import collections
from functools import partial as bind
import torch
import platform
import os

import elements
import embodied
import numpy as np

# 在文件开头添加平台检测
IS_MACOS = platform.system() == 'Darwin'

def grpo_train(make_agent, make_replay, make_env, make_stream, make_logger, args):
  """
  Group Relative Policy Optimization (GRPO) training loop.
  
  This adapts the standard training procedure to implement GRPO, which:
  1. Uses a reference model to compute KL divergence
  2. Computes advantages between generated samples
  3. Applies epsilon clipping for policy ratio
  4. Scales rewards within groups of generated samples
  """
  # Platform-specific optimizations
  if IS_MACOS:
    # Force CPU platform on macOS
    if hasattr(args, 'jax') and hasattr(args.jax, 'platform'):
      if args.jax.platform != 'cpu':
        print(f"Warning: Forcing JAX platform to 'cpu' on macOS in GRPO train (was '{args.jax.platform}')")
        if hasattr(args.jax, '_update'):
          args.jax._update({'platform': 'cpu'})
        else:
          args.jax.platform = 'cpu'
    
    # Disable JAX profiler on macOS to prevent crashes
    if hasattr(args, 'jax'):
      if not hasattr(args.jax, 'profiler'):
        args.jax.profiler = False
      elif args.jax.profiler:
        print("Warning: Disabling JAX profiler on macOS to prevent crashes")
        args.jax.profiler = False
    
    # Set environment variables for macOS
    os.environ['JAX_PLATFORMS'] = 'cpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['SYSTEM_VERSION_COMPAT'] = '0'
    
    # Reduce batch size if needed on macOS to prevent memory issues
    if hasattr(args, 'batch_size') and args.batch_size > 32:
      original_batch_size = args.batch_size
      args.batch_size = min(32, args.batch_size)
      print(f"Warning: Reducing batch size from {original_batch_size} to {args.batch_size} on macOS to prevent memory issues")

  # Create primary agent and reference agent
  agent = make_agent()
  
  # Set up reference agent if beta > 0
  ref_agent = None
  if getattr(args, 'beta', 0.0) > 0.0:
    if getattr(args, 'use_peft', False):
      # For PEFT models, we can disable the adapter to get the reference model
      ref_agent = agent  # Will use adapter disabling mechanism
    else:
      # Otherwise create a separate reference model
      ref_agent = make_agent()  # Clone the agent
  
  replay = make_replay()
  logger = make_logger()

  logdir = elements.Path(args.logdir)
  step = logger.step
  usage = elements.Usage(**args.usage)
  train_agg = elements.Agg()
  epstats = elements.Agg()
  episodes = collections.defaultdict(elements.Agg)
  policy_fps = elements.FPS()
  train_fps = elements.FPS()

  # GRPO specific parameters
  num_generations = getattr(args, 'num_generations', 4)
  beta = getattr(args, 'beta', 0.0)  # KL penalty coefficient
  epsilon_low = getattr(args, 'epsilon', 0.2)
  epsilon_high = getattr(args, 'epsilon_high', None)
  if epsilon_high is None:
    epsilon_high = epsilon_low
  scale_rewards = getattr(args, 'scale_rewards', True)
  loss_type = getattr(args, 'loss_type', 'grpo')

  batch_steps = args.batch_size * args.batch_length
  should_train = elements.when.Ratio(args.train_ratio / batch_steps)
  should_log = embodied.LocalClock(args.log_every)
  should_report = embodied.LocalClock(args.report_every)
  should_save = embodied.LocalClock(args.save_every)

  @elements.timer.section('logfn')
  def logfn(tran, worker):
    episode = episodes[worker]
    tran['is_first'] and episode.reset()
    episode.add('score', tran['reward'], agg='sum')
    episode.add('length', 1, agg='sum')
    episode.add('rewards', tran['reward'], agg='stack')
    for key, value in tran.items():
      if value.dtype == np.uint8 and value.ndim == 3:
        if worker == 0:
          episode.add(f'policy_{key}', value, agg='stack')
      elif key.startswith('log/'):
        assert value.ndim == 0, (key, value.shape, value.dtype)
        episode.add(key + '/avg', value, agg='avg')
        episode.add(key + '/max', value, agg='max')
        episode.add(key + '/sum', value, agg='sum')
    if tran['is_last']:
      result = episode.result()
      logger.add({
          'score': result.pop('score'),
          'length': result.pop('length'),
      }, prefix='episode')
      rew = result.pop('rewards')
      if len(rew) > 1:
        result['reward_rate'] = (np.abs(rew[1:] - rew[:-1]) >= 0.01).mean()
      epstats.add(result)

  fns = [bind(make_env, i) for i in range(args.envs)]
  driver = embodied.Driver(fns, parallel=not args.debug)
  driver.on_step(lambda tran, _: step.increment())
  driver.on_step(lambda tran, _: policy_fps.step())
  driver.on_step(replay.add)
  driver.on_step(logfn)

  stream_train = iter(agent.stream(make_stream(replay, 'train')))
  stream_report = iter(agent.stream(make_stream(replay, 'report')))

  carry_train = [agent.init_train(args.batch_size)]
  carry_report = agent.init_report(args.batch_size)

  # Keep track of old policy log probabilities for GRPO
  old_per_token_logps = None
  # Counter for multi-step iterations
  iteration_counter = 0
  # Number of iterations to perform per batch (μ in the GRPO paper)
  num_iterations = getattr(args, 'num_iterations', 1)

  def compute_advantages(rewards):
    """Compute advantages by normalizing rewards within groups"""
    rewards = torch.tensor(rewards)
    # Reshape to (num_groups, num_generations)
    rewards = rewards.reshape(-1, num_generations)
    
    # Compute mean and std per group
    mean_rewards = rewards.mean(dim=1, keepdim=True)
    std_rewards = rewards.std(dim=1, keepdim=True)
    
    # Normalize to get advantages
    advantages = rewards - mean_rewards
    if scale_rewards:
      advantages = advantages / (std_rewards + 1e-8)
      
    # Reshape back to original shape
    return advantages.reshape(-1).tolist()

  def trainfn(tran, worker):
    nonlocal old_per_token_logps, iteration_counter
    
    if len(replay) < args.batch_size * args.batch_length:
      return
      
    for _ in range(should_train(step)):
      with elements.timer.section('stream_next'):
        batch = next(stream_train)
      
      # Process batch for GRPO
      if 'rewards' in batch:
        # If multiple rewards per sample, compute advantages
        batch['advantages'] = compute_advantages(batch['rewards'])
      
      # Store current policy log probabilities for next iteration
      if num_iterations > 1 and iteration_counter == 0:
        with torch.no_grad():
          # Get log probabilities of current policy (implementation specific)
          # This would need to be adapted based on the actual agent implementation
          old_per_token_logps = agent.get_log_probs(batch)
        
      # Add old log probabilities to batch if available
      if old_per_token_logps is not None:
        batch['old_per_token_logps'] = old_per_token_logps
      
      # Get reference model log probabilities if using KL penalty
      if beta > 0.0 and ref_agent is not None:
        with torch.no_grad():
          if hasattr(ref_agent, 'disable_adapter'):
            # For PEFT models, disable adapter to get reference model outputs
            with ref_agent.disable_adapter():
              batch['ref_per_token_logps'] = ref_agent.get_log_probs(batch)
          else:
            # For separate reference model
            batch['ref_per_token_logps'] = ref_agent.get_log_probs(batch)
      
      # GRPO hyperparameters should NOT be added to the batch
      # as they would trigger assertion errors in the agent.train() method
      # Instead, these parameters are passed separately to the train_grpo method
      grpo_params = {
        'beta': beta,
        'epsilon_low': epsilon_low,
        'epsilon_high': epsilon_high,
        'loss_type': loss_type,
        'scale_rewards': scale_rewards,
        'num_generations': num_generations,
        'num_iterations': num_iterations
      }
      
      # Train the agent using the GRPO-specific method
      carry_train[0], outs, mets = agent.train_grpo(carry_train[0], batch, grpo_params)
      train_fps.step(batch_steps)
      if 'replay' in outs:
        replay.update(outs['replay'])
      train_agg.add(mets, prefix='train')
      
      # Update iteration counter
      iteration_counter = (iteration_counter + 1) % num_iterations
      if iteration_counter == 0:
        # Reset old log probabilities for new batch
        old_per_token_logps = None
        
  driver.on_step(trainfn)

  cp = elements.Checkpoint(logdir / 'ckpt')
  cp.step = step
  cp.agent = agent
  cp.replay = replay
  if args.from_checkpoint:
    elements.checkpoint.load(args.from_checkpoint, dict(
        agent=bind(agent.load, regex=args.from_checkpoint_regex)))
  cp.load_or_save()

  print('Start GRPO training loop')
  policy = lambda *args: agent.policy(*args, mode='train')
  driver.reset(agent.init_policy)
  while step < args.steps:

    driver(policy, steps=10)

    if should_report(step) and len(replay):
      agg = elements.Agg()
      for _ in range(args.consec_report * args.report_batches):
        carry_report, mets = agent.report(carry_report, next(stream_report))
        agg.add(mets)
      logger.add(agg.result(), prefix='report')

    if should_log(step):
      logger.add(train_agg.result())
      logger.add(epstats.result(), prefix='epstats')
      logger.add(replay.stats(), prefix='replay')
      logger.add(usage.stats(), prefix='usage')
      logger.add({'fps/policy': policy_fps.result()})
      logger.add({'fps/train': train_fps.result()})
      logger.add({'timer': elements.timer.stats()['summary']})
      logger.write()

    if should_save(step):
      cp.save()

  logger.close()

# 在训练函数中，更新配置前检查是否在macOS上运行
def train(agent, env, replay, logger, args):
  # 确保在macOS上不使用CUDA
  if IS_MACOS and hasattr(args, 'jax') and hasattr(args.jax, 'platform'):
    if args.jax.platform != 'cpu':
      print(f"Warning: Forcing JAX platform to 'cpu' on macOS (was '{args.jax.platform}')")
      args.jax.platform = 'cpu'
