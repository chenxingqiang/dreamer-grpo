# Group Relative Policy Optimization (GRPO) Implementation

This implementation adds support for the Group Relative Policy Optimization (GRPO) algorithm to the DreamerV3 framework. GRPO is based on the method introduced in the paper ["DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models"](https://huggingface.co/papers/2402.03300).

## Key Features

- Group-wise reward normalization for more stable learning
- Support for KL divergence regularization with reference model
- Multiple optimization iterations per batch
- Compatible with PEFT models
- Multiple loss types: GRPO, BNPO, DR-GRPO

## How to Use

### Basic Usage

To run training with GRPO, use the `train_grpo` script option:

```bash
./macos_run.sh dreamerv3/main.py --configs defaults \
  --script train_grpo \
  --run.train_ratio 32 \
  --beta 0.1 \
  --epsilon 0.2 \
  --num_generations 4 \
  --scale_rewards true \
  --loss_type grpo \
  --logdir ./logs/dreamer/{timestamp}
```

This sets up a GRPO training run with the following parameters:
- Beta (KL coefficient) = 0.1
- Epsilon (clipping parameter) = 0.2
- 4 generations per prompt
- Reward scaling enabled
- Using the standard GRPO loss function

### Using Configuration Templates

You can also use the predefined GRPO configuration templates in `configs.yaml`:

```bash
# 基础GRPO配置
./macos_run.sh dreamerv3/main.py --configs defaults grpo_config

# 带有较高KL惩罚的GRPO
./macos_run.sh dreamerv3/main.py --configs defaults grpo_kl

# 使用BNPO损失函数的GRPO
./macos_run.sh dreamerv3/main.py --configs defaults grpo_bnpo

# 多步迭代的GRPO
./macos_run.sh dreamerv3/main.py --configs defaults grpo_multi
```

### GPU Acceleration on macOS

To use Metal GPU acceleration on Apple Silicon Macs:

1. Install jax-metal:
```bash
pip install jax-metal
```

2. Enable Metal backend:
```bash
./macos_run.sh dreamerv3/main.py --configs defaults grpo_config \
  --jax.platform metal \
  --logdir ./logs/dreamer/{timestamp}
```

### Configuration Parameters

The main GRPO configuration parameters are:

- `beta`: KL penalty coefficient (β in the paper)
- `epsilon`: Lower clipping value (ε_low in the paper)
- `epsilon_high`: Upper clipping value (ε_high in the paper), if None uses epsilon value
- `num_generations`: Number of generations per prompt (G in the paper)
- `num_iterations`: Number of optimization iterations per batch (μ in the paper)
- `scale_rewards`: Whether to scale rewards by the standard deviation
- `loss_type`: Loss type: "grpo", "bnpo", or "dr_grpo"

See `embodied/run/grpo_config.py` for the full list of configuration options.

### Customizing the Reward Function

GRPO works with both standard rewards and custom reward functions. To use a custom reward function:

1. Define your reward function
2. Ensure it returns a tensor of rewards with shape [batch_size]
3. Pass the rewards to the agent through the batch dictionary

## Implementation Details

The GRPO implementation consists of several key components:

- `embodied/run/grpo_train.py`: Main training loop for GRPO
- `embodied/run/grpo_config.py`: Configuration options for GRPO
- `embodied/jax/grpo_loss.py`: Core GRPO loss calculation
- `embodied/jax/agent.py`: Extended with methods for GRPO support

The implementation follows the algorithm described in the DeepSeekMath paper, with the key steps:

1. Generate multiple completions for each prompt
2. Compute rewards for each completion
3. Normalize rewards within each group to get advantages
4. Apply clipped surrogate loss with KL penalty
5. Optionally perform multiple optimization iterations

## References

```bibtex
@article{zhihong2024deepseekmath,
    title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
    author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
    year         = 2024,
    eprint       = {arXiv:2402.03300},
}
``` 