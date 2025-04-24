import jax
import jax.numpy as jnp


def compute_grpo_loss(logits, target_ids, attention_mask, advantages,
                      old_log_probs=None, ref_log_probs=None,
                      epsilon_low=0.2, epsilon_high=None, beta=0.0,
                      loss_type="grpo"):
    """
    Compute the Group Relative Policy Optimization (GRPO) loss.

    Args:
        logits: Model logits with shape [batch_size, sequence_length, vocab_size]
        target_ids: Target token IDs with shape [batch_size, sequence_length]
        attention_mask: Attention mask with shape [batch_size, sequence_length]
        advantages: Advantages with shape [batch_size]
        old_log_probs: Log probabilities from the old policy (for multi-step optimization)
        ref_log_probs: Log probabilities from the reference model (for KL divergence)
        epsilon_low: Lower clipping bound
        epsilon_high: Upper clipping bound, defaults to epsilon_low if None
        beta: KL penalty coefficient
        loss_type: Type of loss to use: "grpo", "bnpo", or "dr_grpo"

    Returns:
        GRPO loss value, and metrics dictionary
    """
    if epsilon_high is None:
        epsilon_high = epsilon_low

    # Compute log probabilities for the current policy
    log_probs = compute_token_log_probs(logits, target_ids)

    # If old_log_probs is None, we're in the first iteration
    if old_log_probs is None:
        old_log_probs = jax.lax.stop_gradient(log_probs)

    # Compute policy ratio
    ratio = jnp.exp(log_probs - old_log_probs)

    # Compute clipped ratio
    clipped_ratio = jnp.clip(ratio, 1.0 - epsilon_low, 1.0 + epsilon_high)

    # Compute the surrogate loss terms
    surrogate1 = ratio * advantages[:, jnp.newaxis]
    surrogate2 = clipped_ratio * advantages[:, jnp.newaxis]
    surrogate_loss = -jnp.minimum(surrogate1, surrogate2)

    # Add KL divergence penalty if reference model is provided
    kl_div = None
    if beta > 0.0 and ref_log_probs is not None:
        kl_div = compute_kl_divergence(ref_log_probs, log_probs)
        surrogate_loss = surrogate_loss + beta * kl_div

    # Apply attention mask to consider only valid tokens
    valid_tokens = attention_mask.sum()

    # Compute the final loss based on the specified loss type
    if loss_type == "grpo":
        # Average over valid tokens in each sequence, then over batch
        per_example_loss = (surrogate_loss * attention_mask).sum(axis=1) / jnp.maximum(attention_mask.sum(axis=1), 1.0)
        loss = per_example_loss.mean()
    elif loss_type == "bnpo":
        # Average over all valid tokens in the batch
        loss = (surrogate_loss * attention_mask).sum() / jnp.maximum(valid_tokens, 1.0)
    elif loss_type == "dr_grpo":
        # Average over all possible token positions (including padding)
        loss = (surrogate_loss * attention_mask).sum() / (surrogate_loss.shape[0] * surrogate_loss.shape[1])
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    # Compute metrics for logging
    metrics = {}

    # Compute clipping metrics
    low_clipped = ((ratio < 1.0 - epsilon_low) & (advantages[:, jnp.newaxis] < 0))
    high_clipped = ((ratio > 1.0 + epsilon_high) & (advantages[:, jnp.newaxis] > 0))
    region_clipped = low_clipped | high_clipped

    # Apply attention mask to the clipping metrics
    low_clip_ratio = (low_clipped * attention_mask).sum() / jnp.maximum(valid_tokens, 1.0)
    high_clip_ratio = (high_clipped * attention_mask).sum() / jnp.maximum(valid_tokens, 1.0)
    region_clip_ratio = (region_clipped * attention_mask).sum() / jnp.maximum(valid_tokens, 1.0)

    metrics['clip_ratio_low'] = low_clip_ratio
    metrics['clip_ratio_high'] = high_clip_ratio
    metrics['clip_ratio'] = region_clip_ratio

    if kl_div is not None:
        metrics['kl_divergence'] = (kl_div * attention_mask).sum() / jnp.maximum(valid_tokens, 1.0)

    return loss, metrics


def compute_token_log_probs(logits, target_ids):
    """
    Compute the log probabilities of the target tokens.

    Args:
        logits: Model logits with shape [batch_size, sequence_length, vocab_size]
        target_ids: Target token IDs with shape [batch_size, sequence_length]

    Returns:
        Log probabilities with shape [batch_size, sequence_length]
    """
    # Get log softmax of logits
    log_probs = jax.nn.log_softmax(logits, axis=-1)

    # Gather log probs for the target tokens
    batch_size, seq_len = target_ids.shape
    batch_indices = jnp.arange(batch_size)[:, jnp.newaxis]
    seq_indices = jnp.arange(seq_len)[jnp.newaxis, :]
    token_log_probs = log_probs[batch_indices, seq_indices, target_ids]

    return token_log_probs


def compute_kl_divergence(ref_log_probs, log_probs):
    """
    Compute the KL divergence between the reference and current policy.
    KL(ref || current) = sum(ref * (log(ref) - log(current)))

    For log probabilities, this simplifies to:
    KL(ref || current) = sum(exp(ref_log_probs) * (ref_log_probs - log_probs))

    Args:
        ref_log_probs: Log probabilities from the reference model
        log_probs: Log probabilities from the current model

    Returns:
        KL divergence with shape [batch_size, sequence_length]
    """
    return jnp.exp(ref_log_probs) * (ref_log_probs - log_probs) - 1.0


def compute_advantages(rewards, num_generations, scale=True):
    """
    Compute advantages by normalizing rewards within groups.

    Args:
        rewards: Array of rewards with shape [batch_size]
        num_generations: Number of generations per prompt
        scale: Whether to scale advantages by the standard deviation

    Returns:
        Advantages with shape [batch_size]
    """
    # Reshape to [num_prompts, num_generations]
    rewards = rewards.reshape(-1, num_generations)

    # Compute mean and standard deviation per group
    mean_rewards = rewards.mean(axis=1, keepdims=True)

    # Compute advantages
    advantages = rewards - mean_rewards

    # Scale by standard deviation if requested
    if scale:
        std_rewards = rewards.std(axis=1, keepdims=True)
        advantages = advantages / (std_rewards + 1e-8)

    # Reshape back to original shape
    return advantages.reshape(-1)