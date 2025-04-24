import dataclasses


@dataclasses.dataclass
class GRPOConfig:
    """
    Configuration class for Group Relative Policy Optimization (GRPO).
    
    This defines the parameters for the GRPO algorithm as described in the paper
    "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models".
    """
    
    # General parameters for GRPO
    beta: float = 0.0  # KL penalty coefficient (β in the paper)
    epsilon: float = 0.2  # Lower clipping value (ε_low in the paper)
    epsilon_high: float = None  # Upper clipping value (ε_high in the paper), if None uses epsilon value
    num_generations: int = 4  # Number of generations per prompt (G in the paper)
    num_iterations: int = 1  # Number of optimization iterations per batch (μ in the paper)
    scale_rewards: bool = True  # Whether to scale rewards by the standard deviation
    loss_type: str = "grpo"  # Loss type: "grpo", "bnpo", or "dr_grpo"
    
    # Model and adapter options
    use_peft: bool = False  # Whether to use Parameter-Efficient Fine-Tuning (PEFT)
    disable_dropout: bool = True  # Whether to disable dropout during training
    
    # Generation parameters
    temperature: float = 1.0  # Sampling temperature
    top_p: float = 1.0  # Top-p (nucleus) sampling parameter
    top_k: int = None  # Top-k sampling parameter (None for no top-k)
    min_p: float = None  # Min-p sampling parameter (None for no min-p)
    repetition_penalty: float = 1.0  # Repetition penalty for generation
    max_prompt_length: int = None  # Maximum prompt length (None for no truncation)
    max_completion_length: int = 128  # Maximum completion length (|o_i| in the paper)
    mask_truncated_completions: bool = True  # Whether to mask out completions that were truncated
    
    # Logging parameters
    log_completions: bool = True  # Whether to log completions
    wandb_log_unique_prompts: bool = True  # Whether to log only unique prompts to wandb
    num_completions_to_print: int = 3  # Number of completions to print for debugging
    
    # Advanced acceleration options
    use_vllm: bool = False  # Whether to use vLLM for faster generation
    vllm_server_host: str = "localhost"  # vLLM server host
    vllm_server_port: int = 8000  # vLLM server port
    vllm_server_timeout: int = 300  # vLLM server connection timeout in seconds
    vllm_guided_decoding_regex: str = None  # Regex pattern for guided decoding in vLLM
    use_liger_loss: bool = False  # Whether to use Liger kernel for fast loss computation
    cache_implementation: str = "default"  # Cache implementation for generation 