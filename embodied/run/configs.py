"""
Configuration handling for the embodied framework.
"""

from . import grpo_config

# Export configuration classes
GRPOConfig = grpo_config.GRPOConfig

def make_config(name, **kwargs):
    """
    Factory function to create configuration objects by name.
    
    Args:
        name: Name of the configuration type to create
        **kwargs: Parameters to pass to the configuration constructor
        
    Returns:
        A configuration object of the requested type
    """
    if name.lower() == 'grpo':
        return GRPOConfig(**kwargs)
    else:
        raise ValueError(f"Unknown configuration type: {name}") 