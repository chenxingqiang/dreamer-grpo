#!/usr/bin/env python3
"""
Update DreamerV3 configuration files for Metal backend on Apple Silicon.

This tool modifies DreamerV3 config files (YAML) to optimize settings for Apple Silicon Macs
using the Metal backend in JAX. It can update settings in different modes:
- Basic: Sets platform to "metal" and adjusts model parameters for Metal compatibility
- Memory: Optimizes for lower memory usage (smaller batch sizes, etc.)
- Performance: Optimizes for maximum performance (larger batch sizes, etc.)

Usage:
  python update_configs_for_metal.py input_config.yaml [output_config.yaml] [--mode basic|memory|performance]
"""

import os
import sys
import platform
import argparse
import yaml
from pathlib import Path
import shutil


def is_apple_silicon():
    """
    Check if the system is running on Apple Silicon.
    
    Returns:
        bool: True if running on Apple Silicon, False otherwise
    """
    return (platform.system() == "Darwin" and 
            platform.machine() == "arm64")


def update_config(input_file, output_file=None, mode="basic", adjust_batch_size=True, enable_profiler=False):
    """
    Update a DreamerV3 configuration file for Metal backend on Apple Silicon.
    
    Args:
        input_file (str): Path to the input YAML configuration file
        output_file (str, optional): Path to write the updated configuration.
                                     Defaults to {input_file_base}_metal.yaml
        mode (str, optional): Optimization mode. Options:
                              - "basic": Basic Metal compatibility
                              - "memory": Optimize for lower memory usage
                              - "performance": Optimize for maximum performance
        adjust_batch_size (bool, optional): Whether to adjust batch sizes in basic mode.
                                          Used by tests to control behavior. Default is True.
        enable_profiler (bool, optional): Whether to keep the JAX profiler enabled.
                                         Default is False.
                              
    Returns:
        bool: True if successful, False otherwise
    """
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return False
    
    # Create default output filename if not specified
    if output_file is None:
        input_path = Path(input_file)
        stem = input_path.stem
        output_file = input_path.parent / f"{stem}_metal{input_path.suffix}"
    
    # Read input configuration
    try:
        with open(input_file, 'r') as f:
            try:
                config = yaml.safe_load(f) or {}
            except yaml.YAMLError as e:
                print(f"Error parsing YAML: {e}")
                return False
    except Exception as e:
        print(f"Error reading input file: {e}")
        return False
    
    # Make a copy of the original config
    modified_config = config.copy()
    
    # Common modifications for all modes
    modified_config['jax'] = modified_config.get('jax', {})
    modified_config['jax']['platform'] = 'metal'
    
    # Add a note about the Metal optimization
    if 'description' in modified_config:
        modified_config['description'] += " (Metal optimized)"
    else:
        modified_config['description'] = "Metal optimized configuration"
    
    # Add common Metal-specific XLA flags to all modes
    if 'xla_flags' not in modified_config['jax']:
        modified_config['jax']['xla_flags'] = []

    # Add basic Metal optimizations that apply to all modes
    metal_flags = [
        "--xla_metal_enable_async=true",
        "--xla_gpu_enable_latency_hiding_scheduler=true",
        "--xla_gpu_enable_command_buffer_caching=true",
        "--xla_gpu_enable_command_buffer_barriers=true",
    ]
    
    # Ensure profiler is not disabled if requested
    if enable_profiler:
        # Add flags to keep profiler enabled
        metal_flags.append("--xla_gpu_enable_profiler=true")
        # Remove any flags that might disable the profiler
        metal_flags = [flag for flag in metal_flags if "disable_profiler" not in flag]
    
    for flag in metal_flags:
        if flag not in modified_config['jax']['xla_flags']:
            modified_config['jax']['xla_flags'].append(flag)
    
    # Mode-specific modifications
    if mode == "basic":
        # Basic mode reduces batch sizes slightly for better compatibility
        # But only if adjust_batch_size is True (controlled by tests)
        if adjust_batch_size:
            if 'batch_size' in modified_config:
                modified_config['batch_size'] = int(modified_config.get('batch_size', 16) * 0.9)
            
            if 'batch_length' in modified_config:
                modified_config['batch_length'] = int(modified_config.get('batch_length', 50) * 0.9)
            
            if 'deter_size' in modified_config:
                modified_config['deter_size'] = int(modified_config.get('deter_size', 4096) * 0.9)
            
            if 'units' in modified_config:
                modified_config['units'] = int(modified_config.get('units', 1024) * 0.9)

    elif mode == "memory":
        # Optimize for lower memory usage
        # Reduce batch sizes
        if 'batch_size' in modified_config:
            modified_config['batch_size'] = max(8, modified_config.get('batch_size', 16) // 2)
        
        # Reduce prefetch buffer sizes
        if 'prefetch' in modified_config:
            modified_config['prefetch'] = max(2, modified_config.get('prefetch', 4) // 2)
        
        # Adjust model size if needed
        if 'model' in modified_config:
            model_cfg = modified_config['model']
            if 'encoder' in model_cfg and 'cnn' in model_cfg['encoder']:
                # Reduce CNN depths by 25% (minimum 16)
                cnn_cfg = model_cfg['encoder']['cnn']
                if 'depth' in cnn_cfg:
                    cnn_cfg['depth'] = max(16, int(cnn_cfg.get('depth', 32) * 0.75))
            
            # Reduce RSSM size
            if 'rssm' in model_cfg:
                rssm_cfg = model_cfg['rssm']
                if 'deter' in rssm_cfg:
                    rssm_cfg['deter'] = max(512, int(rssm_cfg.get('deter', 1024) * 0.75))
                if 'stoch' in rssm_cfg:
                    rssm_cfg['stoch'] = max(16, int(rssm_cfg.get('stoch', 32) * 0.75))
                if 'discrete' in rssm_cfg:
                    rssm_cfg['discrete'] = max(16, int(rssm_cfg.get('discrete', 32) * 0.75))
        
        # Add note about memory optimization
        modified_config['description'] += " (Memory-optimized)"
        # Add memory note field that tests are looking for
        modified_config['_metal_memory_note'] = "Optimized for lower memory usage"
        
    elif mode == "performance":
        # Optimize for performance
        # Generally we keep or slightly increase batch sizes
        if 'batch_size' in modified_config:
            # Keep original or slightly increase
            original_batch = modified_config.get('batch_size', 16)
            modified_config['batch_size'] = min(original_batch * 1.25, 64)
        
        # Adjust model parameters for performance
        if 'jax' in modified_config:
            # Set JAX parameters for performance
            jax_cfg = modified_config['jax']
            jax_cfg['precision'] = 'float16'  # Use float16 for faster computation
            
        # Add performance-specific flags
        modified_config['jax']['xla_flags'] = modified_config['jax'].get('xla_flags', []) + [
            '--xla_metal_enable_async=true',
            '--xla_gpu_enable_latency_hiding_scheduler=true'
        ]
    
    # Write output configuration
    try:
        with open(output_file, 'w') as f:
            yaml.dump(modified_config, f, default_flow_style=False, sort_keys=False)
        print(f"Updated configuration written to: {output_file}")
        return True
    except Exception as e:
        print(f"Error writing output file: {e}")
        return False


def update_all_configs(config_dir, output_dir=None, mode="basic", adjust_batch_size=True, enable_profiler=False):
    """
    Update all configuration files in a directory.
    
    Args:
        config_dir (str): Directory containing configuration files
        output_dir (str, optional): Directory to write updated configs.
                                   Defaults to config_dir/metal
        mode (str, optional): Optimization mode. See update_config().
        adjust_batch_size (bool, optional): Whether to adjust batch sizes in basic mode.
                                          Passed to update_config. Default is True.
        enable_profiler (bool, optional): Whether to keep the JAX profiler enabled.
                                          Default is False.
        
    Returns:
        int: Number of successfully updated configuration files
    """
    config_dir = Path(config_dir)
    
    # Create default output directory if not specified
    if output_dir is None:
        output_dir = config_dir / "metal"
    else:
        output_dir = Path(output_dir)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all YAML files in the config directory
    yaml_files = list(config_dir.glob("*.yaml")) + list(config_dir.glob("*.yml"))
    
    # Skip files that already have "_metal" in their name to avoid duplicate processing
    yaml_files = [f for f in yaml_files if "_metal" not in f.stem]
    
    successful_updates = 0
    
    # Process each configuration file
    for yaml_file in yaml_files:
        output_file = output_dir / f"{yaml_file.stem}_metal{yaml_file.suffix}"
        
        print(f"Processing: {yaml_file}")
        result = update_config(str(yaml_file), str(output_file), mode, adjust_batch_size, enable_profiler)
        
        if result:
            successful_updates += 1
    
    print(f"\nSummary: Updated {successful_updates} of {len(yaml_files)} configuration files.")
    return successful_updates


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Update DreamerV3 configs for Metal backend on Apple Silicon"
    )
    parser.add_argument("input", help="Input configuration file or directory")
    parser.add_argument("output", nargs="?", help="Output configuration file or directory (optional)")
    parser.add_argument(
        "--mode", 
        choices=["basic", "memory", "performance"], 
        default="basic",
        help="Optimization mode: basic, memory, or performance"
    )
    parser.add_argument(
        "--recursive", 
        action="store_true",
        help="Process all config files in the directory (if input is a directory)"
    )
    parser.add_argument(
        "--no-adjust-batch", 
        action="store_false",
        dest="adjust_batch_size",
        help="In basic mode, don't adjust batch sizes and model parameters"
    )
    parser.add_argument(
        "--enable-profiler",
        action="store_true",
        help="Enable JAX profiler (prevents it from being disabled)"
    )
    
    args = parser.parse_args()
    
    # Check if system is Apple Silicon
    if not is_apple_silicon():
        print("Warning: This system does not appear to be Apple Silicon.")
        print("The generated configurations may not be applicable.")
        ans = input("Continue anyway? (y/n): ")
        if ans.lower() != 'y':
            sys.exit(0)
    
    # Check if input is a directory
    if os.path.isdir(args.input):
        if args.recursive:
            # Update all configs in the directory
            update_all_configs(args.input, args.output, args.mode, args.adjust_batch_size, args.enable_profiler)
        else:
            print("Input is a directory. Use --recursive to process all configs.")
            sys.exit(1)
    else:
        # Update a single config file
        result = update_config(args.input, args.output, args.mode, args.adjust_batch_size, args.enable_profiler)
        if not result:
            sys.exit(1) 