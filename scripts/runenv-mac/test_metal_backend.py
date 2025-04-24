#!/usr/bin/env python3
"""
Test script to verify Metal backend configuration for DreamerV3.
This script checks if JAX is correctly configured with Metal backend on Apple Silicon Macs.
"""

import os
import sys
import time
import platform

# Print system information
print("System information:")
print(f"  Platform: {platform.system()} {platform.release()}")
print(f"  Processor: {platform.processor()}")
print(f"  Python version: {platform.python_version()}")
print("")

# Check environment variables
print("Environment configuration:")
print(f"  JAX_PLATFORMS: {os.environ.get('JAX_PLATFORMS', 'not set')}")
print(f"  ENABLE_PJRT_COMPATIBILITY: {os.environ.get('ENABLE_PJRT_COMPATIBILITY', 'not set')}")
print(f"  XLA_PYTHON_CLIENT_MEM_FRACTION: {os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION', 'not set')}")
print("")

# Import JAX
print("Importing JAX...")
try:
    import jax
    import jax.numpy as jnp
    print(f"JAX version: {jax.__version__}")
    
    # Print JAX configuration
    print("JAX configuration:")
    print(f"  Available platforms: {jax.lib.xla_bridge.get_backend().platform_version}")
    print(f"  Current platform: {jax.devices()[0].platform}")
    print(f"  Device count: {jax.device_count()}")
    print(f"  Devices: {jax.devices()}")
    
    # Check if we're using Metal
    is_using_metal = any("Metal" in str(d) for d in jax.devices())
    if is_using_metal:
        print("✅ Successfully using Metal backend")
    else:
        print("❌ Not using Metal backend")
        
    print("")
    
    # Run a simple test
    print("Running matrix multiplication test...")
    
    # Create some test matrices
    size = 2000
    print(f"  Creating {size}x{size} matrices...")
    
    # Define function to benchmark
    def benchmark_matmul(size):
        # Create matrices on device
        x = jnp.ones((size, size))
        y = jnp.ones((size, size))
        
        # Warm-up run
        _ = jnp.matmul(x, y)
        jax.device_get(_)  # Force completion
        
        # Timed run
        start_time = time.time()
        result = jnp.matmul(x, y)
        jax.device_get(result)  # Force completion
        end_time = time.time()
        
        return result, end_time - start_time
    
    # Benchmark
    result, duration = benchmark_matmul(size)
    print(f"  Matrix multiplication completed in {duration:.4f} seconds")
    print(f"  Result shape: {result.shape}, sum: {result.sum()}")
    
    # Add another test - simple neural network forward pass
    print("\nRunning neural network forward pass test...")
    
    def create_simple_network():
        import jax.random as random
        key = random.PRNGKey(0)
        
        # Create a simple 2-layer network
        weights1 = random.normal(key, (1000, 1000))
        weights2 = random.normal(key, (1000, 100))
        
        # Define forward pass
        def forward(x):
            hidden = jnp.maximum(0, jnp.dot(x, weights1))  # ReLU activation
            output = jnp.dot(hidden, weights2)
            return output
        
        return forward
    
    # Create and benchmark network
    try:
        network = create_simple_network()
        inputs = jnp.ones((100, 1000))
        
        # Warm-up
        _ = network(inputs)
        jax.device_get(_)
        
        # Benchmark
        iterations = 10
        start_time = time.time()
        for _ in range(iterations):
            result = network(inputs)
            jax.device_get(result)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / iterations
        print(f"  Average forward pass time ({iterations} iterations): {avg_time:.6f} seconds")
        print(f"  Output shape: {result.shape}")
        
    except Exception as e:
        print(f"  Error in neural network test: {e}")
    
    print("\nOverall Metal backend test result:")
    if is_using_metal:
        print("✅ JAX is correctly configured with Metal backend")
    else:
        print("⚠️ JAX is running but not using Metal backend")
    
except Exception as e:
    print(f"\n❌ Error initializing JAX: {e}")
    import traceback
    traceback.print_exc()
    print("\nJAX initialization failed. Please check your configuration.")
    sys.exit(1) 