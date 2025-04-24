#!/usr/bin/env python3
"""Test script to verify JAX import fix works properly."""

import os
# Force CPU backend for test
os.environ['JAX_PLATFORMS'] = 'cpu'

try:
    # First, test if our agent can be imported
    print("Attempting to import the agent module...")
    from embodied.jax.agent import Agent
    print("Successfully imported the Agent class!")

    # Then test basic JAX functionality
    import jax
    import jax.numpy as jnp

    print(f"JAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")

    # Simple JAX operation
    x = jnp.ones((3, 3))
    y = jnp.ones((3, 3))
    z = jnp.matmul(x, y)
    print(f"JAX matrix multiplication result: {z}")

    print("All tests passed! JAX is working correctly on this system.")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()