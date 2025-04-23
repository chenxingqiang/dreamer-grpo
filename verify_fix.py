#!/usr/bin/env python3
"""
Verification script to check that our JAX fix works with DreamerV3 code.
This script tests importing key modules and JAX initialization.
"""

import os
import sys

# Set environment variables for JAX on macOS
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['SYSTEM_VERSION_COMPAT'] = '0'

print("Starting verification of JAX fix...")

try:
    # First import JAX and verify devices are available
    print("Testing JAX import and device availability...")
    import jax
    print(f"JAX version: {jax.__version__}")
    print(f"Available devices: {jax.devices()}")
    
    # Test importing the agent module
    print("Testing import of the agent module...")
    from embodied.jax.agent import Agent
    print("Successfully imported the Agent class")
    
    # Test other critical imports
    print("Testing other critical imports...")
    import embodied
    import ninjax
    import elements
    print("All critical imports successful")
    
    print("\n✅ All verification tests passed!")
    print("The JAX backend initialization fix is working correctly.")
    
except Exception as e:
    print(f"\n❌ Verification failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)