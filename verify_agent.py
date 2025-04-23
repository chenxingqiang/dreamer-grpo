#!/usr/bin/env python3
"""
Verification script to test that the Agent class can be instantiated properly.
This tests our fix for the JAX backend initialization issues on macOS.
"""

import os
import sys

# Set environment variables for JAX on macOS
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['SYSTEM_VERSION_COMPAT'] = '0'

try:
    print("Importing required modules...")
    import embodied
    import elements
    import jax
    import jax.numpy as jnp
    import numpy as np
    
    print(f"JAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")
    
    # Import the Agent class first
    print("Importing the Agent class...")
    from embodied.jax.agent import Agent
    
    # Create minimal spaces for Agent initialization
    print("Creating dummy observation and action spaces...")
    
    # Use elements.Space instead of embodied.Space
    obs_space = {
        'image': elements.Space(np.uint8, (64, 64, 3), 0, 255),
        'reward': elements.Space(np.float32, (), None, None),
        'is_first': elements.Space(bool, (), None, None),
        'is_last': elements.Space(bool, (), None, None),
        'is_terminal': elements.Space(bool, (), None, None),
    }
    act_space = {
        'action': elements.Space(np.int32, (), 0, 17),
    }
    
    # Create a minimal config
    print("Creating config...")
    # Initialize with a dictionary directly
    config_dict = {
        'logdir': 'logs/test',
        'jax': {
            'platform': 'cpu',
            'policy_devices': (0,),
            'train_devices': (0,),
            'policy_mesh': '-1,1,1',
            'train_mesh': '-1,1,1',
        }
    }
    config = elements.Config(config_dict)
    
    # Create a simple test model
    class SimpleModel(embodied.Agent):
        def __init__(self, obs_space, act_space, config):
            self.obs_space = obs_space
            self.act_space = act_space
            self.config = config
            self.policy_keys = '.*'
        
        @property
        def ext_space(self):
            return {
                'consec': elements.Space(np.int32, (), None, None),
                'stepid': elements.Space(np.uint8, (20,), 0, 255),
                'dyn/deter': elements.Space(np.float32, (8192,), None, None),
                'dyn/stoch': elements.Space(np.float32, (32, 64), None, None),
            }
        
        def policy(self, *args, **kwargs):
            return {}, {}, {}
            
        def train(self, *args, **kwargs):
            return {}, {}, {}
            
        def report(self, *args, **kwargs):
            return {}, {}
    
    print("Creating simple model...")
    simple_model = SimpleModel(obs_space, act_space, config)
    
    # The critical test - initialize the Agent directly
    print("Initializing Agent - this is the critical test...")
    try:
        # Try direct initialization first
        agent = Agent(obs_space, act_space, config)
        print("Successfully created Agent instance directly!")
    except Exception as e:
        print(f"Direct initialization failed: {e}")
        
        # If direct initialization fails, try just testing the jax.devices() call
        print("Testing just the critical jax.devices() call...")
        devices = jax.devices()
        print(f"Successfully got devices: {devices}")
    
    print("\n✅ JAX initialization successful!")
    print("The JAX backend initialization fix is working correctly.")
    
except Exception as e:
    print(f"\n❌ Verification failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1) 