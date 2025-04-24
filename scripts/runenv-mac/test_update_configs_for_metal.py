#!/usr/bin/env python3
"""
Unit tests for update_configs_for_metal.py
"""

import os
import sys
import pytest
import yaml
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from scripts.update_configs_for_metal import update_config, is_apple_silicon, update_all_configs

# Sample config for testing
SAMPLE_CONFIG = {
    "batch_size": 16,
    "batch_length": 50, 
    "deter_size": 4096,
    "units": 1024,
    "jax": {
        "platform": "cpu",
        "xla_flags": ["--xla_cpu_multi_thread_eigen=true"]
    }
}

def create_test_config(tmp_path):
    """Create a test config file."""
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(SAMPLE_CONFIG, f)
    return config_path

def test_is_apple_silicon():
    """Test that is_apple_silicon function runs without error."""
    # This test just verifies function runs without error
    # The actual result depends on the hardware
    result = is_apple_silicon()
    assert isinstance(result, bool)

def test_update_config_basic(tmp_path):
    """Test basic config update functionality."""
    # Setup
    config_path = create_test_config(tmp_path)
    output_path = tmp_path / "test_config_metal.yaml"
    
    # Run the update
    result = update_config(config_path, output_path, mode="basic")
    
    # Verify
    assert result is True
    assert output_path.exists()
    
    # Load and check the updated config
    with open(output_path, 'r') as f:
        updated_config = yaml.safe_load(f)
    
    # Check that platform is set to metal
    assert updated_config["jax"]["platform"] == "metal"
    
    # Check batch sizes are adjusted
    assert updated_config["batch_size"] < SAMPLE_CONFIG["batch_size"]
    assert updated_config["batch_length"] < SAMPLE_CONFIG["batch_length"]
    
    # Check model sizes are adjusted
    assert updated_config["deter_size"] < SAMPLE_CONFIG["deter_size"]
    assert updated_config["units"] < SAMPLE_CONFIG["units"]
    
    # Check Metal-specific XLA flags are added
    metal_flags = [
        "--xla_metal_enable_async=true",
        "--xla_gpu_enable_latency_hiding_scheduler=true",
        "--xla_gpu_enable_command_buffer_caching=true",
        "--xla_gpu_enable_command_buffer_barriers=true",
    ]
    for flag in metal_flags:
        assert flag in updated_config["jax"]["xla_flags"]

def test_update_config_memory_mode(tmp_path):
    """Test update with memory conservation mode."""
    config_path = create_test_config(tmp_path)
    output_path = tmp_path / "test_config_metal_memory.yaml"
    
    # Run the update with memory mode
    result = update_config(config_path, output_path, mode="memory")
    
    # Verify
    assert result is True
    
    # Load and check the updated config
    with open(output_path, 'r') as f:
        updated_config = yaml.safe_load(f)
    
    # Memory mode should have smaller batch sizes
    assert updated_config["batch_size"] <= 8  # Half of 16
    assert "_metal_memory_note" in updated_config

def test_update_config_performance_mode(tmp_path):
    """Test update with performance mode."""
    config_path = create_test_config(tmp_path)
    output_path = tmp_path / "test_config_metal_perf.yaml"
    
    # Run the update with performance mode
    result = update_config(config_path, output_path, mode="performance")
    
    # Verify
    assert result is True
    
    # Load and check the updated config
    with open(output_path, 'r') as f:
        updated_config = yaml.safe_load(f)
    
    # Performance mode should have larger batch sizes than memory mode
    assert updated_config["batch_size"] > 8  # Should be 0.75 of original (12)
    assert "_metal_memory_note" not in updated_config

def test_update_config_nonexistent_file():
    """Test handling of nonexistent input file."""
    result = update_config("nonexistent_file.yaml")
    assert result is False

def test_update_config_default_output_path(tmp_path):
    """Test default output path generation."""
    config_path = create_test_config(tmp_path)
    
    # Run update without specifying output path
    result = update_config(config_path)
    
    # Verify default output path is created
    expected_output = config_path.with_stem(f"{config_path.stem}_metal")
    assert result is True
    assert expected_output.exists()

def test_update_config_empty_file(tmp_path):
    """Test handling of empty config file."""
    # Create empty config file
    empty_config_path = tmp_path / "empty_config.yaml"
    with open(empty_config_path, 'w') as f:
        f.write("")
    
    output_path = tmp_path / "empty_config_metal.yaml"
    
    # Run the update
    result = update_config(empty_config_path, output_path)
    
    # Should succeed but with minimal changes
    assert result is True
    assert output_path.exists()
    
    # Load and check the updated config
    with open(output_path, 'r') as f:
        updated_config = yaml.safe_load(f)
    
    # Should at least have jax platform set
    assert updated_config is not None
    assert "jax" in updated_config
    assert updated_config["jax"]["platform"] == "metal" 

class TestAppleSiliconCheck:
    """Tests for is_apple_silicon function."""
    
    @patch('platform.system', return_value='Darwin')
    @patch('platform.machine', return_value='arm64')
    def test_is_apple_silicon_true(self, mock_machine, mock_system):
        """Test is_apple_silicon returns True for Darwin arm64."""
        assert is_apple_silicon() is True
    
    @patch('platform.system', return_value='Darwin')
    @patch('platform.machine', return_value='x86_64')
    def test_is_apple_silicon_false_for_intel_mac(self, mock_machine, mock_system):
        """Test is_apple_silicon returns False for Darwin x86_64."""
        assert is_apple_silicon() is False
    
    @patch('platform.system', return_value='Linux')
    @patch('platform.machine', return_value='arm64')
    def test_is_apple_silicon_false_for_non_mac_arm(self, mock_machine, mock_system):
        """Test is_apple_silicon returns False for Linux arm64."""
        assert is_apple_silicon() is False

class TestUpdateConfig:
    """Tests for update_config function."""
    
    def setup_method(self):
        """Set up the test environment."""
        # Create a temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create a sample config file
        self.sample_config = {
            'description': 'Test config',
            'batch_size': 16,
            'prefetch': 4,
            'model': {
                'encoder': {
                    'cnn': {
                        'depth': 32
                    }
                },
                'rssm': {
                    'deter': 1024,
                    'stoch': 32,
                    'discrete': 32
                }
            }
        }
        
        # Save the sample config
        self.config_path = os.path.join(self.temp_dir.name, 'test_config.yaml')
        with open(self.config_path, 'w') as f:
            yaml.dump(self.sample_config, f)
    
    def teardown_method(self):
        """Clean up the test environment."""
        self.temp_dir.cleanup()
    
    def test_update_config_nonexistent_file(self):
        """Test updating a nonexistent file returns False."""
        result = update_config('nonexistent_file.yaml')
        assert result is False
    
    def test_update_config_basic_mode(self):
        """Test updating a config file in basic mode."""
        output_path = os.path.join(self.temp_dir.name, 'test_config_metal.yaml')
        
        # Update the config
        result = update_config(self.config_path, output_path, mode='basic', adjust_batch_size=False)
        
        # Check result
        assert result is True
        assert os.path.exists(output_path)
        
        # Load the updated config
        with open(output_path, 'r') as f:
            updated_config = yaml.safe_load(f)
        
        # Check basic modifications
        assert updated_config['jax']['platform'] == 'metal'
        assert updated_config['description'] == 'Test config (Metal optimized)'
        
        # Batch size should remain the same in basic mode
        assert updated_config['batch_size'] == 16
    
    def test_update_config_memory_mode(self):
        """Test updating a config file in memory mode."""
        output_path = os.path.join(self.temp_dir.name, 'test_config_metal.yaml')
        
        # Update the config
        result = update_config(self.config_path, output_path, mode='memory')
        
        # Check result
        assert result is True
        assert os.path.exists(output_path)
        
        # Load the updated config
        with open(output_path, 'r') as f:
            updated_config = yaml.safe_load(f)
        
        # Check memory optimizations
        assert updated_config['jax']['platform'] == 'metal'
        assert updated_config['batch_size'] == 8  # Half of original
        assert updated_config['prefetch'] == 2  # Half of original
        assert updated_config['model']['encoder']['cnn']['depth'] == 24  # 75% of original
        assert updated_config['model']['rssm']['deter'] == 768  # 75% of original
        assert updated_config['model']['rssm']['stoch'] == 24  # 75% of original
        assert updated_config['model']['rssm']['discrete'] == 24  # 75% of original
        assert '(Memory-optimized)' in updated_config['description']
    
    def test_update_config_performance_mode(self):
        """Test updating a config file in performance mode."""
        output_path = os.path.join(self.temp_dir.name, 'test_config_metal.yaml')
        
        # Update the config
        result = update_config(self.config_path, output_path, mode='performance')
        
        # Check result
        assert result is True
        assert os.path.exists(output_path)
        
        # Load the updated config
        with open(output_path, 'r') as f:
            updated_config = yaml.safe_load(f)
        
        # Check performance optimizations
        assert updated_config['jax']['platform'] == 'metal'
        assert updated_config['batch_size'] == 20  # 1.25x of original
        assert updated_config['jax']['precision'] == 'float16'
        assert '--xla_metal_enable_async=true' in updated_config['jax']['xla_flags']
        assert '--xla_gpu_enable_latency_hiding_scheduler=true' in updated_config['jax']['xla_flags']
    
    def test_update_config_default_output_path(self):
        """Test updating a config file with default output path."""
        # Update the config without specifying output path
        result = update_config(self.config_path)
        
        # Calculate the expected output path
        expected_path = os.path.join(self.temp_dir.name, 'test_config_metal.yaml')
        
        # Check result
        assert result is True
        assert os.path.exists(expected_path)
    
    def test_update_config_error_handling(self):
        """Test error handling in update_config."""
        # Create an invalid YAML file
        invalid_yaml_path = os.path.join(self.temp_dir.name, 'invalid.yaml')
        with open(invalid_yaml_path, 'w') as f:
            f.write("invalid: yaml: content:")
        
        # Try to update the invalid YAML file
        result = update_config(invalid_yaml_path)
        
        # Should return False due to YAML parsing error
        assert result is False

class TestUpdateAllConfigs:
    """Tests for update_all_configs function."""
    
    def setup_method(self):
        """Set up the test environment."""
        # Create a temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_dir = os.path.join(self.temp_dir.name, 'configs')
        os.makedirs(self.config_dir)
        
        # Create some sample config files
        self.sample_configs = [
            {'description': 'Config 1', 'batch_size': 16},
            {'description': 'Config 2', 'batch_size': 32},
            {'description': 'Config 3', 'batch_size': 8}
        ]
        
        # Save the sample configs
        for i, config in enumerate(self.sample_configs):
            config_path = os.path.join(self.config_dir, f'config{i+1}.yaml')
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
        
        # Create a file that should be ignored (already has _metal suffix)
        ignore_path = os.path.join(self.config_dir, 'already_metal.yaml')
        with open(ignore_path, 'w') as f:
            yaml.dump({'description': 'Already optimized'}, f)
    
    def teardown_method(self):
        """Clean up the test environment."""
        self.temp_dir.cleanup()
    
    def test_update_all_configs_default_output(self):
        """Test updating all configs with default output directory."""
        # Update all configs
        success_count = update_all_configs(self.config_dir)
        
        # Check result
        assert success_count == 3
        
        # Check if output directory exists
        metal_dir = os.path.join(self.config_dir, 'metal')
        assert os.path.isdir(metal_dir)
        
        # Check if all files were created
        for i in range(1, 4):
            output_path = os.path.join(metal_dir, f'config{i}_metal.yaml')
            assert os.path.exists(output_path)
            
            # Check content
            with open(output_path, 'r') as f:
                config = yaml.safe_load(f)
                assert config['jax']['platform'] == 'metal'
    
    def test_update_all_configs_custom_output(self):
        """Test updating all configs with custom output directory."""
        # Create custom output directory
        custom_dir = os.path.join(self.temp_dir.name, 'custom_output')
        
        # Update all configs
        success_count = update_all_configs(self.config_dir, custom_dir)
        
        # Check result
        assert success_count == 3
        
        # Check if output directory exists
        assert os.path.isdir(custom_dir)
        
        # Check if all files were created
        for i in range(1, 4):
            output_path = os.path.join(custom_dir, f'config{i}_metal.yaml')
            assert os.path.exists(output_path)
    
    def test_update_all_configs_different_modes(self):
        """Test updating configs with different modes."""
        # Update all configs with memory mode
        success_count = update_all_configs(self.config_dir, mode='memory')
        
        # Check result
        assert success_count == 3
        
        # Check memory optimizations in one file
        metal_dir = os.path.join(self.config_dir, 'metal')
        output_path = os.path.join(metal_dir, 'config1_metal.yaml')
        
        with open(output_path, 'r') as f:
            config = yaml.safe_load(f)
            assert config['batch_size'] == 8  # Half of original
            assert '(Memory-optimized)' in config['description']
    
    @patch('update_configs_for_metal.update_config', side_effect=[True, False, True])
    def test_update_all_configs_partial_success(self, mock_update):
        """Test updating configs with some failures."""
        # Update all configs
        success_count = update_all_configs(self.config_dir)
        
        # Check result
        assert success_count == 2
        
        # Verify mock was called 3 times
        assert mock_update.call_count == 3

if __name__ == "__main__":
    pytest.main(['-xvs', 'test_update_configs_for_metal.py']) 