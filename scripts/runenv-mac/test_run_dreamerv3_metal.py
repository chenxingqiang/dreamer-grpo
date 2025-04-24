#!/usr/bin/env python3
"""
Unit tests for run_dreamerv3_metal.py
"""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile

# Import functions from the script
sys.path.append('.')
from scripts.run_dreamerv3_metal import (
    is_apple_silicon, 
    setup_metal_environment, 
    find_dreamerv3_main, 
    find_config_file,
    run_dreamerv3
)

def test_is_apple_silicon():
    """Test is_apple_silicon function."""
    # The actual result depends on the hardware, but we can ensure it runs
    result = is_apple_silicon()
    assert isinstance(result, bool)

def test_setup_metal_environment():
    """Test that Metal environment setup sets expected environment variables."""
    with patch.dict(os.environ, {}, clear=True):
        # Run the function
        setup_metal_environment()
        
        # Check if environment variables are set correctly
        assert os.environ.get('JAX_PLATFORMS') == 'metal'
        assert os.environ.get('ENABLE_PJRT_COMPATIBILITY') == 'true'
        assert os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION') == '0.85'
        assert 'XLA_FLAGS' in os.environ
        
        # Check that XLA flags contain expected values
        xla_flags = os.environ.get('XLA_FLAGS', '')
        assert '--xla_metal_enable_async=true' in xla_flags
        assert '--xla_gpu_enable_latency_hiding_scheduler=true' in xla_flags

def test_find_dreamerv3_main_existing_file():
    """Test finding main.py when it exists."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a fake main.py file
        temp_path = Path(tmpdir)
        main_path = temp_path / "main.py"
        main_path.touch()
        
        # Test with current directory set to temp directory
        with patch('os.getcwd', return_value=str(temp_path)):
            result = find_dreamerv3_main()
            assert result == str(main_path)

def test_find_dreamerv3_main_not_found():
    """Test behavior when main.py can't be found."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Set current directory to an empty temp directory
        with patch('os.getcwd', return_value=tmpdir):
            with patch('os.path.exists', return_value=False):  # Ensure no files are found
                with pytest.raises(FileNotFoundError):
                    find_dreamerv3_main()

def test_find_config_file_metal_variant():
    """Test finding config file prioritizing metal variants."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create sample config files
        temp_path = Path(tmpdir)
        config_dir = temp_path / "configs"
        config_dir.mkdir()
        
        regular_config = config_dir / "atari.yaml"
        metal_config = config_dir / "atari_metal.yaml"
        
        regular_config.touch()
        metal_config.touch()
        
        # Test finding config with a specific name (should prefer metal variant)
        with patch('os.getcwd', return_value=str(temp_path)):
            result = find_config_file("atari", config_dir)
            assert result == str(metal_config)

def test_find_config_file_regular():
    """Test finding regular config file when metal variant doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create sample config file
        temp_path = Path(tmpdir)
        config_dir = temp_path / "configs"
        config_dir.mkdir()
        
        regular_config = config_dir / "atari.yaml"
        regular_config.touch()
        
        # Test finding config with a specific name
        with patch('os.getcwd', return_value=str(temp_path)):
            result = find_config_file("atari", config_dir)
            assert result == str(regular_config)

def test_find_config_file_not_found():
    """Test behavior when config file can't be found."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create an empty config directory
        temp_path = Path(tmpdir)
        config_dir = temp_path / "configs"
        config_dir.mkdir()
        
        # Test finding a non-existent config
        with patch('os.getcwd', return_value=str(temp_path)):
            with pytest.raises(FileNotFoundError):
                find_config_file("nonexistent", config_dir)

@patch('subprocess.run')
def test_run_dreamerv3_successful(mock_run):
    """Test running DreamerV3 with successful execution."""
    # Mock successful subprocess execution
    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_run.return_value = mock_process
    
    # Test running DreamerV3
    with patch('run_dreamerv3_metal.find_dreamerv3_main', return_value="./main.py"):
        with patch('run_dreamerv3_metal.find_config_file', return_value="./configs/atari_metal.yaml"):
            result = run_dreamerv3(
                config="atari", 
                task="pong", 
                logdir="./logdir", 
                steps=10000,
                seed=1
            )
            
            # Verify success
            assert result is True
            
            # Verify process was started with correct arguments
            mock_run.assert_called_once()
            args = mock_run.call_args[0][0]
            assert "python" in args[0]
            assert "./main.py" in args
            assert "--config=./configs/atari_metal.yaml" in args
            assert "--task=pong" in args
            assert "--logdir=./logdir" in args
            assert "--steps=10000" in args
            assert "--seed=1" in args

@patch('subprocess.run')
def test_run_dreamerv3_failure(mock_run):
    """Test running DreamerV3 with execution failure."""
    # Mock failed subprocess execution
    mock_process = MagicMock()
    mock_process.returncode = 1
    mock_run.return_value = mock_process
    
    # Test running DreamerV3
    with patch('run_dreamerv3_metal.find_dreamerv3_main', return_value="./main.py"):
        with patch('run_dreamerv3_metal.find_config_file', return_value="./configs/atari_metal.yaml"):
            result = run_dreamerv3(
                config="atari", 
                task="pong", 
                logdir="./logdir"
            )
            
            # Verify failure
            assert result is False 