# DreamerV3 macOS Quickstart Guide

This quickstart guide helps you get DreamerV3 running on macOS systems, addressing common issues with JAX on Apple hardware.

## Prerequisites

1. macOS (tested on Sonoma 14.x)
2. Python 3.9 or higher
3. pip package manager

## Installation

```bash
# Create a virtual environment (recommended)
python -m venv dreamer_env
source dreamer_env/bin/activate

# Install the right JAX/jaxlib versions for macOS compatibility
pip install jax==0.4.26 jaxlib==0.4.26

# Install other requirements
pip install -r requirements.txt
```

## Running DreamerV3

Use the provided helper script to run any DreamerV3 command:

```bash
# Basic usage
./macos_run.sh dreamerv3/main.py --configs defaults

# With specific options
./macos_run.sh dreamerv3/main.py --configs crafter --run.train_ratio 32
```

The helper script automatically sets the necessary environment variables:
- `JAX_PLATFORMS=cpu`: Forces JAX to use CPU backend
- `SYSTEM_VERSION_COMPAT=0`: Fixes compatibility issues on macOS Sonoma

## Apple GPU (Metal) Support

If you want to try using your Mac's GPU via Metal (experimental):

```bash
# Install Metal plugin
pip install jax-metal
pip install jax==0.4.26 jaxlib==0.4.26
pip install ml_dtypes==0.2.0

# Run with Metal support
ENABLE_PJRT_COMPATIBILITY=1 python dreamerv3/main.py --configs defaults
```

Note: Metal support is experimental and may not work for all models/configurations.

## Troubleshooting

If you encounter JAX backend initialization errors:

1. Make sure you're using the correct JAX/jaxlib versions (0.4.26)
2. Use the `./macos_run.sh` helper script
3. Clear Python cache with `find . -name "__pycache__" -type d -exec rm -rf {} +`
4. If using Metal, set `ENABLE_PJRT_COMPATIBILITY=1`

For more information, see the main README.md file. 