# DreamerV3 macOS Quickstart Guide

This quickstart guide helps you get DreamerV3 running on macOS systems, addressing common issues with JAX on Apple hardware.

## Important Notes for macOS Users

- **CUDA is not supported on macOS** - Apple computers do not support NVIDIA CUDA
- JAX on macOS must use either CPU or Apple's Metal GPU plugin (for Apple Silicon Macs)
- Our patches ensure that CUDA is properly disabled and CPU backend is used by default

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
pip install jax==0.4.26 jaxlib==0.4.26 jax-metal

# Install other requirements
pip install -r requirements.txt
```

## Running DreamerV3 on CPU

Use the provided helper script to run any DreamerV3 command:

```bash
# Basic usage
./macos_run.sh dreamerv3/main.py --configs defaults

# With specific options
./macos_run.sh dreamerv3/main.py --configs crafter --run.train_ratio 32
```

The helper script automatically sets the necessary environment variables:
- `JAX_PLATFORMS=cpu`: Forces JAX to use CPU backend
- `CUDA_VISIBLE_DEVICES=""`: Explicitly disables CUDA
- `SYSTEM_VERSION_COMPAT=0`: Fixes compatibility issues on macOS Sonoma

## Apple GPU (Metal) Support

If you want to try using your Mac's GPU via Metal on Apple Silicon Macs (experimental):

```bash
# 1. Install Metal plugin and compatible JAX versions
pip install jax-metal
pip install jax==0.4.26 jaxlib==0.4.26
pip install ml_dtypes==0.2.0

# 2. Run with Metal support (note the environment variable)
ENABLE_PJRT_COMPATIBILITY=1 python dreamerv3/main.py --configs defaults
```

**Important Metal Notes:**
- Metal support is experimental and may not work for all models
- Metal plugin only works on Apple Silicon (M1/M2/M3) Macs, not Intel Macs
- Training large models may be slower or fail with out-of-memory errors
- Set `ENABLE_PJRT_COMPATIBILITY=1` to prevent API version mismatch errors

## Known Issues and Fixes

### JAX Profiler Crashes

The JAX profiler can cause segmentation faults on macOS. This has been fixed in our helper scripts:

- By default, the profiler is now disabled in `macos_run.sh` to prevent crashes
- You can explicitly enable/disable the profiler with `--enable-profiler` or `--disable-profiler` flags
- We've added safety checks in the agent code to prevent crashes if the profiler is enabled

Example of running with profiler disabled (default):
```bash
./macos_run.sh dreamerv3/main.py --configs defaults
```

Example of running with profiler enabled (use with caution):
```bash
./macos_run.sh --enable-profiler dreamerv3/main.py --configs defaults
```

### Scope Viewer Permission Issues

If you encounter permission errors when running the scope viewer:

```
PermissionError: [Errno 13] Permission denied: '.../scope-0.6.3-py3.11.egg/scope/viewer/dist/index.html'
```

Use our `scope_fix.sh` script to fix the permissions:

```bash
# Fix permissions for scope viewer
./scope_fix.sh
```

This script will:
1. Check if scope is installed and working
2. If not, offer to install it for the current user or in a virtual environment
3. Fix permissions if needed using sudo
4. Provide commands to run the scope viewer

After fixing, you can run the scope viewer normally:
```bash
python -m scope.viewer --basedir ~/logdir --port 8200
```

Alternatively, use the `macos_scope.sh` script which includes workarounds for both the permission issue and JAX conflicts:
```bash
./macos_scope.sh --basedir ~/logdir --port 8200
```

## Troubleshooting

If you encounter JAX backend initialization errors:

1. **Use the right JAX versions**: 
   ```bash
   pip install jax==0.4.26 jaxlib==0.4.26
   ```

2. **Run with the helper script**:
   ```bash
   ./macos_run.sh your_script.py
   ```

3. **Clear Python cache**:
   ```bash
   find . -name "__pycache__" -type d -exec rm -rf {} +
   find . -name "*.pyc" -delete
   ```

4. **Set environment variables manually**:
   ```bash
   SYSTEM_VERSION_COMPAT=0 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" python your_script.py
   ```

5. **For Metal GPU support**:
   ```bash
   ENABLE_PJRT_COMPATIBILITY=1 python your_script.py
   ```

For more information on JAX with Metal, see:
- [Apple's JAX on Metal documentation](https://developer.apple.com/metal/jax/)
- [JAX GitHub issues about Metal](https://github.com/google/jax/issues) 