#!/bin/bash
# Helper script for running DreamerV3 on macOS

# Add a function to clean up and exit gracefully
cleanup_and_exit() {
  local exit_code=$1
  if [ $exit_code -ne 0 ]; then
    echo ""
    echo "======================================================"
    echo "Script exited with error code $exit_code"
    echo ""
    echo "If you encounter JAX backend initialization errors:"
    echo "  1. Make sure you're using jax==0.4.26 jaxlib==0.4.26"
    echo "  2. For Apple Silicon: install jax-metal for GPU acceleration"
    echo "  3. Try running with: SYSTEM_VERSION_COMPAT=0 JAX_PLATFORMS=cpu python your_script.py"
    echo ""
    echo "For more help, see the MACOS_QUICKSTART.md file"
    echo "======================================================"
  else
    echo ""
    echo "======================================================"
    echo "Script completed successfully!"
    echo "======================================================"
  fi
  exit $exit_code
}

# Handle interrupts gracefully
trap 'cleanup_and_exit 0' INT TERM

# Display header
echo "======================================================"
echo "      DreamerV3 macOS Runner Helper Script            "
echo "======================================================"

# Detect macOS architecture
if [ "$(uname -m)" = "arm64" ]; then
  APPLE_SILICON=true
  echo "Detected Apple Silicon (M1/M2/M3) Mac"
else
  APPLE_SILICON=false
  echo "Detected Intel Mac"
fi

# Default configuration
if [ "$APPLE_SILICON" = true ]; then
  # Default to Metal on Apple Silicon
  JAX_PLATFORM=${JAX_PLATFORM:-metal}
else
  # Default to CPU on Intel
  JAX_PLATFORM=${JAX_PLATFORM:-cpu}
fi

DISABLE_PROFILER=true              # Disable JAX profiler by default on macOS to prevent crashes
MEM_FRACTION=0.8                   # Default memory fraction for Metal (80% of GPU memory)

# Parse any specific script flags
ARGS=()
PLATFORM_SET=false                 # Flag to track if platform is explicitly set
MEM_FRACTION_SET=false             # Flag to track if memory fraction is explicitly set

print_help() {
  echo "Usage: $0 [options] <script.py> [arguments]"
  echo ""
  echo "Options:"
  echo "  --jax.platform=cpu|metal    Set JAX platform (default: metal on Apple Silicon, cpu on Intel)"
  echo "  --metal.memory=0.1-0.95     Set memory fraction for Metal backend (default: 0.8)"
  echo "  --enable-profiler           Enable JAX profiler (prevents it from being disabled)"
  echo "  --disable-profiler          Disable JAX profiler (default)"
  echo "  --help                      Display this help message"
  echo ""
  echo "Metal backend is only available on Apple Silicon (M1/M2/M3) Macs."
  echo "For Intel Macs, only the CPU backend is supported."
  echo ""
  echo "Examples:"
  echo "  $0 dreamerv3/main.py --configs crafter"
  echo "  $0 --jax.platform=cpu dreamerv3/main.py --configs defaults"
  echo "  $0 --metal.memory=0.7 dreamerv3/main.py --configs defaults"
  echo ""
  echo "For more options, see README.md and MACOS_QUICKSTART.md"
  exit 0
}

# Parse command line arguments
ENABLE_PROFILER_AFTER=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help)
      # Only handle script help (no further args needed)
      print_help
      ;;
    --jax.platform=*)
      JAX_PLATFORM="${1#*=}"
      PLATFORM_SET=true
      shift
      ;;
    --metal.memory=*)
      MEM_FRACTION="${1#*=}"
      MEM_FRACTION_SET=true
      shift
      ;;
    --enable-profiler)
      DISABLE_PROFILER=false
      shift
      ;;
    --disable-profiler)
      DISABLE_PROFILER=true
      shift
      ;;
    *)
      # Check if it's a script file
      if [[ "$1" == *".py" ]]; then
        # It's a script, remember it and check later args
        SCRIPT_FILE="$1"
        ARGS+=("$1")
        shift
        
        # Process all remaining arguments
        while [[ $# -gt 0 ]]; do
          # Check for profiler flags after script name
          if [[ "$1" == "--enable-profiler" ]]; then
            DISABLE_PROFILER=false
            ENABLE_PROFILER_AFTER=true
            shift
          elif [[ "$1" == "--disable-profiler" ]]; then
            DISABLE_PROFILER=true
            shift
          else
            # Add all other args normally
            ARGS+=("$1")
            shift
          fi
        done
        break
      else
        # Not a script, add as regular arg
        ARGS+=("$1")
        shift
      fi
      ;;
  esac
done

# Validate platform choice
if [ "$JAX_PLATFORM" = "metal" ] && [ "$APPLE_SILICON" = false ]; then
  echo "Warning: Metal backend is only available on Apple Silicon Macs."
  echo "Switching to CPU backend for Intel Mac."
  JAX_PLATFORM="cpu"
fi

if [ "$JAX_PLATFORM" != "cpu" ] && [ "$JAX_PLATFORM" != "metal" ]; then
  echo "Error: Invalid platform '$JAX_PLATFORM'. Supported platforms on macOS: 'cpu', 'metal'"
  echo "For Intel Macs, only 'cpu' is supported."
  echo "For Apple Silicon Macs, both 'cpu' and 'metal' are supported."
  cleanup_and_exit 1
fi

# Validate memory fraction
if [[ "$MEM_FRACTION_SET" = true ]]; then
  if ! [[ "$MEM_FRACTION" =~ ^0*([0-9])(\.[0-9]+)?$ ]] || (( $(echo "$MEM_FRACTION < 0.1" | bc -l) )) || (( $(echo "$MEM_FRACTION > 0.95" | bc -l) )); then
    echo "Error: Metal memory fraction must be between 0.1 and 0.95"
    cleanup_and_exit 1
  fi
fi

# Set environment variables for JAX on macOS
export JAX_PLATFORMS=$JAX_PLATFORM
export SYSTEM_VERSION_COMPAT=0  # Needed for some versions of macOS Sonoma
export CUDA_VISIBLE_DEVICES=""   # Explicitly disable CUDA on macOS

# Configure Metal-specific settings
if [ "$JAX_PLATFORM" = "metal" ]; then
  export ENABLE_PJRT_COMPATIBILITY=1
  export XLA_PYTHON_CLIENT_MEM_FRACTION=$MEM_FRACTION
  export XLA_PYTHON_CLIENT_ALLOCATOR=platform
  
  # Set up XLA flags for Metal
  XLA_FLAGS_ARRAY=(
    "--xla_metal_enable_async=true"
    "--xla_gpu_enable_latency_hiding_scheduler=true"
    "--xla_gpu_enable_command_buffer_caching=true"
    "--xla_gpu_enable_command_buffer_barriers=true"
  )
  
  # Add or remove profiler flags based on setting
  if [ "$DISABLE_PROFILER" = true ]; then
    XLA_FLAGS_ARRAY+=("--xla_gpu_disable_profiler=true")
  else
    XLA_FLAGS_ARRAY+=("--xla_gpu_enable_profiler=true")
  fi
  
  # Join the array into a space-separated string
  export XLA_FLAGS="${XLA_FLAGS_ARRAY[*]}"
else
  # CPU backend flags
  if [ "$DISABLE_PROFILER" = true ]; then
    export XLA_FLAGS="--xla_force_host_platform_device_count=1 --xla_cpu_enable_xprof=false"
  else
    export XLA_FLAGS="--xla_force_host_platform_device_count=1 --xla_cpu_enable_xprof=true"
  fi
fi

# Explicitly add platform argument to ensure config.yaml gets the correct value
if [ "$PLATFORM_SET" = true ]; then
  ARGS+=("--jax.platform=$JAX_PLATFORM")
fi

# Check if Python is available
if ! command -v python &> /dev/null; then
  echo "Error: Python is not installed or not in PATH"
  cleanup_and_exit 1
fi

# Check Python version (need 3.9+)
PYTHON_VERSION=$(python --version | cut -d " " -f 2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]); then
  echo "Error: Python 3.9+ is required, but you have Python $PYTHON_VERSION"
  cleanup_and_exit 1
fi

echo "Using Python $PYTHON_VERSION"

# Clear any Python bytecode cache that might affect the imports
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

# Set PYTHONPATH to include the current directory
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# Run the specified script
if [ ${#ARGS[@]} -eq 0 ]; then
  echo "Usage: $0 [options] <script.py> [arguments]"
  echo "For help, use: $0 --help"
  cleanup_and_exit 1
fi

# Check if we need to handle a help request specially
if [[ "${ARGS[*]}" =~ -h || "${ARGS[*]}" =~ --help || "${ARGS[*]}" =~ -help ]]; then
  if [[ "${ARGS[*]}" =~ dreamerv3/main.py || "${ARGS[*]}" =~ dreamerv3/main ]]; then
    echo "Detected help request for DreamerV3. Adjusting command to properly handle help."
    # Find the script position
    for i in "${!ARGS[@]}"; do
      if [[ "${ARGS[i]}" == *"dreamerv3/main.py"* || "${ARGS[i]}" == *"dreamerv3/main"* ]]; then
        # Add --configs default right after the script name
        ARGS=(${ARGS[@]:0:$i+1} "--configs" "defaults" "--help")
        break
      fi
    done
  fi
fi

echo "Running with:"
echo " - JAX_PLATFORMS=$JAX_PLATFORMS"
echo " - SYSTEM_VERSION_COMPAT=$SYSTEM_VERSION_COMPAT" 
echo " - CUDA is explicitly disabled with CUDA_VISIBLE_DEVICES=''"
if [ "$JAX_PLATFORM" = "metal" ]; then
  echo " - Metal backend enabled with:"
  echo "   - ENABLE_PJRT_COMPATIBILITY=1"
  echo "   - XLA_PYTHON_CLIENT_MEM_FRACTION=$MEM_FRACTION"
fi
echo " - JAX profiler is $([ "$DISABLE_PROFILER" = true ] && echo "disabled" || echo "enabled")"
echo " - XLA_FLAGS=\"$XLA_FLAGS\""
echo ""

# Execute the provided Python script with any additional arguments
echo "======================================================"
echo "Starting DreamerV3 with command: python ${ARGS[@]}"
echo "======================================================"

python "${ARGS[@]}"
cleanup_and_exit $? 