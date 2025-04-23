#!/bin/bash
# Helper script for running DreamerV3 on macOS

# Display header
echo "======================================================"
echo "      DreamerV3 macOS Runner Helper Script            "
echo "======================================================"

# Set environment variables for JAX on macOS
export JAX_PLATFORMS=cpu
export SYSTEM_VERSION_COMPAT=0  # Needed for some versions of macOS Sonoma

# Check if Python is available
if ! command -v python &> /dev/null; then
  echo "Error: Python is not installed or not in PATH"
  exit 1
fi

# Check Python version (need 3.9+)
PYTHON_VERSION=$(python --version | cut -d " " -f 2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]); then
  echo "Error: Python 3.9+ is required, but you have Python $PYTHON_VERSION"
  exit 1
fi

echo "Using Python $PYTHON_VERSION"

# Clear any Python bytecode cache that might affect the imports
find . -name "__pycache__" -type d -exec rm -rf {} +
find . -name "*.pyc" -delete

# Set PYTHONPATH to include the current directory
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# Run the specified script
if [ $# -eq 0 ]; then
  echo "Usage: $0 <script.py> [arguments]"
  echo ""
  echo "Example:"
  echo "  $0 dreamerv3/main.py --configs crafter --run.train_ratio 32"
  echo ""
  echo "For more options, see README.md"
  exit 1
fi

echo "Running with JAX_PLATFORMS=$JAX_PLATFORMS and SYSTEM_VERSION_COMPAT=$SYSTEM_VERSION_COMPAT"

# Execute the provided Python script with any additional arguments
echo "======================================================"
echo "Starting DreamerV3 with command: python $@"
echo "======================================================"
python "$@"

# Check exit code
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  echo ""
  echo "======================================================"
  echo "Script exited with error code $EXIT_CODE"
  echo ""
  echo "If you encounter JAX backend initialization errors,"
  echo "ensure you have the correct versions installed:"
  echo "  - pip install jax==0.4.26 jaxlib==0.4.26"
  echo ""
  echo "For more help, see the macOS section in README.md"
  echo "======================================================"
else
  echo ""
  echo "======================================================"
  echo "Script completed successfully!"
  echo "======================================================"
fi

exit $EXIT_CODE 