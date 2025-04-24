#!/bin/bash
# macOS-specific helper script for running the scope viewer without JAX conflicts

# Add a function to clean up and exit gracefully
cleanup_and_exit() {
  local exit_code=$1
  echo ""
  echo "======================================================"
  if [ $exit_code -eq 0 ]; then
    echo "Scope viewer closed successfully."
  else
    echo "Scope viewer exited with error code ${exit_code}."
    echo "If you continue to have issues, try:"
    echo "1. Install scope manually: pip install --user scope"
    echo "2. Check permissions: sudo chown -R $(whoami) $(pip3 show scope | grep Location | cut -d' ' -f2)/scope"
    echo "3. Run with sudo (not recommended): sudo -E python -m scope.viewer --basedir ~/logdir --port 8200"
  fi
  echo "======================================================"
  exit $exit_code
}

# Handle interrupts gracefully
trap 'cleanup_and_exit 0' INT TERM

# Display header
echo "======================================================"
echo "      macOS Scope Viewer Helper Script                "
echo "======================================================"

# Default values
PORT=8200
BASEDIR="$HOME/logdir"
VENV_DIR="$HOME/.scope_venv"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --port)
      PORT="$2"
      shift 2
      ;;
    --basedir)
      BASEDIR="$2"
      shift 2
      ;;
    --venv)
      VENV_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--port PORT] [--basedir DIRECTORY] [--venv VENV_PATH]"
      cleanup_and_exit 1
      ;;
  esac
done

# Make sure basedir exists
mkdir -p "$BASEDIR"

echo "Starting scope viewer with:"
echo " - Base directory: $BASEDIR"
echo " - Port: $PORT"
echo ""

# Set environment variables for macOS compatibility and to prevent JAX conflicts
export SYSTEM_VERSION_COMPAT=0
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""

# Check if scope is installed for the user
if ! python3 -c "import scope" &>/dev/null; then
  echo "Scope package not found. Installing for current user..."
  pip3 install --user scope fastapi uvicorn
  
  # Check if installation was successful
  if ! python3 -c "import scope" &>/dev/null; then
    echo "Failed to install scope. Trying virtual environment approach..."
    
    # Create a virtual environment for scope if it doesn't exist
    if [ ! -d "$VENV_DIR" ]; then
      echo "Creating virtual environment at $VENV_DIR..."
      python3 -m venv "$VENV_DIR"
    fi
    
    # Activate the virtual environment and install scope
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip
    pip install scope fastapi uvicorn mediapy
    
    # Run scope directly from the virtual environment
    python -m scope.viewer --basedir "$BASEDIR" --port "$PORT"
    result=$?
    
    # Deactivate virtual environment
    deactivate
    
    # Exit with the result from scope viewer
    cleanup_and_exit $result
  fi
  echo "Scope successfully installed."
fi

# Try direct approach - run scope with environment variables
echo "Attempting to run scope viewer directly..."
python3 -m scope.viewer --basedir "$BASEDIR" --port "$PORT"
result=$?

if [ $result -eq 0 ]; then
  cleanup_and_exit 0
else
  # If that fails, try subprocess approach
  echo "Direct approach failed (exit code $result), trying subprocess isolation..."
  python3 -c "
  import os
  import sys
  import subprocess

  # Set up the command
  cmd = [sys.executable, '-m', 'scope.viewer', '--basedir', '$BASEDIR', '--port', '$PORT']

  # Run in a separate process
  try:
      exit_code = subprocess.call(cmd)
      sys.exit(exit_code)
  except Exception as e:
      print(f'Error running scope viewer: {e}')
      sys.exit(1)
  "
  result=$?
  
  if [ $result -eq 0 ]; then
    cleanup_and_exit 0
  else
    # If subprocess fails too, try virtual environment approach
    echo "Subprocess approach failed (exit code $result). Falling back to virtual environment..."
    
    # Create a virtual environment for scope if it doesn't exist
    if [ ! -d "$VENV_DIR" ]; then
      echo "Creating virtual environment at $VENV_DIR..."
      python3 -m venv "$VENV_DIR"
    fi
    
    # Activate the virtual environment and install scope
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip
    pip install scope fastapi uvicorn mediapy
    
    # Run scope directly from the virtual environment
    python -m scope.viewer --basedir "$BASEDIR" --port "$PORT"
    result=$?
    
    # Deactivate virtual environment
    deactivate
    
    # Exit with the result from scope viewer
    cleanup_and_exit $result
  fi
fi 