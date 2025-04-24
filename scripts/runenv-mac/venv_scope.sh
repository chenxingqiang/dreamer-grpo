#!/bin/bash
# Helper script for running the scope viewer in a virtual environment

# Display header
echo "======================================================"
echo "    Scope Viewer Virtual Environment Setup Script     "
echo "======================================================"

# Default values
PORT=8200
BASEDIR="$HOME/logdir"
VENV_DIR="$HOME/.dreamerv3_venv"

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
      exit 1
      ;;
  esac
done

# Check if virtual environment exists, create if not
if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment at $VENV_DIR..."
  python3 -m venv "$VENV_DIR"
  
  # Activate the virtual environment and install required packages
  source "$VENV_DIR/bin/activate"
  pip install --upgrade pip
  
  # Install scope directly (not from requirements.txt)
  echo "Installing scope package and dependencies..."
  pip install scope fastapi uvicorn
  
  # Install other required packages
  echo "Installing other required packages..."
  pip install elements orjson mediapy numpy av pillow
else
  # Just activate the existing virtual environment
  source "$VENV_DIR/bin/activate"
fi

# Make sure basedir exists
mkdir -p "$BASEDIR"

echo "Starting scope viewer with:"
echo " - Base directory: $BASEDIR"
echo " - Port: $PORT"
echo " - Virtual environment: $VENV_DIR"
echo ""

# Set environment variables for macOS compatibility
export SYSTEM_VERSION_COMPAT=0

# Run scope viewer in the virtual environment without sudo
python -m scope.viewer --basedir "$BASEDIR" --port "$PORT"

# Deactivate the virtual environment when done
deactivate 