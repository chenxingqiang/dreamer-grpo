#!/bin/bash
# Helper script for running the scope viewer with the correct permissions

# Display header
echo "======================================================"
echo "        Scope Viewer Helper Script for macOS          "
echo "======================================================"

# Default values
PORT=8080
BASEDIR="$HOME/logdir"

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
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--port PORT] [--basedir DIRECTORY]"
      exit 1
      ;;
  esac
done

echo "Starting scope viewer with:"
echo " - Base directory: $BASEDIR"
echo " - Port: $PORT"
echo ""

# Run scope viewer as the current user but with sudo
# This preserves environment variables and avoids permission issues
sudo -E python3 -m scope.viewer --basedir "$BASEDIR" --port "$PORT" 