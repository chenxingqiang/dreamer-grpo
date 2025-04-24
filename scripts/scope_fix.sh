#!/bin/bash
# Fix script for scope viewer permission issues on macOS

# Add a function to clean up and exit gracefully
cleanup_and_exit() {
  local exit_code=$1
  echo ""
  echo "======================================================"
  if [ $exit_code -eq 0 ]; then
    echo "Scope viewer permissions fixed successfully."
  else
    echo "Failed to fix scope viewer permissions with error code ${exit_code}."
  fi
  echo "======================================================"
  exit $exit_code
}

# Handle interrupts gracefully
trap 'cleanup_and_exit 0' INT TERM

# Display header
echo "======================================================"
echo "      Scope Viewer Permission Fixer for macOS         "
echo "======================================================"

# Get scope location
SCOPE_PATH=$(python3 -c "import scope; print(scope.__path__[0])" 2>/dev/null)
if [ -z "$SCOPE_PATH" ]; then
  echo "Scope package not found. Let's install it."
  echo ""
  echo "Choose an installation method:"
  echo "1. Install for current user (recommended)"
  echo "2. Install in a virtual environment"
  read -p "Enter your choice (1-2): " CHOICE
  
  case $CHOICE in
    1)
      echo "Installing scope for current user..."
      pip3 install --user scope fastapi uvicorn
      SCOPE_PATH=$(python3 -c "import scope; print(scope.__path__[0])" 2>/dev/null)
      if [ -z "$SCOPE_PATH" ]; then
        echo "Failed to install scope. Please try the virtual environment method."
        cleanup_and_exit 1
      fi
      ;;
    2)
      echo "Creating virtual environment at ~/.scope_venv..."
      python3 -m venv ~/.scope_venv
      source ~/.scope_venv/bin/activate
      pip install --upgrade pip
      pip install scope fastapi uvicorn
      
      # Create a launcher script
      cat > ~/scope_launcher.sh << EOL
#!/bin/bash
source ~/.scope_venv/bin/activate
python -m scope.viewer "\$@"
deactivate
EOL
      chmod +x ~/scope_launcher.sh
      
      echo "Virtual environment created and scope installed."
      echo "Use ~/scope_launcher.sh to run scope viewer."
      echo "Example: ~/scope_launcher.sh --basedir ~/logdir --port 8200"
      cleanup_and_exit 0
      ;;
    *)
      echo "Invalid choice. Exiting."
      cleanup_and_exit 1
      ;;
  esac
fi

echo "Found scope installation at: $SCOPE_PATH"

# Check for permissions problems
if [ ! -w "$SCOPE_PATH/viewer/dist" ]; then
  echo "Permission issue detected. Trying to fix..."
  
  # Get parent directory - this is the site-packages or egg directory
  PARENT_DIR=$(dirname "$SCOPE_PATH")
  SCOPE_DIR_NAME=$(basename "$SCOPE_PATH")
  
  echo "Checking parent directory permissions: $PARENT_DIR"
  
  # Ask for sudo to fix permissions
  echo "We need sudo access to fix permissions. You'll be prompted for your password."
  sudo chown -R $(whoami) "$SCOPE_PATH"
  
  if [ $? -eq 0 ]; then
    echo "Permissions fixed successfully!"
    echo ""
    echo "You can now run scope viewer with:"
    echo "python -m scope.viewer --basedir ~/logdir --port 8200"
    cleanup_and_exit 0
  else
    echo "Failed to fix permissions. Let's try installing in user directory."
    
    # Backup plan: Install scope in user's directory
    echo "Installing scope in user directory..."
    pip install --user scope fastapi uvicorn
    
    # Check if it worked
    USER_SCOPE_PATH=$(python -c "import site; print(site.USER_SITE)" 2>/dev/null)
    if [ -n "$USER_SCOPE_PATH" ]; then
      echo "Scope installed in user site-packages: $USER_SCOPE_PATH"
      echo ""
      echo "You can now run scope viewer with:"
      echo "python -m scope.viewer --basedir ~/logdir --port 8200"
      cleanup_and_exit 0
    else
      echo "Failed to install scope in user directory."
      cleanup_and_exit 1
    fi
  fi
else
  echo "No permission issues detected!"
  echo ""
  echo "You can run scope viewer with:"
  echo "python -m scope.viewer --basedir ~/logdir --port 8200"
  cleanup_and_exit 0
fi 