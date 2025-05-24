#!/bin/bash

# 1. Install requirements
echo "Installing Requirements from requirements.txt ..."
pip install -r requirements.txt
echo "Requirements installed!"

# 2. Setup alias to entrypoint script
ALIAS_NAME="simulation"
SCRIPT_NAME="simulation_engine_entrypoint.py"
# Get parent path from this script's pwd
SCRIPT_PARENT_PATH="$(cd "$(dirname "$0")"; pwd)"
SCRIPT_PATH="$SCRIPT_PARENT_PATH/$SCRIPT_NAME"
echo "$SCRIPT_PATH"
# Get shell config path based for zsh or bash
if [ -n "$ZSH_VERSION" ]; then
    SHELL_CONFIG="$HOME/.zshrc"
elif [ -n "$BASH_VERSION" ]; then
    SHELL_CONFIG="$HOME/.bashrc"
else
    echo "Unsupported shell. Add the alias manually:"
    echo "alias $ALIAS_NAME='python \"$SCRIPT_PATH\"'"
    exit 1
fi

# Check if alias already exists in shell config, if not then set
ALIAS_COMMAND="alias $ALIAS_NAME='python \"$SCRIPT_PATH\"'"
if grep -Fxq "$ALIAS_COMMAND" "$SHELL_CONFIG"; then
    echo "Alias already exists in $SHELL_CONFIG"
else
    echo "$ALIAS_COMMAND" >> "$SHELL_CONFIG"
    echo "Alias added to $SHELL_CONFIG"
fi

# Reload
echo "Reloading shell config..."
source "$SHELL_CONFIG"

# Tell user to reload session
echo "You can now run '$ALIAS_NAME' from anywhere in your terminal."
echo "Please run the following to activate it in your current session:"
echo "    source $SHELL_CONFIG"

# Remove alias - commented out
# sed -i.bak "/alias $ALIAS_NAME=.*myscript.py.*/d" "$SHELL_CONFIG"
# Remove from current session
# unalias "$ALIAS_NAME" 2>/dev/null