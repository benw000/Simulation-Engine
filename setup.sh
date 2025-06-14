#!/bin/bash

# TODO:
# Make this pip install upgrade then bengine package

# 1. Install requirements
echo "Installing Requirements from requirements.txt ..."
pip install -r requirements.txt
echo "Requirements installed!"

# 2. Setup alias to entrypoint script
ALIAS_NAME="simulation"
SCRIPT_NAME="simulation_engine/main/simulation_engine_entrypoint.py"
# Get parent path from this script's pwd
SCRIPT_PARENT_PATH="$(cd "$(dirname "$0")"; pwd)"
SCRIPT_PATH="$SCRIPT_PARENT_PATH/$SCRIPT_NAME"
echo "$SCRIPT_PATH"
# Get shell config path based for zsh or bash
case "$SHELL" in
  */zsh)  SHELL_CONFIG="$HOME/.zshrc" ;;
  */bash) SHELL_CONFIG="$HOME/.bashrc" ;;
  *) echo "Unsupported shell. Add the alias manually:";
    echo "alias $ALIAS_NAME='python \"$SCRIPT_PATH\"'"; exit 1 ;;
esac

# Check if alias already exists in shell config, if not then set
ALIAS_COMMAND="alias $ALIAS_NAME='python \"$SCRIPT_PATH\"'"
if grep -Fxq "$ALIAS_COMMAND" "$SHELL_CONFIG"; then
    echo "Alias already exists in $SHELL_CONFIG"
else
    if [ -w "$SHELL_CONFIG" ]; then
        echo "$ALIAS_COMMAND" >> "$SHELL_CONFIG"
        echo "Alias added to $SHELL_CONFIG"
    else
        echo "Error: Cannot write to $SHELL_CONFIG — permission denied."
        echo "Printing permissions on $SHELL_CONFIG file:"
        ls -l "$SHELL_CONFIG"
        echo "If $SHELL_CONFIG is owned by 'root' then change permissions via:"
        echo "sudo chown $(whoami):staff $SHELL_CONFIG"
        echo "(After this please rerun ./setup.sh)"
        echo "Or please add this line manually:"
        echo "$ALIAS_COMMAND"
        exit 1
    fi
fi

# Remove alias - commented out
# sed -i.bak "/$ALIAS_COMMAND" "$SHELL_CONFIG"
# Remove from current session
# unalias "$ALIAS_NAME" 2>/dev/null

# Make sure profiles are in sync
echo '[ -f ~/.zshrc ] && source ~/.zshrc' >> ~/.zprofile
echo '[ -f ~/.bashrc ] && source ~/.bashrc' >> ~/.bash_profile

# Reload
echo "Reloading shell config..."
source "$SHELL_CONFIG"

# Tell user to reload session
echo "You can now run '$ALIAS_NAME' from anywhere in your terminal."
echo "Please run the following to activate it in your current session:"
echo "    source $SHELL_CONFIG"

