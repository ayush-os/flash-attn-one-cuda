#!/bin/bash

# 1. Install uv if not found
if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

# 2. Create virtual environment
echo "Creating virtual environment..."
uv venv

# 3. Activate for the current subshell
source .venv/bin/activate

# 4. Install dependencies and compile the extension
echo "Installing dependencies and compiling CUDA extension..."
uv pip install torch numpy setuptools
uv pip install -e .

echo "------------------------------------------------"
echo "✅ Setup Complete!"
echo "👉 IMPORTANT: Run 'source .venv/bin/activate' in your terminal now."
echo "------------------------------------------------"