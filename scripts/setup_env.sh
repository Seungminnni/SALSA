#!/usr/bin/env bash
set -euo pipefail

# Helper script to set up a Python environment for SALSA
# NOTE: review the commands and run them manually in your shell.

if command -v conda >/dev/null 2>&1; then
  echo "Creating conda environment from requirements.txt (conda)..."
  conda create --name lattice_env --file requirements.txt -y
  echo "Activate it with: conda activate lattice_env"
  echo "If PyTorch wheel doesn't include the correct CUDA version, install the matching PyTorch wheel as described in INSTALL.md"
else
  echo "Conda not found — creating venv and installing pip packages..."
  python3 -m venv venv
  source venv/bin/activate
  pip install -r requirements-pip.txt
  echo "Install PyTorch wheel with matching CUDA version as in INSTALL.md"
fi

# Check git-lfs
if ! command -v git-lfs >/dev/null 2>&1; then
  echo "git-lfs is not installed. Install it (apt/brew) and run 'git lfs install'"
else
  echo "git-lfs available"
  git lfs install || true
fi

echo "Setup script finished. Read INSTALL.md for optional steps (Apex, git-lfs migrate, etc.)"
