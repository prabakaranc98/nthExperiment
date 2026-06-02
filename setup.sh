#!/bin/bash
# setup.sh — bootstrap the nthexperiment environment
# Usage: bash setup.sh [--cuda] [--jax] [--mlx] [--all]

set -e

echo "── nthExperiment Setup ──────────────────────────────────"

# Check uv is installed
if ! command -v uv &> /dev/null; then
    echo "→ Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

echo "→ Creating virtual environment..."
uv venv .venv --python 3.11

echo "→ Installing base dependencies..."
uv pip install -e "."

# Optional extras
for arg in "$@"; do
    case $arg in
        --cuda)
            echo "→ Installing PyTorch (CUDA 12.1)..."
            uv pip install torch torchvision torchaudio \
                --index-url https://download.pytorch.org/whl/cu121
            ;;
        --torch)
            echo "→ Installing PyTorch (CPU)..."
            uv pip install -e ".[torch]"
            ;;
        --jax)
            echo "→ Installing JAX (CUDA 12)..."
            uv pip install -U "jax[cuda12]"
            ;;
        --mlx)
            echo "→ Installing MLX (Apple Silicon)..."
            uv pip install -e ".[mlx]"
            ;;
        --hf)
            echo "→ Installing HuggingFace stack..."
            uv pip install -e ".[hf]"
            ;;
        --tracking)
            echo "→ Installing experiment tracking..."
            uv pip install -e ".[tracking]"
            ;;
        --all)
            echo "→ Installing all extras..."
            uv pip install -e ".[all]"
            ;;
    esac
done

# Set up .env if not present
if [ ! -f .env ]; then
    echo "→ Creating .env from template..."
    cp .env.example .env
    echo "   Edit .env and add your API keys."
fi

# Register Jupyter kernel
echo "→ Registering Jupyter kernel..."
uv run python -m ipykernel install --user --name nthexperiment --display-name "nthExperiment"

echo ""
echo "── Done ──────────────────────────────────────────────────"
echo "   Activate: source .venv/bin/activate"
echo "   Or run:   uv run python your_script.py"
echo "   Jupyter:  uv run jupyter lab"
