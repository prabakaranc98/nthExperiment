#!/bin/bash
# gpu-sync.sh — rsync experiments to remote GPU
# Usage: bash scripts/gpu-sync.sh [experiment-folder]
# Example: bash scripts/gpu-sync.sh experiments/001-resnet-scratch

set -a; source .env; set +a

TARGET=${1:-experiments}
REMOTE_DIR="~/nthexperiment"

echo "Syncing $TARGET → $GPU_USER@$GPU_HOST:$REMOTE_DIR/$TARGET"
rsync -avz --exclude '.venv' --exclude '__pycache__' --exclude '*.pyc' \
    -e "ssh -i $GPU_KEY -p $GPU_PORT" \
    "$TARGET/" \
    "$GPU_USER@$GPU_HOST:$REMOTE_DIR/$TARGET/"
