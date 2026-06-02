#!/bin/bash
# gpu-connect.sh — SSH into remote GPU machine
# Loads config from .env

set -a; source .env; set +a

echo "Connecting to $GPU_USER@$GPU_HOST:$GPU_PORT"
ssh -i "$GPU_KEY" -p "$GPU_PORT" "$GPU_USER@$GPU_HOST"
