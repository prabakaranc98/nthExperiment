#!/bin/bash
# new-experiment.sh — scaffold a new experiment
# Usage: bash scripts/new-experiment.sh 001 "resnet-from-scratch" experiments

set -e

NUM=${1:?Usage: new-experiment.sh <num> <name> [experiments|tiny-experiments]}
NAME=${2:?Usage: new-experiment.sh <num> <name> [experiments|tiny-experiments]}
DIR=${3:-experiments}

FOLDER="$DIR/$NUM-$NAME"

mkdir -p "$FOLDER"
cp _templates/experiment.md "$FOLDER/experiment.md"
touch "$FOLDER/log.md"
touch "$FOLDER/reflection.md"

echo "→ Created $FOLDER"
echo "   Edit $FOLDER/experiment.md to start."
