#!/bin/bash

#SBATCH --job-name=task1_pipeline
#SBATCH --output=logs/task1_pipeline_%j.out
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --ntasks=1

set -euo pipefail

source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate photogeopose

cd "$HOME/PhotoGeoPose"

echo "[1/3] Running Task 1 inference + retrieval"
python task1.py --multi-gpu --topk 5

echo "[2/3] Generating retrieval visualizations"
python visualize_task1_results.py

# echo "[3/3] Evaluating retrieval metrics"
# python evaluate_task1_results.py

echo "Pipeline complete."
