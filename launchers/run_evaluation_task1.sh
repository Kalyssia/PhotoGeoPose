#!/bin/bash

#SBATCH --job-name=eval_task1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate photogeopose

cd $HOME/PhotoGeoPose

python evaluate_task1_results.py