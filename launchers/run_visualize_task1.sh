#!/bin/bash
#SBATCH --job-name=vis_task1
#SBATCH --output=logs/vis_task1_%j.out
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

set -euo pipefail

source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate photogeopose

cd $HOME/PhotoGeoPose

python visualize_task1_results.py \
  --results outputs/topk_results.json \
  --embeddings outputs/embeddings.pt \
  --image-dir /scratch/users/agraillet/images \
  --output-dir outputs/visualizations
