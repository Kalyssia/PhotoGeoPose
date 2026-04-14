#!/bin/bash

#SBATCH --job-name=megaloc_infer
#SBATCH --output=logs/megaloc_visu_epoch1_%j.out
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1

source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate photogeopose

cd $HOME/PhotoGeoPose

python visualize_retrieval.py \
  --results inference_baseline_outputs/retrieval_results.json \
  --image-dir /scratch/users/agraillet/images \
  --output-dir retrieval_viz \
  --topk 5 \
  --max-examples 50 \
  --distance-thresh-m 25.0