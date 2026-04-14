#!/bin/bash

#SBATCH --job-name=megaloc_infer
#SBATCH --output=logs/megaloc_infer_%j.out
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --ntasks=1

source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate photogeopose

cd $HOME/PhotoGeoPose

python inference.py \
  --db-annotations dataset/splits/annotations_train.json \
  --db-images /scratch/users/agraillet/images \
  --query-annotations dataset/splits/annotations_val.json \
  --query-images /scratch/users/agraillet/images \
  --checkpoint megaloc_finetuned_best.pt \
  --output-dir inference_outputs \
  --device cuda \
  --multi-gpu
