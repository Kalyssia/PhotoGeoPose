#!/bin/bash

#SBATCH --job-name=megaloc_eval
#SBATCH --output=logs/megaloc_eval_b_%j.out
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --ntasks=1

source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate photogeopose

cd $HOME/VPR-methods-evaluation

python3 main.py \
  --method megaloc \
  --image_size 322 322 \
  --database_folder /scratch/users/akayembe/my_dataset/database \
  --queries_folder /scratch/users/akayembe/my_dataset/queries \
  --no_labels \
  --num_preds_to_save 5 \
  --log_dir my_dataset_megaloc