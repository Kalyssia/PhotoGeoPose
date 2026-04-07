#!/bin/bash

#SBATCH --job-name=megaloc_eval
#SBATCH --output=logs/megaloc_eval_%j.out
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --ntasks=1

source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate photogeopose

cd $HOME/PhotoGeoPose


python task1.py
