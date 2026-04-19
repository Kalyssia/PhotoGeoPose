#!/bin/bash

#SBATCH --job-name=megaloc_infer
#SBATCH --output=logs/megaloc_infer_%j.out
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --ntasks=1

# To get the log/output files in the correct directory, 
# make sure to create a "logs" directory in the root of the project.

source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate photogeopose

cd $HOME/PhotoGeoPose


python task1.py --multi-gpu
