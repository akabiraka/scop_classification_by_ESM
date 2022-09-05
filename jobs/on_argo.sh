#!/usr/bin/sh

#SBATCH --job-name=scop-ESM
#SBATCH --output=/scratch/akabir4/scop_classification_by_ESM/outputs/argo_logs/scop-%j.out
#SBATCH --error=/scratch/akabir4/scop_classification_by_ESM/outputs/argo_logs/scop-%j.err
#SBATCH --mail-user=<akabir4@gmu.edu>
#SBATCH --mail-type=BEGIN,END,FAIL

##cpu jobs
##SBATCH --partition=all-LoPri
##SBATCH --cpus-per-task=4
##SBATCH --mem=16000MB

##python files for CPU jobs
##python generators/DownloadCleanFasta.py

##GPU jobs
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --mem=32000MB

##nvidia-smi
##python files for GPU jobs
python models/train_val.py
##python models/eval.py