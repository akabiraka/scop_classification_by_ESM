#!/usr/bin/sh

#SBATCH --job-name=scop-ESM
#SBATCH --output=/scratch/akabir4/scop_classification_by_ESM/outputs/argo_logs/scop-%j.out
#SBATCH --error=/scratch/akabir4/scop_classification_by_ESM/outputs/argo_logs/scop-%j.err
#SBATCH --mail-user=<akabir4@gmu.edu>
#SBATCH --mail-type=BEGIN,END,FAIL

##---------------------------CPU jobs-----------------------------
##SBATCH --partition=normal                  # submit   to the normal(default) partition
##SBATCH --cpus-per-task=8                   # Request n   cores per node
##SBATCH --nodes=1                          # Request N nodes
##SBATCH --mem=16000MB                # Request nGB RAM per core
##SBATCH --array=0-10                         # distributed array job   

##python generators/DownloadCleanFasta.py
##python generators/data_helper.py
##python generators/Features.py


##---------------------------GPU jobs-----------------------------
## gpu
#SBATCH --partition=gpuq                    # the DGX only belongs in the 'gpu'  partition
#SBATCH --qos=gpu                           # need to select 'gpu' QoS
#SBATCH --ntasks-per-node=1                 # up to 128; 
#SBATCH --gres=gpu:A100.40gb:1              # up to 8; only request what you need
#SBATCH --mem=32000MB               # memory per CORE; total memory is 1 TB (1,000,000 MB)

##nvidia-smi
##python models/train_val.py
python models/eval.py
