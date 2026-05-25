#!/bin/bash
#SBATCH --job-name=Convnext_CheXpert_Train
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00
#SBATCH --array=1-6
#SBATCH --output=/home/pchalla7/output/convnextfinal%A_%a.out
#SBATCH --error=/home/pchalla7/output/convnextfinal%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pchalla7@asu.edu
#SBATCH --export=NONE

echo "=== Starting ConvNext Training Job (Array Task ${SLURM_ARRAY_TASK_ID}) ==="
date

module purge

# Remove CUDA; no GPU nodes exist
# module load cuda-12.6.1-gcc-12.1.0

# If you have a Python env, activate it here
# source /home/youruser/yourenv/bin/activate

cd "$SLURM_SUBMIT_DIR"

python3 -u final_train.py

echo "=== Training Job Finished (Array Task ${SLURM_ARRAY_TASK_ID}) ==="
date

