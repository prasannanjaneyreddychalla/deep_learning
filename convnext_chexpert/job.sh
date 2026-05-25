#!/bin/bash -l
#SBATCH --job-name=convnext_chexpert
#SBATCH --partition=gpu
#SBATCH --qos=public
#SBATCH --nodes=1
# If you want a FULL A100, uncomment the next line and remove --gpus
# SBATCH --gres=gpu:a100:1
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/pchalla7/output/convnext_%A.out
#SBATCH --error=/scratch/pchalla7/output/convnext_%A.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pchalla7@asu.edu
#SBATCH --export=NONE

set -euo pipefail

echo "=== start on $(hostname) ==="; date
module purge
module load mamba/latest
module load cuda-12.6.1-gcc-12.1.0

# Conda activation can choke under 'set -u'
set +u
source activate /home/pchalla7/.conda/envs/chexpert
set -u
hash -r

PY="/home/pchalla7/.conda/envs/chexpert/bin/python"
echo "python path: $PY"
"$PY" -V || { echo "python not found at $PY"; exit 1; }

# Recreate a sane runtime env (since --export=NONE nuked it)
export HOME="/scratch/pchalla7"
export XDG_CACHE_HOME="/scratch/pchalla7/.cache"
export TORCH_HOME="/scratch/pchalla7/.cache/torch"
export WANDB_MODE=offline
export WANDB_DIR="/scratch/pchalla7/wandb"
export MAIL_TO="pchalla7@asu.edu"

# Threading hints for dataloader/BLAS
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false
# Optional: silence pydantic warning spam
export PYTHONWARNINGS="ignore:UnsupportedFieldAttributeWarning"

cd "$SLURM_SUBMIT_DIR"
mkdir -p outputs /scratch/pchalla7/output "$WANDB_DIR" "$TORCH_HOME" "$XDG_CACHE_HOME"

# Optional GPU telemetry (kill on exit)
if command -v nvidia-smi &>/dev/null; then
  nvidia-smi -L || true
  nvidia-smi dmon -s pucm -d 5 > /scratch/pchalla7/output/convnext_${SLURM_JOB_ID}_dmon.log 2>&1 &
  DMON_PID=$!
fi
trap '[[ -n "${DMON_PID:-}" ]] && kill ${DMON_PID} 2>/dev/null || true' EXIT

# Run the training (absolute python path so PATH weirdness can’t bite)
srun -u "$PY" final_train.py

echo "=== done ==="; date

