#!/bin/bash -l
#SBATCH --job-name=swin_node21
#SBATCH --partition=gpu
#SBATCH --qos=public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1                 # or: --gres=gpu:a30:1 (pick one style)
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --hint=nomultithread     # avoids weird HT core binding
#SBATCH --output=/scratch/pchalla7/output/node21_%A.out
#SBATCH --error=/scratch/pchalla7/output/node21_%A.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pchalla7@asu.edu
#SBATCH --export=NONE

set -euo pipefail
echo "=== start on $(hostname) ==="; date

module purge
module load mamba/latest
# Do NOT load a CUDA module unless your cluster requires it for driver/toolchain

# conda/mamba activation under -u
set +u
source activate /home/pchalla7/.conda/envs/node21-swin
set -u
hash -r

PY="/home/pchalla7/.conda/envs/node21-swin/bin/python"
echo "python path: $PY"; "$PY" -V || { echo "python not found at $PY"; exit 1; }

# runtime env
export HOME="/scratch/pchalla7"
export XDG_CACHE_HOME="/scratch/pchalla7/.cache"
export TORCH_HOME="/scratch/pchalla7/.cache/torch"
export WANDB_MODE=offline
export WANDB_DIR="/scratch/pchalla7/wandb"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false
export PYTHONWARNINGS="ignore:UnsupportedFieldAttributeWarning"
export CUDNN_BENCHMARK=1

# kill default CPU-binding drama for this step
export SLURM_CPU_BIND=none

cd "$SLURM_SUBMIT_DIR"
mkdir -p outputs /scratch/pchalla7/output "$WANDB_DIR" "$TORCH_HOME" "$XDG_CACHE_HOME"

if command -v nvidia-smi &>/dev/null; then
  nvidia-smi -L || true
  nvidia-smi || true
  nvidia-smi dmon -s pucm -d 5 > /scratch/pchalla7/output/swin_${SLURM_JOB_ID}_dmon.log 2>&1 & DMON_PID=$!
fi
trap '[[ -n "${DMON_PID:-}" ]] && kill ${DMON_PID} 2>/dev/null || true' EXIT

# IMPORTANT: set DRY_RUN=False in train.py before launching
# Option A: with srun, no CPU bind
srun --cpu-bind=none -u "$PY" train.py

# Option B (alternative): no srun
# "$PY" -u train.py

echo "=== done ==="; date

