#!/bin/bash
#SBATCH --job-name=sac-push-main
#SBATCH --account=kisski-aai-ss26
#SBATCH --partition=kisski
#SBATCH --gpus=A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=07:00:00
#SBATCH --array=0-19%10
#SBATCH --output=logs/%x_%A_%a.out

# FetchPush-v3 main comparison: sparse vs sparse+HER, 10 seeds each.
# Calibrated from push_pilot.sbatch (job 15197304, 2026-08-07): sparse
# ran at 61 steps/s, sparse+HER at 54 steps/s over 200k steps. At 1M
# steps that projects to ~4.6h / ~5.1h of training alone; 07:00:00 adds
# margin for eval overhead and the node-speed variance seen in the
# reach grid (17-38 min for runs nominally the same size).
# Array layout: 0-9 -> sparse seed 0-9, 10-19 -> sparse+HER seed 0-9.
set -euo pipefail

PROJECT_DIR=/mnt/vast-kisski/home/mustafa.asfari01/u28154/SAC-HER
TOTAL_TIMESTEPS=1000000
id=$SLURM_ARRAY_TASK_ID
her=""
seed=$((id % 10))
[ "$id" -ge 10 ] && her="--her"

module load miniforge3
eval "$(conda shell.bash hook)"
conda activate "$PROJECT_DIR/env"

echo "=== node: $(hostname), push sparse her=$her seed=$seed ==="
nvidia-smi || true

python "$PROJECT_DIR/train.py" \
    --task push \
    --reward-type sparse \
    $her \
    --seed "$seed" \
    --total-timesteps "$TOTAL_TIMESTEPS" \
    --eval-freq 10000 \
    --n-eval-episodes 20 \
    --log-root "$PROJECT_DIR/runs"
