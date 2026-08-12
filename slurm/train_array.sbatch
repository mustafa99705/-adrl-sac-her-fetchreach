#!/bin/bash
#SBATCH --job-name=sac-fetchreach
#SBATCH --account=kisski-aai-ss26
#SBATCH --partition=kisski
#SBATCH --gpus=A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-9
#SBATCH --output=logs/%x_%A_%a.out

# Array layout: task 0-4 -> sparse seeds 0-4, task 5-9 -> dense seeds 0-4.
set -euo pipefail

PROJECT_DIR=/mnt/vast-kisski/home/mustafa.asfari01/u28154/SAC-HER
REWARDS=(sparse dense)
N_SEEDS=5
reward=${REWARDS[$((SLURM_ARRAY_TASK_ID / N_SEEDS))]}
seed=$((SLURM_ARRAY_TASK_ID % N_SEEDS))

module load miniforge3
eval "$(conda shell.bash hook)"
conda activate "$PROJECT_DIR/env"

echo "=== node: $(hostname), reward=$reward seed=$seed ==="
nvidia-smi || true
python -c "import torch; print('cuda available:', torch.cuda.is_available())"

python "$PROJECT_DIR/train.py" \
    --reward-type "$reward" \
    --seed "$seed" \
    --total-timesteps 100000 \
    --eval-freq 2000 \
    --n-eval-episodes 20 \
    --log-root "$PROJECT_DIR/runs"
