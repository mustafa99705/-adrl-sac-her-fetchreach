#!/bin/bash
#SBATCH --job-name=sac-her-fetchreach
#SBATCH --account=kisski-aai-ss26
#SBATCH --partition=kisski
#SBATCH --gpus=A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/%x_%A_%a.out

# SAC + HER on sparse FetchReach-v3, seeds 0-4.
set -euo pipefail

PROJECT_DIR=/mnt/vast-kisski/home/mustafa.asfari01/u28154/SAC-HER
seed=$SLURM_ARRAY_TASK_ID

module load miniforge3
eval "$(conda shell.bash hook)"
conda activate "$PROJECT_DIR/env"

echo "=== node: $(hostname), sparse+HER seed=$seed ==="
nvidia-smi || true
python -c "import torch; print('cuda available:', torch.cuda.is_available())"

python "$PROJECT_DIR/train.py" \
    --reward-type sparse \
    --her \
    --seed "$seed" \
    --total-timesteps 100000 \
    --eval-freq 2000 \
    --n-eval-episodes 20 \
    --log-root "$PROJECT_DIR/runs"
