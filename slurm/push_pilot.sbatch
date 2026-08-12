#!/bin/bash
#SBATCH --job-name=sac-push-pilot
#SBATCH --account=kisski-aai-ss26
#SBATCH --partition=kisski
#SBATCH --gpus=A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/%x_%A_%a.out

# Pilot: calibrate step throughput + sanity-check learning on FetchPush-v3
# before committing the full 30-job grid. Task 0 = sparse, task 1 = sparse+HER.
set -euo pipefail

PROJECT_DIR=/mnt/vast-kisski/home/mustafa.asfari01/u28154/SAC-HER
CONDITIONS=(sparse sparse_her)
condition=${CONDITIONS[$SLURM_ARRAY_TASK_ID]}
her_flag=""
[[ "$condition" == "sparse_her" ]] && her_flag="--her"

module load miniforge3
eval "$(conda shell.bash hook)"
conda activate "$PROJECT_DIR/env"

echo "=== node: $(hostname), push pilot condition=$condition ==="
nvidia-smi || true
python -c "import torch; print('cuda available:', torch.cuda.is_available())"

python "$PROJECT_DIR/train.py" \
    --task push \
    --reward-type sparse \
    $her_flag \
    --seed 0 \
    --total-timesteps 200000 \
    --eval-freq 5000 \
    --n-eval-episodes 20 \
    --log-root "$PROJECT_DIR/runs_pilot"
