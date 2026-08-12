#!/bin/bash
#SBATCH --job-name=sac-push-ablation
#SBATCH --account=kisski-aai-ss26
#SBATCH --partition=kisski
#SBATCH --gpus=A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=07:00:00
#SBATCH --array=0-9%10
#SBATCH --output=logs/%x_%A_%a.out

# HER relabeling-ratio ablation on FetchPush-v3, n_sampled_goal in {1,8},
# 5 seeds each (n_sampled_goal=4 is the main sparse_her condition from
# push_main_array.sbatch, seeds 0-4 -- reused rather than rerun).
# Array layout: 0-4 -> nsg=1 seed 0-4, 5-9 -> nsg=8 seed 0-4.
# Time budget calibrated from push_pilot.sbatch: sparse+HER (nsg=4) ran
# at 54 steps/s over 200k steps; nsg=1 relabels less so should be a
# little faster, nsg=8 relabels more so a little slower -- 07:00:00
# covers both at 1M steps with margin.
set -euo pipefail

PROJECT_DIR=/mnt/vast-kisski/home/mustafa.asfari01/u28154/SAC-HER
TOTAL_TIMESTEPS=1000000
id=$SLURM_ARRAY_TASK_ID
if [ "$id" -lt 5 ]; then nsg=1; seed=$id; else nsg=8; seed=$((id - 5)); fi

module load miniforge3
eval "$(conda shell.bash hook)"
conda activate "$PROJECT_DIR/env"

echo "=== node: $(hostname), push ablation n_sampled_goal=$nsg seed=$seed ==="
nvidia-smi || true

python "$PROJECT_DIR/train.py" \
    --task push \
    --reward-type sparse \
    --her \
    --n-sampled-goal "$nsg" \
    --seed "$seed" \
    --total-timesteps "$TOTAL_TIMESTEPS" \
    --eval-freq 10000 \
    --n-eval-episodes 20 \
    --log-root "$PROJECT_DIR/runs"
