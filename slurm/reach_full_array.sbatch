#!/bin/bash
#SBATCH --job-name=sac-reach-full
#SBATCH --account=kisski-aai-ss26
#SBATCH --partition=kisski
#SBATCH --gpus=A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-24
#SBATCH --output=logs/%x_%A_%a.out

# Completes the FetchReach-v3 4-condition x 10-seed grid required by
# proposal_v2. dense/sparse/sparse_her already have seeds 0-4 from the v1
# study (runs/reach/<cond>/seed{0-4}); this array adds seeds 5-9 for those
# three plus all 10 seeds (0-9) for the new dense_her condition.
#
# Array layout:
#   0-4    dense       seed 5-9
#   5-9    sparse      seed 5-9
#   10-14  sparse_her  seed 5-9
#   15-24  dense_her   seed 0-9
set -euo pipefail

PROJECT_DIR=/mnt/vast-kisski/home/mustafa.asfari01/u28154/SAC-HER
id=$SLURM_ARRAY_TASK_ID

if   [ "$id" -lt 5 ];  then reward=dense;  her=""      ; seed=$((id + 5))
elif [ "$id" -lt 10 ]; then reward=sparse; her=""      ; seed=$((id - 5 + 5))
elif [ "$id" -lt 15 ]; then reward=sparse; her="--her" ; seed=$((id - 10 + 5))
else                        reward=dense;  her="--her" ; seed=$((id - 15))
fi

module load miniforge3
eval "$(conda shell.bash hook)"
conda activate "$PROJECT_DIR/env"

echo "=== node: $(hostname), reach reward=$reward her=$her seed=$seed ==="
nvidia-smi || true

python "$PROJECT_DIR/train.py" \
    --task reach \
    --reward-type "$reward" \
    $her \
    --seed "$seed" \
    --total-timesteps 100000 \
    --eval-freq 2000 \
    --n-eval-episodes 20 \
    --log-root "$PROJECT_DIR/runs"
