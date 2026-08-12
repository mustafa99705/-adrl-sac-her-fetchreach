#!/bin/bash
# Create the project conda environment on the login node (needs internet).
# Usage: bash setup_env.sh
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="$PROJECT_DIR/env"

module load miniforge3

if [ ! -d "$ENV_DIR" ]; then
    conda create -p "$ENV_DIR" python=3.11 -y
fi

# conda activate in non-interactive shells
eval "$(conda shell.bash hook)"
conda activate "$ENV_DIR"

pip install -r "$PROJECT_DIR/requirements.txt"

python - <<'EOF'
import gymnasium
import gymnasium_robotics  # noqa: F401  (registers Fetch envs)

for env_id in ("FetchReach-v3", "FetchReachDense-v3"):
    env = gymnasium.make(env_id)
    obs, _ = env.reset(seed=0)
    assert set(obs.keys()) == {"observation", "achieved_goal", "desired_goal"}, obs.keys()
    assert env.spec.max_episode_steps == 50, env.spec.max_episode_steps
    env.close()
    print(f"{env_id}: OK (obs keys, 50-step episodes)")

import torch
print(f"torch {torch.__version__}, CUDA available here: {torch.cuda.is_available()} "
      "(expected False on login node)")
EOF

echo "Environment ready at $ENV_DIR"
