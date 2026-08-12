"""Render a trained policy's rollouts to an mp4.

Headless MuJoCo rendering: osmesa fails on this cluster (no system OSMesa
lib), egl works from the login node without a GPU (software/virtual EGL
context) -- see MUJOCO_GL below, must be set before mujoco is imported.

Usage:
    python record_video.py --run-dir runs/reach/sparse_her/seed0 \
        --out results/videos/reach_sparse_her.mp4 --episodes 3
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import json
from pathlib import Path

import gymnasium
import gymnasium_robotics  # noqa: F401
import imageio
import numpy as np
from stable_baselines3 import SAC

import env_fix

env_fix.apply_reset_fix()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True,
                   help="run directory containing config.json + best_model.zip")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=123, help="rollout seed (distinct from training)")
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--checkpoint", default="best_model", choices=("best_model", "final_model"))
    return p.parse_args()


def main():
    args = parse_args()
    config = json.loads((args.run_dir / "config.json").read_text())
    env_id = config["env_id"]

    env = gymnasium.make(env_id, render_mode="rgb_array")
    # env= is required to load HerReplayBuffer checkpoints (SB3 needs it to
    # reconstruct the buffer's compute_reward hook); harmless for vanilla runs.
    model = SAC.load(args.run_dir / f"{args.checkpoint}.zip", env=env, device="cpu")

    frames = []
    successes = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        ep_success = False
        while not done:
            frames.append(env.render())
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_success = ep_success or bool(info.get("is_success", False))
            done = terminated or truncated
        successes.append(ep_success)
    env.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.out, frames, fps=args.fps)
    print(f"[record_video] {config.get('condition', '?')} on {env_id}: "
          f"{args.episodes} episodes, successes={successes} -> {args.out} "
          f"({len(frames)} frames)")


if __name__ == "__main__":
    main()
