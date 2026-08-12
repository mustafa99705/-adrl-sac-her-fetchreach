"""Train SAC (optionally with HER) on FetchReach-v3 or FetchPush-v3.

Sparse reward: per-step -1 while the achieved goal is farther than 0.05 m
from the desired goal, 0 once within it. Dense reward: per-step negative
Euclidean distance between achieved and desired goal. Hyperparameters are
identical across reward types, tasks, and HER on/off so that the reward
signal / relabeling strategy is the only experimental variable within each
comparison.

Four conditions, selected via --reward-type {sparse,dense} x --her:
  dense           vanilla SAC, informative reward
  dense_her       SAC + HerReplayBuffer, informative reward
  sparse          vanilla SAC, binary reward
  sparse_her      SAC + HerReplayBuffer, binary reward

--n-sampled-goal controls the HER relabeling ratio (ablation on FetchPush).
"""

import argparse
import json
import platform
import time
from pathlib import Path

import gymnasium
import gymnasium_robotics  # noqa: F401  (registers Fetch envs)
import numpy as np
import stable_baselines3 as sb3
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.her import HerReplayBuffer

import env_fix

env_fix.apply_reset_fix()

# MujocoFetchEnv is the shared base class for both Reach and Push; the reset
# fix above patches it once and applies to every Fetch task.
TASK_ENV_IDS = {
    "reach": {"sparse": "FetchReach-v3", "dense": "FetchReachDense-v3"},
    "push": {"sparse": "FetchPush-v3", "dense": "FetchPushDense-v3"},
}

# rl-baselines3-zoo values for Fetch tasks (gamma/lr/batch/buffer/warmup);
# everything else stays at SB3 defaults. Identical across task/reward/HER.
HYPERPARAMS = dict(
    gamma=0.95,
    learning_rate=1e-3,
    batch_size=256,
    buffer_size=1_000_000,
    learning_starts=1_000,
)


class HerDiagnosticsCallback(BaseCallback):
    """Logs SAC's entropy coefficient and the replay buffer's goal-distance
    distribution (achieved vs. desired goal in sampled minibatches) every
    `log_freq` steps. The goal-distance distribution over the transitions
    actually being trained on is our proxy for the "implicit curriculum"
    HER is expected to induce: early on it is dominated by short,
    just-relabeled distances (near 0, i.e. easy/successful), and should
    broaden as the policy improves and real/far goals become reachable too.

    Works with or without HER (with HER off, replay_buffer.sample still
    works and this simply reports the *fixed* goal distribution / entropy
    coefficient evolution alone).
    """

    def __init__(self, log_freq: int, sample_size: int = 2000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.sample_size = sample_size
        self.timesteps = []
        self.ent_coef = []
        self.goal_dist_mean = []
        self.goal_dist_p10 = []
        self.goal_dist_p50 = []
        self.goal_dist_p90 = []

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq != 0:
            return True
        buffer_size = self.model.replay_buffer.size()
        if buffer_size < 1:
            return True

        # entropy coefficient (auto-tuned: exp(log_ent_coef); fixed: constant)
        if self.model.log_ent_coef is not None:
            ent_coef = float(torch.exp(self.model.log_ent_coef.detach()))
        else:
            ent_coef = float(self.model.ent_coef_tensor)

        batch = self.model.replay_buffer.sample(
            min(self.sample_size, buffer_size), env=self.model._vec_normalize_env
        )
        achieved = batch.observations["achieved_goal"].cpu().numpy()
        desired = batch.observations["desired_goal"].cpu().numpy()
        dist = np.linalg.norm(achieved - desired, axis=1)

        self.timesteps.append(self.num_timesteps)
        self.ent_coef.append(ent_coef)
        self.goal_dist_mean.append(float(dist.mean()))
        self.goal_dist_p10.append(float(np.percentile(dist, 10)))
        self.goal_dist_p50.append(float(np.percentile(dist, 50)))
        self.goal_dist_p90.append(float(np.percentile(dist, 90)))
        return True

    def save(self, path: Path):
        np.savez(
            path,
            timesteps=np.array(self.timesteps),
            ent_coef=np.array(self.ent_coef),
            goal_dist_mean=np.array(self.goal_dist_mean),
            goal_dist_p10=np.array(self.goal_dist_p10),
            goal_dist_p50=np.array(self.goal_dist_p50),
            goal_dist_p90=np.array(self.goal_dist_p90),
        )


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="reach", choices=("reach", "push"))
    p.add_argument("--reward-type", required=True, choices=("sparse", "dense"))
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--her", action="store_true", help="use HER (HerReplayBuffer)")
    p.add_argument("--n-sampled-goal", type=int, default=4,
                   help="HER relabeling ratio (only used with --her)")
    p.add_argument("--total-timesteps", type=int, default=100_000)
    p.add_argument("--eval-freq", type=int, default=2_000)
    p.add_argument("--n-eval-episodes", type=int, default=20)
    p.add_argument("--diag-freq", type=int, default=None,
                   help="entropy-coef / goal-distance logging frequency "
                        "(default: same as --eval-freq)")
    p.add_argument("--log-root", type=Path, default=Path(__file__).parent / "runs")
    args = p.parse_args()
    if args.diag_freq is None:
        args.diag_freq = args.eval_freq
    return args


def main():
    args = parse_args()
    env_id = TASK_ENV_IDS[args.task][args.reward_type]
    condition = f"{args.reward_type}_her" if args.her else args.reward_type
    run_dir = args.log_root / args.task / condition
    if args.her and args.n_sampled_goal != 4:
        run_dir = args.log_root / f"{args.task}_ablation" / f"nsg{args.n_sampled_goal}"
    run_dir = run_dir / f"seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_env = make_vec_env(
        env_id,
        n_envs=1,
        seed=args.seed,
        monitor_dir=str(run_dir),
        monitor_kwargs=dict(info_keywords=("is_success",)),
    )
    eval_env = make_vec_env(
        env_id,
        n_envs=1,
        seed=args.seed + 10_000,
        monitor_kwargs=dict(info_keywords=("is_success",)),
    )

    her_kwargs = {}
    if args.her:
        her_kwargs = dict(
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=dict(
                n_sampled_goal=args.n_sampled_goal,
                goal_selection_strategy="future",
            ),
        )

    model = SAC(
        "MultiInputPolicy",
        train_env,
        seed=args.seed,
        verbose=1,
        tensorboard_log=str(run_dir / "tb"),
        device="auto",
        **HYPERPARAMS,
        **her_kwargs,
    )

    config = dict(
        vars(args),
        log_root=str(args.log_root),
        env_id=env_id,
        condition=condition,
        hyperparams=HYPERPARAMS,
        device=str(model.device),
        versions=dict(
            python=platform.python_version(),
            stable_baselines3=sb3.__version__,
            gymnasium=gymnasium.__version__,
            torch=torch.__version__,
            numpy=np.__version__,
        ),
    )
    config["total_timesteps"] = args.total_timesteps
    config["log_root"] = str(args.log_root)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
    print(f"[train] {env_id} task={args.task} condition={condition} "
          f"n_sampled_goal={args.n_sampled_goal if args.her else '-'} "
          f"seed={args.seed} device={model.device}", flush=True)

    eval_callback = EvalCallback(
        eval_env,
        log_path=str(run_dir),
        best_model_save_path=str(run_dir),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
    )
    diag_callback = HerDiagnosticsCallback(log_freq=args.diag_freq)

    start = time.time()
    model.learn(total_timesteps=args.total_timesteps,
                callback=CallbackList([eval_callback, diag_callback]))
    elapsed = time.time() - start
    print(f"[train] done: {args.total_timesteps} steps in {elapsed:.0f}s "
          f"({args.total_timesteps / elapsed:.0f} steps/s)", flush=True)

    diag_callback.save(run_dir / "her_diagnostics.npz")
    model.save(run_dir / "final_model")
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
