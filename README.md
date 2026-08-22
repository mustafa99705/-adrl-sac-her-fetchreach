# When and Why Does Hindsight Experience Replay Improve SAC under Sparse Rewards?

RL course project. **Claim under test:** vanilla SAC (Stable-Baselines3)
learns quickly with dense rewards but struggles under sparse rewards, and
Hindsight Experience Replay (HER) recovers dense-like learning from the
sparse signal alone — and this benefit grows with task difficulty.

This file covers the full study (four conditions × two tasks × 10 seeds +
HER relabeling-ratio ablation, see `proposal/proposal.tex`). The
project started as a smaller 3-condition, `FetchReach-v3`-only pilot (5
seeds) — that result is superseded but still valid and is folded into the
numbers below. For the complete chronological history of decisions and
exact job/calibration numbers see `EXPERIMENT_LOG.md`; for the full written
analysis see `report/report.tex`.

Four conditions, identical SAC hyperparameters throughout, run on both tasks:

1. **dense** — vanilla SAC, informative reward (−distance)
2. **dense + HER** — dense reward, with SB3's `HerReplayBuffer`
3. **sparse** — vanilla SAC, binary reward (−1 outside 5 cm, else 0)
4. **sparse + HER** — same sparse reward, but with SB3's `HerReplayBuffer`
   (`n_sampled_goal=4`, `future` strategy by default): failed episodes are
   relabeled in the replay buffer as if the achieved position had been the
   goal, manufacturing successful experience instead of waiting for lucky
   exploration.

`FetchPush-v3` (main+ablation) only ran the two central conditions (sparse,
sparse+HER), per the project proposal.

## Reward semantics

Both env variants of each task are identical except for the reward (verified
in [gymnasium-robotics 1.3.1 source](https://github.com/Farama-Foundation/Gymnasium-Robotics/blob/v1.3.1/gymnasium_robotics/envs/fetch/fetch_env.py)):

| Task | Variant | Env ID | Per-step reward |
|---|---|---|---|
| Reach | sparse | `FetchReach-v3` | `-1` if gripper–goal distance > 0.05 m, else `0` |
| Reach | dense | `FetchReachDense-v3` | negative Euclidean distance to goal |
| Push | sparse | `FetchPush-v3` | `-1` if object–goal distance > 0.05 m, else `0` |
| Push | dense | `FetchPushDense-v3` | negative Euclidean distance to goal |

The sparse reward is per-step, not only-at-episode-end: an always-failing
Reach episode returns exactly −50 (50 steps × −1); Push episodes are also 50
steps. It is still uninformative — the agent receives the same −1 regardless
of how close it is outside the 0.05 m success ball, so there is no gradient
toward the goal until exploration stumbles into it. `info["is_success"]` is
emitted every step.

## Experiment design

- **Conditions:** all four on Reach; sparse and sparse+HER on Push (the
  computationally heavier task).
- **Seeds:** 10 per condition on both tasks (80 runs total across the two
  main grids).
- **Budget:** 100,000 steps/run on Reach, 1,000,000 steps/run on Push
  (calibrated from a 200k-step pilot — see `EXPERIMENT_LOG.md`).
- **Evaluation:** 20 deterministic held-out episodes, every 2,000 steps
  (Reach) or 10,000 steps (Push).
- **Ablation:** HER relabeling ratio `n_sampled_goal` ∈ {1, 4, 8} on
  `FetchPush-v3` sparse+HER, 5 seeds per setting (`n_sampled_goal=4` reuses
  the first 5 seeds of the main sparse+HER condition rather than re-running).
- **Mechanism diagnostics:** every run also logs SAC's auto-tuned entropy
  coefficient α and the distribution of ‖achieved − desired goal‖ in sampled
  replay-buffer minibatches (`her_diagnostics.npz`), at the same cadence as
  evaluation.
- **Hyperparameters (identical across every run** — reward type / task / HER
  on-off is the only experimental variable): `MultiInputPolicy` (Dict obs),
  `gamma=0.95`, `learning_rate=1e-3`, `batch_size=256`, `buffer_size=1e6`,
  `learning_starts=1000`, remaining SB3 defaults (`tau=0.005`, `train_freq=1`,
  `gradient_steps=1`, `ent_coef="auto"`, `net_arch=[256,256]`).
  gamma/lr/batch/buffer follow rl-baselines3-zoo's Fetch settings; we deviate
  by not using VecNormalize (avoids a moving-statistics confound).

## Environment bug found and fixed (`env_fix.py`)

gymnasium-robotics 1.3.x (the only release line with the `-v3` Fetch
environments) has a reset bug: `MujocoFetchEnv._reset_sim` calls
`mj_resetData` — restoring the **XML default** initial pose — but never
restores the initial state captured at construction. On Reach this meant
every episode started with the gripper ~0.55 m from the goal-sampling center
instead of the documented ≤ 0.26 m, a ~2.1× larger initial offset
(0.55/0.26). Upstream fixed exactly this in the v4 environments
(gymnasium-robotics 1.4.0), which however dropped the `-v3` IDs we depend on.

`env_fix.py` backports the upstream fix verbatim (restore
`initial_time/qpos/qvel` after `mj_resetData`) into the 1.3.1 class, so every
`-v3` Fetch task behaves as documented. The patch changes no physics, reward,
or task logic and applies identically to every condition and every task.
`train.py` and `record_video.py` apply it at import time.

On the unpatched Reach environment, sparse SAC sat at exactly 0% success
on all 5 seeds and dense plateaued at ~35% -- see `report/report.tex`
§6 for the corrected, patched results.

## Pinned software stack

`-v3` Fetch environments only exist in gymnasium-robotics 1.3.x (1.4+
replaced them with `-v4`), which forces the rest of the pins — see
`requirements.txt`: Python 3.11, numpy 1.26.4 (<2 required by mujoco 3.1.x),
mujoco 3.1.6, gymnasium 1.1.1, gymnasium-robotics 1.3.1, stable-baselines3
2.6.0, torch 2.7.1 (CUDA 12.6 wheel). Headless rendering for
`record_video.py` needs `MUJOCO_GL=egl` (`osmesa` fails on this cluster: no
system OSMesa lib). `tectonic` and `poppler` (installed into `./env` via
conda-forge, does not touch the pins above) are only needed to compile/preview
`report/report.tex` locally — not part of the experiment pipeline itself.

## Reproduce

On the GWDG/KISSKI cluster (login node has internet, compute nodes don't):

```bash
# 1. one-time environment setup (login node)
bash setup_env.sh

# 2. full experiment on the kisski partition (1×A100 per job)
cd slurm && mkdir -p logs
sbatch reach_full_array.sbatch        # FetchReach, 4 conditions x 10 seeds (25 jobs;
                                       # combine with the original 15-job v1 arrays,
                                       # train_array.sbatch + train_her_array.sbatch,
                                       # for the full 40)
sbatch push_pilot.sbatch              # calibration pilot (2 jobs) -- run first
sbatch push_main_array.sbatch         # FetchPush sparse vs sparse+HER, 10 seeds (20 jobs)
sbatch push_ablation_array.sbatch     # n_sampled_goal in {1,8}, 5 seeds (10 jobs)

# 3. after all jobs finish: figures + summary tables (login node)
module load miniforge3 && conda activate ./env
python plot_results.py                # writes results/figures/{fig1-6,*.csv}
```

A single run can also be launched directly:

```bash
python train.py --task reach --reward-type sparse --seed 0 --total-timesteps 100000
python train.py --task push --reward-type sparse --her --n-sampled-goal 8 \
    --seed 0 --total-timesteps 1000000   # Push + HER + custom relabeling ratio
```

Render a trained policy to video:
```bash
python record_video.py --run-dir runs/reach/sparse_her/seed0 \
    --out results/videos/reach_sparse_her.mp4 --episodes 3
```

Check cluster job status without memorizing `squeue` flags: `~/kstatus.sh`
(aliased `kstatus` in `.bashrc`, works from any directory on this cluster).

Outputs per run in `runs/<task>/<condition>/seed{K}/` (ablation runs at
`runs/<task>_ablation/nsg{K}/seed{K}/`, pilot runs at
`runs_pilot/<task>/<condition>/seed0/`): `monitor.csv` (per-episode
return/length/success during training), `evaluations.npz` (eval timesteps,
returns, successes), `her_diagnostics.npz` (entropy coefficient + goal-distance
percentiles), `best_model.zip`, `final_model.zip`, `config.json` (full config
+ package versions), TensorBoard logs in `tb/`.

## Results

All runs: SLURM array on the `kisski` A100 partition (1 GPU per run). Full
figures in `results/figures/`, per-seed numbers in
`results/figures/summary.csv` and `ablation_summary.csv`. Full write-up
with mechanism analysis in `report/report.tex`.

| | FetchReach-v3 (10 seeds/cond.) | | FetchPush-v3 (10 seeds/cond.) | |
|---|---|---|---|---|
| condition | final success | median steps to 90% | condition | final success | median steps to 90% |
| dense | 100.0% | 6,000 | — | — | — |
| dense+HER | 100.0% | 6,000 | — | — | — |
| sparse | 99.4% | 22,000 | sparse | 5.4% | never |
| sparse+HER | 100.0% | 8,000 | sparse+HER | 90.7% | 350,000 |

- On the **easy task**, every condition eventually succeeds; HER's
  contribution is a ~2.75× reduction in steps-to-90% for the sparse reward,
  not the difference between success and failure — random exploration
  occasionally reaches the goal on its own.
- On the **hard task**, plain sparse-reward SAC extracts essentially nothing
  from a full million steps (5.4% final success, 0/10 seeds ≥90%), while HER
  on the identical reward reaches 90.7% success (9/10 seeds) — HER is the
  difference between the task being learned at all.
- **Mechanism:** SAC's auto-tuned entropy coefficient α collapses ~21× under
  plain sparse reward on Push *while the policy is still failing*
  (premature, overconfident convergence with no corrective signal), while
  HER keeps α roughly 8× higher throughout, sustaining the exploration
  needed to eventually discover successful pushes. On Reach, the trained-on
  goal-distance distribution narrows cleanly over training (~0.10 m →
  ~0.02 m) — a visible implicit curriculum; on Push the same signal is
  present but mostly masked by the ~80% HER-relabeled transitions in the
  pooled metric.
- **Ablation:** relabeling ratio saturates rather than scaling —
  `n_sampled_goal=8` learns somewhat faster (median steps-to-90% 260,000 vs.
  ~380,000–400,000) but `{1,4,8}` converge to a similar final success region.

See `report/report.tex` §5–6 for the full results, mechanism analysis,
and ablation discussion, and §8 for limitations (including two seeds/runs that were
increased beyond the proposal's originally specified 5-seed / {1,4}-ablation
design, and one honest caveat about the goal-distance metric on Push).

### Demo videos

Rollouts of a trained policy on each task/condition (`record_video.py`,
deterministic actions, patched env). GIF preview below (1 episode,
~12 fps); click through for the full mp4 (up to 3 episodes, 25 fps).

<table>
<tr>
<td align="center">
<a href="results/videos/reach_dense.mp4"><img src="https://raw.githubusercontent.com/mustafa99705/-adrl-sac-her-fetchreach/main/results/videos/reach_dense.gif" width="260"></a><br/>Reach, dense
</td>
<td align="center">
<a href="results/videos/reach_sparse.mp4"><img src="https://raw.githubusercontent.com/mustafa99705/-adrl-sac-her-fetchreach/main/results/videos/reach_sparse.gif" width="260"></a><br/>Reach, sparse
</td>
</tr>
<tr>
<td align="center">
<a href="results/videos/reach_sparse_her.mp4"><img src="https://raw.githubusercontent.com/mustafa99705/-adrl-sac-her-fetchreach/main/results/videos/reach_sparse_her.gif" width="260"></a><br/>Reach, sparse+HER
</td>
<td align="center">
<a href="results/videos/push_sparse_her.mp4"><img src="https://raw.githubusercontent.com/mustafa99705/-adrl-sac-her-fetchreach/main/results/videos/push_sparse_her.gif" width="260"></a><br/>Push, sparse+HER
</td>
</tr>
</table>

### References

- Andrychowicz et al., *Hindsight Experience Replay*, NeurIPS 2017.
  [arXiv:1707.01495](https://arxiv.org/abs/1707.01495)
- Plappert et al., *Multi-Goal Reinforcement Learning: Challenging Robotics
  Environments and Request for Research*, 2018.
  [arXiv:1802.09464](https://arxiv.org/abs/1802.09464)
- Haarnoja et al., *Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL
  with a Stochastic Actor*, ICML 2018.
- Raffin et al., *Stable-Baselines3: Reliable RL Implementations*, JMLR 2021.
- gymnasium-robotics 1.3.2 release notes (v4 envs "fixing the state
  initialization bug").
  [Releases](https://github.com/Farama-Foundation/Gymnasium-Robotics/releases)
