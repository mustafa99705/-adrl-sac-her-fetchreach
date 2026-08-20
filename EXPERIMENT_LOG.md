# Experiment Log — SAC-HER on FetchReach/FetchPush

Running record of everything done on this project, in order, with the numbers
that will feed the final report. `README.md` documents the v1 methodology in
prose; this file is the chronological ground truth (decisions, job IDs,
timings, raw results) behind both v1 and the v2 expansion.

Project: *When and Why Does Hindsight Experience Replay Improve Soft
Actor-Critic under Sparse Rewards?* — Mustafa Asfari, Mohammad Alkhozami.
Proposal: `proposal/proposal.tex`.

---

## Phase 1 — v1 pilot study (completed before this log started)

**Scope:** 3 conditions × 5 seeds × FetchReach-v3 only, 100,000 steps/run.

**Conditions:** dense, sparse, sparse+HER (`n_sampled_goal=4`, `future`
strategy). Identical SAC hyperparameters across all three — see
[Hyperparameters](#hyperparameters-identical-across-all-runs) below.

**Bug found and fixed — `env_fix.py`:** gymnasium-robotics 1.3.x's
`MujocoFetchEnv._reset_sim` calls `mj_resetData` (XML default arm pose) but
never restores the initial state captured at construction, so every episode
started ~0.55 m from the goal-sampling center instead of the documented
≤0.26 m — task was ~2.5× harder than specified, some goals barely reachable
in 50 steps. `env_fix.py` backports the upstream v4 fix into the 1.3.1 class
(restore `initial_time/qpos/qvel` after `mj_resetData`); changes no physics or
reward logic. Applied at import time in `train.py`. Broken-reset results kept
in `runs_broken_reset/` and `results/figures_broken_reset/` for comparison
(sparse SAC was stuck at 0% success on all 5 seeds under the bug).

**v1 results** (`results/figures/summary.csv`, 5 seeds/condition):

| condition | final success (mean) | median steps to 90% success | success-curve AUC (mean) |
|---|---|---|---|
| dense | 1.00 | 6,000 | 0.944 |
| sparse | 0.98 | 22,000 | 0.804 |
| sparse + HER | 1.00 | 8,000 | 0.931 |

Sparse SAC eventually reaches dense-level final success, but needs **~3.7×**
more steps to get there; sparse+HER recovers dense-like sample efficiency
using only the uninformative sparse signal. This was the core v1 finding and
motivated the larger v2 study (does the gap widen on a harder task?).

---

## Phase 2 — v2 expansion (this session)

`proposal_v2.tex` scoped up the study: 4 conditions (add dense+HER), 10 seeds
(was 5), a second harder task (FetchPush-v3), an HER relabeling-ratio ablation
(`n_sampled_goal` ∈ {1,4,8}), and mechanism diagnostics (entropy-coefficient
evolution, relabeled-goal-distance distribution) to explain *why* HER helps,
not just *whether*.

### Code changes

- **`train.py`** rewritten to take `--task {reach,push}`, support the
  `dense_her` condition, accept `--n-sampled-goal` for the ablation, and log
  diagnostics via a new `HerDiagnosticsCallback` (entropy coefficient +
  goal-distance percentiles every `--diag-freq` steps, saved to
  `her_diagnostics.npz`). Run directory layout:
  `runs/<task>/<condition>/seed<N>/`, with ablation runs (`n_sampled_goal≠4`)
  routed to `runs/<task>_ablation/nsg<k>/seed<N>/` so they never collide with
  the main-grid `sparse_her` (`nsg=4`) runs.
- **`record_video.py`** (new) — renders held-out rollouts of a trained
  checkpoint to mp4 via headless MuJoCo. `osmesa` fails on this cluster (no
  system OSMesa lib); `MUJOCO_GL=egl` works from the login node with no GPU
  needed for rendering.
- **`plot_results.py`** rewritten for two tasks + the new diagnostic/ablation
  plots (fig1–fig6, see its docstring); verified against the existing v1 data
  before trusting it on new data.
- Existing v1 data (`runs/{dense,sparse,sparse_her}/seed{0-4}`) migrated into
  the new layout at `runs/reach/<condition>/seed{0-4}/` with no data loss
  (verified file counts before/after).
- All new code paths (Push env registration, dense+HER, ablation
  `n_sampled_goal`, diagnostics logging) smoke-tested on the login node with
  tiny step counts before anything was submitted to the cluster.

### Hyperparameters (identical across all runs, v1 and v2)

From rl-baselines3-zoo's tuned Fetch settings; everything else is
Stable-Baselines3 defaults (`tau=0.005`, `train_freq=1`, `gradient_steps=1`,
`ent_coef="auto"`, `net_arch=[256,256]`, `MultiInputPolicy`, no VecNormalize):

```
gamma=0.95  learning_rate=1e-3  batch_size=256
buffer_size=1_000_000  learning_starts=1_000
```

### Cluster infrastructure notes

- KISSKI HPC (GWDG), shared course allocation `kisski-aai-ss26` (~38
  students). Login node (`glogin10`) has no GPU — only place code was written
  and smoke-tested. All real training runs through SLURM on the `kisski`
  partition (22 nodes × 4× A100, GPUs allocated per-job via `--gpus=A100:1`).
  `kisski-h100` is the only other partition this account can use; checked and
  it was *more* congested (359 queued jobs on 8 nodes vs. 79 on 22) so never
  used. Shared filesystem (VAST) means results land directly in the project
  folder with no manual transfer between login and compute nodes.
- Built `~/kstatus.sh` (aliased `kstatus` in `.bashrc`) — one-command SLURM
  dashboard: own jobs, ETA, partition load, jobs finished in the last 24h.
- Published two Claude Artifacts (private, links below): a side-by-side video
  comparison of trained policies, and a snapshot status dashboard with a
  diagram of the login-node → SLURM-queue → GPU-node → shared-storage flow.

---

## Job history

| job ID | script | purpose | array size | submitted | outcome |
|---|---|---|---|---|---|
| 15197304 | `push_pilot.sbatch` | calibrate FetchPush throughput/success before committing full grid | 2 (sparse, sparse+HER) | 2026-08-07 16:03 | **COMPLETED** — see below |
| 15197307 | `reach_full_array.sbatch` | finish FetchReach v2 grid: seeds 5–9 for dense/sparse/sparse+HER, all seeds 0–9 for new dense+HER | 25 | 2026-08-07 16:04 | **COMPLETED**, all 25 tasks, no errors |
| 15207358 | `push_main_array.sbatch` | FetchPush main comparison: sparse vs sparse+HER | 20 (10 seeds each) | 2026-08-08 12:03 | **COMPLETED**, all 20 tasks (~2h45m–3h05m each) |
| 15207359 | `push_ablation_array.sbatch` | HER relabeling ratio ablation, n_sampled_goal ∈ {1,8} (nsg=4 reused from job 15207358's sparse+HER) | 10 (5 seeds each) | 2026-08-08 12:03 | **COMPLETED**, all 10 tasks |

Both queue waits were real and substantial: job 15197304/15197307 sat
`PENDING` for roughly 6-7 hours before starting (partition had 79 jobs queued
across the whole course at peak); this is normal shared-HPC contention, not
an error — confirmed via `sacct`/`scontrol` (no error logs exist for the
waiting period because nothing had executed yet).

### FetchReach v2 grid — result

All 40 seed-runs now exist (`runs/reach/<condition>/seed{0-9}/final_model.zip`
for dense, sparse, sparse_her, dense_her). No tracebacks/NaNs/OOM in any of
the 25 new logs. Full 4-condition × 10-seed plots/summary not yet regenerated
(next step, see below).

### FetchPush pilot — result (the key calibration numbers)

200,000 steps, 1 seed each, `eval-freq=5000`, `n-eval-episodes=20`:

| condition | eval success @ 200k steps | training-rollout success (rolling) | throughput |
|---|---|---|---|
| sparse | 0% | 9% | 61 steps/s |
| sparse + HER | 5% | 13% | 54 steps/s |

Confirms the proposal's hypothesis direction: FetchPush is much harder than
FetchReach — at a step count where every FetchReach condition was already
>90% successful, FetchPush success is near zero even with HER. 200k steps is
nowhere near enough; the proposal's ~10⁶-step estimate is right.

**Used directly to calibrate the main grid/ablation jobs:**
- `--total-timesteps 1000000` (was a placeholder in the draft scripts).
- `#SBATCH --time=07:00:00` (draft had a placeholder `06:00:00`): at 1M
  steps, sparse projects to ~4.6h and sparse+HER to ~5.1h of training alone
  from the observed fps; 7h leaves margin for eval overhead and the
  node-to-node speed variance seen in the reach grid (17–38 min for
  nominally identical 100k-step runs).
- `--eval-freq 10000` (roughly the same eval-to-training ratio as the pilot,
  scaled to the 5× longer budget).

All 70 planned runs (40 Reach + 20 Push main + 10 Push ablation) completed
successfully as of 2026-08-09, zero failed tasks.

### Final results (`plot_results.py` on the complete dataset, 2026-08-09)

FetchReach-v3 (10 seeds/condition) — all conditions solve the task; HER
mainly affects speed:

| condition | final success | seeds ≥90% | median steps to 90% | AUC |
|---|---|---|---|---|
| dense | 100.0% | 10/10 | 6,000 | 0.947 |
| dense+HER | 100.0% | 10/10 | 6,000 | 0.945 |
| sparse | 99.4% | 10/10 | 22,000 | 0.826 |
| sparse+HER | 100.0% | 10/10 | 8,000 | 0.929 |

FetchPush-v3 (10 seeds/condition) — sparse essentially fails to learn; HER
recovers the task:

| condition | final success | seeds ≥90% | median steps to 90% | AUC |
|---|---|---|---|---|
| sparse | 5.4% | 0/10 | never | 0.067 |
| sparse+HER | 90.7% | 9/10 | 350,000 | 0.610 |

Mechanism (`her_diagnostics.npz`, exact aggregates): on Push, SAC's
auto-tuned entropy coefficient α collapses ~21× (6.2e-3 → 2.9e-4) under plain
sparse reward *while the policy is still failing* (premature-convergence
failure mode), vs. staying flat around 2.2–2.3e-3 with HER (~8× higher at the
end). On Reach, the trained-on goal-distance median falls cleanly from
~0.10 m to ~0.016–0.020 m over training (clear implicit curriculum); on Push
the same aggregate metric is dominated by the ~80% HER-relabeled transitions
(median pinned near 0) and only shows the curriculum as a mild widen-then-narrow
pattern in the real-goal (p10–p90) band (~0.18 → ~0.21 → ~0.18 m).

Ablation (n_sampled_goal ∈ {1,4,8}, 5 seeds/setting, Push sparse+HER):
saturates rather than scaling — nsg=8 reaches 90% in a median of 260,000
steps vs. ~380,000–400,000 for nsg=1/4, but final success (84–89%) and AUC
(0.55–0.64) are close across all three. Every setting has exactly one
underperforming seed (0.20–0.45 final success) regardless of nsg.

One fig5 cosmetic bug fixed in `plot_results.py`: the goal-distance
suptitle was getting clipped at the top of the PNG (`fig.tight_layout()`
didn't reserve space for `fig.suptitle()`); fixed by shortening the title and
adding `bbox_inches="tight"` to that figure's `savefig` calls.

---

## Artifacts (private links, not yet shared externally)

- Video comparison (dense vs sparse vs sparse+HER on FetchReach, trained
  policies): `https://claude.ai/code/artifact/c8600a7f-b017-4617-975f-2acce21a1574`
- Status dashboard (snapshot, not live):
  `https://claude.ai/code/artifact/06de2c74-cd9b-4522-8ea4-27cf1ed09d50`
- Results gallery (all 11 figures + tables, with Arabic commentary):
  `https://claude.ai/code/artifact/52700202-a710-4d73-802a-af1d70bf5ef6`

---

## Final report

`report/report.tex` — full write-up (abstract, method, setup, results,
mechanism analysis, ablation, discussion/limitations, reproducibility,
conclusion) in the same `adrl`/NeurIPS-variant style as `proposal/proposal_v2.tex`,
citing the exact numbers above. Figures copied to `report/figures/` as PDFs.

**Not compiled to PDF here** — this environment has no LaTeX toolchain and,
more importantly, no copy of `adrl.cls`/`adrl.sty` (the course-provided style
class the proposal depends on; not on CTAN, presumably only in the group's
Overleaf project). Verified the document is otherwise syntactically correct
and well laid out by compiling with `tectonic` (installed into `SAC-HER/env`
via conda-forge, does not touch the pinned RL package versions) against a
plain-`article` stand-in for the `adrl` package — 7 pages, all tables/figures/
citations render correctly, no overfull boxes after fixing one figure-sizing
issue (Figure 3's two subpanels had mismatched aspect ratios). To get a PDF
with the real course styling, compile `report.tex` on Overleaf (same project
as the proposal) or supply `adrl.cls` here.

### Proposal-version mismatch found and corrected (2026-08-09)

`proposal/proposal_v2.tex` on disk (the file this whole v2 session was built
from, pasted at the start of the conversation) says **10 seeds/condition**
and ablation $n_{\text{sampled\_goal}} \in \{1,4,8\}$. The user later pasted
a different text of the same proposal (same title/authors/structure,
otherwise word-for-word identical) specifying **5 seeds/condition**, ablation
$\{1,4\}$ only, and a softer hypothesis (ii) ("substantially improve" rather
than "the difference between learning and not learning at all"). The user
confirmed **the 5-seed/{1,4} version is the actually-submitted/official
proposal** — `proposal_v2.tex` on disk does not match what was submitted.

No experiments needed to be redone: what we ran (10 seeds, $\{1,4,8\}$) is a
strict superset of what the real proposal required, not a different or
smaller design. Fixed `report/report.tex` to describe this honestly: quotes
hypothesis (ii) with the real wording, and the Experimental Setup section now
states the proposed scope (5 seeds, $\{1,4\}$) explicitly before explaining
we ran the larger 10-seed/$\{1,4,8\}$ design for tighter confidence bands on
FetchPush's noisier seeds. `proposal_v2.tex` on disk itself was **not**
edited — it still has the 10-seed/$\{1,4,8\}$/harder-hypothesis text; if that
file is meant to reflect the real submitted proposal it should be corrected
too, but that's a decision for the user (it may be an intentional later
planning draft, not the submitted one).

## What's left

Nothing experimental. Only: (1) get a real PDF via Overleaf/adrl.cls, (2) any
content revisions the authors want after reading it.
