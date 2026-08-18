"""Aggregate SAC/HER runs into learning curves, mechanism plots, and summary
tables, per proposal_v2.

Reads runs/<task>/<condition>/seed*/{evaluations.npz,monitor.csv,
her_diagnostics.npz} and runs/push_ablation/nsg<k>/seed*/... and writes to
results/figures/:
  fig1_<task>_success_rate.{png,pdf}   eval success rate vs timesteps
  fig2_<task>_returns.{png,pdf}        eval mean return vs timesteps
  fig3_<task>_train_success.{png,pdf}  training-time rolling success rate
  fig4_<task>_entropy_coef.{png,pdf}   SAC's auto-tuned entropy coefficient
  fig5_<task>_goal_distance.{png,pdf}  sampled-batch goal-distance percentiles
                                        (HER's implicit curriculum)
  fig6_push_ablation.{png,pdf}         success rate by HER relabeling ratio
  summary.csv               per-seed: task, condition, final success,
                             steps-to-90%, success-curve AUC
  ablation_summary.csv      same, for the n_sampled_goal ablation
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Quiet, considered chart chrome: no top/right spines, recessive muted-gray
# axis text, hairline gridlines (drawn per-axes below with GRID_COLOR).
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": "#52514e",
    "xtick.color": "#898781",
    "ytick.color": "#898781",
    "axes.axisbelow": True,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

TASK_CONDITIONS = {
    "reach": ["dense", "sparse", "dense_her", "sparse_her"],
    "push": ["sparse", "sparse_her"],
}
TASK_ENV_NAME = {"reach": "FetchReach-v3", "push": "FetchPush-v3"}
# Validated categorical palette (fixed slot order, CVD-safe on adjacent
# pairs): slot1 blue, slot2 orange, slot3 aqua, slot4 yellow. Assigned once
# per condition identity and reused across every figure, so e.g. sparse+HER
# is the same yellow whether it appears alongside all four conditions
# (Reach) or just sparse (Push).
COLORS = {
    "dense": "#2a78d6",
    "sparse": "#eb6834",
    "dense_her": "#1baf7a",
    "sparse_her": "#eda100",
}
LABELS = {
    "dense": "dense",
    "sparse": "sparse",
    "dense_her": "dense + HER",
    "sparse_her": "sparse + HER",
}
GRID_COLOR = "#e1e0d9"
SEED_LINE_ALPHA = 0.12
BAND_ALPHA = 0.12
ABLATION_NSG = (1, 4, 8)
ABLATION_COLORS = {1: "#2a78d6", 4: "#008300", 8: "#4a3aa7"}
SUCCESS_TARGET = 0.9


def _seed_dirs(root: Path):
    return sorted(root.glob("seed*"), key=lambda p: int(p.name.removeprefix("seed")))


def load_condition(cond_dir: Path):
    """Return per-seed eval curves from all seed*/evaluations.npz under cond_dir."""
    timesteps, success, returns, seeds = None, [], [], []
    for sd in _seed_dirs(cond_dir):
        npz_path = sd / "evaluations.npz"
        if not npz_path.exists():
            print(f"[warn] missing {npz_path}, skipping")
            continue
        data = np.load(npz_path)
        ts = data["timesteps"]
        if timesteps is None:
            timesteps = ts
        elif len(ts) != len(timesteps):
            n = min(len(ts), len(timesteps))
            print(f"[warn] {sd} has {len(ts)} evals vs {len(timesteps)}; truncating to {n}")
            timesteps = timesteps[:n]
            success = [s[:n] for s in success]
            returns = [r[:n] for r in returns]
            ts = ts[:n]
        success.append(data["successes"][: len(timesteps)].mean(axis=1))
        returns.append(data["results"][: len(timesteps)].mean(axis=1))
        seeds.append(int(sd.name.removeprefix("seed")))
    if timesteps is None:
        raise FileNotFoundError(f"no evaluations.npz found under {cond_dir}")
    return dict(timesteps=timesteps, success=np.array(success),
                returns=np.array(returns), seeds=seeds)


def load_diagnostics(cond_dir: Path):
    """Return per-seed her_diagnostics.npz arrays, aligned to the shortest
    seed's timestep grid (all seeds share the same --diag-freq by construction)."""
    timesteps, ent_coef, gd_mean, gd_p10, gd_p50, gd_p90, seeds = (
        None, [], [], [], [], [], []
    )
    for sd in _seed_dirs(cond_dir):
        npz_path = sd / "her_diagnostics.npz"
        if not npz_path.exists():
            continue
        d = np.load(npz_path)
        if len(d["timesteps"]) == 0:
            continue
        ts = d["timesteps"]
        if timesteps is None or len(ts) < len(timesteps):
            timesteps = ts
        ent_coef.append(d["ent_coef"])
        gd_mean.append(d["goal_dist_mean"])
        gd_p10.append(d["goal_dist_p10"])
        gd_p50.append(d["goal_dist_p50"])
        gd_p90.append(d["goal_dist_p90"])
        seeds.append(int(sd.name.removeprefix("seed")))
    if timesteps is None:
        return None
    n = len(timesteps)
    return dict(
        timesteps=timesteps,
        ent_coef=np.array([a[:n] for a in ent_coef]),
        goal_dist_mean=np.array([a[:n] for a in gd_mean]),
        goal_dist_p10=np.array([a[:n] for a in gd_p10]),
        goal_dist_p50=np.array([a[:n] for a in gd_p50]),
        goal_dist_p90=np.array([a[:n] for a in gd_p90]),
        seeds=seeds,
    )


def load_monitor(run_dir: Path):
    csv_path = next(iter(sorted(run_dir.glob("*monitor.csv"))), None)
    if csv_path is None:
        return None
    df = pd.read_csv(csv_path, skiprows=1)
    steps = df["l"].cumsum()
    rolling = df["is_success"].rolling(100, min_periods=10).mean()
    return steps, rolling


def fig_success_rate(data, task, out_dir: Path):
    conditions = TASK_CONDITIONS[task]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cond in conditions:
        if cond not in data:
            continue
        d = data[cond]
        color = COLORS[cond]
        for row in d["success"]:
            ax.plot(d["timesteps"], row, color=color, alpha=SEED_LINE_ALPHA, lw=0.6)
        mean, std = d["success"].mean(axis=0), d["success"].std(axis=0)
        ax.plot(d["timesteps"], mean, color=color, lw=2,
                label=f"{LABELS[cond]} (n={len(d['seeds'])})")
        ax.fill_between(d["timesteps"], mean - std, mean + std, color=color, alpha=BAND_ALPHA)
    ax.axhline(SUCCESS_TARGET, color=GRID_COLOR, ls=(0, (4, 3)), lw=1.2, zorder=0.5)
    ax.text(0.992, SUCCESS_TARGET, "90% ", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=7.5, color="#898781")
    ax.set_xlabel("environment steps")
    ax.set_ylabel("evaluation success rate")
    ax.set_title(f"SAC on {TASK_ENV_NAME[task]}: " + " vs ".join(LABELS[c] for c in conditions if c in data))
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=1.0, color=GRID_COLOR, lw=0.7)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig1_{task}_success_rate.{ext}", dpi=200)
    plt.close(fig)


def fig_returns(data, task, out_dir: Path):
    conditions = [c for c in TASK_CONDITIONS[task] if c in data]
    fig, axes = plt.subplots(1, len(conditions), figsize=(5.5 * len(conditions), 4.5))
    for ax, cond in zip(np.atleast_1d(axes), conditions):
        d = data[cond]
        color = COLORS[cond]
        for row in d["returns"]:
            ax.plot(d["timesteps"], row, color=color, alpha=SEED_LINE_ALPHA, lw=0.6)
        mean, std = d["returns"].mean(axis=0), d["returns"].std(axis=0)
        ax.plot(d["timesteps"], mean, color=color, lw=2)
        ax.fill_between(d["timesteps"], mean - std, mean + std, color=color, alpha=BAND_ALPHA)
        if cond.startswith("sparse"):
            ax.axhline(-50, color="gray", ls="--", lw=1)
            ax.text(0.02, 0.06, "always-fail = -50", transform=ax.transAxes,
                    color="gray", fontsize=9)
        ax.set_ylabel("episode return")
        ax.set_title(f"{LABELS[cond]}")
        ax.set_xlabel("environment steps")
        ax.grid(alpha=1.0, color=GRID_COLOR, lw=0.7)
    fig.suptitle(f"{TASK_ENV_NAME[task]}: evaluation returns (scales not comparable across panels)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig2_{task}_returns.{ext}", dpi=200)
    plt.close(fig)


def fig_train_success(runs_dir: Path, task: str, data, out_dir: Path):
    conditions = [c for c in TASK_CONDITIONS[task] if c in data]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cond in conditions:
        color = COLORS[cond]
        shown = False
        for seed in data[cond]["seeds"]:
            mon = load_monitor(runs_dir / task / cond / f"seed{seed}")
            if mon is None:
                continue
            steps, rolling = mon
            ax.plot(steps, rolling, color=color, alpha=0.35, lw=0.8,
                    label=LABELS[cond] if not shown else None)
            shown = True
    ax.set_xlabel("environment steps")
    ax.set_ylabel("training success rate (rolling 100 episodes)")
    ax.set_title(f"{TASK_ENV_NAME[task]}: training-time success (exploration policy)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=1.0, color=GRID_COLOR, lw=0.7)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig3_{task}_train_success.{ext}", dpi=200)
    plt.close(fig)


def fig_entropy_coef(diag_data, task, out_dir: Path):
    conditions = [c for c in TASK_CONDITIONS[task] if c in diag_data]
    if not conditions:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cond in conditions:
        d = diag_data[cond]
        color = COLORS[cond]
        mean = d["ent_coef"].mean(axis=0)
        std = d["ent_coef"].std(axis=0)
        ax.plot(d["timesteps"], mean, color=color, lw=2,
                label=f"{LABELS[cond]} (n={len(d['seeds'])})")
        ax.fill_between(d["timesteps"], mean - std, mean + std, color=color, alpha=BAND_ALPHA)
    ax.set_xlabel("environment steps")
    ax.set_ylabel(r"entropy coefficient $\alpha$ (auto-tuned)")
    ax.set_title(f"{TASK_ENV_NAME[task]}: SAC entropy coefficient over training")
    ax.set_yscale("log")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=1.0, color=GRID_COLOR, lw=0.7, which="both")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig4_{task}_entropy_coef.{ext}", dpi=200)
    plt.close(fig)


def fig_goal_distance(diag_data, task, out_dir: Path):
    """Plot p10/p50/p90 as three distinct lines rather than a median+band: a
    shaded p10-p90 band visually hides that the median stays pinned near
    zero (HER-relabeled transitions) while the upper percentiles move
    independently as training progresses -- the actual curriculum signal.
    """
    her_conditions = [c for c in TASK_CONDITIONS[task] if c.endswith("_her") and c in diag_data]
    if not her_conditions:
        return
    fig, axes = plt.subplots(1, len(her_conditions),
                              figsize=(5.5 * len(her_conditions), 4.5), squeeze=False)
    for ax, cond in zip(axes[0], her_conditions):
        d = diag_data[cond]
        color = COLORS[cond]
        p10 = d["goal_dist_p10"].mean(axis=0)
        p50 = d["goal_dist_p50"].mean(axis=0)
        p90 = d["goal_dist_p90"].mean(axis=0)
        ax.plot(d["timesteps"], p90, color=color, lw=1.5, ls=":", alpha=0.8, label="p90")
        ax.plot(d["timesteps"], p50, color=color, lw=2, label="median (p50)")
        ax.plot(d["timesteps"], p10, color=color, lw=1.5, ls="--", alpha=0.8, label="p10")
        ax.set_title(f"{LABELS[cond]} (n={len(d['seeds'])})")
        ax.set_xlabel("environment steps")
        ax.set_ylabel("|achieved goal - desired goal| in sampled minibatch (m)")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=1.0, color=GRID_COLOR, lw=0.7)
    fig.suptitle(f"{TASK_ENV_NAME[task]}: goal-distance of trained-on transitions", fontsize=12)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig5_{task}_goal_distance.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_ablation(runs_dir: Path, out_dir: Path):
    """Success-rate curves for the FetchPush-v3 n_sampled_goal ablation.
    nsg=4 reuses runs/push/sparse_her/seed{0-4} (the main-grid condition)
    rather than rerunning it."""
    curves = {}
    for nsg in ABLATION_NSG:
        if nsg == 4:
            cond_dir = runs_dir / "push" / "sparse_her"
        else:
            cond_dir = runs_dir / "push_ablation" / f"nsg{nsg}"
        if not cond_dir.exists():
            continue
        # ablation uses 5 seeds (0-4); cap the reused nsg=4 condition to match
        all_seeds = _seed_dirs(cond_dir)
        seed_dirs = [s for s in all_seeds if int(s.name.removeprefix("seed")) < 5]
        if not seed_dirs:
            continue
        timesteps, success = None, []
        for sd in seed_dirs:
            npz_path = sd / "evaluations.npz"
            if not npz_path.exists():
                continue
            data = np.load(npz_path)
            ts = data["timesteps"]
            if timesteps is None or len(ts) < len(timesteps):
                timesteps = ts
            success.append(data["successes"].mean(axis=1))
        if timesteps is None:
            continue
        n = len(timesteps)
        curves[nsg] = dict(timesteps=timesteps,
                            success=np.array([s[:n] for s in success]))
    if not curves:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for nsg, d in curves.items():
        color = ABLATION_COLORS[nsg]
        mean, std = d["success"].mean(axis=0), d["success"].std(axis=0)
        ax.plot(d["timesteps"], mean, color=color, lw=2,
                label=f"n_sampled_goal={nsg} (n={d['success'].shape[0]})")
        ax.fill_between(d["timesteps"], mean - std, mean + std, color=color, alpha=BAND_ALPHA)
    ax.axhline(SUCCESS_TARGET, color=GRID_COLOR, ls=(0, (4, 3)), lw=1.2, zorder=0.5)
    ax.text(0.992, SUCCESS_TARGET, "90% ", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=7.5, color="#898781")
    ax.set_xlabel("environment steps")
    ax.set_ylabel("evaluation success rate")
    ax.set_title("FetchPush-v3 sparse+HER: relabeling-ratio ablation")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=1.0, color=GRID_COLOR, lw=0.7)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig6_push_ablation.{ext}", dpi=200)
    plt.close(fig)

    rows = []
    for nsg, d in curves.items():
        ts = d["timesteps"]
        for i, curve in enumerate(d["success"]):
            reached = np.nonzero(curve >= SUCCESS_TARGET)[0]
            rows.append(dict(
                n_sampled_goal=nsg,
                final_success=curve[-5:].mean(),
                steps_to_90pct=int(ts[reached[0]]) if len(reached) else None,
                success_auc=np.trapz(curve, ts) / ts[-1],
            ))
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "ablation_summary.csv", index=False)
    print("\n=== ablation (FetchPush-v3 sparse+HER, n_sampled_goal) ===")
    print(df.groupby("n_sampled_goal").agg(
        final_success_mean=("final_success", "mean"),
        final_success_median=("final_success", "median"),
        auc_mean=("success_auc", "mean"),
        steps_to_90pct_median=("steps_to_90pct", "median"),
    ).to_string())


def summarize(all_data, out_dir: Path):
    rows = []
    for task, data in all_data.items():
        for cond, d in data.items():
            ts = d["timesteps"]
            for seed, curve in zip(d["seeds"], d["success"]):
                reached = np.nonzero(curve >= SUCCESS_TARGET)[0]
                rows.append(dict(
                    task=task,
                    condition=cond,
                    seed=seed,
                    final_success=curve[-5:].mean(),
                    steps_to_90pct=int(ts[reached[0]]) if len(reached) else None,
                    success_auc=np.trapz(curve, ts) / ts[-1],
                ))
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "summary.csv", index=False)

    agg = df.groupby(["task", "condition"]).agg(
        final_success_mean=("final_success", "mean"),
        final_success_median=("final_success", "median"),
        final_success_std=("final_success", "std"),
        auc_mean=("success_auc", "mean"),
        steps_to_90pct_median=("steps_to_90pct", "median"),
        seeds_reaching_90pct=("steps_to_90pct", "count"),
        n_seeds=("seed", "count"),
    )
    print("\n=== per-seed summary ===")
    print(df.to_string(index=False))
    print("\n=== per-condition summary ===")
    print(agg.to_string())
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-dir", type=Path, default=Path(__file__).parent / "runs")
    p.add_argument("--out-dir", type=Path,
                   default=Path(__file__).parent / "results" / "figures")
    p.add_argument("--tasks", nargs="+", default=["reach", "push"], choices=("reach", "push"))
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_data = {}
    for task in args.tasks:
        data, diag_data = {}, {}
        for cond in TASK_CONDITIONS[task]:
            cond_dir = args.runs_dir / task / cond
            if not cond_dir.exists():
                print(f"[warn] {cond_dir} does not exist yet, skipping")
                continue
            try:
                data[cond] = load_condition(cond_dir)
            except FileNotFoundError as e:
                print(f"[warn] {e}")
                continue
            diag = load_diagnostics(cond_dir)
            if diag is not None:
                diag_data[cond] = diag
        if not data:
            continue
        all_data[task] = data
        fig_success_rate(data, task, args.out_dir)
        fig_returns(data, task, args.out_dir)
        fig_train_success(args.runs_dir, task, data, args.out_dir)
        fig_entropy_coef(diag_data, task, args.out_dir)
        fig_goal_distance(diag_data, task, args.out_dir)

    if "push" in args.tasks:
        fig_ablation(args.runs_dir, args.out_dir)

    summarize(all_data, args.out_dir)
    print(f"\nfigures + summary tables written to {args.out_dir}")


if __name__ == "__main__":
    main()
