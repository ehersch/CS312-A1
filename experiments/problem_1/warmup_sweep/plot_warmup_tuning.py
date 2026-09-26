from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import numpy as np
import wandb

from experiments.problem_1.warmup_sweep.launch_warmup_tuning import EXPERIMENT_KEY, WARMUP_PERCENTS
from utils import WANDB_ENTITY, WANDB_PROJECT


PLOT_DIR = Path(__file__).resolve().parent / "plots"
OUT_PATH = PLOT_DIR / f"{EXPERIMENT_KEY}_loss_summary.png"
PROJECT_PATH = f"{WANDB_ENTITY}/{WANDB_PROJECT}"

PURPLE = "#7030A0"
LIGHT_BLUE = "#8CD9FF"
RED = "#e74c3c"
DARK = "#111111"
GRAY = "#777777"
SWEEP_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "warmup_percent_purple_light_blue",
    [PURPLE, LIGHT_BLUE],
)
SWEEP_NORM = mcolors.Normalize(vmin=min(WARMUP_PERCENTS), vmax=max(WARMUP_PERCENTS))
LEFT_X_TICKS = [100, 300, 1_000, 3_000, 10_000]
LEFT_X_TICK_LABELS = ["100", "300", "1k", "3k", "10k"]
LEFT_Y_TICKS = [3, 4, 5, 6]
LEFT_Y_TICK_LABELS = ["3", "4", "5", "6"]


@dataclass(frozen=True)
class WarmupTuningRun:
    warmup_percent: float
    val_loss: float
    train_loss: float | None
    val_steps: np.ndarray
    val_losses: np.ndarray
    name: str
    state: str
    created_at: str
    url: str


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["P052", "Palatino", "Palatino Linotype", "serif"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "P052",
            "mathtext.it": "P052:italic",
            "mathtext.bf": "P052:bold",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.18,
        }
    )


def finite_float(value) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def run_warmup_percent(run) -> float | None:
    return finite_float(run.config.get("warmup_percent"))


def expected_warmup_percent(warmup_percent: float | None) -> float | None:
    if warmup_percent is None:
        return None
    for expected in WARMUP_PERCENTS:
        if math.isclose(warmup_percent, expected, rel_tol=1e-12, abs_tol=0.0):
            return expected
    return None


def run_created_at(run) -> str:
    return str(getattr(run, "created_at", None) or getattr(run, "createdAt", "") or "")


def created_at_sort_key(created_at: str) -> float:
    if not created_at:
        return 0.0
    try:
        return datetime.fromisoformat(created_at.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def is_exact_experiment_run(run) -> bool:
    return (run.name or "").endswith(f"-{EXPERIMENT_KEY}")


def final_metric(run, metric_name: str) -> float | None:
    value = finite_float(run.summary.get(metric_name))
    if value is not None:
        return value

    latest_step = -1
    latest_value = None
    for row in run.scan_history(keys=["optimizer_step", metric_name]):
        value = finite_float(row.get(metric_name))
        if value is None:
            continue
        step = int(row.get("optimizer_step", latest_step + 1))
        if step >= latest_step:
            latest_step = step
            latest_value = value
    return latest_value


def metric_trajectory(run, metric_name: str) -> tuple[np.ndarray, np.ndarray]:
    values_by_step = {}
    for row in run.scan_history(keys=["optimizer_step", metric_name]):
        value = finite_float(row.get(metric_name))
        step = finite_float(row.get("optimizer_step"))
        if value is None or step is None:
            continue
        values_by_step[int(step)] = value

    if not values_by_step:
        for index, row in enumerate(run.scan_history(keys=[metric_name])):
            value = finite_float(row.get(metric_name))
            if value is None:
                continue
            step = finite_float(row.get("optimizer_step"))
            if step is None:
                step = finite_float(row.get("_step"))
            if step is None:
                step = index
            values_by_step[int(step)] = value

    items = sorted(values_by_step.items())
    if not items:
        return np.array([]), np.array([])
    steps, losses = zip(*items, strict=True)
    return np.array(steps), np.array(losses)


def matching_runs() -> list[WarmupTuningRun]:
    api = wandb.Api()
    runs = api.runs(
        PROJECT_PATH,
        filters={"tags": {"$in": [EXPERIMENT_KEY]}},
    )

    latest_by_value = {}
    for run in runs:
        if EXPERIMENT_KEY not in set(run.tags or []) or not is_exact_experiment_run(run):
            continue
        warmup_percent = expected_warmup_percent(run_warmup_percent(run))
        if warmup_percent is None:
            continue
        created_at = run_created_at(run)
        sort_key = created_at_sort_key(created_at)
        current = latest_by_value.get(warmup_percent)
        if current is None or sort_key > current[0]:
            latest_by_value[warmup_percent] = (sort_key, created_at, run)

    collected = []
    for warmup_percent in WARMUP_PERCENTS:
        candidate = latest_by_value.get(warmup_percent)
        if candidate is None:
            continue
        _, created_at, run = candidate
        val_steps, val_losses = metric_trajectory(run, "val_loss")
        if val_losses.size:
            val_loss = float(val_losses[-1])
        else:
            val_loss = final_metric(run, "val_loss")
        if val_loss is None:
            continue
        collected.append(
            WarmupTuningRun(
                warmup_percent=warmup_percent,
                val_loss=val_loss,
                train_loss=final_metric(run, "train_loss"),
                val_steps=val_steps,
                val_losses=val_losses,
                name=run.name,
                state=run.state,
                created_at=created_at,
                url=run.url,
            )
        )
    return sorted(collected, key=lambda item: item.warmup_percent)


def plot_runs(runs: list[WarmupTuningRun]) -> Path:
    if not runs:
        raise RuntimeError(
            f"No W&B runs found for tag {EXPERIMENT_KEY!r} in {PROJECT_PATH}."
        )

    set_style()
    warmup_percents = np.array([run.warmup_percent for run in runs])
    val_losses = np.array([run.val_loss for run in runs])
    best_index = int(np.argmin(val_losses))

    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        figsize=(10, 4.2),
        gridspec_kw={"width_ratios": [1.35, 1.0]},
    )

    for run in runs:
        if not run.val_losses.size:
            continue
        color = SWEEP_CMAP(SWEEP_NORM(run.warmup_percent))
        ax_left.plot(
            run.val_steps + 1,
            run.val_losses,
            color=color,
            linewidth=1.8,
            alpha=0.9,
            label=f"{run.warmup_percent:g}",
            zorder=4,
        )

    ax_left.set_xlabel("Optimizer step")
    ax_left.set_ylabel("Validation loss")
    ax_left.set_title("Loss Trajectories")
    ax_left.set_xscale("log")
    ax_left.set_yscale("log")
    ax_left.set_xticks(LEFT_X_TICKS)
    ax_left.set_xticklabels(LEFT_X_TICK_LABELS)
    ax_left.set_yticks(LEFT_Y_TICKS)
    ax_left.set_yticklabels(LEFT_Y_TICK_LABELS)
    ax_left.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax_left.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax_left.grid(True, which="major", linestyle=":", alpha=0.3)
    ax_left.grid(True, which="minor", axis="y", linestyle=":", alpha=0.15)
    ax_left.legend(title="Warmup fraction", frameon=False, fontsize=8.5, title_fontsize=9, ncol=2)

    ax_right.plot(
        warmup_percents,
        val_losses,
        color=GRAY,
        linewidth=1.0,
        alpha=0.45,
        zorder=3,
    )
    ax_right.scatter(
        warmup_percents,
        val_losses,
        s=80,
        c=warmup_percents,
        cmap=SWEEP_CMAP,
        norm=SWEEP_NORM,
        edgecolor=DARK,
        linewidth=1.2,
        zorder=5,
    )
    ax_right.scatter(
        [warmup_percents[best_index]],
        [val_losses[best_index]],
        s=150,
        facecolors="none",
        edgecolor=RED,
        linewidth=2.0,
        zorder=6,
        label=f"best: {warmup_percents[best_index]:g}",
    )

    ax_right.set_xscale("linear")
    ax_right.set_xticks(list(WARMUP_PERCENTS))
    ax_right.set_xticklabels(
        [f"{lr:g}" for lr in WARMUP_PERCENTS],
        rotation=20,
        ha="right",
    )
    ax_right.set_xlabel("Warmup fraction")
    ax_right.set_ylabel("Final validation loss")
    ax_right.set_title("Final Loss")
    ax_right.margins(y=0.12)
    ax_right.grid(True, which="major", linestyle=":", alpha=0.3)
    ax_right.grid(True, which="minor", axis="y", linestyle=":", alpha=0.15)
    ax_right.legend(frameon=False, loc="best")

    for lr, loss in zip(warmup_percents, val_losses, strict=True):
        ax_right.annotate(
            f"{loss:.3f}",
            xy=(lr, loss),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=GRAY,
            clip_on=False,
        )

    sm = mpl.cm.ScalarMappable(norm=SWEEP_NORM, cmap=SWEEP_CMAP)
    sm.set_array([])
    colorbar = fig.colorbar(sm, ax=[ax_left, ax_right], fraction=0.03, pad=0.02)
    colorbar.set_label("Warmup fraction")
    colorbar.set_ticks(list(WARMUP_PERCENTS))
    colorbar.set_ticklabels([f"{lr:g}" for lr in WARMUP_PERCENTS])

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH)
    plt.close(fig)
    return OUT_PATH


def main() -> None:
    runs = matching_runs()
    for run in runs:
        print(
            f"warmup={run.warmup_percent:g} val_loss={run.val_loss:.6f} "
            f"eval_points={len(run.val_losses)} state={run.state} "
            f"created_at={run.created_at} name={run.name}"
        )
    print(plot_runs(runs).resolve())


if __name__ == "__main__":
    main()
