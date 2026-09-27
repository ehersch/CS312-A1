"""Scheduler comparisons using the existing LR plot's style and W&B helpers."""

from dataclasses import dataclass, field
from pathlib import Path
import math

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import wandb

from experiments.problem_1.lr_sweep.plot_lr_tuning import (
    PURPLE,
    LIGHT_BLUE,
    RED,
    created_at_sort_key,
    finite_float,
    final_metric,
    metric_trajectory,
    run_created_at,
    run_learning_rate,
    set_style,
)
from utils import WANDB_ENTITY, WANDB_PROJECT


@dataclass(frozen=True)
class SweepSpec:
    experiment_key: str
    parameter: str
    parameter_label: str
    values: tuple
    learning_rates: tuple
    plot_dir: Path
    fixed_config: dict = field(default_factory=dict)
    output_name: str = "loss_summary"


@dataclass(frozen=True)
class PairedRun:
    learning_rate: float
    value: float | str
    val_loss: float | None
    progress: np.ndarray
    val_losses: np.ndarray
    name: str
    state: str
    created_at: str
    url: str


def expected_value(value, choices):
    if isinstance(value, str) and value in choices:
        return value
    value = finite_float(value)
    if value is not None:
        for choice in choices:
            if math.isclose(value, choice, rel_tol=1e-12, abs_tol=0.0):
                return choice
    return None


def matching_runs(spec: SweepSpec, api=None) -> list[PairedRun]:
    if api is None:
        api = wandb.Api()
    candidates = api.runs(
        f"{WANDB_ENTITY}/{WANDB_PROJECT}",
        filters={"tags": {"$in": [spec.experiment_key]}},
    )
    latest = {}
    for run in candidates:
        if spec.experiment_key not in set(run.tags or []):
            continue
        if not (run.name or "").endswith(f"-{spec.experiment_key}"):
            continue
        if any(run.config.get(key) != value for key, value in spec.fixed_config.items()):
            continue
        lr = expected_value(run_learning_rate(run), spec.learning_rates)
        value = expected_value(run.config.get(spec.parameter), spec.values)
        if lr is None or value is None:
            continue
        key = (value, lr)
        created_at = run_created_at(run)
        sort_key = created_at_sort_key(created_at)
        if key not in latest or sort_key > latest[key][0]:
            latest[key] = (sort_key, created_at, run)

    collected = []
    for value in spec.values:
        for lr in spec.learning_rates:
            candidate = latest.get((value, lr))
            if candidate is None:
                print(f"Missing: {spec.parameter}={value} lr={lr:g}")
                continue
            _, created_at, run = candidate
            steps, losses = metric_trajectory(run, "val_loss")
            total_steps = finite_float(run.config.get("total_steps"))
            if total_steps is not None and total_steps > 0:
                # Training logs zero-based optimizer_step. Normalize for batch-size comparisons.
                progress = (steps + 1) / total_steps
            else:
                print(f"No total_steps for {run.name}; omitting its trajectory.")
                progress, losses = np.array([]), np.array([])

            # Never rank an interrupted run's early loss as a completed result.
            val_loss = None
            if run.state == "finished":
                if "val_loss" in run.summary:
                    val_loss = finite_float(run.summary.get("val_loss"))
                else:
                    val_loss = final_metric(run, "val_loss")
            collected.append(
                PairedRun(
                    learning_rate=lr,
                    value=value,
                    val_loss=val_loss,
                    progress=progress,
                    val_losses=losses,
                    name=run.name,
                    state=run.state,
                    created_at=created_at,
                    url=run.url,
                )
            )
    return collected


def plot_runs(spec: SweepSpec, runs: list[PairedRun]) -> Path:
    set_style()
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(11, 4.2),
        gridspec_kw={"width_ratios": [1.35, 1.0]},
        layout="constrained",
    )
    cmap = mcolors.LinearSegmentedColormap.from_list("paired_purple_blue", [PURPLE, LIGHT_BLUE])
    colors = {value: cmap(i / max(1, len(spec.values) - 1)) for i, value in enumerate(spec.values)}
    styles = ("-", "--", ":", "-.")
    by_pair = {(run.value, run.learning_rate): run for run in runs}

    for value in spec.values:
        for i, lr in enumerate(spec.learning_rates):
            run = by_pair.get((value, lr))
            if run is None or not run.progress.size:
                continue
            status = "" if run.state == "finished" else f" ({run.state})"
            ax_left.plot(
                run.progress, run.val_losses,
                color=colors[value], linestyle=styles[i % len(styles)],
                linewidth=1.8, alpha=0.9,
                label=f"{value}, LR {lr:g}{status}",
            )
    ax_left.set_xlabel("Training fraction (optimizer steps / total steps)")
    ax_left.set_ylabel("Validation loss")
    ax_left.set_title("Loss Trajectories")
    ax_left.set_xlim(0, 1)
    if ax_left.lines:
        ax_left.legend(title=spec.parameter_label, frameon=False, fontsize=8, title_fontsize=9)
    else:
        ax_left.text(0.5, 0.5, "No trajectories available", transform=ax_left.transAxes, ha="center")

    unavailable = []
    for value in spec.values:
        losses = []
        for lr in spec.learning_rates:
            run = by_pair.get((value, lr))
            loss = run.val_loss if run is not None else None
            losses.append(np.nan if loss is None else loss)
            if loss is None:
                status = "missing" if run is None else run.state
                if run is not None and run.state == "finished":
                    status = "no finite final loss"
                unavailable.append(f"{value} / LR {lr:g}: {status}")
        losses = np.array(losses)
        ax_right.plot(
            spec.learning_rates, losses, marker="o", markersize=7,
            color=colors[value], markeredgecolor="#111111", linewidth=1.5,
            label=f"{value}",
        )
        if np.isfinite(losses).any():
            best = int(np.nanargmin(losses))
            ax_right.scatter(
                [spec.learning_rates[best]], [losses[best]],
                s=145, facecolors="none", edgecolor=RED, linewidth=2, zorder=6,
            )
            for lr, loss in zip(spec.learning_rates, losses):
                if np.isfinite(loss):
                    ax_right.annotate(
                        f"{loss:.3f}", (lr, loss), xytext=(0, 8),
                        textcoords="offset points", ha="center", fontsize=8,
                    )
    ax_right.set_xscale("log")
    ax_right.set_xticks(spec.learning_rates)
    ax_right.set_xticklabels([f"{lr:g}" for lr in spec.learning_rates])
    ax_right.set_xlabel("Learning rate")
    ax_right.set_ylabel("Final validation loss")
    ax_right.set_title("Final Loss (finished runs)")
    ax_right.margins(y=0.2)
    ax_right.legend(title=spec.parameter_label, frameon=False)
    for ax in (ax_left, ax_right):
        ax.grid(True, which="major", linestyle=":", alpha=0.3)
    caption = "Red circles: best sampled LR for each setting."
    if unavailable:
        caption += "\nUnavailable final results: " + "; ".join(unavailable)
    import textwrap

    fig.supxlabel("\n".join(textwrap.wrap(caption, width=115)), fontsize=8)
    spec.plot_dir.mkdir(parents=True, exist_ok=True)
    out_path = spec.plot_dir / f"{spec.experiment_key}_{spec.output_name}.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_experiment(spec: SweepSpec) -> Path:
    runs = matching_runs(spec)
    for run in runs:
        loss = "unavailable" if run.val_loss is None else f"{run.val_loss:.6f}"
        print(
            f"{spec.parameter}={run.value} lr={run.learning_rate:g} "
            f"val_loss={loss} state={run.state} created_at={run.created_at} "
            f"name={run.name} url={run.url}"
        )
    path = plot_runs(spec, runs)
    print(path.resolve())
    return path
