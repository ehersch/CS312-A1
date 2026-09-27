"""Plot measured scaling results; default to d4-d7 before preregistration."""
from pathlib import Path
import csv

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import wandb

from experiments.a1_basics.p2_scaling_law_reliability import PART_A_RUNS as RUNS
from experiments.lr_tuning.plot_lr_tuning import (
    set_style, finite_float, final_metric, run_created_at, created_at_sort_key,
)
from train import training_run_name
from utils import WANDB_ENTITY, WANDB_PROJECT

PLOT_DIR = Path(__file__).resolve().parent / "plots" / "p2_scaling"
# Output prefix only: the original launcher does not add experiment tags.
EXPERIMENT_KEY = "p2_scaling"
VARIANTS = (
    ("baseline", "Baseline", 0.003, "linear", 0.0),
    ("constant", "Constant LR", 0.003, "constant", 0.0),
    ("dropout", "Linear LR, dropout 0.2", 0.003, "linear", 0.2),
    ("high_lr", "Linear LR, learning rate 0.03", 0.03, "linear", 0.0),
)


def collect_results(max_depth=7, api=None):
    if api is None:
        api = wandb.Api()
    configs = sorted(
        (c for c in RUNS if 4 <= c.model_config.num_hidden_layers <= max_depth),
        key=lambda c: c.model_config.num_hidden_layers,
    )
    expected = {training_run_name(c): c for c in configs}
    latest = {}
    for run in api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}", filters={"displayName": {"$in": list(expected)}}):
        if run.name not in expected:
            continue
        config = expected[run.name]
        # Check the studied fields as well as the original launcher's exact name.
        if any(run.config.get(field) != getattr(config, field)
               for field in ("learning_rate", "lr_schedule", "dropout")):
            continue
        timestamp = created_at_sort_key(run_created_at(run))
        if run.name not in latest or timestamp > latest[run.name][0]:
            latest[run.name] = (timestamp, run)

    rows = []
    for name, config in expected.items():
        variant = next(v[0] for v in VARIANTS if v[2:] == (config.learning_rate, config.lr_schedule, config.dropout))
        candidate = latest.get(name)
        run = candidate[1] if candidate else None
        params = finite_float(run.config.get("parameter_count")) if run else None
        loss = None
        state = run.state if run else "missing"
        if run and state == "finished":
            loss = (finite_float(run.summary["val_loss"]) if "val_loss" in run.summary
                    else final_metric(run, "val_loss"))
        if state == "finished" and (loss is None or loss <= 0 or params is None or params <= 0):
            state = "invalid final metrics"
            loss = None
        rows.append(dict(
            variant=variant, depth=config.model_config.num_hidden_layers,
            parameter_count=params, val_loss=loss, state=state,
            run_name=name, url=run.url if run else "",
        ))
    return rows


def draw_panel(ax, rows, title):
    valid = [r for r in rows if r['state'] == 'finished' and r['val_loss'] is not None]
    # NaNs break lines at missing depths rather than implying a measured point.
    ax.plot([r['parameter_count'] or np.nan for r in rows],
            [r['val_loss'] if r in valid else np.nan for r in rows],
            color="#7030A0", marker="o", linewidth=1.8)
    for row in valid:
        ax.annotate(f"d{row['depth']}: {row['val_loss']:.3f}",
                    (row['parameter_count'], row['val_loss']), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    if valid:
        ticks = sorted({r['parameter_count'] for r in valid})
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{x / 1e6:.1f}M" for x in ticks])
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.yaxis.set_minor_formatter(mticker.FormatStrFormatter("%.2f"))
    else:
        ax.set_xlim(1e6, 1e8)
        ax.set_ylim(2, 10)
        ax.text(.5, .5, "No completed results", transform=ax.transAxes, ha="center")
    unavailable = [f"d{r['depth']}: {r['state']}" for r in rows if r not in valid]
    if unavailable:
        ax.text(.03, .03, "\n".join(unavailable), transform=ax.transAxes,
                fontsize=8, va="bottom", bbox=dict(facecolor="white", alpha=.8, edgecolor="none"))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_title(title)
    ax.set_xlabel("Model parameters (log scale)")
    ax.set_ylabel("Final validation loss (log scale)")
    ax.margins(x=.18, y=.25)
    ax.grid(True, which="both", linestyle=":", alpha=.25)


def plot_results(rows, max_depth=7, plot_dir=PLOT_DIR):
    set_style()
    plot_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{EXPERIMENT_KEY}_d4-d{max_depth}"
    csv_path = plot_dir / f"{prefix}_results.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=['variant', 'depth', 'parameter_count', 'val_loss', 'state', 'run_name', 'url'])
        writer.writeheader()
        writer.writerows(rows)
    outputs = [csv_path]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    for ax, (key, title, *_) in zip(axes.flat, VARIANTS):
        subset = [r for r in rows if r['variant'] == key]
        draw_panel(ax, subset, title)
        single, single_ax = plt.subplots(figsize=(6, 4.5), layout="constrained")
        draw_panel(single_ax, subset, title)
        path = plot_dir / f"{prefix}_{key}.png"
        single.savefig(path)
        plt.close(single)
        outputs.append(path)
    fig.suptitle(f"Scaling measurements: d4-d{max_depth} (fixed training tokens)")
    path = plot_dir / f"{prefix}_summary.png"
    fig.savefig(path)
    plt.close(fig)
    outputs.append(path)
    return outputs


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-depth", type=int, choices=(7, 9), default=7,
                        help="Use 9 only after recording predictions and completing d8/d9.")
    args = parser.parse_args()
    rows = collect_results(args.max_depth)
    for row in rows:
        print(f"{row['variant']} d{row['depth']} loss={row['val_loss']} state={row['state']} {row['url']}")
    for path in plot_results(rows, args.max_depth):
        print(path.resolve())


if __name__ == "__main__":
    main()
