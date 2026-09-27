"""Compare baseline/dropout data scaling, reusing untagged Part A 600k runs."""
from pathlib import Path
import csv

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import wandb

from experiments.problem_2.data_scaling.launch_data_scaling import (
    EXPERIMENT_KEY, TRAIN_SEQUENCE_COUNTS, DROPOUTS, build_runs, make_config,
)
from experiments.lr_tuning.plot_lr_tuning import (
    set_style, finite_float, final_metric, run_created_at, created_at_sort_key,
)
from train import training_run_name
from utils import WANDB_ENTITY, WANDB_PROJECT

PLOT_DIR = Path(__file__).resolve().parent / "plots"


def comparison_configs():
    return build_runs() + [make_config(600_000, d, existing_endpoint=True) for d in DROPOUTS]


def collect_results(api=None):
    if api is None:
        api = wandb.Api()
    expected = {training_run_name(c): c for c in comparison_configs()}
    latest = {}
    for run in api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}", filters={"displayName": {"$in": list(expected)}}):
        if run.name not in expected:
            continue
        c = expected[run.name]
        keys = ("num_train_sequences", "num_epochs", "dropout", "batch_size", "learning_rate", "lr_schedule", "weight_decay", "warmup_percent", "model_seed", "data_seed")
        if any(run.config.get(k) != getattr(c, k) for k in keys):
            continue
        if run.config.get("model_config") != c.model_config.to_dict():
            continue
        if c.run_name_suffix and EXPERIMENT_KEY not in set(run.tags or []):
            continue
        timestamp = created_at_sort_key(run_created_at(run))
        if run.name not in latest or timestamp > latest[run.name][0]:
            latest[run.name] = (timestamp, run)
    rows = []
    for name, c in expected.items():
        candidate = latest.get(name)
        run = candidate[1] if candidate else None
        state = run.state if run else "missing"
        loss = None
        if run and state == "finished":
            loss = finite_float(run.summary['val_loss']) if 'val_loss' in run.summary else final_metric(run, 'val_loss')
            if loss is None or loss <= 0:
                loss, state = None, "invalid final loss"
        rows.append(dict(dropout=c.dropout, sequences=c.num_train_sequences,
                         tokens=c.num_train_sequences*c.train_dataset.context_length,
                         val_loss=loss, state=state, reused_endpoint=c.run_name_suffix is None,
                         run_name=name, url=run.url if run else ""))
    return sorted(rows, key=lambda r: (r['dropout'], r['sequences']))


def plot_results(rows, plot_dir=PLOT_DIR):
    set_style()
    plot_dir.mkdir(parents=True, exist_ok=True)
    csv_path = plot_dir / f"{EXPERIMENT_KEY}_results.csv"
    with csv_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['dropout','sequences','tokens','val_loss','state','reused_endpoint','run_name','url'])
        writer.writeheader()
        writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(8, 5.5), layout='constrained')
    for dropout, color in zip(DROPOUTS, ('#7030A0', '#409DC5')):
        subset = [r for r in rows if r['dropout']==dropout]
        x = [r['tokens']/1e6 for r in subset]
        y = [r['val_loss'] if r['val_loss'] is not None else np.nan for r in subset]
        ax.plot(x, y, marker='o', color=color, linewidth=1.8,
                label='Baseline' if dropout==0 else f'Dropout {dropout:g}')
        for r in subset:
            if r['val_loss'] is not None:
                ax.annotate(f"{r['val_loss']:.3f}", (r['tokens']/1e6, r['val_loss']),
                            xytext=(0, 8 if dropout else -14), textcoords='offset points',
                            ha='center', fontsize=9, color=color)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ticks=sorted({r['tokens']/1e6 for r in rows})
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{x:g}' for x in ticks])
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    ax.yaxis.set_minor_formatter(mticker.FormatStrFormatter('%.2f'))
    if not any(r['val_loss'] is not None for r in rows):
        ax.set_ylim(2, 10)
        ax.text(.5,.5,'No completed results',transform=ax.transAxes,ha='center')
    ax.set_xlabel('Training tokens (millions, log scale)')
    ax.set_ylabel('Final validation loss (log scale)')
    ax.set_title('d8 data scaling: baseline vs. dropout')
    ax.legend(frameon=False)
    ax.grid(True, which='both', linestyle=':', alpha=.3)
    ax.margins(x=.12,y=.2)
    unavailable=[f"dropout {r['dropout']:g}, {r['sequences']//1000}k: {r['state']}" for r in rows if r['val_loss'] is None]
    caption='One epoch; 600k-sequence endpoints reused from Part A when available.'
    if unavailable:
        caption+='\nUnavailable: '+ '; '.join(unavailable)
    import textwrap
    fig.supxlabel('\n'.join(textwrap.wrap(caption,100)),fontsize=8)
    path=plot_dir/f'{EXPERIMENT_KEY}_loss_summary.png'
    fig.savefig(path)
    plt.close(fig)
    return path, csv_path


def main():
    rows=collect_results()
    for r in rows:
        print(f"dropout={r['dropout']:g} sequences={r['sequences']} loss={r['val_loss']} state={r['state']} {r['url']}")
    for p in plot_results(rows):
        print(p.resolve())


if __name__ == '__main__':
    main()
