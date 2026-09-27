# Problem 1(c): scheduler sweeps

All 30 runs use the default d8 model, batch size, weight decay, training tokens,
and epoch count. Only scheduler, peak LR, and warmup fraction vary.

| Group | Schedules | Warmup | LRs | Runs |
|---|---|---|---|---|
| Schedule comparison | linear, cos, constant, wsd0.2 | 0.01 | 0.001, 0.003, 0.009 | 12 |
| Additional WSD fractions | wsd0.1, wsd0.5 | 0.01 | same | 6 |
| Longer warmup | linear, cos, constant, wsd0.2 | 0.1 | same | 12 |

In this repository, `constant` ignores warmup. Its three longer-warmup runs are
repeat controls, not different LR schedules. Thus there are 30 labeled runs but
27 distinct training recipes. WSD fractions specify the fraction of total steps
spent in the final linear decay; the default-warmup wsd0.2 runs are reused in the
WSD comparison without launching duplicates.

Run from the repository root:

```bash
bash experiments/problem_1/scheduler_sweeps/run_all.sh
```

The script executes two separate commands:

```bash
uv run python -u -m experiments.problem_1.scheduler_sweeps.launch_scheduler_tuning --wait
uv run python -m experiments.problem_1.scheduler_sweeps.plot_scheduler_tuning
```

The launcher submits one detached Modal app with at most two concurrent H100
jobs. `--wait` keeps the local command waiting for those calls. Keep the local
process alive for automatic plotting. Without `--wait`, the launcher returns
after submission; run the plotting command yourself after training finishes.
Avoid launching while other sweeps consume your environment's two GPU slots.

Plots are saved in `plots/`: a schedule comparison, a WSD-fraction comparison,
and one warmup comparison for each of the four schedule families. The plots use
the latest matching run for each full configuration, omit unfinished runs from
final-loss rankings, and explicitly report unavailable results. The Bash script
attempts plotting available results even if some training calls failed, then
returns a nonzero status for those failures.

Change `EXPERIMENT_KEY` before a changed experiment. All plots import that key
and the LR grid from the launcher. The WSD comparison uses fractions 0.1, 0.2,
and 0.5; update its values as well if changing the WSD grid.
