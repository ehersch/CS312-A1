from pathlib import Path

from experiments.problem_1.paired_sweeps.warmup_lr_sweep.launch_warmup_lr_tuning import (
    EXPERIMENT_KEY,
    LEARNING_RATES,
    WARMUP_PERCENTS,
)
from experiments.problem_1.paired_sweeps.plot_utils import SweepSpec, plot_experiment


SPEC = SweepSpec(
    experiment_key=EXPERIMENT_KEY,
    parameter="warmup_percent",
    parameter_label="Warmup fraction",
    values=WARMUP_PERCENTS,
    learning_rates=LEARNING_RATES,
    plot_dir=Path(__file__).resolve().parent / "plots",
)


def main() -> None:
    plot_experiment(SPEC)


if __name__ == "__main__":
    main()
