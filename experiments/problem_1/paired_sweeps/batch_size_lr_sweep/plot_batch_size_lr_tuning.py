from pathlib import Path

from experiments.problem_1.paired_sweeps.batch_size_lr_sweep.launch_batch_size_lr_tuning import (
    EXPERIMENT_KEY,
    LEARNING_RATES,
    BATCH_SIZES,
)
from experiments.problem_1.paired_sweeps.plot_utils import SweepSpec, plot_experiment


SPEC = SweepSpec(
    experiment_key=EXPERIMENT_KEY,
    parameter="batch_size",
    parameter_label="Batch size",
    values=BATCH_SIZES,
    learning_rates=LEARNING_RATES,
    plot_dir=Path(__file__).resolve().parent / "plots",
)


def main() -> None:
    plot_experiment(SPEC)


if __name__ == "__main__":
    main()
