from pathlib import Path

from experiments.problem_1.paired_sweeps.weight_decay_lr_sweep.launch_weight_decay_lr_tuning import (
    EXPERIMENT_KEY,
    LEARNING_RATES,
    WEIGHT_DECAYS,
)
from experiments.problem_1.paired_sweeps.plot_utils import SweepSpec, plot_experiment


SPEC = SweepSpec(
    experiment_key=EXPERIMENT_KEY,
    parameter="weight_decay",
    parameter_label="Weight decay",
    values=WEIGHT_DECAYS,
    learning_rates=LEARNING_RATES,
    plot_dir=Path(__file__).resolve().parent / "plots",
)


def main() -> None:
    plot_experiment(SPEC)


if __name__ == "__main__":
    main()
