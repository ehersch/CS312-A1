from pathlib import Path

from experiments.problem_1.scheduler_sweeps.launch_scheduler_tuning import (
    EXPERIMENT_KEY,
    LEARNING_RATES,
    SCHEDULES,
    WARMUP_PERCENTS,
)
from experiments.problem_1.scheduler_sweeps.plot_utils import SweepSpec, plot_experiment


PLOT_DIR = Path(__file__).resolve().parent / "plots"


def build_specs():
    common = dict(experiment_key=EXPERIMENT_KEY, learning_rates=LEARNING_RATES, plot_dir=PLOT_DIR)
    specs = [
        SweepSpec(
            **common,
            parameter="lr_schedule",
            parameter_label="Schedule (1% warmup; constant ignores warmup)",
            values=SCHEDULES,
            fixed_config={"warmup_percent": WARMUP_PERCENTS[0]},
            output_name="schedule_comparison",
        ),
        SweepSpec(
            **common,
            parameter="lr_schedule",
            parameter_label="WSD decay fraction (1% warmup)",
            values=("wsd0.1", "wsd0.2", "wsd0.5"),
            fixed_config={"warmup_percent": WARMUP_PERCENTS[0]},
            output_name="wsd_fraction_comparison",
        ),
    ]
    for schedule in SCHEDULES:
        note = " (ignored; repeat controls)" if schedule == "constant" else ""
        specs.append(SweepSpec(
            **common,
            parameter="warmup_percent",
            parameter_label=f"{schedule}: warmup fraction{note}",
            values=WARMUP_PERCENTS,
            fixed_config={"lr_schedule": schedule},
            output_name=f"{schedule}_warmup_comparison",
        ))
    return specs


def main() -> None:
    for spec in build_specs():
        plot_experiment(spec)


if __name__ == "__main__":
    main()
