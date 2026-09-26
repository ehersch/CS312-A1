from experiments.problem_1.paired_sweeps.batch_size_lr_sweep.plot_batch_size_lr_tuning import (
    SPEC as BATCH_SIZE_SPEC,
)
from experiments.problem_1.paired_sweeps.warmup_lr_sweep.plot_warmup_lr_tuning import (
    SPEC as WARMUP_SPEC,
)
from experiments.problem_1.paired_sweeps.weight_decay_lr_sweep.plot_weight_decay_lr_tuning import (
    SPEC as WEIGHT_DECAY_SPEC,
)
from experiments.problem_1.paired_sweeps.plot_utils import plot_experiment


def main() -> None:
    for spec in (BATCH_SIZE_SPEC, WARMUP_SPEC, WEIGHT_DECAY_SPEC):
        plot_experiment(spec)


if __name__ == "__main__":
    main()
