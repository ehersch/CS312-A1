from experiments.problem_1.paired_sweeps.batch_size_lr_sweep.launch_batch_size_lr_tuning import (
    build_runs as batch_size_runs,
)
from experiments.problem_1.paired_sweeps.warmup_lr_sweep.launch_warmup_lr_tuning import (
    build_runs as warmup_runs,
)
from experiments.problem_1.paired_sweeps.weight_decay_lr_sweep.launch_weight_decay_lr_tuning import (
    build_runs as weight_decay_runs,
)


def build_runs():
    return batch_size_runs() + warmup_runs() + weight_decay_runs()


def main() -> None:
    from modal_train import launch_training_jobs

    launch_training_jobs(build_runs(), max_parallel_runs=2, gpu="H100")


if __name__ == "__main__":
    main()
