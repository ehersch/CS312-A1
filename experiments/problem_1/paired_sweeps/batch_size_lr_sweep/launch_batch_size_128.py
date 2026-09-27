"""Add only the three batch-128 runs to the existing batch-size/LR experiment."""

from experiments.problem_1.paired_sweeps.batch_size_lr_sweep.launch_batch_size_lr_tuning import (
    EXPERIMENT_KEY,
    LEARNING_RATES,
)


def build_runs():
    from train import TrainConfig

    return [
        TrainConfig(
            batch_size=128,
            learning_rate=learning_rate,
            run_name_suffix=EXPERIMENT_KEY,
            wandb_tags=(EXPERIMENT_KEY,),
        )
        for learning_rate in LEARNING_RATES
    ]


def main() -> None:
    from modal_train import launch_training_jobs

    launch_training_jobs(build_runs(), max_parallel_runs=2, gpu="H100")


if __name__ == "__main__":
    main()
