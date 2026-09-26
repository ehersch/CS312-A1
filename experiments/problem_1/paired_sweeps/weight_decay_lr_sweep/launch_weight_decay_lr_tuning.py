EXPERIMENT_KEY = "p1b-weight-decay-lr-v1"
LEARNING_RATES = (1e-3, 3e-3, 1e-2)
WEIGHT_DECAYS = (0.1, 1.0)


def build_runs():
    from train import TrainConfig

    return [
        TrainConfig(
            learning_rate=learning_rate,
            weight_decay=value,
            run_name_suffix=EXPERIMENT_KEY,
            wandb_tags=(EXPERIMENT_KEY,),
        )
        for value in WEIGHT_DECAYS
        for learning_rate in LEARNING_RATES
    ]


def main() -> None:
    from modal_train import launch_training_jobs

    launch_training_jobs(build_runs(), max_parallel_runs=2, gpu="H100")


if __name__ == "__main__":
    main()
