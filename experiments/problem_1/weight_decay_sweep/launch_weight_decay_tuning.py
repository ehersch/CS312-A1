EXPERIMENT_KEY = "p1a-weight-decay-v1"
WEIGHT_DECAYS = (0.01, 0.03, 0.1, 0.3, 1.0)

def build_runs():
    from train import TrainConfig

    return [
        TrainConfig(
            weight_decay=weight_decay,
            run_name_suffix=EXPERIMENT_KEY,
            wandb_tags=(EXPERIMENT_KEY,),
        )
        for weight_decay in WEIGHT_DECAYS
    ]


def main() -> None:
    from modal_train import launch_training_jobs

    launch_training_jobs(build_runs(), max_parallel_runs=2, gpu="H100")


if __name__ == "__main__":
    main()
