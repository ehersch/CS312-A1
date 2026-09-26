EXPERIMENT_KEY = "p1a-warmup-v1"
WARMUP_PERCENTS = (0.001, 0.003, 0.01, 0.03, 0.1)

def build_runs():
    from train import TrainConfig

    return [
        TrainConfig(
            warmup_percent=warmup,
            run_name_suffix=EXPERIMENT_KEY,
            wandb_tags=(EXPERIMENT_KEY,),
        )
        for warmup in WARMUP_PERCENTS
    ]


def main() -> None:
    from modal_train import launch_training_jobs

    launch_training_jobs(build_runs(), max_parallel_runs=2, gpu="H100")


if __name__ == "__main__":
    main()
