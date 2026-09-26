EXPERIMENT_KEY = "batch-size-tuning-v1"
BATCH_SIZE = (16, 32, 64, 128, 256)

def build_runs():
    from train import TrainConfig

    return [
        TrainConfig(
            batch_size=batch_size,
            run_name_suffix=EXPERIMENT_KEY,
            wandb_tags=(EXPERIMENT_KEY,),
        )
        for batch_size in BATCH_SIZE
    ]


def main() -> None:
    from modal_train import launch_training_jobs

    launch_training_jobs(build_runs(), max_parallel_runs=2, gpu="H100")


if __name__ == "__main__":
    main()
