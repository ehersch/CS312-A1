from modal_train import launch_training_jobs
from train import TrainConfig


RUNS = [
    TrainConfig(
        deterministic=True,
        run_name_suffix="deterministic-reference-1",
    ),
    TrainConfig(
        deterministic=True,
        run_name_suffix="deterministic-reference-2",
    ),
]


# TODO: Vary model_seed and data_seed separately and together.
# For hardware nondeterminism, run the same config on at least two GPU
# types by passing `gpu=...` to launch_training_jobs from a local experiment.

"""from modal_train import launch_training_jobs
from train import TrainConfig

EXPERIMENT_KEY = "p3a-joint-variation-v1"

RUNS = [
    TrainConfig(
        model_seed=42,
        data_seed=42,
        deterministic=False,
        run_name_suffix=f"{EXPERIMENT_KEY}-h100-1",
        wandb_tags=(EXPERIMENT_KEY,),
    ),
    TrainConfig(
        model_seed=43,
        data_seed=43,
        deterministic=False,
        run_name_suffix=f"{EXPERIMENT_KEY}-a100-2",
        wandb_tags=(EXPERIMENT_KEY,),
    ),
    TrainConfig(
        model_seed=44,
        data_seed=44,
        deterministic=False,
        run_name_suffix=f"{EXPERIMENT_KEY}-h100-3",
        wandb_tags=(EXPERIMENT_KEY,),
    ),
]

GPUS = ("H100", "A100", "H100")


def main():
    import modal

    # Wait for each run before submitting the next.
    for config, gpu in zip(RUNS, GPUS):
        jobs = launch_training_jobs(
            [config],
            gpu=gpu,
            max_parallel_runs=1,
        )
        for job in jobs:
            if not job.get("skipped"):
                modal.FunctionCall.from_id(job["modal_call_id"]).get()


if __name__ == "__main__":
    main()"""


def main():
    launch_training_jobs(RUNS)


if __name__ == "__main__":
    main()
