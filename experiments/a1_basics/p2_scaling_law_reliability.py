from modal_train import launch_training_jobs
from model_config import depth_model_config
from train import TrainConfig


RUNS = []
seen_configs = set()
for depth in range(4, 10):
    for learning_rate, lr_schedule, dropout in [
        (0.003, "linear", 0.0),
        (0.003, "constant", 0.0),
        (0.003, "linear", 0.2),
        (0.03, "linear", 0.0),
    ]:
        config_key = (depth, learning_rate, lr_schedule, dropout)
        if config_key in seen_configs:
            continue
        seen_configs.add(config_key)
        RUNS.append(
            TrainConfig(
                model_config=depth_model_config(depth),
                learning_rate=learning_rate,
                lr_schedule=lr_schedule,
                dropout=dropout,
            )
        )


# TODO: Add your own slope-bending and scaling-law-breaking interventions for
# parts (b) and (c). The generated run name will encode the config changes.

# Preserve Part A configurations for its plotting script.
PART_A_RUNS = RUNS

# --- PART B: data scaling at fixed d8, baseline vs. dropout ---
EXPERIMENT_KEY = "p2b-data-scaling-v1"
TRAIN_SEQUENCE_COUNTS = (75_000, 150_000, 300_000)
DROPOUTS = (0.0, 0.2)

def build_part_b_runs():
    """Six d8 runs: baseline/dropout at 75k, 150k, and 300k sequences."""
    return [
        TrainConfig(
            model_config=depth_model_config(8),
            num_train_sequences=num_train_sequences,
            num_epochs=1.0,
            dropout=dropout,
            run_name_suffix=EXPERIMENT_KEY,
            wandb_tags=(EXPERIMENT_KEY,),
        )
        for num_train_sequences in TRAIN_SEQUENCE_COUNTS
        for dropout in DROPOUTS
    ]


# --- PART C: extreme LR across model-size and data scaling ---
PART_C_EXPERIMENT_KEY = "p2c-high-lr-v1"
PART_C_LEARNING_RATES = (0.1, 0.3)
PART_C_DEPTHS = tuple(range(4, 10))


def build_part_c_runs():
    """18 runs; test instability rather than assume these LRs break scaling.

    Model-size axis: d4-d9 at 600k sequences, two LRs (12 runs).
    Data axis: d8 at 75k/150k/300k sequences, two LRs (6 runs).
    The d8/600k configurations from the model-size sweep are also the final
    data-scaling endpoints; do not launch them again. Compare against existing
    Part A model-size baselines and Part B data-scaling baselines.
    """
    model_and_data_sizes = [
        (depth, 600_000) for depth in PART_C_DEPTHS
    ] + [
        (8, num_train_sequences) for num_train_sequences in TRAIN_SEQUENCE_COUNTS
    ]
    return [
        TrainConfig(
            model_config=depth_model_config(depth),
            num_train_sequences=num_train_sequences,
            num_epochs=1.0,
            learning_rate=learning_rate,
            lr_schedule="linear",
            dropout=0.0,
            run_name_suffix=PART_C_EXPERIMENT_KEY,
            wandb_tags=(PART_C_EXPERIMENT_KEY,),
        )
        for depth, num_train_sequences in model_and_data_sizes
        for learning_rate in PART_C_LEARNING_RATES
    ]


# Preserve existing Part B wrappers; Part C must be explicitly selected.
RUNS = build_part_b_runs()


def main():
    import argparse
    from train import training_run_name

    parser = argparse.ArgumentParser(description="Problem 2 scaling experiments")
    parser.add_argument("--part", choices=("a", "b", "c"), default="b")
    parser.add_argument("--dry-run", action="store_true", help="Print configurations without submitting jobs.")
    parser.add_argument("--wait", action="store_true", help="Wait for submitted calls to finish.")
    args = parser.parse_args()
    runs = {"a": lambda: PART_A_RUNS, "b": build_part_b_runs, "c": build_part_c_runs}[args.part]()
    print(f"Part {args.part.upper()}: {len(runs)} runs")
    if args.dry_run:
        for config in runs:
            print(training_run_name(config))
        return
    jobs = launch_training_jobs(runs, max_parallel_runs=2, gpu="H100")
    if args.wait:
        import modal
        failures = []
        for job in jobs:
            if job.get("skipped"):
                continue
            call_id = job["modal_call_id"]
            print(f"Waiting for job {job['job_index']}: {call_id}", flush=True)
            try:
                modal.FunctionCall.from_id(call_id).get()
            except Exception as exc:
                failures.append(call_id)
                print(f"Job {call_id} failed: {exc}", flush=True)
        if failures:
            raise SystemExit(f"{len(failures)} call(s) failed. Inspect the errors before retrying.")


if __name__ == "__main__":
    main()
