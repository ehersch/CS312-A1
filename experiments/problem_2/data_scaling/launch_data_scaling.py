"""Part 2(b): six d8 data-scaling runs; existing 600k endpoints are not launched."""

from experiments.a1_basics.p2_scaling_law_reliability import (
    EXPERIMENT_KEY, TRAIN_SEQUENCE_COUNTS, DROPOUTS, build_part_b_runs,
)


def make_config(sequences, dropout, *, existing_endpoint=False):
    from model_config import depth_model_config
    from train import TrainConfig

    return TrainConfig(
        model_config=depth_model_config(8),
        num_train_sequences=sequences,
        num_epochs=1.0,
        dropout=dropout,
        run_name_suffix=None if existing_endpoint else EXPERIMENT_KEY,
        wandb_tags=() if existing_endpoint else (EXPERIMENT_KEY,),
    )


def build_runs():
    return build_part_b_runs()


def main():
    import argparse
    from modal_train import launch_training_jobs

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    jobs = launch_training_jobs(build_runs(), max_parallel_runs=2, gpu="H100")
    if not args.wait:
        return
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
        raise SystemExit(f"{len(failures)} call(s) failed; inspect errors before retrying.")


if __name__ == "__main__":
    main()
