"""Part 1(c): 30 runs studying schedule, peak LR, WSD fraction, and warmup."""

EXPERIMENT_KEY = "p1c-schedulers-v1"
LEARNING_RATES = (0.001, 0.003, 0.009)
SCHEDULES = ("linear", "cos", "constant", "wsd0.2")
EXTRA_WSD_SCHEDULES = ("wsd0.1", "wsd0.5")
WARMUP_PERCENTS = (0.01, 0.1)


def build_runs():
    from train import TrainConfig

    # 18 default-warmup runs + 12 longer-warmup runs. Reuse wsd0.2
    # results for both the schedule comparison and the WSD-fraction plot.
    # ConstantScheduler ignores warmup: its second group is a repeat control.
    return [
        TrainConfig(
            learning_rate=lr,
            lr_schedule=schedule,
            warmup_percent=warmup,
            run_name_suffix=EXPERIMENT_KEY,
            wandb_tags=(EXPERIMENT_KEY,),
        )
        for warmup, schedules in (
            (WARMUP_PERCENTS[0], SCHEDULES + EXTRA_WSD_SCHEDULES),
            (WARMUP_PERCENTS[1], SCHEDULES),
        )
        for schedule in schedules
        for lr in LEARNING_RATES
    ]


def main() -> None:
    import argparse

    from modal_train import launch_training_jobs

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait", action="store_true", help="Wait for all submitted calls before exiting.")
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
        raise SystemExit(f"{len(failures)} call(s) failed; inspect the errors above. Available results can still be plotted.")


if __name__ == "__main__":
    main()
