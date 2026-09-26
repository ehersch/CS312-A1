#!/usr/bin/env bash
# Submit all paired sweeps, wait for their detached Modal jobs, then plot.
# Keep this process running for automatic plotting (e.g. in tmux).
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$SCRIPT_DIR/../../.."

uv run python -u - <<'PY'
import modal

from experiments.problem_1.paired_sweeps.launch_all import build_runs
from experiments.problem_1.paired_sweeps.plot_all import main as plot_all
from modal_train import launch_training_jobs

jobs = launch_training_jobs(build_runs(), max_parallel_runs=2, gpu="H100")
failures = []
for job in jobs:
    if job.get("skipped"):
        continue
    call_id = job["modal_call_id"]
    print(f"Waiting for job {job['job_index']}: {call_id}", flush=True)
    try:
        # All jobs were already submitted; waiting here does not serialize training.
        modal.FunctionCall.from_id(call_id).get()
    except Exception as exc:
        failures.append(call_id)
        print(f"Job {call_id} failed: {exc}", flush=True)

print("All submitted calls have been checked. Generating plots...", flush=True)
plot_all()

if failures:
    raise SystemExit(
        f"{len(failures)} job(s) failed. Plots include available results; "
        "inspect the reported Modal calls before interpreting missing results."
    )
print("Finished. Plots are in each paired sweep's plots/ directory.")
PY
