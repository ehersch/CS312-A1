#!/usr/bin/env bash
# Train only d4-d7, then plot. Keep this process alive for automatic plotting.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$SCRIPT_DIR/../.."

training_status=0
# Import the original RUNS without executing its main(), so this wrapper can
# limit concurrency and wait for completion without changing the source file.
uv run python -u - <<'PY' || training_status=$?
import modal
from experiments.a1_basics.p2_scaling_law_reliability import RUNS
from modal_train import launch_training_jobs

assert len(RUNS) == 16, "First stage must contain exactly 16 runs."
assert {c.model_config.name for c in RUNS} == {"d4", "d5", "d6", "d7"}, "First stage must only train d4-d7."
jobs = launch_training_jobs(RUNS, max_parallel_runs=2, gpu="H100")
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
    raise SystemExit(f"{len(failures)} call(s) failed. Plotting available results next.")
PY
if [[ "$training_status" -eq 130 || "$training_status" -eq 143 ]]; then
    exit "$training_status"
fi
uv run python -m experiments.a1_basics.plot_p2_scaling_law_reliability
exit "$training_status"
