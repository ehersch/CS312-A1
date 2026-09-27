#!/usr/bin/env bash
# Keep this process alive for automatic plotting. Neither command runs on import.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$SCRIPT_DIR/../../.."

# Separate uv commands: wait for all training calls, then plot available results.
training_status=0
uv run python -u -m experiments.problem_1.scheduler_sweeps.launch_scheduler_tuning --wait || training_status=$?
if [[ "$training_status" -eq 130 || "$training_status" -eq 143 ]]; then
    exit "$training_status"
fi
uv run python -m experiments.problem_1.scheduler_sweeps.plot_scheduler_tuning
exit "$training_status"
