#!/usr/bin/env bash
# Six new runs, then plot; leave this process running for automatic plotting.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$SCRIPT_DIR/../../.."
training_status=0
uv run python -u -m experiments.problem_2.data_scaling.launch_data_scaling --wait || training_status=$?
if [[ "$training_status" -eq 130 || "$training_status" -eq 143 ]]; then
    exit "$training_status"
fi
uv run python -m experiments.problem_2.data_scaling.plot_data_scaling
exit "$training_status"
