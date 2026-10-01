#!/usr/bin/env bash
# Resources (including 30-minute generation / 24-hour spectral time) set by controller.
set -euo pipefail
repo=${REPO_ROOT:?}
job_file=${1:?recovery job JSON}
cd "$repo"
exec "${PYTHON:-/usr/bin/python3.11}" "$repo/src/model/scripts/slurm/timeout_recovery.py" worker --root "${PILOT_ROOT:?}" --job-file "$job_file"
