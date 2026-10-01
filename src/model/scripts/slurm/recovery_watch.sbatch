#!/usr/bin/env bash
set -euo pipefail
repo=${REPO_ROOT:?}
cd "$repo"
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec "${PYTHON:-/usr/bin/python3.11}" "$repo/src/model/scripts/slurm/timeout_recovery.py" reconcile --root "${PILOT_ROOT:?}"
