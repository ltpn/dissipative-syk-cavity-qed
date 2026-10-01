#!/usr/bin/env bash
# Site resources are supplied by sbatch or submit_spectral_pipeline.sh.
set -euo pipefail
repo=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$repo"
stage=${1:?Usage: pipeline_stage.sbatch STAGE [pipeline options]}
shift
args=("$stage" "$@")
if [[ "$stage" != postprocess && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    args+=(--shard-index "$SLURM_ARRAY_TASK_ID" --shard-count "${SLURM_ARRAY_TASK_COUNT:?}")
fi
cpus=${SLURM_CPUS_PER_TASK:-1}
export JULIA_NUM_THREADS=${JULIA_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=$cpus
export OMP_NUM_THREADS=1
exec "${JULIA:-julia}" --project="$repo/src/environment" \
    "$repo/src/model/scripts/pipeline.jl" "${args[@]}" --blas-threads "$cpus"
