#!/usr/bin/env bash
set -euo pipefail
repo=${REPO_ROOT:?}
root=${PILOT_ROOT:?}
stage=${1:?Usage: physical_pilot_worker.sh STAGE SHARD COUNT}
case "$stage" in generate|eigen|svd) ;; *) echo "Invalid stage: $stage" >&2; exit 2;; esac
shard=${2:?point shard index}
count=${3:?total production points}
threads=${OPENBLAS_THREADS_PER_WORKER:-64}
echo "host=$(hostname) shard=$shard/$count pid=$$"
awk '/Cpus_allowed_list|Mems_allowed_list/ {print}' /proc/self/status
echo "$(date -Is) starting $stage shard=$shard BLAS_threads=$threads"
args=(--shard-index "$shard" --shard-count "$count" --blas-threads "$threads")
if [[ $stage == generate ]]; then
    args+=(--config "$root/config.toml" --output "$root/liouvillians")
else
    args+=(--input "$root/liouvillians" --output "$root/$stage")
fi
"${JULIA:?}" --project="$repo/src/environment" \
    "$repo/src/model/scripts/pipeline.jl" "$stage" "${args[@]}"
echo "$(date -Is) completed $stage shard=$shard"
