#!/bin/bash -l
#SBATCH --job-name=spectra_fig3_ds
#SBATCH --hint=multithread
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err

# fig3_multimode_dualsocket.sbatch
#
# Each array
# element takes one node and launches TWO Julia workers,
# each pinned to one socket via `srun --cpu-bind=verbose,sockets
# --cpus-per-task=128`, with OPENBLAS_NUM_THREADS = 16 per worker.  The
# driver runs a SINGLE delta_tilde value (--delta-tildes ${FIG3_DELTA_TILDE})
# and each worker gets one seed via the sweep-style shard-index / count.
#
# Sharding rule:
#   total_shards = array_nodes * 2      (each shard = 1 seed)
#   global_shard_index = node_index * 2 + worker
# With array_nodes chosen such that total_shards == n_seeds, every worker
# does exactly one svdvals(L_c) at K = 14400.
#
# Required environment:
#   FIG3_N_ORB           system size (e.g. 10)
#   FIG3_N_SEEDS         per-delta_tilde seed count (e.g. 64)
#   FIG3_CACHE_DIR       absolute path to the shared multimode cache dir
#   FIG3_DELTA_TILDE     single delta_tilde value for this sbatch invocation
#
# Optional environment:
#   FIG3_FILLING                     pass `--filling <k>` (default: half-filling)
#   WORKERS_PER_NODE                 default 2 (socket-pinned; do not change)
#   CPUS_PER_WORKER                  default 128 (one socket)
#   JULIA_THREADS_PER_WORKER         default 1
#   OPENBLAS_THREADS_PER_WORKER      default 16

set -euo pipefail

repo=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
scratch_root=${repo}/data/slurm

n_orb=${FIG3_N_ORB:?set FIG3_N_ORB (system size, e.g. 10)}
n_seeds=${FIG3_N_SEEDS:?set FIG3_N_SEEDS (per-delta_tilde seed count, e.g. 64)}
cache_dir=${FIG3_CACHE_DIR:?set FIG3_CACHE_DIR (absolute path)}
delta_tilde=${FIG3_DELTA_TILDE:?set FIG3_DELTA_TILDE (single value, e.g. 0.1)}
filling_arg=""
if [[ -n "${FIG3_FILLING:-}" ]]; then
    filling_arg="--filling ${FIG3_FILLING}"
fi
workers_per_node=${WORKERS_PER_NODE:-2}
cpus_per_worker=${CPUS_PER_WORKER:-128}
threads_per_worker=${JULIA_THREADS_PER_WORKER:-1}
openblas_threads_per_worker=${OPENBLAS_THREADS_PER_WORKER:-16}

if [[ "${workers_per_node}" != "2" ]]; then
    echo "ERROR: fig3_multimode_dualsocket.sbatch is hard-coded for" >&2
    echo "       WORKERS_PER_NODE = 2 (one worker per socket).  Got ${workers_per_node}." >&2
    exit 2
fi

mkdir -p "${scratch_root}/logs"
mkdir -p "${cache_dir}"
cd "${repo}"

export JULIA_PROJECT="${repo}/src/environment"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

array_min=${SLURM_ARRAY_TASK_MIN:-0}
array_max=${SLURM_ARRAY_TASK_MAX:-0}
array_id=${SLURM_ARRAY_TASK_ID:-0}
node_index=$((array_id - array_min))
array_nodes=$((array_max - array_min + 1))
total_shards=$((array_nodes * workers_per_node))

log_root="${scratch_root}/fig3_ds/logs/${SLURM_JOB_ID}"
mkdir -p "${log_root}"

# Slugify delta_tilde for filenames (e.g. "0.1" -> "0p1").
dtilde_slug=$(printf '%s' "${delta_tilde}" | tr '.' 'p' | tr '-' 'm')

echo "host                       = $(hostname)"
echo "slurm_job_id               = ${SLURM_JOB_ID}"
echo "array_id                   = ${array_id}"
echo "array_nodes                = ${array_nodes}"
echo "workers_per_node           = ${workers_per_node} (socket-pinned)"
echo "cpus_per_worker            = ${cpus_per_worker}"
echo "threads_per_worker         = ${threads_per_worker}"
echo "openblas_threads_per_worker= ${openblas_threads_per_worker}"
echo "n_orb                      = ${n_orb}"
echo "filling_arg                = ${filling_arg}"
echo "n_seeds                    = ${n_seeds}"
echo "delta_tilde                = ${delta_tilde}"
echo "total_shards               = ${total_shards}"
echo "cache_dir                  = ${cache_dir}"
echo "julia                      = $(command -v "${JULIA:-julia}")"

pids=()
for worker in $(seq 1 "${workers_per_node}"); do
    global_shard_index=$((node_index * workers_per_node + worker))
    worker_tag=$(printf 'socket%d' $((worker - 1)))

    out="${log_root}/dt_${dtilde_slug}_node_$(printf '%03d' "${node_index}")_worker_${worker_tag}.out"
    err="${log_root}/dt_${dtilde_slug}_node_$(printf '%03d' "${node_index}")_worker_${worker_tag}.err"
    echo "launch dt=${delta_tilde} worker=${worker_tag} shard=${global_shard_index}/${total_shards}"

    srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task="${cpus_per_worker}" \
        --distribution=block:block --cpu-bind=verbose,sockets \
        /usr/bin/env JULIA_NUM_THREADS="${threads_per_worker}" \
            OPENBLAS_NUM_THREADS="${openblas_threads_per_worker}" \
            OMP_NUM_THREADS=1 \
            JULIA_PROJECT="${JULIA_PROJECT}" \
            "${JULIA:-julia}" --project="${JULIA_PROJECT}" \
            src/model/scripts/analysis/figures/figure3_multimode_unfolded.jl \
            --build-cache-only --with-dynamics --delta-cd 1.0 \
            --n-orb "${n_orb}" \
            ${filling_arg} \
            --n-seeds "${n_seeds}" \
            --delta-tildes "${delta_tilde}" \
            --shard-index "${global_shard_index}" \
            --shard-count "${total_shards}" \
            --cache-dir "${cache_dir}" > "${out}" 2> "${err}" &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done

exit "${status}"
