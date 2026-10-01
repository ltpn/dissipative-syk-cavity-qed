#!/bin/bash -l
#SBATCH --job-name=spectra_prod
#SBATCH --hint=multithread
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err

set -euo pipefail

repo=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
scratch_root=${repo}/data/slurm
config_path=${PRODUCTION_CONFIG:?set PRODUCTION_CONFIG to a tracked production config}
workers_per_node=${WORKERS_PER_NODE:-16}
cpus_per_worker=${CPUS_PER_WORKER:-16}
threads_per_worker=${JULIA_THREADS_PER_WORKER:-2}
# OPENBLAS_THREADS_PER_WORKER: optional override for the per-worker BLAS
# blocks used up to N_orb = 8, where each worker's dense eigen(L) fits well
# inside its private memory allocation.  For larger K (e.g. N=10 f=3 with
# K = 14400), allow explicit multi-threaded BLAS so the O(K^3) eigen /
# svdvals actually use the 32-core-per-worker slice.  Keeps
# OMP_NUM_THREADS=1 so Julia's task-parallel loops stay serialized.
openblas_threads_per_worker=${OPENBLAS_THREADS_PER_WORKER:-1}

source "${repo}/src/model/scripts/slurm/batch_helpers.sh"

mkdir -p "${scratch_root}/logs"
cd "${repo}"

export JULIA_PROJECT="${repo}/src/environment"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

merged_dir=$("${JULIA:-julia}" --project="${JULIA_PROJECT}" -e 'using TOML; print(String(TOML.parsefile(ARGS[1])["output"]["data_subdir"]))' "${config_path}")
[[ "${merged_dir}" == /* ]] || merged_dir="${repo}/data/${merged_dir}"
run_root=$(dirname "${merged_dir}")
config_store="${run_root}/configs"
mkdir -p "${run_root}/merged" "${run_root}/shards" "${run_root}/figures"

array_min=${SLURM_ARRAY_TASK_MIN:-0}
array_max=${SLURM_ARRAY_TASK_MAX:-0}
array_id=${SLURM_ARRAY_TASK_ID:-0}
node_index=$((array_id - array_min))
array_nodes=$((array_max - array_min + 1))
total_shards=$((array_nodes * workers_per_node))

archive_production_config "${config_path}" "${config_store}" "${array_id}" "${array_min}"

log_root="${scratch_root}/production/logs/${SLURM_JOB_ID}"
mkdir -p "${log_root}"

echo "host=$(hostname)"
echo "slurm_job_id=${SLURM_JOB_ID}"
echo "array_id=${array_id}"
echo "array_nodes=${array_nodes}"
echo "workers_per_node=${workers_per_node}"
echo "cpus_per_worker=${cpus_per_worker}"
echo "threads_per_worker=${threads_per_worker}"
echo "openblas_threads_per_worker=${openblas_threads_per_worker}"
echo "config_path=${config_path}"
echo "merged_dir=${merged_dir}"
echo "run_root=${run_root}"
echo "julia=$(command -v "${JULIA:-julia}")"

pids=()
for worker in $(seq 1 "${workers_per_node}"); do
    global_shard_index=$((node_index * workers_per_node + worker))
    worker_tag=$(printf '%02d' "${worker}")
    worker_dir="${run_root}/shards/node_$(printf '%03d' "${node_index}")/worker_${worker_tag}"
    mkdir -p "${worker_dir}"

    out="${log_root}/worker_${worker_tag}.out"
    err="${log_root}/worker_${worker_tag}.err"
    echo "launch worker=${worker_tag} shard=${global_shard_index}/${total_shards} dir=${worker_dir}"

    srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task="${cpus_per_worker}" \
        --distribution=block:block --cpu-bind=verbose,cores \
        /usr/bin/env JULIA_NUM_THREADS="${threads_per_worker}" \
            OPENBLAS_NUM_THREADS="${openblas_threads_per_worker}" \
            OMP_NUM_THREADS=1 \
            JULIA_PROJECT="${JULIA_PROJECT}" \
            "${JULIA:-julia}" --project="${JULIA_PROJECT}" \
            src/model/scripts/sweeps/spectrum_sweep.jl \
            --config "${config_path}" \
            --shard-index "${global_shard_index}" \
            --shard-count "${total_shards}" \
            --data-subdir "${worker_dir}" > "${out}" 2> "${err}" &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done

exit "${status}"