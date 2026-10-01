#!/bin/bash -l
#SBATCH --job-name=fig3_m10_prod
#SBATCH --hint=multithread
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --array=0-3
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err

set -euo pipefail

[[ -n "${SLURM_JOB_ID:-}" ]] || {
    echo "ERROR: production must run under SLURM" >&2
    exit 2
}

repo=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
scratch_root=${repo}/data/slurm
config_path=${repo}/src/model/configs/figure3_syk4_dtilde0p01_m10_n10_f3.toml
calibration_path=${repo}/data/syk4_reference/m10/calibration/figure3_syk4_dtilde0p01_m10_n10_f3.jld2
workers_per_node=16
cpus_per_worker=16

cd "${repo}"
mkdir -p "${scratch_root}/logs"
export JULIA_PROJECT="${repo}/src/environment"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

[[ -f "${calibration_path}" ]] || {
    echo "ERROR: calibration payload missing: ${calibration_path}" >&2
    exit 2
}
"${JULIA:-julia}" --project="${JULIA_PROJECT}" -e '
using JLD2
jldopen(ARGS[1], "r") do f
    @assert f["target_delta_tilde"] == 0.01
    @assert f["n_random_jumps"] == 10
    @assert f["n_cavity_jumps"] == 1
end
' "${calibration_path}"

merged_dir=$("${JULIA:-julia}" --project="${JULIA_PROJECT}" -e \
    'using TOML; print(TOML.parsefile(ARGS[1])["output"]["data_subdir"])' \
    "${config_path}")
[[ "${merged_dir}" == /* ]] || merged_dir="${repo}/data/${merged_dir}"
run_root=$(dirname "${merged_dir}")
mkdir -p "${run_root}/shards" "${run_root}/configs"
cp "${config_path}" "${run_root}/configs/config_job_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.toml"

array_min=${SLURM_ARRAY_TASK_MIN:-0}
array_max=${SLURM_ARRAY_TASK_MAX:-3}
array_id=${SLURM_ARRAY_TASK_ID:?missing SLURM_ARRAY_TASK_ID}
node_index=$((array_id - array_min))
array_nodes=$((array_max - array_min + 1))
total_shards=$((array_nodes * workers_per_node))
[[ "${total_shards}" == "64" ]] || {
    echo "ERROR: expected exactly 64 total shards, got ${total_shards}" >&2
    exit 2
}

log_root="${scratch_root}/figure3_syk4_m10/logs/${SLURM_ARRAY_JOB_ID}/${array_id}"
mkdir -p "${log_root}"
echo "host=$(hostname) job=${SLURM_JOB_ID} array=${array_id} shards=${total_shards}"

pids=()
for worker in $(seq 1 "${workers_per_node}"); do
    global_shard_index=$((node_index * workers_per_node + worker))
    worker_tag=$(printf '%02d' "${worker}")
    worker_dir="${run_root}/shards/node_$(printf '%03d' "${node_index}")/worker_${worker_tag}"
    mkdir -p "${worker_dir}"
    out="${log_root}/worker_${worker_tag}.out"
    err="${log_root}/worker_${worker_tag}.err"

    srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task="${cpus_per_worker}" \
        --distribution=block:block --cpu-bind=verbose,cores \
        /usr/bin/env JULIA_NUM_THREADS=1 \
            OPENBLAS_NUM_THREADS="${cpus_per_worker}" OMP_NUM_THREADS=1 \
            JULIA_PROJECT="${JULIA_PROJECT}" \
            "${JULIA:-julia}" --project="${JULIA_PROJECT}" \
            src/model/scripts/sweeps/spectrum_sweep.jl \
            --config "${config_path}" \
            --shard-index "${global_shard_index}" \
            --shard-count "${total_shards}" \
            --data-subdir "${worker_dir}" >"${out}" 2>"${err}" &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    wait "${pid}" || status=1
done
exit "${status}"
