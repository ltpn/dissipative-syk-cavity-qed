#!/bin/bash -l
#SBATCH --job-name=cgs_b0_prod
#SBATCH --hint=multithread
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --array=0-39
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err

set -euo pipefail

[[ -n "${SLURM_JOB_ID:-}" ]] || {
    echo "ERROR: production must run under Slurm" >&2
    exit 2
}

repo=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
config_path=${repo}/src/model/configs/cgs_survival_beta0_n8_q4.toml
output_root=${repo}/data/cgs_survival
calibration_path=${output_root}/calibration/figure3_diss_syk_n8_q4_m300.jld2
compute_driver=${repo}/src/model/scripts/analysis/figures/cgs_survival_compute.jl
project=${repo}/src/environment
workers_per_node=16
cpus_per_worker=16
total_shards=640
dataset_keys=(
    fig1_eta_0p1 fig1_eta_0p4 fig1_eta_1 fig1_eta_2 fig1_diss_syk
    fig3_dtilde_0p01 fig3_dtilde_0p1 fig3_dtilde_1 fig3_dtilde_10 fig3_diss_syk
)

array_min=${SLURM_ARRAY_TASK_MIN:-0}
array_max=${SLURM_ARRAY_TASK_MAX:-39}
array_id=${SLURM_ARRAY_TASK_ID:?missing SLURM_ARRAY_TASK_ID}
[[ "${array_min}" == 0 && "${array_max}" == 39 ]] || {
    echo "ERROR: production requires exactly array indices 0-39" >&2
    exit 2
}
node_index=$((array_id - array_min))
[[ $(((array_max - array_min + 1) * workers_per_node)) == "${total_shards}" ]] || exit 2
[[ "${#dataset_keys[@]}" == 10 ]] || exit 2
[[ -f "${calibration_path}" ]] || {
    echo "ERROR: calibrated Figure-3 control is missing: ${calibration_path}" >&2
    exit 2
}

log_root=${repo}/data/slurm/cgs_survival/${SLURM_ARRAY_JOB_ID}/${array_id}
mkdir -p "${log_root}" "${output_root}"
cd "${repo}"
export STORE="${repo}/data"
export JULIA_PROJECT=${project}
export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

pids=()
for worker in $(seq 1 "${workers_per_node}"); do
    global_shard_index=$((node_index * workers_per_node + worker))
    dataset_index=$(((global_shard_index - 1) / 64 + 1))
    seed=$(((global_shard_index - 1) % 64 + 1))
    [[ "${global_shard_index}" -ge 1 && "${global_shard_index}" -le "${total_shards}" ]] || exit 2
    dataset_key=${dataset_keys[$((dataset_index - 1))]}
    worker_tag=$(printf '%02d' "${worker}")
    srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task="${cpus_per_worker}" \
        --distribution=block:block --cpu-bind=cores \
        /usr/bin/env JULIA_NUM_THREADS=1 \
            OPENBLAS_NUM_THREADS="${cpus_per_worker}" OMP_NUM_THREADS=1 \
            JULIA_PROJECT="${JULIA_PROJECT}" \
            "${JULIA:-julia}" --project="${JULIA_PROJECT}" "${compute_driver}" \
            --config "${config_path}" --dataset-key "${dataset_key}" \
            --seed "${seed}" --output-root "${output_root}" \
            --figure3-calibration "${calibration_path}" \
            >"${log_root}/worker_${worker_tag}_${dataset_key}_seed_${seed}.out" \
            2>"${log_root}/worker_${worker_tag}_${dataset_key}_seed_${seed}.err" &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    wait "${pid}" || status=1
done
exit "${status}"
