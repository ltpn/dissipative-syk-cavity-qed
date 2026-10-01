#!/bin/bash -l
#SBATCH --job-name=cgs_b0_cal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

[[ -n "${SLURM_JOB_ID:-}" ]] || {
    echo "ERROR: calibration must run under Slurm" >&2
    exit 2
}

repo=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
config_path=${repo}/src/model/configs/cgs_survival_beta0_n8_q4.toml
baseline_calibration=${repo}/data/syk4/n8/calibration.jld2
output_root=${repo}/data/cgs_survival
output=${output_root}/calibration/figure3_diss_syk_n8_q4_m300.jld2
smoke_copy=${output_root}/validation/smoke/calibration/figure3_diss_syk_n8_q4_m300.jld2
source_path=${repo}/src/model/scripts/analysis/figures/calibrate_cgs_figure3_syk4_reference.jl
project=${repo}/src/environment

mkdir -p "${repo}/data/slurm/cgs_survival" "$(dirname "${output}")" \
    "$(dirname "${smoke_copy}")" "${output_root}/configs"
cd "${repo}"
export STORE="${repo}/data"
export JULIA_PROJECT=${project}
export JULIA_NUM_THREADS=64
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

validate_calibration() {
    "${JULIA:-julia}" --project="${JULIA_PROJECT}" -e '
        using JLD2
        jldopen(ARGS[1], "r") do f
            @assert f["n_orb"] == 8
            @assert f["filling"] == 4
            @assert f["n_seeds"] == 64
            @assert f["n_random_jumps"] == 300
            @assert f["n_cavity_jumps"] == 1
            @assert f["target_delta_tilde"] == 0.01
        end
    ' "${output}" "${config_path}" "${baseline_calibration}" "${source_path}"
}

if [[ -f "${output}" ]]; then
    validate_calibration || {
        echo "ERROR: existing calibration is invalid; preserving it: ${output}" >&2
        exit 2
    }
    echo "validated existing calibration: ${output}"
else
    srun --nodes=1 --ntasks=1 --cpus-per-task=64 \
        "${JULIA:-julia}" --project="${JULIA_PROJECT}" -t 64 "${source_path}" \
        --config "${config_path}" \
        --baseline-calibration "${baseline_calibration}" \
        --n-seeds 64 --n-random-jumps 300 --target-delta-tilde 0.01 \
        --output "${output}"
    validate_calibration
fi

config_copy=${output_root}/configs/cgs_survival_beta0_n8_q4.toml
config_tmp=${config_copy}.tmp.${SLURM_JOB_ID}
cp "${config_path}" "${config_tmp}"
mv "${config_tmp}" "${config_copy}"

smoke_tmp=${smoke_copy}.tmp.${SLURM_JOB_ID}
cp "${output}" "${smoke_tmp}"
mv "${smoke_tmp}" "${smoke_copy}"
