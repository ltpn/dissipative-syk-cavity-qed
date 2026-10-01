#!/usr/bin/env bash

set -euo pipefail

repo=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}
export REPO_ROOT="${repo}"
cd "${repo}"
export STORE="${repo}/data"
slurm_dir=${repo}/src/model/scripts/slurm
config_path=${repo}/src/model/configs/cgs_survival_beta0_n8_q4.toml
output_root=${repo}/data/cgs_survival
calibration_path=${output_root}/calibration/figure3_diss_syk_n8_q4_m300.jld2
smoke_manifest=${output_root}/validation/smoke/manifests/smoke.toml
project=${repo}/src/environment
julia_bin=${JULIA:-julia}

usage() {
    echo "usage: $0 --smoke | --production-after-smoke SMOKE_VALIDATION_JOB_ID" >&2
    exit 2
}

job_id() {
    local raw=$1
    printf '%s' "${raw%%;*}"
}

require_completed_job() {
    local requested=$1
    [[ "${requested}" =~ ^[0-9]+$ ]] || {
        echo "ERROR: invalid Slurm job ID: ${requested}" >&2
        exit 2
    }
    local state
    state=$(sacct -n -X -j "${requested}" --format=State | awk 'NF { print $1; exit }')
    [[ "${state}" == "COMPLETED" ]] || {
        echo "ERROR: Slurm job ${requested} is not completed successfully (state=${state:-missing})" >&2
        exit 2
    }
}

validate_smoke_artifacts() {
    [[ -f "${smoke_manifest}" ]] || {
        echo "ERROR: promoted smoke manifest is missing: ${smoke_manifest}" >&2
        exit 2
    }
    [[ -f "${calibration_path}" ]] || {
        echo "ERROR: calibration is missing: ${calibration_path}" >&2
        exit 2
    }
    "${julia_bin}" --project="${project}" -e '
        include(ARGS[1])
        cgs_validate_manifest(ARGS[2])
        using JLD2
        jldopen(ARGS[3], "r") do f
            @assert f["n_orb"] == 8 && f["filling"] == 4
            @assert f["n_seeds"] == 64 && f["n_random_jumps"] == 300
            @assert f["n_cavity_jumps"] == 1 && f["target_delta_tilde"] == 0.01
        end
    ' "${repo}/src/model/scripts/analysis/figures/cgs_survival_merge_validate.jl" \
      "${smoke_manifest}" "${calibration_path}"
}

mkdir -p "${output_root}/manifests" "${repo}/data/slurm/cgs_survival"
cd "${repo}"
export STORE="${repo}/data"

case "${1:-}" in
    --smoke)
        [[ $# == 1 ]] || usage
        calibration_job=$(job_id "$(sbatch --parsable \
            "${slurm_dir}/cgs_survival_beta0_calibrate.sbatch")")
        smoke_job=$(job_id "$(sbatch --parsable \
            --dependency="afterok:${calibration_job}" \
            "${slurm_dir}/cgs_survival_beta0_smoke.sbatch")")
        smoke_validation_job=$(job_id "$(sbatch --parsable \
            --dependency="afterok:${smoke_job}" \
            --export="ALL,CGS_MODE=smoke" \
            "${slurm_dir}/cgs_survival_beta0_merge_render.sbatch")")
        printf 'calibration_job_id=%s\nsmoke_job_id=%s\nsmoke_validation_job_id=%s\n' \
            "${calibration_job}" "${smoke_job}" "${smoke_validation_job}"
        ;;
    --production-after-smoke)
        [[ $# == 2 ]] || usage
        smoke_validation_job=$2
        require_completed_job "${smoke_validation_job}"
        validate_smoke_artifacts
        calibration_job=$(job_id "$(sbatch --parsable \
            "${slurm_dir}/cgs_survival_beta0_calibrate.sbatch")")
        production_job=$(job_id "$(sbatch --parsable \
            --dependency="afterok:${calibration_job}" \
            "${slurm_dir}/cgs_survival_beta0_production.sbatch")")
        merge_render_job=$(job_id "$(sbatch --parsable \
            --dependency="afterok:${production_job}" \
            --export="ALL,CGS_MODE=production" \
            "${slurm_dir}/cgs_survival_beta0_merge_render.sbatch")")
        printf 'calibration_job_id=%s\nproduction_job_id=%s\nmerge_render_job_id=%s\n' \
            "${calibration_job}" "${production_job}" "${merge_render_job}"
        ;;
    *)
        usage
        ;;
esac
