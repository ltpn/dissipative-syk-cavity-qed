#!/usr/bin/env bash

set -euo pipefail

repo=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}
export REPO_ROOT="${repo}"
cd "${repo}"
slurm_dir=${repo}/src/model/scripts/slurm

dependency=()
[[ -z "${MULTIMODE_MERGE_JOB_IDS:-}" ]] || dependency=(--dependency="afterok:${MULTIMODE_MERGE_JOB_IDS}")
calibration_job=$(sbatch --parsable "${dependency[@]}" "${slurm_dir}/figure3_syk4_m10_calibrate.sbatch")
calibration_job=${calibration_job%%;*}
canary_job=$(sbatch --parsable --dependency="afterok:${calibration_job}" \
    "${slurm_dir}/figure3_syk4_m10_canary.sbatch")
canary_job=${canary_job%%;*}
production_job=$(sbatch --parsable --dependency="afterok:${canary_job}" \
    "${slurm_dir}/figure3_syk4_m10_production.sbatch")
production_job=${production_job%%;*}
validation_job=$(sbatch --parsable --dependency="afterok:${production_job}" \
    "${slurm_dir}/figure3_syk4_m10_merge_validate.sbatch")
validation_job=${validation_job%%;*}
render_job=$(sbatch --parsable --dependency="afterok:${validation_job}" \
    "${slurm_dir}/figure3_syk4_m10_render.sbatch")
render_job=${render_job%%;*}

printf 'calibration_job_id=%s\ncanary_job_id=%s\nproduction_job_id=%s\nvalidation_job_id=%s\nrender_job_id=%s\n' \
    "${calibration_job}" "${canary_job}" "${production_job}" \
    "${validation_job}" "${render_job}"
