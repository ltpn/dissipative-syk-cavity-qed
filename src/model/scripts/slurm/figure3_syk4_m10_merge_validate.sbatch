#!/bin/bash -l
#SBATCH --job-name=fig3_m10_val
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

[[ -n "${SLURM_JOB_ID:-}" ]] || {
    echo "ERROR: merge/validation must run under SLURM" >&2
    exit 2
}
: "${CALIBRATION_JOB_ID:?missing CALIBRATION_JOB_ID}"
: "${PRODUCTION_JOB_ID:?missing PRODUCTION_JOB_ID}"

repo=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
scratch_root=${repo}/data/slurm
output_root=${repo}/data/syk4_reference/m10
config_path=${repo}/src/model/configs/figure3_syk4_dtilde0p01_m10_n10_f3.toml
calibration_path=${output_root}/calibration/figure3_syk4_dtilde0p01_m10_n10_f3.jld2
spectra_dir=${output_root}/norb_10_f3/merged/spectra
validation_path=${output_root}/validation.toml

cd "${repo}"
mkdir -p "${scratch_root}/logs"
export JULIA_PROJECT="${repo}/src/environment"
export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

"${JULIA:-julia}" --project="${JULIA_PROJECT}" \
    src/model/scripts/sweeps/merge_spectrum_shards.jl \
    --config "${config_path}"

"${JULIA:-julia}" --project="${JULIA_PROJECT}" \
    src/model/scripts/analysis/figures/validate_figure3_syk4_reference.jl \
    --config "${config_path}" --spectra-dir "${spectra_dir}" \
    --calibration "${calibration_path}" --seed-range 1:64 \
    --output "${validation_path}" \
