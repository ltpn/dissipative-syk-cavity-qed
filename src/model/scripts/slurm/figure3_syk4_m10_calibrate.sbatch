#!/bin/bash -l
#SBATCH --job-name=fig3_m10_cal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

[[ -n "${SLURM_JOB_ID:-}" ]] || {
    echo "ERROR: calibration must run under SLURM" >&2
    exit 2
}

repo=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
scratch_root=${repo}/data/slurm
store_root=${repo}/data
output_root=${store_root}/syk4_reference/m10
physical_cache=${store_root}/multimode/n10_f3/multimode_centered__norb=10__f=3__ngrid=160__M=300__deltacd=1__dtilde=0p01.jld2
baseline_calibration=${store_root}/syk4/n10_f3/calibration.jld2
baseline_config=${repo}/src/model/configs/csr_ladder_l3b_n10_f3.toml
# The config that generated the multimode ensemble behind physical_cache; the
# calibration rebuilds that ensemble's Hamiltonians from it.
multimode_config=${MULTIMODE_CONFIG:?set MULTIMODE_CONFIG to the N=10 multimode ensemble config}
output=${output_root}/calibration/figure3_syk4_dtilde0p01_m10_n10_f3.jld2

cd "${repo}"
mkdir -p "${scratch_root}/logs" "$(dirname "${output}")"
export JULIA_PROJECT="${repo}/src/environment"
export JULIA_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

echo "host=$(hostname) slurm_job_id=${SLURM_JOB_ID} output=${output}"
srun --nodes=1 --ntasks=1 --cpus-per-task=32 \
    "${JULIA:-julia}" --project="${JULIA_PROJECT}" -t 32 \
    src/model/scripts/analysis/figures/calibrate_figure3_syk4_reference.jl \
    --physical-cache "${physical_cache}" \
    --multimode-config "${multimode_config}" \
    --baseline-calibration "${baseline_calibration}" \
    --baseline-config "${baseline_config}" \
    --delta-tilde 0.01 --n-seeds 64 --n-random-jumps 10 \
    --output "${output}"
