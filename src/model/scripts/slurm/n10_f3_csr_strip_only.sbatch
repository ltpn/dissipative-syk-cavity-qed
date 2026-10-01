#!/bin/bash -l
#SBATCH --job-name=spectra_csr_strip
#SBATCH --hint=multithread
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Render the nine-panel complex-spacing-ratio strip.

set -euo pipefail

repo=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
scratch_root=${repo}/data/slurm
mkdir -p "${scratch_root}/logs"
cd "${repo}"

export JULIA_PROJECT="${repo}/src/environment"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export JULIA_NUM_THREADS=32

PRODUCTION_SPECTRA=${repo}/data/physical/n10_f3/merged/spectra
CONTROL_SPECTRA=${repo}/data/syk4/n10_f3/merged/spectra
FIGDIR=${repo}/figures

echo "host=$(hostname)  slurm_job_id=${SLURM_JOB_ID}"
echo "=== CSR strip (9 panels: 4 physical + 4 multimode + SYK) ==="
time "${JULIA:-julia}" --project="${JULIA_PROJECT}" \
    src/model/scripts/analysis/csr_synthetic_ladder/plot_csr_strip.jl \
    --n-orb 10 --filling 3 --n-seeds 64 \
    --physical-dir "${PRODUCTION_SPECTRA}" \
    --control-dir "${CONTROL_SPECTRA}" \
    --multimode-cache-dir "${repo}/data/multimode/n10_f3" \
    --output-dir "${FIGDIR}"

echo "== done =="
ls -lh "${FIGDIR}"/spectral_statistics__csr_strip* 2>&1 || true
