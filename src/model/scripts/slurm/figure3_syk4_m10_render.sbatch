#!/bin/bash -l
#SBATCH --job-name=fig3_m10_render
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

[[ -n "${SLURM_JOB_ID:-}" ]] || {
    echo "ERROR: render must run under SLURM" >&2
    exit 2
}

repo=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
scratch_root=${repo}/data/slurm
store_root=${repo}/data
output_root=${store_root}/syk4_reference/m10
config_path=${repo}/src/model/configs/figure3_syk4_dtilde0p01_m10_n10_f3.toml
spectra_dir=${output_root}/norb_10_f3/merged/spectra
validation_path=${output_root}/validation.toml
physical_cache_dir=${store_root}/multimode/n10_f3
figure_dir=${repo}/figures
inset_n6=${store_root}/multimode/n6_f3/multimode_centered__norb=6__f=3__ngrid=160__M=300__deltacd=1__dtilde=0p01.jld2
inset_n8=${store_root}/multimode/n8_f4/multimode_centered__norb=8__f=4__ngrid=160__M=300__deltacd=1__dtilde=0p01.jld2

cd "${repo}"
mkdir -p "${scratch_root}/logs" "${scratch_root}/fontconfig" "${figure_dir}"
export JULIA_PROJECT="${repo}/src/environment"
export XDG_CACHE_HOME="${scratch_root}/fontconfig"
export JULIA_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

[[ -f "${validation_path}" ]] || {
    echo "ERROR: validation payload missing: ${validation_path}" >&2
    exit 2
}

srun --nodes=1 --ntasks=1 --cpus-per-task=16 \
    "${JULIA:-julia}" --project="${JULIA_PROJECT}" -t 16 \
    src/model/scripts/analysis/figures/figure3_multimode_unfolded.jl \
    --delta-cd 1.0 --n-orb 10 --filling 3 --n-seeds 64 \
    --output-dir "${figure_dir}" --cache-dir "${physical_cache_dir}" \
    --inset-comparison-cache "6:${inset_n6}" \
    --inset-comparison-cache "8:${inset_n8}" \
    --syk4-reference-config "${config_path}" \
    --syk4-reference-spectra-dir "${spectra_dir}" \
    --syk4-reference-validation "${validation_path}" \
    --syk4-reference-eta 2.0 --syk4-reference-seeds 1:64
