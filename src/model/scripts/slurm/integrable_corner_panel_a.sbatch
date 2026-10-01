#!/bin/bash -l
#SBATCH --job-name=corner_a
#SBATCH --hint=multithread
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

repo=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
scratch_root=${repo}/data/slurm
output_root=${CORNER_OUTPUT_ROOT:?set CORNER_OUTPUT_ROOT}
julia_bin=${JULIA:-julia}

mkdir -p "${scratch_root}/logs" "${output_root}"
cd "${repo}"
export JULIA_PROJECT="${repo}/src/environment"
export JULIA_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=64
export OMP_NUM_THREADS=1

"${julia_bin}" --project="${JULIA_PROJECT}" -t "${JULIA_NUM_THREADS}" \
  src/model/scripts/analysis/figures/integrable_corner_validate_panel_a.jl \
  --seed 1 --output "${output_root}/panel_a_n10_f3_seed1.jld2"
