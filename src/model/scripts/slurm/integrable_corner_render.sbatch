#!/bin/bash -l
#SBATCH --job-name=corner_render
#SBATCH --hint=multithread
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

repo=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
scratch_root=${repo}/data/slurm
output_root=${CORNER_OUTPUT_ROOT:?set CORNER_OUTPUT_ROOT}
figure_dir=${CORNER_FIGURE_DIR:-${repo}/figures}
julia_bin=${JULIA:-julia}

mkdir -p "${scratch_root}/logs" "${scratch_root}/.cache" "${figure_dir}"
cd "${repo}"
export JULIA_PROJECT="${repo}/src/environment"
export XDG_CACHE_HOME="${scratch_root}/.cache"
export JULIA_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

"${julia_bin}" --project="${JULIA_PROJECT}" -t "${JULIA_NUM_THREADS}" \
  src/model/scripts/analysis/figures/integrable_corner_render.jl \
  --cache-dir "${output_root}" \
  --panel-a "${output_root}/panel_a_n10_f3_seed1.jld2" \
  --output-dir "${figure_dir}"
