#!/bin/bash -l
#SBATCH --job-name=corner_b
#SBATCH --hint=multithread
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err

set -euo pipefail

repo=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
scratch_root=${repo}/data/slurm
output_root=${CORNER_OUTPUT_ROOT:?set CORNER_OUTPUT_ROOT}
n_orb=${CORNER_N_ORB:?set CORNER_N_ORB}
filling=${CORNER_FILLING:?set CORNER_FILLING}
seed=${SLURM_ARRAY_TASK_ID:?run as an array}
julia_bin=${JULIA:-julia}

mkdir -p "${scratch_root}/logs" "${output_root}"
cd "${repo}"
export JULIA_PROJECT="${repo}/src/environment"
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

output_file=$(printf '%s/corner__norb=%d__filling=%d__seed=%d.jld2' \
  "${output_root}" "${n_orb}" "${filling}" "${seed}")

"${julia_bin}" --project="${JULIA_PROJECT}" -t "${JULIA_NUM_THREADS}" \
  src/model/scripts/analysis/figures/integrable_corner_compute.jl \
  --n-orb "${n_orb}" --filling "${filling}" --seed "${seed}" \
  --row-block 32 --time-block 16 --output "${output_file}"
