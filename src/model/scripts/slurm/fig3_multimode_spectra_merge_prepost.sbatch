#!/bin/bash -l
#SBATCH --job-name=spectra_fig3_eigs_merge
#SBATCH --hint=multithread
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Merge and validate the four Figure 3 ensembles.

set -euo pipefail

repo=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
scratch_root=${repo}/data/slurm
cache_dir=${FIG3_CACHE_DIR:?set FIG3_CACHE_DIR}
n_orb=${FIG3_N_ORB:-10}
filling=${FIG3_FILLING:-3}
n_seeds=${FIG3_N_SEEDS:-64}
shard_count=${FIG3_SHARD_COUNT:-64}
# This production wrapper deliberately locks the four Figure 3 values.
# Do not pass a comma-separated list through `sbatch --export`: Slurm uses
# commas to delimit exported variables and would silently truncate the value.
delta_tildes=0.01,0.1,1.0,10.0
expected_dim=${FIG3_EXPECTED_DIM:-14400}

mkdir -p "${scratch_root}/logs"
cd "${repo}"
export JULIA_PROJECT="${repo}/src/environment"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

"${JULIA:-julia}" --project="${JULIA_PROJECT}" \
    src/model/scripts/analysis/figures/merge_fig3_multimode_shards.jl \
    --cache-dir "${cache_dir}" --n-orb "${n_orb}" --filling "${filling}" \
    --shard-count "${shard_count}" --n-seeds "${n_seeds}" \
    --delta-tildes "${delta_tildes}" --require-eigenvalues --delta-cd 1.0

"${JULIA:-julia}" --project="${JULIA_PROJECT}" -e '
using JLD2, Printf
cache_dir, n_orb_s, filling_s, n_seeds_s, expected_dim_s = ARGS[1:5]
n_orb = parse(Int, n_orb_s); filling = parse(Int, filling_s)
n_seeds = parse(Int, n_seeds_s); expected_dim = parse(Int, expected_dim_s)
filling_tag = "__f=$(filling)"
for dt in ARGS[6:end]
    slug = replace(@sprintf("%.6g", parse(Float64, dt)), "." => "p", "-" => "m")
    path = joinpath(cache_dir,
        "multimode_centered__norb=$(n_orb)$(filling_tag)__ngrid=160__M=300__deltacd=1__dtilde=$(slug).jld2")
    JLD2.jldopen(path, "r") do f
        seeds = Int.(f["seeds"])
        haskey(f, "L_eigvals") || error("missing L_eigvals: $path")
        spectra = f["L_eigvals"]
        seeds == collect(1:n_seeds) || error("seed mismatch: $path")
        length(spectra) == n_seeds || error("spectrum-count mismatch: $path")
        all(length(values) == expected_dim for values in spectra) ||
            error("eigenvalue dimension mismatch: $path")
        println("validated ", path, " seeds=", length(seeds), " K=", expected_dim)
    end
end
' "${cache_dir}" "${n_orb}" "${filling}" "${n_seeds}" "${expected_dim}" \
  $(printf '%s' "${delta_tildes}" | tr ',' ' ')
