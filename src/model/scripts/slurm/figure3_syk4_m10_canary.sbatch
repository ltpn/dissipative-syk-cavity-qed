#!/bin/bash -l
#SBATCH --job-name=fig3_m10_canary
#SBATCH --hint=multithread
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

[[ -n "${SLURM_JOB_ID:-}" ]] || {
    echo "ERROR: canary must run under SLURM" >&2
    exit 2
}

repo=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
scratch_root=${repo}/data/slurm
output_root=${repo}/data/syk4_reference/m10
config_path=${repo}/src/model/configs/figure3_syk4_dtilde0p01_m10_n10_f3.toml
canary_dir=${output_root}/canary/seed_001

cd "${repo}"
mkdir -p "${scratch_root}/logs" "${canary_dir}"
export JULIA_PROJECT="${repo}/src/environment"
export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=32
export OMP_NUM_THREADS=1

srun --nodes=1 --ntasks=1 --cpus-per-task=32 \
    "${JULIA:-julia}" --project="${JULIA_PROJECT}" \
    src/model/scripts/sweeps/spectrum_sweep.jl \
    --config "${config_path}" --shard-index 1 --shard-count 64 \
    --data-subdir "${canary_dir}"

"${JULIA:-julia}" --project="${JULIA_PROJECT}" -e '
using JLD2
spectra_dir = joinpath(ARGS[1], "spectra")
paths = filter(p -> endswith(p, ".jld2"), readdir(spectra_dir; join = true))
@assert length(paths) == 1
jldopen(only(paths), "r") do f
    metadata = f["metadata"]
    @assert metadata["n_spontaneous_emission_jumps"] == 10
    @assert metadata["n_cavity_loss_jumps"] == 1
    @assert metadata["n_total_jumps"] == 11
    @assert all(isfinite, f["L_svd_S_centered"])
    @assert all(isfinite, f["times"])
    @assert all(isfinite, f["populations_t"])
    @assert f["dynamics_trace_residual_max"] < 1e-8
    @assert f["dynamics_particle_residual_max"] < 1e-8
end
' "${canary_dir}"
