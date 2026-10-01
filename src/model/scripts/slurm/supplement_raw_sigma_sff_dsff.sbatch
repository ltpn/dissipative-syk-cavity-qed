#!/bin/bash -l
#SBATCH --job-name=raw_sff_supp
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

[[ -n "${SLURM_JOB_ID:-}" ]] || {
    echo "ERROR: supplementary rendering must run under SLURM" >&2
    exit 2
}

repo=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
scratch_root=${repo}/data/slurm
store_root=${repo}/data
figure_dir=${repo}/figures
figure1_manifest=${figure_dir}/figure_1__manifest.toml
figure3_manifest=${figure_dir}/figure_3__manifest.toml

fig1_spectra=${store_root}/physical/n10_f3/merged/spectra
fig1_control=${store_root}/syk4/n10_f3/merged/spectra
fig3_cache_root=${store_root}/multimode/n10_f3
fig3_control=${store_root}/syk4_reference/m10/norb_10_f3/merged/spectra

[[ -d "${figure_dir}" ]] || {
    echo "ERROR: figure directory missing: ${figure_dir}" >&2
    exit 2
}
for manifest in "${figure1_manifest}" "${figure3_manifest}"; do
    [[ -f "${manifest}" ]] || {
        echo "ERROR: source manifest missing: ${manifest}" >&2
        exit 2
    }
done

for eta in 0.1 0.4 1.0 2.0; do
    for seed in $(seq 1 64); do
        path=${fig1_spectra}/physical__eta=${eta}__gamma=1.0__tol=1.0e-10__seed=${seed}.jld2
        [[ -f "${path}" ]] || {
            echo "ERROR: Figure-1 spectrum missing: ${path}" >&2
            exit 2
        }
    done
done
for seed in $(seq 1 64); do
    path=${fig1_control}/synthetic_l3b_bdi_syk4__eta=2.0__gamma=1.0__tol=1.0e-10__seed=${seed}.jld2
    [[ -f "${path}" ]] || {
        echo "ERROR: Figure-1 control spectrum missing: ${path}" >&2
        exit 2
    }
done

for cache in \
    multimode_centered__norb=10__f=3__ngrid=160__M=300__deltacd=1__dtilde=0p01.jld2 \
    multimode_centered__norb=10__f=3__ngrid=160__M=300__deltacd=1__dtilde=0p1.jld2 \
    multimode_centered__norb=10__f=3__ngrid=160__M=300__deltacd=1__dtilde=1.jld2 \
    multimode_centered__norb=10__f=3__ngrid=160__M=300__deltacd=1__dtilde=10.jld2; do
    path=${fig3_cache_root}/${cache}
    [[ -f "${path}" ]] || {
        echo "ERROR: Figure-3 cache missing: ${path}" >&2
        exit 2
    }
done
for seed in $(seq 1 64); do
    path=${fig3_control}/synthetic_l3b_bdi_syk4_fig3_m10__eta=2.0__gamma=1.0__tol=1.0e-10__seed=${seed}.jld2
    [[ -f "${path}" ]] || {
        echo "ERROR: Figure-3 control spectrum missing: ${path}" >&2
        exit 2
    }
done

cd "${repo}"
mkdir -p "${scratch_root}/logs" "${scratch_root}/fontconfig"
export JULIA_PROJECT="${repo}/src/environment"
export XDG_CACHE_HOME="${scratch_root}/fontconfig"
export JULIA_NUM_THREADS=64
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

srun --nodes=1 --ntasks=1 --cpus-per-task=64 \
    "${JULIA:-julia}" --project="${JULIA_PROJECT}" -t 64 \
    src/model/scripts/analysis/figures/supplement_raw_sigma_sff_dsff.jl \
    --figure1-manifest "${figure_dir}/figure_1__manifest.toml" \
    --figure3-manifest "${figure_dir}/figure_3__manifest.toml" \
    --output-dir "${figure_dir}"
