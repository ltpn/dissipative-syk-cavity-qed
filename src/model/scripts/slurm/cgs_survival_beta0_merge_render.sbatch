#!/bin/bash -l
#SBATCH --job-name=cgs_b0_merge
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

[[ -n "${SLURM_JOB_ID:-}" ]] || {
    echo "ERROR: merge/render must run under Slurm" >&2
    exit 2
}
: "${CGS_MODE:?CGS_MODE must be smoke or production}"

repo=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
config_path=${repo}/src/model/configs/cgs_survival_beta0_n8_q4.toml
production_root=${repo}/data/cgs_survival
project=${repo}/src/environment
mode=${CGS_MODE}

case "${mode}" in
    smoke)
        output_root=${production_root}/validation/smoke
        figure_dir=${output_root}/figures
        expected_seeds=1:2
        smoke_flag=--smoke
        ;;
    production)
        output_root=${production_root}
        figure_dir=${repo}/figures
        expected_seeds=1:64
        smoke_flag=
        ;;
    *)
        echo "ERROR: CGS_MODE must be smoke or production" >&2
        exit 2
        ;;
esac

mkdir -p "${repo}/data/slurm/cgs_survival" "${figure_dir}" \
    "${output_root}/manifests"
cd "${repo}"
export STORE="${repo}/data"
export JULIA_PROJECT=${project}
export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export XDG_CACHE_HOME=${repo}/data/slurm/.cache
mkdir -p "${XDG_CACHE_HOME}/fontconfig"

merge_driver=${repo}/src/model/scripts/analysis/figures/cgs_survival_merge_validate.jl
render_driver=${repo}/src/model/scripts/analysis/figures/cgs_survival_supplement.jl
plotdata=${output_root}/plotdata/cgs_survival_beta0_n8_q4.jld2
manifest=${output_root}/manifests/${mode}.toml

srun --nodes=1 --ntasks=1 --cpus-per-task=16 \
    "${JULIA:-julia}" --project="${JULIA_PROJECT}" "${merge_driver}" \
    --config "${config_path}" --output-root "${output_root}" \
    --mode "${mode}" --expected-seeds "${expected_seeds}"

"${JULIA:-julia}" --project="${JULIA_PROJECT}" "${render_driver}" \
    --plotdata "${plotdata}" --manifest "${manifest}" \
    --output-dir "${figure_dir}" ${smoke_flag}

figure_manifest=${figure_dir}/figure.toml
manifest_copy=${output_root}/manifests/figure.toml
manifest_tmp=${manifest_copy}.tmp.${SLURM_JOB_ID}
cp "${figure_manifest}" "${manifest_tmp}"
mv "${manifest_tmp}" "${manifest_copy}"
