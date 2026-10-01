#!/usr/bin/env bash
set -euo pipefail
root=${PILOT_ROOT:?}
repo=${REPO_ROOT:?}
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=64 OMP_NUM_THREADS=1
jl=("${JULIA:?}" --project="$repo/src/environment")
"${jl[@]}" -e '
    include(joinpath(ARGS[1],"src/model/SpectralPostprocessing.jl"))
    P = SpectralPostprocessing
    P.LinearAlgebra.BLAS.set_num_threads(64)
    root = ARGS[2]
    cfg = P.SpectralArtifacts.load_pipeline_config(joinpath(root,"config.toml"))
    selection = joinpath(root,"recovery","selection.toml")
    result = P.run_postprocessing(cfg,joinpath(root,"analysis");
        eigen_dir=joinpath(root,"eigen"),svd_dir=joinpath(root,"svd"),seeds=[1,2],
        replacements=isfile(selection) ? selection : nothing)
    result.failed == 0 || error("Postprocessing failed")
    write(joinpath(root,"analysis_directory.txt"),result.directory*"\n")
' "$repo" "$root"
analysis=$(cat "$root/analysis_directory.txt")
reference=${AI_REFERENCE:-$root/ai_dagger/reference.jld2}
if [[ ! -f $reference ]]; then
    "${jl[@]}" "$repo/src/model/scripts/analysis/figures/generate_ai_dagger_dsff_reference.jl" \
        --cache-dir "$root/ai_dagger/cache" --output "$reference"
fi
"${jl[@]}" "$repo/src/model/scripts/analysis/figures/physical_pilot_figures.jl" \
    "$root/config.toml" "$analysis" "$reference" "$root/figures"
"${jl[@]}" "$repo/src/model/scripts/analysis/figures/figure2_dynamics.jl" \
    --config "$root/config.toml" --spectra-dir "$analysis/spectra" \
    --seeds 1:2 --no-l2 --output-dir "$root/figures"
"${jl[@]}" "$repo/src/model/scripts/verify_physical_pilot.jl" "$root"
date -Is > "$root/PIPELINE_COMPLETED"
