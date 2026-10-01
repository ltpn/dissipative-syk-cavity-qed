#!/usr/bin/env bash
set -euo pipefail
repo=${REPO_ROOT:-$PWD}
root=${1:?absolute scratch run directory}
[[ $root == "${SCRATCH:?}/"* ]] || { echo 'Run directory must be in SCRATCH' >&2; exit 2; }
[[ ! -e $root ]] || { echo "Preserving existing $root; prepare requires a new directory" >&2; exit 2; }
mkdir -p "$root/source" "$root/logs" "$root/preflight/logs"
cp -a "$repo/julia" "$root/source/"
cp "$repo/src/model/configs/physical_n10_q4_r16.toml" "$root/config.toml"
git -C "$repo" rev-parse HEAD > "$root/source_revision.txt"
git -C "$repo" diff > "$root/source.patch"
find "$root/source" -type f -exec sha256sum {} + > "$root/source_sha256.txt"
"${JULIA:?}" --project="$repo/src/environment" -e '
    using TOML
    include(joinpath(ARGS[1],"src/model/SpectralPipeline.jl"))
    root = ARGS[2]
    cfg = SpectralPipeline.load_pipeline_config(joinpath(root,"config.toml"))
    points = SpectralPipeline.pipeline_manifest(cfg)["points"]
    open(joinpath(root,"pilot_shards.tsv"),"w") do io
        for eta in cfg["grid"]["etas"]
            indices = [only(findall(p->p["eta"]==eta && p["seed"]==seed,points)) for seed in (1,2)]
            println(io,"$eta $(indices[1]) $(indices[2]) $(length(points))")
        end
    end
    cfg["numerics"]["n_orb"] = 4
    cfg["numerics"]["filling"] = 2
    cfg["numerics"]["n_grid"] = 16
    cfg["numerics"]["time_grid"] = Dict("t_min"=>0.01,"t_max"=>1e6,"n"=>40)
    cfg["analysis"] = Dict("dsff"=>Dict("minimum_effective_count"=>1.0,"density_bins"=>256))
    open(joinpath(root,"preflight/config.toml"),"w") do io
        TOML.print(io,cfg;sorted=true)
    end
' "$repo" "$root"
cp "$root/pilot_shards.tsv" "$root/preflight/pilot_shards.tsv"
echo "$root"
