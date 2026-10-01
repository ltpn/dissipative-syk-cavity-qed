#!/usr/bin/env julia
# The Figure 3 m10 calibration reads only per-seed r_HD (and shape metadata)
# from the multimode delta_tilde = 0.01 cache. r_HD is fixed at generation: the
# legacy export copies it verbatim from each generate artifact's r_HD_frobenius.
# This builds the same cache fields directly from the saved Liouvillians, so the
# reference ensemble can be calibrated and started while the multimode eigen,
# SVD and dynamics stages are still running.
#
#   julia --project=src/environment multimode_rhd_from_generation.jl \
#       --config MULTIMODE.toml --liouvillians DIR --delta-tilde 0.01 --output PATH
using JLD2, TOML
include(joinpath(@__DIR__, "..", "..", "..", "SpectralArtifacts.jl"))
using .SpectralArtifacts

function main(args = ARGS)
    opts = Dict{String,String}()
    i = 1
    while i <= length(args)
        key = args[i]; startswith(key, "--") && i < length(args) || error("expected --key value pairs")
        opts[key[3:end]] = args[i + 1]; i += 2
    end
    for key in ("config", "liouvillians", "delta-tilde", "output")
        haskey(opts, key) || error("missing --$key")
    end
    cfg = SpectralArtifacts.load_pipeline_config(abspath(opts["config"]))
    cfg["model"] == "multimode" || error("config is not a multimode ensemble")
    delta_tilde = parse(Float64, opts["delta-tilde"])
    dir = abspath(opts["liouvillians"])
    manifest = read_manifest(joinpath(dir, "manifest.jld2"))
    manifest["physics_id"] == SpectralArtifacts.physics_identity(cfg) ||
        error("Liouvillians in $dir were not generated from this config")
    points = sort(filter(p -> p["delta_tilde"] == delta_tilde, manifest["points"]); by = p -> p["seed"])
    seeds = [p["seed"] for p in points]
    n_seeds = Int(cfg["grid"]["seed_count"])
    seeds == collect(1:n_seeds) || error("manifest lacks seeds 1:$n_seeds at delta_tilde=$delta_tilde")
    expected = Dict("generation_id" => manifest["generation_id"])
    r_hd = [Float64(read_artifact(joinpath(dir, p["filename"]); stage = "generate",
                expected = merge(expected, Dict("point_id" => p["point_id"])),
                fields = ["r_HD_frobenius"])["r_HD_frobenius"]) for p in points]
    num = cfg["numerics"]
    output = abspath(opts["output"])
    mkpath(dirname(output))
    jldsave(output * ".tmp"; n_orb = Int(num["n_orb"]), filling = Int(num["filling"]),
        delta_tilde, n_seeds, seeds, r_HD = r_hd, source = "generation artifacts",
        liouvillians = dir, generation_id = manifest["generation_id"],
        physics_id = manifest["physics_id"])
    mv(output * ".tmp", output; force = false)
    println("wrote $output: $n_seeds seeds, r_HD median $(sort(r_hd)[cld(n_seeds, 2)])")
end
abspath(PROGRAM_FILE) == (@__FILE__) && main()
