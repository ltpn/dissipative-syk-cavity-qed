#!/usr/bin/env julia
# merge_fig3_multimode_shards.jl
#
# Combine per-shard multimode cache files produced by
# `figure3_multimode_unfolded.jl --build-cache-only --shard-index N --shard-count M`
# into the canonical `multimode_centered__...__dtilde=X.jld2` cache that the
# plotting driver expects.
#
# Usage:
#   julia --project=src/environment \
#       src/model/scripts/analysis/figures/merge_fig3_multimode_shards.jl \
#       --cache-dir <dir> --n-orb 8 --shard-count 32 --n-seeds 128

if Sys.isapple()
    @eval using AppleAccelerate
end

using JLD2
using Printf
using LinearAlgebra

const HERE = @__DIR__
const LAMB_DIR = abspath(joinpath(HERE, "..", "..", ".."))

# Load only the definitions we need from the main driver.  The driver's own
# `main` entry point is guarded by `if abspath(PROGRAM_FILE) == @__FILE__`,
# so `include` only defines helpers.
include(joinpath(LAMB_DIR, "..", "SYK_setup.jl"))
include(joinpath(LAMB_DIR, "LambDickeModel.jl"))
include(joinpath(LAMB_DIR, "HarmonicOscillatorGrid.jl"))
include(joinpath(LAMB_DIR, "Speckle.jl"))
include(joinpath(LAMB_DIR, "OverlapIntegrals.jl"))
include(joinpath(LAMB_DIR, "JumpFactorization.jl"))
include(joinpath(LAMB_DIR, "PhysicalSYK.jl"))
include(joinpath(LAMB_DIR, "LiouvillianIntegration.jl"))
include(joinpath(LAMB_DIR, "Dynamics.jl"))
include(joinpath(LAMB_DIR, "MultimodeHamiltonian.jl"))
include(joinpath(LAMB_DIR, "scripts", "analysis", "figures",
                 "prl_style.jl"))
include(joinpath(LAMB_DIR, "scripts", "analysis", "figures",
                 "_common_sff_helpers.jl"))
include(joinpath(LAMB_DIR, "scripts", "analysis", "figures",
                 "figure3_multimode_unfolded.jl"))

function _parse_merge_cli(argv)
    cache_dir   = ""
    n_orb       = 6
    filling::Union{Nothing,Int} = nothing
    shard_count = 1
    n_seeds     = 128
    require_eigenvalues = false
    delta_tildes = collect(Float64, DEFAULT_DELTA_TILDES_PLOT)
    # Must match the `--delta-cd` used for the shard build: it is part of every
    # shard and merged cache filename.
    delta_cd = DELTA_CD_OVER_2PI_MHZ
    i = 1
    while i <= length(argv)
        a = argv[i]
        if a == "--cache-dir"; i += 1; cache_dir = argv[i]
        elseif startswith(a, "--cache-dir="); cache_dir = split(a, "=", limit=2)[2]
        elseif a == "--n-orb"; i += 1; n_orb = parse(Int, argv[i])
        elseif startswith(a, "--n-orb="); n_orb = parse(Int, split(a, "=", limit=2)[2])
        elseif a == "--filling"; i += 1; filling = parse(Int, argv[i])
        elseif startswith(a, "--filling="); filling = parse(Int, split(a, "=", limit=2)[2])
        elseif a == "--shard-count"; i += 1; shard_count = parse(Int, argv[i])
        elseif startswith(a, "--shard-count="); shard_count = parse(Int, split(a, "=", limit=2)[2])
        elseif a == "--n-seeds"; i += 1; n_seeds = parse(Int, argv[i])
        elseif startswith(a, "--n-seeds="); n_seeds = parse(Int, split(a, "=", limit=2)[2])
        elseif a == "--delta-tildes"; i += 1; delta_tildes = _parse_delta_tildes(argv[i])
        elseif startswith(a, "--delta-tildes="); delta_tildes = _parse_delta_tildes(split(a, "=", limit=2)[2])
        elseif a == "--delta-cd"; i += 1; delta_cd = parse(Float64, argv[i])
        elseif startswith(a, "--delta-cd="); delta_cd = parse(Float64, split(a, "=", limit=2)[2])
        elseif a == "--require-eigenvalues"; require_eigenvalues = true
        else error("unrecognized argument: $a")
        end
        i += 1
    end
    isempty(cache_dir) && error("--cache-dir required")
    isdir(cache_dir)  || error("cache-dir not found: $cache_dir")
    shard_count >= 1  || error("--shard-count must be >= 1")
    n_seeds     >= 1  || error("--n-seeds must be >= 1")
    (isfinite(delta_cd) && delta_cd > 0.0) ||
        error("--delta-cd must be positive and finite; got $delta_cd")
    return (cache_dir = abspath(cache_dir),
            n_orb = n_orb,
            filling = filling,
            shard_count = shard_count,
            n_seeds = n_seeds,
            require_eigenvalues = require_eigenvalues,
            delta_tildes = delta_tildes,
            delta_cd = delta_cd)
end

function merge_main(argv = ARGS)
    cli = _parse_merge_cli(argv)
    global N_ORB     = Int(cli.n_orb)
    global FILLING   = cli.filling === nothing ? nothing : Int(cli.filling)
    filling_effective = FILLING === nothing ? div(N_ORB, 2) : FILLING
    global D_HILBERT = binomial(N_ORB, filling_effective)
    global K_LIOUV   = D_HILBERT^2
    global DELTA_TILDES_PLOT = tuple(cli.delta_tildes...)
    global DELTA_CD_OVER_2PI_MHZ = Float64(cli.delta_cd)
    println("=" ^ 78)
    println("Merging multimode per-shard caches")
    println("cache-dir:    $(cli.cache_dir)")
    println("N_orb:        $(cli.n_orb)")
    println("filling:      $filling_effective")
    println("shard-count:  $(cli.shard_count)")
    println("n-seeds:      $(cli.n_seeds)")
    println("Delta_cd/2pi: $(cli.delta_cd) MHz")
    println("require eigs: $(cli.require_eigenvalues)")
    println("delta_tildes: $(cli.delta_tildes)")
    println("=" ^ 78)
    n_ok = 0
    n_missing = 0
    for dt in cli.delta_tildes
        @printf("\n--- delta_tilde = %g ---\n", dt)
        res = merge_shard_caches(cli.cache_dir, dt, cli.shard_count, cli.n_seeds;
                                 require_L_eigvals = cli.require_eigenvalues)
        @printf("  merged %d seeds -> %s\n", res.n_seeds, res.path)
        if res.n_seeds == cli.n_seeds
            n_ok += 1
        else
            n_missing += 1
        end
    end
    println("\nMerge summary: complete=$n_ok  short=$n_missing (out of $(length(cli.delta_tildes)))")
    return n_missing == 0 ? 0 : 1
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(merge_main(ARGS))
end
