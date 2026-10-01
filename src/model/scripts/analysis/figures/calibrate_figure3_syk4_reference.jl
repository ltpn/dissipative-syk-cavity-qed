#!/usr/bin/env julia

# Usage:
#   julia calibrate_figure3_syk4_reference.jl \
#     --physical-cache PATH --multimode-config PATH \
#     --baseline-calibration PATH --baseline-config PATH --delta-tilde 0.01 \
#     --n-random-jumps 10 [--n-seeds N] \
#     --output PATH
#
# --n-seeds defaults to the baseline config's seed_count and must equal it.

using JLD2
using LinearAlgebra
using Random: MersenneTwister
using SparseArrays: spzeros
using Statistics: mean, median
using TOML

const HERE = @__DIR__
const LAMB_DIR = abspath(joinpath(HERE, "..", "..", ".."))
const JULIA_DIR = abspath(joinpath(LAMB_DIR, ".."))

include(joinpath(JULIA_DIR, "SYK_setup.jl"))
include(joinpath(LAMB_DIR, "LambDickeModel.jl"))
include(joinpath(LAMB_DIR, "HarmonicOscillatorGrid.jl"))
include(joinpath(LAMB_DIR, "Speckle.jl"))
include(joinpath(LAMB_DIR, "OverlapIntegrals.jl"))
include(joinpath(LAMB_DIR, "JumpFactorization.jl"))
include(joinpath(LAMB_DIR, "PhysicalSYK.jl"))
include(joinpath(LAMB_DIR, "LiouvillianIntegration.jl"))
include(joinpath(LAMB_DIR, "SyntheticLadder.jl"))
include(joinpath(LAMB_DIR, "MultimodeHamiltonian.jl"))
include(joinpath(LAMB_DIR, "Figure3SYK4Calibration.jl"))
include(joinpath(LAMB_DIR, "SpectralArtifacts.jl"))

using .Figure3SYK4Calibration: coherent_superoperator_norm,
    solve_component_scales

# The m300 reference is defined at N = 10 and delta_tilde = 0.01; filling and
# ensemble size follow the SYK4 baseline config, which must agree with the
# multimode cache and the baseline calibration.
const LOCKED_N_ORB = 10
const LOCKED_DELTA_TILDE = 0.01
const LOCKED_N_CAVITY_JUMPS = 1
const LOCKED_CAVITY_RATE = 0.2

function _parse_cli(argv)
    opts = Dict{String,String}(
        "delta-tilde" => string(LOCKED_DELTA_TILDE),
        "n-seeds" => "",
        "n-random-jumps" => "300",
    )
    i = 1
    while i <= length(argv)
        arg = argv[i]
        startswith(arg, "--") || error("unexpected positional argument: $arg")
        if occursin('=', arg)
            key, value = split(arg[3:end], '='; limit = 2)
            opts[key] = value
        else
            key = arg[3:end]
            i += 1
            i <= length(argv) || error("--$key requires a value")
            opts[key] = argv[i]
        end
        i += 1
    end
    required = ("physical-cache", "multimode-config", "baseline-calibration",
                "baseline-config", "output")
    for key in required
        haskey(opts, key) || error("missing required option --$key")
        isempty(opts[key]) && error("--$key must not be empty")
    end
    return opts
end

function _read_jld2(path::AbstractString)
    isfile(path) || error("JLD2 input does not exist: $path")
    return JLD2.jldopen(path, "r") do file
        Dict{String,Any}(String(key) => file[key] for key in keys(file))
    end
end

"""
Multimode parameters of the physical ensemble, built from its own pipeline
config exactly as `SpectralPipeline.build_point` does, so the rebuilt
Hamiltonians are the ones the ensemble used. MultimodeParams' constructor
defaults differ from the configs, so parameters must never be restated here.
"""
function _physical_params(cfg::AbstractDict, delta_tilde::Real)
    num = cfg["numerics"]
    params = Dict{Symbol,Any}(Symbol(k) => v for (k, v) in num if k != "time_grid")
    params[:weight_type] = Symbol(get(params, :weight_type, "speckle"))
    params[:delta_tilde] = Float64(delta_tilde)
    return MultimodeParams(; params...)
end

"""
Tie the r_HD cache to the multimode config: same model, N, filling and
delta_tilde, and the same physics identity when the cache records one.
"""
function _validate_multimode_config(cfg, cache, delta_tilde)
    cfg["model"] == "multimode" || error("--multimode-config is not a multimode ensemble")
    num = cfg["numerics"]
    Int(num["n_orb"]) == LOCKED_N_ORB || error("multimode config n_orb mismatch")
    Int(num["filling"]) == Int(cache["filling"]) || error("multimode config/cache filling mismatch")
    delta_tilde in Float64.(cfg["grid"]["delta_tildes"]) ||
        error("multimode config has no delta_tilde=$delta_tilde")
    Int(cfg["grid"]["seed_count"]) == Int(cache["n_seeds"]) ||
        error("multimode config/cache seed count mismatch")
    for key in ("n_grid", "mode_cutoff")
        haskey(cache, key) && Int(cache[key]) != Int(num[key]) &&
            error("multimode config/cache $key mismatch")
    end
    haskey(cache, "delta_cd_over_2pi_mhz") &&
        Float64(cache["delta_cd_over_2pi_mhz"]) != Float64(cfg["loss"]["delta_cd_over_2pi_mhz"]) &&
        error("multimode config/cache delta_cd mismatch")
    physics_id = SpectralArtifacts.physics_identity(cfg)
    haskey(cache, "physics_id") && cache["physics_id"] != physics_id &&
        error("r_HD cache was not produced from this multimode config")
    return physics_id
end

function _physical_targets(params::MultimodeParams, r_hd::Vector{Float64})
    n_seeds = length(r_hd)
    spans = Vector{Float64}(undef, n_seeds)
    lh_norms = Vector{Float64}(undef, n_seeds)
    Threads.@threads for seed in 1:n_seeds
        block = multimode_hamiltonian_block(params; seed = seed)
        H = Matrix{ComplexF64}(block.H)
        evals = eigvals(Hermitian(0.5 .* (H .+ H')))
        spans[seed] = Float64(maximum(evals) - minimum(evals))
        lh_norms[seed] = coherent_superoperator_norm(H)
    end
    ld_norms = lh_norms ./ r_hd
    return spans, lh_norms, ld_norms
end

function _baseline_targets(baseline::AbstractDict, config::AbstractDict,
                           n_seeds::Integer, n_random_jumps::Integer)
    n_orb = Int(baseline["n_orb"])
    filling = Int(baseline["filling"])
    seeds = config["seeds"]
    offset = Int(seeds["l3b_offset"])
    lambda_star = Float64(baseline["lambda_star_60"])
    sigma_g = Float64(baseline["sigma_g"])
    basis = generate_vectors(n_orb, filling)
    d = length(basis)

    unit_spans = Vector{Float64}(undef, n_seeds)
    baseline_ld_norms = Vector{Float64}(undef, n_seeds)
    Threads.@threads for seed in 1:n_seeds
        rng_h = MersenneTwister(offset + seed)
        tensor = random_real_syk4_tensor(rng_h, n_orb, 1.0)
        H = Matrix{ComplexF64}(syk4_block_from_tensor(n_orb, basis, tensor))
        evals = eigvals(Hermitian(0.5 .* (H .+ H')))
        unit_spans[seed] = Float64(maximum(evals) - minimum(evals))

        rng_jump = MersenneTwister(offset + seed + 1_000_000)
        rng_cavity = MersenneTwister(offset + seed + 2_000_000)
        one_body_jumps = random_symmetric_jumps_equal_weight(
            rng_jump, n_orb, n_random_jumps, sqrt(lambda_star))
        cavity_op = random_symmetric_cavity_op(rng_cavity, n_orb, sigma_g)
        jumps = many_body_jump_operators_from_matrices(
            n_orb, basis, one_body_jumps; gamma_eff = 1.0)
        append!(jumps, many_body_jump_operators_from_matrices(
            n_orb, basis, [cavity_op]; gamma_eff = LOCKED_CAVITY_RATE))
        LD = assemble_lindblad_liouvillian(
            spzeros(ComplexF64, d, d), jumps)
        baseline_ld_norms[seed] = norm(LD)
    end
    return unit_spans, baseline_ld_norms
end

function _validate_inputs(cache, baseline, config, delta_tilde, n_seeds)
    delta_tilde == LOCKED_DELTA_TILDE ||
        error("Figure 3 calibration requires delta_tilde=$(LOCKED_DELTA_TILDE)")
    n_seeds == Int(config["grid"]["seed_count"]) ||
        error("Figure 3 calibration requires n_seeds equal to the baseline config seed_count")
    Int(cache["n_orb"]) == LOCKED_N_ORB || error("physical cache n_orb mismatch")
    Float64(cache["delta_tilde"]) == delta_tilde ||
        error("physical cache delta_tilde mismatch")
    Int(cache["n_seeds"]) == n_seeds || error("physical cache seed count mismatch")
    Int.(cache["seeds"]) == collect(1:n_seeds) ||
        error("physical cache must contain ordered seeds 1:$n_seeds")
    Int(baseline["n_orb"]) == LOCKED_N_ORB || error("baseline n_orb mismatch")
    filling = Int(config["numerics"]["filling"])
    Int(baseline["filling"]) == filling || error("baseline filling mismatch")
    Int(cache["filling"]) == filling || error("physical cache filling mismatch")
    Int(config["numerics"]["n_orb"]) == LOCKED_N_ORB ||
        error("baseline config n_orb mismatch")
    String.(config["grid"]["configs"]) == ["synthetic_l3b_bdi_syk4"] ||
        error("baseline config must be the Figure 1/2 L3b model")
    r_hd = Float64.(cache["r_HD"])
    length(r_hd) == n_seeds || error("physical r_HD length mismatch")
    all(isfinite, r_hd) && all(>(0), r_hd) ||
        error("physical r_HD values must be finite and positive")
    return r_hd, filling
end

function main(argv = ARGS)
    opts = _parse_cli(argv)
    physical_cache = abspath(opts["physical-cache"])
    multimode_config = abspath(opts["multimode-config"])
    baseline_calibration = abspath(opts["baseline-calibration"])
    baseline_config = abspath(opts["baseline-config"])
    output = abspath(opts["output"])
    delta_tilde = parse(Float64, opts["delta-tilde"])
    n_random_jumps = parse(Int, opts["n-random-jumps"])
    n_random_jumps in (10, 300) || error("Figure 3 calibration requires 10 or 300 random jumps")

    cache = _read_jld2(physical_cache)
    baseline = _read_jld2(baseline_calibration)
    isfile(baseline_config) || error("baseline config does not exist: $baseline_config")
    config = TOML.parsefile(baseline_config)
    n_seeds = isempty(opts["n-seeds"]) ? Int(config["grid"]["seed_count"]) : parse(Int, opts["n-seeds"])
    r_hd, filling = _validate_inputs(cache, baseline, config, delta_tilde, n_seeds)
    multimode_cfg = SpectralArtifacts.load_pipeline_config(multimode_config)
    multimode_physics_id = _validate_multimode_config(multimode_cfg, cache, delta_tilde)

    physical_spans, physical_lh_norms, physical_ld_norms =
        _physical_targets(_physical_params(multimode_cfg, delta_tilde), r_hd)
    unit_spans, baseline_ld_norms =
        _baseline_targets(baseline, config, n_seeds, n_random_jumps)
    scales = solve_component_scales(
        physical_spans, physical_ld_norms, unit_spans, baseline_ld_norms)

    payload = Dict{String,Any}(
        "target_delta_tilde" => delta_tilde,
        "physical_cache" => physical_cache,
        "multimode_config" => multimode_config,
        "multimode_physics_id" => multimode_physics_id,
        "physical_seed_range" => [1, n_seeds],
        "n_orb" => LOCKED_N_ORB,
        "filling" => filling,
        "n_seeds" => n_seeds,
        "n_random_jumps" => n_random_jumps,
        "n_cavity_jumps" => LOCKED_N_CAVITY_JUMPS,
        "sigma_syk4_bdi_span" => scales.sigma_syk4,
        "figure3_dissipator_rate_scale" => scales.dissipator_rate_scale,
        "lambda_star_60" => Float64(baseline["lambda_star_60"]),
        "sigma_g" => Float64(baseline["sigma_g"]),
        "physical_H_span_mean" => mean(physical_spans),
        "physical_LH_norm_mean" => mean(physical_lh_norms),
        "physical_LD_norm_mean" => mean(physical_ld_norms),
        "unit_syk_H_span_mean" => mean(unit_spans),
        "baseline_LD_norm_mean" => mean(baseline_ld_norms),
        "physical_r_HD_median" => median(r_hd),
        "physical_H_spans" => physical_spans,
        "physical_LH_norms" => physical_lh_norms,
        "physical_LD_norms" => physical_ld_norms,
        "physical_r_HD" => r_hd,
        "unit_syk_H_spans" => unit_spans,
        "baseline_LD_norms" => baseline_ld_norms,
        "baseline_calibration" => baseline_calibration,
        "baseline_config" => baseline_config,
    )

    mkpath(dirname(output))
    tmp = output * ".tmp"
    paths = SpectralArtifacts.DatasetPaths
    payload = paths.relative_metadata(payload)
    payload["path_root"] = relpath(paths.dataset_root(),dirname(output))
    JLD2.jldopen(tmp, "w") do file
        for (key, value) in payload
            file[key] = value
        end
    end
    mv(tmp, output; force = true)
    println("wrote Figure 3 SYK4 calibration: $output")
    println("  n_random_jumps=$n_random_jumps")
    println("  sigma_syk4_bdi_span=$(scales.sigma_syk4)")
    println("  dissipator_rate_scale=$(scales.dissipator_rate_scale)")
    return output
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
