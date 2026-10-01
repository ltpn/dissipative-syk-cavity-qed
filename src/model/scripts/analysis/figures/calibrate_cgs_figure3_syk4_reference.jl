#!/usr/bin/env julia

using JLD2
using LinearAlgebra
using Random: MersenneTwister
using SparseArrays: spzeros
using Statistics: mean

const CGS_CAL_HERE = @__DIR__
const CGS_CAL_LAMB_DIR = abspath(joinpath(CGS_CAL_HERE, "..", "..", ".."))
const CGS_CAL_JULIA_DIR = abspath(joinpath(CGS_CAL_LAMB_DIR, ".."))
const CGS_CAL_REPO_ROOT = abspath(joinpath(CGS_CAL_JULIA_DIR, ".."))

include(joinpath(CGS_CAL_JULIA_DIR, "SYK_setup.jl"))
include(joinpath(CGS_CAL_LAMB_DIR, "LambDickeModel.jl"))
include(joinpath(CGS_CAL_LAMB_DIR, "HarmonicOscillatorGrid.jl"))
include(joinpath(CGS_CAL_LAMB_DIR, "Speckle.jl"))
include(joinpath(CGS_CAL_LAMB_DIR, "OverlapIntegrals.jl"))
include(joinpath(CGS_CAL_LAMB_DIR, "JumpFactorization.jl"))
include(joinpath(CGS_CAL_LAMB_DIR, "PhysicalSYK.jl"))
include(joinpath(CGS_CAL_LAMB_DIR, "LiouvillianIntegration.jl"))
include(joinpath(CGS_CAL_LAMB_DIR, "SyntheticLadder.jl"))
include(joinpath(CGS_CAL_LAMB_DIR, "MultimodeHamiltonian.jl"))
include(joinpath(CGS_CAL_LAMB_DIR, "Figure3SYK4Calibration.jl"))
include(joinpath(CGS_CAL_LAMB_DIR, "CGSModelBuilders.jl"))

using .Figure3SYK4Calibration: coherent_superoperator_norm,
    solve_component_scales, validate_component_metrics

function cgs_parse_calibration_cli(argv)
    options = Dict{String,String}(
        "n-seeds" => "64",
        "n-random-jumps" => "300",
        "target-delta-tilde" => "0.01",
    )
    index = 1
    while index <= length(argv)
        argument = argv[index]
        startswith(argument, "--") || error("unexpected positional argument: $argument")
        if occursin('=', argument)
            key, value = split(argument[3:end], '='; limit = 2)
            options[key] = value
        else
            key = argument[3:end]
            index += 1
            index <= length(argv) || error("--$key requires a value")
            options[key] = argv[index]
        end
        index += 1
    end
    for key in ("config", "baseline-calibration", "output")
        haskey(options, key) || error("missing required option --$key")
        isempty(options[key]) && error("--$key must not be empty")
    end
    return options
end

function cgs_calibration_payload(physical_spans,
                                 physical_lh_norms,
                                 physical_ld_norms,
                                 unit_syk_spans,
                                 baseline_ld_norms;
                                 n_orb::Integer,
                                 filling::Integer,
                                 target_delta_tilde::Real,
                                 n_random_jumps::Integer,
                                 n_cavity_jumps::Integer,
                                 delta0_over_2pi_mhz::Real,
                                 kappa_over_2pi_mhz::Real,
                                 baseline::AbstractDict)
    collections = (physical_spans, physical_lh_norms, physical_ld_norms,
                   unit_syk_spans, baseline_ld_norms)
    sample_count = length(physical_spans)
    sample_count > 0 || throw(ArgumentError("calibration samples must be nonempty"))
    all(values -> length(values) == sample_count, collections) ||
        throw(DimensionMismatch("calibration sample arrays must have equal lengths"))
    all(values -> all(isfinite, values) && all(>(0), values), collections) ||
        throw(ArgumentError("calibration samples must be finite and positive"))
    for key in ("lambda_star_60", "sigma_g")
        haskey(baseline, key) || throw(ArgumentError("baseline lacks $key"))
    end
    delta0 = Float64(delta0_over_2pi_mhz)
    kappa = Float64(kappa_over_2pi_mhz)
    isfinite(delta0) && delta0 > 0 || throw(ArgumentError(
        "Delta_0/2pi must be finite and positive"))
    isfinite(kappa) && kappa >= 0 || throw(ArgumentError(
        "kappa/2pi must be finite and nonnegative"))

    scales = solve_component_scales(
        physical_spans, physical_ld_norms, unit_syk_spans, baseline_ld_norms)
    synthetic_span_mean = scales.sigma_syk4 * mean(unit_syk_spans)
    synthetic_ld_mean = scales.dissipator_rate_scale * mean(baseline_ld_norms)
    payload = Dict{String,Any}(
        "n_orb" => Int(n_orb),
        "filling" => Int(filling),
        "target_delta_tilde" => Float64(target_delta_tilde),
        "n_seeds" => sample_count,
        "n_random_jumps" => Int(n_random_jumps),
        "n_cavity_jumps" => Int(n_cavity_jumps),
        "delta_cd_over_2pi_mhz" => delta0,
        "delta_0_over_2pi_mhz" => delta0,
        "kappa_over_2pi_mhz" => kappa,
        "physical_cavity_loss_rate_over_energy" => kappa / delta0,
        "sigma_syk4_bdi_span" => scales.sigma_syk4,
        "figure3_dissipator_rate_scale" => scales.dissipator_rate_scale,
        "lambda_star_60" => Float64(baseline["lambda_star_60"]),
        "sigma_g" => Float64(baseline["sigma_g"]),
        "physical_H_spans" => Float64.(physical_spans),
        "physical_LH_norms" => Float64.(physical_lh_norms),
        "physical_LD_norms" => Float64.(physical_ld_norms),
        "physical_r_HD" => Float64.(physical_lh_norms ./ physical_ld_norms),
        "unit_syk_H_spans" => Float64.(unit_syk_spans),
        "baseline_LD_norms" => Float64.(baseline_ld_norms),
        "physical_H_span_mean" => mean(physical_spans),
        "physical_LH_norm_mean" => mean(physical_lh_norms),
        "physical_LD_norm_mean" => mean(physical_ld_norms),
        "unit_syk_H_span_mean" => mean(unit_syk_spans),
        "baseline_LD_norm_mean" => mean(baseline_ld_norms),
        "synthetic_H_span_mean" => synthetic_span_mean,
        "synthetic_LD_norm_mean" => synthetic_ld_mean,
    )
    return payload
end

function cgs_write_calibration(path::AbstractString, payload::AbstractDict)
    output = abspath(path)
    mkpath(dirname(output))
    temporary = output * ".tmp.$(getpid())"
    try
        JLD2.jldopen(temporary, "w") do file
            for (key, value) in payload
                file[String(key)] = value
            end
        end
        JLD2.jldopen(temporary, "r") do file
            file["n_random_jumps"] == payload["n_random_jumps"] ||
                error("temporary calibration failed identity validation")
        end
        mv(temporary, output; force = true)
    finally
        isfile(temporary) && rm(temporary; force = true)
    end
    return output
end

function _cgs_read_calibration_input(path::AbstractString)
    input = abspath(path)
    isfile(input) || error("calibration input does not exist: $input")
    return JLD2.jldopen(input, "r") do file
        Dict{String,Any}(String(key) => file[key] for key in keys(file))
    end
end

function _cgs_physical_calibration_samples(cfg::AbstractDict,
                                           seeds,
                                           target_delta_tilde::Real,
                                           delta0_over_2pi_mhz::Real)
    count = length(seeds)
    spans = Vector{Float64}(undef, count)
    lh_norms = Vector{Float64}(undef, count)
    ld_norms = Vector{Float64}(undef, count)
    Threads.@threads for sample_index in eachindex(seeds)
        seed = seeds[sample_index]
        block = multimode_open_system_block(
            _cgs_multimode_params(cfg, target_delta_tilde),
            _cgs_multimode_loss(cfg, delta0_over_2pi_mhz);
            seed = seed)
        H = Matrix{ComplexF64}(block.H)
        energies = eigvals(Hermitian(0.5 .* (H .+ H')))
        spans[sample_index] = Float64(maximum(energies) - minimum(energies))
        lh_norms[sample_index] = coherent_superoperator_norm(H)
        L_H = assemble_lindblad_liouvillian(H, [])
        ld_norms[sample_index] = norm(block.L - L_H)
    end
    return spans, lh_norms, ld_norms
end

function _cgs_baseline_calibration_samples(cfg::AbstractDict,
                                           baseline::AbstractDict,
                                           seeds,
                                           n_random_jumps::Integer)
    _cgs_validate_ladder_calibration(baseline, cfg)
    n_orb = Int(cfg["system"]["n_orb"])
    filling = Int(cfg["system"]["filling"])
    offset = Int(cfg["figure3_control"]["seed_offset"])
    cavity_rate = Float64(cfg["figure3_control"]["cavity_loss_rate_over_J"])
    basis = generate_vectors(n_orb, filling)
    d = length(basis)
    unit_spans = Vector{Float64}(undef, length(seeds))
    ld_norms = Vector{Float64}(undef, length(seeds))
    Threads.@threads for sample_index in eachindex(seeds)
        seed = seeds[sample_index]
        rng_H = MersenneTwister(offset + seed)
        rng_jump = MersenneTwister(offset + seed + 1_000_000)
        rng_cavity = MersenneTwister(offset + seed + 2_000_000)
        tensor = random_real_syk4_tensor(rng_H, n_orb, 1.0)
        H = Matrix{ComplexF64}(syk4_block_from_tensor(n_orb, basis, tensor))
        energies = eigvals(Hermitian(0.5 .* (H .+ H')))
        unit_spans[sample_index] = Float64(maximum(energies) - minimum(energies))

        one_body_jumps = random_symmetric_jumps_equal_weight(
            rng_jump, n_orb, n_random_jumps,
            sqrt(Float64(baseline["lambda_star_60"])))
        cavity = random_symmetric_cavity_op(
            rng_cavity, n_orb, Float64(baseline["sigma_g"]))
        jumps = many_body_jump_operators_from_matrices(
            n_orb, basis, one_body_jumps; gamma_eff = 1.0)
        append!(jumps, many_body_jump_operators_from_matrices(
            n_orb, basis, [cavity]; gamma_eff = cavity_rate))
        L_D = assemble_lindblad_liouvillian(
            spzeros(ComplexF64, d, d), jumps)
        ld_norms[sample_index] = norm(L_D)
    end
    return unit_spans, ld_norms
end

function _cgs_require_store_output(path::AbstractString)
    store = get(ENV, "STORE", "")
    isempty(store) && error("STORE must be set for production calibration")
    output = abspath(path)
    store_root = abspath(store)
    separator = string(Base.Filesystem.path_separator)
    output == store_root || startswith(output, store_root * separator) ||
        error("calibration output must be beneath STORE: $output")
    return output
end

function cgs_calibration_main(argv = ARGS)
    options = cgs_parse_calibration_cli(argv)
    config_path = abspath(options["config"])
    baseline_path = abspath(options["baseline-calibration"])
    output = _cgs_require_store_output(options["output"])
    n_seeds = parse(Int, options["n-seeds"])
    n_random_jumps = parse(Int, options["n-random-jumps"])
    target_delta_tilde = parse(Float64, options["target-delta-tilde"])
    n_seeds == 64 || error("CGS Figure-3 calibration requires 64 seeds")
    n_random_jumps == 300 || error("CGS Figure-3 calibration requires 300 random jumps")
    target_delta_tilde == 0.01 ||
        error("CGS Figure-3 calibration requires target delta_tilde=0.01")

    cfg = cgs_load_config(config_path; production = true)
    delta0 = if haskey(options, "delta-cd-over-2pi-mhz")
        parse(Float64, options["delta-cd-over-2pi-mhz"])
    else
        only(_cgs_delta0_values(cfg, :figure3))
    end
    delta0 in _cgs_delta0_values(cfg, :figure3) || error(
        "requested Delta_0/2pi=$delta0 is absent from the Figure-3 grid")
    kappa = Float64(cfg["figure3"]["kappa_over_2pi_mhz"])
    baseline = _cgs_read_calibration_input(baseline_path)
    seeds = collect(1:n_seeds)
    physical_spans, physical_lh_norms, physical_ld_norms =
        _cgs_physical_calibration_samples(
            cfg, seeds, target_delta_tilde, delta0)
    unit_spans, baseline_ld_norms = _cgs_baseline_calibration_samples(
        cfg, baseline, seeds, n_random_jumps)
    payload = cgs_calibration_payload(
        physical_spans, physical_lh_norms, physical_ld_norms,
        unit_spans, baseline_ld_norms;
        n_orb = Int(cfg["system"]["n_orb"]),
        filling = Int(cfg["system"]["filling"]),
        target_delta_tilde = target_delta_tilde,
        n_random_jumps = n_random_jumps,
        n_cavity_jumps = 1,
        delta0_over_2pi_mhz = delta0,
        kappa_over_2pi_mhz = kappa,
        baseline = baseline)

    merge!(payload, Dict{String,Any}(
        "seeds" => seeds,
        "seed_range" => [1, n_seeds],
    ))
    validate_component_metrics(payload;
        rtol = Float64(cfg["validation"]["calibration_relative_tolerance"]),
        expected_n_random_jumps = 300,
        expected_n_cavity_jumps = 1,
        expected_n_orb = 8,
        expected_filling = 4,
        expected_target_delta_tilde = 0.01)
    cgs_write_calibration(output, payload)
    println("wrote CGS Figure-3 calibration: $output")
    println("  sigma_syk4_bdi_span=", payload["sigma_syk4_bdi_span"])
    println("  figure3_dissipator_rate_scale=",
            payload["figure3_dissipator_rate_scale"])
    println("  Delta_0/2pi [MHz]=", payload["delta_0_over_2pi_mhz"])
    return output
end

if abspath(PROGRAM_FILE) == @__FILE__
    cgs_calibration_main()
end
