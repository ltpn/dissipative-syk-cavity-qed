#!/usr/bin/env julia

using JLD2
using LinearAlgebra

const CGS_COMPUTE_HERE = @__DIR__
const CGS_COMPUTE_LAMB_DIR = abspath(joinpath(CGS_COMPUTE_HERE, "..", "..", ".."))
const CGS_COMPUTE_JULIA_DIR = abspath(joinpath(CGS_COMPUTE_LAMB_DIR, ".."))
const CGS_COMPUTE_REPO_ROOT = abspath(joinpath(CGS_COMPUTE_JULIA_DIR, ".."))

include(joinpath(CGS_COMPUTE_JULIA_DIR, "SYK_setup.jl"))
include(joinpath(CGS_COMPUTE_LAMB_DIR, "LambDickeModel.jl"))
include(joinpath(CGS_COMPUTE_LAMB_DIR, "HarmonicOscillatorGrid.jl"))
include(joinpath(CGS_COMPUTE_LAMB_DIR, "Speckle.jl"))
include(joinpath(CGS_COMPUTE_LAMB_DIR, "OverlapIntegrals.jl"))
include(joinpath(CGS_COMPUTE_LAMB_DIR, "JumpFactorization.jl"))
include(joinpath(CGS_COMPUTE_LAMB_DIR, "PhysicalSYK.jl"))
include(joinpath(CGS_COMPUTE_LAMB_DIR, "LiouvillianIntegration.jl"))
include(joinpath(CGS_COMPUTE_LAMB_DIR, "SyntheticLadder.jl"))
include(joinpath(CGS_COMPUTE_LAMB_DIR, "MultimodeHamiltonian.jl"))
include(joinpath(CGS_COMPUTE_LAMB_DIR, "CGSSurvival.jl"))
include(joinpath(CGS_COMPUTE_LAMB_DIR, "CGSModelBuilders.jl"))

using .CGSSurvival

function cgs_parse_compute_cli(argv)
    options = Dict{String,String}()
    flags = Set{String}()
    index = 1
    while index <= length(argv)
        argument = argv[index]
        startswith(argument, "--") || error("unexpected positional argument: $argument")
        key = argument[3:end]
        if key in ("fixture", "force")
            push!(flags, key)
        elseif occursin('=', argument)
            key, value = split(argument[3:end], '='; limit = 2)
            options[key] = value
        else
            index += 1
            index <= length(argv) || error("--$key requires a value")
            options[key] = argv[index]
        end
        index += 1
    end
    for key in ("config", "dataset-key", "seed", "output-root")
        haskey(options, key) || error("missing required option --$key")
    end
    options["fixture"] = string("fixture" in flags)
    options["force"] = string("force" in flags)
    return options
end

function cgs_validation_thresholds(cfg::AbstractDict)
    validation = cfg["validation"]
    return Dict{String,Float64}(
        "h_residual_max" => Float64(validation["h_residual_max"]),
        "l_residual_max" => Float64(validation["l_residual_max"]),
        "trace_preservation_residual_max" =>
            Float64(validation["trace_preservation_residual_max"]),
        "f0_error_max" => Float64(validation["f0_error_max"]),
        "curve_max_imaginary" => Float64(validation["curve_max_imaginary"]),
        "curve_probability_tolerance" =>
            Float64(validation["curve_probability_tolerance"]),
        "reconstruction_residual_max" =>
            Float64(validation["reconstruction_residual_max"]),
    )
end

function _cgs_compute_require_store_output(output_root::AbstractString)
    store = get(ENV, "STORE", "")
    isempty(store) && error("STORE must be set for production CGS computation")
    root = abspath(output_root)
    store_root = abspath(store)
    separator = string(Base.Filesystem.path_separator)
    root == store_root || startswith(root, store_root * separator) ||
        error("production output root must be beneath STORE: $root")
    return root
end

function _cgs_compute_fixture_model()
    H = ComplexF64[0.0 0.0; 0.0 0.7]
    jump = sqrt(0.3) .* ComplexF64[0 1; 0 0]
    Id = Matrix{ComplexF64}(I, 2, 2)
    JdagJ = jump' * jump
    L = -1im .* (kron(Id, H) .- kron(transpose(H), Id))
    L .+= kron(conj(jump), jump)
    L .-= 0.5 .* kron(Id, JdagJ)
    L .-= 0.5 .* kron(transpose(JdagJ), Id)
    metadata = Dict{String,Any}(
        "dataset_key" => "fixture",
        "family" => "fixture",
        "parameter_name" => "toy_rate",
        "parameter" => 0.3,
        "seed" => 1,
        "toy" => true,
    )
    return (L = Matrix{ComplexF64}(L), H = H, metadata = metadata)
end

function _cgs_find_dataset_spec(cfg::AbstractDict, key::AbstractString)
    matches = filter(spec -> spec.key == key, cgs_dataset_specs(cfg))
    length(matches) == 1 || error("unknown or duplicate dataset key: $key")
    return only(matches)
end

function _cgs_control_calibration_path(cfg::AbstractDict,
                                       spec::CGSDatasetSpec,
                                       options::AbstractDict)
    spec.control || return nothing
    override_key = spec.family == :figure1 ?
        "figure1-calibration" : "figure3-calibration"
    if haskey(options, override_key)
        return abspath(options[override_key])
    end

    if spec.family == :figure1
        return _cgs_store_path(cfg,
            String(cfg["figure1_control"]["calibration_store_relative"]))
    end
    return joinpath(
        _cgs_store_path(cfg, String(cfg["output"]["store_subdir"])),
        String(cfg["figure3_control"]["calibration_relative"]))
end

function _cgs_build_production_model(cfg::AbstractDict,
                                     spec::CGSDatasetSpec,
                                     seed::Integer,
                                     options::AbstractDict)
    calibration_path = _cgs_control_calibration_path(cfg, spec, options)
    if !spec.control
        return cgs_build_model(spec, seed, cfg)
    end
    isfile(calibration_path) || error("control calibration not found: $calibration_path")
    calibration = _cgs_load_jld2(calibration_path)
    build = spec.family == :figure1 ?
        cgs_build_model(spec, seed, cfg; ladder_calibration = calibration) :
        cgs_build_model(spec, seed, cfg; figure3_calibration = calibration)
    return build
end

function _cgs_seed_payload(build,
                           cfg::AbstractDict,
                           times::AbstractVector;
                           production::Bool,
                           dataset_key::AbstractString,
                           family::AbstractString,
                           parameter_name::AbstractString,
                           parameter::Real,
                           delta0_over_2pi_mhz::Union{Nothing,Real} = nothing,
                           seed::Integer,
                           n_orb::Integer,
                           filling::Integer)
    state = beta_zero_cgs(build.H)
    spectral = exact_cgs_spectral_data(build.L, state.rho, times)
    reconstructed = reconstruct_survival(
        spectral.eigenvalues, spectral.weights, times)
    diagnostics = merge(copy(state.diagnostics), copy(spectral.diagnostics))
    diagnostics["stored_reconstruction_residual"] =
        maximum(abs, reconstructed - spectral.complex_curve)
    steady_indices = findall(abs.(spectral.eigenvalues) .<= 1e-8)
    payload = Dict{String,Any}(
        "identity/production" => production,
        "identity/dataset_key" => String(dataset_key),
        "identity/family" => String(family),
        "identity/parameter_name" => String(parameter_name),
        "identity/parameter" => Float64(parameter),
        "identity/seed" => Int(seed),
        "dimensions/n_orb" => Int(n_orb),
        "dimensions/filling" => Int(filling),
        "dimensions/hilbert" => size(build.H, 1),
        "dimensions/liouvillian" => size(build.L, 1),
        "model/resolved_metadata" => build.metadata,
        "cgs/energies" => state.energies,
        "cgs/eigenvectors" => state.eigenvectors,
        "cgs/pivots" => state.pivots,
        "cgs/phase_corrections" => state.phase_corrections,
        "cgs/psi" => state.psi,
        "spectral/eigenvalues" => spectral.eigenvalues,
        "spectral/left_overlaps" => spectral.left_overlaps,
        "spectral/right_coefficients" => spectral.right_coefficients,
        "spectral/weights" => spectral.weights,
        "survival/times" => Float64.(times),
        "survival/complex_curve" => spectral.complex_curve,
        "survival/real_curve" => spectral.real_curve,
        "survival/steady_mode_indices" => steady_indices,
        "survival/steady_mode_weights" => spectral.weights[steady_indices],
        "diagnostics/values" => diagnostics,
        "diagnostics/pass" => true,
    )
    if delta0_over_2pi_mhz !== nothing
        delta0 = Float64(delta0_over_2pi_mhz)
        isfinite(delta0) && delta0 > 0 || throw(ArgumentError(
            "Delta_0 identity must be finite and positive"))
        payload["identity/delta_0_over_2pi_mhz"] = delta0
        payload["identity/delta_cd_over_2pi_mhz"] = delta0
    end
    return payload
end

function cgs_compute_main(argv = ARGS)
    options = cgs_parse_compute_cli(argv)
    fixture = parse(Bool, options["fixture"])
    force = parse(Bool, options["force"])
    config_path = abspath(options["config"])
    dataset_key = options["dataset-key"]
    seed = parse(Int, options["seed"])
    seed > 0 || error("seed must be positive")
    cfg = cgs_load_config(config_path; production = true)
    thresholds = cgs_validation_thresholds(cfg)
    output_root = fixture ? abspath(options["output-root"]) :
        _cgs_compute_require_store_output(options["output-root"])
    canonical_production_root = get(ENV, "STORE", "") == "" ? "" :
        abspath(joinpath(ENV["STORE"], String(cfg["output"]["store_subdir"])))
    if fixture && !isempty(canonical_production_root)
        output_root == canonical_production_root &&
            error("fixture mode cannot write into the production root")
    end
    output = joinpath(output_root, "seeds", dataset_key,
                      "seed_$(lpad(seed, 3, '0')).jld2")
    if isfile(output) && !force
        read_seed_payload(output; validate = true, thresholds = thresholds)
        result = (path = output, status = :skipped)
        println("validated existing CGS seed payload: $output")
        return result
    end

    if fixture
        dataset_key == "fixture" || error("fixture mode requires dataset-key=fixture")
        seed == 1 || error("fixture mode requires seed=1")
        build = _cgs_compute_fixture_model()
        times = [0.0, 0.2, 1.0]
        payload = _cgs_seed_payload(build, cfg, times;
            production = false,
            dataset_key = dataset_key,
            family = "fixture",
            parameter_name = "toy_rate",
            parameter = 0.3,
            seed = seed,
            n_orb = 2,
            filling = 1)
    else
        spec = _cgs_find_dataset_spec(cfg, dataset_key)
        seed in spec.seed_range || error("seed $seed is outside $(spec.seed_range)")
        build = _cgs_build_production_model(cfg, spec, seed, options)
        payload = _cgs_seed_payload(build, cfg, cgs_time_grid(cfg);
            production = true,
            dataset_key = spec.key,
            family = String(spec.family),
            parameter_name = String(spec.parameter_name),
            parameter = spec.parameter,
            delta0_over_2pi_mhz = spec.delta0_over_2pi_mhz,
            seed = seed,
            n_orb = Int(cfg["system"]["n_orb"]),
            filling = Int(cfg["system"]["filling"]))
    end
    result = write_seed_payload(output, payload;
        thresholds = thresholds, force = force)
    println("$(result.status) CGS seed payload: $(result.path)")
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    cgs_compute_main()
end
