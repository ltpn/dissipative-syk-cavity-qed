module Figure3SYK4Reference

using JLD2
using Printf
using TOML
include(joinpath(@__DIR__, "..", "..", "..", "DatasetPaths.jl"))

export load_syk4_reference

_fmt(x::Real) = string(Float64(x))

function _seed_path(spectra_dir::AbstractString, config::AbstractString,
                    eta::Real, gamma::Real, tol::Real, seed::Integer)
    filename = @sprintf("%s__eta=%s__gamma=%s__tol=%s__seed=%d.jld2",
                        config, _fmt(eta), _fmt(gamma), _fmt(tol), seed)
    return joinpath(spectra_dir, filename)
end

function _single_value(values, label::AbstractString)
    length(values) == 1 ||
        error("SYK4+diss. reference requires one $label; got $(collect(values))")
    return only(values)
end

function _reference_contract(config_name::AbstractString)
    config_name == "synthetic_l3b_bdi_syk4_fig3_m300" &&
        return (n_random_jumps = 300,)
    config_name == "synthetic_l3b_bdi_syk4_fig3_m10" &&
        return (n_random_jumps = 10,)
    error("unsupported Figure 3 SYK4 reference config: $config_name")
end

"""
    load_syk4_reference(config_path; spectra_dir_override, eta, seeds,
                        expected_n_orb, expected_filling)

Load the trace-centered singular values and stored occupation dynamics for the
canonical L3b `SYK4+diss.` reference ensemble. Every requested seed is
required, and all dynamics arrays must share one exact native time grid.
"""
function load_syk4_reference(
        config_path::AbstractString;
        spectra_dir_override::AbstractString = "",
        validation_path::AbstractString,
        eta::Real = 2.0,
        seeds = 1:1,
        expected_n_orb::Integer,
        expected_filling::Integer)
    config_abs = abspath(config_path)
    isfile(config_abs) ||
        error("SYK4+diss. reference config not found: $config_abs")
    cfg = DatasetPaths.read_manifest(config_abs)
    haskey(cfg, "grid") || error("SYK4+diss. config lacks [grid]: $config_abs")
    haskey(cfg, "numerics") ||
        error("SYK4+diss. config lacks [numerics]: $config_abs")
    haskey(cfg, "output") ||
        error("SYK4+diss. config lacks [output]: $config_abs")

    grid = cfg["grid"]
    numerics = cfg["numerics"]
    config_name = String(_single_value(grid["configs"], "grid config"))
    contract = _reference_contract(config_name)
    gamma = Float64(_single_value(grid["gammas"], "gamma"))
    configured_etas = Float64.(grid["etas"])
    any(value -> isapprox(value, eta; rtol = 0, atol = 32eps(Float64)),
        configured_etas) ||
        error("SYK4+diss. eta=$(Float64(eta)) is absent from config etas=$(configured_etas)")
    svd_tol = Float64(grid["baseline_svd_tol"])
    seed_count = Int(grid["seed_count"])
    n_orb = Int(numerics["n_orb"])
    filling = haskey(numerics, "filling") ?
        Int(numerics["filling"]) : div(n_orb, 2)
    n_orb == expected_n_orb ||
        error("SYK4+diss. n_orb=$n_orb does not match Figure 3 n_orb=$expected_n_orb")
    filling == expected_filling ||
        error("SYK4+diss. filling=$filling does not match Figure 3 filling=$expected_filling")
    requested_seeds = Int.(collect(seeds))
    isempty(requested_seeds) && error("SYK4+diss. seed range is empty")
    all(seed -> 1 <= seed <= seed_count, requested_seeds) ||
        error("SYK4+diss. seeds $(requested_seeds) exceed configured range 1:$seed_count")
    length(unique(requested_seeds)) == length(requested_seeds) ||
        error("SYK4+diss. seed range contains duplicates: $(requested_seeds)")

    spectra_dir = if isempty(spectra_dir_override)
        abspath(joinpath(@__DIR__, "..", "..", "..", "..", "..", "data", String(cfg["output"]["data_subdir"]), "spectra"))
    else
        abspath(spectra_dir_override)
    end
    isdir(spectra_dir) ||
        error("SYK4+diss. spectra directory not found: $spectra_dir")

    validation_abs = abspath(validation_path)
    isfile(validation_abs) ||
        error("Figure 3 SYK4 validation file not found: $validation_abs")
    validation = DatasetPaths.read_manifest(validation_abs)
    get(validation, "validated", false) === true ||
        error("Figure 3 SYK4 ensemble has not passed validation")
    Float64(validation["target_delta_tilde"]) == 0.01 ||
        error("Figure 3 SYK4 validation target must be delta_tilde=0.01")
    String(validation["config_name"]) == config_name ||
        error("Figure 3 SYK4 validation config name mismatch")
    ispath(String(validation["config"])) && realpath(String(validation["config"])) == realpath(config_abs) ||
        error("Figure 3 SYK4 validation config path mismatch")
    ispath(String(validation["spectra_dir"])) && realpath(String(validation["spectra_dir"])) == realpath(spectra_dir) ||
        error("Figure 3 SYK4 validation spectra directory mismatch")
    Int(validation["n_random_jumps"]) == contract.n_random_jumps ||
        error("Figure 3 SYK4 validation must contain $(contract.n_random_jumps) random jumps")
    Int(validation["n_cavity_jumps"]) == 1 ||
        error("Figure 3 SYK4 validation must contain one cavity jump")
    validated_range = Int.(validation["seed_range"])
    length(validated_range) == 2 || error("invalid validated seed range")
    Int(validation["n_seeds"]) == validated_range[2] - validated_range[1] + 1 ||
        error("validated seed count is inconsistent with its range")
    all(seed -> validated_range[1] <= seed <= validated_range[2],
        requested_seeds) || error("requested seeds exceed the validated range")
    Float64(validation["H_span_relative_error"]) <= 0.02 ||
        error("validated Hamiltonian span exceeds the two-percent tolerance")
    Float64(validation["LD_norm_relative_error"]) <= 0.02 ||
        error("validated dissipator norm exceeds the two-percent tolerance")

    haskey(numerics, "calibration_jld2") ||
        error("Figure 3 SYK4 config lacks numerics.calibration_jld2")
    calibration_path = abspath(String(numerics["calibration_jld2"]))
    isfile(calibration_path) ||
        error("Figure 3 SYK4 calibration payload not found: $calibration_path")
    # Compare file identity, not spelling: the same calibration is reached through
    # a snapshot's data symlink or through its target.
    isfile(String(validation["calibration"])) &&
        realpath(String(validation["calibration"])) == realpath(calibration_path) ||
        error("Figure 3 SYK4 validation calibration path mismatch")
    calibration = JLD2.jldopen(calibration_path, "r") do file
        return (
            target_delta_tilde = Float64(file["target_delta_tilde"]),
            n_random_jumps = Int(file["n_random_jumps"]),
            n_cavity_jumps = Int(file["n_cavity_jumps"]),
        )
    end
    calibration.target_delta_tilde == 0.01 ||
        error("Figure 3 SYK4 calibration payload has the wrong target")
    calibration.n_random_jumps == contract.n_random_jumps &&
        calibration.n_cavity_jumps == 1 ||
        error("Figure 3 SYK4 calibration payload has the wrong jump topology")

    sigmas_by_seed = Vector{Vector{Float64}}()
    times_ref = nothing
    populations_sum = nothing
    for seed in requested_seeds
        path = _seed_path(spectra_dir, config_name, eta, gamma, svd_tol, seed)
        isfile(path) || error("SYK4+diss. seed file not found: $path")
        point = JLD2.jldopen(path, "r") do f
            haskey(f, "L_svd_S_centered") ||
                error("SYK4+diss. seed $seed lacks L_svd_S_centered: $path")
            haskey(f, "times") ||
                error("SYK4+diss. seed $seed lacks times: $path")
            haskey(f, "populations_t") ||
                error("SYK4+diss. seed $seed lacks populations_t: $path")
            return (
                sigmas = sort(Vector{Float64}(f["L_svd_S_centered"])),
                times = Vector{Float64}(f["times"]),
                populations = Matrix{Float64}(f["populations_t"]),
            )
        end
        isempty(point.sigmas) &&
            error("SYK4+diss. seed $seed has no centered singular values: $path")
        size(point.populations) == (expected_n_orb, length(point.times)) ||
            error("SYK4+diss. seed $seed populations_t has size $(size(point.populations)); expected ($expected_n_orb, $(length(point.times)))")
        if times_ref === nothing
            times_ref = point.times
            populations_sum = copy(point.populations)
        else
            point.times == times_ref ||
                error("SYK4+diss. seed $seed uses a different native time grid")
            populations_sum .+= point.populations
        end
        push!(sigmas_by_seed, point.sigmas)
    end

    n_used = length(sigmas_by_seed)
    return (
        config_path = config_abs,
        config_name = config_name,
        spectra_dir = spectra_dir,
        validation_path = validation_abs,
        calibration_path = calibration_path,
        target_delta_tilde = calibration.target_delta_tilde,
        n_random_jumps = calibration.n_random_jumps,
        n_cavity_jumps = calibration.n_cavity_jumps,
        physical_H_span_mean = Float64(validation["physical_H_span_mean"]),
        synthetic_H_span_mean = Float64(validation["synthetic_H_span_mean"]),
        H_span_relative_error = Float64(validation["H_span_relative_error"]),
        physical_LD_norm_mean = Float64(validation["physical_LD_norm_mean"]),
        synthetic_LD_norm_mean = Float64(validation["synthetic_LD_norm_mean"]),
        LD_norm_relative_error = Float64(validation["LD_norm_relative_error"]),
        eta = Float64(eta),
        gamma = gamma,
        svd_tol = svd_tol,
        seed_range = [first(requested_seeds), last(requested_seeds)],
        n_requested = length(requested_seeds),
        n_used = n_used,
        sigmas = sigmas_by_seed,
        times = times_ref::Vector{Float64},
        mean_populations = populations_sum ./ n_used,
    )
end

end
