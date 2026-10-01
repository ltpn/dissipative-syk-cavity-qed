#!/usr/bin/env julia

# Usage:
#   julia validate_figure3_syk4_reference.jl \
#     --config PATH --spectra-dir PATH --calibration PATH \
#     --seed-range 1:64 --output PATH \
#     [--calibration-job-id ID] [--production-job-id ID]

module Figure3SYK4Validation

using JLD2
using Printf
using Statistics: mean
using TOML
include(joinpath(@__DIR__, "..", "..", "..", "DatasetPaths.jl"))

const HERE = @__DIR__
const LAMB_DIR = abspath(joinpath(HERE, "..", "..", ".."))
include(joinpath(LAMB_DIR, "Figure3SYK4Calibration.jl"))
using .Figure3SYK4Calibration: relative_error, validate_component_metrics

export validate_reference, main

function _seed_path(spectra_dir::AbstractString, config::AbstractString,
                    eta::Real, gamma::Real, tol::Real, seed::Integer)
    filename = @sprintf("%s__eta=%s__gamma=%s__tol=%s__seed=%d.jld2",
                        config, string(Float64(eta)), string(Float64(gamma)),
                        string(Float64(tol)), seed)
    return joinpath(spectra_dir, filename)
end

function _read_jld2(path::AbstractString)
    isfile(path) || error("JLD2 input not found: $path")
    return JLD2.jldopen(path, "r") do file
        Dict{String,Any}(String(key) => file[key] for key in keys(file))
    end
end

function _one(values, label)
    length(values) == 1 || error("expected one $label, got $(collect(values))")
    return only(values)
end

function _reference_contract(config_name::AbstractString)
    config_name == "synthetic_l3b_bdi_syk4_fig3_m300" &&
        return (n_random_jumps = 300,)
    config_name == "synthetic_l3b_bdi_syk4_fig3_m10" &&
        return (n_random_jumps = 10,)
    error("unsupported Figure 3 SYK4 reference config: $config_name")
end

function validate_reference(config_path::AbstractString,
                            spectra_dir::AbstractString,
                            calibration_path::AbstractString,
                            seeds,
                            output_path::AbstractString;
                            component_rtol::Real = 0.02,
                            conservation_tol::Real = 1e-8)
    config_abs = abspath(config_path)
    spectra_abs = abspath(spectra_dir)
    calibration_abs = abspath(calibration_path)
    output_abs = abspath(output_path)
    isfile(config_abs) || error("config not found: $config_abs")
    isdir(spectra_abs) || error("spectra directory not found: $spectra_abs")
    cfg = DatasetPaths.read_manifest(config_abs)
    grid = cfg["grid"]
    numerics = cfg["numerics"]
    config_name = String(_one(grid["configs"], "grid config"))
    contract = _reference_contract(config_name)
    calibration = _read_jld2(calibration_abs)

    Float64(calibration["target_delta_tilde"]) == 0.01 ||
        error("calibration target_delta_tilde must be 0.01")
    Int(calibration["n_random_jumps"]) == contract.n_random_jumps ||
        error("calibration must specify $(contract.n_random_jumps) random jumps")
    Int(calibration["n_cavity_jumps"]) == 1 ||
        error("calibration must specify one cavity jump")

    eta = Float64(_one(grid["etas"], "eta"))
    gamma = Float64(_one(grid["gammas"], "gamma"))
    svd_tol = Float64(grid["baseline_svd_tol"])
    n_orb = Int(numerics["n_orb"])
    filling = Int(numerics["filling"])
    n_orb == Int(calibration["n_orb"]) || error("config/calibration n_orb mismatch")
    filling == Int(calibration["filling"]) ||
        error("config/calibration filling mismatch")

    requested = Int.(collect(seeds))
    isempty(requested) && error("seed range is empty")
    length(unique(requested)) == length(requested) || error("duplicate seeds")
    requested == collect(first(requested):last(requested)) ||
        error("seed range must be contiguous and ordered")
    length(requested) <= Int(grid["seed_count"]) ||
        error("seed range exceeds configured seed count")

    h_spans = Float64[]
    lh_norms = Float64[]
    ld_norms = Float64[]
    r_hd = Float64[]
    trace_residuals = Float64[]
    particle_residuals = Float64[]
    times_ref = nothing
    required_keys = (
        "L_svd_S_centered", "times", "entropy_t", "populations_t",
        "r_HD_frobenius", "metadata", "H_spectral_span",
        "L_H_frobenius", "L_D_frobenius",
        "dynamics_trace_residual_max", "dynamics_particle_residual_max",
    )

    for seed in requested
        path = _seed_path(spectra_abs, config_name, eta, gamma, svd_tol, seed)
        isfile(path) || error("missing seed $seed: $path")
        point = JLD2.jldopen(path, "r") do file
            for key in required_keys
                haskey(file, key) || error("seed $seed lacks $key")
            end
            metadata = Dict{String,Any}(file["metadata"])
            Int(metadata["seed"]) == seed || error("seed metadata mismatch for $seed")
            Int(metadata["n_spontaneous_emission_jumps"]) ==
                contract.n_random_jumps ||
                error("seed $seed does not contain $(contract.n_random_jumps) random jumps")
            Int(metadata["n_cavity_loss_jumps"]) == 1 ||
                error("seed $seed does not contain one cavity jump")
            expected_total_jumps = contract.n_random_jumps + 1
            Int(metadata["n_total_jumps"]) == expected_total_jumps ||
                error("seed $seed does not contain $expected_total_jumps total jumps")
            sigmas = Vector{Float64}(file["L_svd_S_centered"])
            times = Vector{Float64}(file["times"])
            entropy = Vector{Float64}(file["entropy_t"])
            populations = Matrix{Float64}(file["populations_t"])
            isempty(sigmas) && error("seed $seed has empty centered singular values")
            isempty(times) && error("seed $seed has empty time grid")
            length(entropy) == length(times) || error("seed $seed entropy length mismatch")
            size(populations) == (n_orb, length(times)) ||
                error("seed $seed population shape mismatch")
            return (
                times = times,
                h_span = Float64(file["H_spectral_span"]),
                lh_norm = Float64(file["L_H_frobenius"]),
                ld_norm = Float64(file["L_D_frobenius"]),
                ratio = Float64(file["r_HD_frobenius"]),
                trace_residual = Float64(file["dynamics_trace_residual_max"]),
                particle_residual =
                    Float64(file["dynamics_particle_residual_max"]),
            )
        end
        all(isfinite, (point.h_span, point.lh_norm, point.ld_norm, point.ratio,
                       point.trace_residual, point.particle_residual)) ||
            error("seed $seed contains non-finite validation metrics")
        point.h_span > 0 && point.lh_norm > 0 && point.ld_norm > 0 ||
            error("seed $seed contains non-positive component metrics")
        isapprox(point.ratio, point.lh_norm / point.ld_norm;
                 rtol = 1e-10, atol = 0) ||
            error("seed $seed r_HD is inconsistent with component norms")
        point.trace_residual < conservation_tol ||
            error("seed $seed trace residual fails tolerance")
        point.particle_residual < conservation_tol ||
            error("seed $seed particle residual fails tolerance")
        if times_ref === nothing
            times_ref = point.times
        else
            point.times == times_ref || error("seed $seed uses a different time grid")
        end
        push!(h_spans, point.h_span)
        push!(lh_norms, point.lh_norm)
        push!(ld_norms, point.ld_norm)
        push!(r_hd, point.ratio)
        push!(trace_residuals, point.trace_residual)
        push!(particle_residuals, point.particle_residual)
    end

    metrics = Dict{String,Any}(
        "physical_H_span_mean" => Float64(calibration["physical_H_span_mean"]),
        "synthetic_H_span_mean" => mean(h_spans),
        "physical_LD_norm_mean" => Float64(calibration["physical_LD_norm_mean"]),
        "synthetic_LD_norm_mean" => mean(ld_norms),
        "n_random_jumps" => contract.n_random_jumps,
        "n_cavity_jumps" => 1,
    )
    validate_component_metrics(metrics; rtol = component_rtol,
        expected_n_random_jumps = contract.n_random_jumps)
    h_error = relative_error(metrics["synthetic_H_span_mean"],
                             metrics["physical_H_span_mean"])
    ld_error = relative_error(metrics["synthetic_LD_norm_mean"],
                              metrics["physical_LD_norm_mean"])

    report = Dict{String,Any}(
        "validated" => true,
        "config_name" => config_name,
        "seed_range" => [first(requested), last(requested)],
        "n_seeds" => length(requested),
        "n_random_jumps" => contract.n_random_jumps,
        "n_cavity_jumps" => 1,
        "n_total_jumps" => contract.n_random_jumps + 1,
        "target_delta_tilde" => 0.01,
        "config" => config_abs,
        "spectra_dir" => spectra_abs,
        "calibration" => calibration_abs,
        "physical_H_span_mean" => metrics["physical_H_span_mean"],
        "synthetic_H_span_mean" => metrics["synthetic_H_span_mean"],
        "H_span_relative_error" => h_error,
        "physical_LD_norm_mean" => metrics["physical_LD_norm_mean"],
        "synthetic_LD_norm_mean" => metrics["synthetic_LD_norm_mean"],
        "LD_norm_relative_error" => ld_error,
        "synthetic_LH_norm_mean" => mean(lh_norms),
        "synthetic_r_HD_mean" => mean(r_hd),
        "trace_residual_max" => maximum(trace_residuals),
        "particle_residual_max" => maximum(particle_residuals),
        "component_relative_tolerance" => Float64(component_rtol),
        "conservation_tolerance" => Float64(conservation_tol),

    )

    mkpath(dirname(output_abs))
    tmp = output_abs * ".tmp"
    open(tmp, "w") do io
        DatasetPaths.print_manifest(io, report; sorted=true, path=output_abs)
    end
    mv(tmp, output_abs; force = true)
    return report
end

function _parse_seed_range(value::AbstractString)
    parts = split(value, ':'; limit = 2)
    length(parts) == 2 || error("--seed-range must have the form FIRST:LAST")
    first_seed, last_seed = parse.(Int, parts)
    first_seed <= last_seed || error("--seed-range is descending")
    return first_seed:last_seed
end

function _parse_cli(argv)
    opts = Dict{String,String}(

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
    for key in ("config", "spectra-dir", "calibration", "seed-range", "output")
        haskey(opts, key) || error("missing required option --$key")
    end
    return opts
end

function main(argv = ARGS)
    opts = _parse_cli(argv)
    report = validate_reference(
        opts["config"], opts["spectra-dir"], opts["calibration"],
        _parse_seed_range(opts["seed-range"]), opts["output"])

    println("validated Figure 3 SYK4 reference: $(opts["output"])")
    println("  H span relative error=$(report["H_span_relative_error"])")
    println("  LD norm relative error=$(report["LD_norm_relative_error"])")
    return report
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    Figure3SYK4Validation.main()
end
