#!/usr/bin/env julia

using JLD2
using LinearAlgebra
using Printf
using Random
using SHA
using Statistics: mean, quantile
using TOML
include(joinpath(@__DIR__, "..", "..", "..", "DatasetPaths.jl"))

const HERE = @__DIR__
const LAMB_DIR = abspath(joinpath(HERE, "..", "..", ".."))
const REPO_ROOT = abspath(joinpath(LAMB_DIR, "..", ".."))
const MODULE_PATH = joinpath(LAMB_DIR, "ComplexDSFFAIDaggerReference.jl")
include(MODULE_PATH)
using .ComplexDSFFAIDaggerReference
using .ComplexDSFFAIDaggerReference: _atomic_jld2

const DEFAULT_MATRIX_SIZES = [128, 256, 512]
const DEFAULT_SEED_COUNTS = [512, 256, 128]
const DEFAULT_RNG_SEED = 20260804
const DEFAULT_THETA = pi / 4
const DEFAULT_ALPHA_TILDE = 1.0
const DEFAULT_THRESHOLD = 0.95
const CACHE_VERSION = 1

mutable struct CLIOptions
    matrix_sizes::Vector{Int}
    seed_counts::Vector{Int}
    rng_seed::Int
    n_boot::Int
    u_min::Float64
    u_max::Float64
    u_points::Int
    cache_dir::String
    output::String
    force::Bool
    smoke::Bool
end

_default_root() = joinpath(REPO_ROOT, "data", "ai_dagger")

function _usage()
    println("""
Usage: generate_ai_dagger_dsff_reference.jl [options]

  --matrix-sizes LIST  matrix dimensions (default 128,256,512)
  --seed-counts LIST   independent matrices per size (default 512,256,128)
  --rng-seed N         deterministic base seed (default 20260804)
  --n-boot N           seed-cluster bootstrap draws (default 500)
  --u-min X            minimum spacing-scaled time (default 1e-3)
  --u-max X            maximum spacing-scaled time (default 1e2)
  --u-points N         logarithmic grid length (default 500)
  --cache-dir PATH     resumable spectrum and finite-size caches
  --output PATH        final extrapolated JLD2 reference
  --smoke              cap sizes, seeds, grid, and bootstraps
  --force              ignore reusable derived caches
  --help                show this help

The reference uses a complex-symmetric Gaussian ensemble, the circular-law
filter f(z)=exp(-|z|^2), the density time kappa=t/sqrt(rho_eff), and an
extrapolation linear in 1/N.
""")
end

_parse_int_list(value) = parse.(Int, strip.(split(value, ',')))

function parse_cli(args = ARGS)
    root = _default_root()
    matrix_sizes = copy(DEFAULT_MATRIX_SIZES)
    seed_counts = copy(DEFAULT_SEED_COUNTS)
    rng_seed = DEFAULT_RNG_SEED
    n_boot = 500
    u_min = 1e-3
    u_max = 1e2
    u_points = 500
    cache_dir = joinpath(root, "cache")
    output = joinpath(root, "gaussian_ai_dagger_dsff_reference.jld2")
    force = false
    smoke = false
    index = 1
    while index <= length(args)
        argument = args[index]
        if argument == "--help"
            _usage()
            exit(0)
        elseif argument == "--force"
            force = true
        elseif argument == "--smoke"
            smoke = true
        else
            index < length(args) ||
                throw(ArgumentError("missing value for $argument"))
            value = args[index + 1]
            if argument == "--matrix-sizes"
                matrix_sizes = _parse_int_list(value)
            elseif argument == "--seed-counts"
                seed_counts = _parse_int_list(value)
            elseif argument == "--rng-seed"
                rng_seed = parse(Int, value)
            elseif argument == "--n-boot"
                n_boot = parse(Int, value)
            elseif argument == "--u-min"
                u_min = parse(Float64, value)
            elseif argument == "--u-max"
                u_max = parse(Float64, value)
            elseif argument == "--u-points"
                u_points = parse(Int, value)
            elseif argument == "--cache-dir"
                cache_dir = abspath(value)
            elseif argument == "--output"
                output = abspath(value)
            else
                throw(ArgumentError("unknown option $argument"))
            end
            index += 1
        end
        index += 1
    end
    length(matrix_sizes) == length(seed_counts) >= 2 ||
        throw(ArgumentError("matrix sizes and seed counts must have equal length >= 2"))
    all(>(1), matrix_sizes) || throw(ArgumentError("matrix sizes must exceed one"))
    length(unique(matrix_sizes)) == length(matrix_sizes) ||
        throw(ArgumentError("matrix sizes must be distinct"))
    all(>(1), seed_counts) || throw(ArgumentError("seed counts must exceed one"))
    n_boot > 0 || throw(ArgumentError("n_boot must be positive"))
    0 < u_min < u_max || throw(ArgumentError("u limits must be ordered and positive"))
    u_points >= 40 || throw(ArgumentError("u_points must be at least 40"))
    if smoke
        matrix_sizes = min.(matrix_sizes, [24 + 16(i - 1)
                                           for i in eachindex(matrix_sizes)])
        seed_counts = min.(seed_counts, 12)
        n_boot = min(n_boot, 40)
        u_points = min(u_points, 100)
    end
    return CLIOptions(matrix_sizes, seed_counts, rng_seed, n_boot,
                      u_min, u_max, u_points, cache_dir, output, force, smoke)
end

_log_range(lo, hi, points) =
    exp.(range(log(Float64(lo)), log(Float64(hi)); length = Int(points)))

function _sha256_file(path::AbstractString)
    return open(path, "r") do io
        bytes2hex(SHA.sha256(io))
    end
end

function _token(parts...)
    encoded = String[]
    for part in parts
        representation = repr(part)
        push!(encoded, string(ncodeunits(representation), ':', representation))
    end
    return bytes2hex(SHA.sha256(codeunits(join(encoded, '|'))))
end

_matrix_seed(base::Int, size::Int, index::Int) =
    mod(base + 1_000_003size + 104_729index, typemax(Int))

function _spectrum_path(options::CLIOptions, size::Int, index::Int)
    return joinpath(options.cache_dir, "spectra", "N=$size",
                    @sprintf("spectrum__seed=%06d.jld2", index))
end

function _load_or_compute_spectrum(options::CLIOptions, size::Int, index::Int)
    path = _spectrum_path(options, size, index)
    seed = _matrix_seed(options.rng_seed, size, index)
    token = _token(CACHE_VERSION, "complex_symmetric_gaussian", size, seed)
    if isfile(path) && !options.force
        cached = JLD2.jldopen(path, "r") do file
            haskey(file, "token") && String(file["token"]) == token || return nothing
            Int(file["matrix_size"]) == size || return nothing
            return ComplexF64.(file["eigenvalues"])
        end
        cached === nothing || return cached, path
    end
    rng = MersenneTwister(seed)
    matrix = gaussian_ai_dagger_matrix(rng, size)
    matrix == transpose(matrix) || error("generated matrix is not complex symmetric")
    eigenvalues = eigvals!(matrix)
    _atomic_jld2(path) do file
        file["token"] = token
        file["matrix_size"] = size
        file["matrix_seed"] = seed
        file["eigenvalues"] = eigenvalues
    end
    return eigenvalues, path
end

function _load_size_spectra(options::CLIOptions, size::Int, count::Int)
    spectra = Vector{Vector{ComplexF64}}(undef, count)
    paths = Vector{String}(undef, count)
    for index in 1:count
        spectra[index], paths[index] =
            _load_or_compute_spectrum(options, size, index)
        if index == 1 || index == count || index % max(1, div(count, 8)) == 0
            @printf("    N=%d spectra %d/%d\n", size, index, count)
        end
    end
    return spectra, paths
end

function _finite_path(options::CLIOptions, size::Int, count::Int)
    return joinpath(options.cache_dir, "finite_size",
                    "finite__N=$(size)__seeds=$(count).jld2")
end

function _load_or_compute_finite(options::CLIOptions, size::Int, count::Int,
                                 spectra, spectrum_paths, u)
    input_hashes = _sha256_file.(spectrum_paths)
    token = _token(CACHE_VERSION, size, count, input_hashes, u, options.n_boot,
                   DEFAULT_THETA, DEFAULT_ALPHA_TILDE)
    path = _finite_path(options, size, count)
    if isfile(path) && !options.force
        cached = JLD2.jldopen(path, "r") do file
            haskey(file, "token") && String(file["token"]) == token || return nothing
            return (
                u = Float64.(file["u"]), curve = Float64.(file["curve"]),
                median = Float64.(file["median"]), lower = Float64.(file["lower"]),
                upper = Float64.(file["upper"]), samples = Float64.(file["samples"]),
                spacing = Float64(file["spacing"]),
                spacing_by_seed = Float64.(file["spacing_by_seed"]),
                plateau = Float64(file["plateau"]),
                plateau_by_seed = Float64.(file["plateau_by_seed"]),
                token = token, cache_path = path)
        end
        cached === nothing || begin
            @printf("    finite-size cache N=%d spacing=%.6g P=%.6g\n",
                    size, cached.spacing, cached.plateau)
            return cached
        end
    end
    result = compute_finite_size_reference(
        spectra, u; theta = DEFAULT_THETA,
        alpha_tilde = DEFAULT_ALPHA_TILDE, n_boot = options.n_boot,
        rng = MersenneTwister(_matrix_seed(options.rng_seed, size, count + 1)))
    _atomic_jld2(path) do file
        file["token"] = token
        file["matrix_size"] = size
        file["seed_count"] = count
        file["u"] = result.u
        file["curve"] = result.curve
        file["median"] = result.median
        file["lower"] = result.lower
        file["upper"] = result.upper
        file["samples"] = result.samples
        file["spacing"] = result.spacing
        file["spacing_by_seed"] = result.spacing_by_seed
        file["plateau"] = result.plateau
        file["plateau_by_seed"] = result.plateau_by_seed
    end
    @printf("    finite-size result N=%d spacing=%.6g P=%.6g\n",
            size, result.spacing, result.plateau)
    return merge(result, (token = token, cache_path = path))
end

function _plateau_diagnostic(curve, samples)
    start = max(1, floor(Int, 0.9length(curve)))
    empirical = mean(@view curve[start:end])
    sample_means = vec(mean(@view(samples[:, start:end]); dims = 2))
    lower = quantile(sample_means, 0.025)
    upper = quantile(sample_means, 0.975)
    return (mean = empirical, lower = lower, upper = upper,
            consistent_95 = lower <= 1 <= upper,
            start_index = start)
end

function _minimum_successful_calibrations(options::CLIOptions)
    requested = options.smoke ? max(2, div(options.n_boot, 4)) : 250
    return min(requested, options.n_boot)
end

function _write_manifest(path, options, reference_path, finite_results,
                         calibration, diagnostic)
    finite_table = Dict{String,Any}()
    for (index, size) in enumerate(options.matrix_sizes)
        result = finite_results[index]
        finite_table[string(size)] = Dict(
            "seed_count" => options.seed_counts[index],
            "spacing" => result.spacing,
            "plateau" => result.plateau,
            "effective_density" => gaussian_filter_effective_density(
                result.plateau, DEFAULT_ALPHA_TILDE, DEFAULT_ALPHA_TILDE),
            "spacing_times_sqrt_effective_density" => result.spacing * sqrt(
                gaussian_filter_effective_density(
                    result.plateau, DEFAULT_ALPHA_TILDE,
                    DEFAULT_ALPHA_TILDE)),
            "cache_path" => result.cache_path)
    end
    payload = Dict{String,Any}(
        "matrix_sizes" => options.matrix_sizes,
        "seed_counts" => options.seed_counts,
        "rng_seed" => options.rng_seed,
        "theta" => DEFAULT_THETA,
        "alpha_tilde" => DEFAULT_ALPHA_TILDE,
        "heisenberg_threshold" => DEFAULT_THRESHOLD,
        "chi" => calibration.chi,
        "chi_bootstrap_median" => calibration.median,
        "chi_bootstrap_lower" => calibration.lower,
        "chi_bootstrap_upper" => calibration.upper,
        "chi_bootstrap_successful" => calibration.successful,
        "chi_bootstrap_requested" => calibration.requested,
        "late_plateau_mean" => diagnostic.mean,
        "late_plateau_95_lower" => diagnostic.lower,
        "late_plateau_95_upper" => diagnostic.upper,
        "late_plateau_consistent_95" => diagnostic.consistent_95,
        "weighted_spacing_grid" => Dict(
            "min" => options.u_min, "max" => options.u_max,
            "points" => options.u_points),
        "n_bootstrap" => options.n_boot,
        "smoke" => options.smoke,
        "cache_dir" => options.cache_dir,
        "finite_sizes" => finite_table,
        "reference_jld2" => reference_path,
    )
    mkpath(dirname(path))
    temporary = path * ".tmp.$(getpid())"
    try
        open(temporary, "w") do io
            DatasetPaths.print_manifest(io, payload; sorted = true, path=path)
        end
        mv(temporary, path; force = true)
    finally
        isfile(temporary) && rm(temporary; force = true)
    end
    return path
end

function main(args = ARGS)
    options = parse_cli(args)
    mkpath(options.cache_dir)
    mkpath(dirname(options.output))
    BLAS.set_num_threads(max(1, min(Sys.CPU_THREADS, 16)))
    u = _log_range(options.u_min, options.u_max, options.u_points)
    println("Gaussian AI-dagger complex-DSFF reference")
    println("  sizes:  $(options.matrix_sizes)")
    println("  seeds:  $(options.seed_counts)")
    println("  cache:  $(options.cache_dir)")
    println("  output: $(options.output)")

    finite_results = Any[]
    spectrum_paths_by_size = Vector{Vector{String}}()
    for (size, count) in zip(options.matrix_sizes, options.seed_counts)
        println("  loading/generating N=$size")
        spectra, spectrum_paths = _load_size_spectra(options, size, count)
        push!(spectrum_paths_by_size, spectrum_paths)
        push!(finite_results, _load_or_compute_finite(
            options, size, count, spectra, spectrum_paths, u))
        GC.gc()
    end

    extrapolated = density_scaled_reference_curves(
        options.matrix_sizes, u,
        [result.curve for result in finite_results],
        [result.samples for result in finite_results],
        [result.spacing for result in finite_results],
        [result.plateau for result in finite_results];
        alpha_x = DEFAULT_ALPHA_TILDE,
        alpha_y = DEFAULT_ALPHA_TILDE)
    calibration = calibrate_extrapolated_reference(
        extrapolated.kappa, extrapolated.curve, extrapolated.samples;
        threshold = DEFAULT_THRESHOLD,
        minimum_successful = _minimum_successful_calibrations(options))
    diagnostic = _plateau_diagnostic(extrapolated.curve, extrapolated.samples)
    @printf("  chi_AI-dagger = %.8g [%.8g, %.8g] (%d/%d bootstraps)\n",
            calibration.chi, calibration.lower, calibration.upper,
            calibration.successful, calibration.requested)
    @printf("  late plateau = %.6g, 95%% [%.6g, %.6g], consistent=%s\n",
            diagnostic.mean, diagnostic.lower, diagnostic.upper,
            diagnostic.consistent_95)

    metadata = Dict{String,Any}(
        "matrix_sizes" => options.matrix_sizes,
        "seed_counts" => options.seed_counts,
        "rng_seed" => options.rng_seed,
        "theta" => DEFAULT_THETA,
        "alpha_tilde" => DEFAULT_ALPHA_TILDE,
        "heisenberg_threshold" => DEFAULT_THRESHOLD,)
    write_ai_dagger_reference(options.output;
        metadata = metadata, u = extrapolated.u, curve = extrapolated.curve,
        lower = extrapolated.lower, upper = extrapolated.upper,
        samples = extrapolated.samples, chi = calibration.chi,
        chi_samples = calibration.chi_samples,
        finite_size_systematic = extrapolated.finite_size_systematic)
    JLD2.jldopen(options.output, "a+") do file
        file["curve_median"] = extrapolated.median
        file["slope_in_inverse_size"] = extrapolated.slope
        file["calibration_u"] = calibration.calibration.u
        file["calibration_raw"] = calibration.calibration.raw
        file["calibration_fit"] = calibration.calibration.fit
        file["calibration_ramp_start_u"] =
            calibration.calibration.ramp_start_u
        file["late_plateau_mean"] = diagnostic.mean
        file["late_plateau_95_lower"] = diagnostic.lower
        file["late_plateau_95_upper"] = diagnostic.upper
        file["late_plateau_consistent_95"] = diagnostic.consistent_95
        for (index, size) in enumerate(options.matrix_sizes)
            result = finite_results[index]
            prefix = "finite_sizes/N_$size"
            file["$prefix/seed_count"] = options.seed_counts[index]
            file["$prefix/curve"] = result.curve
            file["$prefix/lower"] = result.lower
            file["$prefix/upper"] = result.upper
            file["$prefix/spacing"] = result.spacing
            file["$prefix/plateau"] = result.plateau
            file["$prefix/effective_density"] =
                extrapolated.effective_densities[index]
            file["$prefix/density_factor"] =
                extrapolated.density_factors[index]
            file["$prefix/cache_path"] = result.cache_path
        end
    end
    manifest_path = replace(options.output, r"\.jld2$" => "__manifest.toml")
    _write_manifest(manifest_path, options, options.output, finite_results,
                    calibration, diagnostic)
    options.smoke || diagnostic.consistent_95 ||
        error("large-N AI-dagger late plateau is inconsistent with one")
    println("  wrote $(options.output)")
    println("  wrote $manifest_path")
    return (reference = options.output, manifest = manifest_path,
            chi = calibration.chi, plateau = diagnostic)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
