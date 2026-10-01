#!/usr/bin/env julia
#
# Standalone BDI-dagger complex DSFF for the N=10,
# filling-3 spectra behind final PRL Figure 1.  This script reads only stored
# `L_eigvals`; it never rebuilds a Liouvillian and never edits Figure 1.
#
# Complex-DSFF estimator
# -----------------------------------
# The implementation is based on the following primary sources:
#
#   * arXiv:2405.01641v1 (Li, Yan, Prosen, Chan), especially its complex-DOS
#     unfolding and Gaussian spectral-window prescription.  This motivates
#     the fixed conformal map and the HWHM-derived filter used below.
#   * arXiv:2211.01650 (Garcia-Garcia, Sa, Verbaarschot), for the numerical
#     AI-dagger complex-symmetric DSFF, its quadratic mesoscopic ramp, and the
#     natural two-dimensional time scaling t/sqrt(rho).
#   * arXiv:2410.24043 (Kulkarni, Kawabata, Ryu), for analytical AI-dagger
#     characteristic-polynomial/two-point results.  Its controlled result is
#     a large-separation asymptote, not a closed mean-spacing DSFF curve.
#   * arXiv:2212.00474 (Sa, Ribeiro, Prosen), for the many-body Lindbladian
#     symmetry classification and the distinction between the full
#     BDI-dagger constraint and generic complex-bulk correlations.
#   * arXiv:2103.05001 (Li, Prosen, Chan), for the original connected DSFF
#     framework and the analytical diagonal plateau in the Ginibre case.
#
# What this code actually does for canonical Figure 1:
#
#   1. Remove only the steady mode, abs(lambda) <= 1e-8.  Real-axis modes and
#      both members of every conjugate pair are retained.
#   2. Apply the non-adaptive beta=1/2 map
#          g(z) = -im * (z-z0)^(1/2),
#      with z0 fitted from the ensemble-averaged raw DOS.  Fit the mapped-DOS
#      maximum and HWHM widths, then use
#          f = exp[-alpha_x(x-mu_x)^2-alpha_y(y-mu_y)^2],
#          alpha_m = Delta_D,m^(-2).
#   3. Compute the unbiased ensemble-connected DSFF and divide by the
#      analytical filtered plateau P = mean_seed sum_n f_n^2.  The plateau is
#      never forced to one using the observed late-time curve.
#   4. Do NOT compare ensembles using the Prosen weighted spacing
#          s_tilde = mean_n[d_nn f_n f_nn].
#      It includes the nonuniversal occupancy factor mean_n[f_n f_nn] and in
#      the first trial displaced the Gaussian AI-dagger curve by almost one
#      decade.  Instead this project derives, from integral(f^2),
#          rho_eff = 2P*sqrt(alpha_x*alpha_y)/pi,
#          kappa   = abs(tau)/sqrt(rho_eff).
#      This density formula assumes the mapped density is locally flat over
#      the Gaussian window; that approximation is a caveat for the paper.
#   5. Fix the common horizontal constant with an independent numerical
#      Gaussian complex-symmetric ensemble X=X^T at sizes 128,256,512 and
#      seed counts 512,256,128.  Curves are put on a common kappa grid,
#      extrapolated linearly in 1/N, and the post-dip isotonic 0.95 crossing
#      defines chi_AI-dagger.  This is numerical RMT, not a closed analytical
#      BDI-dagger curve and not GinUE.
#
# Locked production diagnostics:
#
#   chi_AI-dagger = 7.155686, bootstrap 68% interval [6.596051, 8.980567]
#   extrapolated late plateau = 0.996084, 95% interval [0.968884, 1.020414]
#   physical 0.95 crossings in displayed x: 0.84,1.27,1.13,1.02 as eta grows
#   L3b target-model crossing in displayed x: 1.50
#
# Why AI-dagger is legitimate only here: the full Liouvillian is BDI-dagger,
# whose global conjugation constraint produces real-axis modes and mirror
# correlations.  For beta=1/2 and theta=pi/4 the Gaussian window selects one
# off-axis lobe and the mirror term dephases.  Direct evaluation with the
# production filters bounds the largest conjugate-pair correction at the
# first displayed point by about 1.1e-4 (eta=2; about 1e-33 for L3b), so the
# panel probes local AI-dagger bulk correlations to plotting accuracy.  At
# theta=0 the conjugate partners have identical projected coordinates, the
# mirror term does not dephase, and this AI-dagger calibration MUST NOT be
# reused.  The red L3b BDI-dagger curve remains in panel (b) precisely as the
# finite-size/global-symmetry check.
#
# Longer derivation and provenance:
# Reference generator: generate_ai_dagger_dsff_reference.jl
# Reference implementation: ComplexDSFFAIDaggerReference.jl

using CairoMakie
using JLD2
using LaTeXStrings
using Printf
using Random
using SHA
using Statistics: mean, median, quantile
using TOML
include(joinpath(@__DIR__, "..", "..", "..", "DatasetPaths.jl"))

const HERE = @__DIR__
const SCRIPT_PATH = abspath(@__FILE__)
const LAMB_DIR = abspath(joinpath(HERE, "..", "..", ".."))
const REPO_ROOT = abspath(joinpath(LAMB_DIR, "..", ".."))
const COMPLEX_DSFF_MODULE_PATH = joinpath(LAMB_DIR, "ComplexDSFFUnfolding.jl")
const COMPLEX_DSFF_PLOTDATA_PATH = joinpath(LAMB_DIR, "ComplexDSFFPlotData.jl")
const AI_DAGGER_REFERENCE_MODULE_PATH =
    joinpath(LAMB_DIR, "ComplexDSFFAIDaggerReference.jl")

if !isdefined(@__MODULE__, :ComplexDSFFUnfolding)
    include(COMPLEX_DSFF_MODULE_PATH)
end
using .ComplexDSFFUnfolding

if !isdefined(@__MODULE__, :ComplexDSFFPlotData)
    include(COMPLEX_DSFF_PLOTDATA_PATH)
end
using .ComplexDSFFPlotData

if !isdefined(@__MODULE__, :ComplexDSFFAIDaggerReference)
    include(AI_DAGGER_REFERENCE_MODULE_PATH)
end
using .ComplexDSFFAIDaggerReference
using .ComplexDSFFAIDaggerReference: atomic_jld2 as _atomic_jld2

if !isdefined(@__MODULE__, :PRLStyle)
    include(joinpath(HERE, "prl_style.jl"))
end
using .PRLStyle

const DEFAULT_MANIFEST = joinpath(
    REPO_ROOT, "figures", "figure_1__manifest.toml")
const CACHE_VERSION = 4
const STEADY_TOL_DEFAULT = 1.0e-8
const PLATEAU_U_MIN = 1.0e2
const PLATEAU_U_MAX = 1.0e4
const PLATEAU_POINTS = 192
const PHYSICAL_ETAS = (0.1, 0.4, 1.0, 2.0)
const RAYS = ((key = "theta_pi4", theta = pi / 4),)

mutable struct CLIOptions
    manifest::String
    output_dir::String
    cache_dir::String
    ai_dagger_reference::String
    n_seeds::Int
    n_boot::Int
    density_bins::Int
    density_sigma::Float64
    plot_points::Int
    fixed_betas::Vector{Float64}
    plot_x_min::Float64
    plot_x_max::Float64
    force::Bool
    smoke::Bool
    minimum_effective_count::Float64
end

_default_cache_dir() = joinpath(REPO_ROOT, "data", "complex_dsff")

function _usage()
    println("""
Usage: figure1b_complex_dsff_unfolding.jl [options]

  --manifest PATH       Figure 1 TOML manifest
  --output-dir PATH     PDF/PNG/plot-data destination
  --cache-dir PATH      resumable JLD2 cache root
  --ai-dagger-reference PATH
                        large-N Gaussian AI-dagger JLD2 calibration
  --plot-x-min X        normalized-time lower limit (default 10^-2.5)
  --plot-x-max X        normalized-time upper limit (default 10)
  --n-seeds N           number of seeds to load (default 64)
  --n-boot N            seed-cluster bootstrap samples (default 500)
  --density-bins N      density histogram bins per axis (default 256)
  --density-sigma X     Gaussian smoothing width in bins (default 2)
  --plot-points N       common x-grid points (default 400)
  --smoke               cap work at 2 seeds and reduced grids
  --minimum-effective-count X
                        DSFF density-fit threshold (default 500; 20 with --smoke).
                        Lower it only for tiny test ensembles; it is recorded
                        in the manifest.
  --force               ignore reusable caches
  --help                show this help
""")
end

function parse_cli(args = ARGS)
    manifest = abspath(DEFAULT_MANIFEST)
    output_dir = abspath(dirname(DEFAULT_MANIFEST))
    cache_dir = abspath(_default_cache_dir())
    ai_dagger_reference = joinpath(REPO_ROOT, "data", "ai_dagger", "gaussian_ai_dagger_dsff_reference.jld2")
    n_seeds = 64
    n_boot = 500
    density_bins = 256
    density_sigma = 2.0
    plot_points = 400
    minimum_effective_count = NaN  # NaN => 500, or 20 with --smoke
    fixed_betas = [0.5]
    plot_x_min = 10.0^-2.5
    plot_x_max = 10.0
    force = false
    smoke = false

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--help"
            _usage()
            exit(0)
        elseif arg == "--force"
            force = true
        elseif arg == "--smoke"
            smoke = true
        else
            i < length(args) || throw(ArgumentError("missing value for $arg"))
            value = args[i + 1]
            if arg == "--manifest"
                manifest = abspath(value)
            elseif arg == "--output-dir"
                output_dir = abspath(value)
            elseif arg == "--cache-dir"
                cache_dir = abspath(value)
            elseif arg == "--ai-dagger-reference"
                ai_dagger_reference = abspath(value)
            elseif arg == "--plot-x-min"
                plot_x_min = parse(Float64, value)
            elseif arg == "--plot-x-max"
                plot_x_max = parse(Float64, value)
            elseif arg == "--n-seeds"
                n_seeds = parse(Int, value)
            elseif arg == "--n-boot"
                n_boot = parse(Int, value)
            elseif arg == "--density-bins"
                density_bins = parse(Int, value)
            elseif arg == "--density-sigma"
                density_sigma = parse(Float64, value)
            elseif arg == "--plot-points"
                plot_points = parse(Int, value)
            elseif arg == "--minimum-effective-count"
                minimum_effective_count = parse(Float64, value)
            else
                throw(ArgumentError("unknown option $arg"))
            end
            i += 1
        end
        i += 1
    end
    n_seeds >= 2 || throw(ArgumentError("n_seeds must be at least 2"))
    n_boot > 0 || throw(ArgumentError("n_boot must be positive"))
    density_bins >= 8 || throw(ArgumentError("density_bins must be at least 8"))
    plot_points >= 20 || throw(ArgumentError("plot_points must be at least 20"))
    0 < plot_x_min < plot_x_max ||
        throw(ArgumentError("plot x limits must satisfy 0 < min < max"))
    if smoke
        n_seeds = min(n_seeds, 2)
        n_boot = min(n_boot, 40)
        density_bins = min(density_bins, 96)
        plot_points = min(plot_points, 100)
    end
    return CLIOptions(manifest, output_dir, cache_dir, ai_dagger_reference,
                      n_seeds,
                      n_boot, density_bins, density_sigma,
                      plot_points, unique(fixed_betas),
                      plot_x_min, plot_x_max, force, smoke,
                      isnan(minimum_effective_count) ? (smoke ? 20.0 : 500.0) : minimum_effective_count)
end

function _require_manifest_value(manifest::Dict{String,Any}, key::String,
                                 expected)
    haskey(manifest, key) ||
        throw(ArgumentError("production manifest is missing '$key'"))
    manifest[key] == expected ||
        throw(ArgumentError("production manifest '$key' must equal $expected, got $(manifest[key])"))
    return nothing
end

"""
Reject manifests that do not describe an N = 10 production input.

The paper protocol (N, Delta_cd, etas, steady tolerance, gamma) is locked; the
filling and ensemble size are free but must be self-consistent: the requested
seed count matches both the main ensemble and the L3b overlay, and the recorded
dimensions match C(N, Q).
"""
function validate_production_manifest(manifest::Dict{String,Any},
                                      options::CLIOptions)
    options.smoke && return nothing
    n_seeds = options.n_seeds
    n_seeds > 0 || throw(ArgumentError("production requires a positive seed count"))
    _require_manifest_value(manifest, "n_orb", 10)
    haskey(manifest, "filling") ||
        throw(ArgumentError("production manifest is missing 'filling'"))
    hilbert_dim = binomial(10, Int(manifest["filling"]))
    _require_manifest_value(manifest, "delta_cd_over_2pi_mhz", 1.0)
    _require_manifest_value(manifest, "n_seeds", n_seeds)
    _require_manifest_value(manifest, "seed_range", Any[1, n_seeds])
    _require_manifest_value(manifest, "steady_tol", STEADY_TOL_DEFAULT)
    _require_manifest_value(manifest, "K_liouville", hilbert_dim^2)
    _require_manifest_value(manifest, "hilbert_dim", hilbert_dim)
    _require_manifest_value(manifest, "etas", Any[0.1, 0.4, 1.0, 2.0])
    haskey(manifest, "l2_overlay") ||
        throw(ArgumentError("production manifest is missing the L3b overlay"))
    overlay = manifest["l2_overlay"]
    overlay isa Dict{String,Any} ||
        throw(ArgumentError("L3b overlay metadata has an invalid type"))
    _require_manifest_value(overlay, "n_seeds", n_seeds)
    _require_manifest_value(overlay, "seed_range", Any[1, n_seeds])
    _require_manifest_value(overlay, "gamma", 1.0)
    return nothing
end

"Ensure every loaded seed has the locked number of nonsteady levels."
function validate_spectral_dimensions(spectra_by_key::AbstractDict,
                                      expected_levels::Integer)
    expected_levels > 0 || throw(ArgumentError("expected_levels must be positive"))
    for (key, spectra) in spectra_by_key
        for (seed, spectrum) in enumerate(spectra)
            length(spectrum) == expected_levels ||
                throw(ArgumentError("$key seed $seed has $(length(spectrum)) levels; expected $expected_levels"))
        end
    end
    return nothing
end

_fmt(value::Real) = string(Float64(value))

function spectrum_filename(config::AbstractString, eta::Real, gamma::Real,
                           tolerance::Real, seed::Integer)
    return @sprintf("%s__eta=%s__gamma=%s__tol=%s__seed=%d.jld2",
                    config, _fmt(eta), _fmt(gamma), _fmt(tolerance), Int(seed))
end

function load_spectrum_file(path::AbstractString;
                            steady_tol::Real = STEADY_TOL_DEFAULT)
    tolerance = Float64(steady_tol)
    isfinite(tolerance) && tolerance >= 0 ||
        throw(ArgumentError("steady_tol must be finite and nonnegative"))
    isfile(path) || throw(ArgumentError("missing spectrum: $path"))
    values = JLD2.jldopen(path, "r") do file
        haskey(file, "L_eigvals") ||
            throw(ArgumentError("L_eigvals missing from $path"))
        ComplexF64.(file["L_eigvals"])
    end
    all(isfinite, real.(values)) && all(isfinite, imag.(values)) ||
        throw(ArgumentError("nonfinite raw eigenvalues in $path"))
    filtered = ComplexF64[z for z in values if abs(z) > tolerance]
    isempty(filtered) && throw(ArgumentError("no nonsteady eigenvalues in $path"))
    return filtered
end

function _dataset_metadata(manifest::Dict{String,Any})
    physical_dir = String(manifest["spectra_dir"])
    gamma = Float64(manifest["delta_cd_over_2pi_mhz"])
    tolerance = Float64(manifest["svd_tol"])
    datasets = NamedTuple[]
    for eta in Float64.(manifest["etas"])
        eta in PHYSICAL_ETAS || continue
        key = "eta_" * replace(_fmt(eta), "." => "p")
        push!(datasets, (key = key,
                         config = "physical", eta = eta, gamma = gamma,
                         tolerance = tolerance, spectra_dir = physical_dir))
    end
    length(datasets) == length(PHYSICAL_ETAS) ||
        throw(ArgumentError("Figure 1 manifest does not contain all physical eta values"))

    overlay = manifest["l2_overlay"]
    config_name = String(overlay["config_name"])
    occursin("l3b", lowercase(config_name)) ||
        throw(ArgumentError("Figure 1 overlay is not the required L3b control"))
    push!(datasets, (key = "l3b",
                     config = config_name, eta = Float64(overlay["eta"]),
                     gamma = Float64(overlay["gamma"]),
                     tolerance = Float64(overlay["svd_tol"]),
                     spectra_dir = String(overlay["spectra_dir"])))
    return datasets
end

function _load_dataset(meta, n_seeds::Int, steady_tol::Float64)
    spectra = Vector{Vector{ComplexF64}}(undef, n_seeds)
    paths = Vector{String}(undef, n_seeds)
    for seed in 1:n_seeds
        path = joinpath(meta.spectra_dir,
                        spectrum_filename(meta.config, meta.eta, meta.gamma,
                                          meta.tolerance, seed))
        paths[seed] = path
        spectra[seed] = load_spectrum_file(path; steady_tol = steady_tol)
    end
    return spectra, paths
end

function spec_summary(spec::UnfoldingSpec)
    map = spec.map
    scores = [Dict("beta" => Float64(score.beta),
                   "flatness" => Float64(score.flatness),
                   "n_eff" => Float64(score.n_eff),
                   "selected" => Bool(score.selected))
              for score in spec.candidate_scores]
    return Dict{String,Any}(
        "beta" => map.beta,
        "z0" => map.z0,
        "mu_x" => spec.filter.mu_x,
        "mu_y" => spec.filter.mu_y,
        "delta_x" => spec.filter.delta_x,
        "delta_y" => spec.filter.delta_y,
        "alpha_x" => spec.filter.alpha_x,
        "alpha_y" => spec.filter.alpha_y,
        "flatness" => spec.flatness,
        "n_eff_median" => spec.n_eff_median,
        "n_eff_by_seed" => spec.n_eff_by_seed,
        "candidate_scores" => scores,
        "density_x_bounds" => [first(spec.density.x), last(spec.density.x)],
        "density_y_bounds" => [first(spec.density.y), last(spec.density.y)],
    )
end

function _fingerprint(parts...)
    payload = join(string.(parts), '\n')
    return bytes2hex(SHA.sha256(codeunits(payload)))
end

function _source_fingerprint(paths::Vector{String})
    details = String[]
    for path in paths
        push!(details, string(DatasetPaths.relative_metadata(path), '|', _sha256_file(path)))
    end
    return _fingerprint(details...)
end

function preprocess_cache_token(options::CLIOptions, source_token::AbstractString,
                                policy, steady_tol::Real)
    minimum_effective_count = options.minimum_effective_count
    return _fingerprint(
        CACHE_VERSION, source_token, policy_slug(policy), options.n_seeds,
        options.density_bins, options.density_sigma,
        minimum_effective_count, Float64(steady_tol), options.smoke)
end

function _deterministic_seed(parts...)
    digest = SHA.sha256(codeunits(join(string.(parts), '|')))
    value = zero(UInt32)
    for i in 1:4
        value |= UInt32(digest[i]) << (8 * (i - 1))
    end
    return Int(value)
end

function _write_spec(file, prefix::AbstractString, spec::UnfoldingSpec)
    summary = spec_summary(spec)
    for key in ("beta", "z0", "mu_x", "mu_y", "delta_x",
                "delta_y", "flatness", "n_eff_median")
        file["$prefix/$key"] = summary[key]
    end
    file["$prefix/n_eff_by_seed"] = spec.n_eff_by_seed
    file["$prefix/density_x"] = spec.density.x
    file["$prefix/density_y"] = spec.density.y
    file["$prefix/density_values"] = spec.density.values
    file["$prefix/candidate_beta"] = Float64[s.beta for s in spec.candidate_scores]
    file["$prefix/candidate_flatness"] = Float64[s.flatness for s in spec.candidate_scores]
    file["$prefix/candidate_n_eff"] = Float64[s.n_eff for s in spec.candidate_scores]
    file["$prefix/candidate_selected"] = Bool[s.selected for s in spec.candidate_scores]
end

function _read_spec(file, prefix::AbstractString)
    map = PowerMap(Float64(file["$prefix/beta"]), Float64(file["$prefix/z0"]))
    filter = GaussianFilter(Float64(file["$prefix/mu_x"]),
                            Float64(file["$prefix/mu_y"]),
                            Float64(file["$prefix/delta_x"]),
                            Float64(file["$prefix/delta_y"]))
    density = DensityGrid(Float64.(file["$prefix/density_x"]),
                          Float64.(file["$prefix/density_y"]),
                          Float64.(file["$prefix/density_values"]))
    betas = Float64.(file["$prefix/candidate_beta"])
    flatnesses = Float64.(file["$prefix/candidate_flatness"])
    counts = Float64.(file["$prefix/candidate_n_eff"])
    selected = Bool.(file["$prefix/candidate_selected"])
    scores = NamedTuple[(beta = betas[i], flatness = flatnesses[i],
                         n_eff = counts[i], selected = selected[i])
                        for i in eachindex(betas)]
    n_eff = Float64.(file["$prefix/n_eff_by_seed"])
    return UnfoldingSpec(map, filter, density,
                         Float64(file["$prefix/flatness"]),
                         Float64(file["$prefix/n_eff_median"]), n_eff, scores)
end

function _preprocess_cache_path(options::CLIOptions, policy, key::String)
    return joinpath(options.cache_dir, policy_slug(policy),
                    "preprocess__$(key)__seeds=$(options.n_seeds).jld2")
end

function _build_preprocessed(options::CLIOptions, meta, spectra,
                             source_token::String, policy,
                             steady_tol::Float64)
    token = preprocess_cache_token(options, source_token, policy, steady_tol)
    cache_path = _preprocess_cache_path(options, policy, meta.key)
    if isfile(cache_path) && !options.force
        cached = JLD2.jldopen(cache_path, "r") do file
            String(file["token"]) == token || return nothing
            spec = _read_spec(file, "spec")
            return (spec = spec,
                    mapped = Vector{Vector{ComplexF64}}(file["mapped_spectra"]),
                    weights = Vector{Vector{Float64}}(file["weights_by_seed"]),
                    plateau_by_seed = Float64.(file["plateau_by_seed"]),
                    spacing_mean = Float64(file["spacing_mean"]),
                    spacing_per_seed = Float64.(file["spacing_per_seed"]),
                    token = token, cache_path = cache_path)
        end
        cached !== nothing && begin
            @printf("  cache %-12s %-11s  %s\n", policy_slug(policy), meta.key,
                    cache_path)
            return cached
        end
    end

    @printf("  fit   %-12s %-11s  (%d seeds)\n", policy_slug(policy), meta.key,
            length(spectra))
    spec = fit_fixed_power_policy(
        spectra, policy.beta; n_bins = options.density_bins,
        smooth_sigma = options.density_sigma,
        minimum_effective_count = options.minimum_effective_count)
    mapped = mapped_spectra(spectra, spec)
    weights = [filter_weights(seed_spectrum, spec.filter)
               for seed_spectrum in mapped]
    plateau_by_seed = Float64[sum(abs2, seed_weights) for seed_weights in weights]
    spacing = ensemble_weighted_spacing(spectra, spec)
    _atomic_jld2(cache_path) do file
        file["token"] = token
        _write_spec(file, "spec", spec)
        file["mapped_spectra"] = mapped
        file["weights_by_seed"] = weights
        file["plateau_by_seed"] = plateau_by_seed
        file["spacing_mean"] = spacing.mean
        file["spacing_per_seed"] = spacing.per_seed
    end
    return (spec = spec, mapped = mapped, weights = weights,
            plateau_by_seed = plateau_by_seed, spacing_mean = spacing.mean,
            spacing_per_seed = spacing.per_seed, token = token,
            cache_path = cache_path)
end

_log_range(lo, hi, n) = exp.(range(log(Float64(lo)), log(Float64(hi)); length = Int(n)))

"Adapt the independent Gaussian AI-dagger reference to the existing curve API."
function _theory_calibration(reference)
    calibration = (
        chi = reference.chi,
        u = reference.u,
        raw = reference.curve,
        fit = reference.curve,
        threshold = reference.threshold,
        ramp_start_u = NaN)
    return (calibration = calibration,
            median = reference.chi_median,
            lower = reference.chi_lower,
            upper = reference.chi_upper,
            samples = reference.chi_samples,
            token = reference.token,
            cache_path = reference.path)
end

"Convert the displayed Heisenberg coordinate to the dataset's physical time."
function _curve_times(x::AbstractVector{<:Real}, calibration, preprocessed)
    chi = calibration.calibration.chi
    plateau = mean(preprocessed.plateau_by_seed)
    filter = preprocessed.spec.filter
    density = gaussian_filter_effective_density(plateau, filter.alpha_x, filter.alpha_y)
    return Float64.(x) .* chi .* sqrt(density)

end

function _curve_cache_path(options, policy, ray_key, dataset_key)
    return joinpath(options.cache_dir, policy_slug(policy),
                    "curve__$(dataset_key)__$(ray_key)__seeds=$(options.n_seeds).jld2")
end

function _load_or_compute_curve(options, policy, ray, meta, preprocessed,
                                calibration)
    x = _log_range(options.plot_x_min, options.plot_x_max, options.plot_points)
    chi = calibration.calibration.chi
    token = _fingerprint(CACHE_VERSION, preprocessed.token, calibration.token,
                         ray.key, options.plot_points, options.plot_x_min,
                         options.plot_x_max, options.n_boot, chi)
    cache_path = _curve_cache_path(options, policy, ray.key, meta.key)
    if isfile(cache_path) && !options.force
        cached = JLD2.jldopen(cache_path, "r") do file
            String(file["token"]) == token || return nothing
            return (x = Float64.(file["x"]),
                    times = Float64.(file["times"]),
                    Z = ComplexF64.(file["Z"]),
                    curve = Float64.(file["curve"]),
                    median = Float64.(file["median"]),
                    lower = Float64.(file["lower"]),
                    upper = Float64.(file["upper"]),
                    samples = Float64.(file["samples"]),
                    plateau = Float64(file["plateau"]),
                    omitted = Int(file["omitted_nonpositive"]),
                    token = token, cache_path = cache_path)
        end
        cached !== nothing && begin
            @printf("  curve cache       %-12s %-9s %-11s\n",
                    policy_slug(policy), ray.key, meta.key)
            return cached
        end
    end

    times = _curve_times(x, calibration, preprocessed)
    Z, _ = partition_traces(preprocessed.mapped, preprocessed.spec.filter,
                            times, ray.theta)
    rng = MersenneTwister(_deterministic_seed(:curve, policy_slug(policy),
                                               ray.key, meta.key,
                                               options.n_seeds, options.n_boot))
    boot = bootstrap_connected_dsff(Z, preprocessed.plateau_by_seed;
                                    n_boot = options.n_boot, rng = rng)
    omitted = count(value -> !(isfinite(value) && value > 0), boot.curve)
    _atomic_jld2(cache_path) do file
        file["token"] = token
        file["x"] = x
        file["times"] = times
        file["Z"] = Z
        file["curve"] = boot.curve
        file["median"] = boot.median
        file["lower"] = boot.lower
        file["upper"] = boot.upper
        file["samples"] = boot.samples
        file["plateau"] = boot.plateau
        file["omitted_nonpositive"] = omitted
    end
    @printf("  curve             %-12s %-9s %-11s omitted=%d\n",
            policy_slug(policy), ray.key, meta.key, omitted)
    return merge(boot, (x = x, times = times, Z = Z, omitted = omitted,
                        token = token, cache_path = cache_path))
end

"Summarize a high-time curve against its ray-resolved projected plateau."
function asymptotic_plateau_diagnostic(curve::AbstractVector{<:Real},
                                       samples::AbstractMatrix{<:Real},
                                       expected_ratio::Real)
    size(samples, 2) == length(curve) ||
        throw(DimensionMismatch("plateau bootstrap grid mismatch"))
    isempty(curve) && throw(ArgumentError("plateau curve cannot be empty"))
    sample_means = vec(mean(samples; dims = 2))
    expected = Float64(expected_ratio)
    lower68 = quantile(sample_means, 0.16)
    upper68 = quantile(sample_means, 0.84)
    lower95 = quantile(sample_means, 0.025)
    upper95 = quantile(sample_means, 0.975)
    return Dict{String,Any}(
        "empirical_mean" => mean(curve),
        "expected_projected_plateau_over_P" => expected,
        "bootstrap_68_lower" => lower68,
        "bootstrap_68_upper" => upper68,
        "bootstrap_95_lower" => lower95,
        "bootstrap_95_upper" => upper95,
        "consistent_68" => lower68 <= expected <= upper68,
        "consistent_95" => lower95 <= expected <= upper95,
    )
end

function _plateau_cache_path(options, policy, ray_key)
    return joinpath(options.cache_dir, policy_slug(policy),
                    "plateau_validation__l3b__$(ray_key)__seeds=$(options.n_seeds).jld2")
end

function _load_or_compute_plateau_validation(options, policy, ray, reference)
    n_points = options.smoke ? 24 : PLATEAU_POINTS
    u = _log_range(PLATEAU_U_MIN, PLATEAU_U_MAX, n_points)
    token = _fingerprint(CACHE_VERSION, reference.token, ray.key, n_points,
                         options.n_boot, PLATEAU_U_MIN, PLATEAU_U_MAX,
                         :projected_plateau)
    cache_path = _plateau_cache_path(options, policy, ray.key)
    if isfile(cache_path) && !options.force
        cached = JLD2.jldopen(cache_path, "r") do file
            String(file["token"]) == token || return nothing
            curve = Float64.(file["curve"])
            samples = Float64.(file["samples"])
            expected = Float64(file["expected_projected_plateau_over_P"])
            return (u = Float64.(file["u"]), curve = curve,
                    median = Float64.(file["median"]),
                    lower = Float64.(file["lower"]),
                    upper = Float64.(file["upper"]), samples = samples,
                    projected_plateau_by_seed =
                        Float64.(file["projected_plateau_by_seed"]),
                    expected_ratio = expected,
                    diagnostic = asymptotic_plateau_diagnostic(
                        curve, samples, expected),
                    token = token, cache_path = cache_path)
        end
        cached !== nothing && begin
            @printf("  plateau cache     %-12s %-9s mean=%.5g expected=%.5g\n",
                    policy_slug(policy), ray.key,
                    cached.diagnostic["empirical_mean"], cached.expected_ratio)
            return cached
        end
    end

    times = u ./ reference.spacing_mean
    Z, _ = partition_traces(reference.mapped, reference.spec.filter,
                            times, ray.theta)
    projected_by_seed = projected_plateau_by_seed(
        reference.mapped, reference.weights, ray.theta)
    expected_ratio = mean(projected_by_seed) / mean(reference.plateau_by_seed)
    rng = MersenneTwister(_deterministic_seed(
        :plateau, policy_slug(policy), ray.key,
        options.n_seeds, options.n_boot))
    boot = bootstrap_connected_dsff(Z, reference.plateau_by_seed;
                                    n_boot = options.n_boot, rng = rng)
    diagnostic = asymptotic_plateau_diagnostic(
        boot.curve, boot.samples, expected_ratio)
    _atomic_jld2(cache_path) do file
        file["token"] = token
        file["u"] = u
        file["Z"] = Z
        file["curve"] = boot.curve
        file["median"] = boot.median
        file["lower"] = boot.lower
        file["upper"] = boot.upper
        file["samples"] = boot.samples
        file["projected_plateau_by_seed"] = projected_by_seed
        file["expected_projected_plateau_over_P"] = expected_ratio
    end
    @printf("  plateau           %-12s %-9s mean=%.5g expected=%.5g 95%%=%s\n",
            policy_slug(policy), ray.key, diagnostic["empirical_mean"],
            expected_ratio, diagnostic["consistent_95"])
    return (u = u, curve = boot.curve, median = boot.median,
            lower = boot.lower, upper = boot.upper, samples = boot.samples,
            projected_plateau_by_seed = projected_by_seed,
            expected_ratio = expected_ratio, diagnostic = diagnostic,
            token = token, cache_path = cache_path)
end

"Common logarithmic y limits for a fixed-beta, single-ray comparison."

function _sha256_file(path::AbstractString)
    return open(path, "r") do io
        bytes2hex(SHA.sha256(io))
    end
end

function _ensemble_parameters(source_manifest::Dict{String,Any})
    overlay = source_manifest["l2_overlay"]
    return Dict{String,Any}(
        "n_orb" => Int(source_manifest["n_orb"]),
        "filling" => Int(source_manifest["filling"]),
        "physical_etas" => Float64.(source_manifest["etas"]),
        "gamma" => Float64(source_manifest["delta_cd_over_2pi_mhz"]),
        "physical_seed_range" => Int.(source_manifest["seed_range"]),
        "physical_n_seeds" => Int(source_manifest["n_seeds"]),
        "l3b_config_name" => String(overlay["config_name"]),
        "l3b_eta" => Float64(overlay["eta"]),
        "l3b_gamma" => Float64(overlay["gamma"]),
        "l3b_seed_range" => Int.(overlay["seed_range"]),
        "l3b_n_seeds" => Int(overlay["n_seeds"]),
        "l3b_spectra_dir" => abspath(String(overlay["spectra_dir"])),
    )
end

function _write_plotdata(path, options, all_results, all_preprocessed,
                         theory_reference)
    _atomic_jld2(path) do file
        file["n_seeds"] = options.n_seeds
        file["fixed_betas"] = options.fixed_betas
        file["ray_keys"] = ["theta_pi4"]
        file["plot_x_min"] = options.plot_x_min
        file["plot_x_max"] = options.plot_x_max
        file["smoke"] = options.smoke
        prefix = "theory/gaussian_ai_dagger"
        file["$prefix/x"] = theory_reference.x
        file["$prefix/curve"] = theory_reference.curve
        file["$prefix/lower"] = theory_reference.lower
        file["$prefix/upper"] = theory_reference.upper
        file["$prefix/finite_size_systematic"] =
            theory_reference.finite_size_systematic
        file["$prefix/chi"] = theory_reference.chi
        file["$prefix/matrix_sizes"] = theory_reference.matrix_sizes
        file["$prefix/seed_counts"] = theory_reference.seed_counts
        file["$prefix/display_x"] = theory_reference.display_x
        file["$prefix/display_curve"] = theory_reference.display_curve
        for (policy, result) in all_results
            pslug = policy_slug(policy)
            for ray in RAYS
                ray_result = result["rays"][ray.key]
                file["$pslug/$(ray.key)/chi"] = ray_result["calibration"].calibration.chi
                file["$pslug/$(ray.key)/chi_bootstrap_samples"] =
                    ray_result["calibration"].samples
                validation = ray_result["plateau_validation"]
                vprefix = "$pslug/$(ray.key)/plateau_validation"
                file["$vprefix/u"] = validation.u
                file["$vprefix/curve"] = validation.curve
                file["$vprefix/lower"] = validation.lower
                file["$vprefix/upper"] = validation.upper
                file["$vprefix/expected_projected_plateau_over_P"] =
                    validation.expected_ratio
                file["$vprefix/projected_plateau_by_seed"] =
                    validation.projected_plateau_by_seed
                for (key, curve) in ray_result["datasets"]
                    prefix = "$pslug/$(ray.key)/$key"
                    file["$prefix/x"] = curve.x
                    file["$prefix/curve"] = curve.curve
                    file["$prefix/lower"] = curve.lower
                    file["$prefix/upper"] = curve.upper
                    file["$prefix/omitted_nonpositive"] = curve.omitted
                end
            end
            for (key, prep) in all_preprocessed[policy]
                prefix = "$pslug/preprocessing/$key"
                _write_spec(file, prefix, prep.spec)
                file["$prefix/weighted_spacing"] = prep.spacing_mean
                file["$prefix/weighted_spacing_by_seed"] = prep.spacing_per_seed
                file["$prefix/plateau_by_seed"] = prep.plateau_by_seed
                file["$prefix/effective_density"] =
                    gaussian_filter_effective_density(
                        mean(prep.plateau_by_seed),
                        prep.spec.filter.alpha_x,
                        prep.spec.filter.alpha_y)
            end
        end
    end
    return path
end

function _write_manifest(path, options, source_manifest, datasets,
                         all_results, all_preprocessed,
                         plotdata,
                         theory_reference)
    manifest = Dict{String,Any}(
        "source_figure1_manifest_input" => options.manifest,
        "ensemble" => _ensemble_parameters(source_manifest),
        "steady_tol" => Float64(get(source_manifest, "steady_tol", STEADY_TOL_DEFAULT)),
        "n_seeds" => options.n_seeds,
        "n_bootstrap" => options.n_boot,
        "fixed_betas" => options.fixed_betas,
        "ray_keys" => ["theta_pi4"],
        "plot_x_min" => options.plot_x_min,
        "plot_x_max" => options.plot_x_max,
        "density_bins" => options.density_bins,
        "density_smoothing_sigma_bins" => options.density_sigma,
        "gaussian_filter_alpha_tilde" => 1.0,
        "heisenberg_threshold" => 0.95,
        "plateau_validation_u_min" => PLATEAU_U_MIN,
        "plateau_validation_u_max" => PLATEAU_U_MAX,
        "plateau_validation_points" => options.smoke ? 24 : PLATEAU_POINTS,
        "plotdata_jld2" => plotdata,
        "cache_dir" => options.cache_dir,
        "smoke" => options.smoke,
        "minimum_effective_count" => options.minimum_effective_count,
    )
    manifest["gaussian_ai_dagger_reference"] = Dict(
        "path" => theory_reference.path,
        "matrix_sizes" => theory_reference.matrix_sizes,
        "seed_counts" => theory_reference.seed_counts,
        "rng_seed" => theory_reference.rng_seed,
        "theta" => theory_reference.theta,
        "alpha_tilde" => theory_reference.alpha_tilde,
        "chi" => theory_reference.chi,
        "chi_bootstrap_lower" => theory_reference.chi_lower,
        "chi_bootstrap_median" => theory_reference.chi_median,
        "chi_bootstrap_upper" => theory_reference.chi_upper,)
    dataset_table = Dict{String,Any}()
    for meta in datasets
        dataset_table[meta.key] = Dict(
            "config" => meta.config,
            "eta" => meta.eta, "gamma" => meta.gamma,
            "svd_tol" => meta.tolerance, "spectra_dir" => meta.spectra_dir)
    end
    manifest["datasets"] = dataset_table

    policy_table = Dict{String,Any}()
    for (policy, result) in all_results
        pslug = policy_slug(policy)
        ptable = Dict{String,Any}("preprocessing" => Dict{String,Any}(),
                                 "rays" => Dict{String,Any}())
        for (key, prep) in all_preprocessed[policy]
            summary = spec_summary(prep.spec)
            summary["weighted_spacing"] = prep.spacing_mean
            summary["weighted_spacing_by_seed"] = prep.spacing_per_seed
            summary["analytic_plateau_mean"] = mean(prep.plateau_by_seed)
            summary["effective_density"] = gaussian_filter_effective_density(
                mean(prep.plateau_by_seed), prep.spec.filter.alpha_x,
                prep.spec.filter.alpha_y)
            summary["cache_path"] = prep.cache_path
            ptable["preprocessing"][key] = summary
        end
        for ray in RAYS
            ray_result = result["rays"][ray.key]
            calibration = ray_result["calibration"]
            validation = ray_result["plateau_validation"]
            validation_table = copy(validation.diagnostic)
            validation_table["cache_path"] = validation.cache_path
            rtable = Dict{String,Any}(
                "theta" => ray.theta,
                "chi" => calibration.calibration.chi,
                "chi_bootstrap_median" => calibration.median,
                "chi_bootstrap_lower" => calibration.lower,
                "chi_bootstrap_upper" => calibration.upper,
                "ramp_start_u" => calibration.calibration.ramp_start_u,
                "asymptotic_plateau_validation" => validation_table,
                "curves" => Dict{String,Any}())
            for (key, curve) in ray_result["datasets"]
                entry = Dict{String,Any}(
                    "omitted_nonpositive" => curve.omitted,
                    "cache_path" => curve.cache_path)
                rtable["curves"][key] = entry
            end
            ptable["rays"][ray.key] = rtable
        end
        policy_table[pslug] = ptable
    end
    manifest["policies"] = policy_table

    mkpath(dirname(path))
    tmp = path * ".tmp.$(getpid())"
    try
        open(tmp, "w") do io
            DatasetPaths.print_manifest(io, manifest; sorted=true, path=path)
        end
        mv(tmp, path; force = true)
    finally
        isfile(tmp) && rm(tmp; force = true)
    end
    return path
end

function _run_policy(options, policy, datasets, spectra_by_key,
                     source_tokens_by_key, steady_tol, theory_reference)
    println("\n--- policy: $(policy_slug(policy)) ---")
    preprocessed = Dict{String,Any}()
    for meta in datasets
        preprocessed[meta.key] = _build_preprocessed(
            options, meta, spectra_by_key[meta.key],
            source_tokens_by_key[meta.key], policy, steady_tol)
    end
    reference = preprocessed["l3b"]
    result = Dict{String,Any}("rays" => Dict{String,Any}())
    for ray in RAYS
        calibration = _theory_calibration(theory_reference)
        plateau_validation = _load_or_compute_plateau_validation(
            options, policy, ray, reference)
        datasets_result = Dict{String,Any}()
        for meta in datasets
            datasets_result[meta.key] = _load_or_compute_curve(
                options, policy, ray, meta, preprocessed[meta.key], calibration)
        end
        result["rays"][ray.key] = Dict("calibration" => calibration,
                                       "plateau_validation" => plateau_validation,
                                       "datasets" => datasets_result)
    end
    return preprocessed, result
end

function main(args = ARGS)
    options = parse_cli(args)
    isfile(options.manifest) || throw(ArgumentError("manifest not found: $(options.manifest)"))
    source_manifest = DatasetPaths.read_manifest(options.manifest)
    validate_production_manifest(source_manifest, options)
    datasets = _dataset_metadata(source_manifest)
    theory_reference = load_ai_dagger_reference(
        options.ai_dagger_reference; expected_theta = pi / 4)
    if !options.smoke
        theory_reference.matrix_sizes == [128, 256, 512] ||
            throw(ArgumentError("production AI-dagger sizes must be 128,256,512"))
        theory_reference.seed_counts == [512, 256, 128] ||
            throw(ArgumentError("production AI-dagger seed counts must be 512,256,128"))
    end
    steady_tol = Float64(get(source_manifest, "steady_tol", STEADY_TOL_DEFAULT))
    mkpath(options.cache_dir)
    mkpath(options.output_dir)

    println("BDI-dagger complex DSFF")
    println("  manifest:   $(options.manifest)")
    println("  cache:      $(options.cache_dir)")
    println("  output:     $(options.output_dir)")
    println("  seeds:      $(options.n_seeds)")
    requests = FixedPowerPolicy.(options.fixed_betas)
    isempty(requests) && throw(ArgumentError("no unfolding policies requested"))
    println("  policies:   $(join(policy_slug.(requests), ", "))")
    println("  rays:       $(join(["theta_pi4"], ", "))")
    println("  plot x:     [$(options.plot_x_min), $(options.plot_x_max)]")
    println("  Julia threads: $(Threads.nthreads())")
    @printf("  AI-dagger: chi=%.8g sizes=%s\n",
            theory_reference.chi, theory_reference.matrix_sizes)

    spectra_by_key = Dict{String,Vector{Vector{ComplexF64}}}()
    source_tokens_by_key = Dict{String,String}()
    println("\n--- loading stored spectra ---")
    for meta in datasets
        spectra, paths = _load_dataset(meta, options.n_seeds, steady_tol)
        spectra_by_key[meta.key] = spectra
        source_tokens_by_key[meta.key] = _source_fingerprint(paths)
        real_counts = [count(z -> abs(imag(z)) <= 1e-12, seed_spectrum)
                       for seed_spectrum in spectra]
        @printf("  %-11s seeds=%d levels/seed=%d real-axis median=%g\n",
                meta.key, length(spectra), length(first(spectra)), median(real_counts))
    end
    options.smoke || validate_spectral_dimensions(
        spectra_by_key, Int(source_manifest["K_liouville"]) - 1)

    all_preprocessed = Dict{Any,Any}()
    all_results = Dict{Any,Any}()
    for policy in requests
        preprocessed, result = _run_policy(options, policy, datasets,
                                           spectra_by_key,
                                           source_tokens_by_key, steady_tol,
                                           theory_reference)
        all_preprocessed[policy] = preprocessed
        all_results[policy] = result
        GC.gc()
    end
    plotdata_name = "figure_1_fixed_beta_dsff__plotdata.jld2"
    manifest_name = "figure_1_fixed_beta_dsff__manifest.toml"
    plotdata = joinpath(options.output_dir, plotdata_name)
    _write_plotdata(plotdata, options, all_results, all_preprocessed,
                    theory_reference)
    manifest_path = joinpath(options.output_dir, manifest_name)
    _write_manifest(manifest_path, options, source_manifest, datasets,
                    all_results,
                    all_preprocessed, plotdata,
                    theory_reference)
    println("\nWrote plot data: $plotdata")
    println("Wrote manifest:  $manifest_path")

    incompatible = String[]
    if !options.smoke
        parsed = DatasetPaths.read_manifest(manifest_path)
        for (policy, ptable) in parsed["policies"]
            for (ray, rtable) in ptable["rays"]
                diag = rtable["asymptotic_plateau_validation"]
                Bool(diag["consistent_95"]) ||
                    push!(incompatible, "$policy/$ray")
            end
        end
    end
    isempty(incompatible) ||
        error("L3b asymptotic projected-plateau check failed for: $(join(incompatible, ", "))")
    return (plotdata = plotdata, manifest = manifest_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
