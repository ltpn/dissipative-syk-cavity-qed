#!/usr/bin/env julia
# Figure 1: trace-centered, unfolded sigma-SFF and fixed-beta complex DSFF.
# The main ensemble has 10 orbitals, 3 particles, and 64 disorder realizations;
# eta = 2 insets use 6 and 8 orbitals with 128 realizations each.
# Centered singular values are harvested from stored spectra or rebuilt from
# the configured model and cached in data/sigma_sff. The DSFF panel reads the
# plot data produced by figure1b_complex_dsff_unfolding.jl.

if Sys.isapple()
    @eval using AppleAccelerate
end

using CairoMakie
using JLD2
using LaTeXStrings
using LinearAlgebra
using Printf
using Random
using SparseArrays: SparseMatrixCSC, sparse
using Statistics: mean, median
using TOML
include(joinpath(@__DIR__, "..", "..", "..", "DatasetPaths.jl"))

const HERE = @__DIR__
include(joinpath(HERE, "_figure_export.jl"))
const LAMB_DIR = abspath(joinpath(HERE, "..", "..", ".."))       # src/model
const MODULE_DIR = LAMB_DIR                                        # alias used in included modules
const REPO_ROOT  = abspath(joinpath(LAMB_DIR, "..", ".."))       # repo root
const DATA_ROOT  = joinpath(REPO_ROOT, "data")
const FIGURE_ROOT = joinpath(REPO_ROOT, "figures")

# Physical-model include chain (order matches spectrum_sweep.jl).
include(joinpath(LAMB_DIR, "..", "SYK_setup.jl"))
include(joinpath(LAMB_DIR, "LambDickeModel.jl"))
include(joinpath(LAMB_DIR, "HarmonicOscillatorGrid.jl"))
include(joinpath(LAMB_DIR, "Speckle.jl"))
include(joinpath(LAMB_DIR, "OverlapIntegrals.jl"))
include(joinpath(LAMB_DIR, "JumpFactorization.jl"))
include(joinpath(LAMB_DIR, "PhysicalSYK.jl"))
include(joinpath(LAMB_DIR, "LiouvillianIntegration.jl"))
include(joinpath(LAMB_DIR, "SyntheticLadder.jl"))
include(joinpath(LAMB_DIR, "IntegrableCorner.jl"))

include(joinpath(HERE, "prl_style.jl"))
include(joinpath(HERE, "MagicLaTeX.jl"))
include(joinpath(HERE, "_common_sff_helpers.jl"))
include(joinpath(HERE, "_figure1_plot_smoothing.jl"))
include(joinpath(LAMB_DIR, "ComplexDSFFPlotData.jl"))
using .PRLStyle
using .MagicLaTeX
using .Figure1PlotSmoothing: sgolay_log_positive
using .ComplexDSFFPlotData
using .IntegrableCorner: folded_goe_form_factor

const DEFAULT_CONFIG = joinpath(LAMB_DIR, "configs",
                                 "physical_n10_f3.toml")
const DEFAULT_OUTPUT_DIR = joinpath(FIGURE_ROOT)
const DEFAULT_CACHE_DIR  = joinpath(DATA_ROOT, "sigma_sff", "n10_f3")
const DEFAULT_INSET_FIGURE1_MANIFESTS = (
    (n_orb = 6, filling = 3, path = joinpath(
        FIGURE_ROOT, "source_n6.toml")),
    (n_orb = 8, filling = 4, path = joinpath(
        FIGURE_ROOT, "source_n8.toml")),
)

# Calibrated real SYK4 control overlay.
# One extra curve on both panels at eta = DEFAULT_L2_ETA using the same
# calibration + build recipe as `spectrum_sweep.jl` L2 branch.
#
# = SYK4+dissipators) via `--l2-config`.  `L2_CONFIG_NAME` is populated at
# runtime from `l2_cfg["grid"]["configs"][1]` so the JLD2 file schema on
# disk (`<config>__eta=...__seed=...jld2`) is matched.  When the ladder
# sweep already wrote per-seed trace-centered SVDs (`L_svd_S_centered`)
# and the caller passes `--l2-spectra-dir <ladder merged/spectra>`, the
# collector HARVESTS them instead of rebuilding — no calibration file
# or synthetic constructor is needed on the plot-time host.
const DEFAULT_L2_CONFIG = joinpath(LAMB_DIR, "configs", "csr_ladder_l3b_n10_f3.toml")
const DEFAULT_L2_ETA    = 2.0
L2_CONFIG_NAME::String  = "synthetic_l3b_bdi_syk4"   # overwritten in main()
const MAGIC_PALETTE = palette_magiclatex(:gem_2024)
const TARGET_MODEL_COLOR = MAGIC_PALETTE[7]
const TARGET_MODEL_STYLE = TARGET_MODEL_LINESTYLE
const LATEX_SCALE = 0.5
const AUTHORING_CANVAS_SIZE_PT = (601.0, 283.0)
const PDF_CROP_MARGIN_PT = 1.0
const FIGURE1_SG_WINDOW = 11
const FIGURE1_SG_DEGREE = 3
const FIGURE1_INSET_GREYS = [
    RGBf(0.72, 0.72, 0.72),
    RGBf(0.46, 0.46, 0.46),
    RGBf(0.20, 0.20, 0.20),
]
source_px(points::Real) = MagicLaTeX.pt2px(points / LATEX_SCALE)

function magic_log_ticks(powers)
    positions = 10.0 .^ collect(powers)
    labels = [latexstring("10^{", power, "}") for power in powers]
    return positions, labels
end

function smooth_figure1_curve(values)
    return sgolay_log_positive(
        positive_or_nan(values);
        window = FIGURE1_SG_WINDOW,
        degree = FIGURE1_SG_DEGREE,
    )
end

# ------------------------------------------------------------------
# Physics constants for this figure
# ------------------------------------------------------------------
const ETAS_PLOT       = (0.1, 0.4, 1.0, 2.0)
const FIGURE1_ETA_VIRIDIS_POSITIONS = collect(range(0.05, 0.90; length = length(ETAS_PLOT)))
const GAMMA           = 1.0                       # Delta_cd/2pi = 1 MHz
const SVD_TOL         = 1.0e-10
const EPSILON_ZERO    = 1.0e-8
const DEFAULT_ANALYSIS_WINDOW = (0.05, 0.95)
const TAU_GRID        = (1e-3, 10.0, 400)         # (min, max, n)
# `eigvals(Symmetric(K x K))` at that size costs ~15 min per realization
# and the full 128-seed ensemble times out even on 4 h budgets.
# System-size dependent RMT reference dimensions.  Overwritten in `main()`
# from the config's `numerics.n_orb` value; defaults kept for N_orb = 6.
K_LIOUV::Int          = 400                       # Liouvillian block dim (default N_orb=6)
D_HILBERT::Int        = 20                        # binomial(6, 3)
FILLING::Int          = 3                         # default half-filling at N_orb=6
const RNG_SEED_SFF_BOOTSTRAP = 20260804
const SFF_N_BOOTSTRAP = 500
const STEADY_TOL      = 1.0e-8                    # matches config numerics.filter.steady_tol

const Y_LOWER_SFF     = 10.0^(-1.5)               # lower y-clip for the sigma-SFF panel
const Y_LOWER_DSFF    = 10.0^(-1.5)               # requested lower clip for complex DSFF
const X_LOWER_SFF     = 10.0^(-2.5)               # lower x-clip for the sigma-SFF panel
const HU_N_BINS       = 40                        # knot bins for the Hamiltonian-unfolding staircase
const HU_DEGREE       = 5                         # requested polynomial degree

# ------------------------------------------------------------------
# I/O helpers (production JLD2 layout, matches sweep_scheduler.jl)
# ------------------------------------------------------------------

log_range(lo, hi, n) = exp.(range(log(Float64(lo)), log(Float64(hi)); length = Int(n)))

# Match the filename convention of `sweep_scheduler.jl:fmt_grid_value`
# (`string(x)` for real values).  This yields e.g. "1.0", "0.4", "1.0e-10".
_fmt(x::Real) = string(Float64(x))
_fmt(x::Integer) = string(x)

function _seed_path(spectra_dir, config, eta, gamma, tol, seed)
    fname = @sprintf("%s__eta=%s__gamma=%s__tol=%s__seed=%d.jld2",
                     config,
                     _fmt(eta), _fmt(gamma), _fmt(tol), seed)
    return joinpath(spectra_dir, fname)
end

# ------------------------------------------------------------------
# LambDickeLiouvillianConfig from TOML  (mirrors `spectrum_sweep.jl:
# make_liouvillian_config` for the `physical` branch that production uses).
# ------------------------------------------------------------------
"""
    _resolve_delta_cd_over_2pi_mhz(num) -> Float64

Cavity-drive detuning in MHz, mirroring
`spectrum_sweep.jl: make_liouvillian_config`.  It is independent of the
swept `gamma_eff` and enters only through
`cavity_loss_rate_over_J = kappa / delta`, so it must never default to the
sweep axis: that would rescale the cavity dissipator as `kappa / gamma`.
The sweep reads the `gamma` axis as the detuning axis in exactly one mode,
`force_gamma_eff_over_J`, which this script does not implement; such a
config is rejected rather than regenerated into spectra that cannot match
the production run.
"""
function _resolve_delta_cd_over_2pi_mhz(num::AbstractDict)
    haskey(num, "force_gamma_eff_over_J") &&
        error("force_gamma_eff_over_J is unsupported here: the sweep then treats " *
              "the `gamma` axis as Delta_cd and pins gamma_eff, which this " *
              "regeneration path cannot reproduce")
    return Float64(get(num, "delta_cd_over_2pi_mhz", 1.0))
end

function build_physical_ldcfg(cfg::AbstractDict, tol::Real, gamma_eff::Real)
    num = cfg["numerics"]
    n_orb = Int(num["n_orb"])
    # Optional custom filling: when absent, LambDickeLiouvillianConfig falls back to div(n_orb,2).
    filling = haskey(num, "filling") ? Int(num["filling"]) : nothing
    common = (n_grid = Int(num["n_grid"]),
              box_length = Float64(num["box_length"]),
              correlation_length = Float64(num["correlation_length"]),
              disorder_strength = Float64(num["disorder_strength"]),
              J = Float64(num["J"]),
              svd_tol = Float64(tol),
              lambda_c_micron = Float64(get(num, "lambda_c_micron", 0.671)),
              kappa_over_2pi_mhz = Float64(get(num, "kappa_over_2pi_mhz", 0.2)),
              delta_cd_over_2pi_mhz = _resolve_delta_cd_over_2pi_mhz(num),
              manual_cavity_loss_rate_over_J =
                  haskey(num, "cavity_loss_rate_over_J") ?
                      Float64(num["cavity_loss_rate_over_J"]) : nothing)
    return LambDickeLiouvillianConfig(; n_orb = n_orb, filling = filling, weight_type = :speckle,
        gamma_eff = Float64(gamma_eff),

        common...)
end

# ------------------------------------------------------------------
# Per-parameter cache of trace-centered singular values (Fig 1)
# ------------------------------------------------------------------
_eta_slug(x)   = replace(@sprintf("%g", x), "." => "p", "-" => "m")
_gamma_slug(x) = replace(@sprintf("%g", x), "." => "p", "-" => "m")

function cache_path_fig1(cache_dir, eta, gamma, n_orb::Integer;
                          filling::Union{Nothing,Integer} = nothing)
    tag = @sprintf("__f=%d", filling === nothing ? div(Int(n_orb), 2) : Int(filling))
    slug = @sprintf("production_centered__norb=%d%s__gamma=%s__eta=%s.jld2",
                    Int(n_orb), tag, _gamma_slug(gamma), _eta_slug(eta))
    return joinpath(cache_dir, slug)
end

"""
Read centered singular values and diagnostics from a production per-seed file.
Return `nothing` only when the file is missing.
"""
function _harvest_centered_from_production(spectra_dir::AbstractString,
                                           config::AbstractString,
                                           eta::Real, gamma::Real,
                                           tol::Real, seed::Integer)
    p = _seed_path(spectra_dir, config, eta, gamma, tol, seed)
    isfile(p) || return nothing
    return JLD2.jldopen(p, "r") do f
        s      = sort(Vector{Float64}(f["L_svd_S_centered"]))
        mu_re = Float64(f["L_trace_shift_mu_re"])
        mu_im = Float64(f["L_trace_shift_mu_im"])
        rHD = Float64(f["r_HD_frobenius"])
        n_j = Int(f["metadata"]["n_total_jumps"])
        return (sigmas = s, mu_re = mu_re, mu_im = mu_im,
                r_HD = rHD, n_jumps = n_j)
    end
end

"""
Load (or build + persist) the per-seed trace-centered singular values of
the production physical Liouvillian at (`eta`, `gamma`).  Also returns the
per-seed complex trace shift `mu` and the per-seed coherent-to-dissipative
Frobenius ratio `r_HD` for parameters.

When `spectra_dir` is given, existing production files supply the centered
payload directly instead of rebuilding.

**Timeout resilience**: after every completed seed (harvested or rebuilt),
the current state is flushed to the target JLD2 atomically.  This makes
the builder resumable at seed-level granularity on SIGTERM.
"""
function collect_or_build_centered_sigmas_fig1(cfg::AbstractDict,
                                                  eta::Real, gamma::Real,
                                                  tol::Real, seeds,
                                                  cache_dir::AbstractString,
                                                  n_orb::Integer;
                                                  filling::Union{Nothing,Integer} = nothing,
                                                  spectra_dir::Union{Nothing,AbstractString} = nothing,
                                                  production_config::AbstractString = "physical")
    mkpath(cache_dir)
    path = cache_path_fig1(cache_dir, eta, gamma, n_orb; filling = filling)
    sigmas  = Vector{Vector{Float64}}()
    mu_re   = Float64[]
    mu_im   = Float64[]
    r_HD    = Float64[]
    n_jumps = Int[]
    if isfile(path)
        JLD2.jldopen(path, "r") do f
            for arr in f["sigmas"]; push!(sigmas, Vector{Float64}(arr)); end
            append!(mu_re,   Vector{Float64}(f["mu_re"]))
            append!(mu_im,   Vector{Float64}(f["mu_im"]))
            append!(r_HD,    Vector{Float64}(f["r_HD"]))
            append!(n_jumps, Vector{Int}(f["n_jumps"]))
        end
    end
    n_seeds = length(seeds)
    have = length(sigmas)
    if have >= n_seeds
        return (sigmas  = sigmas[1:n_seeds],
                mu_re   = mu_re[1:n_seeds],
                mu_im   = mu_im[1:n_seeds],
                r_HD    = r_HD[1:n_seeds],
                n_jumps = n_jumps[1:n_seeds],
                path    = path)
    end
    n_harvested = 0
    for (idx, seed) in enumerate(seeds)
        idx <= have && continue
        t0 = time()
        harvested = spectra_dir === nothing ? nothing :
            _harvest_centered_from_production(spectra_dir, production_config,
                                              eta, gamma, tol, seed)
        if harvested !== nothing
            push!(sigmas,  harvested.sigmas)
            push!(mu_re,   harvested.mu_re)
            push!(mu_im,   harvested.mu_im)
            push!(r_HD,    harvested.r_HD)
            push!(n_jumps, harvested.n_jumps)
            n_harvested += 1
            @printf("    eta=%-4g seed=%3d  K=%d  |mu|=%.3g  r_HD=%.3g  (harvested)  %.1fs\n",
                    eta, seed, length(harvested.sigmas),
                    hypot(harvested.mu_re, harvested.mu_im),
                    harvested.r_HD, time() - t0)
            flush(stdout)
        else
            error("Missing saved spectrum for $production_config eta=$eta seed=$seed. Run pipeline generate, eigen, svd and postprocess, then pass --spectra-dir.")
        end
        # Incremental atomic flush after each seed so timeout doesn't lose work.
        tmp = path * ".tmp"
        JLD2.jldopen(tmp, "w") do f
            f["sigmas"]  = sigmas
            f["mu_re"]   = mu_re
            f["mu_im"]   = mu_im
            f["r_HD"]    = r_HD
            f["n_jumps"] = n_jumps
            f["eta"]     = Float64(eta)
            f["gamma"]   = Float64(gamma)
            f["svd_tol"] = Float64(tol)
            f["n_seeds"] = length(sigmas)
            f["n_orb"]   = Int(n_orb)
            f["filling"] = filling === nothing ? div(Int(n_orb), 2) : Int(filling)
        end
        mv(tmp, path; force = true)
    end
    if n_harvested > 0
        @printf("    -> eta=%-4g summary: harvested=%d\n",
                eta, n_harvested)
    end
    return (sigmas = sigmas, mu_re = mu_re, mu_im = mu_im,
            r_HD = r_HD, n_jumps = n_jumps, path = path)
end

"""Inline JLD2->Dict loader for the CSR-ladder calibration payload."""
function _load_ladder_calibration(path::AbstractString)
    isfile(path) || error("CSR ladder calibration file not found: $path")
    JLD2.jldopen(path, "r") do f
        d = Dict{String,Any}()
        for k in keys(f); d[k] = f[k]; end
        return d
    end
end

"""
Build a `LambDickeLiouvillianConfig` matching the
`synthetic_l3b_bdi_syk4` branch of `spectrum_sweep.jl` for a
single (seed, eta, gamma_eff).  Reads the calibration JLD2 pointed to
by the L2 TOML config (`numerics.calibration_jld2`).
"""
function build_l2_ldcfg_from_toml(l2_cfg::AbstractDict, tol::Real,
                                   gamma_eff::Real, seed::Integer)
    num   = l2_cfg["numerics"]
    seeds = l2_cfg["seeds"]
    n_orb = Int(num["n_orb"])
    filling = haskey(num, "filling") ? Int(num["filling"]) : nothing
    common = (n_grid = Int(num["n_grid"]),
              box_length = Float64(num["box_length"]),
              correlation_length = Float64(num["correlation_length"]),
              disorder_strength = Float64(num["disorder_strength"]),
              J = Float64(num["J"]),
              svd_tol = Float64(tol),
              lambda_c_micron = Float64(get(num, "lambda_c_micron", 0.671)),
              kappa_over_2pi_mhz = Float64(get(num, "kappa_over_2pi_mhz", 0.2)),
              delta_cd_over_2pi_mhz = _resolve_delta_cd_over_2pi_mhz(num),
              manual_cavity_loss_rate_over_J =
                  haskey(num, "cavity_loss_rate_over_J") ?
                      Float64(num["cavity_loss_rate_over_J"]) : nothing)
    calib   = _load_ladder_calibration(String(num["calibration_jld2"]))
    offset  = Int(seeds["l3b_offset"])
    n_orb_calib = Int(calib["n_orb"])
    n_orb == n_orb_calib ||
        error("L2: config n_orb=$n_orb ≠ calibration n_orb=$n_orb_calib")
    f_calib = Int(calib["filling"])
    f_config = filling === nothing ? div(n_orb, 2) : filling
    f_calib == f_config ||
        error("L2: config filling=$f_config ≠ calibration filling=$f_calib")
    sigma_g = Float64(calib["sigma_g"])
    rng_H = MersenneTwister(offset + Int(seed))
    rng_jump = MersenneTwister(offset + Int(seed) + 1_000_000)
    rng_cavity = MersenneTwister(offset + Int(seed) + 2_000_000)
    H_tensor = random_real_syk4_tensor(rng_H, n_orb, Float64(calib["sigma_syk4_bdi_span"]))
    jumps = random_symmetric_jumps_equal_weight(
        rng_jump, n_orb, 60, sqrt(Float64(calib["lambda_star_60"])))
    cavity_op = random_symmetric_cavity_op(rng_cavity, n_orb, sigma_g)
    return LambDickeLiouvillianConfig(; n_orb = n_orb, filling = filling, weight_type = :uniform,
        gamma_eff = Float64(gamma_eff),
        synthetic_hamiltonian_tensor = H_tensor,
        synthetic_jump_matrices = jumps,
        synthetic_cavity_loss_operator = cavity_op,
        common...)
end

function cache_path_l2(cache_dir, eta, gamma, n_orb::Integer;
                       filling::Union{Nothing,Integer} = nothing)
    tag = @sprintf("__f=%d", filling === nothing ? div(Int(n_orb), 2) : Int(filling))
    slug = @sprintf("l2_centered__norb=%d%s__gamma=%s__eta=%s.jld2",
                    Int(n_orb), tag, _gamma_slug(gamma), _eta_slug(eta))
    return joinpath(cache_dir, slug)
end

"""
Build (or load) the per-seed trace-centered singular values of the L2
Liouvillian at (`eta`, `gamma_eff`).  Seed-level atomic flush so the
builder is timeout-resilient at N_orb = 8.

When `spectra_dir` is supplied, existing production files supply the centered
spectra and diagnostics directly.
"""
function collect_or_build_l2_centered_sigmas(l2_cfg::AbstractDict, eta::Real,
                                              gamma_eff::Real, tol::Real, seeds,
                                              cache_dir::AbstractString,
                                              n_orb::Integer;
                                              filling::Union{Nothing,Integer} = nothing,
                                              spectra_dir::Union{Nothing,AbstractString} = nothing)
    mkpath(cache_dir)
    path = cache_path_l2(cache_dir, eta, gamma_eff, n_orb; filling = filling)
    sigmas  = Vector{Vector{Float64}}()
    mu_re   = Float64[]
    mu_im   = Float64[]
    r_HD    = Float64[]
    n_jumps = Int[]
    if isfile(path)
        JLD2.jldopen(path, "r") do f
            for arr in f["sigmas"]; push!(sigmas, Vector{Float64}(arr)); end
            append!(mu_re,   Vector{Float64}(f["mu_re"]))
            append!(mu_im,   Vector{Float64}(f["mu_im"]))
            append!(r_HD,    Vector{Float64}(f["r_HD"]))
            append!(n_jumps, Vector{Int}(f["n_jumps"]))
        end
    end
    n_seeds = length(seeds)
    have = length(sigmas)
    if have >= n_seeds
        return (sigmas  = sigmas[1:n_seeds],
                mu_re   = mu_re[1:n_seeds],
                mu_im   = mu_im[1:n_seeds],
                r_HD    = r_HD[1:n_seeds],
                n_jumps = n_jumps[1:n_seeds],
                path    = path)
    end
    for (idx, seed) in enumerate(seeds)
        idx <= have && continue
        t0 = time()
        # Read centered spectra and diagnostics from the production sweep.
        harvested = spectra_dir === nothing ? nothing :
            _harvest_centered_from_production(String(spectra_dir),
                                              L2_CONFIG_NAME,
                                              eta, gamma_eff, tol, seed)
        if harvested !== nothing
            push!(sigmas,  harvested.sigmas)
            push!(mu_re,   harvested.mu_re)
            push!(mu_im,   harvested.mu_im)
            push!(r_HD,    harvested.r_HD)
            push!(n_jumps, harvested.n_jumps)
            @printf("    L2  eta=%-4g seed=%3d  K=%d  |mu|=%.3g  r_HD=%.3g  (harvested)  %.1fs\n",
                    eta, seed, length(harvested.sigmas),
                    hypot(harvested.mu_re, harvested.mu_im),
                    harvested.r_HD, time() - t0)
            flush(stdout)
        else
            error("Missing saved control spectrum for $L2_CONFIG_NAME eta=$eta seed=$seed. Run the staged pipeline, then pass --l2-spectra-dir.")
        end
        tmp = path * ".tmp"
        JLD2.jldopen(tmp, "w") do f
            f["sigmas"]  = sigmas
            f["mu_re"]   = mu_re
            f["mu_im"]   = mu_im
            f["r_HD"]    = r_HD
            f["n_jumps"] = n_jumps
            f["eta"]     = Float64(eta)
            f["gamma"]   = Float64(gamma_eff)
            f["svd_tol"] = Float64(tol)
            f["n_seeds"] = length(sigmas)
            f["n_orb"]   = Int(n_orb)
            f["config"]  = L2_CONFIG_NAME
        end
        mv(tmp, path; force = true)
    end
    return (sigmas = sigmas, mu_re = mu_re, mu_im = mu_im,
            r_HD = r_HD, n_jumps = n_jumps, path = path)
end

# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
function _parse_cli(argv)
    spectra_dir = ""
    config_path = DEFAULT_CONFIG
    output_dir  = DEFAULT_OUTPUT_DIR
    cache_dir   = DEFAULT_CACHE_DIR
    seeds_str   = ""    # empty => use 1:config.seed_count
    window      = DEFAULT_ANALYSIS_WINDOW
    l2_config   = DEFAULT_L2_CONFIG
    l2_eta      = DEFAULT_L2_ETA
    l2_seeds_str = ""    # empty => use 1:l2_cfg.grid.seed_count
    l2_spectra_dir_override = ""  # empty => derive from config's data_subdir
    complex_dsff_plotdata = ""
    complex_dsff_manifest = ""
    inset_manifests = NamedTuple[]  # empty => DEFAULT_INSET_FIGURE1_MANIFESTS
    include_l2  = true
    i = 1
    while i <= length(argv)
        a = argv[i]
        if a == "--config"; i += 1; config_path = argv[i]
        elseif startswith(a, "--config="); config_path = split(a, "=", limit = 2)[2]
        elseif a == "--spectra-dir"
            i += 1
            i <= length(argv) || error("--spectra-dir requires a path")
            spectra_dir = argv[i]
        elseif startswith(a, "--spectra-dir="); spectra_dir = split(a, "=", limit=2)[2]
        elseif a == "--output-dir"; i += 1; output_dir = argv[i]
        elseif startswith(a, "--output-dir="); output_dir = split(a, "=", limit = 2)[2]
        elseif a == "--cache-dir"; i += 1; cache_dir = argv[i]
        elseif startswith(a, "--cache-dir="); cache_dir = split(a, "=", limit = 2)[2]
        elseif a == "--seeds"; i += 1; seeds_str = argv[i]
        elseif startswith(a, "--seeds="); seeds_str = split(a, "=", limit = 2)[2]
        elseif a == "--analysis-window"; i += 1; window = _parse_window(argv[i])
        elseif startswith(a, "--analysis-window="); window = _parse_window(split(a, "=", limit = 2)[2])
        elseif a == "--l2-config"; i += 1; l2_config = argv[i]
        elseif startswith(a, "--l2-config="); l2_config = split(a, "=", limit = 2)[2]
        elseif a == "--l2-eta"; i += 1; l2_eta = parse(Float64, argv[i])
        elseif startswith(a, "--l2-eta="); l2_eta = parse(Float64, split(a, "=", limit = 2)[2])
        elseif a == "--l2-seeds"; i += 1; l2_seeds_str = argv[i]
        elseif startswith(a, "--l2-seeds="); l2_seeds_str = split(a, "=", limit = 2)[2]
        elseif a == "--l2-spectra-dir"; i += 1; l2_spectra_dir_override = argv[i]
        elseif startswith(a, "--l2-spectra-dir="); l2_spectra_dir_override = split(a, "=", limit = 2)[2]
        elseif a == "--complex-dsff-plotdata"; i += 1; complex_dsff_plotdata = argv[i]
        elseif startswith(a, "--complex-dsff-plotdata="); complex_dsff_plotdata = split(a, "=", limit = 2)[2]
        elseif a == "--complex-dsff-manifest"; i += 1; complex_dsff_manifest = argv[i]
        elseif startswith(a, "--complex-dsff-manifest="); complex_dsff_manifest = split(a, "=", limit = 2)[2]
        elseif a == "--inset-manifest" || startswith(a, "--inset-manifest=")
            value = a == "--inset-manifest" ? (i += 1; argv[i]) : split(a, "=", limit = 2)[2]
            push!(inset_manifests, _parse_inset_manifest(value))
        elseif a == "--no-l2"; include_l2 = false
        else error("unrecognized argument: $a")
        end
        i += 1
    end
    output_dir = abspath(output_dir)
    isempty(complex_dsff_plotdata) &&
        (complex_dsff_plotdata = joinpath(
            output_dir, "figure_1_fixed_beta_dsff__plotdata.jld2"))
    isempty(complex_dsff_manifest) &&
        (complex_dsff_manifest = joinpath(
            output_dir, "figure_1_fixed_beta_dsff__manifest.toml"))
    return (config = abspath(config_path),
             spectra_dir = isempty(spectra_dir) ? "" : abspath(spectra_dir),
             output_dir = output_dir,
             cache_dir  = abspath(cache_dir),
             seeds_str = String(seeds_str),
             window = window,
             l2_config = abspath(l2_config),
             l2_eta = Float64(l2_eta),
             l2_seeds_str = String(l2_seeds_str),
             l2_spectra_dir_override = String(l2_spectra_dir_override),
             complex_dsff_plotdata = abspath(complex_dsff_plotdata),
             complex_dsff_manifest = abspath(complex_dsff_manifest),
             inset_manifests = isempty(inset_manifests) ?
                 collect(DEFAULT_INSET_FIGURE1_MANIFESTS) : inset_manifests,
             include_l2 = include_l2)
end

function _seed_range(str, default_n)
    isempty(str) && return collect(1:default_n)
    occursin(":", str) || error("--seeds must be lo:hi (got $str)")
    parts = split(str, ":")
    return collect(parse(Int, parts[1]):parse(Int, parts[2]))
end

"""
Parse `--inset-manifest N:PATH`: a finite-size source manifest for the eta = 2
inset. The filling is read from the manifest's config (default N/2, as in the
pipeline configs).
"""
function _parse_inset_manifest(value::AbstractString)
    parts = split(value, ':'; limit = 2)
    length(parts) == 2 || error("--inset-manifest must be N:PATH (got $value)")
    n_orb = parse(Int, parts[1])
    path = abspath(String(parts[2]))
    isfile(path) || error("finite-size inset manifest not found: $path")
    numerics = DatasetPaths.read_manifest(String(DatasetPaths.read_manifest(path)["config"]))["numerics"]
    Int(numerics["n_orb"]) == n_orb || error("inset manifest $path is not N=$n_orb")
    filling = haskey(numerics, "filling") ? Int(numerics["filling"]) : div(n_orb, 2)
    return (n_orb = n_orb, filling = filling, path = path)
end

"Resolve the eta=2 finite-size centered-sigma caches from their manifests."
function _resolve_default_inset_caches(specs = DEFAULT_INSET_FIGURE1_MANIFESTS)
    resolved = NamedTuple[]
    for spec in specs
        isfile(spec.path) || error("finite-size inset manifest not found: $(spec.path)")
        manifest = DatasetPaths.read_manifest(spec.path)
        etas = Float64.(manifest["etas"])
        index = findfirst(value -> isapprox(value, 2.0; rtol = 0,
                                             atol = 32eps(Float64)), etas)
        index === nothing && error("finite-size inset manifest lacks eta=2: $(spec.path)")
        cache_paths = String.(manifest["cache_paths"])
        length(cache_paths) == length(etas) ||
            error("finite-size inset manifest cache list is inconsistent: $(spec.path)")
        cache_path = cache_paths[index]
        if !isfile(cache_path)
            cfg = DatasetPaths.read_manifest(String(manifest["config"]))
            result = collect_or_build_centered_sigmas_fig1(
                cfg, 2.0, GAMMA, SVD_TOL, 1:Int(manifest["n_seeds"]),
                dirname(cache_path), spec.n_orb; filling = spec.filling,
                spectra_dir = String(manifest["spectra_dir"]))
            result.path == abspath(cache_path) ||
                abspath(result.path) == abspath(cache_path) ||
                error("finite-size cache path disagrees with source manifest")
        end
        push!(resolved, (n_orb = spec.n_orb,
                         filling = spec.filling,
                         manifest_path = abspath(spec.path),
                         cache_path = abspath(cache_path)))
    end
    return resolved
end

function _load_complex_dsff_inputs(cli, n_seeds::Integer)
    isfile(cli.complex_dsff_plotdata) ||
        error("fixed-beta complex-DSFF plot data not found: $(cli.complex_dsff_plotdata)")
    isfile(cli.complex_dsff_manifest) ||
        error("fixed-beta complex-DSFF manifest not found: $(cli.complex_dsff_manifest)")
    parameters = DatasetPaths.read_manifest(cli.complex_dsff_manifest)
    validate_plotdata_binding(cli.complex_dsff_plotdata, parameters)
    Int(parameters["n_seeds"]) == n_seeds ||
        error("complex DSFF plot data has $(parameters["n_seeds"]) seeds; Figure 1 renders $n_seeds")
    betas = Float64.(parameters["fixed_betas"])
    any(beta -> isapprox(beta, CANONICAL_DSFF_BETA;
                         rtol = 0, atol = 32eps(Float64)), betas) ||
        error("fixed-beta manifest does not contain beta=$(CANONICAL_DSFF_BETA)")
    CANONICAL_DSFF_RAY_KEY in String.(parameters["ray_keys"]) ||
        error("fixed-beta manifest does not contain $(CANONICAL_DSFF_RAY_KEY)")
    isapprox(Float64(parameters["plot_x_min"]), X_LOWER_SFF; rtol = 1e-12) ||
        error("fixed-beta manifest has the wrong lower x limit")
    isapprox(Float64(parameters["plot_x_max"]), TAU_GRID[2]; rtol = 1e-12) ||
        error("fixed-beta manifest has the wrong upper x limit")
    curves = load_canonical_fixed_beta_curves(
        cli.complex_dsff_plotdata;
        x_min = X_LOWER_SFF,
        x_max = TAU_GRID[2],
        n_points = TAU_GRID[3])
    theory = load_gaussian_ai_dagger_curve(cli.complex_dsff_plotdata)
    collections = Vector{Vector{Float64}}()
    for curve in values(curves)
        push!(collections, curve.curve, curve.lower, curve.upper)
    end
    automatic_limits = rounded_log_limits(collections; include = 1.0)
    ylimits = (Y_LOWER_DSFF, automatic_limits[2])
    return (curves = curves, theory = theory,
            ylimits = ylimits, parameters = parameters)
end

function _current_ensemble_parameters(cli, seeds, n_orb::Int, filling::Int)
    isfile(cli.l2_config) || error("L3b config not found: $(cli.l2_config)")
    l3b_cfg = DatasetPaths.read_manifest(cli.l2_config)
    l3b_grid = l3b_cfg["grid"]
    l3b_name = String(l3b_grid["configs"][1])
    l3b_seed_count = Int(l3b_grid["seed_count"])
    l3b_seeds = isempty(cli.l2_seeds_str) ? collect(1:l3b_seed_count) :
                 _seed_range(cli.l2_seeds_str, l3b_seed_count)
    l3b_spectra_dir = isempty(cli.l2_spectra_dir_override) ?
        abspath(joinpath(DATA_ROOT, String(l3b_cfg["output"]["data_subdir"]), "spectra")) :
        abspath(cli.l2_spectra_dir_override)
    return Dict{String,Any}(
        "n_orb" => n_orb,
        "filling" => filling,
        "physical_etas" => collect(ETAS_PLOT),
        "gamma" => GAMMA,
        "physical_seed_range" => [first(seeds), last(seeds)],
        "physical_n_seeds" => length(seeds),
        "l3b_config_name" => l3b_name,
        "l3b_eta" => cli.l2_eta,
        "l3b_gamma" => GAMMA,
        "l3b_seed_range" => [first(l3b_seeds), last(l3b_seeds)],
        "l3b_n_seeds" => length(l3b_seeds),
        "l3b_spectra_dir" => l3b_spectra_dir,
    )
end

# ------------------------------------------------------------------
# Main driver
# ------------------------------------------------------------------
function main(argv = ARGS)
    cli = _parse_cli(argv)
    isfile(cli.config) || error("config not found: $(cli.config)")
    cfg = DatasetPaths.read_manifest(cli.config)
    grid = cfg["grid"]; outcfg = cfg["output"]
    seed_count = Int(grid["seed_count"])
    seeds = _seed_range(cli.seeds_str, seed_count)
    baseline_tol = Float64(grid["baseline_svd_tol"])
    (baseline_tol ≈ SVD_TOL) ||
        @warn "config baseline_svd_tol=$baseline_tol differs from hardcoded SVD_TOL=$SVD_TOL"

    # System-size overrides for the RMT reference dimensions and the cache
    # filename slug (defaults hardwired for the N=6 canonical figures).
    n_orb    = Int(cfg["numerics"]["n_orb"])
    filling  = haskey(cfg["numerics"], "filling") ? Int(cfg["numerics"]["filling"]) : div(n_orb, 2)
    d_hilb   = binomial(n_orb, filling)
    global D_HILBERT = d_hilb
    global K_LIOUV   = d_hilb^2
    global FILLING   = filling

    orig_data = isabspath(outcfg["data_subdir"]) ?
                    String(outcfg["data_subdir"]) :
                    abspath(joinpath(REPO_ROOT, "data",
                                      String(outcfg["data_subdir"])))
    spectra_dir = isempty(cli.spectra_dir) ? joinpath(orig_data, "spectra") : cli.spectra_dir
    isdir(spectra_dir) || error("spectra directory not found: $spectra_dir")

    # SFF tau grid.
    taus = log_range(TAU_GRID[1], TAU_GRID[2], TAU_GRID[3])
    qs   = 2π .* taus
    complex_dsff = _load_complex_dsff_inputs(cli, length(seeds))
    validate_ensemble_compatibility(
        complex_dsff.parameters,
        _current_ensemble_parameters(cli, seeds, n_orb, filling))

    println("=" ^ 78)
    println("Final Figure 1  (sigma-SFF + DSFF theta=pi/4 at Delta_cd/2pi = $GAMMA MHz)")
    println("config:       $(cli.config)")
    println("spectra dir:  $spectra_dir")
    println("output dir:   $(cli.output_dir)")
    println("etas (plot):  $(collect(ETAS_PLOT))")
    println("seeds:        $(first(seeds)):$(last(seeds)) (n=$(length(seeds)))")
    println("analysis wnd: [$(cli.window[1]), $(cli.window[2])]")
    println("=" ^ 78)

    # ---- LEFT panel data (trace-centered, rebuilt Liouvillian) --------
    println("\n--- Building + trace-centering physical Liouvillians (cached) ---")
    println("cache dir:    $(cli.cache_dir)")
    sff_bootstrap = Vector{Any}(undef, length(ETAS_PLOT))
    n_used_per_eta = Int[]
    r_HD_pooled   = Float64[]
    mu_re_pooled  = Float64[]
    mu_im_pooled  = Float64[]
    n_jumps_pooled = Int[]
    cache_paths   = String[]
    for (i, eta) in enumerate(ETAS_PLOT)
        @printf("\n--- eta = %g ---\n", eta)
        c = collect_or_build_centered_sigmas_fig1(cfg, eta, GAMMA, baseline_tol,
                                                     seeds, cli.cache_dir, n_orb;
                                                     filling = filling,
                                                     spectra_dir = spectra_dir)
        push!(cache_paths, c.path)
        append!(r_HD_pooled,    c.r_HD)
        append!(mu_re_pooled,   c.mu_re)
        append!(mu_im_pooled,   c.mu_im)
        append!(n_jumps_pooled, c.n_jumps)
        res = bootstrap_kunf_ham(c.sigmas, qs;
            analysis_window = cli.window,
            epsilon_zero = EPSILON_ZERO,
            n_bins = HU_N_BINS, degree = HU_DEGREE,
            n_boot = SFF_N_BOOTSTRAP,
            rng = MersenneTwister(RNG_SEED_SFF_BOOTSTRAP + i))
        sff_bootstrap[i] = res
        push!(n_used_per_eta, res.n_used)
        @printf("  eta=%-4g  seeds used=%3d  M_bulk=%d\n",
                eta, res.n_used, res.M)
    end

    # Finite-size inset at eta=2: reuse the current N=10 result and compute
    # the identical trace-centered, Hamiltonian-unfolded estimator from the
    # stored N=6 and N=8 centered-sigma caches.
    println("\n--- Preparing eta=2 finite-size sigma-SFF inset ---")
    finite_size_inset = NamedTuple[]
    for spec in _resolve_default_inset_caches(cli.inset_manifests)
        centered = load_centered_sigma_cache(spec.cache_path;
            expected_n_orb = spec.n_orb,
            filling = spec.filling,
            eta = 2.0,
            gamma = GAMMA,
            svd_tol = baseline_tol)
        result = kunf_ham(centered.sigmas, qs;
            analysis_window = cli.window,
            epsilon_zero = EPSILON_ZERO,
            n_bins = HU_N_BINS,
            degree = HU_DEGREE)
        push!(finite_size_inset, (
            n_orb = spec.n_orb,
            filling = spec.filling,
            K = centered.K,
            n_seeds = result.n_used,
            curve = result.k_unf,
            cache_path = spec.cache_path,
            manifest_path = spec.manifest_path,
        ))
        @printf("    N=%2d  f=%d  K=%5d  n_seeds=%d  from %s\n",
                spec.n_orb, spec.filling, centered.K, result.n_used,
                basename(spec.cache_path))
    end
    eta2_index = findfirst(eta -> isapprox(eta, 2.0; rtol = 0,
                                           atol = 32eps(Float64)), ETAS_PLOT)
    eta2_index === nothing && error("physical eta grid lacks eta=2")
    push!(finite_size_inset, (
        n_orb = n_orb,
        filling = filling,
        K = K_LIOUV,
        n_seeds = sff_bootstrap[eta2_index].n_used,
        curve = sff_bootstrap[eta2_index].curve,
        cache_path = abspath(cache_paths[eta2_index]),
        manifest_path = "current run",
    ))
    sort!(finite_size_inset; by = entry -> entry.n_orb)
    @printf("    N=%2d  f=%d  K=%5d  n_seeds=%d  from current run\n",
            n_orb, filling, K_LIOUV, sff_bootstrap[eta2_index].n_used)

    # ---- Physical-ensemble diagnostics for the RMT match ---------------
    r_HD_median = median(r_HD_pooled)
    @printf("\n--- Physical ensemble diagnostics ---\n")
    @printf("    r_HD median (||L_H||_F / ||L_D||_F): %.4g  (min=%.3g max=%.3g)\n",
            r_HD_median, minimum(r_HD_pooled), maximum(r_HD_pooled))
    @printf("    |mu = tr(L)/D|  mean = %.3g\n",
            mean(sqrt.(mu_re_pooled.^2 .+ mu_im_pooled.^2)))
    M_dissipators = maximum(n_jumps_pooled)
    @printf("    n_total_jumps (model)  = %d\n", M_dissipators)

    # Same unit-plateau analytical folded-GOE reference used in Figure 5:
    # K_f-GOE(t/t_Hei) = K_GOE(2|t/t_Hei|).
    goe_curve = folded_goe_form_factor(taus)

    l2_bootstrap = nothing
    l2_manifest = Dict{String,Any}(
        "config"  => cli.include_l2 ? cli.l2_config : "",
    )
    if cli.include_l2
        println("\n--- Ladder overlay (CSR synthetic ladder $(cli.l2_config)) ---")
        isfile(cli.l2_config) ||
            error("L2 config not found: $(cli.l2_config)")
        l2_cfg = DatasetPaths.read_manifest(cli.l2_config)
        l2_grid = l2_cfg["grid"]
        # Bind the ladder config identifier used in JLD2 filenames
        # (`<config>__eta=...__seed=...jld2`) from the ladder TOML.  This is
        # what the sweep driver writes and what fig 1 consumes.
        global L2_CONFIG_NAME = String(l2_grid["configs"][1])
        println("  ladder config name: $L2_CONFIG_NAME")
        l2_seed_count = Int(l2_grid["seed_count"])
        l2_seeds = isempty(cli.l2_seeds_str) ?
            collect(1:l2_seed_count) : _seed_range(cli.l2_seeds_str, l2_seed_count)
        l2_num = l2_cfg["numerics"]
        l2_n_orb = Int(l2_num["n_orb"])
        l2_n_orb == n_orb ||
            error("ladder n_orb=$l2_n_orb ≠ physical n_orb=$n_orb (fig 1 pipeline assumes matched)")
        l2_filling = haskey(l2_num, "filling") ? Int(l2_num["filling"]) : div(l2_n_orb, 2)
        l2_filling == filling ||
            error("ladder filling=$l2_filling ≠ physical filling=$filling (fig 1 pipeline assumes matched)")
        l2_spectra_dir = isempty(cli.l2_spectra_dir_override) ?
            joinpath(DATA_ROOT, String(l2_cfg["output"]["data_subdir"]), "spectra") :
            abspath(cli.l2_spectra_dir_override)
        !isdir(l2_spectra_dir) &&
            @warn "ladder spectra dir not found; centered-SFF harvest disabled" path=l2_spectra_dir
        # tol for L2: baseline tol of the L2 grid (same 1e-10 as physical)
        l2_tol = Float64(l2_grid["baseline_svd_tol"])
        @printf("  ladder config:  %s\n", cli.l2_config)
        @printf("  ladder eta:     %g\n", cli.l2_eta)
        @printf("  ladder seeds:   1:%d (n=%d)\n", last(l2_seeds), length(l2_seeds))
        @printf("  ladder spectra: %s\n", l2_spectra_dir)
        # LEFT panel data (trace-centered harvest-or-rebuild + cache)
        println("  Harvesting/building ladder centered singular values ...")
        l2_cache = collect_or_build_l2_centered_sigmas(l2_cfg,
                        cli.l2_eta, GAMMA, l2_tol, l2_seeds,
                        cli.cache_dir, n_orb; filling = filling,
                        spectra_dir = isdir(l2_spectra_dir) ? l2_spectra_dir : nothing)
        l2_kunf_res = bootstrap_kunf_ham(l2_cache.sigmas, qs;
            analysis_window = cli.window,
            epsilon_zero = EPSILON_ZERO,
            n_bins = HU_N_BINS, degree = HU_DEGREE,
            n_boot = SFF_N_BOOTSTRAP,
            rng = MersenneTwister(RNG_SEED_SFF_BOOTSTRAP + 200))
        l2_bootstrap = l2_kunf_res
        @printf("  L2 SFF: seeds used=%d, M_bulk=%d\n",
                l2_kunf_res.n_used, l2_kunf_res.M)
        l2_manifest["n_seeds_used_dsff"] =
            Int(complex_dsff.parameters["n_seeds"])
        l2_manifest["eta"]             = cli.l2_eta
        l2_manifest["gamma"]           = GAMMA
        l2_manifest["svd_tol"]         = l2_tol
        l2_manifest["seed_range"]      = [first(l2_seeds), last(l2_seeds)]
        l2_manifest["n_seeds"]         = length(l2_seeds)
        l2_manifest["n_seeds_used_sff"]= l2_kunf_res.n_used
        l2_manifest["cache_path"]      = l2_cache.path
        l2_manifest["spectra_dir"]     = l2_spectra_dir
        l2_manifest["config_name"]     = L2_CONFIG_NAME
    end

    # ---- Plot ---------------------------------------------------------
    mkpath(cli.output_dir)
    suffix   = _window_suffix(cli.window)
    pdf_path = joinpath(cli.output_dir, "figure_1_sigma_sff_dsff_theta0$(suffix).pdf")
    png_path = joinpath(cli.output_dir, "figure_1_sigma_sff_dsff_theta0$(suffix).png")

    eta_colormap = cgrad(:viridis)
    eta_colors = [eta_colormap[position]
                  for position in FIGURE1_ETA_VIRIDIS_POSITIONS]
    size_colors = FIGURE1_INSET_GREYS
    main_width = source_px(1.15)
    reference_width = source_px(1.35)
    base_theme = theme_magiclatex(
        PaletteName = :gem_2024,
        FigureSize = MagicLaTeX.pt2px.(AUTHORING_CANVAS_SIZE_PT),
    )
    compact_theme = merge(base_theme, Theme(
        figure_padding = source_px.((2.0, 2.0, 1.5, 1.5)),
    ))

    fig = with_theme(compact_theme) do
        figure = Figure(backgroundcolor = :white)
        axis_common = (
            aspect = 0.95,
            xscale = log10,
            yscale = log10,
            xgridvisible = false,
            ygridvisible = false,
            xticks = magic_log_ticks(-2:1),
            xlabelpadding = source_px(1.5),
            ylabelpadding = source_px(1.5),
            xticklabelpad = source_px(1.3),
            yticklabelpad = source_px(1.3),
        )
        # Both panels show the connected form factor divided by its diagonal
        # plateau K_inf: K_inf=M for the unit-weight sigma-SFF window, while
        # K_inf=P=<sum_n f_n^2> for the Gaussian-filtered complex DSFF.
        ax_left = Axis(figure[1, 1];
            axis_common...,
            xlabel = L"t/t_{\mathrm{Hei}}",
            ylabel = L"\mathrm{\sigma SFF}",
            yticks = magic_log_ticks(-1:1),
        )
        ax_right = Axis(figure[1, 2];
            axis_common...,
            xlabel = L"|\tau|/\tau_{\mathrm{Hei}}",
            yticks = magic_log_ticks(-1:2),
        )
        Label(figure[1, 2, Right()], L"\mathrm{DSFF}";
            rotation = pi / 2,
            fontsize = ax_right.ylabelsize[],
            font = ax_right.ylabelfont[],
            color = ax_right.ylabelcolor[],
        )

        # -- LEFT panel ------------------------------------------------
        eta_lines = Any[]
        for (i, _) in enumerate(ETAS_PLOT)
            boot = sff_bootstrap[i]
            color = eta_colors[i]
            lower = positive_or_nan(boot.lower)
            upper = positive_or_nan(boot.upper)
            valid = @. isfinite(lower) & isfinite(upper) & (upper >= lower)
            lower[.!valid] .= NaN
            upper[.!valid] .= NaN
            band!(ax_left, taus, lower, upper; color = (color, 0.14))
            line = lines!(ax_left, taus, smooth_figure1_curve(boot.curve);
                color = color, linewidth = main_width)
            push!(eta_lines, line)
        end

        goe_line = lines!(ax_left, taus, goe_curve;
            color = :black, linestyle = :dash, linewidth = reference_width)

        target_sff_line = nothing
        if l2_bootstrap !== nothing
            target_lower = positive_or_nan(l2_bootstrap.lower)
            target_upper = positive_or_nan(l2_bootstrap.upper)
            target_valid = @. isfinite(target_lower) & isfinite(target_upper) &
                               (target_upper >= target_lower)
            target_lower[.!target_valid] .= NaN
            target_upper[.!target_valid] .= NaN
            band!(ax_left, taus, target_lower, target_upper;
                color = (TARGET_MODEL_COLOR, 0.14))
            target_sff_line = lines!(
                ax_left, taus, smooth_figure1_curve(l2_bootstrap.curve);
                color = TARGET_MODEL_COLOR, linewidth = reference_width)
        end
        hlines!(ax_left, [1.0]; color = :gray45, linestyle = :dot,
            linewidth = source_px(0.7))

        # -- RIGHT panel -----------------------------------------------
        for (i, eta) in enumerate(ETAS_PLOT)
            key = "eta_" * replace(_fmt(eta), "." => "p")
            curve = complex_dsff.curves[key]
            color = eta_colors[i]
            lower = positive_or_nan(curve.lower)
            upper = positive_or_nan(curve.upper)
            valid = @. isfinite(lower) & isfinite(upper) & (upper >= lower)
            lower[.!valid] .= NaN
            upper[.!valid] .= NaN
            band!(ax_right, curve.x, lower, upper; color = (color, 0.14))
            lines!(ax_right, curve.x, smooth_figure1_curve(curve.curve);
                color = color, linewidth = main_width)
        end

        l3b_curve = complex_dsff.curves["l3b"]
        l3b_lower = positive_or_nan(l3b_curve.lower)
        l3b_upper = positive_or_nan(l3b_curve.upper)
        l3b_valid = @. isfinite(l3b_lower) & isfinite(l3b_upper) &
                        (l3b_upper >= l3b_lower)
        l3b_lower[.!l3b_valid] .= NaN
        l3b_upper[.!l3b_valid] .= NaN
        band!(ax_right, l3b_curve.x, l3b_lower, l3b_upper;
            color = (TARGET_MODEL_COLOR, 0.14))
        dsff_target_line = lines!(
            ax_right, l3b_curve.x, smooth_figure1_curve(l3b_curve.curve);
            color = TARGET_MODEL_COLOR, linewidth = reference_width)

        ai_theory = complex_dsff.theory
        ai_line = lines!(ax_right, ai_theory.display_x,
            smooth_figure1_curve(ai_theory.display_curve);
            color = :black, linestyle = :dash, linewidth = reference_width)
        hlines!(ax_right, [1.0]; color = :gray45, linestyle = :dot,
            linewidth = source_px(0.7))

        xlims!(ax_left, X_LOWER_SFF, TAU_GRID[2])
        xlims!(ax_right, X_LOWER_SFF, TAU_GRID[2])
        ylims!(ax_left, Y_LOWER_SFF, 10.0)
        ylims!(ax_right, Y_LOWER_DSFF, 100.0)

        axislegend(
            ax_left,
            eta_lines,
            Any[L"0.1", L"0.4", L"1", L"2"],
            L"\eta=";
            position = :rt,
            orientation = :horizontal,
            titleposition = :left,
            nbanks = 2,
            framevisible = false,
            patchsize = source_px.((6.0, 2.5)),
            padding = source_px.((0.5, 0.5, 0.5, 0.5)),
            margin = (0, 0, 0, 0),
            rowgap = source_px(0.5),
            colgap = source_px(0.8),
            patchlabelgap = source_px(0.6),
            titlegap = source_px(0.8),
        )
        left_reference_lines = Any[goe_line]
        left_reference_labels = Any["f-GOE"]
        if target_sff_line !== nothing
            push!(left_reference_lines, target_sff_line)
            push!(left_reference_labels, L"\mathrm{diss.\ SYK}")
        end
        axislegend(
            ax_left,
            left_reference_lines,
            left_reference_labels;
            position = (0.98, 0.79),
            orientation = :horizontal,
            nbanks = 1,
            framevisible = false,
            patchsize = source_px.((12.0, 2.5)),
            padding = source_px.((0.5, 0.5, 0.5, 0.5)),
            margin = (0, 0, 0, 0),
            colgap = source_px(2.0),
            patchlabelgap = source_px(1.0),
        )
        axislegend(
            ax_right,
            [ai_line],
            [L"\mathrm{AI}^{\dagger}\ \mathrm{RMT}"];
            position = :rt,
            nbanks = 1,
            framevisible = false,
            patchsize = source_px.((12.0, 2.5)),
            padding = source_px.((0.5, 0.5, 0.5, 0.5)),
            margin = (0, 0, 0, 0),
            patchlabelgap = source_px(1.0),
        )

        # Draw panel labels after the legends so the overlay cannot hide them.
        for (axis, label, label_y) in ((ax_left, L"\mathrm{(a)}", 0.99),
                                       (ax_right, L"\mathrm{(b)}", 0.99))
            text!(axis, 0.012, label_y; text = label, space = :relative,
                align = (:left, :top), color = :black)
        end

        inset = Axis(
            figure[1, 1];
            tellwidth = false,
            tellheight = false,
            width = Relative(0.36),
            height = Relative(0.30),
            halign = 0.74,
            valign = 0.14,
            alignmode = Inside(),
            xscale = log10,
            yscale = log10,
            yaxisposition = :right,
            xticks = magic_log_ticks((-2, 0, 1)),
            yticks = magic_log_ticks((-1, 0, 1)),
            xticklabelsvisible = true,
            yticklabelsvisible = true,
            xticksvisible = true,
            yticksvisible = true,
            xticklabelsize = source_px(5.0),
            yticklabelsize = source_px(5.0),
            xticklabelpad = source_px(0.7),
            yticklabelpad = source_px(0.7),
            xticksize = source_px(2.0),
            yticksize = source_px(2.0),
            xtickwidth = source_px(0.45),
            ytickwidth = source_px(0.45),
            xminorticksvisible = false,
            yminorticksvisible = false,
            xgridvisible = false,
            ygridvisible = false,
            spinewidth = source_px(0.55),
            backgroundcolor = (:white, 0.90),
        )
        translate!(inset.blockscene, 0, 0, 500)
        translate!(inset.scene, 0, 0, 500)
        for (i, entry) in enumerate(finite_size_inset)
            curve = smooth_figure1_curve(entry.curve)
            valid = @. isfinite(taus) & (taus > 0) & isfinite(curve)
            any(valid) || continue
            lines!(inset, taus[valid], curve[valid];
                color = size_colors[i], linewidth = source_px(0.85))
        end
        hlines!(inset, [1.0]; color = :gray45, linestyle = :dot,
            linewidth = source_px(0.55))
        xlims!(inset, X_LOWER_SFF, TAU_GRID[2])
        ylims!(inset, Y_LOWER_SFF, 10.0)

        colgap!(figure.layout, source_px(-14.0))
        colsize!(figure.layout, 1, Relative(0.50))
        colsize!(figure.layout, 2, Relative(0.50))
        figure
    end

    source_size_points = mktempdir() do temporary_dir
        uncropped_pdf = joinpath(temporary_dir, "figure-uncropped.pdf")
        save(uncropped_pdf, fig)
        crop_pdf_hires(uncropped_pdf, pdf_path; margin_pt = PDF_CROP_MARGIN_PT)
    end
    rasterize_pdf_png(pdf_path, png_path)
    println("wrote $pdf_path")
    println("wrote $png_path")

    # ---- Companion manifest ------------------------------------------
    manifest_path = joinpath(cli.output_dir, "figure_1__manifest$(suffix).toml")
    open(manifest_path, "w") do io
        DatasetPaths.print_manifest(io, Dict{String,Any}(
            "config"           => cli.config,
            "cache_dir"        => cli.cache_dir,
            "cache_paths"      => cache_paths,
            "spectra_dir"      => spectra_dir,
            "etas"             => collect(ETAS_PLOT),
            "delta_cd_over_2pi_mhz" => GAMMA,
            "svd_tol"          => baseline_tol,
            "seed_range"       => [first(seeds), last(seeds)],
            "n_seeds"          => length(seeds),
            "n_seeds_used_sff" => n_used_per_eta,
            "n_seeds_used_dsff"=> fill(Int(complex_dsff.parameters["n_seeds"]), length(ETAS_PLOT)),
            "n_orb"            => n_orb,
            "filling"          => filling,
            "hilbert_dim"      => d_hilb,
            "K_liouville"      => K_LIOUV,
            "r_HD_median"      => r_HD_median,
            "r_HD_min"         => minimum(r_HD_pooled),
            "r_HD_max"         => maximum(r_HD_pooled),
            "authoring_canvas_size_points" => collect(AUTHORING_CANVAS_SIZE_PT),
            "source_figure_size_points" => source_size_points,
            "final_figure_size_points_at_columnwidth" =>
                source_size_points .* LATEX_SCALE,
            "intended_latex_scale" => LATEX_SCALE,
            "palette_positions" => FIGURE1_ETA_VIRIDIS_POSITIONS,
            "curve_smoothing" => Dict(
                "filter" => "Savitzky-Golay",
                "domain" => "log10(y)",
                "polynomial_degree" => FIGURE1_SG_DEGREE,
                "window_points" => FIGURE1_SG_WINDOW,
                "applies_to" =>
                    "physical, target-model, inset, and DSFF central curves",
                "uncertainty_ribbons" => "unfiltered",
            ),
            "analysis_window"  => collect(cli.window),
            "hu_n_bins"        => HU_N_BINS,
            "hu_degree"        => HU_DEGREE,
            "tau_grid"         => Dict("min" => TAU_GRID[1],
                                        "max" => TAU_GRID[2],
                                        "n"   => TAU_GRID[3]),
            "complex_dsff"     => Dict(
                "beta" => CANONICAL_DSFF_BETA,
                "theta" => CANONICAL_DSFF_THETA,
                "ray_key" => CANONICAL_DSFF_RAY_KEY,
                "heisenberg_chi" => complex_dsff.theory.chi,
                "ai_dagger_matrix_sizes" =>
                    complex_dsff.theory.matrix_sizes,
                "ai_dagger_seed_counts" =>
                    complex_dsff.theory.seed_counts,
                "plotdata_jld2" => cli.complex_dsff_plotdata,
                "parameters_manifest" => cli.complex_dsff_manifest,
                "n_seeds" => Int(complex_dsff.parameters["n_seeds"]),
                "x_min" => X_LOWER_SFF,
                "x_max" => TAU_GRID[2],
                "y_limits" => collect(complex_dsff.ylimits),
                "omitted_nonpositive" => Dict(
                    key => curve.omitted
                    for (key, curve) in complex_dsff.curves),
            ),
            "steady_tol"       => STEADY_TOL,
            "y_lower_clip_sff" => Y_LOWER_SFF,
            "y_lower_clip_dsff" => Y_LOWER_DSFF,
            "x_lower_clip_sff" => X_LOWER_SFF,
            "finite_size_inset" => Dict(
                "eta" => 2.0,
                "system_sizes" => [entry.n_orb for entry in finite_size_inset],
                "fillings" => [entry.filling for entry in finite_size_inset],
                "K" => [entry.K for entry in finite_size_inset],
                "n_seeds" => [entry.n_seeds for entry in finite_size_inset],
                "cache_paths" => [entry.cache_path for entry in finite_size_inset],
                "source_manifests" => [entry.manifest_path for entry in finite_size_inset],
                "width_relative" => 0.36,
                "height_relative" => 0.30,
                "horizontal_alignment" => 0.74,
                "vertical_alignment" => 0.14,
                "major_xtick_powers" => [-2, 0, 1],
                "major_ytick_powers" => [-1, 0, 1],
                "palette_rgb" => [
                    [0.72, 0.72, 0.72],
                    [0.46, 0.46, 0.46],
                    [0.20, 0.20, 0.20],
                ],
                "analysis_window" => collect(cli.window),
            ),
            "sigma_sff_bootstrap" => Dict(
                "n_bootstrap" => SFF_N_BOOTSTRAP,
                "interval" => 0.68,
                "quantiles" => [0.16, 0.84],
                "rng_seed_base" => RNG_SEED_SFF_BOOTSTRAP,
            ),
            "folded_goe"       => Dict(
            ),
            "l2_overlay"       => l2_manifest,
            "outputs"          => [pdf_path, png_path],
        ); path=manifest_path)
    end
    println("wrote $manifest_path")

    return (pdf = pdf_path, png = png_path, manifest = manifest_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
