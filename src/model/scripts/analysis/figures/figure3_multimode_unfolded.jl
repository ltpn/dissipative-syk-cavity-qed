#!/usr/bin/env julia
# figure3_multimode_unfolded.jl
#
# Final production Figure 3 for the Lamb-Dicke chaos manuscript.
#
# Hamiltonian-unfolded sigma-SFF of the multimode open-system Liouvillian
# (cavity-loss-only dissipators, SI Eq. S23) for four values of the
# effective disorder strength delta tilde{omega} at fixed
#   Delta_cd/2pi = 0.2 MHz, N_orb = 10, filling = 3, n_grid = 160,
#   mode_cutoff = 300,
# The CLI retains support for smaller diagnostic systems and parameter subsets.
#
# Data flow
#   1. For each (delta_tilde, seed) build the fixed-filling Liouvillian via
#      `multimode_open_system_block` and cache its singular values to a
#      per-parameter JLD2 (skipped if the cache already exists).
#   2. Apply Hamiltonian unfolding (polynomial-5 staircase) on the
#      requested bulk window (canonical: [0.05, 0.95]) and compute the
#      R/(R-1)-unbiased
#      connected sigma-SFF via `kunf_ham`.
#   3. Render a compact two-panel MagicLaTeX figure with a fixed,
#      magma-derived delta tilde{omega} palette and top-panel smoothing.

if Sys.isapple()
    @eval using AppleAccelerate
end

using CairoMakie
using JLD2
using LaTeXStrings
using LinearAlgebra
using Printf
using Random
using Statistics: mean, median
using TOML
include(joinpath(@__DIR__, "..", "..", "..", "DatasetPaths.jl"))

const HERE = @__DIR__
include(joinpath(HERE, "_figure_export.jl"))
const LAMB_DIR = abspath(joinpath(HERE, "..", "..", ".."))         # src/model
const REPO_ROOT = abspath(joinpath(LAMB_DIR, "..", ".."))          # repo root
const DATA_ROOT = joinpath(REPO_ROOT, "data")
const FIGURE_ROOT = joinpath(REPO_ROOT, "figures")

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
include(joinpath(LAMB_DIR, "IntegrableCorner.jl"))

include(joinpath(HERE, "_common_sff_helpers.jl"))
include(joinpath(HERE, "MagicLaTeX.jl"))
include(joinpath(HERE, "_figure1_plot_smoothing.jl"))
include(joinpath(HERE, "_figure3_layout.jl"))
include(joinpath(HERE, "_figure3_syk4_reference.jl"))

using .MagicLaTeX
using .Figure1PlotSmoothing: sgolay_log_positive
using .Figure3SYK4Reference: load_syk4_reference
using .IntegrableCorner: folded_goe_form_factor

# Small helper reused from `ComplexDSFFPlotData.positive_or_nan`: mask
# non-positive / non-finite values so log-scale `lines!` / `band!`
# renders cleanly (matches the fig 1 usage).
positive_or_nan(values) = Float64[
    isfinite(v) && v > 0 ? v : NaN for v in values]

# post-filter level count across all seeds.  multimode sigmas at
# K=14400 occasionally drop one value below `epsilon_zero`, so seeds
# split between K=14400 and K=14399 (bulk length 12961 vs 12960).  This
# preprocessing step filters each seed and truncates them all to the
# minimum retained length so the strict check passes without biasing
# the estimator (identical to what `sigma_sff_traces` would have done
# had it enforced a common K itself).
function _equalize_sigma_counts(sigmas_by_seed, epsilon_zero)
    filts = [sort!(filter(s -> s > epsilon_zero, Vector{Float64}(sig)))
             for sig in sigmas_by_seed]
    isempty(filts) && return filts
    K_min = minimum(length, filts)
    return [filts[i][1:K_min] for i in eachindex(filts)]
end

const DEFAULT_OUTPUT_DIR = joinpath(FIGURE_ROOT)
const DEFAULT_CACHE_DIR  = joinpath(DATA_ROOT, "multimode", "n10_f3")

# ------------------------------------------------------------------
# Physics constants for this figure
# ------------------------------------------------------------------
const DEFAULT_DELTA_TILDES_PLOT = (0.01, 0.1, 1.0, 10.0)   # canonical
# Overwritten by the `--delta-tildes` CLI flag in main() so short debug
# canaries can restrict the loop to a subset.
DELTA_TILDES_PLOT = DEFAULT_DELTA_TILDES_PLOT
N_ORB::Int              = 10
FILLING::Union{Int,Nothing} = nothing
const N_GRID            = 160
const BOX_LENGTH        = 15.0
const ZETA              = 1.0
const MODE_CUTOFF       = 300
const WEIGHT_TYPE       = :speckle
const SPECKLE_GRAINS    = 17.0
const DISORDER_STRENGTH = 1.0
const DRIVE_WAVEVECTOR  = 0.0
const ENERGY_SCALE      = 1.0

const KAPPA_OVER_2PI_MHZ    = 0.2
# multimode figure can be produced at the Table-S1 multimode value
# (0.2 MHz, i.e. kappa/Delta = 1) or at the single-mode spontaneous-emission
# value (1 MHz, i.e. kappa/Delta = 0.2).  Delta enters ONLY through
# kappa/Delta_cd, which sets the dissipator strength: the Hamiltonian is built
# in units lambda^2/Delta_cd = 1 and the mode weights depend on delta_tilde
# alone, so two runs differing only in `--delta-cd` share the same coherent
# operator and differ purely in the dissipation-to-interaction ratio.
# The value is part of every cache filename, so runs do not collide.
DELTA_CD_OVER_2PI_MHZ::Float64 = 1.0
const GAMMA_OVER_2PI_MHZ    = 5.9
const DELTA_DA_OVER_2PI_MHZ = 3000.0
const RATE_SCALE            = 1.0

const DEFAULT_N_SEEDS = 64
const EPSILON_ZERO    = 1.0e-8
const DEFAULT_ANALYSIS_WINDOW = (0.05, 0.95)
const TAU_GRID        = (1e-3, 10.0, 400)
# with `--with-dynamics`.  Matches the production `[numerics.time_grid]`
# defaults so the multimode dynamics panel is directly comparable to
# figure 2 (physical + L3b) on the same log-t range.
const TIME_GRID_DYN   = (1.0e-2, 1.0e6, 540)     # (t_min, t_max, n)
const DYNAMICS_ENTROPY_EIG_TOL = 1.0e-12
const Y_LOWER_SFF     = 10.0^(-1.5)              # lower y-clip for the sigma-SFF panel
const X_LOWER_SFF     = 10.0^(-2.5)              # lower x-clip for the sigma-SFF panel
const HU_N_BINS       = 40                       # knot bins for the Hamiltonian-unfolding staircase
const HU_DEGREE       = 5                        # requested polynomial degree

# RMT reference ensembles (symmetry-class-matched, per Kawabata et al.
# arXiv:2307.08218).  Class BDI-dagger.  The random-Lindbladian coherent/
# dissipative Frobenius ratio is set to the median r_HD of the physical
# ensemble measured at run time.  K_LIOUV and D_HILBERT are overwritten
# from N_ORB in main().
K_LIOUV::Int         = 400                # = binomial(6, 3)^2  (Liouvillian dim)
D_HILBERT::Int       = 20                 # = binomial(6, 3)
# runs (K = 14400 at N=10 f=3 costs ~15 min per real-symmetric eigen so
# window).
# in fig 3 (analogous to `RNG_SEED_SFF_BOOTSTRAP` in fig 1).  A single
# fixed seed here means every re-render produces bit-identical bands.
const RNG_SEED_SFF_BOOTSTRAP = 20260804
const N_BOOT_SFF     = 500

# Canonical single-column presentation.  The authoring canvas is placed at
# 50% in the manuscript, matching Figures 1 and 2.
const FIGURE3_LATEX_SCALE = 0.5
const FIGURE3_AUTHORING_CANVAS_SIZE_PT = (601.0, 283.0)
const PDF_CROP_MARGIN_PT = 1.0
const FIGURE3_LEGEND_LABEL_PT = 7.2
const FIGURE3_SG_WINDOW = 11
const FIGURE3_SG_DEGREE = 3
const FIGURE3_CURVE_COLOR_RGB = [
    [0.9968296, 0.7724726, 0.5373508],
    [0.9666100, 0.4361110, 0.3596695],
    [0.7458284, 0.2254362, 0.4656836],
    [0.4555515, 0.1268050, 0.5073730],
]
const FIGURE3_CURVE_COLORS = [RGBf(rgb...) for rgb in FIGURE3_CURVE_COLOR_RGB]
const FIGURE3_INSET_GREYS = [
    RGBf(0.72, 0.72, 0.72),
    RGBf(0.46, 0.46, 0.46),
    RGBf(0.20, 0.20, 0.20),
]
const FIGURE3_DYNAMICS_YLABEL_DOWNSHIFT_PT = 2.75
const FIGURE3_TARGET_MODEL_COLOR_RGB = [0.05, 0.34, 0.20]
const FIGURE3_TARGET_MODEL_COLOR = RGBf(FIGURE3_TARGET_MODEL_COLOR_RGB...)
const FIGURE3_INSET_HALIGN = 0.69

figure3_source_px(points::Real) =
    MagicLaTeX.pt2px(points / FIGURE3_LATEX_SCALE)

function figure3_smooth_curve(values)
    return sgolay_log_positive(
        positive_or_nan(values);
        window = FIGURE3_SG_WINDOW,
        degree = FIGURE3_SG_DEGREE,
    )
end

function figure3_curve_colors(n::Integer)
    n >= 1 || throw(ArgumentError(
        "Figure 3 requires at least one parameter curve"))
    n == length(FIGURE3_CURVE_COLORS) && return copy(FIGURE3_CURVE_COLORS)
    ramp = cgrad(FIGURE3_CURVE_COLORS)
    return n == 1 ? [ramp[0.5]] :
        [ramp[position] for position in range(0.0, 1.0; length = n)]
end

log_range(lo, hi, n) = exp.(range(log(Float64(lo)), log(Float64(hi)); length = Int(n)))

# ------------------------------------------------------------------
# multimode model constructors and per-parameter cache
# ------------------------------------------------------------------
function build_params(delta_tilde)
    return MultimodeParams(
        n_orb = N_ORB,
        filling = FILLING,
        box_length = BOX_LENGTH,
        n_grid = N_GRID,
        zeta = ZETA,
        mode_cutoff = MODE_CUTOFF,
        delta_tilde = Float64(delta_tilde),
        weight_type = WEIGHT_TYPE,
        speckle_grains_per_side = SPECKLE_GRAINS,
        speckle_correlation_length = nothing,
        disorder_strength = DISORDER_STRENGTH,
        drive_wavevector = DRIVE_WAVEVECTOR,
        energy_scale = ENERGY_SCALE,
    )
end

function build_loss()
    return MultimodeCavityLossParams(
        kappa_over_2pi_mhz = KAPPA_OVER_2PI_MHZ,
        delta_cd_over_2pi_mhz = DELTA_CD_OVER_2PI_MHZ,
        gamma_over_2pi_mhz = GAMMA_OVER_2PI_MHZ,
        delta_da_over_2pi_mhz = DELTA_DA_OVER_2PI_MHZ,
        rate_scale = RATE_SCALE,
    )
end

_dtilde_slug(dt) = replace(@sprintf("%.6g", dt), "." => "p", "-" => "m")

"""
Cache path for one (delta_tilde, shard) tuple.  When `shard_count == 1`
this reduces to the canonical `multimode_centered__...` filename used by
the plotting mode of `figure3_multimode_unfolded.jl`.
"""
function cache_path(cache_dir, delta_tilde;
                    shard_index::Integer = 1, shard_count::Integer = 1)
    filling_resolved = FILLING === nothing ? div(N_ORB, 2) : Int(FILLING)
    f_tag = @sprintf("__f=%d", filling_resolved)
    if shard_count > 1
        slug = @sprintf("multimode_shard__norb=%d%s__ngrid=%d__M=%d__deltacd=%s__dtilde=%s__shard=%03d_of_%03d.jld2",
                        N_ORB, f_tag, N_GRID, MODE_CUTOFF,
                        _dtilde_slug(DELTA_CD_OVER_2PI_MHZ),
                        _dtilde_slug(delta_tilde),
                        shard_index, shard_count)
    else
        slug = @sprintf("multimode_centered__norb=%d%s__ngrid=%d__M=%d__deltacd=%s__dtilde=%s.jld2",
                        N_ORB, f_tag, N_GRID, MODE_CUTOFF,
                        _dtilde_slug(DELTA_CD_OVER_2PI_MHZ),
                        _dtilde_slug(delta_tilde))
    end
    return joinpath(cache_dir, slug)
end

"""Return the 1-based seed list assigned to shard `(shard_index, shard_count)`."""
function shard_seed_range(n_seeds::Integer, shard_index::Integer, shard_count::Integer)
    return [s for s in 1:Int(n_seeds) if mod(s - 1, Int(shard_count)) + 1 == Int(shard_index)]
end

"""Validate that one complex spectrum is stored for every listed seed."""
function validate_seed_spectra(seeds::AbstractVector{<:Integer}, spectra;
                               expected_dim::Union{Nothing,Integer} = nothing)
    length(spectra) == length(seeds) || throw(ArgumentError(
        "L_eigvals/seed count mismatch: $(length(spectra)) spectra for $(length(seeds)) seeds"))
    length(unique(Int.(seeds))) == length(seeds) ||
        throw(ArgumentError("duplicate seed identifiers in Figure 3 cache"))
    for (seed, values) in zip(seeds, spectra)
        vals = ComplexF64.(values)
        expected_dim === nothing || length(vals) == Int(expected_dim) ||
            throw(ArgumentError(
                "seed $seed has $(length(vals)) eigenvalues; expected $(Int(expected_dim))"))
        all(z -> isfinite(real(z)) && isfinite(imag(z)), vals) ||
            throw(ArgumentError("seed $seed contains non-finite L_eigvals"))
    end
    return nothing
end

"""Load optional seed-resolved `L_eigvals` and enforce seed alignment."""
function load_optional_L_eigvals(file, seeds::AbstractVector{<:Integer};
                                 expected_dim::Union{Nothing,Integer} = nothing)
    haskey(file, "L_eigvals") || return nothing
    spectra = [ComplexF64.(values) for values in file["L_eigvals"]]
    validate_seed_spectra(seeds, spectra; expected_dim = expected_dim)
    return spectra
end

"""
Merge per-shard JLD2 cache files for one `delta_tilde` into the canonical
combined cache used by the plotting driver.  Reads
`multimode_shard__...__shard=NN_of_MM.jld2` for `NN = 1..MM`, concatenates
seeds in the natural (global) order, and writes
`multimode_centered__...__dtilde=X.jld2`.
"""
function merge_shard_caches(cache_dir::AbstractString, delta_tilde::Real,
                             shard_count::Integer, n_seeds::Integer;
                             require_L_eigvals::Bool = false)
    target = cache_path(cache_dir, delta_tilde)
    seed_to_sigmas       = Dict{Int,Vector{Float64}}()
    seed_to_mu_re        = Dict{Int,Float64}()
    seed_to_mu_im        = Dict{Int,Float64}()
    seed_to_r_HD         = Dict{Int,Float64}()
    seed_to_n_jumps      = Dict{Int,Int}()
    seed_to_jump_norm_sq = Dict{Int,Vector{Float64}}()
    seed_to_L_eigvals    = Dict{Int,Vector{ComplexF64}}()
    # `entropy_t` + `populations_t` + a shared `times` grid, the merger
    # propagates them into the merged cache; otherwise they are dropped
    # and the fig 3 plot pass renders the sigma-SFF panel only.
    seed_to_entropy   = Dict{Int,Vector{Float64}}()
    seed_to_pops      = Dict{Int,Matrix{Float64}}()
    seed_to_trace_res = Dict{Int,Float64}()
    seed_to_part_res  = Dict{Int,Float64}()
    dyn_times_ref::Union{Nothing,Vector{Float64}} = nothing
    for idx in 1:Int(shard_count)
        p = cache_path(cache_dir, delta_tilde; shard_index = idx, shard_count = Int(shard_count))
        isfile(p) || continue
        JLD2.jldopen(p, "r") do f
            local_seeds = Int.(f["seeds"])
            local_sigs  = f["sigmas"]
            local_mu_re = Float64.(f["mu_re"])
            local_mu_im = Float64.(f["mu_im"])
            local_rHD   = Float64.(f["r_HD"])
            local_nj    = Int.(f["n_jumps"])
            local_jns = f["jump_norm_sq"]
            local_eigs = load_optional_L_eigvals(f, local_seeds;
                                                 expected_dim = K_LIOUV)
            # Dynamics arrays are optional (only present when the shard
            # was built with `--with-dynamics`).  All shards for the
            # same delta_tilde must share the same `times` grid.
            local_ent   = haskey(f, "entropy_t")     ? f["entropy_t"]     : nothing
            local_pops  = haskey(f, "populations_t") ? f["populations_t"] : nothing
            local_tres  = haskey(f, "dynamics_trace_residual_max") ? Float64.(f["dynamics_trace_residual_max"]) : nothing
            local_pres  = haskey(f, "dynamics_particle_residual_max") ? Float64.(f["dynamics_particle_residual_max"]) : nothing
            if haskey(f, "times")
                local_times = Vector{Float64}(f["times"])
                if dyn_times_ref === nothing
                    dyn_times_ref = local_times
                elseif length(dyn_times_ref) != length(local_times) ||
                       maximum(abs.(dyn_times_ref .- local_times)) > 1e-12
                    error("merge_shard_caches: shard $idx dt=$delta_tilde uses a different dynamics time grid; refusing to merge")
                end
            end
            for (i, s) in enumerate(local_seeds)
                seed_to_sigmas[s]  = Vector{Float64}(local_sigs[i])
                seed_to_mu_re[s]   = local_mu_re[i]
                seed_to_mu_im[s]   = local_mu_im[i]
                seed_to_r_HD[s]    = local_rHD[i]
                seed_to_n_jumps[s] = local_nj[i]
                seed_to_jump_norm_sq[s] = Vector{Float64}(local_jns[i])
                if local_eigs !== nothing
                    seed_to_L_eigvals[s] = local_eigs[i]
                end
                if local_ent !== nothing && local_pops !== nothing
                    seed_to_entropy[s] = Vector{Float64}(local_ent[i])
                    seed_to_pops[s]    = Matrix{Float64}(local_pops[i])
                end
                if local_tres  !== nothing; seed_to_trace_res[s] = local_tres[i];  end
                if local_pres  !== nothing; seed_to_part_res[s]  = local_pres[i];  end
            end
        end
    end
    sorted_seeds = sort(collect(keys(seed_to_sigmas)))
    expected_seeds = collect(1:Int(n_seeds))
    sorted_seeds == expected_seeds || error(
        "merge_shard_caches: expected seeds 1:$(n_seeds) for dt=$delta_tilde, " *
        "found $(sorted_seeds); refusing to replace canonical cache")
    if require_L_eigvals &&
       !all(haskey(seed_to_L_eigvals, seed) for seed in expected_seeds)
        error("merge_shard_caches: complete L_eigvals required for dt=$delta_tilde; " *
              "refusing to replace canonical cache")
    end
    sigmas   = [seed_to_sigmas[s]  for s in sorted_seeds]
    mu_re    = [seed_to_mu_re[s]   for s in sorted_seeds]
    mu_im    = [seed_to_mu_im[s]   for s in sorted_seeds]
    r_HD     = [seed_to_r_HD[s]    for s in sorted_seeds]
    n_jumps  = [seed_to_n_jumps[s] for s in sorted_seeds]
    jump_norm_sq = [seed_to_jump_norm_sq[s] for s in sorted_seeds]
    have_L_eigvals = !isempty(sorted_seeds) &&
        all(haskey(seed_to_L_eigvals, s) for s in sorted_seeds)
    L_eigvals = have_L_eigvals ?
        [seed_to_L_eigvals[s] for s in sorted_seeds] : nothing
    # Same all-or-nothing rule for the dynamics arrays.
    have_dyn = dyn_times_ref !== nothing &&
        !isempty(seed_to_entropy) && !isempty(seed_to_pops) &&
        all(haskey(seed_to_entropy, s) && haskey(seed_to_pops, s) for s in sorted_seeds)
    tmp = target * ".tmp"
    JLD2.jldopen(tmp, "w") do f
        f["sigmas"]  = sigmas
        f["mu_re"]   = mu_re
        f["mu_im"]   = mu_im
        f["r_HD"]    = r_HD
        f["n_jumps"] = n_jumps
        f["jump_norm_sq"] = jump_norm_sq
        if L_eigvals !== nothing
            f["L_eigvals"] = L_eigvals
        end
        f["seeds"]                 = sorted_seeds
        f["delta_tilde"]           = Float64(delta_tilde)
        f["delta_cd_over_2pi_mhz"] = DELTA_CD_OVER_2PI_MHZ
        f["n_orb"]                 = N_ORB
        f["n_grid"]                = N_GRID
        f["mode_cutoff"]           = MODE_CUTOFF
        f["n_seeds"]               = length(sigmas)
        if have_dyn
            f["times"]         = dyn_times_ref
            f["entropy_t"]     = [seed_to_entropy[s] for s in sorted_seeds]
            f["populations_t"] = [seed_to_pops[s]    for s in sorted_seeds]
            if all(haskey(seed_to_trace_res, s) for s in sorted_seeds)
                f["dynamics_trace_residual_max"]    = [seed_to_trace_res[s] for s in sorted_seeds]
            end
            if all(haskey(seed_to_part_res,  s) for s in sorted_seeds)
                f["dynamics_particle_residual_max"] = [seed_to_part_res[s]  for s in sorted_seeds]
            end
        end
    end
    mv(tmp, target; force = true)
    return (path = target, n_seeds = length(sigmas),
            n_eigval_seeds = L_eigvals === nothing ? 0 : length(L_eigvals))
end

"""
Load (or build + persist) the per-seed trace-centered singular values of
the multimode open-system Liouvillian at `delta_tilde`.  Also returns
the per-seed complex trace shift `mu = tr(L)/D` and the per-seed
coherent-to-dissipative Frobenius ratio `r_HD = ||L_H||_F / ||L_D||_F`,
both used downstream to (a) match the random-Lindbladian reference
and (b) document the RMT centering in the manifest.

When `shard_count > 1`, only the seeds assigned to `shard_index` are
built and persisted to a per-shard cache file (see `cache_path`).  Callers
running the plotting path should first merge shard caches with
`merge_shard_caches`.

**Timeout resilience**: after every successfully built seed, the current
state of all arrays is flushed to the target JLD2 file atomically (via
`.tmp` + `mv`).  This makes the builder resumable at the granularity of
a single seed; on SIGTERM the next invocation picks up from the last
completed seed.
"""
function collect_or_build_singulars(delta_tilde, n_seeds, cache_dir;
                                     shard_index::Integer = 1,
                                     shard_count::Integer = 1,
                                     allow_build::Bool = false,
                                     with_dynamics::Bool = false,
                                     dyn_times::Union{Nothing,AbstractVector} = nothing)
    mkpath(cache_dir)
    path = cache_path(cache_dir, delta_tilde;
                      shard_index = shard_index, shard_count = shard_count)
    my_seeds = shard_count > 1 ?
        shard_seed_range(n_seeds, shard_index, shard_count) :
        collect(1:Int(n_seeds))
    sigmas   = Vector{Vector{Float64}}()
    mu_re    = Float64[]
    mu_im    = Float64[]
    r_HD     = Float64[]
    n_jumps  = Int[]
    # Squared Frobenius norms of the cavity-loss jump operators per seed.
    jump_norm_sq = Vector{Vector{Float64}}()
    L_eigvals_by_seed = Dict{Int,Vector{ComplexF64}}()
    # `dynamics_from_eigen` and store the per-seed dynamics arrays
    # (entropy_t, populations_t) alongside the sigma-SFF payload.  One
    # eigen(L) per seed at K = 14400 costs ~30 min on top of the
    # existing ~25 min svdvals(Lc).  `dyn_times` is a shared time grid
    # (all seeds must use it identically).  Empty when not enabled.
    dyn_entropy      = Vector{Vector{Float64}}()
    dyn_populations  = Vector{Matrix{Float64}}()
    dyn_trace_res    = Float64[]
    dyn_particle_res = Float64[]
    seeds_done = Int[]
    if isfile(path)
        JLD2.jldopen(path, "r") do f
            for arr in f["sigmas"]; push!(sigmas, Vector{Float64}(arr)); end
            append!(mu_re,   Vector{Float64}(f["mu_re"]))
            append!(mu_im,   Vector{Float64}(f["mu_im"]))
            append!(r_HD,    Vector{Float64}(f["r_HD"]))
            append!(n_jumps, Vector{Int}(f["n_jumps"]))
            append!(seeds_done, Int.(f["seeds"]))
            if !isempty(seeds_done)
                cached_eigs = load_optional_L_eigvals(f, seeds_done;
                                                      expected_dim = K_LIOUV)
                if cached_eigs !== nothing
                    for (seed, values) in zip(seeds_done, cached_eigs)
                        L_eigvals_by_seed[seed] = values
                    end
                end
            end
            for arr in f["jump_norm_sq"]; push!(jump_norm_sq, Vector{Float64}(arr)); end
            haskey(f, "entropy_t") &&
                for arr in f["entropy_t"]; push!(dyn_entropy, Vector{Float64}(arr)); end
            haskey(f, "populations_t") &&
                for arr in f["populations_t"]; push!(dyn_populations, Matrix{Float64}(arr)); end
            haskey(f, "dynamics_trace_residual_max") && append!(dyn_trace_res,    Vector{Float64}(f["dynamics_trace_residual_max"]))
            haskey(f, "dynamics_particle_residual_max") && append!(dyn_particle_res, Vector{Float64}(f["dynamics_particle_residual_max"]))
        end
    end
    already_done = Set{Int}(seeds_done)
    pending_seeds = [s for s in my_seeds if !(s in already_done)]
    times_vec = with_dynamics ?
        (dyn_times === nothing ?
            log_range(TIME_GRID_DYN[1], TIME_GRID_DYN[2], TIME_GRID_DYN[3]) :
            collect(Float64, dyn_times)) :
        Float64[]
    if isempty(pending_seeds)
        return (sigmas = sigmas, mu_re = mu_re, mu_im = mu_im,
                r_HD = r_HD, n_jumps = n_jumps,
                jump_norm_sq = jump_norm_sq,
                dyn_entropy = dyn_entropy,
                dyn_populations = dyn_populations,
                dyn_trace_res = dyn_trace_res,
                dyn_particle_res = dyn_particle_res,
                times = times_vec,
                L_eigvals = all(haskey(L_eigvals_by_seed, s) for s in seeds_done) ?
                    [L_eigvals_by_seed[s] for s in seeds_done] : Vector{Vector{ComplexF64}}(),
                path = path, seeds = seeds_done)
    end
    allow_build || error("Missing saved multimode spectra for delta_tilde=$delta_tilde seeds=$pending_seeds. Run the staged pipeline and pass its multimode export directory via --cache-dir.")
    p    = build_params(delta_tilde)
    loss = build_loss()
    for seed in pending_seeds
        t0 = time()
        result = multimode_open_system_block(p, loss; seed = seed)
        Lmat = Matrix(result.L)
        Hmat = Matrix(result.H)
        Lc, mu = trace_center_superoperator(Lmat)
        s = sort(svdvals(Lc))
        # Jump-mode statistics: squared Frobenius norm of every cavity-loss
        # ~30 s per seed rebuild during the plot pass.
        w = Float64[norm(Matrix(J))^2 for J in result.jump_operators]
        # Optional dynamics compute: one eigen(L) whose vectors feed the
        # dense propagation via `dynamics_from_eigen`.  No second
        # diagonalization is needed for the sigma-SFF (which uses
        # svdvals(Lc), not the eigenvalues).
        dyn_result = nothing
        if with_dynamics
            F = eigen(Lmat)
            L_eigvals_by_seed[seed] = ComplexF64.(F.values)
            dyn_result = dynamics_from_eigen(F.values, F.vectors,
                                              result.basis_states, times_vec;
                                              n_orb = Int(result.metadata["n_orb"]),
                                              entropy_eig_tol = DYNAMICS_ENTROPY_EIG_TOL)
        end
        push!(sigmas,  s)
        push!(mu_re,   real(mu))
        push!(mu_im,   imag(mu))
        push!(r_HD,    coherent_to_dissipative_norm_ratio(Hmat, Lmat))
        push!(n_jumps, Int(result.metadata["n_cavity_loss_jumps"]))
        push!(jump_norm_sq, w)
        if dyn_result !== nothing
            push!(dyn_entropy,      copy(dyn_result.entropy_t))
            push!(dyn_populations,  copy(dyn_result.populations_t))
            push!(dyn_trace_res,    Float64(dyn_result.metadata["trace_residual_max"]))
            push!(dyn_particle_res, Float64(dyn_result.metadata["particle_residual_max"]))
        end
        push!(seeds_done, seed)
        # Incremental atomic flush after each seed so timeout doesn't lose work.
        tmp = path * ".tmp"
        JLD2.jldopen(tmp, "w") do f
            f["sigmas"]                = sigmas
            f["mu_re"]                 = mu_re
            f["mu_im"]                 = mu_im
            f["r_HD"]                  = r_HD
            f["n_jumps"]               = n_jumps
            f["jump_norm_sq"]          = jump_norm_sq
            f["seeds"]                 = seeds_done
            f["delta_tilde"]           = Float64(delta_tilde)
            f["delta_cd_over_2pi_mhz"] = DELTA_CD_OVER_2PI_MHZ
            f["n_orb"]                 = N_ORB
            f["n_grid"]                = N_GRID
            f["mode_cutoff"]           = MODE_CUTOFF
            f["n_seeds"]               = length(sigmas)
            f["shard_index"]           = Int(shard_index)
            f["shard_count"]           = Int(shard_count)
            if all(haskey(L_eigvals_by_seed, s) for s in seeds_done)
                spectra = [L_eigvals_by_seed[s] for s in seeds_done]
                validate_seed_spectra(seeds_done, spectra; expected_dim = K_LIOUV)
                f["L_eigvals"] = spectra
            end
            if with_dynamics && !isempty(dyn_entropy)
                f["times"]                          = times_vec
                f["entropy_t"]                      = dyn_entropy
                f["populations_t"]                  = dyn_populations
                f["dynamics_trace_residual_max"]    = dyn_trace_res
                f["dynamics_particle_residual_max"] = dyn_particle_res
            end
        end
        mv(tmp, path; force = true)
        if with_dynamics
            @printf("    delta_tilde=%-7g seed=%3d  K=%d  |mu|=%.3g  r_HD=%.3g  M=%d  trace_res=%.2g  %.1fs  (flushed %d/%d)\n",
                    delta_tilde, seed, length(s), abs(mu), r_HD[end], length(w),
                    dyn_trace_res[end],
                    time() - t0,
                    length(seeds_done), length(my_seeds))
        else
            @printf("    delta_tilde=%-7g seed=%3d  K=%d  |mu|=%.3g  r_HD=%.3g  M=%d  %.1fs  (flushed %d/%d)\n",
                    delta_tilde, seed, length(s), abs(mu), r_HD[end], length(w),
                    time() - t0,
                    length(seeds_done), length(my_seeds))
        end
        flush(stdout)
    end
    return (sigmas = sigmas, mu_re = mu_re, mu_im = mu_im,
            r_HD = r_HD, n_jumps = n_jumps,
            jump_norm_sq = jump_norm_sq,
            dyn_entropy = dyn_entropy,
            dyn_populations = dyn_populations,
            dyn_trace_res = dyn_trace_res,
            dyn_particle_res = dyn_particle_res,
            times = times_vec,
            L_eigvals = all(haskey(L_eigvals_by_seed, s) for s in seeds_done) ?
                [L_eigvals_by_seed[s] for s in seeds_done] : Vector{Vector{ComplexF64}}(),
            path = path, seeds = seeds_done)
end

function _rasterize_pdf_288(input_path::AbstractString,
                            output_base::AbstractString)
    pdftocairo = Sys.which("pdftocairo")
    if pdftocairo !== nothing
        run(Cmd([pdftocairo, "-png", "-singlefile", "-r", "288",
                 input_path, output_base]))
    else
        gs = Sys.which("gs")
        gs === nothing && error("PNG export requires Ghostscript (`gs`)")
        run(Cmd([gs, "-q", "-dNOPAUSE", "-dBATCH", "-sDEVICE=pngalpha",
                 "-r288", "-sOutputFile=$(output_base).png", input_path]))
    end
    return output_base * ".png"
end

# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
function _parse_cli(argv)
    output_dir = DEFAULT_OUTPUT_DIR
    cache_dir  = DEFAULT_CACHE_DIR
    n_seeds    = DEFAULT_N_SEEDS
    window     = DEFAULT_ANALYSIS_WINDOW
    n_orb      = N_ORB                           # allow override via --n-orb
    filling::Union{Nothing,Int} = nothing        # nothing => half-filling default
    shard_index::Int = 1
    shard_count::Int = 1
    build_cache_only = false
    delta_tildes = collect(Float64, DEFAULT_DELTA_TILDES_PLOT)
    delta_cd::Float64 = DELTA_CD_OVER_2PI_MHZ   # allow override via --delta-cd
    # that also computes eigen(L) + dense propagation (`dynamics_from_eigen`)
    # and saves `times`, `entropy_t`, `populations_t` alongside the
    # sigma-SFF payload.  Cost: one `eigen(L)` per seed (~30 min at
    # K = 14400) in addition to the existing `svdvals(Lc)`.
    with_dynamics::Bool = false
    # `--inset-comparison-cache <N>:<path>` (repeatable) is a merged
    # cache built for a *different* system size at the same
    # delta_tilde selected via `--inset-comparison-delta-tilde`
    # (defaults to 0.01).  Rendered as an overlay of kunf_ham curves
    # inside the top panel, labeled by N.
    inset_comparison_caches::Vector{Tuple{Int,String}} = Tuple{Int,String}[]
    inset_comparison_delta_tilde::Float64 = 0.01
    syk4_reference_config = ""
    syk4_reference_spectra_dir = ""
    syk4_reference_validation = ""
    syk4_reference_eta::Float64 = 2.0
    syk4_reference_seeds = Int[]
    i = 1
    while i <= length(argv)
        a = argv[i]
        if a == "--output-dir"; i += 1; output_dir = argv[i]
        elseif startswith(a, "--output-dir="); output_dir = split(a, "=", limit = 2)[2]
        elseif a == "--cache-dir"; i += 1; cache_dir = argv[i]
        elseif startswith(a, "--cache-dir="); cache_dir = split(a, "=", limit = 2)[2]
        elseif a == "--n-seeds"; i += 1; n_seeds = parse(Int, argv[i])
        elseif startswith(a, "--n-seeds="); n_seeds = parse(Int, split(a, "=", limit = 2)[2])
        elseif a == "--analysis-window"; i += 1; window = _parse_window(argv[i])
        elseif startswith(a, "--analysis-window="); window = _parse_window(split(a, "=", limit = 2)[2])
        elseif a == "--n-orb"; i += 1; n_orb = parse(Int, argv[i])
        elseif startswith(a, "--n-orb="); n_orb = parse(Int, split(a, "=", limit = 2)[2])
        elseif a == "--filling"; i += 1; filling = parse(Int, argv[i])
        elseif startswith(a, "--filling="); filling = parse(Int, split(a, "=", limit = 2)[2])
        elseif a == "--shard-index"; i += 1; shard_index = parse(Int, argv[i])
        elseif startswith(a, "--shard-index="); shard_index = parse(Int, split(a, "=", limit = 2)[2])
        elseif a == "--shard-count"; i += 1; shard_count = parse(Int, argv[i])
        elseif startswith(a, "--shard-count="); shard_count = parse(Int, split(a, "=", limit = 2)[2])
        elseif a == "--build-cache-only"; build_cache_only = true
            build_cache_only = true
        elseif a == "--delta-tildes"; i += 1; delta_tildes = _parse_delta_tildes(argv[i])
        elseif startswith(a, "--delta-tildes="); delta_tildes = _parse_delta_tildes(split(a, "=", limit = 2)[2])
        elseif a == "--delta-cd"; i += 1; delta_cd = parse(Float64, argv[i])
        elseif startswith(a, "--delta-cd="); delta_cd = parse(Float64, split(a, "=", limit = 2)[2])
        elseif a == "--with-dynamics"; with_dynamics = true
        elseif a == "--inset-comparison-cache"
            i += 1
            push!(inset_comparison_caches, _parse_inset_comparison_cache(argv[i]))
        elseif startswith(a, "--inset-comparison-cache=")
            push!(inset_comparison_caches,
                  _parse_inset_comparison_cache(split(a, "=", limit = 2)[2]))
        elseif a == "--inset-comparison-delta-tilde"
            i += 1; inset_comparison_delta_tilde = parse(Float64, argv[i])
        elseif startswith(a, "--inset-comparison-delta-tilde=")
            inset_comparison_delta_tilde = parse(Float64, split(a, "=", limit = 2)[2])
        elseif a == "--syk4-reference-config"
            i += 1; syk4_reference_config = argv[i]
        elseif startswith(a, "--syk4-reference-config=")
            syk4_reference_config = split(a, "=", limit = 2)[2]
        elseif a == "--syk4-reference-spectra-dir"
            i += 1; syk4_reference_spectra_dir = argv[i]
        elseif startswith(a, "--syk4-reference-spectra-dir=")
            syk4_reference_spectra_dir = split(a, "=", limit = 2)[2]
        elseif a == "--syk4-reference-validation"
            i += 1; syk4_reference_validation = argv[i]
        elseif startswith(a, "--syk4-reference-validation=")
            syk4_reference_validation = split(a, "=", limit = 2)[2]
        elseif a == "--syk4-reference-eta"
            i += 1; syk4_reference_eta = parse(Float64, argv[i])
        elseif startswith(a, "--syk4-reference-eta=")
            syk4_reference_eta = parse(Float64, split(a, "=", limit = 2)[2])
        elseif a == "--syk4-reference-seeds"
            i += 1; syk4_reference_seeds = _parse_seed_range(argv[i])
        elseif startswith(a, "--syk4-reference-seeds=")
            syk4_reference_seeds = _parse_seed_range(split(a, "=", limit = 2)[2])
        else error("unrecognized argument: $a")
        end
        i += 1
    end
    shard_count >= 1 || error("--shard-count must be >= 1, got $shard_count")
    (1 <= shard_index <= shard_count) ||
        error("--shard-index must satisfy 1 <= shard-index <= shard-count; got index=$shard_index count=$shard_count")
    !isempty(syk4_reference_config) && isempty(syk4_reference_seeds) &&
        error("--syk4-reference-seeds is required with --syk4-reference-config")
    !isempty(syk4_reference_config) && isempty(syk4_reference_validation) &&
        error("--syk4-reference-validation is required with --syk4-reference-config")
    isempty(syk4_reference_config) && !isempty(syk4_reference_seeds) &&
        error("--syk4-reference-config is required with --syk4-reference-seeds")
    isempty(syk4_reference_config) && !isempty(syk4_reference_validation) &&
        error("--syk4-reference-config is required with --syk4-reference-validation")
    filling === nothing && n_orb == 10 && (filling = 3)
    (isfinite(delta_cd) && delta_cd > 0.0) ||
        error("--delta-cd must be positive and finite; got $delta_cd")
    return (output_dir = abspath(output_dir),
             cache_dir = abspath(cache_dir),
             n_seeds = n_seeds,
             window = window,
             n_orb = n_orb,
             filling = filling,
             shard_index = shard_index,
             shard_count = shard_count,
             build_cache_only = build_cache_only,
             delta_tildes = delta_tildes,
             delta_cd = delta_cd,
             with_dynamics = with_dynamics,
             inset_comparison_caches = inset_comparison_caches,
             inset_comparison_delta_tilde = inset_comparison_delta_tilde,
             syk4_reference_config = isempty(syk4_reference_config) ?
                "" : abspath(syk4_reference_config),
             syk4_reference_spectra_dir = isempty(syk4_reference_spectra_dir) ?
                "" : abspath(syk4_reference_spectra_dir),
             syk4_reference_validation = isempty(syk4_reference_validation) ?
                "" : abspath(syk4_reference_validation),
             syk4_reference_eta = syk4_reference_eta,
             syk4_reference_seeds = syk4_reference_seeds)
end

function _parse_seed_range(s::AbstractString)
    parts = split(strip(s), ":")
    length(parts) == 2 ||
        error("seed range must be lo:hi; got '$s'")
    lo = parse(Int, strip(parts[1]))
    hi = parse(Int, strip(parts[2]))
    1 <= lo <= hi || error("seed range must satisfy 1 <= lo <= hi; got '$s'")
    return collect(lo:hi)
end

"""Parse `--delta-tildes 0.1,1.0,10.0` (or a single value) into a `Vector{Float64}`."""
function _parse_delta_tildes(s::AbstractString)
    parts = split(replace(String(s), ";" => ","), ",")
    vs = Float64[]
    for p in parts
        stripped = strip(p)
        isempty(stripped) && continue
        push!(vs, parse(Float64, stripped))
    end
    isempty(vs) && error("--delta-tildes must contain at least one value; got '$s'")
    return vs
end

"""
    _parse_inset_comparison_cache(spec) -> (n_orb::Int, path::String)

Parse `--inset-comparison-cache N:/absolute/path.jld2` (or `N=/absolute/path.jld2`).
The `N` is the system size that will label the overlay curve in the inset;
the path must be a merged sigma-SFF cache produced by the same driver.
"""
function _parse_inset_comparison_cache(s::AbstractString)
    # Split on the FIRST occurrence of ':' or '=' only so that absolute
    # paths containing further ':' characters remain intact.
    sep_idx = something(findfirst(c -> c == ':' || c == '=', s), 0)
    sep_idx == 0 && error(
        "--inset-comparison-cache expects `<N>:<path>` (or `<N>=<path>`); got '$s'")
    n_orb = tryparse(Int, s[1:prevind(s, sep_idx)])
    n_orb === nothing &&
        error("--inset-comparison-cache: could not parse system size in '$s'")
    path = strip(s[nextind(s, sep_idx):end])
    isempty(path) &&
        error("--inset-comparison-cache: empty path in '$s'")
    isfile(path) ||
        error("--inset-comparison-cache: file not found: $path")
    return (Int(n_orb), String(path))
end

"""
    _load_sigmas_only(path) -> Vector{Vector{Float64}}

Read the per-seed singular-value vectors from a merged multimode cache
file, without loading any of the optional dynamics / diagnostics arrays.
Used by the finite-size comparison inset.
"""
function _load_sigmas_only(path::AbstractString)
    sigmas = Vector{Vector{Float64}}()
    JLD2.jldopen(path, "r") do f
        haskey(f, "sigmas") || error("_load_sigmas_only: cache missing 'sigmas': $path")
        for arr in f["sigmas"]
            push!(sigmas, Vector{Float64}(arr))
        end
    end
    return sigmas
end

# ------------------------------------------------------------------
# Main driver
# ------------------------------------------------------------------
function main(argv = ARGS)
    cli = _parse_cli(argv)

    # Rebind system-size globals from CLI before anything else.
    global N_ORB     = Int(cli.n_orb)
    # `--filling` is optional: when omitted, fall back to div(N_ORB, 2)
    # MultimodeParams guard 0 < filling < n_orb.
    global FILLING   = cli.filling === nothing ? nothing : Int(cli.filling)
    filling_effective = FILLING === nothing ? div(N_ORB, 2) : FILLING
    global D_HILBERT = binomial(N_ORB, filling_effective)
    global K_LIOUV   = D_HILBERT^2
    global DELTA_TILDES_PLOT = tuple(cli.delta_tildes...)
    # Must be rebound before any `cache_path` / `build_loss` call: it appears
    # in every cache filename and sets kappa/Delta_cd.
    global DELTA_CD_OVER_2PI_MHZ = Float64(cli.delta_cd)

    taus = log_range(TAU_GRID[1], TAU_GRID[2], TAU_GRID[3])
    qs   = 2π .* taus

    println("=" ^ 78)
    println("Final Figure 3  (multimode open-system HU sigma-SFF)")
    println("output dir:            $(cli.output_dir)")
    println("sigma cache dir:       $(cli.cache_dir)")
    println("delta_tildes:          $(collect(DELTA_TILDES_PLOT))")
    @printf("Delta_cd/2pi:          %g MHz   (kappa/Delta = %g)\n",
            DELTA_CD_OVER_2PI_MHZ,
            KAPPA_OVER_2PI_MHZ / DELTA_CD_OVER_2PI_MHZ)
    println("n_orb, n_grid, M:      $N_ORB, $N_GRID, $MODE_CUTOFF")
    println("filling:               $filling_effective (of $N_ORB)")
    println("K_liouv, d_hilbert:    $K_LIOUV, $D_HILBERT")
    println("n_seeds:               $(cli.n_seeds)")
    println("shard:                 $(cli.shard_index)/$(cli.shard_count)")
    println("build-cache-only:      $(cli.build_cache_only)")
    println("with-dynamics:         $(cli.with_dynamics)")
    println("analysis wnd:          [$(cli.window[1]), $(cli.window[2])]")
    println("=" ^ 78)
    flush(stdout)

    # Cache-build branch: build the shard's subset of the caches and exit.
    # Callers running under Slurm should launch one process per shard, then
    # merge with `merge_fig3_multimode_shards.jl`; the final plotting run
    # reads the merged `multimode_centered__...` files with shard_count = 1.
    if cli.build_cache_only
        for (i, dt) in enumerate(DELTA_TILDES_PLOT)
            @printf("\n--- (build-only) delta_tilde = %g shard %d/%d ---\n",
                    dt, cli.shard_index, cli.shard_count)
            c = collect_or_build_singulars(dt, cli.n_seeds, cli.cache_dir;
                                          allow_build = true,
                                            shard_index = cli.shard_index,
                                            shard_count = cli.shard_count,
                                            with_dynamics = cli.with_dynamics)
            @printf("    wrote %d seeds -> %s\n", length(c.sigmas), c.path)
        end
        println("\nCache build complete (shard $(cli.shard_index) of $(cli.shard_count)).")
        return 0
    end

    sff_boots  = Vector{Any}(undef, length(DELTA_TILDES_PLOT))
    n_used_list = Int[]
    K_list      = Int[]
    cache_paths = String[]
    r_HD_pooled = Float64[]
    mu_re_pooled = Float64[]
    mu_im_pooled = Float64[]
    n_jumps_pooled = Int[]
    for (i, dt) in enumerate(DELTA_TILDES_PLOT)
        @printf("\n--- delta_tilde = %g ---\n", dt)
        flush(stdout)
        c = collect_or_build_singulars(dt, cli.n_seeds, cli.cache_dir;
                                        shard_index = cli.shard_index,
                                        shard_count = cli.shard_count)
        push!(cache_paths, c.path)
        push!(K_list, isempty(c.sigmas) ? 0 : length(first(c.sigmas)))
        append!(r_HD_pooled,    c.r_HD)
        append!(mu_re_pooled,   c.mu_re)
        append!(mu_im_pooled,   c.mu_im)
        append!(n_jumps_pooled, c.n_jumps)
        # Bootstrap the sigma-SFF after equalizing post-filter counts.
        eq_sigmas = _equalize_sigma_counts(c.sigmas, EPSILON_ZERO)
        res = bootstrap_kunf_ham(eq_sigmas, qs;
                        analysis_window = cli.window,
                        epsilon_zero = EPSILON_ZERO,
                        n_bins = HU_N_BINS, degree = HU_DEGREE,
                        n_boot = N_BOOT_SFF,
                        rng = MersenneTwister(RNG_SEED_SFF_BOOTSTRAP + i))
        sff_boots[i]  = res
        push!(n_used_list, res.n_used)
        @printf("    n_seeds=%3d  K=%d  M_bulk=%d\n",
                length(c.sigmas), K_list[end], res.M)
    end

    # ---- RMT references (symmetry-class matched) -----------------------
    r_HD_median = median(r_HD_pooled)
    @printf("\n--- Physical ensemble diagnostics ---\n")
    @printf("    r_HD median (||L_H||_F / ||L_D||_F): %.4g  (min=%.3g max=%.3g)\n",
            r_HD_median, minimum(r_HD_pooled), maximum(r_HD_pooled))
    @printf("    |mu = tr(L)/D|  mean = %.3g\n",
            mean(sqrt.(mu_re_pooled.^2 .+ mu_im_pooled.^2)))
    M_dissipators = maximum(n_jumps_pooled)   # deterministic given (n_orb, n_grid, mode_cutoff)
    @printf("    n_cavity_loss_jumps (model) = %d  (used for random-Lindbladian M)\n",
            M_dissipators)

    # Same unit-plateau analytical folded-GOE reference used in Figure 5:
    # K_f-GOE(t/t_Hei) = K_GOE(2|t/t_Hei|).
    goe_curve = folded_goe_form_factor(taus)

    mkpath(cli.output_dir)
    suffix   = _window_suffix(cli.window)

    # caches (populated only when the sweep was invoked with
    # `--with-dynamics`).  When all retained caches carry `times`, `entropy_t`
    # and `populations_t` on a shared time grid, the figure switches to a
    # 2-panel side-by-side layout with the occupation of orbital j = filling
    # layout with only the sigma-SFF + inset.
    dyn_times_ref::Union{Nothing,Vector{Float64}} = nothing
    dyn_pops_avg  = Vector{Union{Nothing,Vector{Float64}}}(undef, length(DELTA_TILDES_PLOT))
    dyn_n_seeds   = zeros(Int, length(DELTA_TILDES_PLOT))
    for (i, dt) in enumerate(DELTA_TILDES_PLOT)
        dyn_pops_avg[i] = nothing
        isfile(cache_paths[i]) || continue
        JLD2.jldopen(cache_paths[i], "r") do f
            haskey(f, "times") && haskey(f, "populations_t") || return
            t_local = Vector{Float64}(f["times"])
            if dyn_times_ref === nothing
                dyn_times_ref = t_local
            elseif length(dyn_times_ref) != length(t_local) ||
                   maximum(abs.(dyn_times_ref .- t_local)) > 1e-12
                @warn "figure 3 dynamics: time grids disagree across dts; skipping bottom panel" dt
                return
            end
            pops_list = f["populations_t"]  # Vector{Matrix{Float64}} (n_orb x T)
            isempty(pops_list) && return
            n = size(pops_list[1], 1); T = size(pops_list[1], 2)
            j = filling_effective   # last-filled-orbital convention (matches fig 2)
            pop_sum = zeros(Float64, T)
            n_used = 0
            for P in pops_list
                size(P) == (n, T) || continue
                @views pop_sum .+= P[j, :]
                n_used += 1
            end
            n_used == 0 && return
            dyn_pops_avg[i] = pop_sum ./ n_used
            dyn_n_seeds[i]  = n_used
        end
    end
    have_dynamics = dyn_times_ref !== nothing &&
        all(v -> v !== nothing, dyn_pops_avg)
    have_dynamics && @printf("\n--- Dynamics panel: n_seeds per dt = %s (j = filling = %d) ---\n",
                              string(dyn_n_seeds), filling_effective)

    syk4_ref = nothing
    syk4_boot = nothing
    if !isempty(cli.syk4_reference_config)
        println("\n--- Loading calibrated SYK4+diss. reference ---")
        syk4_ref = load_syk4_reference(cli.syk4_reference_config;
            spectra_dir_override = cli.syk4_reference_spectra_dir,
            validation_path = cli.syk4_reference_validation,
            eta = cli.syk4_reference_eta,
            seeds = cli.syk4_reference_seeds,
            expected_n_orb = N_ORB,
            expected_filling = filling_effective)
        syk4_boot = bootstrap_kunf_ham(
            _equalize_sigma_counts(syk4_ref.sigmas, EPSILON_ZERO), qs;
            analysis_window = cli.window,
            epsilon_zero = EPSILON_ZERO,
            n_bins = HU_N_BINS,
            degree = HU_DEGREE,
            n_boot = N_BOOT_SFF,
            rng = MersenneTwister(RNG_SEED_SFF_BOOTSTRAP + 200))
        @printf("    config=%s  eta=%g  gamma=%g  seeds=%d\n",
                syk4_ref.config_name, syk4_ref.eta, syk4_ref.gamma,
                syk4_ref.n_used)
    end

    pdf_path = joinpath(cli.output_dir,
        have_dynamics ?
            "figure_3_multimode_unfolded_sigma_sff_and_occupation$(suffix).pdf" :
            "figure_3_multimode_unfolded_sigma_sff$(suffix).pdf")
    png_path = joinpath(cli.output_dir,
        have_dynamics ?
            "figure_3_multimode_unfolded_sigma_sff_and_occupation$(suffix).png" :
            "figure_3_multimode_unfolded_sigma_sff$(suffix).png")

    n_dt = length(DELTA_TILDES_PLOT)
    curve_colors = figure3_curve_colors(n_dt)
    main_width = figure3_source_px(1.15)
    reference_width = figure3_source_px(1.35)
    base_theme = theme_magiclatex(
        PaletteName = :gem_2024,
        FigureSize = MagicLaTeX.pt2px.(FIGURE3_AUTHORING_CANVAS_SIZE_PT),
    )
    compact_theme = merge(base_theme, Theme(
        figure_padding = figure3_source_px.((2.0, 2.0, 1.5, 1.5)),
    ))

    fig = with_theme(compact_theme) do
        figure = Figure(backgroundcolor = :white)
        axis_common = (
            aspect = 1.0,
            xgridvisible = false,
            ygridvisible = false,
            xlabelpadding = figure3_source_px(1.5),
            xticklabelpad = figure3_source_px(1.3),
            yticklabelpad = figure3_source_px(1.3),
        )
        ax = Axis(figure[1, 1];
            axis_common...,
            xscale = log10,
            yscale = log10,
            xlabel = L"t/t_{\mathrm{Hei}}",
            ylabel = L"\mathrm{\sigma SFF}",
            ylabelpadding = figure3_source_px(1.5),
            xticks = ([1e-2, 1e-1, 1.0, 1e1],
                      [L"10^{-2}", L"10^{-1}", L"10^{0}", L"10^{1}"]),
        )

        parameter_lines = Any[]
    for (i, _) in enumerate(DELTA_TILDES_PLOT)
        boot = sff_boots[i]
        color = curve_colors[i]
        # Shaded 68 % seed-cluster bootstrap band (fig 1 style).
        lower = positive_or_nan(boot.lower)
        upper = positive_or_nan(boot.upper)
        valid = @. isfinite(lower) & isfinite(upper) & (upper >= lower)
        lower[.!valid] .= NaN
        upper[.!valid] .= NaN
        band!(ax, taus, lower, upper; color = (color, 0.13))
        line = lines!(ax, taus, figure3_smooth_curve(boot.curve);
            color = color, linewidth = main_width)
        push!(parameter_lines, line)
    end
    # Folded-GOE reference (class-AI / BDI-dagger singular-value proxy).
    goe_line = lines!(ax, taus, goe_curve;
        color = :black, linestyle = :dash, linewidth = reference_width)
    reference_lines = Any[goe_line]
    reference_labels = Any["f-GOE"]
    if syk4_boot !== nothing
        smoothed_syk4 = figure3_smooth_curve(syk4_boot.curve)
        mask = @. isfinite(taus) & (taus > 0) &
                  isfinite(smoothed_syk4) & (smoothed_syk4 > 0)
        lines!(ax, taus[mask], smoothed_syk4[mask];
            color = FIGURE3_TARGET_MODEL_COLOR,
            linewidth = reference_width)
    end
    # Reference: RMT plateau K_unf -> 1 on the connected sigma-SFF.
    hlines!(ax, [1.0]; color = :gray45,
        linestyle = :dot, linewidth = figure3_source_px(0.7))
    xlims!(ax, X_LOWER_SFF, 1.7 * taus[end])
    ylims!(ax, Y_LOWER_SFF, nothing)

    # In-panel legend for the reference lines only.  The delta_tilde
    # colour key sits in the right dynamics panel.  Top-right is above
    # the plateau region and away from the (a) label and inset.
    axislegend(
        ax,
        reference_lines,
        reference_labels;
        position = (0.95, 0.96),
        framevisible = false,
        labelsize = figure3_source_px(FIGURE3_LEGEND_LABEL_PT),
        patchsize = figure3_source_px.((12.0, 2.5)),
        padding = figure3_source_px.((0.5, 0.5, 0.5, 0.5)),
        margin = (0, 0, 0, 0),
        rowgap = figure3_source_px(0.5),
        patchlabelgap = figure3_source_px(1.0),
    )

    # -- Inset: finite-size comparison of the connected sigma-SFF at a
    # single delta_tilde.  Overlays the CURRENT-N kunf curve at
    # `cli.inset_comparison_delta_tilde` (default 0.01) together with
    # the same curve computed from each `--inset-comparison-cache
    # <N>:<path>` argument.  The comparison caches are merged sigma-SFF
    # files produced by earlier runs of THIS driver at other N values.
    # The inset is skipped silently when no comparison caches were
    # supplied on the CLI.
    dt_target = cli.inset_comparison_delta_tilde
    idx_target = findfirst(dt -> isapprox(dt, dt_target; atol = 1e-12),
                            DELTA_TILDES_PLOT)
    inset_enabled = !isempty(cli.inset_comparison_caches) && idx_target !== nothing
    if !isempty(cli.inset_comparison_caches) && idx_target === nothing
        @warn "inset skipped: delta_tilde=$dt_target not in main sweep"
    end
    if inset_enabled
        println("\n--- Rendering sigma-SFF finite-size comparison inset ---")
        # Collect (N, kunf) pairs.  Main N first, then each user cache.
        inset_curves  = Vector{Tuple{Int,Int,Vector{Float64}}}()
        push!(inset_curves,
              (cli.n_orb, length(first(_load_sigmas_only(cache_paths[idx_target]))),
               sff_boots[idx_target].curve))
        @printf("    N=%2d  K=%5d  (base)\n",
                cli.n_orb, length(first(_load_sigmas_only(cache_paths[idx_target]))))
        for (n_ins, path_ins) in cli.inset_comparison_caches
            t0 = time()
            sig_ins = _load_sigmas_only(path_ins)
            K_ins = isempty(sig_ins) ? 0 : length(first(sig_ins))
            res_ins = kunf_ham(_equalize_sigma_counts(sig_ins, EPSILON_ZERO), qs;
                                analysis_window = cli.window,
                                epsilon_zero = EPSILON_ZERO,
                                n_bins = HU_N_BINS, degree = HU_DEGREE)
            push!(inset_curves, (n_ins, K_ins, res_ins.k_unf))
            @printf("    N=%2d  K=%5d  n_seeds=%d  (%.1fs)  from %s\n",
                    n_ins, K_ins, length(sig_ins), time() - t0, basename(path_ins))
        end
        # Sort by N so the smallest system is drawn last (topmost).
        sort!(inset_curves, by = t -> t[1])

        # Bottom-right placement leaves the upper reference legend clear.
        inset = Axis(figure[1, 1];
                     tellwidth = false, tellheight = false,
                     width = Relative(0.39), height = Relative(0.33),
                     halign = FIGURE3_INSET_HALIGN, valign = 0.19,
                     alignmode = Inside(),
                     xscale = log10, yscale = log10,
                     yaxisposition = :right,
                     xlabel = "", ylabel = "",
                     xticklabelsvisible = true,
                     yticklabelsvisible = true,
                     xticksvisible = true, yticksvisible = true,
                     xticks = ([1e-2, 1.0, 1e1],
                               [L"10^{-2}", L"10^{0}", L"10^{1}"]),
                     yticks = ([1e-1, 1.0], [L"10^{-1}", L"10^{0}"]),
                     xticklabelsize = figure3_source_px(5.2),
                     yticklabelsize = figure3_source_px(5.2),
                     xticklabelpad = figure3_source_px(0.8),
                     yticklabelpad = figure3_source_px(0.8),
                     xminorticksvisible = false, yminorticksvisible = false,
                     xgridvisible = false, ygridvisible = false,
                     backgroundcolor = (:white, 0.88),
                     spinewidth = figure3_source_px(0.5))
        translate!(inset.blockscene, 0, 0, 500)
        translate!(inset.scene,      0, 0, 500)
        # Match the light-to-dark finite-size palette used by Figure 1.
        inset_palette = FIGURE3_INSET_GREYS
        for (i_ins, (_n_ins, _K_ins, curve)) in enumerate(inset_curves)
            color = inset_palette[mod1(i_ins, length(inset_palette))]
            smoothed_curve = figure3_smooth_curve(curve)
            mask = @. isfinite(taus) & (taus > 0) &
                       isfinite(smoothed_curve) & (smoothed_curve > 0)
            any(mask) || continue
            lines!(inset, taus[mask], smoothed_curve[mask];
                    color = color, linewidth = main_width * 0.85)
        end
        # Reference RMT plateau kappa -> 1 (dotted, matches main panel).
        hlines!(inset, [1.0]; color = :gray45, linestyle = :dot,
                linewidth = figure3_source_px(0.5))
        xlims!(inset, X_LOWER_SFF, taus[end])
        ylims!(inset, Y_LOWER_SFF, nothing)
    end

    # -- Right panel: occupation of orbital j = filling vs t --
    # arrays on a shared time grid (build with `--with-dynamics`).
    # Same delta_tilde colours as the top panel; steady-state uniform-
    # density reference line at filling / n_orb.
    if have_dynamics
        ax_bot = Axis(figure[1, 2];
                     axis_common...,
                     xscale = log10,
                     xlabel = L"tJ",
                     ylabel = latexstring(@sprintf("n_{%d}(t)",
                                                    filling_effective)),
                     ylabelpadding = figure3_source_px(-1.25),
                     xticks = ([1e-1, 1e1, 1e3, 1e5],
                                [L"10^{-1}", L"10^{1}",
                                 L"10^{3}", L"10^{5}"]))
        ax_bot.yaxis.elements[:labeltext].offset[] = Vec2f(
            0, -figure3_source_px(FIGURE3_DYNAMICS_YLABEL_DOWNSHIFT_PT))
        dynamics_lines = Any[]
        for (i, _) in enumerate(DELTA_TILDES_PLOT)
            curve = dyn_pops_avg[i]
            curve === nothing && continue
            keep = findall(k -> dyn_times_ref[k] > 0 && isfinite(curve[k]),
                            eachindex(curve))
            isempty(keep) && continue
            line = lines!(ax_bot, dyn_times_ref[keep], curve[keep];
                    color = curve_colors[i], linewidth = main_width)
            push!(dynamics_lines, line)
        end
        syk4_dynamics_line = nothing
        if syk4_ref !== nothing
            keep = findall(
                k -> syk4_ref.times[k] > 0 &&
                     syk4_ref.times[k] <= 1.0e5 &&
                     isfinite(syk4_ref.mean_populations[filling_effective, k]),
                eachindex(syk4_ref.times))
            isempty(keep) && error(
                "SYK4+diss. reference has no finite occupation points in Figure 3's time window")
            syk4_dynamics_line = lines!(
                ax_bot,
                syk4_ref.times[keep],
                syk4_ref.mean_populations[filling_effective, keep];
                color = FIGURE3_TARGET_MODEL_COLOR,
                linewidth = reference_width)
        end
        # Uniform-density steady-state reference (filling / n_orb).
        reference_occupation = Float64(filling_effective) / Float64(N_ORB)
        occupation_line = hlines!(ax_bot, [reference_occupation];
            color = :gray45, linestyle = :dot,
            linewidth = figure3_source_px(0.7))
        # PI to focus on the transient + short-time thermalization
        # regime).  Start at 10^{-1} so the very-early-time transient
        # is visible.
        xlims!(ax_bot, 4.0e-2, 1.0e5)
        parameter_labels = Any[
            latexstring(@sprintf("%g", dt)) for dt in DELTA_TILDES_PLOT
        ]
        axislegend(
            ax_bot,
            dynamics_lines,
            parameter_labels,
            L"\delta\tilde{\omega}=";
            position = (0.96, 0.88),
            orientation = :horizontal,
            titleposition = :left,
            nbanks = 2,
            framevisible = false,
            labelsize = figure3_source_px(FIGURE3_LEGEND_LABEL_PT),
            titlesize = figure3_source_px(FIGURE3_LEGEND_LABEL_PT),
            patchsize = figure3_source_px.((6.0, 2.5)),
            padding = figure3_source_px.((0.5, 0.5, 0.5, 0.5)),
            margin = (0, 0, 0, 0),
            rowgap = figure3_source_px(0.5),
            colgap = figure3_source_px(0.8),
            patchlabelgap = figure3_source_px(0.6),
            titlegap = figure3_source_px(0.8),
        )
        axislegend(
            ax_bot,
            [occupation_line],
            [L"\nu"];
            position = (0.96, 0.52),
            framevisible = false,
            labelsize = figure3_source_px(FIGURE3_LEGEND_LABEL_PT),
            patchsize = figure3_source_px.((8.0, 2.5)),
            padding = figure3_source_px.((0.5, 0.5, 0.5, 0.5)),
            margin = (0, 0, 0, 0),
            patchlabelgap = figure3_source_px(0.8),
        )
        if syk4_dynamics_line !== nothing
            axislegend(
                ax_bot,
                [syk4_dynamics_line],
                [L"\mathrm{diss.\ SYK}"];
                position = (0.96, 0.72),
                framevisible = false,
                labelsize = figure3_source_px(FIGURE3_LEGEND_LABEL_PT),
                patchsize = figure3_source_px.((6.0, 2.5)),
                padding = figure3_source_px.((0.5, 0.5, 0.5, 0.5)),
                margin = (0, 0, 0, 0),
                patchlabelgap = figure3_source_px(0.8),
            )
        end
        text!(ax_bot, 0.012, 0.84; text = L"\mathrm{(b)}",
              space = :relative, align = (:left, :top), color = :black)
    end

        text!(ax, 0.012, 0.96; text = L"\mathrm{(a)}",
              space = :relative, align = (:left, :top), color = :black)
        configure_figure3_layout!(figure, have_dynamics, figure3_source_px)
        figure
    end

    output_base = splitext(pdf_path)[1]
    source_size_points = mktempdir() do temporary_dir
        uncropped_pdf = joinpath(temporary_dir, "figure-uncropped.pdf")
        save(uncropped_pdf, fig)
        crop_pdf_hires(uncropped_pdf, pdf_path; margin_pt = PDF_CROP_MARGIN_PT)
    end
    _rasterize_pdf_288(pdf_path, output_base)
    println("\nwrote $pdf_path")
    println("wrote $png_path")

    manifest_path = joinpath(cli.output_dir, "figure_3__manifest$(suffix).toml")
    open(manifest_path, "w") do io
        DatasetPaths.print_manifest(io, Dict{String,Any}(
            "delta_tildes"              => collect(DELTA_TILDES_PLOT),
            "dynamics_n_seeds_used"     => dyn_n_seeds,
            "dynamics_time_range"       => have_dynamics ?
                [minimum(dyn_times_ref), maximum(dyn_times_ref)] : Float64[],
            "authoring_canvas_size_points" =>
                collect(FIGURE3_AUTHORING_CANVAS_SIZE_PT),
            "source_figure_size_points" => source_size_points,
            "final_figure_size_points_at_columnwidth" =>
                source_size_points .* FIGURE3_LATEX_SCALE,
            "intended_latex_scale"      => FIGURE3_LATEX_SCALE,
            "curve_colormap_rgb"        => FIGURE3_CURVE_COLOR_RGB,
            "control_curve_color_rgb"   => FIGURE3_TARGET_MODEL_COLOR_RGB,
            "top_sgolay_window"         => FIGURE3_SG_WINDOW,
            "top_sgolay_degree"         => FIGURE3_SG_DEGREE,
            "legend_label_size_points"  => FIGURE3_LEGEND_LABEL_PT,
            "delta_cd_over_2pi_mhz"     => DELTA_CD_OVER_2PI_MHZ,
            "n_orb"                     => N_ORB,
            "filling"                   => filling_effective,
            "hilbert_dim"               => D_HILBERT,
            "n_grid"                    => N_GRID,
            "box_length"                => BOX_LENGTH,
            "mode_cutoff"               => MODE_CUTOFF,
            "zeta"                      => ZETA,
            "speckle_grains_per_side"   => SPECKLE_GRAINS,
            "disorder_strength"         => DISORDER_STRENGTH,
            "drive_wavevector"          => DRIVE_WAVEVECTOR,
            "energy_scale"              => ENERGY_SCALE,
            "kappa_over_2pi_mhz"        => KAPPA_OVER_2PI_MHZ,
            "gamma_over_2pi_mhz"        => GAMMA_OVER_2PI_MHZ,
            "delta_da_over_2pi_mhz"     => DELTA_DA_OVER_2PI_MHZ,
            "rate_scale"                => RATE_SCALE,
            "r_HD_median"               => r_HD_median,
            "r_HD_min"                  => minimum(r_HD_pooled),
            "r_HD_max"                  => maximum(r_HD_pooled),
            "n_seeds"                   => cli.n_seeds,
            "n_seeds_used"              => n_used_list,
            "K_liouville"               => K_list,
            "analysis_window"           => collect(cli.window),
            "hu_n_bins"                 => HU_N_BINS,
            "hu_degree"                 => HU_DEGREE,
            "tau_grid"                  => Dict("min" => TAU_GRID[1],
                                                 "max" => TAU_GRID[2],
                                                 "n"   => TAU_GRID[3]),
            "epsilon_zero"              => EPSILON_ZERO,
            "y_lower_clip"              => Y_LOWER_SFF,
            "x_lower_clip"              => X_LOWER_SFF,
            "cache_paths"               => cache_paths,
            "finite_size_inset" => Dict(
                "delta_tilde"           => cli.inset_comparison_delta_tilde,
                "palette_rgb"           => [
                    [0.72, 0.72, 0.72],
                    [0.46, 0.46, 0.46],
                    [0.20, 0.20, 0.20],
                ],
                "comparison_caches"     => [Dict("n_orb" => n, "path" => p)
                                             for (n, p) in cli.inset_comparison_caches],
            ),
            "syk4_diss_reference" => syk4_ref === nothing ?
                Dict() : Dict(
                    "config"               => syk4_ref.config_path,
                    "config_name"          => syk4_ref.config_name,
                    "spectra_dir"          => syk4_ref.spectra_dir,
                    "validation"           => syk4_ref.validation_path,
                    "calibration"          => syk4_ref.calibration_path,
                    "target_delta_tilde"   => syk4_ref.target_delta_tilde,
                    "n_random_jumps"       => syk4_ref.n_random_jumps,
                    "n_cavity_jumps"       => syk4_ref.n_cavity_jumps,
                    "physical_H_span_mean" => syk4_ref.physical_H_span_mean,
                    "synthetic_H_span_mean" => syk4_ref.synthetic_H_span_mean,
                    "H_span_relative_error" => syk4_ref.H_span_relative_error,
                    "physical_LD_norm_mean" => syk4_ref.physical_LD_norm_mean,
                    "synthetic_LD_norm_mean" => syk4_ref.synthetic_LD_norm_mean,
                    "LD_norm_relative_error" => syk4_ref.LD_norm_relative_error,
                    "eta"                  => syk4_ref.eta,
                    "gamma"                => syk4_ref.gamma,
                    "svd_tol"              => syk4_ref.svd_tol,
                    "seed_range"           => syk4_ref.seed_range,
                    "n_seeds_requested"    => syk4_ref.n_requested,
                    "n_seeds_used"         => syk4_ref.n_used,
                    "bootstrap_rng_seed"   =>
                        RNG_SEED_SFF_BOOTSTRAP + 200,
                    "panels"               => have_dynamics ?
                        ["sigma_sff", "occupation"] : ["sigma_sff"],
                ),
            "sff_bootstrap"             => Dict(
                "n_boot"    => N_BOOT_SFF,
                "quantiles" => [0.16, 0.84],
                "rng_seed"  => RNG_SEED_SFF_BOOTSTRAP,
            ),
            "folded_goe"                => Dict(
            ),
            "outputs"                   => [pdf_path, png_path],
        ); path=manifest_path)
    end
    println("wrote $manifest_path")

    return (pdf = pdf_path, png = png_path, manifest = manifest_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
