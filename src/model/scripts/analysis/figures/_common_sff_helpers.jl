#!/usr/bin/env julia
# _common_sff_helpers.jl
#
# Shared numerical primitives used by all final-PRL sigma-SFF figures:
#
#   - `trace_center_superoperator(L)`               : L -> L - (tr L / D) * I
#   - `hamiltonian_unfold_values(spectrum; ...)`    : per-realization polynomial
#                                                     staircase unfolding
#   - `kunf_ham(sigmas_by_seed, qs; ...)`           : R/(R-1)-unbiased connected
#                                                     sigma-SFF estimator
#   - `coherent_to_dissipative_norm_ratio(H, L)`    : r_HD from a physical build
#
# All routines follow the Kawabata et al. (arXiv:2307.08218) prescription for
# trace-centered sigma-SFF universality tests; see the accompanying README for
# the physical motivation.

using JLD2
using LinearAlgebra
using Polynomials: Polynomials
using Random: AbstractRNG, MersenneTwister, rand, randn
using Printf: @sprintf
using Statistics: mean, quantile

# ------------------------------------------------------------------
# Trace centering
# ------------------------------------------------------------------
"""
    trace_center_superoperator(L) -> (L_c, mu)

Subtract the deterministic scalar shift `mu = tr(L)/D` from a superoperator
matrix `L` (dimension `D`).  This is the Kawabata et al. sigma-SFF
convention: singular values of `L_c = L - mu * I` are the physical object,
not those of the unshifted `L`.
"""
function trace_center_superoperator(L::AbstractMatrix)
    D = size(L, 1)
    size(L, 2) == D || throw(DimensionMismatch("L must be square"))
    Lc = Matrix{ComplexF64}(L)
    mu = tr(Lc) / D
    @inbounds for i in 1:D
        Lc[i, i] -= mu
    end
    return Lc, mu
end

# ------------------------------------------------------------------
# Hamiltonian-unfolding staircase (per realization)
# ------------------------------------------------------------------
"""
    hamiltonian_unfold_values(spectrum; n_bins = 40, degree = 5)

Return the polynomial-fit staircase evaluated on `spectrum`.  The knot grid
is `n_bins + 1` evenly-spaced points across `[min(x), max(x)]`, so the
requested polynomial degree is always attainable regardless of the
physical scale of the bulk.  This deviates deliberately from
`SYK_setup.jl:hamiltonian_unfolding`, whose absolute `step = 0.1` collapses
to a low degree for the narrow Liouvillian-singular-value bulks in this
study; see the the figure drivers give estimator details.
"""
function hamiltonian_unfold_values(spectrum::AbstractVector{<:Real};
                                     n_bins::Integer = 40, degree::Integer = 5)
    xs = sort(collect(Float64.(spectrum)))
    n = length(xs); n >= 2 || return copy(xs)
    xmin, xmax = first(xs), last(xs)
    xmax > xmin || return zeros(Float64, n)
    E = collect(range(xmin, xmax; length = Int(n_bins) + 1))
    eta_counts = Float64[searchsortedlast(xs, e) for e in E]
    dstar = length(unique(xs))
    fit_degree = min(Int(degree), length(E) - 1, dstar - 1)
    fit_degree = max(fit_degree, 0)
    p = Polynomials.fit(E, eta_counts, fit_degree)
    return Float64[p(x) for x in xs]
end

# ------------------------------------------------------------------
# Unbiased connected sigma-SFF estimator
# ------------------------------------------------------------------
"Per-seed unfolded spectral traces and retained level counts."
function sigma_sff_traces(sigmas_by_seed, qs;
                          analysis_window, epsilon_zero,
                          n_bins::Integer = 40, degree::Integer = 5)
    aw = (Float64(analysis_window[1]), Float64(analysis_window[2]))
    nq = length(qs)
    Z = Matrix{ComplexF64}(undef, length(sigmas_by_seed), nq)
    levels = Vector{Int}(undef, length(sigmas_by_seed))
    n_used = 0
    for sigmas in sigmas_by_seed
        filt = sort!(Float64.(filter(s -> s > epsilon_zero, sigmas)))
        K = length(filt); K < 8 && continue
        lo = max(1, Int(ceil(aw[1] * K)))
        hi = min(K, Int(floor(aw[2] * K)))
        hi - lo < 3 && continue
        bulk = filt[lo:hi]
        xi = hamiltonian_unfold_values(bulk; n_bins = n_bins, degree = degree)
        n_used += 1
        levels[n_used] = length(bulk)
        @inbounds for qi in 1:nq
            q = qs[qi]; acc = ComplexF64(0)
            for j in eachindex(xi); acc += cis(-q * xi[j]); end
            Z[n_used, qi] = acc
        end
    end
    return (Z = Z[1:n_used, :], levels = levels[1:n_used], n_used = n_used)
end

"R/(R-1)-unbiased connected sigma-SFF from per-seed traces."
function connected_sigma_sff(Z::AbstractMatrix{<:Complex},
                             levels::AbstractVector{<:Real})
    R, nq = size(Z)
    R >= 2 || throw(ArgumentError("connected estimator requires at least two seeds"))
    length(levels) == R || throw(DimensionMismatch("level-count mismatch"))
    all(==(first(levels)), levels) ||
        throw(ArgumentError(
            "connected sigma-SFF requires a common retained level count across seeds"))
    plateau = Float64(first(levels))
    plateau > 0 || throw(ArgumentError("level counts must be positive"))
    correction = R / (R - 1)
    curve = zeros(Float64, nq)
    for qi in 1:nq
        zmean = sum(@view Z[:, qi]) / R
        mean_abs2 = sum(abs2, @view Z[:, qi]) / R
        curve[qi] = correction * (mean_abs2 - abs2(zmean)) / plateau
    end
    return curve
end

"Cluster-bootstrap a connected sigma-SFF over seed rows."
function bootstrap_connected_sigma_sff(Z::AbstractMatrix{<:Complex},
                                       levels::AbstractVector{<:Real};
                                       n_boot::Integer = 500,
                                       rng::AbstractRNG = MersenneTwister(0))
    R, nq = size(Z)
    R >= 2 || throw(ArgumentError("at least two seeds are required"))
    length(levels) == R || throw(DimensionMismatch("level-count mismatch"))
    n_boot = Int(n_boot)
    n_boot > 0 || throw(ArgumentError("n_boot must be positive"))
    curve = connected_sigma_sff(Z, levels)
    samples = zeros(Float64, n_boot, nq)
    bootstrap_centering = R / (R - 1)
    for b in 1:n_boot
        indices = rand(rng, 1:R, R)
        samples[b, :] .= bootstrap_centering .* connected_sigma_sff(
            @view(Z[indices, :]), @view(levels[indices]))
    end
    median_curve = zeros(Float64, nq)
    lower = zeros(Float64, nq)
    upper = zeros(Float64, nq)
    for qi in 1:nq
        column = @view samples[:, qi]
        median_curve[qi] = quantile(column, 0.5)
        lower[qi] = quantile(column, 0.16)
        upper[qi] = quantile(column, 0.84)
    end
    return (curve = curve, median = median_curve, lower = lower,
            upper = upper, samples = samples)
end

"""
    kunf_ham(sigmas_by_seed, qs; analysis_window, epsilon_zero, ...)

R/(R-1)-unbiased connected sigma-SFF estimator using per-realization
Hamiltonian unfolding on the analysis-window bulk.  `sigmas_by_seed[r]`
is the sorted singular-value list of realization `r`.
"""
function kunf_ham(sigmas_by_seed, qs;
                  analysis_window, epsilon_zero,
                  n_bins::Integer = 40, degree::Integer = 5)
    traces = sigma_sff_traces(sigmas_by_seed, qs;
        analysis_window = analysis_window, epsilon_zero = epsilon_zero,
        n_bins = n_bins, degree = degree)
    nq = length(qs)
    traces.n_used < 2 && return (
        k_unf = fill(NaN, nq),
        M = isempty(traces.levels) ? 0 : last(traces.levels),
        n_used = traces.n_used)
    curve = connected_sigma_sff(traces.Z, traces.levels)
    return (k_unf = curve, M = last(traces.levels), n_used = traces.n_used)
end

"Hamiltonian-unfold, then seed-bootstrap the connected sigma-SFF."
function bootstrap_kunf_ham(sigmas_by_seed, qs;
                            analysis_window, epsilon_zero,
                            n_bins::Integer = 40, degree::Integer = 5,
                            n_boot::Integer = 500,
                            rng::AbstractRNG = MersenneTwister(0))
    traces = sigma_sff_traces(sigmas_by_seed, qs;
        analysis_window = analysis_window, epsilon_zero = epsilon_zero,
        n_bins = n_bins, degree = degree)
    traces.n_used >= 2 ||
        throw(ArgumentError("at least two usable seeds are required"))
    boot = bootstrap_connected_sigma_sff(
        traces.Z, traces.levels; n_boot = n_boot, rng = rng)
    return merge(boot, (
        M = last(traces.levels),
        n_used = traces.n_used,
        levels = traces.levels,
    ))
end

# ------------------------------------------------------------------
# Coherent-to-dissipative Frobenius ratio for a physical Liouvillian
# ------------------------------------------------------------------
"""
    coherent_to_dissipative_norm_ratio(H, L) -> r_HD

Given a physical many-body Hamiltonian `H` (Hermitian, dim `d`) and the
assembled Lindblad Liouvillian `L` (dim `D = d^2`), return
`r_HD = ||L_H||_F / ||L_D||_F` where `L_H = -i (I x H - H^T x I)` is the
pure coherent piece and `L_D = L - L_H` the pure dissipative piece.
"""
function coherent_to_dissipative_norm_ratio(H::AbstractMatrix, L::AbstractMatrix)
    d = size(H, 1)
    Id  = Matrix{ComplexF64}(I, d, d)
    Hc  = Matrix{ComplexF64}(H)
    LH  = -1im .* (kron(Id, Hc) .- kron(transpose(Hc), Id))
    LD  = Matrix{ComplexF64}(L) .- LH
    return norm(LH) / norm(LD)
end

# ------------------------------------------------------------------
# On-disk cache for the RMT reference singular values
# ------------------------------------------------------------------

"Load and validate a trace-centered singular-value ensemble cache."
function load_centered_sigma_cache(path::AbstractString;
                                   expected_n_orb::Integer,
                                   filling::Integer,
                                   eta::Real,
                                   gamma::Real,
                                   svd_tol::Real)
    isfile(path) || throw(ArgumentError("centered sigma cache not found: $path"))
    n_orb = Int(expected_n_orb)
    filling = Int(filling)
    0 <= filling <= n_orb || throw(ArgumentError("invalid filling"))
    expected_K = binomial(n_orb, filling)^2
    return JLD2.jldopen(path, "r") do file
        for key in ("sigmas", "eta", "gamma", "svd_tol", "n_seeds")
            haskey(file, key) ||
                throw(ArgumentError("centered sigma cache lacks `$key`: $path"))
        end
        isapprox(Float64(file["eta"]), Float64(eta); rtol = 0,
                 atol = 32eps(Float64)) ||
            throw(ArgumentError("sigma cache has the wrong eta: $path"))
        isapprox(Float64(file["gamma"]), Float64(gamma); rtol = 0,
                 atol = 32eps(Float64)) ||
            throw(ArgumentError("sigma cache has the wrong gamma: $path"))
        isapprox(Float64(file["svd_tol"]), Float64(svd_tol); rtol = 0,
                 atol = 32eps(Float64)) ||
            throw(ArgumentError("sigma cache has the wrong SVD tolerance: $path"))
        Int(file["n_orb"]) == n_orb ||
            throw(ArgumentError("sigma cache has the wrong system size: $path"))
        sigmas = [Vector{Float64}(values) for values in file["sigmas"]]
        Int(file["n_seeds"]) == length(sigmas) ||
            throw(ArgumentError("sigma cache seed count is inconsistent: $path"))
        isempty(sigmas) && throw(ArgumentError("sigma cache is empty: $path"))
        all(length(values) == expected_K for values in sigmas) ||
            throw(ArgumentError(
                "sigma cache dimension does not match N=$n_orb, filling=$filling: $path"))
        return (sigmas = sigmas, K = expected_K, n_seeds = length(sigmas),
                n_orb = n_orb, filling = filling, path = abspath(path))
    end
end

"""Parse `--analysis-window lo,hi` (or `lo:hi`) into a `(Float64, Float64)`."""
function _parse_window(s::AbstractString)
    parts = split(replace(String(s), ":" => ","), ",")
    length(parts) == 2 || error("--analysis-window must be 'lo,hi', got '$s'")
    lo = parse(Float64, strip(parts[1]))
    hi = parse(Float64, strip(parts[2]))
    (0.0 <= lo < hi <= 1.0) || error("analysis window must satisfy 0 <= lo < hi <= 1")
    return (lo, hi)
end

"""Filename suffix for a non-default analysis window, e.g. `__aw=0p05-0p95`."""
function _window_suffix(window)
    window == DEFAULT_ANALYSIS_WINDOW && return ""
    to_slug(x) = replace(@sprintf("%g", x), "." => "p", "-" => "m")
    return string("__aw=", to_slug(window[1]), "-", to_slug(window[2]))
end
