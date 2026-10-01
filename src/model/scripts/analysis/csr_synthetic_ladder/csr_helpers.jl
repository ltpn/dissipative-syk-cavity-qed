using Statistics: mean, quantile

const STEADY_TOL = 1.0e-8
const DISTANCE_TOL = 1.0e-12
const CSR_BULK_FRACTION = 0.5
const CSR_RMAX_QUANTILE = 0.999
const CSR_IM_AXIS_TOL = 1.0e-6

"""
    complex_spacing_ratios_bulk(spectrum;
                                 bulk_fraction = CSR_BULK_FRACTION,
                                 rmax_quantile = CSR_RMAX_QUANTILE,
                                 im_axis_tol   = CSR_IM_AXIS_TOL,
                                 distance_tol  = DISTANCE_TOL)
        -> (ratios, centroid, r_max, n_bulk_refs, n_full, n_pre_filter)

Compute z_i = (lambda_NN - lambda_i) / (lambda_NNN - lambda_i) only for
those reference eigenvalues lambda_i whose distance to the spectrum
centroid c = mean(spectrum) satisfies |lambda_i - c| <= bulk_fraction *
r_max, with r_max defined by the `rmax_quantile`-quantile of {|lambda -
c|} (default 99.9-percentile).  NN and NNN are still selected from the
(filtered) spectrum, so bulk reference points can pair with near-edge
neighbours -- this matches the paper's convention, where only the
reference point z_i is required to lie in the bulk region.

Before bulk selection, eigenvalues with `|Im lambda| < im_axis_tol` are
removed to eliminate the BDI-dagger real-eigenvalue subset (a symmetry
artefact that contributes an Im zeta = 0 stripe to the CSR density).

The caller is expected to have already dropped steady modes; no
additional filtering is applied here.
"""
function complex_spacing_ratios_bulk(spectrum::AbstractVector{<:Complex};
                                      bulk_fraction::Real = CSR_BULK_FRACTION,
                                      rmax_quantile::Real = CSR_RMAX_QUANTILE,
                                      im_axis_tol::Real   = CSR_IM_AXIS_TOL,
                                      distance_tol::Real  = DISTANCE_TOL)
    raw = ComplexF64.(spectrum)
    n_pre_filter = length(raw)
    # Drop the numerically-real subset (BDI-dagger stripe).
    values = filter(z -> abs(imag(z)) >= Float64(im_axis_tol), raw)
    n_full = length(values)
    n_full >= 4 || return (ComplexF64[], complex(NaN, NaN), NaN, 0, n_full, n_pre_filter)
    c = mean(values)
    dists = abs.(values .- c)
    r_max = quantile(dists, Float64(rmax_quantile))
    r_max > 0 || return (ComplexF64[], c, r_max, 0, n_full, n_pre_filter)
    r_bulk = Float64(bulk_fraction) * r_max
    tol = Float64(distance_tol)
    ratios = ComplexF64[]
    n_bulk_refs = 0
    # the reference-eigenvalue loop and `empty!` it each iteration.  The
    # previous per-reference `Tuple{Float64,ComplexF64}[]` allocation
    # produced O(K) allocations per reference -> O(K^2) per seed at K = 14400,
    # dominating wall time via GC.  Buffer reuse is the reason CSR now fits
    # into a reasonable prepost budget when paired with @threads across seeds.
    candidates = Tuple{Float64,ComplexF64}[]
    sizehint!(candidates, n_full)
    distinct = ComplexF64[]
    sizehint!(distinct, 2)
    for i in eachindex(values)
        # Reference-point cut: only bulk eigenvalues contribute z_i.
        dists[i] <= r_bulk || continue
        n_bulk_refs += 1
        lambda = values[i]
        empty!(candidates)
        has_degenerate_neighbor = false
        @inbounds for j in eachindex(values)
            i == j && continue
            d = abs(lambda - values[j])
            if d <= tol
                has_degenerate_neighbor = true
                break
            end
            push!(candidates, (d, values[j]))
        end
        has_degenerate_neighbor && continue
        # Only the two closest distinct neighbours are needed. Two scans
        # preserve the stable distance sort's tie order without sorting the
        # full spectrum for every bulk reference point.
        nearest = first(candidates)
        for candidate in candidates
            candidate[1] < nearest[1] && (nearest = candidate)
        end
        next_distance = Inf
        next_neighbor = nearest[2]
        for (distance, neighbor) in candidates
            distance < next_distance || continue
            abs(nearest[2] - neighbor) <= tol && continue
            next_distance = distance
            next_neighbor = neighbor
        end
        isfinite(next_distance) || continue
        empty!(distinct)
        push!(distinct, nearest[2], next_neighbor)
        denom = lambda - distinct[2]
        abs(denom) <= tol && continue
        push!(ratios, (lambda - distinct[1]) / denom)
    end
    return (ratios, c, r_max, n_bulk_refs, n_full, n_pre_filter)
end
