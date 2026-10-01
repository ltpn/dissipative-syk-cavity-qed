module ComplexDSFFUnfolding

using Random
using Statistics: mean, median, quantile, std

export PowerMap,
       DensityGrid,
       GaussianFilter,
       UnfoldingSpec,
       apply_unfolding,
       mapped_spectra,
       estimate_density,
       raw_real_axis_mode,
       density_peak_hwhm,
       filter_weight,
       filter_weights,
       effective_count,
       exact_nearest_neighbours,
       weighted_nearest_spacing,
       ensemble_weighted_spacing,
       fit_unfolding_candidate,
       fit_fixed_power_policy,
       partition_traces,
       connected_dsff,
       filtered_plateau,
       projected_plateau_by_seed,
       projected_plateau,
       bootstrap_connected_dsff,
       isotonic_nondecreasing,
       calibrate_heisenberg,
       bootstrap_heisenberg

"Power map `-im * (z-z0)^beta` with branch cut `[z0,+Inf)`."
struct PowerMap
    beta::Float64
    z0::Float64
    function PowerMap(beta::Real, z0::Real)
        b = Float64(beta)
        0 < b <= 1 || throw(ArgumentError("beta must lie in (0,1]"))
        return new(b, Float64(z0))
    end
end

"A Cartesian density estimate; `values[ix,iy]` corresponds to `(x[ix],y[iy])`."
struct DensityGrid
    x::Vector{Float64}
    y::Vector{Float64}
    values::Matrix{Float64}
    function DensityGrid(x::AbstractVector{<:Real}, y::AbstractVector{<:Real},
                         values::AbstractMatrix{<:Real})
        xv = Float64.(x)
        yv = Float64.(y)
        size(values) == (length(xv), length(yv)) ||
            throw(DimensionMismatch("density dimensions must match coordinate axes"))
        issorted(xv) && issorted(yv) ||
            throw(ArgumentError("density axes must be sorted"))
        return new(xv, yv, Float64.(values))
    end
end

"An anisotropic Gaussian filter parameterized by its HWHM-derived scales."
struct GaussianFilter
    mu_x::Float64
    mu_y::Float64
    delta_x::Float64
    delta_y::Float64
    alpha_x::Float64
    alpha_y::Float64
    function GaussianFilter(mu_x::Real, mu_y::Real,
                            delta_x::Real, delta_y::Real)
        dx = Float64(delta_x)
        dy = Float64(delta_y)
        (dx > 0 || isinf(dx)) || throw(ArgumentError("delta_x must be positive"))
        (dy > 0 || isinf(dy)) || throw(ArgumentError("delta_y must be positive"))
        ax = isinf(dx) ? 0.0 : inv(dx^2)
        ay = isinf(dy) ? 0.0 : inv(dy^2)
        return new(Float64(mu_x), Float64(mu_y), dx, dy, ax, ay)
    end
end

"Ensemble-level conformal-map and filter specification."
struct UnfoldingSpec
    map::PowerMap
    filter::GaussianFilter
    density::DensityGrid
    flatness::Float64
    n_eff_median::Float64
    n_eff_by_seed::Vector{Float64}
    candidate_scores::Vector{NamedTuple}
end

apply_unfolding(zs::AbstractVector{<:Complex}, map::PowerMap) =
    ComplexF64[apply_unfolding(z, map) for z in zs]

mapped_spectra(spectra::AbstractVector, spec::UnfoldingSpec) =
    [apply_unfolding(seed_spectrum, spec.map) for seed_spectrum in spectra]

function apply_unfolding(z::Complex, map::PowerMap)
    w = ComplexF64(z) - map.z0
    r = abs(w)
    r == 0 && return 0.0 + 0.0im
    phi = atan(imag(w), real(w))
    phi < 0 && (phi += 2pi)
    return -im * r^map.beta * cis(map.beta * phi)
end

function _smooth_histogram(values::Matrix{Float64}, sigma::Float64)
    sigma > 0 || return copy(values)
    radius = max(1, ceil(Int, 4sigma))
    kernel = exp.(-0.5 .* ((-radius:radius) ./ sigma) .^ 2)
    kernel ./= sum(kernel)
    nx, ny = size(values)
    tmp = zeros(Float64, nx, ny)
    out = zeros(Float64, nx, ny)
    for ix in 1:nx, iy in 1:ny
        acc = 0.0
        for (ki, offset) in enumerate(-radius:radius)
            jx = ix + offset
            1 <= jx <= nx || continue
            acc += kernel[ki] * values[jx, iy]
        end
        tmp[ix, iy] = acc
    end
    for ix in 1:nx, iy in 1:ny
        acc = 0.0
        for (ki, offset) in enumerate(-radius:radius)
            jy = iy + offset
            1 <= jy <= ny || continue
            acc += kernel[ki] * tmp[ix, jy]
        end
        out[ix, iy] = acc
    end
    return out
end

"Estimate the normalized ensemble-averaged Cartesian density."
function estimate_density(spectra::AbstractVector;
                          n_bins::Integer = 256,
                          quantile_bounds = (0.001, 0.999),
                          pad_fraction::Real = 0.05,
                          smooth_sigma::Real = 2.0)
    n_bins = Int(n_bins)
    n_bins >= 8 || throw(ArgumentError("n_bins must be at least 8"))
    qlo, qhi = Float64.(quantile_bounds)
    0 <= qlo < qhi <= 1 || throw(ArgumentError("invalid quantile bounds"))
    points = ComplexF64[]
    sizehint!(points, sum(length, spectra))
    for seed_spectrum in spectra
        append!(points, ComplexF64.(seed_spectrum))
    end
    isempty(points) && throw(ArgumentError("spectra cannot be empty"))
    xs = real.(points)
    ys = imag.(points)
    all(isfinite, xs) && all(isfinite, ys) ||
        throw(ArgumentError("spectra must be finite"))

    xlo, xhi = quantile(xs, qlo), quantile(xs, qhi)
    ylo, yhi = quantile(ys, qlo), quantile(ys, qhi)
    function padded_bounds(lo, hi)
        width = hi - lo
        if !(width > 0)
            width = max(abs(lo), 1.0) * 1e-6
        end
        padding = Float64(pad_fraction) * width
        return lo - padding, hi + padding
    end
    xlo, xhi = padded_bounds(xlo, xhi)
    ylo, yhi = padded_bounds(ylo, yhi)
    dx = (xhi - xlo) / n_bins
    dy = (yhi - ylo) / n_bins
    counts = zeros(Float64, n_bins, n_bins)
    for z in points
        x, y = real(z), imag(z)
        (xlo <= x <= xhi && ylo <= y <= yhi) || continue
        ix = clamp(floor(Int, (x - xlo) / dx) + 1, 1, n_bins)
        iy = clamp(floor(Int, (y - ylo) / dy) + 1, 1, n_bins)
        counts[ix, iy] += 1
    end
    sum(counts) > 0 || throw(ArgumentError("no points inside density bounds"))
    smoothed = _smooth_histogram(counts, Float64(smooth_sigma))
    smoothed ./= sum(smoothed) * dx * dy
    xcenters = collect(range(xlo + dx / 2; step = dx, length = n_bins))
    ycenters = collect(range(ylo + dy / 2; step = dy, length = n_bins))
    return DensityGrid(xcenters, ycenters, smoothed)
end

"Real coordinate of the raw density mode on the grid row closest to `y=0`."
function raw_real_axis_mode(grid::DensityGrid)
    iy = argmin(abs.(grid.y))
    ix = argmax(@view grid.values[:, iy])
    return grid.x[ix]
end

function _linear_crossing(x1, y1, x2, y2, target)
    y1 == y2 && return (x1 + x2) / 2
    t = (target - y1) / (y2 - y1)
    return x1 + clamp(t, 0.0, 1.0) * (x2 - x1)
end

function _halfmax_crossings(axis::Vector{Float64}, cut::Vector{Float64},
                            peak_index::Int, halfmax::Float64)
    left = nothing
    for i in (peak_index - 1):-1:1
        if cut[i] <= halfmax <= cut[i + 1] || cut[i] >= halfmax >= cut[i + 1]
            left = _linear_crossing(axis[i], cut[i], axis[i + 1], cut[i + 1],
                                    halfmax)
            break
        end
    end
    right = nothing
    for i in peak_index:(length(axis) - 1)
        if cut[i] >= halfmax >= cut[i + 1] || cut[i] <= halfmax <= cut[i + 1]
            right = _linear_crossing(axis[i], cut[i], axis[i + 1], cut[i + 1],
                                     halfmax)
            break
        end
    end
    left === nothing && throw(ArgumentError("density has no left HWHM crossing"))
    right === nothing && throw(ArgumentError("density has no right HWHM crossing"))
    right > left || throw(ArgumentError("invalid HWHM crossing order"))
    return Float64(left), Float64(right)
end

"Return `(peak, (Delta_x,Delta_y))` from interpolated line-cut HWHM values."
function density_peak_hwhm(grid::DensityGrid)
    all(isfinite, grid.values) || throw(ArgumentError("density must be finite"))
    peak_value = maximum(grid.values)
    peak_value > 0 || throw(ArgumentError("density maximum must be positive"))

    # Deterministic conjugate-lobe tie break: prefer y <= 0, then smallest
    # |y|, then the lexicographically first grid cell.
    tol = max(eps(peak_value) * 16, abs(peak_value) * 1e-14)
    candidates = Tuple{Int,Int}[]
    for ix in eachindex(grid.x), iy in eachindex(grid.y)
        abs(grid.values[ix, iy] - peak_value) <= tol && push!(candidates, (ix, iy))
    end
    sort!(candidates; by = ij -> begin
        y = grid.y[ij[2]]
        (y > 0 ? 1 : 0, abs(y), ij[1], ij[2])
    end)
    ix, iy = first(candidates)
    halfmax = peak_value / 2
    xl, xr = _halfmax_crossings(grid.x, vec(grid.values[:, iy]), ix, halfmax)
    yl, yr = _halfmax_crossings(grid.y, vec(grid.values[ix, :]), iy, halfmax)
    return (grid.x[ix], grid.y[iy]), ((xr - xl) / 2, (yr - yl) / 2)
end

function filter_weight(z::Complex, filter::GaussianFilter)
    dx = real(z) - filter.mu_x
    dy = imag(z) - filter.mu_y
    return exp(-filter.alpha_x * dx^2 - filter.alpha_y * dy^2)
end

filter_weights(zs::AbstractVector{<:Complex}, filter::GaussianFilter) =
    Float64[filter_weight(z, filter) for z in zs]

function effective_count(weights::AbstractVector{<:Real})
    s1 = sum(weights)
    s2 = sum(abs2, weights)
    s2 > 0 || return 0.0
    return Float64(s1^2 / s2)
end

function _ring_cells(ix::Int, iy::Int, radius::Int)
    radius == 0 && return Tuple{Int,Int}[(ix, iy)]
    cells = Tuple{Int,Int}[]
    lo_x, hi_x = ix - radius, ix + radius
    lo_y, hi_y = iy - radius, iy + radius
    for cx in lo_x:hi_x
        push!(cells, (cx, lo_y))
        push!(cells, (cx, hi_y))
    end
    for cy in (lo_y + 1):(hi_y - 1)
        push!(cells, (lo_x, cy))
        push!(cells, (hi_x, cy))
    end
    return cells
end

"Exact nearest-neighbour indices and distances from a dependency-free cell list."
function exact_nearest_neighbours(zs::AbstractVector{<:Complex})
    n = length(zs)
    n >= 2 || throw(ArgumentError("at least two points are required"))
    points = ComplexF64.(zs)
    xs = real.(points)
    ys = imag.(points)
    all(isfinite, xs) && all(isfinite, ys) ||
        throw(ArgumentError("points must be finite"))
    xmin, xmax = extrema(xs)
    ymin, ymax = extrema(ys)
    span = max(xmax - xmin, ymax - ymin)
    cell_size = span > 0 ? span / max(sqrt(n), 1.0) : 1.0

    cell_of(x, y) = (floor(Int, (x - xmin) / cell_size),
                     floor(Int, (y - ymin) / cell_size))
    cells = Dict{Tuple{Int,Int},Vector{Int}}()
    point_cells = Vector{Tuple{Int,Int}}(undef, n)
    for i in eachindex(points)
        key = cell_of(xs[i], ys[i])
        point_cells[i] = key
        push!(get!(cells, key, Int[]), i)
    end
    occupied = collect(keys(cells))
    min_cx = minimum(first, occupied)
    max_cx = maximum(first, occupied)
    min_cy = minimum(last, occupied)
    max_cy = maximum(last, occupied)

    nearest = zeros(Int, n)
    distances = fill(Inf, n)
    for i in eachindex(points)
        ix, iy = point_cells[i]
        radius = 0
        while true
            for key in _ring_cells(ix, iy, radius)
                for j in get(cells, key, Int[])
                    i == j && continue
                    d = abs(points[i] - points[j])
                    if d < distances[i] ||
                       (d == distances[i] && (nearest[i] == 0 || j < nearest[i]))
                        distances[i] = d
                        nearest[i] = j
                    end
                end
            end

            covers_all = ix - radius <= min_cx && ix + radius >= max_cx &&
                         iy - radius <= min_cy && iy + radius >= max_cy
            if isfinite(distances[i])
                left = xmin + (ix - radius) * cell_size
                right = xmin + (ix + radius + 1) * cell_size
                bottom = ymin + (iy - radius) * cell_size
                top = ymin + (iy + radius + 1) * cell_size
                outside_lower_bound = minimum((xs[i] - left, right - xs[i],
                                               ys[i] - bottom, top - ys[i]))
                distances[i] <= max(outside_lower_bound, 0.0) && break
            end
            covers_all && break
            radius += 1
        end
        nearest[i] != 0 || error("nearest-neighbour search failed")
    end
    return nearest, distances
end

function weighted_nearest_spacing(zs::AbstractVector{<:Complex},
                                  weights::AbstractVector{<:Real})
    length(zs) == length(weights) || throw(DimensionMismatch("weights mismatch"))
    nearest, distances = exact_nearest_neighbours(zs)
    return mean(Float64(distances[i] * weights[i] * weights[nearest[i]])
                for i in eachindex(zs))
end

function ensemble_weighted_spacing(spectra::AbstractVector, spec::UnfoldingSpec)
    transformed = mapped_spectra(spectra, spec)
    per_seed = zeros(Float64, length(transformed))
    Threads.@threads for r in eachindex(transformed)
        weights = filter_weights(transformed[r], spec.filter)
        per_seed[r] = weighted_nearest_spacing(transformed[r], weights)
    end
    all(isfinite, per_seed) && all(>(0), per_seed) ||
        throw(ArgumentError("weighted spacings must be finite and positive"))
    return (mean = mean(per_seed), per_seed = per_seed)
end

function _flatness_score(grid::DensityGrid, filter::GaussianFilter)
    samples = Float64[]
    for ix in eachindex(grid.x), iy in eachindex(grid.y)
        abs(grid.x[ix] - filter.mu_x) <= filter.delta_x || continue
        abs(grid.y[iy] - filter.mu_y) <= filter.delta_y || continue
        push!(samples, grid.values[ix, iy])
    end
    length(samples) >= 4 || throw(ArgumentError("too few density bins in HWHM region"))
    m = mean(samples)
    m > 0 || throw(ArgumentError("nonpositive density in HWHM region"))
    return std(samples; corrected = false) / m
end

"Fit one declared conformal map and its ensemble-level Gaussian filter."
function fit_unfolding_candidate(spectra::AbstractVector,
                                 map::PowerMap;
                                 n_bins::Integer = 256,
                                 quantile_bounds = (0.001, 0.999),
                                 pad_fraction::Real = 0.05,
                                 smooth_sigma::Real = 2.0)
    transformed = [apply_unfolding(seed_spectrum, map) for seed_spectrum in spectra]
    density = estimate_density(transformed; n_bins = n_bins,
                               quantile_bounds = quantile_bounds,
                               pad_fraction = pad_fraction,
                               smooth_sigma = smooth_sigma)
    peak, widths = density_peak_hwhm(density)
    filter = GaussianFilter(peak[1], peak[2], widths[1], widths[2])
    n_eff = Float64[effective_count(filter_weights(seed_spectrum, filter))
                    for seed_spectrum in transformed]
    flatness = _flatness_score(density, filter)
    beta = map.beta
    scores = NamedTuple[(beta = beta, flatness = flatness,
                         n_eff = median(n_eff), selected = true)]
    return UnfoldingSpec(map, filter, density, flatness,
                         median(n_eff), n_eff, scores)
end

"Fit a declared non-adaptive power map and its mapped-DOS Gaussian filter."
function fit_fixed_power_policy(spectra::AbstractVector, beta::Real;
                                n_bins::Integer = 256,
                                quantile_bounds = (0.001, 0.999),
                                pad_fraction::Real = 0.05,
                                smooth_sigma::Real = 2.0,
                                minimum_effective_count::Real = 500.0)
    b = Float64(beta)
    0 < b <= 1 || throw(ArgumentError("beta must lie in (0,1]"))
    raw_density = estimate_density(spectra; n_bins = n_bins,
                                   quantile_bounds = quantile_bounds,
                                   pad_fraction = pad_fraction,
                                   smooth_sigma = smooth_sigma)
    z0 = raw_real_axis_mode(raw_density)
    fit = fit_unfolding_candidate(spectra, PowerMap(b, z0);
                                  n_bins = n_bins,
                                  quantile_bounds = quantile_bounds,
                                  pad_fraction = pad_fraction,
                                  smooth_sigma = smooth_sigma)
    fit.n_eff_median >= minimum_effective_count ||
        throw(ArgumentError("fixed-power candidate fails N_eff threshold"))
    return fit
end

"Per-seed filtered spectral traces and their filter weights."
function partition_traces(spectra::AbstractVector,
                          filter::GaussianFilter,
                          times::AbstractVector{<:Real}, theta::Real)
    R = length(spectra)
    R >= 1 || throw(ArgumentError("at least one spectrum is required"))
    nt = length(times)
    Z = zeros(ComplexF64, R, nt)
    weights_by_seed = Vector{Vector{Float64}}(undef, R)
    ctheta, stheta = cos(Float64(theta)), sin(Float64(theta))
    Threads.@threads for r in 1:R
        zs = ComplexF64.(spectra[r])
        ws = filter_weights(zs, filter)
        weights_by_seed[r] = ws
        projection = real.(zs) .* ctheta .+ imag.(zs) .* stheta
        for ti in eachindex(times)
            t = Float64(times[ti])
            acc = 0.0 + 0.0im
            @inbounds for j in eachindex(zs)
                acc += ws[j] * cis(t * projection[j])
            end
            Z[r, ti] = acc
        end
    end
    return Z, weights_by_seed
end

"Unbiased `R/(R-1)` connected ensemble form factor from per-seed traces."
function connected_dsff(Z::AbstractMatrix{<:Complex})
    R, nt = size(Z)
    R >= 2 || throw(ArgumentError("connected estimator requires at least two seeds"))
    out = zeros(Float64, nt)
    correction = R / (R - 1)
    for j in 1:nt
        m = sum(@view Z[:, j]) / R
        m2 = sum(abs2, @view Z[:, j]) / R
        out[j] = correction * (m2 - abs2(m))
    end
    return out
end

function filtered_plateau(weights_by_seed::AbstractVector)
    isempty(weights_by_seed) && throw(ArgumentError("weights cannot be empty"))
    return mean(sum(abs2, weights) for weights in weights_by_seed)
end

"""
    projected_plateau_by_seed(spectra, weights_by_seed, theta;
                              conjugate_tolerance=1e-8)

Return the infinite-time diagonal contribution after grouping levels with the
same coordinate projected onto the DSFF ray. Generic rays group only exactly
equal floating-point projections. On the real ray of a BDI-dagger spectrum,
numerically resolved conjugate pairs are matched explicitly because they have
the same mathematical real projection and contribute a coherent cross term.
"""
function projected_plateau_by_seed(spectra::AbstractVector,
                                   weights_by_seed::AbstractVector,
                                   theta::Real;
                                   conjugate_tolerance::Real = 1e-8)
    length(spectra) == length(weights_by_seed) ||
        throw(DimensionMismatch("spectra and weights must have equal seed counts"))
    isempty(spectra) && throw(ArgumentError("at least one spectrum is required"))
    conjugate_tolerance >= 0 ||
        throw(ArgumentError("conjugate_tolerance must be nonnegative"))
    ctheta, stheta = cos(Float64(theta)), sin(Float64(theta))
    result = zeros(Float64, length(spectra))
    for r in eachindex(spectra)
        zs = ComplexF64.(spectra[r])
        weights = Float64.(weights_by_seed[r])
        length(zs) == length(weights) ||
            throw(DimensionMismatch("spectrum and weight lengths differ for seed $r"))
        isempty(zs) && continue
        if abs(stheta) <= 64eps(Float64) &&
           abs(abs(ctheta) - 1.0) <= 64eps(Float64)
            # BDI-dagger conjugation produces an exact degeneracy of the real
            # projection. Pair in the complex plane, where an absolute
            # tolerance scaled to the spectrum is meaningful and cannot merge
            # unrelated nearby projected levels.
            result[r] = sum(abs2, weights)
            scale = max(1.0, maximum(abs, zs))
            tolerance = Float64(conjugate_tolerance) * scale
            upper = findall(z -> imag(z) > 0, zs)
            lower = findall(z -> imag(z) < 0, zs)
            order = sortperm(lower; by = i -> real(zs[i]))
            lower = lower[order]
            lower_real = real.(zs[lower])
            used = falses(length(lower))
            for i in upper
                target = conj(zs[i])
                lo = searchsortedfirst(lower_real, real(target) - tolerance)
                hi = searchsortedlast(lower_real, real(target) + tolerance)
                best = 0
                best_distance = Inf
                for position in lo:hi
                    used[position] && continue
                    distance = abs(zs[lower[position]] - target)
                    if distance < best_distance
                        best = position
                        best_distance = distance
                    end
                end
                if best != 0 && best_distance <= tolerance
                    j = lower[best]
                    result[r] += 2 * weights[i] * weights[j]
                    used[best] = true
                end
            end
        else
            projection = real.(zs) .* ctheta .+ imag.(zs) .* stheta
            groups = Dict{Float64,Float64}()
            for i in eachindex(projection)
                groups[projection[i]] = get(groups, projection[i], 0.0) + weights[i]
            end
            result[r] = sum(abs2, values(groups))
        end
    end
    return result
end

"Ensemble mean of the ray-resolved infinite-time filtered plateau."
projected_plateau(spectra::AbstractVector, weights_by_seed::AbstractVector,
                  theta::Real; kwargs...) =
    mean(projected_plateau_by_seed(spectra, weights_by_seed, theta; kwargs...))

function _curve_quantiles(samples::Matrix{Float64})
    nt = size(samples, 2)
    med = zeros(Float64, nt)
    lower = zeros(Float64, nt)
    upper = zeros(Float64, nt)
    for j in 1:nt
        column = @view samples[:, j]
        med[j] = quantile(column, 0.5)
        lower[j] = quantile(column, 0.16)
        upper[j] = quantile(column, 0.84)
    end
    return med, lower, upper
end

"Cluster bootstrap the plateau-normalized connected DSFF over seed rows."
function bootstrap_connected_dsff(Z::AbstractMatrix{<:Complex},
                                  plateau_by_seed::AbstractVector{<:Real};
                                  n_boot::Integer = 500,
                                  rng::AbstractRNG = MersenneTwister(0))
    R, nt = size(Z)
    length(plateau_by_seed) == R || throw(DimensionMismatch("plateau mismatch"))
    R >= 2 || throw(ArgumentError("at least two seeds are required"))
    n_boot = Int(n_boot)
    n_boot > 0 || throw(ArgumentError("n_boot must be positive"))
    plateau = mean(plateau_by_seed)
    plateau > 0 || throw(ArgumentError("plateau must be positive"))
    curve = connected_dsff(Z) ./ plateau
    samples = zeros(Float64, n_boot, nt)
    # A bootstrap draw samples from the finite empirical distribution, whose
    # variance is (R-1)/R times the unbiased sample variance represented by the
    # base connected estimator.  Re-center each draw onto the same unbiased
    # convention before forming percentile ribbons.
    bootstrap_centering = R / (R - 1)
    for b in 1:n_boot
        indices = rand(rng, 1:R, R)
        p = mean(@view plateau_by_seed[indices])
        samples[b, :] .= bootstrap_centering .*
                         connected_dsff(@view Z[indices, :]) ./ p
    end
    med, lower, upper = _curve_quantiles(samples)
    return (curve = curve, median = med, lower = lower, upper = upper,
            samples = samples, plateau = plateau)
end

"Pool-adjacent-violators isotonic regression with unit weights."
function isotonic_nondecreasing(values::AbstractVector{<:Real})
    isempty(values) && return Float64[]
    means = Float64[]
    weights = Float64[]
    starts = Int[]
    stops = Int[]
    for (i, value) in enumerate(values)
        push!(means, Float64(value))
        push!(weights, 1.0)
        push!(starts, i)
        push!(stops, i)
        while length(means) >= 2 && means[end - 1] > means[end]
            w = weights[end - 1] + weights[end]
            m = (means[end - 1] * weights[end - 1] +
                 means[end] * weights[end]) / w
            means[end - 1] = m
            weights[end - 1] = w
            stops[end - 1] = stops[end]
            pop!(means); pop!(weights); pop!(starts); pop!(stops)
        end
    end
    fitted = similar(Float64.(values))
    for b in eachindex(means)
        fitted[starts[b]:stops[b]] .= means[b]
    end
    return fitted
end

function _log_bin_curve(u::AbstractVector{<:Real}, curve::AbstractVector{<:Real},
                        bins_per_decade::Int)
    length(u) == length(curve) || throw(DimensionMismatch("curve mismatch"))
    bins_per_decade > 0 || throw(ArgumentError("bins_per_decade must be positive"))
    keep = [i for i in eachindex(u) if isfinite(u[i]) && u[i] > 0 &&
            isfinite(curve[i])]
    length(keep) >= 2 || throw(ArgumentError("insufficient finite calibration data"))
    logs = log10.(Float64.(u[keep]))
    lo = minimum(logs)
    bin_ids = floor.(Int, (logs .- lo) .* bins_per_decade)
    ubin = Float64[]
    ybin = Float64[]
    for b in minimum(bin_ids):maximum(bin_ids)
        local_indices = findall(==(b), bin_ids)
        isempty(local_indices) && continue
        source_indices = keep[local_indices]
        push!(ubin, 10.0^mean(log10.(Float64.(u[source_indices]))))
        push!(ybin, median(Float64.(curve[source_indices])))
    end
    return ubin, ybin
end

"Calibrate the dimensionless BDI-dagger plateau-onset constant."
function calibrate_heisenberg(u::AbstractVector{<:Real},
                              normalized_curve::AbstractVector{<:Real};
                              threshold::Real = 0.95,
                              bins_per_decade::Integer = 20,
                              ramp_search_fraction::Real = 0.70)
    0 < threshold < 1 || throw(ArgumentError("threshold must lie in (0,1)"))
    0 < ramp_search_fraction <= 1 ||
        throw(ArgumentError("ramp_search_fraction must lie in (0,1]"))
    ubin, ybin = _log_bin_curve(u, normalized_curve, Int(bins_per_decade))
    # Filtering produces an early weighted-count-variance bump.  The
    # Heisenberg plateau belongs to the ramp after the dip, so an isotonic fit
    # over the full curve can never be used: a broad early bump can force an
    # artificial crossing at the first sampled time.  Locate the dip within
    # the pre-plateau portion of the logarithmic grid and fit only from there.
    search_stop = clamp(floor(Int, ramp_search_fraction * length(ybin)),
                        1, length(ybin))
    ramp_start = argmin(@view ybin[1:search_stop])
    ramp_u = ubin[ramp_start:end]
    ramp_raw = ybin[ramp_start:end]
    fit = isotonic_nondecreasing(ramp_raw)
    crossing = findfirst(>=(Float64(threshold)), fit)
    crossing === nothing &&
        throw(ArgumentError("calibration curve does not reach threshold"))
    if crossing == 1 || fit[crossing] == fit[crossing - 1]
        chi = ramp_u[crossing]
    else
        fraction = (Float64(threshold) - fit[crossing - 1]) /
                   (fit[crossing] - fit[crossing - 1])
        logchi = log(ramp_u[crossing - 1]) + fraction *
                 (log(ramp_u[crossing]) - log(ramp_u[crossing - 1]))
        chi = exp(logchi)
    end
    return (chi = chi, u = ramp_u, raw = ramp_raw, fit = fit,
            threshold = Float64(threshold), ramp_start_u = first(ramp_u))
end

"Bootstrap the L3b BDI-dagger Heisenberg calibration over seed rows."
function bootstrap_heisenberg(Z::AbstractMatrix{<:Complex},
                              plateau_by_seed::AbstractVector{<:Real},
                              u::AbstractVector{<:Real};
                              n_boot::Integer = 500,
                              rng::AbstractRNG = MersenneTwister(0),
                              threshold::Real = 0.95,
                              bins_per_decade::Integer = 20)
    R, nt = size(Z)
    nt == length(u) || throw(DimensionMismatch("time grid mismatch"))
    length(plateau_by_seed) == R || throw(DimensionMismatch("plateau mismatch"))
    base_curve = connected_dsff(Z) ./ mean(plateau_by_seed)
    calibration = calibrate_heisenberg(u, base_curve; threshold = threshold,
                                       bins_per_decade = bins_per_decade)
    samples = Float64[]
    bootstrap_centering = R / (R - 1)
    for _ in 1:Int(n_boot)
        indices = rand(rng, 1:R, R)
        curve = bootstrap_centering .*
                connected_dsff(@view Z[indices, :]) ./
                mean(@view plateau_by_seed[indices])
        try
            cal = calibrate_heisenberg(u, curve; threshold = threshold,
                                       bins_per_decade = bins_per_decade)
            isfinite(cal.chi) && push!(samples, cal.chi)
        catch err
            err isa ArgumentError || rethrow()
        end
    end
    minimum_required = min(Int(n_boot), max(10, ceil(Int, n_boot / 2)))
    length(samples) >= minimum_required ||
        throw(ArgumentError("too few finite bootstrap Heisenberg calibrations"))
    return (calibration = calibration,
            median = quantile(samples, 0.5),
            lower = quantile(samples, 0.16),
            upper = quantile(samples, 0.84),
            samples = samples)
end

end # module
