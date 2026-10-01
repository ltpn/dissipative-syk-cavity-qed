module RawConnectedFormFactors

using Random: AbstractRNG, MersenneTwister, rand
using Statistics: mean, quantile

export remove_steady_modes,
       raw_traces,
       unit_weight_plateau_by_seed,
       connected_form_factor,
       bootstrap_connected_form_factor,
       analyze_sigma_sff,
       analyze_dsff,
       positive_or_nan

"Remove only complex levels whose modulus does not exceed `tolerance`."
function remove_steady_modes(spectrum; tolerance::Real = 1e-8)
    tolerance >= 0 ||
        throw(ArgumentError("steady tolerance must be nonnegative"))
    values = ComplexF64.(spectrum)
    all(isfinite, values) ||
        throw(ArgumentError("spectrum contains non-finite values"))
    return values[abs.(values) .> Float64(tolerance)]
end

"Evaluate unit-weight spectral traces for seed-resolved real coordinates."
function raw_traces(levels_by_seed, times)
    isempty(levels_by_seed) &&
        throw(ArgumentError("at least one seed is required"))
    ts = Float64.(times)
    all(isfinite, ts) || throw(ArgumentError("times must be finite"))

    levels = Vector{Vector{Float64}}(undef, length(levels_by_seed))
    for r in eachindex(levels_by_seed)
        xs = Float64.(levels_by_seed[r])
        isempty(xs) && throw(ArgumentError("seed $r has no levels"))
        all(isfinite, xs) ||
            throw(ArgumentError("seed $r has non-finite levels"))
        levels[r] = xs
    end

    Z = Matrix{ComplexF64}(undef, length(levels), length(ts))
    Threads.@threads for r in eachindex(levels)
        xs = levels[r]
        for ti in eachindex(ts)
            acc = 0.0 + 0.0im
            t = ts[ti]
            @inbounds for x in xs
                acc += cis(t * x)
            end
            Z[r, ti] = acc
        end
    end
    return Z
end

"Exact unit-weight diagonal plateau, including exact coordinate multiplicities."
function unit_weight_plateau_by_seed(levels_by_seed)
    isempty(levels_by_seed) &&
        throw(ArgumentError("at least one seed is required"))
    result = Vector{Float64}(undef, length(levels_by_seed))
    for r in eachindex(levels_by_seed)
        xs = Float64.(levels_by_seed[r])
        isempty(xs) && throw(ArgumentError("seed $r has no levels"))
        all(isfinite, xs) ||
            throw(ArgumentError("seed $r has non-finite levels"))
        counts = Dict{Float64,Int}()
        for x in xs
            counts[x] = get(counts, x, 0) + 1
        end
        result[r] = sum(Float64(count)^2 for count in values(counts))
    end
    return result
end

"Unbiased ensemble-connected form factor normalized by the mean plateau."
function connected_form_factor(Z::AbstractMatrix{<:Complex}, plateau_by_seed)
    R, nt = size(Z)
    R >= 2 || throw(ArgumentError("at least two seeds are required"))
    length(plateau_by_seed) == R ||
        throw(DimensionMismatch("plateau mismatch"))
    plateaus = Float64.(plateau_by_seed)
    all(isfinite, plateaus) ||
        throw(ArgumentError("plateaus must be finite"))
    plateau = mean(plateaus)
    plateau > 0 || throw(ArgumentError("mean plateau must be positive"))
    correction = R / (R - 1)
    curve = Vector{Float64}(undef, nt)
    for ti in 1:nt
        column = @view Z[:, ti]
        curve[ti] = correction *
            (mean(abs2, column) - abs2(mean(column))) / plateau
    end
    return curve
end

"Whole-seed bootstrap of the plateau-normalized connected form factor."
function bootstrap_connected_form_factor(
        Z::AbstractMatrix{<:Complex}, plateau_by_seed;
        n_boot::Integer = 500,
        rng::AbstractRNG = MersenneTwister(0))
    R, nt = size(Z)
    R >= 2 || throw(ArgumentError("at least two seeds are required"))
    n_boot > 0 || throw(ArgumentError("n_boot must be positive"))
    plateaus = Float64.(plateau_by_seed)
    length(plateaus) == R || throw(DimensionMismatch("plateau mismatch"))

    curve = connected_form_factor(Z, plateaus)
    samples = Matrix{Float64}(undef, Int(n_boot), nt)
    recenter = R / (R - 1)
    for b in 1:Int(n_boot)
        indices = rand(rng, 1:R, R)
        samples[b, :] .= recenter .* connected_form_factor(
            @view(Z[indices, :]), @view(plateaus[indices]))
    end
    curve_quantile(p) = Float64[
        quantile(@view(samples[:, ti]), p) for ti in 1:nt]
    return (
        curve = curve,
        median = curve_quantile(0.5),
        lower = curve_quantile(0.16),
        upper = curve_quantile(0.84),
        samples = samples,
        plateau = mean(plateaus),
    )
end

function _analyze_levels(levels, times; n_boot, rng)
    isempty(levels) && throw(ArgumentError("at least one seed is required"))
    counts = length.(levels)
    all(==(first(counts)), counts) ||
        throw(ArgumentError("all seeds must have the same level count"))
    Z = raw_traces(levels, times)
    plateau = unit_weight_plateau_by_seed(levels)
    bootstrap = bootstrap_connected_form_factor(
        Z, plateau; n_boot = n_boot, rng = rng)
    return merge(bootstrap, (
        traces = Z,
        level_counts = counts,
        plateau_by_seed = plateau,
    ))
end

"Analyze every supplied trace-centered singular value with unit weight."
function analyze_sigma_sff(
        sigmas, times;
        n_boot::Integer = 500,
        rng::AbstractRNG = MersenneTwister(0))
    levels = [Float64.(values) for values in sigmas]
    return _analyze_levels(levels, times; n_boot = n_boot, rng = rng)
end

"Analyze raw nonsteady complex levels on the requested DSFF ray."
function analyze_dsff(
        spectra, times;
        theta::Real = pi / 4,
        steady_tolerance::Real = 1e-8,
        n_boot::Integer = 500,
        rng::AbstractRNG = MersenneTwister(0))
    filtered = [
        remove_steady_modes(values; tolerance = steady_tolerance)
        for values in spectra
    ]
    removed = [
        length(spectra[i]) - length(filtered[i]) for i in eachindex(spectra)
    ]
    ctheta, stheta = cos(Float64(theta)), sin(Float64(theta))
    projected = [
        Float64[real(z) * ctheta + imag(z) * stheta for z in values]
        for values in filtered
    ]
    analysis = _analyze_levels(projected, times; n_boot = n_boot, rng = rng)
    return merge(analysis, (
        removed_counts = removed,
        projected_levels = projected,
    ))
end

"Mask values that cannot be drawn on a logarithmic axis without clipping."
positive_or_nan(values) = Float64[
    isfinite(value) && value > 0 ? Float64(value) : NaN
    for value in values
]

end
