module ComplexDSFFAIDaggerReference

using JLD2
using LinearAlgebra
using Random
using SHA
using Statistics: mean, quantile

include(joinpath(@__DIR__, "atomic_jld2.jl"))
include(joinpath(@__DIR__, "ComplexDSFFUnfolding.jl"))
using .ComplexDSFFUnfolding

export gaussian_ai_dagger_matrix,
       circular_filter_weights,
       gaussian_filter_effective_density,
       linear_inverse_size_extrapolation,
       extrapolate_reference_curves,
       density_scaled_reference_curves,
       calibrate_extrapolated_reference,
       compute_finite_size_reference,
       write_ai_dagger_reference,
       load_ai_dagger_reference,
       ai_dagger_reference_token

"Generate an invariant Gaussian complex-symmetric (AI-dagger) matrix."
function gaussian_ai_dagger_matrix(rng::AbstractRNG, n::Integer)
    dimension = Int(n)
    dimension > 0 || throw(ArgumentError("matrix dimension must be positive"))
    scale = inv(sqrt(2dimension))
    independent = scale .* (
        randn(rng, dimension, dimension) .+
        im .* randn(rng, dimension, dimension))
    return Matrix{ComplexF64}((independent .+ transpose(independent)) ./ sqrt(2))
end

"Analytical circular-law Gaussian-filter weights at strength alpha_tilde."
function circular_filter_weights(zs::AbstractVector{<:Complex};
                                 alpha_tilde::Real = 1.0)
    alpha = Float64(alpha_tilde)
    alpha > 0 || throw(ArgumentError("alpha_tilde must be positive"))
    return exp.(-alpha .* abs2.(ComplexF64.(zs)))
end

"Infer the local two-dimensional density from a Gaussian-filter plateau."
function gaussian_filter_effective_density(plateau::Real,
                                           alpha_x::Real,
                                           alpha_y::Real)
    value = Float64(plateau)
    ax = Float64(alpha_x)
    ay = Float64(alpha_y)
    isfinite(value) && value > 0 ||
        throw(ArgumentError("plateau must be finite and positive"))
    isfinite(ax) && ax > 0 ||
        throw(ArgumentError("alpha_x must be finite and positive"))
    isfinite(ay) && ay > 0 ||
        throw(ArgumentError("alpha_y must be finite and positive"))
    # For locally constant density rho and
    # f = exp(-alpha_x*x^2-alpha_y*y^2),
    # P = rho * integral(f^2) = rho*pi/(2sqrt(alpha_x*alpha_y)).
    return 2value * sqrt(ax * ay) / pi
end

function _log_interpolate(x::AbstractVector{<:Real},
                          y::AbstractVector{<:Real},
                          targets::AbstractVector{<:Real})
    length(x) == length(y) ||
        throw(DimensionMismatch("interpolation coordinates differ"))
    length(x) >= 2 || throw(ArgumentError("interpolation grid is too short"))
    source = Float64.(x)
    values = Float64.(y)
    destination = Float64.(targets)
    all(isfinite, source) && all(>(0), source) && issorted(source) &&
        all(diff(source) .> 0) ||
        throw(ArgumentError("interpolation grid must be finite, positive, and increasing"))
    all(isfinite, values) ||
        throw(ArgumentError("interpolation values must be finite"))
    all(isfinite, destination) && all(>(0), destination) ||
        throw(ArgumentError("interpolation targets must be finite and positive"))
    tolerance = 64eps(Float64)
    first(destination) >= first(source) * (1 - tolerance) &&
        last(destination) <= last(source) * (1 + tolerance) ||
        throw(ArgumentError("interpolation targets extend outside the source grid"))

    log_source = log.(source)
    output = similar(destination)
    for (index, target) in pairs(destination)
        log_target = log(target)
        right = searchsortedfirst(log_source, log_target)
        if right <= 1
            output[index] = values[1]
        elseif right > length(source)
            output[index] = values[end]
        else
            left = right - 1
            fraction = (log_target - log_source[left]) /
                       (log_source[right] - log_source[left])
            output[index] = muladd(fraction,
                                   values[right] - values[left],
                                   values[left])
        end
    end
    return output
end

function _size_parameters(value::Real, count::Integer, name::AbstractString)
    result = fill(Float64(value), Int(count))
    all(entry -> isfinite(entry) && entry > 0, result) ||
        throw(ArgumentError("$name must be finite and positive"))
    return result
end

function _size_parameters(values::AbstractVector{<:Real}, count::Integer,
                          name::AbstractString)
    length(values) == count ||
        throw(DimensionMismatch("one $name is required per matrix size"))
    result = Float64.(values)
    all(entry -> isfinite(entry) && entry > 0, result) ||
        throw(ArgumentError("$name must be finite and positive"))
    return result
end

"Fit `values[N_index, point] = intercept[point] + slope[point]/N`."
function linear_inverse_size_extrapolation(sizes::AbstractVector{<:Integer},
                                           values::AbstractMatrix{<:Real})
    length(sizes) == size(values, 1) ||
        throw(DimensionMismatch("one value row is required per matrix size"))
    length(sizes) >= 2 ||
        throw(ArgumentError("at least two matrix sizes are required"))
    all(>(0), sizes) || throw(ArgumentError("matrix sizes must be positive"))
    length(unique(sizes)) == length(sizes) ||
        throw(ArgumentError("matrix sizes must be distinct"))
    all(isfinite, values) ||
        throw(ArgumentError("extrapolation values must be finite"))

    inverse_sizes = inv.(Float64.(sizes))
    x_mean = mean(inverse_sizes)
    centered = inverse_sizes .- x_mean
    denominator = sum(abs2, centered)
    denominator > 0 || throw(ArgumentError("inverse-size grid is singular"))
    matrix = Float64.(values)
    y_mean = vec(mean(matrix; dims = 1))
    slope = vec(transpose(centered) * matrix) ./ denominator
    intercept = y_mean .- slope .* x_mean
    return (intercept = intercept, slope = slope,
            inverse_sizes = inverse_sizes)
end

"Extrapolate central and seed-bootstrap finite-size curves on a common u grid."
function extrapolate_reference_curves(
        sizes::AbstractVector{<:Integer},
        u::AbstractVector{<:Real},
        curves::AbstractVector,
        samples::AbstractVector)
    n_sizes = length(sizes)
    length(curves) == n_sizes ||
        throw(DimensionMismatch("one central curve is required per size"))
    length(samples) == n_sizes ||
        throw(DimensionMismatch("one bootstrap matrix is required per size"))
    n_points = length(u)
    n_points > 0 || throw(ArgumentError("u grid cannot be empty"))
    all(curve -> length(curve) == n_points, curves) ||
        throw(DimensionMismatch("finite-size curve grids differ"))
    n_boot = size(first(samples), 1)
    n_boot > 0 || throw(ArgumentError("bootstrap matrices cannot be empty"))
    all(matrix -> size(matrix) == (n_boot, n_points), samples) ||
        throw(DimensionMismatch("finite-size bootstrap grids differ"))

    central_values = reduce(vcat,
        [permutedims(Float64.(curve)) for curve in curves])
    central_fit = linear_inverse_size_extrapolation(sizes, central_values)
    extrapolated_samples = zeros(Float64, n_boot, n_points)
    sample_values = zeros(Float64, n_sizes, n_points)
    for bootstrap_index in 1:n_boot
        for size_index in 1:n_sizes
            sample_values[size_index, :] .= samples[size_index][bootstrap_index, :]
        end
        extrapolated_samples[bootstrap_index, :] .=
            linear_inverse_size_extrapolation(sizes, sample_values).intercept
    end

    median_curve = zeros(Float64, n_points)
    lower = zeros(Float64, n_points)
    upper = zeros(Float64, n_points)
    for point in 1:n_points
        column = @view extrapolated_samples[:, point]
        median_curve[point] = quantile(column, 0.50)
        lower[point] = quantile(column, 0.16)
        upper[point] = quantile(column, 0.84)
    end
    largest_size_index = argmax(sizes)
    finite_size_systematic = abs.(Float64.(curves[largest_size_index]) .-
                                  central_fit.intercept)
    return (u = Float64.(u), curve = central_fit.intercept,
            slope = central_fit.slope, median = median_curve,
            lower = lower, upper = upper, samples = extrapolated_samples,
            finite_size_systematic = finite_size_systematic)
end

"""
Put finite-size curves on the first-principles density scale
`kappa = t/sqrt(rho_eff)` before extrapolating in `1/N`.

The input coordinate is the weighted-spacing variable `u=t*s_N`.
Consequently `kappa=u/(s_N*sqrt(rho_eff))`.  Interpolation is linear in
`log(kappa)`, matching the logarithmic sampling grid.
"""
function density_scaled_reference_curves(
        sizes::AbstractVector{<:Integer},
        u::AbstractVector{<:Real},
        curves::AbstractVector,
        samples::AbstractVector,
        spacings::AbstractVector{<:Real},
        plateaus::AbstractVector{<:Real};
        alpha_x::Union{Real,AbstractVector{<:Real}} = 1.0,
        alpha_y::Union{Real,AbstractVector{<:Real}} = 1.0)
    n_sizes = length(sizes)
    length(spacings) == length(plateaus) == n_sizes ||
        throw(DimensionMismatch("spacing and plateau schedules must match sizes"))
    length(curves) == length(samples) == n_sizes ||
        throw(DimensionMismatch("curve schedules must match sizes"))
    all(value -> isfinite(value) && value > 0, u) &&
        issorted(u) && all(diff(Float64.(u)) .> 0) ||
        throw(ArgumentError("u grid must be finite, positive, and increasing"))
    axes_x = _size_parameters(alpha_x, n_sizes, "alpha_x")
    axes_y = _size_parameters(alpha_y, n_sizes, "alpha_y")
    spacing_values = Float64.(spacings)
    all(value -> isfinite(value) && value > 0, spacing_values) ||
        throw(ArgumentError("spacings must be finite and positive"))
    densities = [gaussian_filter_effective_density(
                     plateaus[index], axes_x[index], axes_y[index])
                 for index in 1:n_sizes]
    density_factors = spacing_values .* sqrt.(densities)
    kappa_grids = [Float64.(u) ./ density_factors[index]
                   for index in 1:n_sizes]
    kappa_min = maximum(first, kappa_grids)
    kappa_max = minimum(last, kappa_grids)
    kappa_min < kappa_max ||
        throw(ArgumentError("density-scaled finite-size grids do not overlap"))
    kappa = exp.(range(log(kappa_min), log(kappa_max); length = length(u)))
    kappa[1] = kappa_min
    kappa[end] = kappa_max

    interpolated_curves = [_log_interpolate(
                               kappa_grids[index], curves[index], kappa)
                           for index in 1:n_sizes]
    interpolated_samples = Matrix{Float64}[]
    for index in 1:n_sizes
        source_samples = samples[index]
        size(source_samples, 2) == length(u) ||
            throw(DimensionMismatch("bootstrap curve grid differs at size index $index"))
        destination = zeros(Float64, size(source_samples, 1), length(kappa))
        for bootstrap_index in axes(source_samples, 1)
            destination[bootstrap_index, :] .= _log_interpolate(
                kappa_grids[index], @view(source_samples[bootstrap_index, :]),
                kappa)
        end
        push!(interpolated_samples, destination)
    end
    extrapolated = extrapolate_reference_curves(
        sizes, kappa, interpolated_curves, interpolated_samples)
    return merge(extrapolated,
                 (kappa = kappa,
                  effective_densities = densities,
                  density_factors = density_factors,
                  interpolated_curves = interpolated_curves))
end

"Calibrate the 0.95 plateau crossing of an extrapolated reference and its bootstraps."
function calibrate_extrapolated_reference(
        u::AbstractVector{<:Real},
        curve::AbstractVector{<:Real},
        samples::AbstractMatrix{<:Real};
        threshold::Real = 0.95,
        bins_per_decade::Integer = 20,
        minimum_successful::Integer = max(10, ceil(Int, size(samples, 1) / 2)))
    length(u) == length(curve) == size(samples, 2) ||
        throw(DimensionMismatch("calibration grids differ"))
    required = Int(minimum_successful)
    1 <= required <= size(samples, 1) ||
        throw(ArgumentError("minimum_successful is outside the bootstrap range"))
    calibration = calibrate_heisenberg(
        u, curve; threshold = threshold, bins_per_decade = bins_per_decade)
    chi_samples = Float64[]
    for bootstrap_index in axes(samples, 1)
        try
            sample_calibration = calibrate_heisenberg(
                u, @view(samples[bootstrap_index, :]);
                threshold = threshold, bins_per_decade = bins_per_decade)
            isfinite(sample_calibration.chi) && sample_calibration.chi > 0 &&
                push!(chi_samples, sample_calibration.chi)
        catch err
            err isa ArgumentError || rethrow()
        end
    end
    length(chi_samples) >= required ||
        throw(ArgumentError("too few successful extrapolated calibrations"))
    return (chi = calibration.chi, calibration = calibration,
            chi_samples = chi_samples,
            median = quantile(chi_samples, 0.50),
            lower = quantile(chi_samples, 0.16),
            upper = quantile(chi_samples, 0.84),
            successful = length(chi_samples),
            requested = size(samples, 1))
end

"Compute the circular-filtered connected DSFF for one matrix size."
function compute_finite_size_reference(
        spectra::AbstractVector,
        u::AbstractVector{<:Real};
        theta::Real = pi / 4,
        alpha_tilde::Real = 1.0,
        n_boot::Integer = 500,
        rng::AbstractRNG = MersenneTwister(0))
    length(spectra) >= 2 ||
        throw(ArgumentError("at least two spectra are required"))
    all(!isempty, spectra) || throw(ArgumentError("spectra cannot be empty"))
    all(value -> isfinite(value) && value > 0, u) ||
        throw(ArgumentError("u values must be finite and positive"))
    alpha = Float64(alpha_tilde)
    alpha > 0 || throw(ArgumentError("alpha_tilde must be positive"))

    complex_spectra = [ComplexF64.(spectrum) for spectrum in spectra]
    weights_by_seed = [circular_filter_weights(spectrum;
                                               alpha_tilde = alpha)
                       for spectrum in complex_spectra]
    spacing_by_seed = [weighted_nearest_spacing(
                           complex_spectra[index], weights_by_seed[index])
                       for index in eachindex(complex_spectra)]
    spacing = mean(spacing_by_seed)
    isfinite(spacing) && spacing > 0 ||
        throw(ArgumentError("weighted spacing must be finite and positive"))

    delta = inv(sqrt(alpha))
    filter = GaussianFilter(0.0, 0.0, delta, delta)
    times = Float64.(u) ./ spacing
    traces, partition_weights = partition_traces(
        complex_spectra, filter, times, theta)
    plateau_by_seed = Float64[sum(abs2, weights)
                              for weights in partition_weights]
    bootstrap = bootstrap_connected_dsff(
        traces, plateau_by_seed; n_boot = n_boot, rng = rng)
    return (u = Float64.(u), times = times, Z = traces,
            curve = bootstrap.curve, median = bootstrap.median,
            lower = bootstrap.lower, upper = bootstrap.upper,
            samples = bootstrap.samples, spacing = spacing,
            spacing_by_seed = spacing_by_seed,
            plateau = bootstrap.plateau,
            plateau_by_seed = plateau_by_seed,
            alpha_tilde = alpha, theta = Float64(theta))
end

_atomic_jld2(writer::Function, path::AbstractString) = atomic_jld2(writer, abspath(path))

"Write a self-describing extrapolated Gaussian AI-dagger DSFF reference."
function write_ai_dagger_reference(
        path::AbstractString;
        metadata::AbstractDict,
        u::AbstractVector{<:Real},
        curve::AbstractVector{<:Real},
        lower::AbstractVector{<:Real},
        upper::AbstractVector{<:Real},
        samples::AbstractMatrix{<:Real},
        chi::Real,
        chi_samples::AbstractVector{<:Real},
        finite_size_systematic::AbstractVector{<:Real})
    required_metadata = (
        "matrix_sizes", "seed_counts", "rng_seed", "theta",
        "alpha_tilde", "heisenberg_threshold")
    all(key -> haskey(metadata, key), required_metadata) ||
        throw(ArgumentError("AI-dagger reference metadata is incomplete"))
    n_points = length(u)
    all(array -> length(array) == n_points,
        (curve, lower, upper, finite_size_systematic)) ||
        throw(DimensionMismatch("reference curve lengths differ"))
    size(samples, 2) == n_points ||
        throw(DimensionMismatch("reference bootstrap grid differs"))
    Float64(chi) > 0 || throw(ArgumentError("chi must be positive"))
    return _atomic_jld2(path) do file
        for (key, value) in metadata
            file[String(key)] = value
        end
        file["u"] = Float64.(u)
        file["curve"] = Float64.(curve)
        file["lower"] = Float64.(lower)
        file["upper"] = Float64.(upper)
        file["samples"] = Float64.(samples)
        file["chi"] = Float64(chi)
        file["chi_samples"] = Float64.(chi_samples)
        file["finite_size_systematic"] = Float64.(finite_size_systematic)
    end
end

"Content token used to bind downstream curve caches to one reference file."
function ai_dagger_reference_token(path::AbstractString)
    isfile(path) || throw(ArgumentError("AI-dagger reference not found: $path"))
    return open(path, "r") do io
        bytes2hex(SHA.sha256(io))
    end
end

"Load and strictly validate a Gaussian AI-dagger DSFF reference."
function load_ai_dagger_reference(path::AbstractString;
                                  expected_theta::Real = pi / 4)
    isfile(path) || throw(ArgumentError("AI-dagger reference not found: $path"))
    token = ai_dagger_reference_token(path)
    return JLD2.jldopen(path, "r") do file
        required = (
            "matrix_sizes", "seed_counts", "rng_seed", "theta",
            "alpha_tilde", "heisenberg_threshold",
            "u", "curve", "lower", "upper",
            "samples", "chi", "chi_samples", "finite_size_systematic")
        all(key -> haskey(file, key), required) ||
            throw(ArgumentError("AI-dagger reference is incomplete"))
        theta = Float64(file["theta"])
        isapprox(theta, Float64(expected_theta); rtol = 0,
                 atol = 64eps(Float64)) ||
            throw(ArgumentError("AI-dagger ray differs from requested theta"))
        alpha = Float64(file["alpha_tilde"])
        isapprox(alpha, 1.0; rtol = 0, atol = 64eps(Float64)) ||
            throw(ArgumentError("AI-dagger reference requires alpha_tilde=1"))
        threshold = Float64(file["heisenberg_threshold"])
        isapprox(threshold, 0.95; rtol = 0, atol = 64eps(Float64)) ||
            throw(ArgumentError("AI-dagger Heisenberg threshold differs"))
        sizes = Int.(file["matrix_sizes"])
        seed_counts = Int.(file["seed_counts"])
        length(sizes) == length(seed_counts) >= 2 ||
            throw(ArgumentError("AI-dagger size schedule is invalid"))
        all(>(0), sizes) && all(>(1), seed_counts) ||
            throw(ArgumentError("AI-dagger size schedule must be positive"))
        u = Float64.(file["u"])
        curve = Float64.(file["curve"])
        lower = Float64.(file["lower"])
        upper = Float64.(file["upper"])
        systematic = Float64.(file["finite_size_systematic"])
        n_points = length(u)
        all(array -> length(array) == n_points,
            (curve, lower, upper, systematic)) ||
            throw(DimensionMismatch("AI-dagger curve lengths differ"))
        samples = Float64.(file["samples"])
        size(samples, 2) == n_points ||
            throw(DimensionMismatch("AI-dagger bootstrap grid differs"))
        chi = Float64(file["chi"])
        isfinite(chi) && chi > 0 ||
            throw(ArgumentError("AI-dagger chi must be finite and positive"))
        chi_samples = Float64.(file["chi_samples"])
        all(value -> isfinite(value) && value > 0, chi_samples) ||
            throw(ArgumentError("AI-dagger chi bootstrap is invalid"))
        display_u, display_curve = if haskey(file, "calibration_u") &&
                                      haskey(file, "calibration_fit")
            candidate_u = Float64.(file["calibration_u"])
            candidate_curve = Float64.(file["calibration_fit"])
            length(candidate_u) == length(candidate_curve) ||
                throw(DimensionMismatch("AI-dagger display curve lengths differ"))
            candidate_u, candidate_curve
        else
            u, curve
        end
        return (
            path = abspath(path), token = token, matrix_sizes = sizes,
            seed_counts = seed_counts, rng_seed = Int(file["rng_seed"]),
            theta = theta, alpha_tilde = alpha,
            threshold = threshold, u = u, x = u ./ chi,
            display_u = display_u, display_x = display_u ./ chi,
            display_curve = display_curve,
            curve = curve, lower = lower, upper = upper,
            samples = samples, chi = chi, chi_samples = chi_samples,
            chi_median = quantile(chi_samples, 0.50),
            chi_lower = quantile(chi_samples, 0.16),
            chi_upper = quantile(chi_samples, 0.84),
            finite_size_systematic = systematic,
        )
    end
end

end # module
