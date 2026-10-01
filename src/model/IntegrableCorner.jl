module IntegrableCorner

using Combinatorics: combinations
using LinearAlgebra: Symmetric, eigen
using Polynomials: Polynomials
using Statistics: mean

export fixed_filling_corner_data,
       corner_center,
       corner_eigenvalue,
       streamed_staircase,
       streamed_unfolded_trace,
       connected_corner_sff,
       exact_corner_plateau,
       goe_form_factor,
       folded_goe_form_factor

"Single-particle and fixed-filling data needed by the integrable-corner formulas."
function fixed_filling_corner_data(g::AbstractMatrix{<:Real}, filling::Integer;
                                   J::Real = 1.0)
    n = size(g, 1)
    size(g, 2) == n || throw(DimensionMismatch("g must be square"))
    0 < filling < n || throw(ArgumentError("filling must satisfy 0 < filling < size(g,1)"))
    all(isfinite, g) || throw(ArgumentError("g contains non-finite values"))
    isapprox(g, transpose(g); rtol = 0, atol = 64eps(Float64) * max(1, maximum(abs, g))) ||
        throw(ArgumentError("g must be real symmetric"))
    Jf = Float64(J)
    isfinite(Jf) || throw(ArgumentError("J must be finite"))

    eig = eigen(Symmetric(Matrix{Float64}(g)))
    epsilon = Vector{Float64}(eig.values)
    occupations = [collect(Int, occ) for occ in combinations(1:n, Int(filling))]
    f = Float64[sum(@view epsilon[occ]) for occ in occupations]
    epsilon2 = abs2.(epsilon)
    h = Float64[sum(@view epsilon2[occ]) for occ in occupations]
    E = @. -Jf * f^2 + Jf * h
    return (epsilon = epsilon, eigenvectors = Matrix(eig.vectors),
            occupations = occupations, f = f, h = h, E = E)
end

"Trace-centering shift c = -Gamma_tot Var(f), with population variance."
function corner_center(f::AbstractVector{<:Real}, gamma_tot::Real)
    isempty(f) && throw(ArgumentError("f must not be empty"))
    gamma = Float64(gamma_tot)
    isfinite(gamma) && gamma >= 0 ||
        throw(ArgumentError("gamma_tot must be finite and non-negative"))
    values = Float64.(f)
    all(isfinite, values) || throw(ArgumentError("f contains non-finite values"))
    variance = mean(abs2, values) - abs2(mean(values))
    return -gamma * max(variance, 0.0)
end

"Closed-form Liouvillian eigenvalue lambda_mn at the single-channel corner."
function corner_eigenvalue(fm::Real, fn::Real, Em::Real, En::Real,
                           gamma_tot::Real)
    gamma = Float64(gamma_tot)
    delta_f = Float64(fm) - Float64(fn)
    delta_E = Float64(Em) - Float64(En)
    return -im * delta_E - (gamma / 2) * delta_f^2
end

@inline function _corner_sigma(fm::Float64, fn::Float64,
                               Em::Float64, En::Float64,
                               gamma::Float64, center::Float64)
    delta_f = fm - fn
    realpart = -(gamma / 2) * delta_f^2 - center
    imagpart = -(Em - En)
    return hypot(realpart, imagpart)
end

function _validated_vectors(f, E, gamma_tot)
    length(f) == length(E) || throw(DimensionMismatch("f and E lengths differ"))
    isempty(f) && throw(ArgumentError("f and E must not be empty"))
    fv = Float64.(f)
    Ev = Float64.(E)
    all(isfinite, fv) || throw(ArgumentError("f contains non-finite values"))
    all(isfinite, Ev) || throw(ArgumentError("E contains non-finite values"))
    gamma = Float64(gamma_tot)
    isfinite(gamma) && gamma >= 0 ||
        throw(ArgumentError("gamma_tot must be finite and non-negative"))
    return fv, Ev, gamma, corner_center(fv, gamma)
end

"Weighted full-spectrum staircase obtained without materializing d^2 values."
function streamed_staircase(f, E, gamma_tot;
                            n_bins::Integer = 40, degree::Integer = 5)
    fv, Ev, gamma, center = _validated_vectors(f, E, gamma_tot)
    d = length(fv)
    nb = Int(n_bins)
    nb >= 2 || throw(ArgumentError("n_bins must be at least 2"))

    diagonal_sigma = abs(center)
    sigma_min = diagonal_sigma
    sigma_max = diagonal_sigma
    @inbounds for m in 1:d-1, n in m+1:d
        sigma = _corner_sigma(fv[m], fv[n], Ev[m], Ev[n], gamma, center)
        sigma_min = min(sigma_min, sigma)
        sigma_max = max(sigma_max, sigma)
    end

    knots = collect(range(sigma_min, sigma_max; length = nb + 1))
    histogram = zeros(Int, length(knots))
    zero_count = diagonal_sigma <= 1e-8 ? d : 0
    if sigma_max == sigma_min
        histogram[1] = d^2
        zero_count = diagonal_sigma <= 1e-8 ? d^2 : 0
    else
        step = (sigma_max - sigma_min) / nb
        bin_index(sigma) = clamp(ceil(Int, (sigma - sigma_min) / step) + 1,
                                 1, length(knots))
        histogram[bin_index(diagonal_sigma)] += d
        @inbounds for m in 1:d-1, n in m+1:d
            sigma = _corner_sigma(fv[m], fv[n], Ev[m], Ev[n], gamma, center)
            histogram[bin_index(sigma)] += 2
            sigma <= 1e-8 && (zero_count += 2)
        end
    end
    counts = cumsum(histogram)
    counts[end] == d^2 || error("weighted pair count mismatch")
    fit_degree = clamp(Int(degree), 0, min(length(knots) - 1,
                                           length(unique(counts)) - 1))
    polynomial = Polynomials.fit(knots, Float64.(counts), fit_degree)
    return (minimum = sigma_min, maximum = sigma_max,
            knots = knots, counts = counts, polynomial = polynomial,
            degree = fit_degree, total_weight = d^2,
            center = center, diagonal_sigma = diagonal_sigma,
            zero_count = zero_count)
end

"Stream the multiplicity-weighted unfolded Fourier trace Z(q)."
function streamed_unfolded_trace(f, E, gamma_tot, qs;
                                 staircase = streamed_staircase(f, E, gamma_tot),
                                 pair_block_rows::Integer = 64,
                                 time_block::Integer = 32,
                                 threaded::Bool = true)
    fv, Ev, gamma, center = _validated_vectors(f, E, gamma_tot)
    d = length(fv)
    qv = Float64.(qs)
    all(isfinite, qv) || throw(ArgumentError("qs contains non-finite values"))
    row_block = Int(pair_block_rows)
    q_block = Int(time_block)
    row_block > 0 || throw(ArgumentError("pair_block_rows must be positive"))
    q_block > 0 || throw(ArgumentError("time_block must be positive"))
    polynomial = staircase.polynomial
    xi_diagonal = polynomial(abs(center))
    row_starts = collect(1:row_block:max(d - 1, 1))
    partials = zeros(ComplexF64, length(qv), length(row_starts))

    function accumulate_block!(block_index)
        rowlo = row_starts[block_index]
        rowhi = min(d - 1, rowlo + row_block - 1)
        rowlo > rowhi && return
        @inbounds for m in rowlo:rowhi, n in m+1:d
            sigma = _corner_sigma(fv[m], fv[n], Ev[m], Ev[n], gamma, center)
            xi = polynomial(sigma)
            for qlo in 1:q_block:length(qv)
                qhi = min(length(qv), qlo + q_block - 1)
                for qi in qlo:qhi
                    partials[qi, block_index] += 2 * cis(-qv[qi] * xi)
                end
            end
        end
        return
    end

    if threaded && Threads.nthreads() > 1
        Threads.@threads :dynamic for block_index in eachindex(row_starts)
            accumulate_block!(block_index)
        end
    else
        for block_index in eachindex(row_starts)
            accumulate_block!(block_index)
        end
    end

    Z = ComplexF64[d * cis(-q * xi_diagonal) for q in qv]
    @inbounds for block_index in eachindex(row_starts), qi in eachindex(qv)
        Z[qi] += partials[qi, block_index]
    end
    return (Z = Z, total_weight = d^2, center = center,
            staircase = staircase)
end

"R/(R-1)-unbiased connected form factor from realization rows."
function connected_corner_sff(Z::AbstractMatrix{<:Complex}, levels)
    R, nq = size(Z)
    R >= 2 || throw(ArgumentError("connected estimator requires at least two seeds"))
    length(levels) == R || throw(DimensionMismatch("level-count mismatch"))
    all(==(first(levels)), levels) ||
        throw(ArgumentError("all realizations must have a common level count"))
    plateau = Float64(first(levels))
    plateau > 0 || throw(ArgumentError("level count must be positive"))
    correction = R / (R - 1)
    curve = zeros(Float64, nq)
    @inbounds for qi in 1:nq
        zmean = sum(@view Z[:, qi]) / R
        mean_abs2 = sum(abs2, @view Z[:, qi]) / R
        curve[qi] = correction * (mean_abs2 - abs2(zmean)) / plateau
    end
    return curve
end

exact_corner_plateau(d::Integer) = 3.0 - 2.0 / Int(d)

"Connected GOE spectral form factor with unit late-time plateau."
function goe_form_factor(tau::Real)
    t = abs(Float64(tau))
    if t <= 1.0
        return 2t - t * log1p(2t)
    end
    return 2.0 - t * log1p(2.0 / (2t - 1.0))
end
goe_form_factor(taus::AbstractArray) = goe_form_factor.(taus)

folded_goe_form_factor(tau::Real) = goe_form_factor(2abs(Float64(tau)))
folded_goe_form_factor(taus::AbstractArray) = folded_goe_form_factor.(taus)

end
