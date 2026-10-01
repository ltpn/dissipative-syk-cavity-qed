# Angular-averaged emission kernels and jump factorization.

using LinearAlgebra: Symmetric, eigen, norm, svd, I

"""
    spherical_bessel_j0(x) -> Float64

Return `j0(x) = sin(x) / x` with the removable singularity `j0(0) = 1`.
For very small `|x|`, use the Taylor series through `x^4` to avoid
cancellation in validation tests.
"""
function spherical_bessel_j0(x::Real)
    y = Float64(x)
    ay = abs(y)
    if ay < sqrt(eps(Float64))
        y2 = y * y
        return 1.0 - y2 / 6.0 + y2 * y2 / 120.0
    end
    return sin(y) / y
end

"""
    JumpFactorization

Result of diagonalizing the pair-index Kossakowski matrix.

Fields:
- `eigenvalues`: retained raw eigenvalues in descending order.
- `jump_matrices`: retained `sqrt(max(lambda,0)) * M_alpha` matrices.
- `mode_matrices`: retained unscaled `M_alpha` matrices.
- `all_eigenvalues`: all raw eigenvalues in descending order, including tiny
  numerical negative values for PSD diagnostics.
- `retained_rank`: number of retained modes.
- `participation_ratio`: PR of the nonnegative kernel spectrum.
- `discarded_weight`: positive spectral weight below the retention threshold.
- `min_eig`: minimum raw eigenvalue.
- `svd_tol`: tolerance used for retention.
- `reconstruction_error`: relative Frobenius error from retained jump modes.
"""
struct JumpFactorization
    eigenvalues::Vector{Float64}
    jump_matrices::Vector{Matrix{Float64}}
    mode_matrices::Vector{Matrix{Float64}}
    all_eigenvalues::Vector{Float64}
    retained_rank::Int
    participation_ratio::Float64
    discarded_weight::Float64
    min_eig::Float64
    svd_tol::Float64
    reconstruction_error::Float64
end

"""
    kernel_tensor(eta, params, coefficients) -> Array{Float64,4}

Compute the rank-4 spontaneous-emission tensor
`K[j,k,l,m](eta)` on the  midpoint grid:

    du^4 * sum_{a,b} F[a,j,k] * j0(eta * |u_a-u_b|) * F[b,l,m]

with `F[a,j,k] = w(u_a) * phi_j(u_a) * phi_k(u_a)`.

The eta-zero limit is evaluated from the  overlap matrix exactly as
`K_jk * K_lm`; this is the same grid formula with `j0(0)=1`, but avoids
materializing an all-ones grid kernel.
"""
function kernel_tensor(eta::Real,
                       params::SingleParticleParams,
                       coefficients::Coefficients)
    _validate_kernel_inputs(eta, params, coefficients)
    n_orb = params.orbitals.n_orb

    if Float64(eta) == 0.0
        K = coefficients.K_jk
        T = Array{Float64,4}(undef, n_orb, n_orb, n_orb, n_orb)
        @inbounds for m in 1:n_orb, l in 1:n_orb, k in 1:n_orb, j in 1:n_orb
            T[j, k, l, m] = K[j, k] * K[l, m]
        end
        return T
    end

    u, du, Phi = build_orbitals_2d(params.grid, params.orbitals)
    w = build_weight(params.grid, params.weight)
    F = _pair_weighted_orbitals(Phi, w)
    grid_kernel = _spherical_grid_kernel(u, Float64(eta))
    K_pair = (du^4) .* (transpose(F) * (grid_kernel * F))
    K_pair = 0.5 .* (K_pair .+ transpose(K_pair))
    return reshape(K_pair, n_orb, n_orb, n_orb, n_orb)
end

"""
    kernel_pair_matrix(eta, params, coefficients) -> Symmetric{Float64,Matrix{Float64}}

Return the column-major pair-index matrix with
`p_C(j,k) = j + (k-1)N`, symmetrized as `(K + K') / 2` before
factorization.
"""
function kernel_pair_matrix(eta::Real,
                            params::SingleParticleParams,
                            coefficients::Coefficients)
    T = kernel_tensor(eta, params, coefficients)
    n_orb = size(T, 1)
    K_pair = reshape(T, n_orb^2, n_orb^2)
    K_pair = 0.5 .* (K_pair .+ transpose(K_pair))
    return Symmetric(Matrix{Float64}(K_pair))
end

"""
    jump_matrices_from_kernel(K_pair; svd_tol=1e-8, retain_method=:hermitian)

Factorize a real symmetric positive semidefinite pair-index kernel into
retained one-body Lindblad jump matrices. The returned `jump_matrices` are
`sqrt(lambda_alpha) * M_alpha`; do not multiply them by `sqrt(Gamma_eff)` here.
"""
function jump_matrices_from_kernel(K_pair;
                                   svd_tol::Real = 1e-8,
                                   retain_method::Symbol = :hermitian)
    tol = Float64(svd_tol)
    tol >= 0.0 || throw(ArgumentError("svd_tol must be non-negative, got $tol"))

    A = Matrix{Float64}(K_pair)
    n_pair_1, n_pair_2 = size(A)
    n_pair_1 == n_pair_2 ||
        throw(DimensionMismatch("K_pair must be square, got $(size(A))"))
    n_orb_float = sqrt(n_pair_1)
    n_orb = round(Int, n_orb_float)
    n_orb^2 == n_pair_1 ||
        throw(ArgumentError("K_pair dimension $n_pair_1 is not a perfect square"))

    A_sym = 0.5 .* (A .+ transpose(A))
    vals, vecs = _factorize_pair_kernel(A_sym, retain_method)

    order = sortperm(vals; rev = true)
    vals = Float64.(vals[order])
    vecs = vecs[:, order]

    lambda1 = isempty(vals) ? 0.0 : maximum((vals[1], 0.0))
    positive_floor = eps(Float64) * max(lambda1, 1.0) * max(length(vals), 1)
    threshold = tol == 0.0 ? positive_floor : max(tol * lambda1, positive_floor)

    positive_vals = max.(vals, 0.0)
    total_positive = sum(positive_vals)
    pr_denom = sum(abs2, positive_vals)
    participation_ratio =
        pr_denom == 0.0 ? 0.0 : (total_positive * total_positive) / pr_denom

    retained = findall(lambda -> lambda >= threshold, vals)
    retained_vals = Float64[]
    mode_matrices = Matrix{Float64}[]
    jump_matrices = Matrix{Float64}[]
    for idx in retained
        lambda = vals[idx]
        lambda > 0.0 || continue
        M = reshape(vecs[:, idx], n_orb, n_orb)
        M = Matrix(0.5 .* (M .+ transpose(M)))
        mnorm = norm(M)
        mnorm > 0.0 || continue
        M ./= mnorm
        push!(retained_vals, lambda)
        push!(mode_matrices, M)
        push!(jump_matrices, sqrt(max(lambda, 0.0)) .* M)
    end

    retained_positive = sum(max.(retained_vals, 0.0))
    discarded_weight = total_positive == 0.0 ? 0.0 :
                       max(total_positive - retained_positive, 0.0) / total_positive

    fact_for_error = JumpFactorization(retained_vals,
                                       jump_matrices,
                                       mode_matrices,
                                       vals,
                                       length(jump_matrices),
                                       participation_ratio,
                                       discarded_weight,
                                       isempty(vals) ? 0.0 : minimum(vals),
                                       tol,
                                       0.0)
    K_rec = reconstruct_kernel(fact_for_error)
    rec_err = norm(A_sym .- K_rec) / max(norm(A_sym), eps(Float64))

    return JumpFactorization(retained_vals,
                             jump_matrices,
                             mode_matrices,
                             vals,
                             length(jump_matrices),
                             participation_ratio,
                             discarded_weight,
                             isempty(vals) ? 0.0 : minimum(vals),
                             tol,
                             rec_err)
end

"""
    reconstruct_kernel(factorization) -> Matrix{Float64}

Reconstruct the retained pair-index kernel as
`sum(vec(J_alpha) * vec(J_alpha)')`, where `J_alpha =
sqrt(lambda_alpha) * M_alpha`.
"""
function reconstruct_kernel(factorization::JumpFactorization)
    if isempty(factorization.jump_matrices)
        n_pair = length(factorization.all_eigenvalues)
        return zeros(Float64, n_pair, n_pair)
    end
    n_orb = size(factorization.jump_matrices[1], 1)
    K = zeros(Float64, n_orb^2, n_orb^2)
    for J in factorization.jump_matrices
        v = vec(J)
        K .+= v * transpose(v)
    end
    return K
end

# ============================================================================
# Internal helpers
# ============================================================================

function _validate_kernel_inputs(eta::Real,
                                 params::SingleParticleParams,
                                 coefficients::Coefficients)
    eta_f = Float64(eta)
    isfinite(eta_f) || throw(ArgumentError("eta must be finite, got $eta"))
    eta_f >= 0.0 || throw(ArgumentError("eta must be non-negative, got $eta"))
    validate_params(params)

    n_orb = params.orbitals.n_orb
    size(coefficients.K_jk) == (n_orb, n_orb) ||
        throw(DimensionMismatch("coefficients.K_jk has shape $(size(coefficients.K_jk)); expected $((n_orb,n_orb))"))
    size(coefficients.g_jk) == (n_orb, n_orb) ||
        throw(DimensionMismatch("coefficients.g_jk has shape $(size(coefficients.g_jk)); expected $((n_orb,n_orb))"))
    coefficients.grid == params.grid ||
        throw(ArgumentError("coefficient grid does not match params grid"))
    coefficients.orbitals.n_orb == params.orbitals.n_orb ||
        throw(ArgumentError("coefficient orbitals do not match params orbitals"))
    coefficients.orbitals.quanta == params.orbitals.quanta ||
        throw(ArgumentError("coefficient orbital ordering does not match params"))
    coefficients.weight == params.weight ||
        throw(ArgumentError("coefficient weight parameters do not match params"))
    return nothing
end

function _pair_weighted_orbitals(Phi::AbstractArray{<:Real,3},
                                 w::AbstractMatrix{<:Real})
    n_grid_1, n_grid_2, n_orb = size(Phi)
    size(w) == (n_grid_1, n_grid_2) ||
        throw(DimensionMismatch("weight size $(size(w)) does not match orbital grid $((n_grid_1,n_grid_2))"))
    Phi_flat = reshape(Phi, n_grid_1 * n_grid_2, n_orb)
    w_flat = vec(w)
    F = Matrix{Float64}(undef, length(w_flat), n_orb^2)
    @inbounds for k in 1:n_orb, j in 1:n_orb
        p = j + (k - 1) * n_orb
        for a in eachindex(w_flat)
            F[a, p] = w_flat[a] * Phi_flat[a, j] * Phi_flat[a, k]
        end
    end
    return F
end

function _spherical_grid_kernel(u::AbstractVector{<:Real}, eta::Float64)
    n_grid = length(u)
    n_sites = n_grid * n_grid
    x = Vector{Float64}(undef, n_sites)
    y = Vector{Float64}(undef, n_sites)
    @inbounds for j in 1:n_grid, i in 1:n_grid
        idx = i + (j - 1) * n_grid
        x[idx] = Float64(u[i])
        y[idx] = Float64(u[j])
    end

    G = Matrix{Float64}(undef, n_sites, n_sites)
    @inbounds for b in 1:n_sites
        xb = x[b]
        yb = y[b]
        for a in 1:n_sites
            dx = x[a] - xb
            dy = y[a] - yb
            G[a, b] = spherical_bessel_j0(eta * sqrt(dx * dx + dy * dy))
        end
    end
    return G
end

function _factorize_pair_kernel(A::AbstractMatrix{<:Real}, retain_method::Symbol)
    if retain_method == :hermitian
        decomp = eigen(Symmetric(Matrix{Float64}(A)))
        return decomp.values, decomp.vectors
    elseif retain_method == :svd
        decomp = svd(Matrix{Float64}(A))
        return decomp.S, decomp.U
    else
        throw(ArgumentError("retain_method must be :hermitian or :svd, got $retain_method"))
    end
end
