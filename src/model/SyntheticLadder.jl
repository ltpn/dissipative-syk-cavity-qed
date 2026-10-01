# Real SYK4 tensors and Hermitian random control channels.

using LinearAlgebra: Symmetric, norm
using Random: AbstractRNG, MersenneTwister, randn

# ============================================================================
# L1 -- Mean-shifted Gaussian white-noise weight
# ============================================================================


# ============================================================================
# L2, L3, L4 -- Random single-particle overlap matrix g
# ============================================================================

"""
    random_symmetric_g(rng, n_orb, sigma_g) -> Matrix{Float64}

Return a real symmetric `n_orb x n_orb` Gaussian matrix with the GOE
(Wigner-Dyson) second-moment structure:

    Var[G[i,i]] = 2 * sigma_g^2   (diagonal)
    Var[G[i,j]] = sigma_g^2       (off-diagonal, i != j)

which gives

    E[||G||_F^2] = sigma_g^2 * n_orb * (n_orb + 1).

The caller should choose `sigma_g` so this equals the calibration target
`<||g_phys||_F^2>`; see `ladder_sigma_g`.

Implementation: draw `M` with iid `N(0, 2 sigma_g^2)`, then symmetrize as
`(M + M^T) / 2`. The Wigner scaling factor `sqrt(2)` is applied to the raw
Gaussian so the diagonal survives the halving intact while the off-diagonal
pair-sum lands with variance `sigma_g^2`.
"""
function random_symmetric_g(rng::AbstractRNG, n_orb::Integer, sigma_g::Real)
    n = Int(n_orb)
    n > 0 || throw(ArgumentError("n_orb must be positive, got $n_orb"))
    sg = Float64(sigma_g)
    sg >= 0 || throw(ArgumentError("sigma_g must be >= 0, got $sigma_g"))
    scale = sqrt(2.0) * sg
    M = scale .* randn(rng, Float64, n, n)
    return Matrix(0.5 .* (M .+ transpose(M)))
end

# ============================================================================
# L2 -- Wishart-random Kossakowski pair-index matrix
# ============================================================================


# ============================================================================
# L3, L4 -- Equal-weight independent random symmetric jump operators
# ============================================================================

"""
    random_symmetric_jumps_equal_weight(rng, n_orb, n_jumps, sqrt_lambda) -> Vector{Matrix{ComplexF64}}

Return `n_jumps` independent one-body jump matrices. Each is a real symmetric
Gaussian matrix with iid entries, normalized to `||m||_F = 1`, then scaled by
`sqrt_lambda`. Returned as `Vector{Matrix{ComplexF64}}` for direct plugging
into `LambDickeLiouvillianConfig(synthetic_jump_matrices = ...)`.

`sqrt(Gamma_eff)` is applied later inside the Liouvillian builder; do NOT
include it here.
"""
function random_symmetric_jumps_equal_weight(rng::AbstractRNG,
                                              n_orb::Integer,
                                              n_jumps::Integer,
                                              sqrt_lambda::Real)
    n = Int(n_orb)
    n > 0 || throw(ArgumentError("n_orb must be positive, got $n_orb"))
    nj = Int(n_jumps)
    nj >= 0 || throw(ArgumentError("n_jumps must be non-negative, got $n_jumps"))
    scale = Float64(sqrt_lambda)
    isfinite(scale) && scale >= 0 ||
        throw(ArgumentError("sqrt_lambda must be finite and >= 0, got $sqrt_lambda"))

    jumps = Vector{Matrix{ComplexF64}}(undef, nj)
    for a in 1:nj
        M = randn(rng, Float64, n, n)
        M = 0.5 .* (M .+ transpose(M))
        f = norm(M)
        f > 0 || (M .= 0.0; f = 1.0)
        M ./= f
        jumps[a] = ComplexF64.(scale .* M)
    end
    return jumps
end

# ============================================================================
# L2, L3, L4 -- Independent cavity-loss operator (real symmetric)
# ============================================================================

"""
    random_symmetric_cavity_op(rng, n_orb, sigma_g) -> Matrix{ComplexF64}

Return an independent real symmetric Gaussian matrix with the same second-
moment structure as `random_symmetric_g(rng, n_orb, sigma_g)`, cast to
`Matrix{ComplexF64}` for direct assignment as
`synthetic_cavity_loss_operator`.
"""
function random_symmetric_cavity_op(rng::AbstractRNG, n_orb::Integer, sigma_g::Real)
    G = random_symmetric_g(rng, n_orb, sigma_g)
    return ComplexF64.(G)
end

# ============================================================================
# L4 -- Full SYK4 tensor
# ============================================================================


# ============================================================================
# L3b (BDI†) -- Real SYK4 tensor (Hermitian AND real -> real symmetric H)
# ============================================================================

"""
    random_real_syk4_tensor(rng, n_orb, sigma_syk4) -> Array{Float64,4}

Return a rank-4 tensor `T[j,l,k,m]` populated only at the unique combinations
`c in Combinations_SYK4(n_orb)` with **real** Gaussian couplings

    T[c] = sigma_syk4 * randn(rng),

for BOTH diagonal and off-diagonal combinations. Second moment matches
the complex ensemble: `E[T[c]^2] = sigma_syk4^2`.

The resulting many-body Hamiltonian is real symmetric in the number basis
(all `T[c]` real ⇒ all matrix elements ⟨α|H|β⟩ real). Combined with real
symmetric jumps and cavity op (see `random_symmetric_jumps_equal_weight`,
`random_symmetric_cavity_op`), the vectorized Liouvillian satisfies
`L^T = L` (complex symmetric), placing it in symmetry class BDI† of the
Sá–Ribeiro–Prosen classification (arXiv:2307.08218).

Consumed by `syk4_block_from_tensor` (which reads only the
`Combinations_SYK4(n_orb)` positions). The output dtype is `Float64` but
`syk4_block_from_tensor` promotes to `ComplexF64` internally.
"""
function random_real_syk4_tensor(rng::AbstractRNG, n_orb::Integer, sigma_syk4::Real)
    n = Int(n_orb)
    n > 0 || throw(ArgumentError("n_orb must be positive, got $n_orb"))
    sJ = Float64(sigma_syk4)
    sJ >= 0 || throw(ArgumentError("sigma_syk4 must be >= 0, got $sigma_syk4"))
    T = zeros(Float64, n, n, n, n)
    for c in Combinations_SYK4(n)
        T[c[1], c[2], c[3], c[4]] = sJ * randn(rng)
    end
    return T
end

# ============================================================================
# Calibration derivations
# ============================================================================

"""
    ladder_sigma_g(g_frobenius2_mean, n_orb) -> Float64

Return the `sigma_g` for `random_symmetric_g` that satisfies
`E[||G||_F^2] = g_frobenius2_mean`. Uses the GOE Wigner normalization
`E[||G||_F^2] = sigma_g^2 * n_orb * (n_orb + 1)`.
"""
function ladder_sigma_g(g_frobenius2_mean::Real, n_orb::Integer)
    n = Int(n_orb)
    n > 0 || throw(ArgumentError("n_orb must be positive"))
    gf2 = Float64(g_frobenius2_mean)
    gf2 >= 0 || throw(ArgumentError("g_frobenius2_mean must be >= 0"))
    return sqrt(gf2 / (n * (n + 1)))
end


"""
    ladder_lambda_star(K_pair_trace_mean, n_jumps) -> Float64

Return the equal-weight `lambda_star` for L3/L4 that satisfies
`n_jumps * lambda_star = K_pair_trace_mean`.
"""
function ladder_lambda_star(K_pair_trace_mean::Real, n_jumps::Integer)
    nj = Int(n_jumps)
    nj > 0 || throw(ArgumentError("n_jumps must be positive"))
    tr_mean = Float64(K_pair_trace_mean)
    tr_mean >= 0 || throw(ArgumentError("K_pair_trace_mean must be >= 0"))
    return tr_mean / nj
end
