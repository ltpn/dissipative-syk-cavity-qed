# Single-particle overlap matrices in the harmonic-oscillator basis.

using LinearAlgebra: Symmetric

"""
    overlap_matrix(Φ::Array{Float64,3}, W::AbstractMatrix{<:Real}, du::Real) -> Matrix{Float64}

Generic  Gram matmul on the 2D grid. Returns the real symmetric
matrix `M[a, b] = Δu² · Σ_{ij} W[i, j] · Φ[i, j, a] · Φ[i, j, b]`.

The output is symmetrized as `(M + M^T) / 2` to suppress floating-point
asymmetry coming from BLAS reductions; this keeps every subsequent calculation’s
"is K_jk symmetric?" check trivially passing.
"""
function overlap_matrix(Φ::AbstractArray{<:Real,3},
                        W::AbstractMatrix{<:Real},
                        du::Real)
    N1, N2, n_orb = size(Φ)
    if size(W) != (N1, N2)
        gsz = (N1, N2)
        throw(DimensionMismatch(
            "weight kernel size $(size(W)) does not match grid $gsz"))
    end
    n_orb == 0 && return zeros(Float64, 0, 0)
    Φ_flat = reshape(Φ, N1 * N2, n_orb)
    W_flat = vec(W)
    # Compute Φ_w[k, a] = W_flat[k] * Φ_flat[k, a]
    Φ_w = Φ_flat .* W_flat
    M = (du^2) .* (transpose(Φ_w) * Φ_flat)
    # Symmetrize defensively.
    return Matrix(0.5 .* (M .+ transpose(M)))
end

"""
    compute_overlap_set(Φ, u, du, w) -> (K, X_x, X_y, T)

Compute the four single-particle overlap matrices declared in the symbol
table of `docs/model.md`:

    K[a, b]     = Δu² Σ w(u) φ_a φ_b
    X_x[a, b]   = Δu² Σ u_x w(u) φ_a φ_b
    X_y[a, b]   = Δu² Σ u_y w(u) φ_a φ_b
    T[a, b]     = Δu² Σ |u|² w(u) φ_a φ_b

`u::Vector{Float64}` is the 1D grid (length `n_grid`); `du` is the spacing;
`w::AbstractMatrix{Float64}` is the (uniform or speckle-derived) weight on
the 2D grid. Returns four `Matrix{Float64}` of shape `(n_orb, n_orb)`.
"""
function compute_overlap_set(Φ::AbstractArray{<:Real,3},
                             u::AbstractVector{<:Real},
                             du::Real,
                             w::AbstractMatrix{<:Real})
    N = length(u)
    if size(w) != (N, N)
        gsz = (N, N)
        throw(DimensionMismatch(
            "weight size $(size(w)) does not match grid $gsz"))
    end
    # Build the auxiliary weight kernels in-place.
    K_W   = w
    Xx_W  = u .* w                  # u_x dependence is along the first axis
    Xy_W  = transpose(u) .* w       # u_y dependence is along the second axis
    T_W   = (u.^2 .+ transpose(u.^2)) .* w
    K   = overlap_matrix(Φ, K_W,  du)
    X_x = overlap_matrix(Φ, Xx_W, du)
    X_y = overlap_matrix(Φ, Xy_W, du)
    T   = overlap_matrix(Φ, T_W,  du)
    return K, X_x, X_y, T
end

"""
    compute_coefficients(p::SingleParticleParams) -> Coefficients

End-to-end  builder for a single configuration. Builds the orbital
array, materializes the weight field, and returns the full `Coefficients`
record. The baseline beam-shape choice `g(u) = g_d(u) = 1` makes
`g_jk = K_jk` ( §1.3); both fields are populated to keep the symbol
table contract complete.

"""
function compute_coefficients(p::SingleParticleParams)
    validate_params(p)
    u, du, Φ = build_orbitals_2d(p.grid, p.orbitals)
    w = build_weight(p.grid, p.weight)
    K, X_x, X_y, T = compute_overlap_set(Φ, u, du, w)
    g = copy(K)  # baseline g(r) = g_d(r) = 1 ⇒ g_jk = K_jk
    return Coefficients(K, g, (X_x, X_y), T, p.grid, p.orbitals, p.weight)
end
