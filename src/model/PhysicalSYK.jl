# Cavity-mediated quartic interactions in a fixed-filling basis.

using LinearAlgebra: norm

"""
    physical_syk_scaling(n_orb; J=1.0) -> Float64

Return the cavity-eliminated coherent prefactor J = λ²/Δ. It is independent
of N. With the paper's scale E = 4λ²/Δ, Γeff/J = 4Γeff/E.
"""
function physical_syk_scaling(n_orb::Integer; J::Real = 1.0)
    n = Int(n_orb)
    n > 0 || throw(ArgumentError("n_orb must be positive, got $n_orb"))
    Jf = Float64(J)
    isfinite(Jf) || throw(ArgumentError("J must be finite, got $J"))
    return Jf
end

"""
    build_physical_syk_tensor(g_matrices; weights=nothing, J=1.0)

Build the antisymmetrized physical low-rank quartic tensor
`J_tensor[j,l,k,m]` consumed by the shared SYK low-rank Hamiltonian builders
(`SYK_lr_block_full` / `SYK_lr_block_direct`, which sum over the UNRESTRICTED
`(j,l,k,m)` index product).

The leading `0.25` is the antisymmetrization normalizer: the 4-term tensor below
over-counts the operator by 4 under that unrestricted sum, and the 1/4 restores
the 1x bare quartic `+(lambda^2/Delta_cd) sum g_{jk} g_{lm} c_j^dg c_l^dg c_k c_m`,
i.e. the quartic part of the paper's `H = -(lambda^2/Delta_cd) F^2`. Do NOT add
this 1/4 to `generate_J_list_lr` (the independent full-Hamiltonian check keeps the over-counted
convention; see model.md). The conjugated form makes the operator Hermitian
for any input, but it equals `-J F^2` only when each `g` is HERMITIAN (real
symmetric in the baseline); a non-Hermitian `g` yields the Hermitized operator,
not `-J F^2` (a warning is emitted).

`g_matrices` may be a single square matrix or a vector of square matrices.
`weights` defaults to one weight per matrix. No spectral-width normalization is applied.
"""
function build_physical_syk_tensor(g_matrices; weights = nothing, J::Real = 1.0)
    matrices = _coerce_matrix_list(g_matrices)
    n_orb = _validate_matrix_list(matrices)
    weights_vec = _coerce_weights(weights, length(matrices))
    scale = physical_syk_scaling(n_orb; J = J)

    tensor = zeros(ComplexF64, n_orb, n_orb, n_orb, n_orb)
    @inbounds for a in eachindex(matrices)
        G = matrices[a]
        w = weights_vec[a]
        # The built operator equals -J F^2 only for Hermitian G (real symmetric in
        # the baseline). The conjugated 4-term form below is Hermitian for any G, but
        # for non-Hermitian G it is the Hermitized object, not -J F^2 -> warn.
        herm_res = norm(G .- G') / max(norm(G), eps(Float64))
        herm_res > 1e-10 && @warn string("build_physical_syk_tensor: g_matrices[", a,
            "] is not Hermitian (residual ", herm_res,
            "); built tensor is the Hermitized operator, not -J F^2.")
        # 0.25 = antisymmetrization normalizer (see header convention lock): the
        # unrestricted SYK_lr_block_* sum over this 4-term tensor over-counts the
        # operator by 4, and 1/4 restores the 1x bare-prefactor quartic.
        for m in 1:n_orb, k in 1:n_orb, l in 1:n_orb, j in 1:n_orb
            tensor[j, l, k, m] += 0.25 * scale * w * (
                G[j, k] * conj(G[m, l]) -
                G[l, k] * conj(G[m, j]) -
                G[j, m] * conj(G[k, l]) +
                G[l, m] * conj(G[k, j])
            )
        end
    end
    return tensor
end

# ============================================================================
# Internal helpers
# ============================================================================

_coerce_matrix_list(g::AbstractMatrix) = [_coerce_single_matrix(g)]

function _coerce_matrix_list(g_matrices::AbstractVector)
    isempty(g_matrices) && throw(ArgumentError("g_matrices must not be empty"))
    return [_coerce_single_matrix(g) for g in g_matrices]
end

function _coerce_single_matrix(g::AbstractMatrix)
    size(g, 1) == size(g, 2) ||
        throw(DimensionMismatch("matrix must be square, got $(size(g))"))
    all(_isfinite_number, g) ||
        throw(ArgumentError("matrix contains non-finite entries"))
    return Matrix{ComplexF64}(g)
end

function _validate_matrix_list(matrices::AbstractVector{<:AbstractMatrix})
    isempty(matrices) && throw(ArgumentError("g_matrices must not be empty"))
    n_orb = size(matrices[1], 1)
    for G in matrices
        size(G) == (n_orb, n_orb) ||
            throw(DimensionMismatch("all matrices must have shape $((n_orb,n_orb)); got $(size(G))"))
    end
    return n_orb
end

function _coerce_weights(weights, n_matrices::Int)
    weights === nothing && return ones(Float64, n_matrices)
    length(weights) == n_matrices ||
        throw(DimensionMismatch("weights length $(length(weights)) does not match $n_matrices matrices"))
    out = Float64.(collect(weights))
    all(isfinite, out) || throw(ArgumentError("weights contain non-finite entries"))
    return out
end

_isfinite_number(x::Real) = isfinite(x)
_isfinite_number(x::Complex) = isfinite(real(x)) && isfinite(imag(x))
