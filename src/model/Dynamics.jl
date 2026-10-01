# Exact density-matrix dynamics, entropy, and orbital populations.

using LinearAlgebra: Hermitian, eigvals, eigen, tr

const DYNAMICS_DEFAULT_ENTROPY_EIG_TOL = 1.0e-12
"""
    initial_product_state_dm(basis_states) -> Matrix{ComplexF64}

Return the density matrix `rho(0) = |Psi(0)><Psi(0)|` with
`|Psi(0)> = |1,...,1,0,...,0>` in the fixed-filling basis defined by
`basis_states`.  The product state corresponds to the first element of
`generate_vectors(N_orb, N_orb / 2)`, which by convention occupies orbitals
`1, ..., N_orb / 2` (lowest-`n_x + n_y` orbitals in the
`HarmonicOscillatorGrid.jl` ordering).

The result lives in the fixed-filling subspace and has size `D x D` with
`D = length(basis_states)`.
"""
function initial_product_state_dm(basis_states)
    d = length(basis_states)
    d > 0 || throw(ArgumentError("basis_states must be non-empty"))
    rho = zeros(ComplexF64, d, d)
    rho[1, 1] = one(ComplexF64)
    return rho
end

"""
    von_neumann_entropy(rho; eig_tol=1e-12) -> Float64

Return the von Neumann entropy `S = -sum_k p_k log(p_k)` of a density
matrix `rho`.  Eigenvalues are computed from the Hermitian symmetrization
`(rho + rho') / 2` to suppress integration noise, clipped to `[0, infty)`,
and only contributions with `p_k > eig_tol` are accumulated.  The natural
logarithm is used (units: nats).
"""
function von_neumann_entropy(rho::AbstractMatrix; eig_tol::Real = DYNAMICS_DEFAULT_ENTROPY_EIG_TOL)
    sym = (rho .+ adjoint(rho)) ./ 2
    vals = eigvals(Hermitian(Matrix(sym)))
    s = 0.0
    @inbounds for v in vals
        p = real(v)
        p > eig_tol || continue
        s -= p * log(p)
    end
    return s
end

"""
    DynamicsResult(times, entropy_t, populations_t, metadata)

Container returned by `dynamics_from_eigen`.  `populations_t` is an
`n_orb x length(times)` real matrix whose row `j` is `<n_j>(t)`.
"""
struct DynamicsResult
    times::Vector{Float64}
    entropy_t::Vector{Float64}
    populations_t::Matrix{Float64}
    metadata::Dict{String,Any}
end

function _fixed_filling_from_basis(basis_states, n_orb::Integer)
    fillings = unique(Int(sum(state)) for state in basis_states)
    length(fillings) == 1 ||
        throw(ArgumentError(
            "basis_states mix particle-number sectors: $(sort(fillings))"))
    return only(fillings)
end

"""
    dynamics_from_eigen(eigenvalues, eigenvectors, basis_states, times;
                        n_orb=length(first(basis_states)),
                        entropy_eig_tol=$(DYNAMICS_DEFAULT_ENTROPY_EIG_TOL))
        -> DynamicsResult

Exact dense propagation of the product state from a PRECOMPUTED eigen
factorization of the Liouvillian, `L = V diag(lambda) V^{-1}`, with
`eigenvalues = lambda` and `eigenvectors = V`.  Reuses one `eigen(L)` so a
caller that already diagonalized `L` (e.g. for the Liouvillian spectrum / DSFF)
does not pay a second decomposition.  Returns the same arrays and (dense-path)
metadata the `:dense` branch of [`dynamics_from_eigen`](@ref) produces.
"""
function dynamics_from_eigen(eigenvalues::AbstractVector,
                             eigenvectors::AbstractMatrix,
                             basis_states,
                             times::AbstractVector;
                             n_orb::Integer = length(first(basis_states)),
                             entropy_eig_tol::Real = DYNAMICS_DEFAULT_ENTROPY_EIG_TOL)
    n = Int(n_orb)
    d = length(basis_states)
    size(eigenvectors) == (d^2, d^2) ||
        throw(DimensionMismatch("eigenvectors is $(size(eigenvectors)); expected $((d^2, d^2))"))
    length(eigenvalues) == d^2 ||
        throw(DimensionMismatch("eigenvalues has length $(length(eigenvalues)); expected $(d^2)"))
    times_vec = collect(Float64, times)
    T = length(times_vec)
    n_filled = _fixed_filling_from_basis(basis_states, n)
    entropy_t = Vector{Float64}(undef, T)
    populations_t = Matrix{Float64}(undef, n, T)
    trace_residual = 0.0

    occ = Matrix{Float64}(undef, n, d)
    @inbounds for (alpha, state) in enumerate(basis_states)
        length(state) == n ||
            throw(DimensionMismatch("basis state $alpha has length $(length(state)); expected $n"))
        for j in 1:n
            occ[j, alpha] = Float64(state[j])
        end
    end

    V = eigenvectors
    lambda = eigenvalues
    t1 = times_vec[1]
    c = V \ vec(Matrix{ComplexF64}(initial_product_state_dm(basis_states)))
    @inbounds for k in 1:T
        vt = V * (exp.(lambda .* (times_vec[k] - t1)) .* c)
        rho_t = reshape(vt, d, d)
        entropy_t[k] = von_neumann_entropy(rho_t; eig_tol = entropy_eig_tol)
        trace_residual = max(trace_residual, abs(real(tr(rho_t)) - 1.0))
        for j in 1:n
            acc = 0.0
            for alpha in 1:d
                acc += occ[j, alpha] * real(rho_t[alpha, alpha])
            end
            populations_t[j, k] = acc
        end
    end

    particle_residual = 0.0
    @inbounds for k in 1:T
        total = 0.0
        for j in 1:n
            total += populations_t[j, k]
        end
        particle_residual = max(particle_residual, abs(total - n_filled))
    end

    metadata = Dict{String,Any}(
        "n_times" => T, "n_orb" => n,
        "n_filled" => n_filled, "entropy_eig_tol" => float(entropy_eig_tol),
        "trace_residual_max" => trace_residual,
        "particle_residual_max" => particle_residual,
    )
    return DynamicsResult(times_vec, entropy_t, populations_t, metadata)
end
