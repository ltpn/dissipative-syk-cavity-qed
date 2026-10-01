# Fixed-filling Hamiltonian, jump operators, and Lindblad generator.

using LinearAlgebra: I, eigvals, norm
using SparseArrays: SparseMatrixCSC, sparse

"""
    LambDickeLiouvillianConfig(; kwargs...)

Configuration for `build_lamb_dicke_liouvillian(config, seed, eta)`.

Omitted synthetic tensors and jumps use the physical model.
Synthetic Hamiltonians are supplied as a full rank-4 `J[j,l,k,m]` tensor.
Synthetic jumps are supplied as one-body matrices `A[j,k]`;  applies
the global `sqrt(gamma_eff)` multiplier when converting them to many-body
jump blocks.

**Filling**: the fixed-filling many-body block
uses `filling` particles out of `n_orb` orbitals; when omitted, defaults to
`div(n_orb, 2)` (half-filling).  Supplying an
explicit `filling` also relaxes the "n_orb must be even" check.
"""
struct LambDickeLiouvillianConfig
    n_orb::Int
    # Number of particles in the fixed-filling sector.  Defaults to
    # `div(n_orb, 2)` (half-filling) when unspecified in the keyword
    # constructor; explicit values must satisfy `0 < filling < n_orb`.
    filling::Int
    n_grid::Int
    box_length::Float64
    weight_type::Symbol
    correlation_length::Float64
    disorder_strength::Float64
    gamma_eff::Float64
    svd_tol::Float64
    J::Float64
    synthetic_hamiltonian_tensor::Union{Array{ComplexF64,4},Nothing}
    synthetic_jump_matrices::Union{Vector{Matrix{ComplexF64}},Nothing}
    lambda_c_micron::Float64
    kappa_over_2pi_mhz::Float64
    delta_cd_over_2pi_mhz::Float64
    cavity_loss_rate_over_J::Float64
    synthetic_cavity_loss_operator::Union{Matrix{ComplexF64},Nothing}
end

function LambDickeLiouvillianConfig(; n_orb::Integer,
                                      filling::Union{Nothing,Integer} = nothing,
                                      n_grid::Integer = 64,
                                      box_length::Real = 12.0,
                                      weight_type::Symbol = :speckle,
                                      correlation_length::Real = 1.0,
                                      disorder_strength::Real = 1.0,
                                      gamma_eff::Real = 1.0,
                                      svd_tol::Real = 1e-8,
                                      J::Real = 1.0,
                                      synthetic_hamiltonian_tensor = nothing,
                                      synthetic_jump_matrices = nothing,

                                      lambda_c_micron::Real = 0.671,

                                      kappa_over_2pi_mhz::Real = 0.2,
                                      delta_cd_over_2pi_mhz::Real = 1.0,
                                      manual_cavity_loss_rate_over_J::Union{Nothing,Real} = nothing,
                                      synthetic_cavity_loss_operator = nothing,
)
    n = Int(n_orb)
    ng = Int(n_grid)
    L = Float64(box_length)
    xi = Float64(correlation_length)
    f = Float64(disorder_strength)
    gamma = Float64(gamma_eff)
    tol = Float64(svd_tol)
    Jf = Float64(J)
    lambda_c = Float64(lambda_c_micron)

    # Resolve the fixed-filling sector.  When the caller does not specify
    # keep the "n_orb must be even" guard.  When `filling` is given
    # explicitly, we allow arbitrary 0 < filling < n_orb (including odd
    # n_orb): the caller has taken responsibility for the physics choice.
    n > 0 || throw(ArgumentError("n_orb must be positive, got $n_orb"))
    filling_resolved = if filling === nothing
        iseven(n) ||
            throw(ArgumentError("half filling requires positive even n_orb, got $n_orb"))
        div(n, 2)
    else
        f_int = Int(filling)
        0 < f_int < n ||
            throw(ArgumentError("filling must satisfy 0 < filling < n_orb, got filling=$f_int n_orb=$n"))
        f_int
    end
    ng > 0 || throw(ArgumentError("n_grid must be positive, got $n_grid"))
    isfinite(L) && L > 0.0 ||
        throw(ArgumentError("box_length must be positive and finite, got $box_length"))
    weight_type in (:uniform, :speckle) ||
        throw(ArgumentError("weight_type must be :uniform or :speckle, got $weight_type"))
    isfinite(xi) && xi > 0.0 ||
        throw(ArgumentError("correlation_length must be positive and finite, got $correlation_length"))
    isfinite(f) && f >= 0.0 ||
        throw(ArgumentError("disorder_strength must be non-negative and finite, got $disorder_strength"))
    isfinite(gamma) && gamma >= 0.0 ||
        throw(ArgumentError("gamma_eff must be non-negative and finite, got $gamma_eff"))
    isfinite(tol) && tol >= 0.0 ||
        throw(ArgumentError("svd_tol must be non-negative and finite, got $svd_tol"))
    isfinite(Jf) && Jf > 0.0 ||
        throw(ArgumentError("J must be positive and finite, got $J"))
    isfinite(lambda_c) && lambda_c > 0.0 ||
        throw(ArgumentError("lambda_c_micron must be positive and finite, got $lambda_c_micron"))

    kappa = Float64(kappa_over_2pi_mhz)
    delta = Float64(delta_cd_over_2pi_mhz)
    isfinite(kappa) && kappa >= 0.0 ||
        throw(ArgumentError("kappa_over_2pi_mhz must be non-negative and finite, got $kappa_over_2pi_mhz"))
    isfinite(delta) ||
        throw(ArgumentError("delta_cd_over_2pi_mhz must be finite, got $delta_cd_over_2pi_mhz"))
    delta > 0 || throw(ArgumentError("delta_cd_over_2pi_mhz must be positive"))
    resolved_cavity_rate = manual_cavity_loss_rate_over_J === nothing ?
        kappa / delta : Float64(manual_cavity_loss_rate_over_J)
    isfinite(resolved_cavity_rate) && resolved_cavity_rate >= 0 ||
        throw(ArgumentError("cavity loss rate must be nonnegative and finite"))
    tensor = synthetic_hamiltonian_tensor === nothing ? nothing :
             Array{ComplexF64,4}(synthetic_hamiltonian_tensor)
    jumps = synthetic_jump_matrices === nothing ? nothing :
            _coerce_one_body_jump_list(synthetic_jump_matrices)
    cavity_op = if synthetic_cavity_loss_operator === nothing
        nothing
    else
        A = Matrix{ComplexF64}(synthetic_cavity_loss_operator)
        size(A, 1) == size(A, 2) == n ||
            throw(DimensionMismatch("synthetic_cavity_loss_operator has shape $(size(A)); expected $((n,n))"))
        all(isfinite, A) ||
            throw(ArgumentError("synthetic_cavity_loss_operator contains non-finite entries"))
        A
    end
    return LambDickeLiouvillianConfig(n, filling_resolved, ng, L, weight_type,
        xi, f, gamma, tol, Jf, tensor, jumps, lambda_c, kappa, delta,
        resolved_cavity_rate, cavity_op)

end

"""
    LiouvillianBuildResult

Container returned by `build_lamb_dicke_liouvillian`.
"""
struct LiouvillianBuildResult
    L::SparseMatrixCSC{ComplexF64,Int}
    H::SparseMatrixCSC{ComplexF64,Int}
    jump_operators::Vector{SparseMatrixCSC{ComplexF64,Int}}
    one_body_jump_matrices::Vector{Matrix{ComplexF64}}
    basis_states::Vector{Vector{Int}}
    coefficients::Coefficients
    jump_factorization::Union{JumpFactorization,Nothing}
    metadata::Dict{String,Any}
end

"""
    many_body_jump_operators_from_matrices(N, basis_states, jump_matrices; gamma_eff=1.0)

Convert one-body matrices `A[j,k]` into fixed-filling many-body jump blocks
using `dissipator_block_direct`.  The returned blocks include the global
factor `sqrt(gamma_eff)`.
"""
function many_body_jump_operators_from_matrices(N::Integer,
                                                basis_states,
                                                jump_matrices;
                                                gamma_eff::Real = 1.0)
    n_orb = Int(N)
    n_orb > 0 || throw(ArgumentError("N must be positive, got $N"))
    gamma = Float64(gamma_eff)
    isfinite(gamma) && gamma >= 0.0 ||
        throw(ArgumentError("gamma_eff must be non-negative and finite, got $gamma_eff"))
    _validate_basis_states(n_orb, basis_states)

    matrices = _coerce_one_body_jump_list(jump_matrices)
    scale = sqrt(gamma)
    out = Vector{SparseMatrixCSC{ComplexF64,Int}}(undef, length(matrices))
    for (idx, A) in enumerate(matrices)
        size(A) == (n_orb, n_orb) ||
            throw(DimensionMismatch("jump matrix $idx has shape $(size(A)); expected $((n_orb,n_orb))"))
        block = dissipator_block_direct(n_orb, basis_states, vec(A))
        out[idx] = sparse(ComplexF64.(scale .* block))
    end
    return out
end

"""
    assemble_lindblad_liouvillian(H, jump_operators) -> SparseMatrixCSC

Assemble the column-vectorized GKSL Liouvillian from a prebuilt fixed-filling
Hamiltonian block and many-body jump operators.
"""
function assemble_lindblad_liouvillian(H::AbstractMatrix, jump_operators)
    size(H, 1) == size(H, 2) ||
        throw(DimensionMismatch("H must be square, got $(size(H))"))
    d = size(H, 1)
    Hs = sparse(ComplexF64.(H))
    Id = sparse(I, d, d)

    L = -1im .* (kron(Id, Hs) .- kron(transpose(Hs), Id))
    for (idx, J) in enumerate(jump_operators)
        size(J) == (d, d) ||
            throw(DimensionMismatch("jump operator $idx has shape $(size(J)); expected $((d,d))"))
        Js = sparse(ComplexF64.(J))
        JdagJ = Js' * Js
        L .+= kron(conj(Js), Js)
        L .-= 0.5 .* kron(Id, JdagJ)
        L .-= 0.5 .* kron(transpose(JdagJ), Id)
    end
    return sparse(ComplexF64.(L))
end

"""
    liouvillian_trace_preservation_residual(L, dim) -> Float64

Return `||vec(I)' * L||_2` for a `dim x dim` density-matrix block.
"""
function liouvillian_trace_preservation_residual(L::AbstractMatrix, dim::Integer)
    d = Int(dim)
    d > 0 || throw(ArgumentError("dim must be positive, got $dim"))
    size(L) == (d^2, d^2) ||
        throw(DimensionMismatch("L has shape $(size(L)); expected $((d^2,d^2))"))
    trace_vec = vec(Matrix{ComplexF64}(I, d, d))
    return norm(transpose(trace_vec) * L)
end

"""
    left_identity_residual(L, dim) -> Float64

Alias for the trace-preservation residual, named for  reports.
"""
left_identity_residual(L::AbstractMatrix, dim::Integer) =
    liouvillian_trace_preservation_residual(L, dim)

"""
    build_lamb_dicke_liouvillian(config, seed, eta) -> LiouvillianBuildResult

End-to-end  builder for physical/synthetic Hamiltonian and jump sources.
"""
function build_lamb_dicke_liouvillian(config::LambDickeLiouvillianConfig,
                                      seed::Integer,
                                      eta::Real)
    eta_f = Float64(eta)
    isfinite(eta_f) && eta_f >= 0.0 ||
        throw(ArgumentError("eta must be non-negative and finite, got $eta"))

    params = _single_particle_params_from_config(config, seed)
    coefficients = compute_coefficients(params)
    basis_states = generate_vectors(config.n_orb, config.filling)

    H_tensor = _hamiltonian_tensor_from_config(config, coefficients)
    H = sparse(ComplexF64.(syk4_block_from_tensor(config.n_orb, basis_states, H_tensor)))
    one_body_jumps, factorization = _one_body_jumps_from_config(config, params,
                                                                coefficients, eta_f)
    spontaneous_jump_operators = many_body_jump_operators_from_matrices(
        config.n_orb, basis_states, one_body_jumps; gamma_eff = config.gamma_eff)

    effective_cavity_loss_rate = config.cavity_loss_rate_over_J * config.J
    cavity_one_body_jumps = Matrix{ComplexF64}[]
    cavity_jump_operators = SparseMatrixCSC{ComplexF64,Int}[]
    if effective_cavity_loss_rate > 0
        cavity_matrix = _cavity_loss_one_body_matrix(config, coefficients)
        push!(cavity_one_body_jumps, cavity_matrix)
        cavity_jump_operators = many_body_jump_operators_from_matrices(
            config.n_orb, basis_states, cavity_one_body_jumps;
            gamma_eff = effective_cavity_loss_rate)
    end

    jump_operators = vcat(spontaneous_jump_operators, cavity_jump_operators)
    all_one_body_jumps = vcat(one_body_jumps, cavity_one_body_jumps)
    L = assemble_lindblad_liouvillian(H, jump_operators)

    metadata = Dict{String,Any}(
        "n_orb" => config.n_orb,
        "filling" => config.filling,
        "n_grid" => config.n_grid,
        "box_length" => config.box_length,
        "correlation_length" => config.correlation_length,
        "disorder_strength" => config.disorder_strength,
        "seed" => Int(seed),
        "eta" => eta_f,
        "lambda_c_micron" => config.lambda_c_micron,
        "gamma_eff_over_J" => config.gamma_eff / config.J,
        "J" => config.J,
        "svd_tol" => config.svd_tol,
        "effective_gamma_eff" => config.gamma_eff,
        "effective_gamma_eff_over_J" => config.gamma_eff / config.J,
        "retained_jump_rank" => factorization === nothing ? length(one_body_jumps) :
                                factorization.retained_rank,
        "discarded_jump_weight" => factorization === nothing ? 0.0 :
                                  factorization.discarded_weight,
        "kernel_min_eig" => factorization === nothing ? NaN :
                            factorization.min_eig,
        "spontaneous_emission_participation_ratio" => factorization === nothing ? NaN :
                                                      factorization.participation_ratio,
        "kappa_over_2pi_mhz" => config.kappa_over_2pi_mhz,
        "delta_cd_over_2pi_mhz" => config.delta_cd_over_2pi_mhz,
        "cavity_loss_rate_over_J" => config.cavity_loss_rate_over_J,
        "effective_cavity_loss_rate" => effective_cavity_loss_rate,
        "effective_cavity_loss_rate_over_J" => effective_cavity_loss_rate / config.J,
        "n_spontaneous_emission_jumps" => length(spontaneous_jump_operators),
        "n_cavity_loss_jumps" => length(cavity_jump_operators),
        "n_total_jumps" => length(jump_operators),
    )

    return LiouvillianBuildResult(L, H, jump_operators, all_one_body_jumps,
                                  basis_states, coefficients, factorization,
                                  metadata)
end

# ============================================================================
# Hamiltonian block builder
# ============================================================================

"""
    syk4_block_from_tensor(n_orb, basis_states, tensor) -> SparseMatrixCSC

Build the fixed-filling many-body SYK4 Hamiltonian block from a rank-4 coupling
tensor `J[j,l,k,m]` (creation pair `j,l`; annihilation pair `k,m`) using the
generic `SYK4_block_direct` builder from `SYK_setup.jl`.

The tensor is reduced to the per-combination coefficient vector over
`Combinations_SYK4` (the canonical convention: creation pair `j < l`,
annihilation pair `k < m`, one representative per Hermitian-conjugate pair); the
builder's factor-4 weight and explicit conjugate fill reconstruct the full
antisymmetric quartic sum and a Hermitian block.

The tensor already carries the coherent prefactor J and is used directly.
"""
function syk4_block_from_tensor(n_orb::Integer, basis_states,
                                tensor::AbstractArray{<:Number,4})
    n = Int(n_orb)
    size(tensor) == (n, n, n, n) ||
        throw(DimensionMismatch("tensor has shape $(size(tensor)); expected $((n,n,n,n))"))
    combos = Combinations_SYK4(n)
    J_list = ComplexF64[tensor[c[1], c[2], c[3], c[4]] for c in combos]
    return SYK4_block_direct(n, basis_states, J_list)
end

# ============================================================================
# Internal helpers
# ============================================================================

function _single_particle_params_from_config(config::LambDickeLiouvillianConfig,
                                    seed::Integer)
    weight = if config.weight_type == :uniform
        WeightParams(weight_type = :uniform)
    else
        WeightParams(weight_type = :speckle,
                     speckle = SpeckleParams(
                         correlation_length = config.correlation_length,
                         disorder_strength = config.disorder_strength,
                         seed = seed))
    end
    return SingleParticleParams(
        grid = GridParams(box_length = config.box_length,
                          n_grid = config.n_grid),
        orbitals = OrbitalParams(n_orb = config.n_orb),
        weight = weight,
    )
end

function _hamiltonian_tensor_from_config(config::LambDickeLiouvillianConfig,
                                         coefficients::Coefficients)
    if config.synthetic_hamiltonian_tensor === nothing
        return build_physical_syk_tensor(coefficients.g_jk; J = config.J)
    end
    tensor = config.synthetic_hamiltonian_tensor
    size(tensor) == (config.n_orb, config.n_orb, config.n_orb, config.n_orb) ||
        throw(DimensionMismatch("synthetic_hamiltonian_tensor has shape $(size(tensor)); expected $((config.n_orb,config.n_orb,config.n_orb,config.n_orb))"))
    return tensor
end

function _one_body_jumps_from_config(config::LambDickeLiouvillianConfig,
                                     params::SingleParticleParams,
                                     coefficients::Coefficients,
                                     eta::Float64)
    if config.synthetic_jump_matrices === nothing
        K_pair = kernel_pair_matrix(eta, params, coefficients)
        fact = jump_matrices_from_kernel(K_pair; svd_tol = config.svd_tol)
        jumps = [ComplexF64.(J) for J in fact.jump_matrices]
        return jumps, fact
    end
    jumps = config.synthetic_jump_matrices
    for (idx, J) in enumerate(jumps)
        size(J) == (config.n_orb, config.n_orb) ||
            throw(DimensionMismatch("synthetic jump matrix $idx has shape $(size(J)); expected $((config.n_orb,config.n_orb))"))
    end
    return jumps, nothing
end

function _cavity_loss_one_body_matrix(config::LambDickeLiouvillianConfig,
                                      coefficients::Coefficients)
    if config.synthetic_cavity_loss_operator !== nothing
        return config.synthetic_cavity_loss_operator
    end
    return ComplexF64.(coefficients.g_jk)
end

function _coerce_one_body_jump_list(jump_matrices)
    matrices = if jump_matrices isa AbstractMatrix
        [jump_matrices]
    else
        collect(jump_matrices)
    end
    out = Matrix{ComplexF64}[]
    sizehint!(out, length(matrices))
    for (idx, A) in enumerate(matrices)
        size(A, 1) == size(A, 2) ||
            throw(DimensionMismatch("jump matrix $idx must be square, got $(size(A))"))
        all(isfinite, A) ||
            throw(ArgumentError("jump matrix $idx contains non-finite entries"))
        push!(out, Matrix{ComplexF64}(A))
    end
    return out
end

function _validate_basis_states(n_orb::Int, basis_states)
    isempty(basis_states) &&
        throw(ArgumentError("basis_states must not be empty"))
    filling = sum(basis_states[1])
    for (idx, state) in enumerate(basis_states)
        length(state) == n_orb ||
            throw(DimensionMismatch("basis state $idx has length $(length(state)); expected $n_orb"))
        all(x -> x == 0 || x == 1, state) ||
            throw(ArgumentError("basis state $idx contains entries outside {0,1}"))
        sum(state) == filling ||
            throw(ArgumentError("basis state $idx has filling $(sum(state)); expected $filling"))
    end
    return nothing
end
