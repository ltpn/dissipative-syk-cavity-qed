# Multimode cavity-mediated interactions and cavity-loss channels (arXiv:2303.11343).

if Sys.isapple()
    @eval using AppleAccelerate
end
using LinearAlgebra: norm
using SparseArrays: SparseMatrixCSC, sparse

struct MultimodeParams
    n_orb::Int
    # Number of particles in the fixed-filling sector.  Defaults to
    # `div(n_orb, 2)` (half-filling) when not supplied to the keyword
    # constructor.  Explicit values must satisfy `0 < filling < n_orb` and
    filling::Int
    box_length::Float64
    n_grid::Int
    zeta::Float64
    mode_cutoff::Int
    delta_tilde::Float64
    weight_type::Symbol
    speckle_grains_per_side::Float64
    speckle_correlation_length::Union{Nothing,Float64}
    disorder_strength::Float64
    drive_wavevector::Float64
    energy_scale::Float64
end

function MultimodeParams(; n_orb::Integer = 14,
                           filling::Union{Nothing,Integer} = nothing,
                           box_length::Real = 10.0,
                           n_grid::Integer = 200,
                           zeta::Real = 1.0,
                           mode_cutoff::Integer = 240,
                           delta_tilde::Real = 0.1,
                           weight_type::Symbol = :speckle,
                           speckle_grains_per_side::Real = 17.0,
                           speckle_correlation_length = nothing,
                           disorder_strength::Real = 1.0,
                           drive_wavevector::Real = 0.0,
                           energy_scale::Real = 1.0)
    n_int = Int(n_orb)
    filling_resolved = if filling === nothing
        n_int > 0 && iseven(n_int) ||
            throw(ArgumentError("half-filling default requires positive even n_orb, got $(n_orb)"))
        div(n_int, 2)
    else
        f_int = Int(filling)
        (n_int > 0 && 0 < f_int < n_int) ||
            throw(ArgumentError("filling must satisfy 0 < filling < n_orb, got filling=$(f_int) n_orb=$(n_int)"))
        f_int
    end
    p = MultimodeParams(n_int, filling_resolved, Float64(box_length), Int(n_grid),
                        Float64(zeta), Int(mode_cutoff), Float64(delta_tilde),
                        weight_type, Float64(speckle_grains_per_side),
                        speckle_correlation_length === nothing ? nothing : Float64(speckle_correlation_length),
                        Float64(disorder_strength), Float64(drive_wavevector),
                        Float64(energy_scale))
    validate_multimode_params(p)
    return p
end

struct MultimodeCavityModes
    quanta::Vector{Tuple{Int,Int}}
    msum::Vector{Int}
    values::Array{Float64,3}
end

struct MultimodeTensorResult
    params::MultimodeParams
    seed::Int
    u::Vector{Float64}
    du::Float64
    orbitals::OrbitalParams
    intensity::Matrix{Float64}
    weight::Matrix{Float64}
    modes::MultimodeCavityModes
    integrals::Array{ComplexF64,3}
    tensor::Array{ComplexF64,4}
end

struct MultimodeIntegralCache
    params::MultimodeParams
    seed::Int
    u::Vector{Float64}
    du::Float64
    orbitals::OrbitalParams
    intensity::Matrix{Float64}
    weight::Matrix{Float64}
    modes::MultimodeCavityModes
    integrals::Array{ComplexF64,3}
end

struct MultimodeHamiltonianBlock
    H
    basis_states::Vector{Vector{Int64}}
    tensor_result::MultimodeTensorResult
end

struct MultimodeCavityLossParams
    kappa_over_2pi_mhz::Float64
    delta_cd_over_2pi_mhz::Float64
    gamma_over_2pi_mhz::Float64
    delta_da_over_2pi_mhz::Float64
    rate_scale::Float64
end

function MultimodeCavityLossParams(; kappa_over_2pi_mhz::Real = 0.2,
                                     delta_cd_over_2pi_mhz::Real,
                                     gamma_over_2pi_mhz::Real = 5.9,
                                     delta_da_over_2pi_mhz::Real = 3000.0,
                                     rate_scale::Real = 1.0)
    loss = MultimodeCavityLossParams(Float64(kappa_over_2pi_mhz),
                                     Float64(delta_cd_over_2pi_mhz),
                                     Float64(gamma_over_2pi_mhz),
                                     Float64(delta_da_over_2pi_mhz),
                                     Float64(rate_scale))
    validate_multimode_cavity_loss_params(loss)
    return loss
end

struct MultimodeOpenSystemBlock
    L::SparseMatrixCSC{ComplexF64,Int}
    H::SparseMatrixCSC{ComplexF64,Int}
    jump_operators::Vector{SparseMatrixCSC{ComplexF64,Int}}
    one_body_jump_matrices::Vector{Matrix{ComplexF64}}
    basis_states::Vector{Vector{Int64}}
    tensor_result::MultimodeTensorResult
    loss_params::MultimodeCavityLossParams
    metadata::Dict{String,Any}
end

function validate_multimode_params(p::MultimodeParams)
    # Filling is validated in the keyword constructor; here we just make
    # sure the stored (n_orb, filling) pair is internally consistent.
    p.n_orb > 0 ||
        throw(ArgumentError("n_orb must be positive, got $(p.n_orb)"))
    0 < p.filling < p.n_orb ||
        throw(ArgumentError("filling must satisfy 0 < filling < n_orb, got filling=$(p.filling) n_orb=$(p.n_orb)"))
    p.box_length > 0.0 && isfinite(p.box_length) ||
        throw(ArgumentError("box_length must be positive and finite, got $(p.box_length)"))
    p.n_grid > 0 ||
        throw(ArgumentError("n_grid must be positive, got $(p.n_grid)"))
    p.zeta > 0.0 && isfinite(p.zeta) ||
        throw(ArgumentError("zeta must be positive and finite, got $(p.zeta)"))
    p.mode_cutoff >= 0 ||
        throw(ArgumentError("mode_cutoff must be non-negative, got $(p.mode_cutoff)"))
    p.delta_tilde >= 0.0 && isfinite(p.delta_tilde) ||
        throw(ArgumentError("delta_tilde must be non-negative and finite, got $(p.delta_tilde)"))
    p.weight_type in (:uniform, :speckle) ||
        throw(ArgumentError("weight_type must be :uniform or :speckle, got $(p.weight_type)"))
    p.speckle_grains_per_side > 0.0 && isfinite(p.speckle_grains_per_side) ||
        throw(ArgumentError("speckle_grains_per_side must be positive and finite"))
    if p.speckle_correlation_length !== nothing
        p.speckle_correlation_length > 0.0 && isfinite(p.speckle_correlation_length) ||
            throw(ArgumentError("speckle_correlation_length must be positive and finite"))
    end
    p.disorder_strength >= 0.0 && isfinite(p.disorder_strength) ||
        throw(ArgumentError("disorder_strength must be non-negative and finite"))
    isfinite(p.drive_wavevector) ||
        throw(ArgumentError("drive_wavevector must be finite"))
    isfinite(p.energy_scale) ||
        throw(ArgumentError("energy_scale must be finite"))
    return nothing
end

function validate_multimode_cavity_loss_params(loss::MultimodeCavityLossParams)
    loss.kappa_over_2pi_mhz >= 0.0 && isfinite(loss.kappa_over_2pi_mhz) ||
        throw(ArgumentError("kappa_over_2pi_mhz must be non-negative and finite"))
    loss.delta_cd_over_2pi_mhz > 0.0 && isfinite(loss.delta_cd_over_2pi_mhz) ||
        throw(ArgumentError("delta_cd_over_2pi_mhz must be positive and finite"))
    loss.gamma_over_2pi_mhz >= 0.0 && isfinite(loss.gamma_over_2pi_mhz) ||
        throw(ArgumentError("gamma_over_2pi_mhz must be non-negative and finite"))
    loss.delta_da_over_2pi_mhz > 0.0 && isfinite(loss.delta_da_over_2pi_mhz) ||
        throw(ArgumentError("delta_da_over_2pi_mhz must be positive and finite"))
    loss.rate_scale >= 0.0 && isfinite(loss.rate_scale) ||
        throw(ArgumentError("rate_scale must be non-negative and finite"))
    return nothing
end

function multimode_inverse_cantor(m::Integer)
    m >= 0 || throw(ArgumentError("Cantor index must be non-negative, got $m"))
    mf = Int(m)
    s = Int(floor((sqrt(8.0 * mf + 1.0) - 1.0) / 2.0))
    t = div(s * (s + 1), 2)
    ny = mf - t
    nx = s - ny
    return nx, ny
end

function multimode_cavity_modes(u::AbstractVector{<:Real};
                                zeta::Real,
                                mode_cutoff::Integer)
    zeta_f = Float64(zeta)
    zeta_f > 0.0 && isfinite(zeta_f) || throw(ArgumentError("zeta must be positive and finite"))
    M = Int(mode_cutoff)
    M >= 0 || throw(ArgumentError("mode_cutoff must be non-negative"))

    quanta = [multimode_inverse_cantor(m) for m in 0:M]
    msum = [nx + ny for (nx, ny) in quanta]
    nmax = maximum(msum; init = 0)
    psi = ho_1d_basis(zeta_f .* Float64.(collect(u)), nmax)
    N = length(u)
    values = Array{Float64,3}(undef, N, N, M + 1)
    @inbounds for q in 1:(M + 1)
        nx, ny = quanta[q]
        psi_x = view(psi, :, nx + 1)
        psi_y = view(psi, :, ny + 1)
        for j in 1:N, i in 1:N
            values[i, j, q] = zeta_f * psi_x[i] * psi_y[j]
        end
    end
    return MultimodeCavityModes(quanta, msum, values)
end

function multimode_interaction_tensor(p::MultimodeParams; seed::Integer = 1)
    return multimode_tensor_from_cache(multimode_integral_cache(p; seed = seed))
end

function multimode_integral_cache(p::MultimodeParams; seed::Integer = 1)
    validate_multimode_params(p)
    grid = GridParams(box_length = p.box_length, n_grid = p.n_grid)
    orbitals = OrbitalParams(n_orb = p.n_orb)
    u, du, phi = build_orbitals_2d(grid, orbitals)
    intensity, weight = _multimode_intensity_and_weight(p, grid, Int(seed))
    modes = multimode_cavity_modes(u; zeta = p.zeta, mode_cutoff = p.mode_cutoff)
    integrals = multimode_interaction_integrals(phi, u, du, weight, modes;
                                                drive_wavevector = p.drive_wavevector)
    return MultimodeIntegralCache(p, Int(seed), u, du, orbitals, intensity, weight,
                                  modes, integrals)
end

function multimode_tensor_from_cache(cache::MultimodeIntegralCache;
                                     delta_tilde::Real = cache.params.delta_tilde,
                                     energy_scale::Real = cache.params.energy_scale)
    p = _multimode_params_with(cache.params;
                               delta_tilde = delta_tilde,
                               energy_scale = energy_scale)
    tensor = multimode_tensor_from_integrals(cache.integrals, cache.modes;
                                             delta_tilde = p.delta_tilde,
                                             energy_scale = p.energy_scale)
    return MultimodeTensorResult(p, cache.seed, cache.u, cache.du, cache.orbitals,
                                 cache.intensity, cache.weight, cache.modes,
                                 cache.integrals, tensor)
end

function multimode_interaction_integrals(phi::AbstractArray{<:Real,3},
                                         u::AbstractVector{<:Real},
                                         du::Real,
                                         weight::AbstractMatrix{<:Number},
                                         modes::MultimodeCavityModes;
                                         drive_wavevector::Real = 0.0)
    N1, N2, n_orb = size(phi)
    size(weight) == (N1, N2) ||
        throw(DimensionMismatch("weight size $(size(weight)) does not match orbital grid $((N1,N2))"))
    size(modes.values, 1) == N1 && size(modes.values, 2) == N2 ||
        throw(DimensionMismatch("cavity mode grid $(size(modes.values)[1:2]) does not match orbital grid $((N1,N2))"))
    phi_flat = reshape(phi, N1 * N2, n_orb)
    drive = _multimode_drive_profile(u, N2, Float64(drive_wavevector))
    base = ComplexF64.(weight) .* drive
    n_modes = size(modes.values, 3)
    integrals = Array{ComplexF64,3}(undef, n_orb, n_orb, n_modes)
    @inbounds for q in 1:n_modes
        W_flat = vec(base .* modes.values[:, :, q])
        phi_w = phi_flat .* W_flat
        # Eq. (S27) as written: measure `du^2` only. The 1/4 antisymmetrization
        # normalizer belongs to `multimode_tensor_from_integrals`, not here --
        # these overlaps also feed the cavity-loss jumps, which are linear in
        # them (see the convention note at the top of this file).
        integrals[:, :, q] .= (Float64(du)^2) .* (transpose(phi_w) * phi_flat)
    end
    return integrals
end

function multimode_tensor_from_integrals(integrals::AbstractArray{<:Complex,3},
                                         modes::MultimodeCavityModes;
                                         delta_tilde::Real,
                                         energy_scale::Real = 1.0)
    n1, n2, n_modes = size(integrals)
    n1 == n2 || throw(DimensionMismatch("integrals must have shape (N,N,M), got $(size(integrals))"))
    n_modes == length(modes.msum) ||
        throw(DimensionMismatch("integrals mode count $n_modes does not match modes $(length(modes.msum))"))
    delta = Float64(delta_tilde)
    delta >= 0.0 && isfinite(delta) || throw(ArgumentError("delta_tilde must be non-negative and finite"))
    scale = Float64(energy_scale)
    isfinite(scale) || throw(ArgumentError("energy_scale must be finite"))

    tensor = zeros(ComplexF64, n1, n1, n1, n1)
    @inbounds for q in 1:n_modes
        Iq = view(integrals, :, :, q)
        # 0.25 = antisymmetrization normalizer (see the convention note at the
        # top of this file and `PhysicalSYK.build_physical_syk_tensor`): the
        # unrestricted 4-term sum below over-counts the Eq. (S32)/(S33)
        # coupling by 4.
        wq = 0.25 * scale / (1.0 + modes.msum[q] * delta)
        for m_ann in 1:n1, k_ann in 1:n1, l_cre in 1:n1, j_cre in 1:n1
            tensor[j_cre, l_cre, k_ann, m_ann] += wq * (
                Iq[j_cre, k_ann] * conj(Iq[m_ann, l_cre]) -
                Iq[l_cre, k_ann] * conj(Iq[m_ann, j_cre]) -
                Iq[j_cre, m_ann] * conj(Iq[k_ann, l_cre]) +
                Iq[l_cre, m_ann] * conj(Iq[k_ann, j_cre])
            )
        end
    end
    return tensor
end

function multimode_hamiltonian_block(p::MultimodeParams; seed::Integer = 1)
    result = multimode_interaction_tensor(p; seed = seed)
    basis_states = generate_vectors(p.n_orb, p.filling)
    H = _multimode_syk4_block_from_tensor(p.n_orb, basis_states, result.tensor)
    return MultimodeHamiltonianBlock(H, basis_states, result)
end

function multimode_cavity_jump_matrices(cache::MultimodeIntegralCache,
                                        loss::MultimodeCavityLossParams;
                                        delta_tilde::Real = cache.params.delta_tilde)
    validate_multimode_cavity_loss_params(loss)
    rate = multimode_cavity_loss_rate_over_energy(loss)
    rate == 0.0 && return Matrix{ComplexF64}[]

    delta = Float64(delta_tilde)
    delta >= 0.0 && isfinite(delta) ||
        throw(ArgumentError("delta_tilde must be non-negative and finite"))
    loss_integrals = cache.integrals
    matrices = Matrix{ComplexF64}[]
    sizehint!(matrices, size(loss_integrals, 3))
    @inbounds for q in axes(loss_integrals, 3)
        # Leading-order mode weight q_mu = 1/(1 + m_mu * delta_tilde), i.e. the
        # kappa -> 0 limit of Eq. (S35).  See the "Cavity linewidth" note at the
        # top of this file: the correction is dropped on BOTH sides, never one.
        mode_denominator = 1.0 + cache.modes.msum[q] * delta
        push!(matrices, Matrix{ComplexF64}(@view loss_integrals[:, :, q]) ./ mode_denominator)
    end
    return matrices
end

function multimode_open_system_block(p::MultimodeParams,
                                     loss::MultimodeCavityLossParams;
                                     seed::Integer = 1)
    cache = multimode_integral_cache(p; seed = seed)
    result = multimode_tensor_from_cache(cache)
    basis_states = generate_vectors(p.n_orb, p.filling)
    H = sparse(ComplexF64.(_multimode_syk4_block_from_tensor(p.n_orb, basis_states, result.tensor)))
    one_body_jumps = multimode_cavity_jump_matrices(cache, loss;
                                                    delta_tilde = p.delta_tilde)
    rate = multimode_cavity_loss_rate_over_energy(loss)
    jump_operators = isempty(one_body_jumps) ?
        SparseMatrixCSC{ComplexF64,Int}[] :
        many_body_jump_operators_from_matrices(p.n_orb, basis_states,
                                               one_body_jumps; gamma_eff = rate)
    L = assemble_lindblad_liouvillian(H, jump_operators)
    jump_weights = [norm(J)^2 for J in jump_operators]
    pr_denom = sum(abs2, jump_weights)
    d = size(H, 1)
    metadata = Dict{String,Any}(
        "seed" => Int(seed),
        "n_orb" => p.n_orb,
        "filling" => p.filling,
        "hilbert_dim" => d,
        "liouvillian_dim" => d^2,
        "box_length" => p.box_length,
        "n_grid" => p.n_grid,
        "zeta" => p.zeta,
        "mode_cutoff" => p.mode_cutoff,
        "delta_tilde" => p.delta_tilde,
        "weight_type" => String(p.weight_type),
        "speckle_grains_per_side" => p.speckle_grains_per_side,
        "disorder_strength" => p.disorder_strength,
        "drive_wavevector" => p.drive_wavevector,
        "energy_scale" => p.energy_scale,
        "kappa_over_2pi_mhz" => loss.kappa_over_2pi_mhz,
        "delta_cd_over_2pi_mhz" => loss.delta_cd_over_2pi_mhz,
        "kappa_over_delta_cd" => multimode_kappa_over_delta_cd(loss),
        "gamma_over_2pi_mhz" => loss.gamma_over_2pi_mhz,
        "delta_da_over_2pi_mhz" => loss.delta_da_over_2pi_mhz,
        "gamma_over_delta_da" => multimode_gamma_over_delta_da(loss),
        "rate_scale" => loss.rate_scale,
        "effective_cavity_loss_rate_over_energy" => rate,
        "n_spontaneous_emission_jumps" => 0,
        "n_cavity_loss_jumps" => length(jump_operators),
        "cavity_loss_participation_ratio" => pr_denom == 0.0 ? 0.0 : sum(jump_weights)^2 / pr_denom,
        "n_total_jumps" => length(jump_operators),
    )
    return MultimodeOpenSystemBlock(L, H, jump_operators, one_body_jumps,
                                    basis_states, result, loss, metadata)
end

multimode_kappa_over_delta_cd(loss::MultimodeCavityLossParams) =
    loss.kappa_over_2pi_mhz / loss.delta_cd_over_2pi_mhz

multimode_gamma_over_delta_da(loss::MultimodeCavityLossParams) =
    loss.gamma_over_2pi_mhz / loss.delta_da_over_2pi_mhz

multimode_cavity_loss_rate_over_energy(loss::MultimodeCavityLossParams) =
    multimode_kappa_over_delta_cd(loss) * loss.rate_scale

function _multimode_intensity_and_weight(p::MultimodeParams, grid::GridParams, seed::Int)
    if p.weight_type == :uniform
        return zeros(Float64, grid.n_grid, grid.n_grid), weight_field_uniform(grid)
    end
    xi = p.speckle_correlation_length === nothing ?
        p.box_length / p.speckle_grains_per_side :
        p.speckle_correlation_length
    sp = SpeckleParams(correlation_length = xi,
                       disorder_strength = p.disorder_strength,
                       seed = seed)
    intensity = speckle_intensity(grid, sp)
    return intensity, weight_field(intensity, p.disorder_strength)
end

function _multimode_params_with(p::MultimodeParams;
                                delta_tilde::Real = p.delta_tilde,
                                energy_scale::Real = p.energy_scale)
    return MultimodeParams(n_orb = p.n_orb,
                           filling = p.filling,
                           box_length = p.box_length,
                           n_grid = p.n_grid,
                           zeta = p.zeta,
                           mode_cutoff = p.mode_cutoff,
                           delta_tilde = delta_tilde,
                           weight_type = p.weight_type,
                           speckle_grains_per_side = p.speckle_grains_per_side,
                           speckle_correlation_length = p.speckle_correlation_length,
                           disorder_strength = p.disorder_strength,
                           drive_wavevector = p.drive_wavevector,
                           energy_scale = energy_scale)
end

function _multimode_drive_profile(u::AbstractVector{<:Real}, n_y::Int, k::Float64)
    N = length(u)
    profile = Matrix{ComplexF64}(undef, N, n_y)
    if k == 0.0
        fill!(profile, 1.0 + 0.0im)
    else
        @inbounds for j in 1:n_y, i in 1:N
            profile[i, j] = exp(1im * k * Float64(u[i]))
        end
    end
    return profile
end

function _multimode_syk4_block_from_tensor(n_orb::Integer, basis_states,
                                           tensor::AbstractArray{<:Number,4})
    n = Int(n_orb)
    size(tensor) == (n, n, n, n) ||
        throw(DimensionMismatch("tensor has shape $(size(tensor)); expected $((n,n,n,n))"))
    combos = Combinations_SYK4(n)
    J_list = ComplexF64[tensor[c[1], c[2], c[3], c[4]] for c in combos]
    return SYK4_block_direct(n, basis_states, J_list)
end

_multimode_isfinite(z::Complex) = isfinite(real(z)) && isfinite(imag(z))
_multimode_isfinite(x::Real) = isfinite(x)
