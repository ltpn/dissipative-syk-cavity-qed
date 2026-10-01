module CGSSurvival

using LinearAlgebra
using JLD2
using Random: AbstractRNG
using Statistics: mean, quantile

export BetaZeroCGS, ExactCGSSpectralData,
       phase_fix_eigenvectors!, beta_zero_cgs,
       reconstruct_survival, exact_cgs_spectral_data,
       bootstrap_seed_mean,
       validate_seed_payload, write_seed_payload, read_seed_payload

const SEED_PAYLOAD_REQUIRED_KEYS = (
    "identity/production",
    "identity/dataset_key",
    "identity/family",
    "identity/parameter_name",
    "identity/parameter",
    "identity/seed",
    "dimensions/n_orb",
    "dimensions/filling",
    "dimensions/hilbert",
    "dimensions/liouvillian",
    "model/resolved_metadata",
    "cgs/energies",
    "cgs/eigenvectors",
    "cgs/pivots",
    "cgs/phase_corrections",
    "cgs/psi",
    "spectral/eigenvalues",
    "spectral/left_overlaps",
    "spectral/right_coefficients",
    "spectral/weights",
    "survival/times",
    "survival/complex_curve",
    "survival/real_curve",
    "survival/steady_mode_indices",
    "survival/steady_mode_weights",
    "diagnostics/values",
    "diagnostics/pass",
)

struct BetaZeroCGS
    energies::Vector{Float64}
    eigenvectors::Matrix{ComplexF64}
    pivots::Vector{Int}
    phase_corrections::Vector{ComplexF64}
    psi::Vector{ComplexF64}
    rho::Matrix{ComplexF64}
    diagnostics::Dict{String,Float64}
end

struct ExactCGSSpectralData
    eigenvalues::Vector{ComplexF64}
    left_overlaps::Vector{ComplexF64}
    right_coefficients::Vector{ComplexF64}
    weights::Vector{ComplexF64}
    complex_curve::Vector{ComplexF64}
    real_curve::Vector{Float64}
    diagnostics::Dict{String,Float64}
end

function _relative_residual(numerator::Real, denominator::Real)
    return Float64(numerator) / max(Float64(denominator), eps(Float64))
end

function phase_fix_eigenvectors!(vectors::AbstractMatrix)
    nrows, ncols = size(vectors)
    nrows > 0 && ncols > 0 ||
        throw(ArgumentError("eigenvector matrix must be nonempty"))
    pivots = Vector{Int}(undef, ncols)
    corrections = Vector{ComplexF64}(undef, ncols)
    for column_index in axes(vectors, 2)
        column = view(vectors, :, column_index)
        magnitudes = abs.(column)
        maximum_magnitude = maximum(magnitudes)
        maximum_magnitude > 0 ||
            throw(ArgumentError("eigenvector $column_index is zero"))
        pivot = findfirst(==(maximum_magnitude), magnitudes)
        pivot === nothing && error("failed to resolve eigenvector pivot")
        correction = conj(column[pivot]) / maximum_magnitude
        column .*= correction
        column[pivot] = ComplexF64(abs(real(column[pivot])), 0.0)
        pivots[column_index] = pivot
        corrections[column_index] = correction
    end
    return pivots, corrections
end

function _reject_numerical_degeneracies(energies::AbstractVector,
                                         degeneracy_factor::Real)
    factor = Float64(degeneracy_factor)
    isfinite(factor) && factor > 0 ||
        throw(ArgumentError("degeneracy_factor must be positive and finite"))
    length(energies) <= 1 && return nothing
    span = Float64(maximum(energies) - minimum(energies))
    for index in 1:(length(energies) - 1)
        left = Float64(energies[index])
        right = Float64(energies[index + 1])
        scale = max(1.0, abs(left), abs(right), span)
        tolerance = factor * eps(Float64) * scale
        right - left > tolerance || throw(ArgumentError(
            "Hamiltonian eigenvalues $index and $(index + 1) are unresolved " *
            "within tolerance $tolerance"))
    end
    return nothing
end

function beta_zero_cgs(H::AbstractMatrix; degeneracy_factor::Real = 256)
    size(H, 1) == size(H, 2) ||
        throw(DimensionMismatch("Hamiltonian must be square"))
    d = size(H, 1)
    d > 0 || throw(ArgumentError("Hamiltonian must be nonempty"))
    H_dense = Matrix{ComplexF64}(H)
    all(isfinite, H_dense) ||
        throw(ArgumentError("Hamiltonian contains non-finite entries"))
    h_norm = norm(H_dense)
    hermiticity_residual = _relative_residual(
        norm(H_dense - H_dense'), max(h_norm, 1.0))
    hermiticity_residual <= 1e-12 || throw(ArgumentError(
        "Hamiltonian is not Hermitian: relative residual=$hermiticity_residual"))

    H_hermitian = Hermitian(0.5 .* (H_dense .+ H_dense'))
    factorization = eigen(H_hermitian)
    order = sortperm(factorization.values)
    energies = Float64.(factorization.values[order])
    vectors = Matrix{ComplexF64}(factorization.vectors[:, order])
    _reject_numerical_degeneracies(energies, degeneracy_factor)
    pivots, phase_corrections = phase_fix_eigenvectors!(vectors)

    amplitudes = fill(inv(sqrt(Float64(d))), d)
    psi = vectors * amplitudes
    rho = psi * psi'
    eigenpair_residual = _relative_residual(
        norm(H_dense * vectors - vectors * Diagonal(energies)),
        max(norm(H_dense) * norm(vectors), 1.0))
    orthogonality_residual = norm(vectors' * vectors - I) / max(Float64(d), 1.0)
    diagnostics = Dict{String,Float64}(
        "hermiticity_residual" => hermiticity_residual,
        "eigenpair_residual" => eigenpair_residual,
        "orthogonality_residual" => orthogonality_residual,
        "state_norm_error" => abs(real(dot(psi, psi)) - 1.0),
        "rho_trace_error" => abs(real(tr(rho)) - 1.0),
        "rho_purity_error" => abs(real(tr(rho * rho)) - 1.0),
        "rho_hermiticity_residual" => norm(rho - rho'),
    )
    return BetaZeroCGS(energies, vectors, pivots, phase_corrections,
                       psi, rho, diagnostics)
end

function reconstruct_survival(eigenvalues::AbstractVector,
                              weights::AbstractVector,
                              times::AbstractVector)
    length(eigenvalues) == length(weights) || throw(DimensionMismatch(
        "eigenvalues and weights must have equal lengths"))
    all(isfinite, eigenvalues) ||
        throw(ArgumentError("eigenvalues contain non-finite entries"))
    all(isfinite, weights) ||
        throw(ArgumentError("weights contain non-finite entries"))
    all(t -> isfinite(t) && t >= 0, times) ||
        throw(ArgumentError("times must be finite and non-negative"))
    values = Vector{ComplexF64}(undef, length(times))
    for (index, time) in pairs(times)
        values[index] = sum(weights .* exp.(eigenvalues .* Float64(time)))
    end
    return values
end

function exact_cgs_spectral_data(L::AbstractMatrix,
                                 rho::AbstractMatrix,
                                 times::AbstractVector)
    size(L, 1) == size(L, 2) ||
        throw(DimensionMismatch("Liouvillian must be square"))
    size(rho, 1) == size(rho, 2) ||
        throw(DimensionMismatch("rho must be square"))
    d = size(rho, 1)
    size(L) == (d^2, d^2) || throw(DimensionMismatch(
        "Liouvillian shape $(size(L)) is incompatible with rho shape $(size(rho))"))
    L_dense = Matrix{ComplexF64}(L)
    rho_dense = Matrix{ComplexF64}(rho)
    all(isfinite, L_dense) ||
        throw(ArgumentError("Liouvillian contains non-finite entries"))
    all(isfinite, rho_dense) ||
        throw(ArgumentError("rho contains non-finite entries"))

    factorization = eigen(L_dense)
    eigenvalues = ComplexF64.(factorization.values)
    vectors = Matrix{ComplexF64}(factorization.vectors)
    r0 = vec(rho_dense)
    vector_factorization = lu(vectors)
    right_coefficients = ComplexF64.(vector_factorization \ r0)
    left_overlaps = ComplexF64.(vec(r0' * vectors))
    weights = left_overlaps .* right_coefficients
    complex_curve = reconstruct_survival(eigenvalues, weights, times)
    real_curve = Float64.(real.(complex_curve))

    eigenpair_residual = _relative_residual(
        norm(L_dense * vectors - vectors * Diagonal(eigenvalues)),
        max(norm(L_dense) * norm(vectors), 1.0))
    identity_vector = vec(Matrix{ComplexF64}(I, d, d))
    trace_preservation_residual =
        norm(transpose(identity_vector) * L_dense) / max(1.0, norm(L_dense))
    diagnostics = Dict{String,Float64}(
        "right_eigenpair_residual" => eigenpair_residual,
        "eigenvector_condition_number_1" => Float64(cond(vectors, 1)),
        "trace_preservation_residual" => Float64(trace_preservation_residual),
        "f0_error" => Float64(abs(sum(weights) - 1.0)),
        "curve_max_imaginary" => Float64(maximum(abs, imag.(complex_curve))),
        "curve_min_real" => minimum(real_curve),
        "curve_max_real" => maximum(real_curve),
    )
    return ExactCGSSpectralData(eigenvalues, left_overlaps,
                                right_coefficients, weights,
                                complex_curve, real_curve, diagnostics)
end

function bootstrap_seed_mean(curves::AbstractMatrix;
                             n_boot::Integer,
                             rng::AbstractRNG)
    n_seeds, n_times = size(curves)
    n_seeds > 0 || throw(ArgumentError("curves must contain at least one seed"))
    n_times > 0 || throw(ArgumentError("curves must contain at least one time"))
    n_boot > 0 || throw(ArgumentError("n_boot must be positive"))
    all(isfinite, curves) ||
        throw(ArgumentError("curves contain non-finite values"))
    central = Float64.(vec(mean(curves; dims = 1)))
    samples = Matrix{Float64}(undef, Int(n_boot), n_times)
    for bootstrap_index in 1:Int(n_boot)
        seed_indices = rand(rng, 1:n_seeds, n_seeds)
        samples[bootstrap_index, :] .=
            vec(mean(view(curves, seed_indices, :); dims = 1))
    end
    lower = [quantile(view(samples, :, index), 0.16) for index in 1:n_times]
    upper = [quantile(view(samples, :, index), 0.84) for index in 1:n_times]
    return (mean = central, lower = lower, upper = upper)
end

function _payload_diagnostic(diagnostics::AbstractDict, key::AbstractString)
    haskey(diagnostics, key) ||
        throw(ArgumentError("diagnostics lack $key"))
    value = Float64(diagnostics[key])
    isfinite(value) || throw(ArgumentError("diagnostic $key is not finite"))
    return value
end

function _payload_threshold(thresholds::AbstractDict, key::AbstractString)
    haskey(thresholds, key) || throw(ArgumentError("thresholds lack $key"))
    value = Float64(thresholds[key])
    isfinite(value) && value >= 0 ||
        throw(ArgumentError("threshold $key must be finite and non-negative"))
    return value
end

function validate_seed_payload(payload::AbstractDict, thresholds::AbstractDict)
    for key in SEED_PAYLOAD_REQUIRED_KEYS
        haskey(payload, key) || throw(ArgumentError("seed payload lacks $key"))
    end
    Int(payload["identity/seed"]) > 0 ||
        throw(ArgumentError("seed identity must be positive"))

    n_orb = Int(payload["dimensions/n_orb"])
    filling = Int(payload["dimensions/filling"])
    d = Int(payload["dimensions/hilbert"])
    K = Int(payload["dimensions/liouvillian"])
    n_orb > 0 && 0 < filling <= n_orb ||
        throw(ArgumentError("invalid system dimensions"))
    d > 0 || throw(ArgumentError("Hilbert dimension must be positive"))
    K == d^2 || throw(DimensionMismatch(
        "Liouvillian dimension $K does not equal d^2=$(d^2)"))

    energies = Float64.(payload["cgs/energies"])
    eigenvectors = ComplexF64.(payload["cgs/eigenvectors"])
    pivots = Int.(payload["cgs/pivots"])
    corrections = ComplexF64.(payload["cgs/phase_corrections"])
    psi = ComplexF64.(payload["cgs/psi"])
    length(energies) == d || throw(DimensionMismatch("wrong H eigenvalue count"))
    size(eigenvectors) == (d, d) ||
        throw(DimensionMismatch("wrong H eigenvector shape"))
    length(pivots) == length(corrections) == length(psi) == d ||
        throw(DimensionMismatch("wrong CGS metadata length"))
    all(1 .<= pivots .<= d) || throw(ArgumentError("invalid phase pivot"))
    all(isfinite, energies) && all(isfinite, eigenvectors) &&
        all(isfinite, corrections) && all(isfinite, psi) ||
        throw(ArgumentError("CGS state contains non-finite values"))
    abs(real(dot(psi, psi)) - 1.0) <= 1e-10 ||
        throw(ArgumentError("CGS state is not normalized"))

    eigenvalues = ComplexF64.(payload["spectral/eigenvalues"])
    left_overlaps = ComplexF64.(payload["spectral/left_overlaps"])
    right_coefficients = ComplexF64.(payload["spectral/right_coefficients"])
    weights = ComplexF64.(payload["spectral/weights"])
    all(length(values) == K for values in
        (eigenvalues, left_overlaps, right_coefficients, weights)) ||
        throw(DimensionMismatch("Liouvillian spectral arrays must have length $K"))
    all(values -> all(isfinite, values),
        (eigenvalues, left_overlaps, right_coefficients, weights)) ||
        throw(ArgumentError("Liouvillian spectral data contain non-finite values"))

    times = Float64.(payload["survival/times"])
    complex_curve = ComplexF64.(payload["survival/complex_curve"])
    real_curve = Float64.(payload["survival/real_curve"])
    length(times) == length(complex_curve) == length(real_curve) ||
        throw(DimensionMismatch("survival arrays have different lengths"))
    !isempty(times) && times[1] == 0.0 && all(diff(times) .> 0) ||
        throw(ArgumentError("survival times must start at zero and increase"))
    all(isfinite, complex_curve) && all(isfinite, real_curve) ||
        throw(ArgumentError("survival curve contains non-finite values"))
    maximum(abs, real.(complex_curve) - real_curve) <= 1e-13 ||
        throw(ArgumentError("stored real curve does not match complex curve"))
    reconstructed = reconstruct_survival(eigenvalues, weights, times)
    reconstruction_residual = maximum(abs, reconstructed - complex_curve)
    reconstruction_residual <=
        _payload_threshold(thresholds, "reconstruction_residual_max") ||
        throw(ArgumentError("stored survival reconstruction failed"))
    steady_indices = Int.(payload["survival/steady_mode_indices"])
    steady_weights = ComplexF64.(payload["survival/steady_mode_weights"])
    length(steady_indices) == length(steady_weights) ||
        throw(DimensionMismatch("steady-mode arrays have different lengths"))
    all(1 .<= steady_indices .<= K) ||
        throw(ArgumentError("invalid steady-mode index"))
    steady_weights == weights[steady_indices] ||
        throw(ArgumentError("steady-mode weights do not match spectral weights"))

    Bool(payload["diagnostics/pass"]) ||
        throw(ArgumentError("seed payload diagnostics are marked failed"))
    diagnostics = payload["diagnostics/values"]
    diagnostics isa AbstractDict ||
        throw(ArgumentError("diagnostics/values must be a dictionary"))
    h_limit = _payload_threshold(thresholds, "h_residual_max")
    for key in ("hermiticity_residual", "eigenpair_residual",
                "orthogonality_residual")
        _payload_diagnostic(diagnostics, key) <= h_limit ||
            throw(ArgumentError("Hamiltonian diagnostic $key exceeds threshold"))
    end
    _payload_diagnostic(diagnostics, "right_eigenpair_residual") <=
        _payload_threshold(thresholds, "l_residual_max") ||
        throw(ArgumentError("Liouvillian eigenpair residual exceeds threshold"))
    _payload_diagnostic(diagnostics, "trace_preservation_residual") <=
        _payload_threshold(thresholds, "trace_preservation_residual_max") ||
        throw(ArgumentError("trace-preservation residual exceeds threshold"))
    _payload_diagnostic(diagnostics, "f0_error") <=
        _payload_threshold(thresholds, "f0_error_max") ||
        throw(ArgumentError("F(0) error exceeds threshold"))
    _payload_diagnostic(diagnostics, "curve_max_imaginary") <=
        _payload_threshold(thresholds, "curve_max_imaginary") ||
        throw(ArgumentError("survival imaginary residual exceeds threshold"))
    probability_tolerance =
        _payload_threshold(thresholds, "curve_probability_tolerance")
    _payload_diagnostic(diagnostics, "curve_min_real") >= -probability_tolerance ||
        throw(ArgumentError("survival probability falls below zero tolerance"))
    _payload_diagnostic(diagnostics, "curve_max_real") <= 1 + probability_tolerance ||
        throw(ArgumentError("survival probability exceeds one tolerance"))
    _payload_diagnostic(diagnostics, "stored_reconstruction_residual") <=
        _payload_threshold(thresholds, "reconstruction_residual_max") ||
        throw(ArgumentError("stored reconstruction diagnostic exceeds threshold"))
    _payload_diagnostic(diagnostics, "eigenvector_condition_number_1") > 0 ||
        throw(ArgumentError("eigenvector condition number must be positive"))
    return nothing
end

function read_seed_payload(path::AbstractString;
                           validate::Bool = true,
                           thresholds = nothing)
    input = abspath(path)
    isfile(input) || throw(ArgumentError("seed payload not found: $input"))
    payload = JLD2.jldopen(input, "r") do file
        haskey(file, "schema/keys") ||
            throw(ArgumentError("seed payload lacks schema/keys"))
        stored_keys = String.(file["schema/keys"])
        Dict{String,Any}(key => file[key] for key in stored_keys)
    end
    if validate
        thresholds === nothing &&
            throw(ArgumentError("thresholds are required to validate a seed payload"))
        validate_seed_payload(payload, thresholds)
    end
    return payload
end

function write_seed_payload(path::AbstractString,
                            payload::AbstractDict;
                            thresholds,
                            force::Bool = false)
    output = abspath(path)
    validate_seed_payload(payload, thresholds)
    if isfile(output)
        read_seed_payload(output; validate = true, thresholds = thresholds)
        if !force
            return (path = output, status = :skipped)
        end
    end
    mkpath(dirname(output))
    temporary = output * ".tmp.$(getpid()).$(get(ENV, "SLURM_ARRAY_TASK_ID", "local"))"
    try
        keys_to_store = sort!(String.(collect(keys(payload))))
        JLD2.jldopen(temporary, "w") do file
            file["schema/keys"] = keys_to_store
            for key in keys_to_store
                file[key] = payload[key]
            end
        end
        read_seed_payload(temporary; validate = true, thresholds = thresholds)
        mv(temporary, output; force = true)
    finally
        isfile(temporary) && rm(temporary; force = true)
    end
    return (path = output, status = :written)
end

end
