#!/usr/bin/env julia

using JLD2
using LinearAlgebra
using Printf
using SparseArrays

include(joinpath(@__DIR__, "_integrable_corner_physical.jl"))
include(joinpath(CORNER_LAMB_DIR, "..", "SYK_setup.jl"))
include(joinpath(CORNER_LAMB_DIR, "JumpFactorization.jl"))
include(joinpath(CORNER_LAMB_DIR, "PhysicalSYK.jl"))
include(joinpath(CORNER_LAMB_DIR, "LiouvillianIntegration.jl"))

function parse_output(argv)
    output = ""
    seed = 1
    i = 1
    while i <= length(argv)
        argv[i] == "--help" && (println("Usage: julia integrable_corner_validate_panel_a.jl --output FILE [--seed 1]"); exit(0))
        i == length(argv) && error("missing value for $(argv[i])")
        if argv[i] == "--output"; output = argv[i + 1]
        elseif argv[i] == "--seed"; seed = parse(Int, argv[i + 1])
        else; error("unknown argument: $(argv[i])")
        end
        i += 2
    end
    isempty(output) && error("--output is required")
    return abspath(output), seed
end

function analytic_sigmas(data, gamma_tot)
    d = length(data.f)
    center = corner_center(data.f, gamma_tot)
    values = Vector{Float64}(undef, d^2)
    index = 1
    @inbounds for m in 1:d, n in 1:d
        values[index] = abs(corner_eigenvalue(
            data.f[m], data.f[n], data.E[m], data.E[n], gamma_tot) - center)
        index += 1
    end
    sort!(values)
    return values, center
end

function main(argv = ARGS)
    output, seed = parse_output(argv)
    n_orb, filling = 10, 3
    t0 = time()
    g = build_physical_corner_g(n_orb, seed)
    data = fixed_filling_corner_data(g, filling; J = CORNER_J)
    basis = generate_vectors(n_orb, filling)
    F = sparse(ComplexF64.(dissipator_block_direct(n_orb, basis, vec(g))))
    tensor = build_physical_syk_tensor(g; J = CORNER_J)
    H = sparse(ComplexF64.(syk4_block_from_tensor(n_orb, basis, tensor)))
    h_spectrum_residual = maximum(abs.(
        sort(eigvals(Hermitian(Matrix(H)))) .- sort(data.E)))
    jump = sqrt(CORNER_GAMMA_TOT) .* F
    L = assemble_lindblad_liouvillian(H, [jump])
    D_b = size(L, 1)
    numeric_center = tr(L) / D_b
    analytic, center = analytic_sigmas(data, CORNER_GAMMA_TOT)
    center_residual = abs(numeric_center - center)

    Lc = Matrix{ComplexF64}(L)
    @inbounds for index in 1:D_b
        Lc[index, index] -= center
    end
    commutator = Lc * adjoint(Lc)
    mul!(commutator, adjoint(Lc), Lc, -1.0, 1.0)
    normality_residual = norm(commutator) / norm(Lc)^2
    commutator = nothing
    GC.gc()
    numeric = sort!(svdvals!(Lc))
    absolute_residuals = abs.(numeric .- analytic)
    max_absolute_residual = maximum(absolute_residuals)
    max_relative_residual = max_absolute_residual / maximum(analytic)
    passed = normality_residual <= 1e-11 &&
             max_relative_residual <= 1e-10 &&
             center_residual <= 1e-10 && h_spectrum_residual <= 1e-10

    mkpath(dirname(output))
    temporary = output * ".tmp.$(getpid())"
    JLD2.jldopen(temporary, "w") do file
        file["complete"] = true
        file["passed"] = passed
        file["n_orb"] = n_orb
        file["filling"] = filling
        file["seed"] = seed
        file["d"] = length(data.f)
        file["D_b"] = D_b
        file["analytic_sigmas"] = analytic
        file["numeric_sigmas"] = numeric
        file["absolute_residuals"] = absolute_residuals
        file["center_analytic"] = center
        file["center_numeric_re"] = real(numeric_center)
        file["center_numeric_im"] = imag(numeric_center)
        file["center_residual"] = center_residual
        file["normality_residual"] = normality_residual
        file["max_absolute_residual"] = max_absolute_residual
        file["max_relative_residual"] = max_relative_residual
        file["hamiltonian_spectrum_residual"] = h_spectrum_residual
        file["normality_threshold"] = 1e-11
        file["relative_sigma_threshold"] = 1e-10
        file["gamma_tot_over_J"] = CORNER_GAMMA_TOT
        file["elapsed_seconds"] = time() - t0
    end
    mv(temporary, output; force = true)
    @printf("panel a: passed=%s normality=%.3e sigma_rel=%.3e center=%.3e H=%.3e\n",
            passed, normality_residual, max_relative_residual,
            center_residual, h_spectrum_residual)
    passed || error("panel-a validation failed; see $output")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
