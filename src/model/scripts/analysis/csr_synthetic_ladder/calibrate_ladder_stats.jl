#!/usr/bin/env julia
# calibrate_ladder_stats.jl
#
# One-shot CSR-synthetic-ladder calibration script.  Reads the reference
# production config (N=8, eta=2, cavity loss on), rebuilds the physical
# Coefficients + Liouvillian for seeds 1..64, and writes the ensemble
# averages
#
#   mu_w, sigma_w2, g_frobenius2_mean, K_pair_trace_mean, H_frobenius_mean
#
# plus derived synthetic-recipe scales (sigma_g, sigma_M, lambda_star,
# sigma_syk4_bdi_span) to a single JLD2 file consumed by the retained CSR-ladder sweep
# configs.
#
# Run from the repository root:
#
#     julia --project=src/environment \
#         src/model/scripts/analysis/csr_synthetic_ladder/calibrate_ladder_stats.jl \
#         [--n-seeds 64] [--output <path>]

if Sys.isapple() && get(ENV, "LAMBDICKE_NO_APPLE_ACCELERATE", "0") != "1"
    @eval using AppleAccelerate
end

using JLD2
using LinearAlgebra: Hermitian, eigvals, norm, opnorm, tr
using Printf
using Random: MersenneTwister
using Statistics: mean, var
using TOML

const HERE       = @__DIR__
const REPO_ROOT  = abspath(joinpath(HERE, "..", "..", "..", "..", ".."))
const JULIA_DIR  = joinpath(REPO_ROOT, "julia")
const MODULE_DIR = joinpath(JULIA_DIR, "model")

include(joinpath(JULIA_DIR, "SYK_setup.jl"))
include(joinpath(MODULE_DIR, "LambDickeModel.jl"))
include(joinpath(MODULE_DIR, "HarmonicOscillatorGrid.jl"))
include(joinpath(MODULE_DIR, "Speckle.jl"))
include(joinpath(MODULE_DIR, "OverlapIntegrals.jl"))
include(joinpath(MODULE_DIR, "JumpFactorization.jl"))
include(joinpath(MODULE_DIR, "PhysicalSYK.jl"))
include(joinpath(MODULE_DIR, "LiouvillianIntegration.jl"))
include(joinpath(MODULE_DIR, "SyntheticLadder.jl"))

# ============================================================================
# CLI parsing
# ============================================================================

function parse_args(argv)
    n_seeds = 64
    output = nothing
    config_path = joinpath(MODULE_DIR, "configs", "physical_n8.toml")
    eta = 2.0
    i = 1
    while i <= length(argv)
        a = argv[i]
        if a == "--n-seeds"
            i += 1; n_seeds = parse(Int, argv[i])
        elseif startswith(a, "--n-seeds=")
            n_seeds = parse(Int, split(a, "=", limit=2)[2])
        elseif a == "--output"
            i += 1; output = argv[i]
        elseif startswith(a, "--output=")
            output = split(a, "=", limit=2)[2]
        elseif a == "--config"
            i += 1; config_path = argv[i]
        elseif startswith(a, "--config=")
            config_path = split(a, "=", limit=2)[2]
        elseif a == "--eta"
            i += 1; eta = parse(Float64, argv[i])
        elseif startswith(a, "--eta=")
            eta = parse(Float64, split(a, "=", limit=2)[2])
        else
            error("unrecognized argument: $a")
        end
        i += 1
    end
    return (n_seeds = n_seeds, output = output, config_path = abspath(config_path), eta = eta)
end

# ============================================================================
# Physical LambDickeLiouvillianConfig from reference TOML
# ============================================================================

function physical_config_from_toml(cfg::AbstractDict; gamma_eff::Float64 = 1.0)
    num = cfg["numerics"]
    # `filling` is optional; when absent, LambDickeLiouvillianConfig falls back
    filling = haskey(num, "filling") ? Int(num["filling"]) : nothing
    return LambDickeLiouvillianConfig(
        n_orb = Int(num["n_orb"]),
        filling = filling,
        n_grid = Int(num["n_grid"]),
        box_length = Float64(num["box_length"]),
        weight_type = :speckle,
        correlation_length = Float64(num["correlation_length"]),
        disorder_strength = Float64(num["disorder_strength"]),
        gamma_eff = gamma_eff,
        svd_tol = Float64(get(num, "baseline_svd_tol", 1e-10)),
        J = Float64(num["J"]),
        lambda_c_micron = Float64(get(num, "lambda_c_micron", 0.671)),
        kappa_over_2pi_mhz = Float64(get(num, "kappa_over_2pi_mhz", 0.2)),
        delta_cd_over_2pi_mhz = gamma_eff,
    )
end

# ============================================================================
# Per-seed calibration payload
# ============================================================================

struct SeedCalibration
    seed::Int
    w_mean::Float64
    w_var::Float64
    g_frobenius2::Float64
    K_pair_trace::Float64
    H_frobenius::Float64
    H_op_norm::Float64
    H_spectral_span::Float64
end

function calibrate_seed(config::LambDickeLiouvillianConfig, seed::Int, eta::Float64)
    # Build the production coefficients + kernel + physical H at this seed.
    result = build_lamb_dicke_liouvillian(config, seed, eta)
    coeff  = result.coefficients

    # Rebuild w to compute grid mean/variance (fast; already cached, but
    # coeff exposes only the K, X, T aggregates).
    p3 = SingleParticleParams(
        grid = coeff.grid,
        orbitals = coeff.orbitals,
        weight = coeff.weight,
    )
    w = build_weight(p3.grid, p3.weight)

    K_pair = kernel_pair_matrix(eta, p3, coeff)
    Hm = Matrix(result.H)
    # Spectral span of H (max eigenvalue minus min eigenvalue).  This is the
    # quantity that sets the Im-axis span of the vectorized Liouvillian cloud
    # (Im lambda_L = E_mu - E_nu ranges over the eigenvalue-difference set),
    # and hence what the CSR aspect ratio depends on.  Op-norm alone is
    # dominated by the identity-piece offset (mu_w^2 * N) for the physical
    # low-rank quartic H, so it OVER-estimates the coherent scale that L4's
    # zero-mean real SYK4 Hamiltonian needs to hit.
    H_evals = eigvals(Hermitian(0.5 .* (Hm .+ Hm')))
    span = Float64(maximum(H_evals) - minimum(H_evals))
    return SeedCalibration(
        seed,
        Float64(mean(w)),
        Float64(var(w; corrected = false)),
        Float64(norm(coeff.g_jk)^2),
        Float64(tr(K_pair)),
        Float64(norm(Hm)),
        Float64(opnorm(Hm)),
        span,
    )
end

"""
    calibrate_sigma_syk4_bdi_span(config, target_H_span; n_seeds = 8) -> Float64

Solve for `sigma_syk4` for the BDI† ladder level so that
`E[span(H_SYK4_real)] = target_H_span`, where `H_SYK4_real` uses a fully
real SYK4 tensor via `random_real_syk4_tensor`. Same linear scaling in
`sigma_syk4` in the coupling amplitude; empirically the
`unit_mean` differs by an O(1) factor from the complex case because
constraining couplings to be real removes the imaginary-part fluctuation of
each off-diagonal combination, halving that entry's second moment.
"""
function calibrate_sigma_syk4_bdi_span(config::LambDickeLiouvillianConfig,
                                       target_H_span::Float64;
                                       n_seeds::Integer = 8)
    n_orb = config.n_orb
    basis_states = generate_vectors(n_orb, config.filling)
    spans = Float64[]
    for s in 1:n_seeds
        rng = MersenneTwister(90_200 + s)
        T = random_real_syk4_tensor(rng, n_orb, 1.0)
        H = Matrix(syk4_block_from_tensor(n_orb, basis_states, T))
        evals = eigvals(Hermitian(0.5 .* (H .+ H')))
        push!(spans, maximum(evals) - minimum(evals))
    end
    unit_mean = mean(spans)
    return unit_mean > 0 ? target_H_span / unit_mean : 0.0
end

# ============================================================================
# Main
# ============================================================================

function main()
    opts = parse_args(ARGS)
    cfg = TOML.parsefile(opts.config_path)
    physcfg = physical_config_from_toml(cfg)

    default_out = joinpath(
        REPO_ROOT, "data", "calibration",
        @sprintf("calibration_n%d_f%d_eta%s.jld2", physcfg.n_orb, physcfg.filling,
                 replace(@sprintf("%.6g", opts.eta), "." => "p")))
    out_path = opts.output === nothing ? default_out : opts.output
    mkpath(dirname(out_path))

    println("[calibration] config=$(opts.config_path)")
    println("[calibration] eta=$(opts.eta), n_orb=$(physcfg.n_orb), filling=$(physcfg.filling), n_seeds=$(opts.n_seeds)")
    println("[calibration] output=$out_path")

    payloads = Vector{SeedCalibration}(undef, opts.n_seeds)
    t0 = time_ns()
    for s in 1:opts.n_seeds
        payloads[s] = calibrate_seed(physcfg, s, opts.eta)
        wall = (time_ns() - t0) / 1e9
        @printf("[calibration] seed=%3d  wall=%6.1fs  <w>=%.4f  Var[w]=%.4f  ||g||_F^2=%.4f  tr(K)=%.4f  ||H||_F=%.4f  ||H||_op=%.4f  span(H)=%.4f\n",
                s, wall, payloads[s].w_mean, payloads[s].w_var,
                payloads[s].g_frobenius2, payloads[s].K_pair_trace,
                payloads[s].H_frobenius, payloads[s].H_op_norm,
                payloads[s].H_spectral_span)
        flush(stdout)
    end

    mu_w    = mean(p.w_mean for p in payloads)
    sigma_w2 = mean(p.w_var  for p in payloads)
    g_frobenius2_mean = mean(p.g_frobenius2 for p in payloads)
    K_pair_trace_mean = mean(p.K_pair_trace for p in payloads)
    H_frobenius_mean  = mean(p.H_frobenius  for p in payloads)
    H_op_mean         = mean(p.H_op_norm    for p in payloads)
    H_span_mean       = mean(p.H_spectral_span for p in payloads)

    sigma_g = ladder_sigma_g(g_frobenius2_mean, physcfg.n_orb)
    lambda_star = ladder_lambda_star(K_pair_trace_mean, 60)
    sigma_syk4_bdi_span = calibrate_sigma_syk4_bdi_span(physcfg, H_span_mean; n_seeds = 32)
    println()
    println("=== CALIBRATION SUMMARY ===")
    @printf("mu_w                = %.6f\n", mu_w)
    @printf("sigma_w2 = Var[w]   = %.6f\n", sigma_w2)
    @printf("<||g||_F^2>         = %.6f\n", g_frobenius2_mean)
    @printf("<tr(K_pair(eta=%.3g))> = %.6f\n", opts.eta, K_pair_trace_mean)
    @printf("<||H||_F>           = %.6f\n", H_frobenius_mean)
    @printf("<||H||_op>          = %.6f\n", H_op_mean)
    @printf("<span(H)>           = %.6f  (physical Im lambda_L span target)\n", H_span_mean)
    println()
    @printf("Derived dissipator scales:\n")
    @printf("  sigma_g          = %.6f  (cavity-op norm, matches <||g||_F^2>)\n", sigma_g)
    @printf("  lambda_star (60) = %.6f  (L3b equal-weight jumps)\n", lambda_star)
    println()
    @printf("Derived synthetic scales (span-matched; ACTIVE):\n")
    @printf("  sigma_syk4_bdi_span    = %.6f  (L3b random_real_syk4_tensor; BDI† class; matches <span(H)>)\n", sigma_syk4_bdi_span)

    JLD2.jldopen(out_path, "w") do f
        f["mu_w"]              = mu_w
        f["sigma_w2"]          = sigma_w2
        f["g_frobenius2_mean"] = g_frobenius2_mean
        f["K_pair_trace_mean"] = K_pair_trace_mean
        f["H_frobenius_mean"]  = H_frobenius_mean
        f["H_op_mean"]         = H_op_mean
        f["H_span_mean"]       = H_span_mean
        f["sigma_g"]           = sigma_g
        f["lambda_star_60"]    = lambda_star
        f["sigma_syk4_bdi_span"] = sigma_syk4_bdi_span
        f["n_orb"]             = physcfg.n_orb
        f["filling"]           = physcfg.filling
        f["n_grid"]            = physcfg.n_grid
        f["box_length"]        = physcfg.box_length
        f["eta"]               = opts.eta
        f["n_seeds"]           = opts.n_seeds
        f["per_seed"] = Dict(
            "seed"            => [p.seed for p in payloads],
            "w_mean"          => [p.w_mean for p in payloads],
            "w_var"           => [p.w_var for p in payloads],
            "g_frobenius2"    => [p.g_frobenius2 for p in payloads],
            "K_pair_trace"    => [p.K_pair_trace for p in payloads],
            "H_frobenius"     => [p.H_frobenius for p in payloads],
            "H_op_norm"       => [p.H_op_norm for p in payloads],
            "H_spectral_span" => [p.H_spectral_span for p in payloads],
        )
    end
    println("\n[calibration] wrote $out_path")
    return 0
end

exit(main())
