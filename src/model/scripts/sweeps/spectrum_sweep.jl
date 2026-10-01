# Compute the spectra, centered singular values, and dynamics consumed by the figures.

if Sys.isapple()
    @eval using AppleAccelerate
end
using JLD2
using LinearAlgebra: BLAS, eigen, eigvals, kron, norm, svd, svdvals, tr,
                     LAPACKException, I, QRIteration
using Printf
using Random: MersenneTwister
using Statistics: mean, median
using TOML

const HERE = @__DIR__
const JULIA_DIR = abspath(joinpath(HERE, "..", "..", ".."))
const REPO_ROOT = abspath(joinpath(JULIA_DIR, ".."))
const MODULE_DIR = abspath(joinpath(HERE, "..", ".."))
const DATA_ROOT = abspath(joinpath(JULIA_DIR, "..", "data"))

include(joinpath(JULIA_DIR, "SYK_setup.jl"))
include(joinpath(MODULE_DIR, "LambDickeModel.jl"))
include(joinpath(MODULE_DIR, "HarmonicOscillatorGrid.jl"))
include(joinpath(MODULE_DIR, "Speckle.jl"))
include(joinpath(MODULE_DIR, "OverlapIntegrals.jl"))
include(joinpath(MODULE_DIR, "JumpFactorization.jl"))
include(joinpath(MODULE_DIR, "PhysicalSYK.jl"))
include(joinpath(MODULE_DIR, "LiouvillianIntegration.jl"))
include(joinpath(MODULE_DIR, "SyntheticLadder.jl"))
include(joinpath(MODULE_DIR, "Validation.jl"))
include(joinpath(MODULE_DIR, "Dynamics.jl"))
include(joinpath(HERE, "sweep_scheduler.jl"))

const CSR_LADDER_CALIBRATION_LOCK = ReentrantLock()
const CSR_LADDER_CALIBRATION_CACHE = Ref{Union{Nothing,Dict{String,Any}}}(nothing)
const CSR_LADDER_CALIBRATION_PATH_CACHE = Ref{String}("")

"""
    load_csr_ladder_calibration(path) -> Dict{String,Any}

Load and cache the JLD2 calibration payload produced by
`calibrate_ladder_stats.jl`. Thread-safe.
"""
function load_csr_ladder_calibration(path::AbstractString)
    p = String(path)
    lock(CSR_LADDER_CALIBRATION_LOCK) do
        if CSR_LADDER_CALIBRATION_CACHE[] !== nothing &&
           CSR_LADDER_CALIBRATION_PATH_CACHE[] == p
            return CSR_LADDER_CALIBRATION_CACHE[]
        end
        isfile(p) || error("CSR ladder calibration file not found: $p")
        payload = JLD2.jldopen(p, "r") do f
            d = Dict{String,Any}()
            for k in keys(f)
                d[k] = f[k]
            end
            d
        end
        CSR_LADDER_CALIBRATION_CACHE[] = payload
        CSR_LADDER_CALIBRATION_PATH_CACHE[] = p
        return payload
    end
end

# ----------------------------------------------------------------------------
# Config loading + validation
# ----------------------------------------------------------------------------

function parse_cli_args(argv)
    config_path = joinpath(MODULE_DIR, "configs", "physical_smoke_n4.toml")
    shard_index = nothing
    shard_count = nothing
    data_subdir_override = nothing
    i = 1
    while i <= length(argv)
        a = argv[i]
        if a == "--config"
            i += 1
            i <= length(argv) || error("--config requires a path argument")
            config_path = argv[i]
        elseif startswith(a, "--config=")
            config_path = split(a, "=", limit = 2)[2]
        elseif a == "--shard-index"
            i += 1
            i <= length(argv) || error("--shard-index requires an integer argument")
            shard_index = parse(Int, argv[i])
        elseif startswith(a, "--shard-index=")
            shard_index = parse(Int, split(a, "=", limit = 2)[2])
        elseif a == "--shard-count"
            i += 1
            i <= length(argv) || error("--shard-count requires an integer argument")
            shard_count = parse(Int, argv[i])
        elseif startswith(a, "--shard-count=")
            shard_count = parse(Int, split(a, "=", limit = 2)[2])
        elseif a == "--data-subdir"
            i += 1
            i <= length(argv) || error("--data-subdir requires a path argument")
            data_subdir_override = argv[i]
        elseif startswith(a, "--data-subdir=")
            data_subdir_override = split(a, "=", limit = 2)[2]
        else
            error("unrecognized argument: $a")
        end
        i += 1
    end
    (isnothing(shard_index) == isnothing(shard_count)) ||
        error("--shard-index and --shard-count must be provided together")
    return (config_path = abspath(config_path),
            shard_index = shard_index,
            shard_count = shard_count,
            data_subdir_override = data_subdir_override)
end

function load_sweep_config(path::AbstractString)
    isfile(path) || error("config file not found: $path")
    return TOML.parsefile(path)
end

# ----------------------------------------------------------------------------
# Per-point Liouvillian construction
# ----------------------------------------------------------------------------

function make_liouvillian_config(p::SweepPoint, num::AbstractDict, seeds::AbstractDict)
    n_orb = Int(num["n_orb"])
    # Optional filling override.  When absent from the TOML, LambDickeLiouvillianConfig
    # into a custom sector must set `numerics.filling` to a value satisfying
    # 0 < filling < n_orb (see LambDickeLiouvillianConfig for validation).
    filling = haskey(num, "filling") ? Int(num["filling"]) : nothing
    # `force_gamma_eff_over_J` decouples the sweep parameter (`p.gamma`, kept as
    # `delta_cd_over_2pi_mhz` for the cavity-loss rate and file naming) from the
    # spontaneous-emission rate `gamma_eff`. When present in `[numerics]`, the
    # value overrides `p.gamma` in every open-system config branch.  Introduced
    forced_gamma_eff = haskey(num, "force_gamma_eff_over_J") ?
        Float64(num["force_gamma_eff_over_J"]) : nothing
    resolved_gamma_eff = forced_gamma_eff === nothing ? p.gamma : forced_gamma_eff
    common = (n_grid = Int(num["n_grid"]),
              box_length = Float64(num["box_length"]),
              correlation_length = Float64(num["correlation_length"]),
              disorder_strength = Float64(num["disorder_strength"]),
              J = Float64(num["J"]),
              svd_tol = p.tol,
              lambda_c_micron = Float64(get(num, "lambda_c_micron", 0.671)),
              kappa_over_2pi_mhz = Float64(get(num, "kappa_over_2pi_mhz", 0.2)),
              # `delta_cd_over_2pi_mhz` is a detuning in MHz, while `p.gamma`
              # is the dimensionless sweep parameter; identifying the two is
              # only meaningful when `force_gamma_eff_over_J` has repurposed
              # the `gamma` axis as the detuning axis.  Otherwise fall back to
              # the canonical 1.0 MHz so that a multi-point `gammas` grid does
              # not silently rescale the cavity-loss rate `kappa / delta`.
              delta_cd_over_2pi_mhz =
                  Float64(get(num, "delta_cd_over_2pi_mhz",
                              forced_gamma_eff === nothing ? 1.0 : p.gamma)),
              manual_cavity_loss_rate_over_J =
                  haskey(num, "cavity_loss_rate_over_J") ?
                      Float64(num["cavity_loss_rate_over_J"]) : nothing)

    if p.config == "physical"
        return LambDickeLiouvillianConfig(; n_orb = n_orb, filling = filling, weight_type = :speckle,
            gamma_eff = resolved_gamma_eff,
            common...)
    elseif p.config == "synthetic_l3b_bdi_syk4"
        # L3b: real SYK4 H (BDI† symmetry class) + 60 real-symmetric equal-
        # weight jumps + real-symmetric cavity op.  All couplings are real
        # Gaussian, so H is real symmetric in the number basis and each L_μ
        # is real symmetric.  The vectorized Liouvillian satisfies L^T = L
        # (complex symmetric), placing the ensemble in class BDI† of the
        # Sá–Ribeiro–Prosen classification (arXiv:2307.08218).  Uses
        # `sigma_syk4_bdi_span` from the span calibration, matched to
        # `<span(H_phys)>` at unit variance.
        calib = load_csr_ladder_calibration(String(num["calibration_jld2"]))
        offset = Int(seeds["l3b_offset"])
        n_orb_calib = Int(calib["n_orb"])
        n_orb == n_orb_calib || error("L3b: config n_orb=$n_orb ≠ calibration n_orb=$n_orb_calib")
        f_calib = Int(calib["filling"])
        f_config = filling === nothing ? div(n_orb, 2) : filling
        f_calib == f_config || error("L3b: config filling=$f_config ≠ calibration filling=$f_calib")
        n_jumps     = 60
        sigma_g     = Float64(calib["sigma_g"])
        lambda_s    = Float64(calib["lambda_star_60"])
        sigma_syk4  = Float64(calib["sigma_syk4_bdi_span"])
        rng_H   = MersenneTwister(offset + p.seed)
        rng_jmp = MersenneTwister(offset + p.seed + 1_000_000)
        rng_cav = MersenneTwister(offset + p.seed + 2_000_000)
        H_tensor  = random_real_syk4_tensor(rng_H, n_orb, sigma_syk4)
        jumps     = random_symmetric_jumps_equal_weight(rng_jmp, n_orb, n_jumps, sqrt(lambda_s))
        cavity_op = random_symmetric_cavity_op(rng_cav, n_orb, sigma_g)
        return LambDickeLiouvillianConfig(; n_orb = n_orb, filling = filling, weight_type = :uniform,
            gamma_eff = resolved_gamma_eff,
            synthetic_hamiltonian_tensor = H_tensor,
            synthetic_jump_matrices = jumps,
            synthetic_cavity_loss_operator = cavity_op,
            common...)
    elseif p.config in ("synthetic_l3b_bdi_syk4_fig3_m300", "synthetic_l3b_bdi_syk4_fig3_m10")
        expected_n_jumps = p.config == "synthetic_l3b_bdi_syk4_fig3_m300" ? 300 : 10
        calib = load_csr_ladder_calibration(String(num["calibration_jld2"]))
        Float64(calib["target_delta_tilde"]) == 0.01 ||
            error("Figure 3 L3b: target_delta_tilde must be 0.01")
        Int(calib["n_orb"]) == n_orb || error("Figure 3 L3b: n_orb mismatch")
        Int(calib["filling"]) == filling || error("Figure 3 L3b: filling mismatch")
        Int(calib["n_random_jumps"]) == expected_n_jumps ||
            error("Figure 3 L3b: expected $expected_n_jumps random jumps")
        Int(calib["n_cavity_jumps"]) == 1 ||
            error("Figure 3 L3b: expected one cavity jump")
        scale = Float64(calib["figure3_dissipator_rate_scale"])
        isfinite(scale) && scale > 0 ||
            error("Figure 3 L3b: invalid dissipator rate scale")

        offset = Int(seeds["l3b_offset"])
        n_jumps = expected_n_jumps
        rng_H = MersenneTwister(offset + p.seed)
        rng_jmp = MersenneTwister(offset + p.seed + 1_000_000)
        rng_cav = MersenneTwister(offset + p.seed + 2_000_000)
        H_tensor = random_real_syk4_tensor(
            rng_H, n_orb, Float64(calib["sigma_syk4_bdi_span"]))
        jumps = random_symmetric_jumps_equal_weight(
            rng_jmp, n_orb, n_jumps,
            sqrt(Float64(calib["lambda_star_60"])))
        cavity_op = random_symmetric_cavity_op(
            rng_cav, n_orb, Float64(calib["sigma_g"]))
        scaled_common = merge(common, (

            manual_cavity_loss_rate_over_J = 0.2 * scale,
        ))
        return LambDickeLiouvillianConfig(;
            n_orb = n_orb, filling = filling, weight_type = :uniform,
            gamma_eff = resolved_gamma_eff * scale,
            synthetic_hamiltonian_tensor = H_tensor,
            synthetic_jump_matrices = jumps,
            synthetic_cavity_loss_operator = cavity_op,
            scaled_common...)
    else
        error("unknown config: $(p.config)")
    end
end

struct DiagSettings
    times::Vector{Float64}
end

struct MergedDynamicsSettings
    entropy_eig_tol::Float64
end

merged_dynamics_settings(cfg::AbstractDict) = MergedDynamicsSettings(
    Float64(get(get(cfg, "dynamics", Dict()), "entropy_eig_tol", 1e-12)))

# Robust dense eigendecomposition of a Liouvillian.  LAPACK's non-symmetric
# solver `geev` occasionally fails to converge (LAPACKException(1)) on borderline
# near-defective L (observed on ~1 in 1e4 physical production points, e.g.
# eta=0, Gamma_eff/J=10).  We retry with a tiny diagonal jitter
# (jitter = 1e-12 * ||L||, far below every physics tolerance: svd_tol 1e-10,
# distance_tol 1e-12, real_tol 1e-10).  The perturbation shifts eigenvalues by
# at most O(jitter) and leaves the DSFF/CSR/gap diagnostics and the dense
# dynamics S(t)/n_j(t) unchanged at numerical resolution, but reliably lets geev
# converge.  On a second failure the error propagates unchanged.
_eigen_jitter(A) = 1e-12 * norm(A)

function robust_eigen(A::AbstractMatrix; diagnostics = nothing)
    diagnostics === nothing || (diagnostics["diagonal_jitter"] = 0.0)
    try
        return eigen(A)
    catch err
        err isa LAPACKException || rethrow()
        jit = _eigen_jitter(A)
        diagnostics === nothing || (diagnostics["diagonal_jitter"] = jit)
        return eigen(A + jit * Matrix{eltype(A)}(I, size(A, 1), size(A, 2)))
    end
end

# Robust singular values.  Julia's `svdvals(A)` dispatches to LAPACK's
# divide-and-conquer `gesdd!('N', A)` (JOBZ='N': singular VALUES only, no
# left/right singular vectors) -- roughly 2x cheaper than `svd(A).S`, which
# goes through `gesdd!('S', A)` and pays for thin U and V^H before
# discarding them.  `gesdd` occasionally fails to converge
# (LAPACKException(1)) on borderline near-defective Liouvillians (observed
# at eta=0, Gamma_eff/J=10); retry with the classic QR-iteration path
# (`gesvd`, slower but more robust).  There is no `svdvals(A; alg=...)`
# kwarg, so the fallback goes through `svd(A; alg=QRIteration()).S` --
# vectors are recomputed there, but that path fires on ~1 in 1e4 seeds and
# the extra cost is well below the wall-time budget of a full point.
function robust_svdvals(A::AbstractMatrix)
    try
        return svdvals(A)
    catch err
        err isa LAPACKException || rethrow()
        return svd(A; alg = QRIteration()).S
    end
end

function compute_point(p::SweepPoint, result, ds::DiagSettings,
                       dyn_settings::MergedDynamicsSettings)
    L = Matrix(result.L)
    F = robust_eigen(L)
    mu = tr(L) / size(L, 1)
    centered = copy(L)
    for i in axes(centered, 1)
        centered[i, i] -= mu
    end
    H = Matrix(result.H)
    identity = Matrix{ComplexF64}(I, size(H)...)
    LH = -1im .* (kron(identity, H) .- kron(transpose(H), identity))
    h_norm, d_norm = norm(LH), norm(L .- LH)
    energies = real.(eigvals(H))
    dyn = dynamics_from_eigen(F.values, F.vectors, result.basis_states, ds.times;
        n_orb = Int(result.metadata["n_orb"]),
        entropy_eig_tol = dyn_settings.entropy_eig_tol)
    row = Dict{String,Any}("config" => p.config, "eta" => p.eta,
        "gamma" => p.gamma, "svd_tol" => p.tol, "seed" => p.seed)
    payload = Dict{String,Any}(
        "L_eigvals" => ComplexF64.(F.values),
        "L_svd_S_centered" => robust_svdvals(centered),
        "L_trace_shift_mu_re" => real(mu), "L_trace_shift_mu_im" => imag(mu),
        "r_HD_frobenius" => d_norm > 0 ? h_norm / d_norm : Inf,
        "H_spectral_span" => maximum(energies) - minimum(energies),
        "L_H_frobenius" => h_norm, "L_D_frobenius" => d_norm,
        "metadata" => result.metadata,
        "times" => dyn.times, "entropy_t" => dyn.entropy_t,
        "populations_t" => dyn.populations_t,
        "dynamics_trace_residual_max" => dyn.metadata["trace_residual_max"],
        "dynamics_particle_residual_max" => dyn.metadata["particle_residual_max"],
    )
    return row, payload
end

# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

function execution_settings(cfg::AbstractDict)
    exec = get(cfg, "execution", Dict{String,Any}())
    parallel_tasks = max(1, Int(get(exec, "parallel_tasks", 1)))
    blas_threads = max(1, Int(get(exec, "blas_threads",
                                  parallel_tasks > 1 ? 1 : BLAS.get_num_threads())))
    return (parallel_tasks = parallel_tasks,
            blas_threads = blas_threads)
end

function compute_and_save_point(p::SweepPoint, num::AbstractDict, seeds::AbstractDict,
                                ds::DiagSettings, dyn_settings::MergedDynamicsSettings,
                                spectra_dir::AbstractString)
    t0 = time_ns()
    mem0 = Base.gc_live_bytes()
    local row, payload
    status = "ok"
    failure_reason = ""
    try
        ldcfg = make_liouvillian_config(p, num, seeds)
        result = build_lamb_dicke_liouvillian(ldcfg, p.seed, p.eta)
        row, payload = compute_point(p, result, ds, dyn_settings)
    catch err
        status = "failed"
        failure_reason = sprint(showerror, err)
        row = Dict{String,Any}(
            "config" => p.config, "eta" => p.eta, "gamma" => p.gamma,
            "svd_tol" => p.tol, "seed" => p.seed)
        payload = nothing
    end

    wall_seconds = (time_ns() - t0) / 1e9
    mem_mb_delta = (Base.gc_live_bytes() - mem0) / 1e6
    row["wall_seconds"] = wall_seconds
    row["mem_mb_delta"] = mem_mb_delta
    row["status"] = status
    row["failure_reason"] = failure_reason

    if status == "ok"
        jld_path = joinpath(spectra_dir, spectrum_filename(p))
        try
            JLD2.jldopen(jld_path, "w") do f
                for (k, v) in payload
                    f[k] = v
                end
            end
        catch err
            row["status"] = "failed"
            row["failure_reason"] = "jld2 save failed: " * sprint(showerror, err)
        end
    end

    return row

end

function print_progress_line(done::Int, total::Int, p::SweepPoint, row::AbstractDict,
                             wall_seconds::Float64, eta_minutes::Float64)
    @printf("[%4d/%4d] %-24s eta=%-5s gamma=%-4s tol=%-8s seed=%-2d  %s  %.2fs  (ETA %.1f min)\n",
            done, total, p.config, fmt_grid_value(p.eta),
            fmt_grid_value(p.gamma), fmt_grid_value(p.tol), p.seed,
            row["status"] == "ok" ? "ok " : "FAIL", wall_seconds, eta_minutes)
    row["status"] == "ok" || println("        failure: $(row["failure_reason"])")
    return nothing
end

function run_sweep(config_path::AbstractString;
                          shard_index::Union{Nothing,Int} = nothing,
                          shard_count::Union{Nothing,Int} = nothing,
                          data_subdir_override::Union{Nothing,String} = nothing)
    cfg = load_sweep_config(config_path)

    num = cfg["numerics"]
    seeds = get(cfg, "seeds", Dict{String,Any}())
    tgcfg = num["time_grid"]
    outcfg = cfg["output"]
    exec = execution_settings(cfg)
    BLAS.set_num_threads(exec.blas_threads)

    data_subdir = data_subdir_override === nothing ?
        String(outcfg["data_subdir"]) : String(data_subdir_override)
    shard = isnothing(shard_index) ? nothing : validate_shard_spec(shard_index, shard_count)

    out_dir = abspath(joinpath(DATA_ROOT, data_subdir))
    spectra_dir = joinpath(out_dir, "spectra")
    metrics_csv = joinpath(out_dir, "metrics.csv")
    dyn_settings = merged_dynamics_settings(cfg)
    mkpath(spectra_dir)

    times = log_time_grid(Float64(tgcfg["t_min"]), Float64(tgcfg["t_max"]), Int(tgcfg["n"]))
    ds = DiagSettings(times)

    spec = grid_spec_from_config(cfg)
    full_grid = build_sweep_grid(spec)
    grid = shard === nothing ? full_grid : shard_points(full_grid, shard.index, shard.count)
    done = read_done_keys(metrics_csv, spectra_dir)
    pending = SweepPoint[p for p in grid if !(point_key(p) in done)]
    n_total = length(grid)
    n_global = length(full_grid)
    n_done = n_total - length(pending)

    isfile(metrics_csv) || write_metrics_header(metrics_csv)

    println("=" ^ 78)
    println("Spectrum sweep driver")
    println("config: $config_path")
    if shard === nothing
        println("grid points: $n_total   already done: $n_done   pending: $(length(pending))")
    else
        println("grid points: shard $(shard.index)/$(shard.count) -> $n_total assigned of $n_global global   already done: $n_done   pending: $(length(pending))")
    end
    println("execution: parallel_tasks=$(exec.parallel_tasks)  BLAS threads=$(exec.blas_threads)")
    println("output: $out_dir")
    println("=" ^ 78)

    wall_times = Float64[]
    n_ok = Ref(0)
    n_failed = Ref(0)
    completed = Ref(0)
    next_index = Ref(0)
    work_lock = ReentrantLock()
    io_lock = ReentrantLock()
    worker_count = min(exec.parallel_tasks, max(length(pending), 1))
    run_start = time_ns()

    function record_result(p::SweepPoint, row::AbstractDict)
        lock(io_lock) do
            append_metrics_row(metrics_csv, row)
            wall_seconds = Float64(row["wall_seconds"])
            push!(wall_times, wall_seconds)
            if row["status"] == "ok"
                n_ok[] += 1
            else
                n_failed[] += 1
            end
            completed[] += 1
            med = median(wall_times)
            remaining = (length(pending) - completed[]) * med / max(worker_count, 1)
            print_progress_line(completed[], length(pending), p, row,
                                wall_seconds, remaining / 60)
        end
        return nothing
    end

    function run_one(p::SweepPoint)
        row = compute_and_save_point(p, num, seeds, ds, dyn_settings,
                                              spectra_dir)
        record_result(p, row)
        return nothing
    end

    if exec.parallel_tasks <= 1 || length(pending) <= 1
        for p in pending
            run_one(p)
        end
    else
        function take_next_point()
            lock(work_lock) do
                next_index[] += 1
                idx = next_index[]
                return idx <= length(pending) ? pending[idx] : nothing
            end
        end

        function worker_loop()
            while true
                p = take_next_point()
                p === nothing && break
                run_one(p)
            end
            return nothing
        end

        tasks = [Threads.@spawn worker_loop() for _ in 1:worker_count]
        foreach(fetch, tasks)
    end

    total_wall = (time_ns() - run_start) / 1e9

    # Recompute coverage against the full metrics.csv (includes prior runs).
    final_done = read_done_keys(metrics_csv, spectra_dir)
    coverage = n_total == 0 ? 1.0 : length(final_done) / n_total
    println("\nRun summary:")
    println("  this run: ok=$(n_ok[]) failed=$(n_failed[])")
    println("  coverage: $(@sprintf "%.2f%%" (coverage * 100)) ($(length(final_done))/$n_total done)")
    println("  this-run wall: $(@sprintf "%.1f" total_wall) s")
    println("  metrics:  $metrics_csv")
    if !isempty(wall_times)
        println("  median per-point: $(@sprintf "%.2f" median(wall_times)) s; " *
                "max: $(@sprintf "%.2f" maximum(wall_times)) s")
    end

    return (metrics = metrics_csv, coverage = coverage, n_ok = n_ok[], n_failed = n_failed[])
end

if abspath(PROGRAM_FILE) == @__FILE__
    cli = parse_cli_args(ARGS)
    run_sweep(cli.config_path;
                     shard_index = cli.shard_index,
                     shard_count = cli.shard_count,
                     data_subdir_override = cli.data_subdir_override)
end
