#!/usr/bin/env julia

using JLD2
using LinearAlgebra
using Printf

include(joinpath(@__DIR__, "_integrable_corner_physical.jl"))

Base.@kwdef mutable struct CornerComputeCLI
    n_orb::Int = 10
    filling::Int = 3
    seed::Int = 1
    output::String = ""
    n_tau::Int = CORNER_TAU_RANGE[3]
    row_block::Int = 32
    time_block::Int = 16
    force::Bool = false
end

function usage()
    println("""
Usage: julia integrable_corner_compute.jl --n-orb N --filling F --seed S --output FILE [options]

Options:
  --n-tau N       logarithmic tau samples in [1e-3,10] (default 250)
  --row-block N   deterministic upper-triangle row block (default 32)
  --time-block N  Fourier-loop time block (default 16)
  --force         replace a matching existing cache
""")
end

function parse_cli(argv)
    cli = CornerComputeCLI()
    i = 1
    while i <= length(argv)
        arg = argv[i]
        if arg == "--help"
            usage(); exit(0)
        elseif arg == "--force"
            cli.force = true; i += 1; continue
        end
        i == length(argv) && error("missing value for $arg")
        value = argv[i + 1]
        if arg == "--n-orb"; cli.n_orb = parse(Int, value)
        elseif arg == "--filling"; cli.filling = parse(Int, value)
        elseif arg == "--seed"; cli.seed = parse(Int, value)
        elseif arg == "--output"; cli.output = value
        elseif arg == "--n-tau"; cli.n_tau = parse(Int, value)
        elseif arg == "--row-block"; cli.row_block = parse(Int, value)
        elseif arg == "--time-block"; cli.time_block = parse(Int, value)
        else; error("unknown argument: $arg")
        end
        i += 2
    end
    isempty(cli.output) && error("--output is required")
    0 < cli.filling < cli.n_orb || error("filling must satisfy 0 < F < N")
    cli.seed > 0 || error("seed must be positive")
    cli.n_tau >= 8 || error("n-tau must be at least 8")
    return cli
end

function matching_cache(path, cli)
    isfile(path) || return false
    return JLD2.jldopen(path, "r") do file
        get(file, "complete", false) === true &&
        Int(file["n_orb"]) == cli.n_orb &&
        Int(file["filling"]) == cli.filling &&
        Int(file["seed"]) == cli.seed &&
        Int(file["n_tau"]) == cli.n_tau
    end
end

function atomic_write_corner_cache(path, cli, g, data, taus, trace, elapsed)
    mkpath(dirname(abspath(path)))
    temporary = abspath(path) * ".tmp.$(getpid())"
    staircase = trace.staircase
    try
        JLD2.jldopen(temporary, "w") do file
            file["complete"] = true
            file["n_orb"] = cli.n_orb
            file["filling"] = cli.filling
            file["seed"] = cli.seed
            file["d"] = length(data.f)
            file["D_b"] = length(data.f)^2
            file["n_tau"] = cli.n_tau
            file["taus"] = taus
            file["qs"] = 2pi .* taus
            file["Z_re"] = real.(trace.Z)
            file["Z_im"] = imag.(trace.Z)
            file["g"] = g
            file["epsilon"] = data.epsilon
            file["f_values"] = data.f
            file["h_values"] = data.h
            file["E_values"] = data.E
            file["center_c"] = staircase.center
            file["sigma_min"] = staircase.minimum
            file["sigma_max"] = staircase.maximum
            file["diagonal_sigma"] = staircase.diagonal_sigma
            file["zero_count_le_1e-8"] = staircase.zero_count
            file["staircase_knots"] = staircase.knots
            file["staircase_counts"] = staircase.counts
            file["staircase_coefficients"] = collect(staircase.polynomial)
            file["staircase_degree"] = staircase.degree
            file["staircase_n_bins"] = CORNER_N_BINS
            file["J"] = CORNER_J
            file["gamma_eff_over_J"] = CORNER_GAMMA_EFF
            file["kappa_eff_over_J"] = CORNER_KAPPA_EFF
            file["gamma_tot_over_J"] = CORNER_GAMMA_TOT
            file["n_grid"] = CORNER_N_GRID
            file["box_length"] = CORNER_BOX_LENGTH
            file["correlation_length"] = CORNER_CORRELATION_LENGTH
            file["disorder_strength"] = CORNER_DISORDER_STRENGTH
            file["row_block"] = cli.row_block
            file["time_block"] = cli.time_block
            file["julia_threads"] = Threads.nthreads()
            file["elapsed_seconds"] = elapsed
        end
        mv(temporary, abspath(path); force = true)
    finally
        isfile(temporary) && rm(temporary; force = true)
    end
end

function main(argv = ARGS)
    cli = parse_cli(argv)
    if !cli.force && matching_cache(cli.output, cli)
        println("cache already complete: $(abspath(cli.output))")
        return
    end
    t0 = time()
    @printf("corner seed: N=%d filling=%d seed=%d threads=%d\n",
            cli.n_orb, cli.filling, cli.seed, Threads.nthreads())
    g = build_physical_corner_g(cli.n_orb, cli.seed)
    data = fixed_filling_corner_data(g, cli.filling; J = CORNER_J)
    staircase = streamed_staircase(data.f, data.E, CORNER_GAMMA_TOT;
        n_bins = CORNER_N_BINS, degree = CORNER_DEGREE)
    taus = corner_tau_grid(cli.n_tau)
    trace = streamed_unfolded_trace(data.f, data.E, CORNER_GAMMA_TOT,
        2pi .* taus; staircase = staircase, pair_block_rows = cli.row_block,
        time_block = cli.time_block, threaded = true)
    elapsed = time() - t0
    atomic_write_corner_cache(cli.output, cli, g, data, taus, trace, elapsed)
    @printf("wrote %s  d=%d D_b=%d zero_count=%d elapsed=%.1fs\n",
            abspath(cli.output), length(data.f), length(data.f)^2,
            staircase.zero_count, elapsed)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
