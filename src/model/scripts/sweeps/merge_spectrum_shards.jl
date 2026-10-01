#!/usr/bin/env julia
# production shard merge + completeness check.
#
# Run from the repository root after shard-local worker runs finish:
#
#     julia --project=src/environment \
#         src/model/scripts/sweeps/merge_spectrum_shards.jl \
#         --config src/model/configs/production_x0_merged_smoke_n4.toml
#
# The script validates that every expected point appears exactly once with
# `status == "ok"`, that the corresponding spectrum file exists and is
# loadable, then writes a merged metrics.csv and a symlinked spectra/ tree at
# the merged output path encoded in the config.

using JLD2
using TOML

const HERE = @__DIR__
const JULIA_DIR = abspath(joinpath(HERE, "..", "..", ".."))
const MODULE_DIR = abspath(joinpath(HERE, "..", ".."))
const DATA_ROOT = abspath(joinpath(JULIA_DIR, "..", "data"))

include(joinpath(HERE, "sweep_scheduler.jl"))

function parse_cli_args(argv)
    config_path = joinpath(MODULE_DIR, "configs", "physical_smoke_n4.toml")
    shards_root = nothing
    i = 1
    while i <= length(argv)
        a = argv[i]
        if a == "--config"
            i += 1
            i <= length(argv) || error("--config requires a path argument")
            config_path = argv[i]
        elseif startswith(a, "--config=")
            config_path = split(a, "=", limit = 2)[2]
        elseif a == "--shards-root"
            i += 1
            i <= length(argv) || error("--shards-root requires a path argument")
            shards_root = argv[i]
        elseif startswith(a, "--shards-root=")
            shards_root = split(a, "=", limit = 2)[2]
        else
            error("unrecognized argument: $a")
        end
        i += 1
    end
    return (config_path = abspath(config_path),
            shards_root = isnothing(shards_root) ? nothing : abspath(String(shards_root)))
end

merged_out_dir(cfg::AbstractDict) =
    abspath(joinpath(DATA_ROOT, String(cfg["output"]["data_subdir"])))

function find_worker_dirs(shards_root::AbstractString)
    worker_dirs = String[]
    isdir(shards_root) || error("shards root not found: $shards_root")
    for (root, _, files) in walkdir(shards_root)
        "metrics.csv" in files || continue
        push!(worker_dirs, root)
    end
    sort!(worker_dirs)
    isempty(worker_dirs) && error("no shard metrics.csv files found under $shards_root")
    return worker_dirs
end

function load_ok_rows(metrics_csv::AbstractString; allowed_configs = nothing)
    rows = Vector{Dict{String,String}}()
    open(metrics_csv, "r") do io
        eof(io) && error("metrics.csv is empty: $metrics_csv")
        header = parse_csv_line(readline(io))
        header == SWEEP_METRICS_HEADERS ||
            error("metrics.csv header mismatch in $metrics_csv")
        idx = Dict(h => i for (i, h) in enumerate(header))
        for line in eachline(io)
            isempty(strip(line)) && continue
            cells = parse_csv_line(line)
            length(cells) == length(header) ||
                error("metrics.csv row has wrong column count in $metrics_csv")
            cells[idx["status"]] == "ok" || continue
            config_name = String(cells[idx["config"]])
            allowed_configs !== nothing && !(config_name in allowed_configs) && continue
            row = Dict{String,String}(header[i] => cells[i] for i in eachindex(header))
            push!(rows, row)
        end
    end
    return rows
end

function spectrum_path_from_row(row::AbstractDict{String,String}, spectra_dir::AbstractString)
    point = SweepPoint(
        String(row["config"]),
        parse(Float64, row["eta"]),
        parse(Float64, row["gamma"]),
        parse(Float64, row["svd_tol"]),
        parse(Int, row["seed"]),
    )
    return point, joinpath(spectra_dir, spectrum_filename(point))
end

"Cavity-loss rate resolver used by merge validation (mirrors the builder helper)."

function _merge_expected_manual_cavity_rate_over_J(cfg::AbstractDict,
                                                    manual_rate_over_J::Real)
    configs = haskey(cfg, "grid") ?
        String.(get(cfg["grid"], "configs", String[])) : String[]
    "synthetic_l3b_bdi_syk4_fig3_m300" in configs || "synthetic_l3b_bdi_syk4_fig3_m10" in configs ||
        return Float64(manual_rate_over_J)

    num = cfg["numerics"]
    haskey(num, "calibration_jld2") ||
        error("Figure 3 calibrated control requires numerics.calibration_jld2")
    calibration_path = abspath(String(num["calibration_jld2"]))
    isfile(calibration_path) ||
        error("Figure 3 calibration payload not found: $calibration_path")
    expected_jumps = "synthetic_l3b_bdi_syk4_fig3_m300" in configs ? 300 : 10
    scale = JLD2.jldopen(calibration_path, "r") do file
        Int(file["n_random_jumps"]) == expected_jumps || error("expected $expected_jumps random jumps")
        Float64(file["figure3_dissipator_rate_scale"])
    end
    isfinite(scale) && scale > 0 ||
        error("invalid Figure 3 dissipator rate scale: $scale")
    return Float64(manual_rate_over_J) * scale
end

function validate_payload_metadata!(meta, cfg::AbstractDict, eta::Float64,
                                    path::AbstractString)
    num = cfg["numerics"]
    for key in ("n_orb", "n_grid", "box_length", "lambda_c_micron",
                "correlation_length", "disorder_strength", "J", "kappa_over_2pi_mhz")
        haskey(num, key) || continue
        haskey(meta, key) && isapprox(meta[key], num[key]; atol=1e-14, rtol=1e-12) ||
            error("spectrum $key mismatch: $path")
    end
    filling = get(num, "filling", div(num["n_orb"], 2))
    meta["filling"] == filling || error("spectrum filling mismatch: $path")
    delta = Float64(get(num, "delta_cd_over_2pi_mhz", meta["delta_cd_over_2pi_mhz"]))
    meta["delta_cd_over_2pi_mhz"] == delta || error("spectrum detuning mismatch: $path")
    rate = haskey(num, "cavity_loss_rate_over_J") ?
        _merge_expected_manual_cavity_rate_over_J(cfg, num["cavity_loss_rate_over_J"]) :
        Float64(get(num, "kappa_over_2pi_mhz", 0.2)) / delta
    isapprox(meta["cavity_loss_rate_over_J"], rate; atol=1e-14, rtol=1e-12) ||
        error("spectrum cavity loss rate mismatch: $path")
    return nothing
end

function assert_loadable_spectrum(path::AbstractString,
                                  cfg::AbstractDict,
                                  point::SweepPoint)
    isfile(path) || error("missing spectrum file: $path")
    JLD2.jldopen(path, "r") do f
        haskey(f, "metadata") || error("spectrum payload missing metadata: $path")
        validate_payload_metadata!(f["metadata"], cfg, point.eta, path)
        dim = binomial(cfg["numerics"]["n_orb"],
            get(cfg["numerics"], "filling", div(cfg["numerics"]["n_orb"], 2)))
        for key in ("L_eigvals", "L_svd_S_centered")
            values = f[key]
            length(values) == dim^2 && all(isfinite, values) ||
                error("invalid $key: $path")
        end
        times, entropy, populations = f["times"], f["entropy_t"], f["populations_t"]
        length(times) == length(entropy) &&
            size(populations) == (cfg["numerics"]["n_orb"], length(times)) ||
            error("dynamics dimensions mismatch: $path")
    end
    return path
end

function merge_shards(config_path::AbstractString; shards_root::Union{Nothing,String} = nothing)
    cfg = TOML.parsefile(config_path)
    out_dir = merged_out_dir(cfg)
    batch_root = dirname(out_dir)
    shards_root = isnothing(shards_root) ? joinpath(batch_root, "shards") : String(shards_root)
    worker_dirs = find_worker_dirs(shards_root)

    expected_grid = build_sweep_grid(grid_spec_from_config(cfg))
    expected_keys = point_key.(expected_grid)

    row_by_key = Dict{String,Dict{String,String}}()
    spectrum_by_key = Dict{String,String}()
    origin_by_key = Dict{String,String}()

    for worker_dir in worker_dirs
        metrics_csv = joinpath(worker_dir, "metrics.csv")
        spectra_dir = joinpath(worker_dir, "spectra")
        for row in load_ok_rows(metrics_csv)
            point, spectrum_path = spectrum_path_from_row(row, spectra_dir)
            key = point_key(point)
            haskey(row_by_key, key) &&
                error("duplicate ok row for $key in $(origin_by_key[key]) and $worker_dir")
            assert_loadable_spectrum(spectrum_path, cfg, point)
            row_by_key[key] = row
            spectrum_by_key[key] = spectrum_path
            origin_by_key[key] = worker_dir
        end
    end

    found_keys = collect(keys(row_by_key))
    missing_keys = sort!(setdiff(expected_keys, found_keys))
    extra_keys = sort!(setdiff(found_keys, expected_keys))
    isempty(missing_keys) || error("missing $(length(missing_keys)) expected points; first missing key=$(first(missing_keys))")
    isempty(extra_keys) || error("found $(length(extra_keys)) unexpected points; first unexpected key=$(first(extra_keys))")

    merged_metrics = joinpath(out_dir, "metrics.csv")
    merged_spectra = joinpath(out_dir, "spectra")
    rm(merged_metrics; force = true)
    rm(merged_spectra; force = true, recursive = true)
    mkpath(merged_spectra)
    write_metrics_header(merged_metrics)

    for point in expected_grid
        key = point_key(point)
        append_metrics_row(merged_metrics, row_by_key[key])
        target = joinpath(merged_spectra, spectrum_filename(point))
        symlink(spectrum_by_key[key], target)
    end

    println("production shard merge complete")
    println("  worker dirs: $(length(worker_dirs))")
    println("  merged ok points: $(length(row_by_key)) / $(length(expected_grid))")
    println("  merged metrics: $merged_metrics")
    println("  merged spectra: $merged_spectra")
    return (metrics = merged_metrics, spectra = merged_spectra,
            points = length(row_by_key))

end

if abspath(PROGRAM_FILE) == @__FILE__
    cli = parse_cli_args(ARGS)
    merge_shards(cli.config_path; shards_root = cli.shards_root)
end
