# sweep_scheduler.jl
#
# production (Lamb-Dicke chaos study) - resumable sweep sweep scheduler.
#
# This is a deliberately dependency-light file: it depends only on `Base` so
# that the fast unit tests can include it without loading the heavy model /
# plotting stack.  The sweep driver and the aggregation script include this
# file and layer the physics + I/O on top.
#
# Convention lock:
#   - The per-row primary key is (config, eta, gamma, svd_tol, seed).
#   - A row is "done" iff metrics.csv has a row with that key and
#     status == "ok" AND the corresponding .jld2 file exists.
#   - status == "failed" rows are retained in the work list (retried).
#   - Floats are formatted with Base `string`, which is deterministic for the
#     literal Float64 grid values used in production, so file names and CSV keys
#     round-trip exactly.

"""
    SweepPoint(config, eta, gamma, tol, seed)

One unit of production work for the physical or retained synthetic controls.
"""
struct SweepPoint
    config::String
    eta::Float64
    gamma::Float64
    tol::Float64
    seed::Int
end

"""
    SweepGrid(; kwargs...)

Declarative description of the production Cartesian grid.
"""
struct SweepGrid
    configs::Vector{String}
    etas::Vector{Float64}
    gammas::Vector{Float64}
    seeds::Vector{Int}
    baseline_tol::Float64

end

function SweepGrid(; configs, etas, gammas, seeds, baseline_tol)
    any(isempty, (configs, etas, gammas, seeds)) &&
        throw(ArgumentError("grid axes must not be empty"))
    all(isfinite, etas) && all(>=(0), etas) ||
        throw(ArgumentError("etas must be finite and nonnegative"))
    all(isfinite, gammas) && all(>=(0), gammas) ||
        throw(ArgumentError("gammas must be finite and nonnegative"))
    isfinite(baseline_tol) && baseline_tol > 0 ||
        throw(ArgumentError("baseline_tol must be finite and positive"))
    all(>(0), seeds) || throw(ArgumentError("seeds must be positive"))
    return SweepGrid(String.(collect(configs)), Float64.(collect(etas)),
                         Float64.(collect(gammas)), Int.(collect(seeds)),
                         Float64(baseline_tol))
end

# ----------------------------------------------------------------------------
# Canonical formatting / keys / file names
# ----------------------------------------------------------------------------

"Format a grid scalar to its canonical, deterministic string."
fmt_grid_value(x::Real) = string(x)

"Canonical primary key string `config|eta|gamma|tol|seed`."
function point_key(p::SweepPoint)
    return string(p.config, "|", fmt_grid_value(p.eta), "|",
                  fmt_grid_value(p.gamma), "|", fmt_grid_value(p.tol), "|",
                  string(p.seed))
end

"""
    point_key_from_fields(config, eta, gamma, tol, seed)

Build the same canonical key from already-stringified CSV cells.  The grid
values are parsed and re-formatted so that, for example, both `"0.4"` and
`"0.40"` map to the canonical `"0.4"`.
"""
function point_key_from_fields(config, eta, gamma, tol, seed)
    return string(String(config), "|",
                  fmt_grid_value(parse(Float64, String(eta))), "|",
                  fmt_grid_value(parse(Float64, String(gamma))), "|",
                  fmt_grid_value(parse(Float64, String(tol))), "|",
                  string(parse(Int, String(seed))))
end

"Deterministic per-row JLD2 file name for a sweep point."
function spectrum_filename(p::SweepPoint)
    return string(p.config,
                  "__eta=", fmt_grid_value(p.eta),
                  "__gamma=", fmt_grid_value(p.gamma),
                  "__tol=", fmt_grid_value(p.tol),
                  "__seed=", string(p.seed),
                  ".jld2")
end

# ----------------------------------------------------------------------------
# Grid builder
# ----------------------------------------------------------------------------

"Return the deterministic Cartesian product of sweep work items."
function build_sweep_grid(spec::SweepGrid)
    points = SweepPoint[]
    for config in spec.configs
        for eta in spec.etas
            for gamma in spec.gammas
                for seed in spec.seeds
                    push!(points,
                          SweepPoint(config, eta, gamma, spec.baseline_tol, seed))
                end
            end
        end
    end
    return points
end

"Validate a 1-based shard specification `(index, count)` for a work list."
function validate_shard_spec(shard_index::Integer, shard_count::Integer)
    shard_count >= 1 || throw(ArgumentError("shard_count must be >= 1, got $shard_count"))
    1 <= shard_index <= shard_count ||
        throw(ArgumentError("shard_index must satisfy 1 <= shard_index <= shard_count; got index=$shard_index count=$shard_count"))
    return (index = Int(shard_index), count = Int(shard_count))
end

"Return the deterministic sublist assigned to one shard of the global work order."
function shard_points(points::AbstractVector{SweepPoint},
                      shard_index::Integer,
                      shard_count::Integer)
    shard = validate_shard_spec(shard_index, shard_count)
    return SweepPoint[p for (i, p) in enumerate(points)
                      if mod(i - 1, shard.count) + 1 == shard.index]
end

# ----------------------------------------------------------------------------
# CSV helpers (minimal, quote-aware)
# ----------------------------------------------------------------------------

"""
    parse_csv_line(line) -> Vector{String}

Parse one CSV record, honouring double-quoted fields and `""` escapes.  Used
by the resume logic so that free-text columns (warnings, failure reasons)
containing commas do not corrupt key parsing.
"""
function parse_csv_line(line::AbstractString)
    cells = String[]
    buf = IOBuffer()
    in_quotes = false
    i = firstindex(line)
    n = lastindex(line)
    while i <= n
        c = line[i]
        if in_quotes
            if c == '"'
                if i < n && line[nextind(line, i)] == '"'
                    write(buf, '"')
                    i = nextind(line, i)
                else
                    in_quotes = false
                end
            else
                write(buf, c)
            end
        else
            if c == '"'
                in_quotes = true
            elseif c == ','
                push!(cells, String(take!(buf)))
            else
                write(buf, c)
            end
        end
        i = nextind(line, i)
    end
    push!(cells, String(take!(buf)))
    return cells
end

# Key columns the resume logic must locate in metrics.csv.
const SWEEP_KEY_COLUMNS = ("config", "eta", "gamma", "svd_tol", "seed")

"""
    read_done_keys(metrics_csv, spectra_dir; allowed_configs=nothing) -> Set{String}

Return the set of canonical keys that are complete: present in `metrics_csv`
with `status == "ok"` and backed by an existing `.jld2` file in
`spectra_dir`.  Returns an empty set if `metrics_csv` does not exist.

When `allowed_configs` is provided, only rows whose `config` is in that set
are considered complete.
"""
function read_done_keys(metrics_csv::AbstractString, spectra_dir::AbstractString;
                        allowed_configs = nothing)
    done = Set{String}()
    isfile(metrics_csv) || return done
    open(metrics_csv, "r") do io
        header_line = readline(io)
        isempty(header_line) && return
        headers = parse_csv_line(header_line)
        idx = Dict(h => i for (i, h) in enumerate(headers))
        all(haskey(idx, c) for c in SWEEP_KEY_COLUMNS) && haskey(idx, "status") ||
            error("metrics.csv is missing required key columns; found headers: $headers")
        for line in eachline(io)
            isempty(strip(line)) && continue
            cells = parse_csv_line(line)
            length(cells) < length(headers) && continue
            cells[idx["status"]] == "ok" || continue
            config_name = String(cells[idx["config"]])
            allowed_configs !== nothing && !(config_name in allowed_configs) && continue
            key = point_key_from_fields(config_name,
                                        cells[idx["eta"]],
                                        cells[idx["gamma"]],
                                        cells[idx["svd_tol"]],
                                        cells[idx["seed"]])
            # The .jld2 file name is reconstructed from the canonical fields.
            p = SweepPoint(config_name,
                           parse(Float64, String(cells[idx["eta"]])),
                           parse(Float64, String(cells[idx["gamma"]])),
                           parse(Float64, String(cells[idx["svd_tol"]])),
                           parse(Int, String(cells[idx["seed"]])))
            isfile(joinpath(spectra_dir, spectrum_filename(p))) || continue
            push!(done, key)
        end
    end
    return done
end

"""
    pending_points(grid, metrics_csv, spectra_dir) -> Vector{SweepPoint}

Return the grid points that still need to run (`grid` minus the done set).
"""
function pending_points(grid::AbstractVector{SweepPoint},
                        metrics_csv::AbstractString,
                        spectra_dir::AbstractString)
    done = read_done_keys(metrics_csv, spectra_dir)
    return SweepPoint[p for p in grid if !(point_key(p) in done)]
end

# ----------------------------------------------------------------------------
# Test/append helpers (used by the unit tests and the driver)
# ----------------------------------------------------------------------------

"Full metrics.csv header for production (key + diagnostics + bookkeeping)."
const SWEEP_METRICS_HEADERS = String[
    "config", "eta", "gamma", "svd_tol", "seed",
    "wall_seconds", "mem_mb_delta", "status", "failure_reason",
]

"Write the metrics.csv header line (creating the directory as needed)."
function write_metrics_header(metrics_csv::AbstractString)
    mkpath(dirname(metrics_csv))
    open(metrics_csv, "w") do io
        println(io, join(SWEEP_METRICS_HEADERS, ","))
    end
    return metrics_csv
end

"Quote-escape one CSV cell."
function sweep_csv_cell(x)
    x === missing && return ""
    if x isa AbstractFloat
        isnan(x) && return "NaN"
        isinf(x) && return x > 0 ? "Inf" : "-Inf"
    end
    s = string(x)
    if occursin(r"[,\n\"]", s)
        return "\"" * replace(s, "\"" => "\"\"") * "\""
    end
    return s
end

"Append one full metrics row (a `Dict`) to metrics.csv."
function append_metrics_row(metrics_csv::AbstractString, row::AbstractDict)
    open(metrics_csv, "a") do io
        println(io, join((sweep_csv_cell(get(row, h, missing))
                          for h in SWEEP_METRICS_HEADERS), ","))
    end
    return metrics_csv
end

"""
    append_status_row(metrics_csv, p, status; failure_reason="")

Append a minimal key + status row.  Used by the tests and as a fallback row
shape; the driver appends the full diagnostics row instead.
"""
function append_status_row(metrics_csv::AbstractString, p::SweepPoint,
                           status::AbstractString; failure_reason::AbstractString = "")
    row = Dict{String,Any}(
        "config" => p.config,
        "eta" => p.eta,
        "gamma" => p.gamma,
        "svd_tol" => p.tol,
        "seed" => p.seed,
        "status" => status,
        "failure_reason" => failure_reason,
    )
    append_metrics_row(metrics_csv, row)
    return metrics_csv
end

function grid_spec_from_config(cfg::AbstractDict)
    grid = cfg["grid"]
    return SweepGrid(
        configs = String.(grid["configs"]),
        etas = Float64.(grid["etas"]),
        gammas = Float64.(grid["gammas"]),
        seeds = collect(1:Int(grid["seed_count"])),
        baseline_tol = Float64(grid["baseline_svd_tol"]),
    )
end
