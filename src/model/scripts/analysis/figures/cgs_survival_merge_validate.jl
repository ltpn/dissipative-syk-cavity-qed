#!/usr/bin/env julia

using JLD2
using Printf: @sprintf
using Random: MersenneTwister
using TOML

const CGS_MERGE_HERE = @__DIR__
const CGS_MERGE_LAMB_DIR = abspath(joinpath(CGS_MERGE_HERE, "..", "..", ".."))

include(joinpath(CGS_MERGE_LAMB_DIR, "atomic_jld2.jl"))
include(joinpath(CGS_MERGE_LAMB_DIR, "CGSSurvival.jl"))
include(joinpath(CGS_MERGE_LAMB_DIR, "CGSModelBuilders.jl"))

using .CGSSurvival

const CGS_FIGURE1_PALETTE_POSITIONS = [
    0.05, 0.3333333333333333, 0.6166666666666667, 0.9]
const CGS_FIGURE1_CONTROL_RGB = [0.65, 0.0, 0.0]
const CGS_FIGURE3_RGB = [
    [0.9968296, 0.7724726, 0.5373508],
    [0.9666100, 0.4361110, 0.3596695],
    [0.7458284, 0.2254362, 0.4656836],
    [0.4555515, 0.1268050, 0.5073730],
]
const CGS_FIGURE3_CONTROL_RGB = [0.05, 0.34, 0.20]

cgs_seed_path(root::AbstractString, key::AbstractString, seed::Integer) =
    joinpath(abspath(root), "seeds", String(key),
             "seed_$(lpad(Int(seed), 3, '0')).jld2")

_cgs_atomic_jld2(writer::Function, path::AbstractString) = atomic_jld2(writer, abspath(path))

function _cgs_write_merged_dataset(path::AbstractString, payload::AbstractDict)
    return _cgs_atomic_jld2(path) do file
        for (key, value) in payload
            file[String(key)] = value
        end
    end
end

function cgs_read_merged_dataset(path::AbstractString)
    input = abspath(path)
    isfile(input) || throw(ArgumentError("merged dataset not found: $input"))
    return JLD2.jldopen(input, "r") do file
        required = ("dataset_key", "seeds", "times",
                    "payloads",
                    "n_orb", "filling", "hilbert_dim", "liouvillian_dim",
                    "production")
        for key in required
            haskey(file, key) || throw(ArgumentError("merged dataset lacks $key"))
        end
        Dict{String,Any}(key => file[key] for key in required)
    end
end

function _cgs_require_seed_identity(payload::AbstractDict,
                                    dataset_key::AbstractString,
                                    seed::Integer;
                                    expected_n_orb::Integer,
                                    expected_filling::Integer,
                                    production::Bool)
    String(payload["identity/dataset_key"]) == dataset_key ||
        throw(ArgumentError("seed $seed has the wrong dataset key"))
    stored_seed = Int(payload["identity/seed"])
    stored_seed == seed || throw(ArgumentError(
        "file for seed $seed contains seed $stored_seed"))
    Bool(payload["identity/production"]) == production ||
        throw(ArgumentError("seed $seed production marker mismatch"))
    Int(payload["dimensions/n_orb"]) == expected_n_orb ||
        throw(DimensionMismatch("seed $seed has wrong n_orb"))
    Int(payload["dimensions/filling"]) == expected_filling ||
        throw(DimensionMismatch("seed $seed has wrong filling"))
    return nothing
end

function cgs_merge_dataset(root::AbstractString,
                           dataset_key::AbstractString,
                           expected_seeds,
                           thresholds::AbstractDict;
                           expected_n_orb::Integer,
                           expected_filling::Integer,
                           production::Bool)
    seeds = Int.(collect(expected_seeds))
    !isempty(seeds) && length(unique(seeds)) == length(seeds) ||
        throw(ArgumentError("expected seeds must be nonempty and unique"))
    sort(seeds) == seeds || throw(ArgumentError("expected seeds must be sorted"))
    payloads = Vector{Dict{String,Any}}(undef, length(seeds))
    reference_times = nothing
    reference_parameters = nothing
    reference_parameter = nothing
    for (index, seed) in pairs(seeds)
        path = cgs_seed_path(root, dataset_key, seed)
        isfile(path) || throw(ArgumentError(
            "missing seed payload for $dataset_key seed $seed: $path"))
        payload = read_seed_payload(path;
            validate = true, thresholds = thresholds)
        _cgs_require_seed_identity(payload, dataset_key, seed;
            expected_n_orb = expected_n_orb,
            expected_filling = expected_filling,
            production = production)
        times = Float64.(payload["survival/times"])
        parameter = Float64(payload["identity/parameter"])
        parameters = Dict(k => v for (k, v) in payload["model/resolved_metadata"]
                          if k in ("n_orb", "filling", "n_grid", "box_length", "eta",
                                   "delta_tilde", "delta_cd_over_2pi_mhz", "kappa_over_2pi_mhz",
                                   "correlation_length", "disorder_strength", "J",
                                   "cavity_loss_rate_over_J", "gamma_eff_over_J", "n_random_jumps",
                                   "seed_offset", "dissipator_rate_scale"))
        if index == 1
            reference_times = times
            reference_parameters = parameters
            reference_parameter = parameter
        else
            times == reference_times ||
                throw(ArgumentError("seed $seed time grid mismatch"))
            isequal(parameters, reference_parameters) ||
                throw(ArgumentError("seed $seed model parameters mismatch"))
            parameter == reference_parameter ||
                throw(ArgumentError("seed $seed parameter mismatch"))
        end
        payloads[index] = payload
    end
    d = Int(payloads[1]["dimensions/hilbert"])
    K = Int(payloads[1]["dimensions/liouvillian"])
    merged_payload = Dict{String,Any}(
        "dataset_key" => String(dataset_key),
        "seeds" => seeds,
        "times" => reference_times,
        "payloads" => payloads,
        "n_orb" => Int(expected_n_orb),
        "filling" => Int(expected_filling),
        "hilbert_dim" => d,
        "liouvillian_dim" => K,
        "production" => production,
    )
    output = joinpath(abspath(root), "merged", "$(dataset_key).jld2")
    _cgs_write_merged_dataset(output, merged_payload)
    reread = cgs_read_merged_dataset(output)
    reread["seeds"] == seeds || error("merged dataset failed seed validation")
    return (path = output,)
end

function _cgs_source_delta0(seed_payload::AbstractDict)
    metadata = seed_payload["model/resolved_metadata"]
    metadata isa AbstractDict || throw(ArgumentError(
        "model/resolved_metadata must be a dictionary"))
    delta0 = Float64(metadata["delta_0_over_2pi_mhz"])
    isfinite(delta0) && delta0 > 0 || throw(ArgumentError(
        "resolved model Delta_0 must be finite and positive"))
    return delta0
end

function _cgs_validate_merged_inputs(merged::AbstractDict, key::AbstractString)
    String(merged["dataset_key"]) == key ||
        throw(ArgumentError("merged dataset key mismatch"))
    payloads = merged["payloads"]
    length(payloads) == length(merged["seeds"]) > 0 ||
        throw(DimensionMismatch("merged seed payload count mismatch"))
    first_seed = first(payloads)
    delta0 = _cgs_source_delta0(first_seed)
    for (seed, payload) in zip(merged["seeds"], payloads)
        String(payload["identity/dataset_key"]) == key ||
            throw(ArgumentError("merged source contains the wrong seed key"))
        payload["identity/seed"] == seed ||
            throw(ArgumentError("merged seed identity mismatch"))
        payload["identity/family"] == first_seed["identity/family"] ||
            throw(ArgumentError("merged source family mismatch"))
        isapprox(_cgs_source_delta0(payload), delta0; rtol=0, atol=32eps(Float64)) ||
            throw(ArgumentError("merged source detuning mismatch"))
    end
    return nothing
end

function _cgs_curve_label(first_seed::AbstractDict, control::Bool)
    control && return "diss. SYK"
    return @sprintf("%g", Float64(first_seed["identity/parameter"]))
end

function cgs_build_plotdata(merged_paths, dataset_keys;
                            n_boot::Integer,
                            bootstrap_seed::Integer)
    keys = String.(dataset_keys)
    paths = abspath.(String.(merged_paths))
    !isempty(keys) && length(keys) == length(paths) ||
        throw(ArgumentError("dataset keys and paths must have equal nonzero length"))
    length(unique(keys)) == length(keys) ||
        throw(ArgumentError("plot-data output keys must be unique"))
    all(isfile, paths) || throw(ArgumentError("merged plot-data input missing"))
    payload = Dict{String,Any}(
        "dataset_keys" => keys,
        "n_bootstrap" => Int(n_boot),
        "bootstrap_seed" => Int(bootstrap_seed),
        "figure1_palette_positions" => copy(CGS_FIGURE1_PALETTE_POSITIONS),
        "figure1_control_rgb" => copy(CGS_FIGURE1_CONTROL_RGB),
        "figure3_rgb" => deepcopy(CGS_FIGURE3_RGB),
        "figure3_control_rgb" => copy(CGS_FIGURE3_CONTROL_RGB),
    )
    reference_times = nothing
    reference_seeds = nothing
    all_seeds_equal = true
    for (index, key) in enumerate(keys)
        path = paths[index]
        merged = cgs_read_merged_dataset(path)
        _cgs_validate_merged_inputs(merged, key)
        times = Float64.(merged["times"])
        seeds = Int.(merged["seeds"])
        if index == 1
            reference_times = times
            reference_seeds = seeds
            payload["times"] = times
            payload["n_orb"] = Int(merged["n_orb"])
            payload["filling"] = Int(merged["filling"])
            payload["hilbert_dim"] = Int(merged["hilbert_dim"])
            payload["liouvillian_dim"] = Int(merged["liouvillian_dim"])
            payload["production"] = Bool(merged["production"])
        else
            times == reference_times ||
                throw(ArgumentError("merged time grids differ"))
            all_seeds_equal &= seeds == reference_seeds
            Int(merged["n_orb"]) == payload["n_orb"] ||
                throw(DimensionMismatch("merged n_orb values differ"))
            Int(merged["filling"]) == payload["filling"] ||
                throw(DimensionMismatch("merged filling values differ"))
            Bool(merged["production"]) == payload["production"] ||
                throw(ArgumentError("merged production markers differ"))
        end
        seed_payloads = merged["payloads"]
        curves = reduce(vcat, [
            permutedims(Float64.(seed_payload["survival/real_curve"]))
            for seed_payload in seed_payloads
        ])
        summary = bootstrap_seed_mean(curves;
            n_boot = n_boot,
            rng = MersenneTwister(Int(bootstrap_seed) + index - 1))
        first_seed = merged["payloads"][1]
        key = key
        prefix = "datasets/$key"
        control = endswith(key, "diss_syk")
        payload["$prefix/seed_curves"] = curves
        payload["$prefix/mean"] = summary.mean
        payload["$prefix/lower"] = summary.lower
        payload["$prefix/upper"] = summary.upper
        payload["$prefix/label"] = _cgs_curve_label(first_seed, control)
        payload["$prefix/family"] = String(first_seed["identity/family"])
        payload["$prefix/parameter_name"] =
            String(first_seed["identity/parameter_name"])
        payload["$prefix/parameter"] = Float64(first_seed["identity/parameter"])
        payload["$prefix/delta_0_over_2pi_mhz"] =
            _cgs_source_delta0(first_seed)
        payload["$prefix/control"] = control
        payload["$prefix/line_style"] = control ? "dash" : "solid"
        payload["$prefix/seeds"] = seeds
    end
    payload["seeds"] = all_seeds_equal ? reference_seeds : Int[]
    payload["dataset_seed_counts"] = [
        length(payload["datasets/$(key)/seeds"])
        for key in keys
    ]
    return payload
end

function cgs_write_plotdata(path::AbstractString, payload::AbstractDict)
    output = _cgs_atomic_jld2(path) do file
        stored_keys = sort!(String.(collect(keys(payload))))
        file["schema/keys"] = stored_keys
        for key in stored_keys
            file[key] = payload[key]
        end
    end
    return output
end

function cgs_read_plotdata(path::AbstractString)
    input = abspath(path)
    isfile(input) || throw(ArgumentError("plot data not found: $input"))
    return JLD2.jldopen(input, "r") do file
        haskey(file, "schema/keys") || throw(ArgumentError("plot data lack schema/keys"))
        Dict{String,Any}(key => file[key] for key in String.(file["schema/keys"]))
    end
end

function cgs_write_manifest(path::AbstractString; plotdata::AbstractString)
    payload = cgs_read_plotdata(plotdata)
    manifest = Dict("plotdata" => abspath(plotdata),
                    "dataset_keys" => payload["dataset_keys"])
    mkpath(dirname(path))
    open(path, "w") do io
        TOML.print(io, manifest; sorted=true)
    end
    return path
end

function cgs_validate_manifest(path::AbstractString)
    manifest = TOML.parsefile(path)
    payload = cgs_read_plotdata(manifest["plotdata"])
    payload["dataset_keys"] == manifest["dataset_keys"] ||
        throw(ArgumentError("plot-data datasets do not match manifest"))
    return nothing
end

function _cgs_parse_merge_cli(argv)
    options = Dict{String,String}()
    index = 1
    while index <= length(argv)
        argument = argv[index]
        startswith(argument, "--") || error("unexpected positional argument: $argument")
        key = argument[3:end]
        index += 1
        index <= length(argv) || error("--$key requires a value")
        options[key] = argv[index]
        index += 1
    end
    for key in ("config", "output-root", "mode", "expected-seeds")
        haskey(options, key) || error("missing required option --$key")
    end
    return options
end

function _cgs_parse_seed_range(text::AbstractString)
    matched = match(r"^(\d+):(\d+)$", text)
    matched === nothing && error("expected seeds must use first:last syntax")
    first_seed, last_seed = parse.(Int, matched.captures)
    first_seed > 0 && last_seed >= first_seed || error("invalid expected seed range")
    return first_seed:last_seed
end

function cgs_merge_main(argv = ARGS)
    options = _cgs_parse_merge_cli(argv)
    cfg = cgs_load_config(options["config"]; production = true)
    mode = options["mode"]
    mode in ("smoke", "production") || error("mode must be smoke or production")
    expected_seeds = _cgs_parse_seed_range(options["expected-seeds"])
    mode == "production" && expected_seeds != 1:64 &&
        error("production merge requires seeds 1:64")
    root = abspath(options["output-root"])
    store = get(ENV, "STORE", "")
    isempty(store) && error("STORE must be set for CGS merge")
    startswith(root, abspath(store) * string(Base.Filesystem.path_separator)) ||
        error("merge output root must be beneath STORE")
    specs = cgs_dataset_specs(cfg)
    thresholds = Dict{String,Float64}(
        key => Float64(cfg["validation"][key]) for key in
        ("h_residual_max", "l_residual_max",
         "trace_preservation_residual_max", "f0_error_max",
         "curve_max_imaginary", "curve_probability_tolerance",
         "reconstruction_residual_max"))
    merged_paths = String[]
    for spec in specs
        result = cgs_merge_dataset(
            root, spec.key, expected_seeds, thresholds;
            expected_n_orb = 8, expected_filling = 4,
            production = true)
        push!(merged_paths, result.path)
    end
    if mode == "production"
        length(specs) * length(expected_seeds) == 640 ||
            error("production merge must contain exactly 640 seed payloads")
    end
    plotdata_payload = cgs_build_plotdata(
        merged_paths, getfield.(specs, :key);
        n_boot = Int(cfg["bootstrap"]["n_resamples"]),
        bootstrap_seed = Int(cfg["bootstrap"]["rng_seed"]))

    plotdata = joinpath(root, "plotdata", "cgs_survival_beta0_n8_q4.jld2")
    cgs_write_plotdata(plotdata, plotdata_payload)
    seed_paths = [cgs_seed_path(root, spec.key, seed)
     for spec in specs for seed in expected_seeds]

    manifest = joinpath(root, "manifests",
        mode == "production" ? "production.toml" : "smoke.toml")
    cgs_write_manifest(manifest; plotdata=plotdata)
    cgs_validate_manifest(manifest)
    println("validated $(length(seed_paths)) CGS source seed payloads")
    println("wrote plot data: $plotdata")
    println("wrote manifest: $manifest")
    return (plotdata = plotdata, manifest = manifest, merged_paths = merged_paths)
end

if abspath(PROGRAM_FILE) == @__FILE__
    cgs_merge_main()
end
