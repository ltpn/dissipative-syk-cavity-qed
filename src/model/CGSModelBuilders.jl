# CGSModelBuilders.jl
#
# Include-style model dispatch for the beta-zero CGS supplement. Include the
# established Lamb-Dicke, synthetic-ladder, and multimode sources first.

using JLD2
using Printf
using Random: MersenneTwister
using TOML

struct CGSDatasetSpec
    key::String
    family::Symbol
    parameter_name::Symbol
    parameter::Float64
    delta0_over_2pi_mhz::Float64
    control::Bool
    seed_range::UnitRange{Int}
end

function _cgs_delta0_values(cfg::AbstractDict, family::Symbol)
    section = cfg[family == :figure1 ? "figure1" : "figure3"]

    return [Float64(section["delta_cd_over_2pi_mhz"])]
end

function _cgs_require_equal(actual, expected, label)
    actual == expected || throw(ArgumentError(
        "$label mismatch: got $actual, expected $expected"))
    return nothing
end

function _cgs_require_float(actual, expected, label)
    isapprox(Float64(actual), Float64(expected); rtol = 0, atol = 32eps(Float64)) ||
        throw(ArgumentError("$label mismatch: got $actual, expected $expected"))
    return nothing
end

function cgs_validate_config(cfg::AbstractDict; production::Bool = true)
    for section in ("system", "time_grid", "bootstrap", "figure1",
                    "figure1_control", "figure3", "figure3_control",
                    "validation", "output")
        haskey(cfg, section) || throw(ArgumentError("config lacks [$section]"))
    end
    system = cfg["system"]
    first_seed = Int(system["seed_first"])
    last_seed = Int(system["seed_last"])
    0 < Int(system["filling"]) < Int(system["n_orb"]) ||
        throw(ArgumentError("filling must satisfy 0 < filling < n_orb"))
    first_seed > 0 && last_seed >= first_seed ||
        throw(ArgumentError("invalid seed range $first_seed:$last_seed"))
    _cgs_require_float(system["beta"], 0.0, "beta")

    time = cfg["time_grid"]
    Float64(time["t_min"]) > 0 || throw(ArgumentError("t_min must be positive"))
    Float64(time["t_max"]) > Float64(time["t_min"]) ||
        throw(ArgumentError("t_max must exceed t_min"))
    Int(time["n_positive"]) > 1 ||
        throw(ArgumentError("n_positive must exceed one"))

    bootstrap = cfg["bootstrap"]
    Int(bootstrap["n_resamples"]) > 0 ||
        throw(ArgumentError("bootstrap n_resamples must be positive"))
    0 < Float64(bootstrap["lower_quantile"]) <
        Float64(bootstrap["upper_quantile"]) < 1 ||
        throw(ArgumentError("bootstrap quantiles must be ordered inside (0,1)"))

    if production
        _cgs_require_equal(Int(system["n_orb"]), 8, "n_orb")
        _cgs_require_equal(Int(system["filling"]), 4, "filling")
        _cgs_require_equal(first_seed, 1, "first seed")
        _cgs_require_equal(last_seed, 64, "last seed")
        _cgs_require_equal(Int(time["n_positive"]), 600, "positive time count")
        _cgs_require_float(time["t_min"], 1e-3, "t_min")
        _cgs_require_float(time["t_max"], 1e6, "t_max")

        fig1 = cfg["figure1"]
        _cgs_require_equal(Float64.(fig1["etas"]), [0.1, 0.4, 1.0, 2.0],
                           "Figure-1 eta grid")
        _cgs_require_float(fig1["gamma_eff"], 1.0, "Figure-1 gamma_eff")
        _cgs_require_float(fig1["delta_cd_over_2pi_mhz"], 1.0,
                           "Figure-1 Delta_cd/2pi")

        fig1_control = cfg["figure1_control"]
        _cgs_require_equal(Int(fig1_control["n_random_jumps"]), 60,
                           "Figure-1 control jump count")
        _cgs_require_equal(Int(fig1_control["n_cavity_jumps"]), 1,
                           "Figure-1 control cavity count")

        fig3 = cfg["figure3"]
        _cgs_require_equal(Float64.(fig3["delta_tildes"]),
                           [0.01, 0.1, 1.0, 10.0], "Figure-3 delta-tilde grid")
        # kappa/Delta = 0.2, matching the single-mode column of Table S1.
        _cgs_require_float(fig3["delta_cd_over_2pi_mhz"], 1.0,
                           "Figure-3 Delta_cd/2pi")

        fig3_control = cfg["figure3_control"]
        _cgs_require_float(fig3_control["target_delta_tilde"], 0.01,
                           "Figure-3 control target")
        _cgs_require_equal(Int(fig3_control["n_random_jumps"]), 300,
                           "Figure-3 control jump count")
        _cgs_require_equal(Int(fig3_control["n_cavity_jumps"]), 1,
                           "Figure-3 control cavity count")

    end
    return nothing
end

function cgs_load_config(path::AbstractString; production::Bool = true)
    config_path = abspath(path)
    isfile(config_path) || throw(ArgumentError("config not found: $config_path"))
    config_text = read(config_path, String)
    cfg = TOML.parse(config_text)
    cgs_validate_config(cfg; production = production)
    return cfg
end

function cgs_time_grid(cfg::AbstractDict)
    time = cfg["time_grid"]
    positive = 10.0 .^ range(log10(Float64(time["t_min"])),
                             log10(Float64(time["t_max"]));
                             length = Int(time["n_positive"]))
    return vcat(0.0, positive)
end

function _cgs_parameter_slug(value::Real)
    text = @sprintf("%g", Float64(value))
    return replace(text, "." => "p", "-" => "m")
end

function cgs_dataset_specs(cfg::AbstractDict)
    system = cfg["system"]
    seed_range = Int(system["seed_first"]):Int(system["seed_last"])
    specs = CGSDatasetSpec[]
    for delta0 in _cgs_delta0_values(cfg, :figure1)
        prefix = "fig1"
        for eta in Float64.(cfg["figure1"]["etas"])
            push!(specs, CGSDatasetSpec(
                "$(prefix)_eta_$(_cgs_parameter_slug(eta))",
                :figure1, :eta, eta, delta0, false, seed_range))
        end
        push!(specs, CGSDatasetSpec("$(prefix)_diss_syk", :figure1, :eta,
            Float64(cfg["figure1_control"]["eta"]), delta0, true,
            seed_range))
    end
    for delta0 in _cgs_delta0_values(cfg, :figure3)
        prefix = "fig3"
        for delta_tilde in Float64.(cfg["figure3"]["delta_tildes"])
            push!(specs, CGSDatasetSpec(
                "$(prefix)_dtilde_$(_cgs_parameter_slug(delta_tilde))",
                :figure3, :delta_tilde, delta_tilde, delta0, false,
                seed_range))
        end
        push!(specs, CGSDatasetSpec("$(prefix)_diss_syk", :figure3,
            :delta_tilde,
            Float64(cfg["figure3_control"]["target_delta_tilde"]), delta0,
            true, seed_range))
    end
    return specs
end

function _cgs_load_jld2(path::AbstractString)
    isfile(path) || throw(ArgumentError("calibration not found: $path"))
    return JLD2.jldopen(path, "r") do file
        Dict{String,Any}(String(key) => file[key] for key in keys(file))
    end
end

function _cgs_store_path(cfg::AbstractDict, relative::AbstractString)
    store = get(ENV, "STORE", "")
    isempty(store) && throw(ArgumentError("STORE must be set to load production calibration"))
    return abspath(joinpath(store, relative))
end

function _cgs_figure1_config(cfg::AbstractDict,
                             delta0_over_2pi_mhz::Real)
    section = cfg["figure1"]
    system = cfg["system"]
    return LambDickeLiouvillianConfig(
        n_orb = Int(system["n_orb"]),
        filling = Int(system["filling"]),
        n_grid = Int(section["n_grid"]),
        box_length = Float64(section["box_length"]),
        weight_type = :speckle,
        correlation_length = Float64(section["correlation_length"]),
        disorder_strength = Float64(section["disorder_strength"]),
        gamma_eff = Float64(section["gamma_eff"]),
        svd_tol = Float64(section["svd_tol"]),
        J = Float64(section["J"]),
        lambda_c_micron = Float64(section["lambda_c_micron"]),
        kappa_over_2pi_mhz = Float64(section["kappa_over_2pi_mhz"]),
        delta_cd_over_2pi_mhz = Float64(delta0_over_2pi_mhz),
    )
end

function _cgs_metadata(base::AbstractDict, spec::CGSDatasetSpec, seed::Integer)
    metadata = Dict{String,Any}(String(key) => value for (key, value) in base)
    metadata["dataset_key"] = spec.key
    metadata["family"] = String(spec.family)
    metadata["parameter_name"] = String(spec.parameter_name)
    metadata["parameter"] = spec.parameter
    metadata["delta_0_over_2pi_mhz"] = spec.delta0_over_2pi_mhz
    metadata["control"] = spec.control
    metadata["seed"] = Int(seed)
    return metadata
end

function _cgs_build_figure1_physical(spec::CGSDatasetSpec,
                                     seed::Integer,
                                     cfg::AbstractDict)
    result = build_lamb_dicke_liouvillian(
        _cgs_figure1_config(cfg, spec.delta0_over_2pi_mhz), seed,
        spec.parameter)
    return (L = result.L, H = result.H,
            metadata = _cgs_metadata(result.metadata, spec, seed))
end

function _cgs_multimode_params(cfg::AbstractDict, delta_tilde::Real)
    section = cfg["figure3"]
    system = cfg["system"]
    return MultimodeParams(
        n_orb = Int(system["n_orb"]),
        filling = Int(system["filling"]),
        box_length = Float64(section["box_length"]),
        n_grid = Int(section["n_grid"]),
        zeta = Float64(section["zeta"]),
        mode_cutoff = Int(section["mode_cutoff"]),
        delta_tilde = Float64(delta_tilde),
        weight_type = :speckle,
        speckle_grains_per_side = Float64(section["speckle_grains_per_side"]),
        speckle_correlation_length = nothing,
        disorder_strength = Float64(section["disorder_strength"]),
        drive_wavevector = Float64(section["drive_wavevector"]),
        energy_scale = Float64(section["energy_scale"]),
    )
end

function _cgs_multimode_loss(cfg::AbstractDict,
                             delta0_over_2pi_mhz::Real)
    section = cfg["figure3"]
    return MultimodeCavityLossParams(
        kappa_over_2pi_mhz = Float64(section["kappa_over_2pi_mhz"]),
        delta_cd_over_2pi_mhz = Float64(delta0_over_2pi_mhz),
        gamma_over_2pi_mhz = Float64(section["gamma_over_2pi_mhz"]),
        delta_da_over_2pi_mhz = Float64(section["delta_da_over_2pi_mhz"]),
        rate_scale = Float64(section["rate_scale"]),
    )
end

function _cgs_build_figure3_physical(spec::CGSDatasetSpec,
                                     seed::Integer,
                                     cfg::AbstractDict)
    result = multimode_open_system_block(
        _cgs_multimode_params(cfg, spec.parameter),
        _cgs_multimode_loss(cfg, spec.delta0_over_2pi_mhz);
        seed = seed)
    return (L = result.L, H = result.H,
            metadata = _cgs_metadata(result.metadata, spec, seed))
end

function _cgs_synthetic_config(cfg::AbstractDict,
                               H_tensor,
                               jumps,
                               cavity_op;
                               gamma_eff::Real,
                               cavity_rate_over_J::Real,
                               delta0_over_2pi_mhz::Real)
    system = cfg["system"]
    section = cfg["figure1"]
    return LambDickeLiouvillianConfig(
        n_orb = Int(system["n_orb"]),
        filling = Int(system["filling"]),
        n_grid = Int(section["n_grid"]),
        box_length = Float64(section["box_length"]),
        weight_type = :uniform,
        correlation_length = Float64(section["correlation_length"]),
        disorder_strength = Float64(section["disorder_strength"]),
        gamma_eff = Float64(gamma_eff),
        svd_tol = Float64(section["svd_tol"]),
        J = Float64(section["J"]),
        synthetic_hamiltonian_tensor = H_tensor,
        synthetic_jump_matrices = jumps,
        lambda_c_micron = Float64(section["lambda_c_micron"]),
        kappa_over_2pi_mhz = Float64(section["kappa_over_2pi_mhz"]),
        delta_cd_over_2pi_mhz = Float64(delta0_over_2pi_mhz),
        manual_cavity_loss_rate_over_J = Float64(cavity_rate_over_J),
        synthetic_cavity_loss_operator = cavity_op,
    )
end

function _cgs_validate_ladder_calibration(calibration::AbstractDict,
                                          cfg::AbstractDict)
    n_orb = Int(cfg["system"]["n_orb"])
    filling = Int(cfg["system"]["filling"])
    _cgs_require_equal(Int(calibration["n_orb"]), n_orb,
                       "ladder calibration n_orb")
    _cgs_require_equal(Int(calibration["filling"]), filling,
                       "ladder calibration filling")
    for key in ("sigma_syk4_bdi_span", "sigma_g", "lambda_star_60")
        haskey(calibration, key) || throw(ArgumentError(
            "ladder calibration lacks $key"))
        isfinite(Float64(calibration[key])) && Float64(calibration[key]) > 0 ||
            throw(ArgumentError("ladder calibration $key must be positive"))
    end
    return nothing
end

function _cgs_build_control(spec::CGSDatasetSpec,
                            seed::Integer,
                            cfg::AbstractDict,
                            calibration::AbstractDict;
                            figure3::Bool)
    system = cfg["system"]
    control = cfg[figure3 ? "figure3_control" : "figure1_control"]
    _cgs_validate_ladder_calibration(calibration, cfg)
    n_orb = Int(system["n_orb"])
    n_random_jumps = Int(control["n_random_jumps"])
    offset = Int(control["seed_offset"])
    rate_scale = figure3 ? Float64(calibration["figure3_dissipator_rate_scale"]) : 1.0
    if figure3
        _cgs_require_float(calibration["target_delta_tilde"],
                           control["target_delta_tilde"],
                           "Figure-3 calibration target")
        _cgs_require_equal(Int(calibration["n_random_jumps"]), n_random_jumps,
                           "Figure-3 calibration random jumps")
        _cgs_require_equal(Int(calibration["n_cavity_jumps"]), 1,
                           "Figure-3 calibration cavity jumps")
        isfinite(rate_scale) && rate_scale > 0 ||
            throw(ArgumentError("Figure-3 dissipator scale must be positive"))

    end

    cavity_rate = figure3 ?
        Float64(control["cavity_loss_rate_over_J"]) * rate_scale :
        Float64(cfg["figure1"]["kappa_over_2pi_mhz"]) /
            spec.delta0_over_2pi_mhz

    rng_H = MersenneTwister(offset + seed)
    rng_jump = MersenneTwister(offset + seed + 1_000_000)
    rng_cavity = MersenneTwister(offset + seed + 2_000_000)
    H_tensor = random_real_syk4_tensor(
        rng_H, n_orb, Float64(calibration["sigma_syk4_bdi_span"]))
    jumps = random_symmetric_jumps_equal_weight(
        rng_jump, n_orb, n_random_jumps,
        sqrt(Float64(calibration["lambda_star_60"])))
    cavity_op = random_symmetric_cavity_op(
        rng_cavity, n_orb, Float64(calibration["sigma_g"]))
    ldcfg = _cgs_synthetic_config(cfg, H_tensor, jumps, cavity_op;
        gamma_eff = Float64(control["gamma_eff"]) * rate_scale,
        cavity_rate_over_J = cavity_rate,
        delta0_over_2pi_mhz = spec.delta0_over_2pi_mhz)
    result = build_lamb_dicke_liouvillian(
        ldcfg, seed, Float64(control["eta"]))
    metadata = _cgs_metadata(result.metadata, spec, seed)
    metadata["n_random_jumps"] = n_random_jumps
    metadata["n_cavity_jumps"] = 1
    metadata["seed_offset"] = offset
    metadata["dissipator_rate_scale"] = rate_scale
    return (L = result.L, H = result.H, metadata = metadata)
end

function cgs_build_model(spec::CGSDatasetSpec,
                         seed::Integer,
                         cfg::AbstractDict;
                         ladder_calibration = nothing,
                         figure3_calibration = nothing)
    seed in spec.seed_range || throw(ArgumentError(
        "seed $seed is outside $(spec.seed_range) for $(spec.key)"))
    build = if !spec.control
        spec.family == :figure1 ?
            _cgs_build_figure1_physical(spec, seed, cfg) :
            _cgs_build_figure3_physical(spec, seed, cfg)
    elseif spec.family == :figure1
        calibration = ladder_calibration === nothing ? _cgs_load_jld2(
            _cgs_store_path(cfg,
                String(cfg["figure1_control"]["calibration_store_relative"]))) :
            ladder_calibration
        _cgs_build_control(spec, seed, cfg, calibration; figure3 = false)
    else
        calibration = figure3_calibration === nothing ? _cgs_load_jld2(
            joinpath(_cgs_store_path(cfg, String(cfg["output"]["store_subdir"])),
                     String(cfg["figure3_control"]["calibration_relative"]))) :
            figure3_calibration
        _cgs_build_control(spec, seed, cfg, calibration; figure3 = true)
    end
    return build
end
