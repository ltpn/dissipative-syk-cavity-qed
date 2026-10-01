#!/usr/bin/env julia

# Raw, unit-weight connected sigma-SFF and theta=pi/4 DSFF supplement.
# The numerical inputs are stored N=10, filling-3 spectra; this driver never
# rebuilds a Liouvillian or recomputes an eigendecomposition/SVD.

using JLD2
using CairoMakie
using LaTeXStrings
using Printf
using Random: MersenneTwister
using TOML
include(joinpath(@__DIR__, "..", "..", "..", "DatasetPaths.jl"))

const HERE = @__DIR__
include(joinpath(HERE, "_figure_export.jl"))
const LAMB_DIR = abspath(joinpath(HERE, "..", "..", ".."))
const REPO_ROOT = abspath(joinpath(LAMB_DIR, "..", ".."))
const FIGURE_ROOT = joinpath(
    REPO_ROOT, "figures")

include(joinpath(LAMB_DIR, "atomic_jld2.jl"))
const _atomic_jld2 = atomic_jld2
include(joinpath(LAMB_DIR, "RawConnectedFormFactors.jl"))
include(joinpath(LAMB_DIR, "ComplexDSFFPlotData.jl"))
using .RawConnectedFormFactors

const FIG1_KEYS = (
    "eta_0p1", "eta_0p4", "eta_1p0", "eta_2p0", "fig1_syk")
const FIG3_KEYS = (
    "dtilde_0p01", "dtilde_0p1", "dtilde_1", "dtilde_10", "fig3_syk")
const ALL_DATASET_KEYS = (FIG1_KEYS..., FIG3_KEYS...)
const DEFAULT_TIME_GRID = 10.0 .^ range(-3, 6; length = 400)
const THETA = pi / 4
const STEADY_TOLERANCE = 1e-8
const N_BOOTSTRAP = 500
# Paper protocol: N = 10. Filling and ensemble size come from the source
# manifests, which must agree with each other (validate_source_manifests).
const EXPECTED_N_ORB = 10
const BOOTSTRAP_SEED_BASE = 2_026_081_800
const FIG1_CONTROL_RGB = [0.65, 0.0, 0.0]
const AUTHORING_CANVAS_SIZE_PT = (1202.0, 380.0)
const INTENDED_LATEX_SCALE = 0.5
const PANEL_X_MIN = [1.0, 1.0, 10.0, 10.0]
const LEGEND_FONT_SIZE_PT = 6.5
const PANEL_ORDER = ["fig1_sigma", "fig1_dsff", "fig3_sigma", "fig3_dsff"]
const LEGEND_LABELS = [
    "0.1", "0.4", "1", "2", "diss. SYK",
    "0.01", "0.1", "1", "10", "diss. SYK",
]
const DEFAULT_FIGURE1_MANIFEST = joinpath(FIGURE_ROOT, "figure_1__manifest.toml")
const DEFAULT_FIGURE3_MANIFEST = joinpath(FIGURE_ROOT, "figure_3__manifest.toml")

pt2px(points::Real) = points * 4 / 3
source_px(points::Real) = pt2px(points / INTENDED_LATEX_SCALE)

function theme_magiclatex(;
        FontSize::Real = 16 * 4 / 3,
        LabelFontSizeMultiplier::Real = 1.1,
        TitleFontSizeMultiplier::Real = 1.1,
        LegendFontSizeMultiplier::Real = 0.9,
        FigureSize = pt2px.((560, 420)),
        PaletteName::Union{Symbol,String} = :gem_2024)
    label_size = FontSize * LabelFontSizeMultiplier
    title_size = FontSize * TitleFontSizeMultiplier
    legend_size = FontSize * LegendFontSizeMultiplier
    return merge(
        Makie.theme_latexfonts(),
        Theme(
            size = FigureSize,
            fontsize = FontSize,
            backgroundcolor = :transparent,
            Axis = (
                xlabelsize = label_size,
                ylabelsize = label_size,
                titlesize = title_size,
                titlefont = :regular,
                spinewidth = pt2px(1.0),
                xtickalign = 1,
                ytickalign = 1,
                xticksize = pt2px(5.0),
                yticksize = pt2px(5.0),
                xtickwidth = pt2px(1.0),
                ytickwidth = pt2px(1.0),
                xticksmirrored = true,
                yticksmirrored = true,
            ),
            Legend = (
                fontsize = legend_size,
                titlesize = legend_size,
                framewidth = pt2px(1.0),
                titlefont = :regular,
            ),
        ),
    )
end

function magic_log_ticks(powers)
    positions = 10.0 .^ collect(powers)
    labels = [latexstring("10^{", power, "}") for power in powers]
    return positions, labels
end

struct DatasetSpec
    key::String
    model::Symbol
    parameter::Float64
    parameter_label::String
    control::Bool
    storage::Symbol
    source::String
    config_name::String
    gamma::Float64
    tolerance::Float64
end

_fmt(x::Real) = string(Float64(x))

function _individual_path(
        dir::AbstractString, config::AbstractString, eta::Real, gamma::Real,
        tolerance::Real, seed::Integer)
    filename = @sprintf(
        "%s__eta=%s__gamma=%s__tol=%s__seed=%d.jld2",
        config, _fmt(eta), _fmt(gamma), _fmt(tolerance), seed)
    return joinpath(dir, filename)
end

"Load seed-resolved centered singular values and complex spectra."
function load_individual_dataset(
        dir::AbstractString, config::AbstractString, eta::Real, gamma::Real,
        tolerance::Real, seeds)
    requested = Int.(collect(seeds))
    sigmas = Vector{Vector{Float64}}()
    spectra = Vector{Vector{ComplexF64}}()
    paths = String[]
    for seed in requested
        path = _individual_path(dir, config, eta, gamma, tolerance, seed)
        isfile(path) || throw(ArgumentError("stored spectrum not found: $path"))
        JLD2.jldopen(path, "r") do file
            haskey(file, "L_svd_S_centered") || throw(ArgumentError(
                "stored spectrum lacks L_svd_S_centered: $path"))
            haskey(file, "L_eigvals") || throw(ArgumentError(
                "stored spectrum lacks L_eigvals: $path"))
            push!(sigmas, Float64.(file["L_svd_S_centered"]))
            push!(spectra, ComplexF64.(file["L_eigvals"]))
        end
        push!(paths, abspath(path))
    end
    return (sigmas = sigmas, spectra = spectra, seeds = requested, paths = paths)
end

"Load requested seed identities from one merged Figure-3 cache."
function load_merged_dataset(path::AbstractString, seeds)
    isfile(path) || throw(ArgumentError("merged spectrum cache not found: $path"))
    requested = Int.(collect(seeds))
    return JLD2.jldopen(path, "r") do file
        for key in ("seeds", "sigmas", "L_eigvals")
            haskey(file, key) ||
                throw(ArgumentError("merged cache lacks $key: $path"))
        end
        stored_seeds = Int.(file["seeds"])
        length(unique(stored_seeds)) == length(stored_seeds) ||
            throw(ArgumentError("merged cache contains duplicate seeds: $path"))
        stored_sigmas = file["sigmas"]
        stored_spectra = file["L_eigvals"]
        length(stored_seeds) == length(stored_sigmas) == length(stored_spectra) ||
            throw(DimensionMismatch("merged cache arrays have different lengths"))
        positions = Dict(seed => i for (i, seed) in enumerate(stored_seeds))
        missing = filter(seed -> !haskey(positions, seed), requested)
        isempty(missing) || throw(ArgumentError(
            "merged cache is missing requested seeds $(missing): $path"))
        indices = [positions[seed] for seed in requested]
        return (
            sigmas = [Float64.(stored_sigmas[i]) for i in indices],
            spectra = [ComplexF64.(stored_spectra[i]) for i in indices],
            seeds = requested,
            paths = [abspath(path)],
        )
    end
end

function validate_dataset_counts(
        data; expected_seeds,
        expected_dimension::Integer,
        production::Bool = true)
    requested = Int.(collect(expected_seeds))
    data.seeds == requested || throw(ArgumentError(
        "seed identity mismatch: got $(data.seeds), expected $requested"))
    length(data.sigmas) == length(requested) == length(data.spectra) ||
        throw(DimensionMismatch("dataset seed arrays have different lengths"))
    for i in eachindex(requested)
        length(data.sigmas[i]) == expected_dimension || throw(ArgumentError(
            "seed $(requested[i]) has $(length(data.sigmas[i])) singular values; " *
            "expected $expected_dimension"))
        length(data.spectra[i]) == expected_dimension || throw(ArgumentError(
            "seed $(requested[i]) has $(length(data.spectra[i])) eigenvalues; " *
            "expected $expected_dimension"))
        all(isfinite, data.sigmas[i]) || throw(ArgumentError(
            "seed $(requested[i]) has non-finite singular values"))
        all(isfinite, data.spectra[i]) || throw(ArgumentError(
            "seed $(requested[i]) has non-finite eigenvalues"))
        if production
            removed = count(z -> abs(z) <= STEADY_TOLERANCE, data.spectra[i])
            removed == 1 || throw(ArgumentError(
                "seed $(requested[i]) has $removed steady modes; expected exactly one"))
        end
    end
    return nothing
end

function _require_equal(actual, expected, label)
    actual == expected ||
        throw(ArgumentError("$label mismatch: got $actual, expected $expected"))
    return nothing
end

function _require_float(actual, expected, label)
    isapprox(Float64(actual), Float64(expected); rtol = 0, atol = 32eps(Float64)) ||
        throw(ArgumentError("$label mismatch: got $actual, expected $expected"))
    return nothing
end

"""
Validate the physics and ensemble metadata locked by the approved design, and
return the shared (filling, n_seeds). N, the eta and delta-tilde grids, Delta_cd,
and the m300 reference (300 random channels plus one cavity channel) are locked;
filling and seed count are free but must agree across every source ensemble.
"""
function validate_source_manifests(fig1, fig3; smoke::Bool = false)
    filling = Int(fig1["filling"])
    n_seeds = Int(fig1["n_seeds"])
    n_seeds > 0 || throw(ArgumentError("Figure 1 seed count must be positive"))
    for (name, manifest) in (("Figure 1", fig1), ("Figure 3", fig3))
        _require_equal(Int(manifest["n_orb"]), EXPECTED_N_ORB, "$name n_orb")
        _require_equal(Int(manifest["filling"]), filling, "$name filling")
        _require_equal(Int(manifest["n_seeds"]), n_seeds, "$name source seed count")
    end
    _require_equal(Float64.(fig1["etas"]), [0.1, 0.4, 1.0, 2.0],
                   "Figure 1 eta grid")
    _require_float(fig1["delta_cd_over_2pi_mhz"], 1.0,
                   "Figure 1 Delta_cd/2pi")
    _require_equal(Int.(fig1["seed_range"]), [1, n_seeds],
                   "Figure 1 seed range")
    l3b = fig1["l2_overlay"]
    _require_equal(Int.(l3b["seed_range"]), [1, n_seeds],
                   "Figure 1 L3b seed range")
    _require_equal(Int(l3b["n_seeds"]), n_seeds, "Figure 1 L3b seed count")

    _require_equal(Float64.(fig3["delta_tildes"]), [0.01, 0.1, 1.0, 10.0],
                   "Figure 3 delta-tilde grid")
    _require_float(fig3["delta_cd_over_2pi_mhz"], 1.0,
                   "Figure 3 Delta_cd/2pi")
    _require_equal(length(fig3["cache_paths"]), 4, "Figure 3 cache count")
    syk = fig3["syk4_diss_reference"]
    _require_equal(Int(syk["n_random_jumps"]), 300,
                   "Figure 3 random-jump count")
    _require_equal(Int(syk["n_cavity_jumps"]), 1,
                   "Figure 3 cavity-jump count")
    _require_float(syk["target_delta_tilde"], 0.01,
                   "Figure 3 SYK target delta-tilde")
    _require_equal(Int.(syk["seed_range"]), [1, n_seeds],
                   "Figure 3 SYK seed range")
    _require_equal(Int(syk["n_seeds_used"]), n_seeds,
                   "Figure 3 SYK seed count")
    smoke # deliberately does not relax source-manifest locks
    return (filling = filling, n_seeds = n_seeds)
end

"Resolve the ten model/control datasets exclusively from the source manifests."
function resolve_dataset_specs(fig1, fig3)
    specs = DatasetSpec[]
    for (key, eta) in zip(FIG1_KEYS[1:4], Float64.(fig1["etas"]))
        push!(specs, DatasetSpec(
            key, :fig1, eta, @sprintf("%g", eta), false, :individual,
            abspath(String(fig1["spectra_dir"])), "physical",
            Float64(fig1["delta_cd_over_2pi_mhz"]), Float64(fig1["svd_tol"])))
    end
    l3b = fig1["l2_overlay"]
    push!(specs, DatasetSpec(
        FIG1_KEYS[5], :fig1, Float64(l3b["eta"]), "diss. SYK", true,
        :individual, abspath(String(l3b["spectra_dir"])),
        String(l3b["config_name"]), Float64(l3b["gamma"]),
        Float64(l3b["svd_tol"])))

    for (key, value, path) in zip(
            FIG3_KEYS[1:4], Float64.(fig3["delta_tildes"]),
            String.(fig3["cache_paths"]))
        push!(specs, DatasetSpec(
            key, :fig3, value, @sprintf("%g", value), false, :merged,
            abspath(path), "multimode", Float64(fig3["delta_cd_over_2pi_mhz"]),
            NaN))
    end
    syk = fig3["syk4_diss_reference"]
    push!(specs, DatasetSpec(
        FIG3_KEYS[5], :fig3, Float64(syk["eta"]), "diss. SYK", true,
        :individual, abspath(String(syk["spectra_dir"])),
        String(syk["config_name"]), Float64(syk["gamma"]),
        Float64(syk["svd_tol"])))
    return specs
end

function load_dataset(spec::DatasetSpec, seeds)
    if spec.storage == :individual
        return load_individual_dataset(
            spec.source, spec.config_name, spec.parameter, spec.gamma,
            spec.tolerance, seeds)
    elseif spec.storage == :merged
        return load_merged_dataset(spec.source, seeds)
    end
    throw(ArgumentError("unknown dataset storage kind $(spec.storage)"))
end

function analyze_loaded_dataset(data, times, index::Integer; n_boot::Integer)
    sigma = analyze_sigma_sff(
        data.sigmas, times; n_boot = n_boot,
        rng = MersenneTwister(BOOTSTRAP_SEED_BASE + 2index - 1))
    dsff = analyze_dsff(
        data.spectra, times; theta = THETA,
        steady_tolerance = STEADY_TOLERANCE, n_boot = n_boot,
        rng = MersenneTwister(BOOTSTRAP_SEED_BASE + 2index))
    sigma_summary = merge(sigma, (
        omitted_nonpositive = count(
            x -> !(isfinite(x) && x > 0), sigma.curve),
    ))
    dsff_summary = merge(dsff, (
        omitted_nonpositive = count(
            x -> !(isfinite(x) && x > 0), dsff.curve),
    ))
    return (sigma = sigma_summary, dsff = dsff_summary)
end

function _write_parameters(file, payload)
    for field in (:dataset_keys, :times, :seeds, :n_orb, :filling, :theta,
                  :steady_tolerance, :n_bootstrap, :smoke,
                  :figure1_palette, :figure3_palette,
                  :figure1_control_rgb, :figure3_control_rgb)
        file[String(field)] = getproperty(payload, field)
    end
end

"Write compact curve data without bootstrap samples or raw spectra."
function write_plotdata(path::AbstractString, payload)
    return _atomic_jld2(path) do file
        _write_parameters(file, payload)
        for key in payload.dataset_keys
            dataset = payload.datasets[key]
            for channel_name in (:sigma, :dsff)
                channel = getproperty(dataset, channel_name)
                prefix = "datasets/$key/$(String(channel_name))"
                for field in (
                        :curve, :median, :lower, :upper, :plateau,
                        :plateau_by_seed, :level_counts, :omitted_nonpositive)
                    file["$prefix/$(String(field))"] = getproperty(channel, field)
                end
                if channel_name == :dsff
                    file["$prefix/removed_counts"] = channel.removed_counts
                end
            end
        end
    end
end

function _flatten_counts(payload, channel::Symbol, field::Symbol)
    Int[
        value
        for key in payload.dataset_keys
        for value in getproperty(getproperty(payload.datasets[key], channel), field)
    ]
end

function _manifest_dict(payload, plotdata::AbstractString; outputs)
    output_paths = abspath.(String.(outputs))
    return Dict{String,Any}(
        "plotdata" => abspath(plotdata),
        "dataset_keys" => String.(payload.dataset_keys),
        "times" => Float64.(payload.times),
        "seeds" => Int.(payload.seeds),
        "n_orb" => Int(payload.n_orb),
        "filling" => Int(payload.filling),
        "theta" => Float64(payload.theta),
        "steady_tolerance" => Float64(payload.steady_tolerance),
        "n_bootstrap" => Int(payload.n_bootstrap),
        "smoke" => Bool(payload.smoke),
        "figure1_palette" => collect(payload.figure1_palette),
        "figure3_palette" => [collect(rgb) for rgb in payload.figure3_palette],
        "figure1_control_rgb" => collect(payload.figure1_control_rgb),
        "figure3_control_rgb" => collect(payload.figure3_control_rgb),
        "n_seeds_by_dataset" => [
            length(payload.datasets[key].sigma.level_counts)
            for key in payload.dataset_keys
        ],
        "sigma_level_counts_flat" =>
            _flatten_counts(payload, :sigma, :level_counts),
        "dsff_level_counts_flat" =>
            _flatten_counts(payload, :dsff, :level_counts),
        "dsff_removed_counts_flat" =>
            _flatten_counts(payload, :dsff, :removed_counts),
        "omitted_nonpositive_sigma" => Dict(
            key => Int(payload.datasets[key].sigma.omitted_nonpositive)
            for key in payload.dataset_keys),
        "omitted_nonpositive_dsff" => Dict(
            key => Int(payload.datasets[key].dsff.omitted_nonpositive)
            for key in payload.dataset_keys),
        "outputs" => output_paths,
    )
end

function write_manifest(
        path::AbstractString, payload, plotdata::AbstractString; outputs)
    manifest = _manifest_dict(payload, plotdata; outputs = outputs)
    mkpath(dirname(path))
    tmp = path * ".tmp.$(getpid())"
    try
        open(tmp, "w") do io
            DatasetPaths.print_manifest(io, manifest; sorted=true, path=path)
        end
        mv(tmp, path; force = true)
    finally
        isfile(tmp) && rm(tmp; force = true)
    end
    return path
end

"Validate every numerical parameter repeated in TOML."
function validate_plotdata_binding(path::AbstractString, parameters)
    isfile(path) || throw(ArgumentError("plot data not found: $path"))
    repeated = ("dataset_keys", "times", "seeds", "n_orb", "filling", "theta",
                "steady_tolerance", "n_bootstrap", "smoke", "figure1_palette",
                "figure3_palette", "figure1_control_rgb", "figure3_control_rgb")
    JLD2.jldopen(path, "r") do file
        for field in repeated
            haskey(parameters, field) ||
                throw(ArgumentError("manifest is missing $field"))
            haskey(file, field) ||
                throw(ArgumentError("plot data is missing $field"))
            file[field] == parameters[field] ||
                throw(ArgumentError("plot-data parameters mismatch for $field"))
        end
    end
    return nothing
end

function _read_channel(file, dataset_key::AbstractString, channel::Symbol)
    prefix = "datasets/$dataset_key/$(String(channel))"
    fields = (
        :curve, :median, :lower, :upper, :plateau, :plateau_by_seed,
        :level_counts, :omitted_nonpositive)
    values = (; (
        field => file["$prefix/$(String(field))"] for field in fields)...)
    if channel == :dsff
        return merge(values, (removed_counts = file["$prefix/removed_counts"],))
    end
    return values
end

"Read the compact plot cache into renderer-ready named tuples."
function read_plotdata(path::AbstractString)
    isfile(path) || throw(ArgumentError("plot data not found: $path"))
    return JLD2.jldopen(path, "r") do file
        dataset_keys = String.(file["dataset_keys"])
        datasets = Dict{String,Any}(
            key => (
                sigma = _read_channel(file, key, :sigma),
                dsff = _read_channel(file, key, :dsff),
            ) for key in dataset_keys)
        return (
            dataset_keys = dataset_keys,
            datasets = datasets,
            times = Float64.(file["times"]),
            seeds = Int.(file["seeds"]),
            n_orb = Int(file["n_orb"]),
            filling = Int(file["filling"]),
            theta = Float64(file["theta"]),
            steady_tolerance = Float64(file["steady_tolerance"]),
            n_bootstrap = Int(file["n_bootstrap"]),
            smoke = Bool(file["smoke"]),
            figure1_palette = Float64.(file["figure1_palette"]),
            figure3_palette = [Float64.(rgb) for rgb in file["figure3_palette"]],
            figure1_control_rgb = Float64.(file["figure1_control_rgb"]),
            figure3_control_rgb = Float64.(file["figure3_control_rgb"]),
        )
    end
end

function _rgb(values)
    length(values) == 3 || throw(ArgumentError("RGB colors need three components"))
    return RGBf(Float32(values[1]), Float32(values[2]), Float32(values[3]))
end

function _draw_raw_curve!(
        axis, times, channel, color;
        reference::Bool = false, x_min::Real = first(times))
    display_mask = times .>= x_min
    any(display_mask) || throw(ArgumentError("display window contains no times"))
    display_times = times[display_mask]
    lower = positive_or_nan(channel.lower[display_mask])
    upper = positive_or_nan(channel.upper[display_mask])
    valid_band = @. isfinite(lower) & isfinite(upper) & (upper >= lower)
    lower[.!valid_band] .= NaN
    upper[.!valid_band] .= NaN
    band!(
        axis, display_times, lower, upper;
        color = (color, reference ? 0.12 : 0.14))
    curve = positive_or_nan(channel.curve[display_mask])
    return lines!(
        axis, display_times, curve; color = color,
        linewidth = source_px(reference ? 1.35 : 1.15))
end

"Render the bound four-panel raw-form-factor cache to PDF and 288-dpi PNG."
function render_figure(
        plotdata_path::AbstractString, manifest_path::AbstractString,
        output_dir::AbstractString; suffix::AbstractString = "")
    isfile(manifest_path) || throw(ArgumentError("manifest not found: $manifest_path"))
    parameters = DatasetPaths.read_manifest(manifest_path)
    validate_plotdata_binding(plotdata_path, parameters)
    payload = read_plotdata(plotdata_path)
    payload.dataset_keys == collect(ALL_DATASET_KEYS) || throw(ArgumentError(
        "unexpected dataset order in plot data: $(payload.dataset_keys)"))
    fig1_colors = [cgrad(:viridis)[position] for position in payload.figure1_palette]
    fig1_control = _rgb(payload.figure1_control_rgb)
    fig3_colors = _rgb.(payload.figure3_palette)
    fig3_control = _rgb(payload.figure3_control_rgb)
    panel_specs = [
        (model = :fig1, channel = :sigma,
         panel = L"\mathrm{(a)}", heading = L"\sigma\mathrm{-SFF}"),
        (model = :fig1, channel = :dsff,
         panel = L"\mathrm{(b)}", heading = L"\mathrm{DSFF},\ \theta=\pi/4"),
        (model = :fig3, channel = :sigma,
         panel = L"\mathrm{(c)}", heading = L"\sigma\mathrm{-SFF}"),
        (model = :fig3, channel = :dsff,
         panel = L"\mathrm{(d)}", heading = L"\mathrm{DSFF},\ \theta=\pi/4"),
    ]

    CairoMakie.activate!()
    base_theme = theme_magiclatex(
        PaletteName = :gem_2024,
        FigureSize = pt2px.(AUTHORING_CANVAS_SIZE_PT),
    )
    compact_theme = merge(base_theme, Theme(
        figure_padding = source_px.((2.0, 8.0, 1.5, 1.5)),
    ))
    fig = with_theme(compact_theme) do
        figure = Figure(backgroundcolor = :white)
        group_handles = Dict{Symbol,Vector{Any}}()
        axes = Axis[]
        for (panel_index, spec) in enumerate(panel_specs)
            tick_powers = spec.model == :fig1 ? (0, 3, 6) : (1, 3, 6)
            axis = Axis(
                figure[1, panel_index];
                aspect = 1.15,
                xscale = log10,
                yscale = log10,
                xgridvisible = false,
                ygridvisible = false,
                xticks = magic_log_ticks(tick_powers),
                xlabel = L"tJ",
                ylabel = panel_index == 1 ? L"K_{\mathrm{c}}/K_{\infty}" : "",
                title = spec.heading,
                titlesize = source_px(6.5),
                xlabelpadding = source_px(1.5),
                ylabelpadding = source_px(1.5),
                titlegap = source_px(0.5),
                xticklabelpad = source_px(1.3),
                yticklabelpad = source_px(1.3),
            )
            push!(axes, axis)
            keys = spec.model == :fig1 ? FIG1_KEYS : FIG3_KEYS
            colors = spec.model == :fig1 ? fig1_colors : fig3_colors
            control = spec.model == :fig1 ? fig1_control : fig3_control
            handles = Any[]
            for i in 1:4
                channel = getproperty(payload.datasets[keys[i]], spec.channel)
                push!(handles, _draw_raw_curve!(
                    axis, payload.times, channel, colors[i];
                    x_min = PANEL_X_MIN[panel_index]))
            end
            control_channel = getproperty(payload.datasets[keys[5]], spec.channel)
            push!(handles, _draw_raw_curve!(
                axis, payload.times, control_channel, control;
                reference = true, x_min = PANEL_X_MIN[panel_index]))
            hlines!(
                axis, [1.0]; color = :gray45, linestyle = :dot,
                linewidth = source_px(0.7))
            xlims!(axis, PANEL_X_MIN[panel_index], last(payload.times))
            text!(
                axis, 0.012, 0.985;
                text = spec.panel,
                space = :relative, align = (:left, :top), color = :black)
            if !haskey(group_handles, spec.model)
                group_handles[spec.model] = handles
            end
        end

        axislegend(
            axes[1], group_handles[:fig1], LEGEND_LABELS[1:5],
            L"\eta=";
            position = :rb, orientation = :horizontal, nbanks = 3,
            framevisible = false, backgroundcolor = (:white, 0.78),
            labelsize = source_px(LEGEND_FONT_SIZE_PT),
            titlesize = source_px(LEGEND_FONT_SIZE_PT),
            patchsize = source_px.((6.0, 2.0)),
            padding = source_px.((0.5, 0.5, 0.5, 0.5)),
            rowgap = source_px(0.2), colgap = source_px(0.8),
            patchlabelgap = source_px(0.5), titlegap = source_px(0.6),
        )
        axislegend(
            axes[3], group_handles[:fig3], LEGEND_LABELS[6:10],
            L"\delta\tilde{\omega}=";
            position = :lb, orientation = :horizontal, nbanks = 3,
            framevisible = false, backgroundcolor = (:white, 0.78),
            labelsize = source_px(LEGEND_FONT_SIZE_PT),
            titlesize = source_px(LEGEND_FONT_SIZE_PT),
            patchsize = source_px.((6.0, 2.0)),
            padding = source_px.((0.5, 0.5, 0.5, 0.5)),
            rowgap = source_px(0.2), colgap = source_px(0.8),
            patchlabelgap = source_px(0.5), titlegap = source_px(0.6),
        )
        colgap!(figure.layout, source_px(0.5))
        for column in 1:4
            colsize!(figure.layout, column, Relative(0.25))
        end
        figure
    end

    mkpath(output_dir)
    base = "supplement_raw_sigma_sff_dsff$(suffix)"
    pdf_path = joinpath(output_dir, "$base.pdf")
    png_path = joinpath(output_dir, "$base.png")
    mktempdir() do temporary_dir
        uncropped_pdf = joinpath(temporary_dir, "uncropped.pdf")
        save(uncropped_pdf, fig)
        crop_pdf_hires(uncropped_pdf, pdf_path)
    end
    rasterize_pdf_png(pdf_path, png_path)
    return (
        pdf = pdf_path,
        png = png_path,
        panel_order = copy(PANEL_ORDER),
        legend_labels = copy(LEGEND_LABELS),
    )
end

function _parse_cli(argv)
    smoke = false
    build_plotdata_only = false
    render_only = false
    help = false
    figure1_manifest = DEFAULT_FIGURE1_MANIFEST
    figure3_manifest = DEFAULT_FIGURE3_MANIFEST
    output_dir = FIGURE_ROOT
    i = 1
    while i <= length(argv)
        argument = argv[i]
        if argument == "--smoke"
            smoke = true
        elseif argument == "--build-plotdata-only"
            build_plotdata_only = true
        elseif argument == "--render-only"
            render_only = true
        elseif argument in ("--figure1-manifest", "--figure3-manifest", "--output-dir")
            i < length(argv) || throw(ArgumentError("$argument requires a value"))
            value = String(argv[i + 1])
            if argument == "--figure1-manifest"
                figure1_manifest = value
            elseif argument == "--figure3-manifest"
                figure3_manifest = value
            else
                output_dir = value
            end
            i += 1
        elseif argument in ("--help", "-h")
            help = true
        else
            throw(ArgumentError("unknown argument: $argument"))
        end
        i += 1
    end
    build_plotdata_only && render_only && throw(ArgumentError(
        "--build-plotdata-only and --render-only are mutually exclusive"))
    return (
        smoke = smoke,
        build_plotdata_only = build_plotdata_only,
        render_only = render_only,
        help = help,
        figure1_manifest = figure1_manifest,
        figure3_manifest = figure3_manifest,
        output_dir = output_dir,
    )
end

function _usage(io::IO = stdout)
    println(io, "Usage: supplement_raw_sigma_sff_dsff.jl [options]")
    println(io, "  --smoke                 use seeds 1:2, 40 times, 20 bootstraps")
    println(io, "  --build-plotdata-only   write bound JLD2/TOML without rendering")
    println(io, "  --render-only           render an existing bound JLD2/TOML pair")
    println(io, "  --figure1-manifest PATH override the Figure-1 source manifest")
    println(io, "  --figure3-manifest PATH override the Figure-3 source manifest")
    println(io, "  --output-dir PATH       output directory")
end

function _artifact_paths(output_dir::AbstractString; smoke::Bool)
    suffix = smoke ? "__smoke" : ""
    base = "supplement_raw_sigma_sff_dsff$suffix"
    return (
        suffix = suffix,
        plotdata = joinpath(output_dir, "$(base)__plotdata.jld2"),
        manifest = joinpath(output_dir, "$(base)__manifest.toml"),
        pdf = joinpath(output_dir, "$base.pdf"),
        png = joinpath(output_dir, "$base.png"),
    )
end

"Build raw curves from the manifest-resolved stored N=10 spectra."
function build_payload(
        figure1_manifest_path::AbstractString,
        figure3_manifest_path::AbstractString;
        smoke::Bool = false)
    for path in (figure1_manifest_path, figure3_manifest_path)
        isfile(path) || throw(ArgumentError("source manifest not found: $path"))
    end
    figure1_path = abspath(figure1_manifest_path)
    figure3_path = abspath(figure3_manifest_path)
    figure1 = DatasetPaths.read_manifest(figure1_path)
    figure3 = DatasetPaths.read_manifest(figure3_path)
    ensemble = validate_source_manifests(figure1, figure3; smoke = smoke)
    expected_dimension = binomial(EXPECTED_N_ORB, ensemble.filling)^2
    specs = resolve_dataset_specs(figure1, figure3)
    length(specs) == length(ALL_DATASET_KEYS) || throw(ArgumentError(
        "resolved $(length(specs)) datasets; expected $(length(ALL_DATASET_KEYS))"))
    [spec.key for spec in specs] == collect(ALL_DATASET_KEYS) || throw(
        ArgumentError("resolved dataset order does not match the scientific lock"))

    seeds = smoke ? collect(1:2) : collect(1:ensemble.n_seeds)
    times = smoke ? 10.0 .^ range(-3, 6; length = 40) : DEFAULT_TIME_GRID
    n_boot = smoke ? 20 : N_BOOTSTRAP
    datasets = Dict{String,Any}()
    for (index, spec) in enumerate(specs)
        @printf("[%2d/%2d] loading and analyzing %-14s\n",
                index, length(specs), spec.key)
        flush(stdout)
        data = load_dataset(spec, seeds)
        validate_dataset_counts(
            data; expected_seeds = seeds,
            expected_dimension = expected_dimension,
            production = true)
        datasets[spec.key] = analyze_loaded_dataset(
            data, times, index; n_boot = n_boot)
    end
    return (
        dataset_keys = collect(ALL_DATASET_KEYS),
        datasets = datasets,
        times = Float64.(times),
        seeds = seeds,
        n_orb = 10,
        filling = ensemble.filling,
        theta = THETA,
        steady_tolerance = STEADY_TOLERANCE,
        n_bootstrap = n_boot,
        smoke = smoke,
        figure1_palette = Float64.(figure1["palette_positions"]),
        figure3_palette = [Float64.(rgb) for rgb in figure3["curve_colormap_rgb"]],
        figure1_control_rgb = copy(FIG1_CONTROL_RGB),
        figure3_control_rgb = Float64.(figure3["control_curve_color_rgb"]),
    )
end

function _rewrite_manifest_outputs(
        manifest_path::AbstractString, outputs::AbstractVector{<:AbstractString})
    manifest = DatasetPaths.read_manifest(manifest_path)
    output_paths = abspath.(String.(outputs))
    all(isfile, output_paths) || throw(ArgumentError(
        "cannot update manifest before every output exists"))
    manifest["outputs"] = output_paths
    manifest["panel_display_x_min"] = copy(PANEL_X_MIN)
    manifest["legend_font_size_points"] = LEGEND_FONT_SIZE_PT
    manifest["authoring_canvas_size_points"] = collect(AUTHORING_CANVAS_SIZE_PT)
    tmp = manifest_path * ".tmp.$(getpid())"
    try
        open(tmp, "w") do io
            DatasetPaths.print_manifest(io, manifest; sorted=true, path=manifest_path)
        end
        mv(tmp, manifest_path; force = true)
    finally
        isfile(tmp) && rm(tmp; force = true)
    end
    return manifest_path
end

function main(argv = ARGS)
    cli = _parse_cli(argv)
    if cli.help
        _usage()
        return nothing
    end
    paths = _artifact_paths(cli.output_dir; smoke = cli.smoke)
    mkpath(cli.output_dir)

    if cli.render_only
        result = render_figure(
            paths.plotdata, paths.manifest, cli.output_dir;
            suffix = paths.suffix)
        _rewrite_manifest_outputs(paths.manifest, [result.pdf, result.png])
        println("Rendered $(result.pdf)")
        println("Rendered $(result.png)")
        return result
    end

    payload = build_payload(
        cli.figure1_manifest, cli.figure3_manifest; smoke = cli.smoke)
    write_plotdata(paths.plotdata, payload)
    write_manifest(paths.manifest, payload, paths.plotdata; outputs = String[])
    println("Wrote $(paths.plotdata)")
    println("Wrote $(paths.manifest)")
    cli.build_plotdata_only && return paths

    result = render_figure(
        paths.plotdata, paths.manifest, cli.output_dir;
        suffix = paths.suffix)
    _rewrite_manifest_outputs(paths.manifest, [result.pdf, result.png])
    println("Rendered $(result.pdf)")
    println("Rendered $(result.png)")
    return result
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__) &&
        get(ENV, "RAW_SFF_SUPPLEMENT_TESTING", "0") != "1"
    main()
end
