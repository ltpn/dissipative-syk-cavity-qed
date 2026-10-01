#!/usr/bin/env julia

using CairoMakie
using LaTeXStrings
using TOML

const CGS_RENDER_HERE = @__DIR__
const CGS_RENDER_LAMB_DIR = abspath(joinpath(CGS_RENDER_HERE, "..", "..", ".."))

include(joinpath(CGS_RENDER_HERE, "cgs_survival_merge_validate.jl"))

const CGS_RENDER_KEYS = [
    "fig1_eta_0p1", "fig1_eta_0p4", "fig1_eta_1", "fig1_eta_2",
    "fig1_diss_syk", "fig3_dtilde_0p01", "fig3_dtilde_0p1",
    "fig3_dtilde_1", "fig3_dtilde_10", "fig3_diss_syk",
]
const CGS_AUTHORING_CANVAS_PT = (720.0, 320.0)
const CGS_LATEX_SCALE = 0.5
const CGS_LEGEND_SIZE_PT = 8.0
const CGS_FONT_SCALE = 1.12
const CGS_CROP_MARGINS_PT = (
    left = 0.0, bottom = 4.0, right = 0.0, top = 0.0)
const CGS_PANEL_LABEL_POSITION = (0.025, 0.925)
const CGS_OUTPUT_BASENAME = "supplement_beta0_cgs_survival_n8_q4"
const CGS_YLABEL = L"\mathrm{SFF}_{\mathrm{fid}}(t)"

cgs_pt2px(points::Real) = points * 4 / 3
cgs_source_px(points::Real) = cgs_pt2px(points / CGS_LATEX_SCALE)

function _cgs_render_rgb(values)
    length(values) == 3 || throw(ArgumentError("RGB colors need three components"))
    return RGBf(Float32(values[1]), Float32(values[2]), Float32(values[3]))
end

function _cgs_log_ticks(powers)
    positions = 10.0 .^ collect(powers)
    labels = [latexstring("10^{", power, "}") for power in powers]
    return positions, labels
end

function _cgs_render_theme(canvas_pt = CGS_AUTHORING_CANVAS_PT;
                           font_scale::Real = 1.0)
    font_size = cgs_source_px(7.0 * font_scale)
    return merge(Makie.theme_latexfonts(), Theme(
        size = cgs_pt2px.(canvas_pt),
        fontsize = font_size,
        backgroundcolor = :white,
        figure_padding = cgs_source_px.((4.0, 7.0, 3.0, 3.0)),
        Axis = (
            xlabelsize = cgs_source_px(7.8 * font_scale),
            ylabelsize = cgs_source_px(7.8 * font_scale),
            titlesize = cgs_source_px(7.4 * font_scale),
            titlefont = :regular,
            spinewidth = cgs_source_px(0.8),
            xtickalign = 1,
            ytickalign = 1,
            xticksize = cgs_source_px(3.5),
            yticksize = cgs_source_px(3.5),
            xtickwidth = cgs_source_px(0.8),
            ytickwidth = cgs_source_px(0.8),
            xticksmirrored = true,
            yticksmirrored = true,
        ),
        Legend = (
            fontsize = cgs_source_px(CGS_LEGEND_SIZE_PT * font_scale),
            titlesize = cgs_source_px(CGS_LEGEND_SIZE_PT * font_scale),
            titlefont = :regular,
        ),
    ))
end

function _cgs_number_label(value::Real)
    rounded = round(Float64(value); digits = 12)
    return isinteger(rounded) ? string(Int(rounded)) : string(rounded)
end

function _cgs_rgb_triplet(color)
    return [Float64(color.r), Float64(color.g), Float64(color.b)]
end

function _cgs_positive_or_nan(values)
    return Float64[isfinite(value) && value > 0 ? value : NaN for value in values]
end

function _cgs_draw_dataset!(axis,
                            times,
                            payload::AbstractDict,
                            key::AbstractString,
                            color;
                            control::Bool)
    prefix = "datasets/$key"
    mean_curve = Float64.(payload["$prefix/mean"])
    lower = _cgs_positive_or_nan(payload["$prefix/lower"])
    upper = _cgs_positive_or_nan(payload["$prefix/upper"])
    valid_band = @. isfinite(lower) & isfinite(upper) & (upper >= lower)
    lower[.!valid_band] .= NaN
    upper[.!valid_band] .= NaN
    band!(axis, times, lower, upper;
          color = (color, control ? 0.13 : 0.16))
    curve = _cgs_positive_or_nan(mean_curve)
    handle = lines!(axis, times, curve;
        color = color,
        linestyle = control ? :dash : :solid,
        linewidth = cgs_source_px(control ? 1.45 : 1.25))
    omitted = count(value -> !(isfinite(value) && value > 0), mean_curve)
    return handle, omitted
end

function _cgs_pdf_hires_bbox(path::AbstractString)
    gs = Sys.which("gs")
    gs === nothing && error("PDF export requires Ghostscript (`gs`)")
    stderr_buffer = IOBuffer()
    run(pipeline(
        Cmd([gs, "-q", "-dNOPAUSE", "-dBATCH", "-sDEVICE=bbox", path]);
        stdout = devnull, stderr = stderr_buffer))
    output = String(take!(stderr_buffer))
    matched = match(
        r"%%HiResBoundingBox:\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)",
        output)
    matched === nothing && error("could not determine PDF bounding box for $path")
    return parse.(Float64, matched.captures)
end

function _cgs_crop_pdf(input_path::AbstractString,
                       output_path::AbstractString;
                       margins_pt = (
                           left = 0.0, bottom = 0.0,
                           right = 0.0, top = 0.0))
    llx, lly, urx, ury = _cgs_pdf_hires_bbox(input_path)
    margins = (
        left = Float64(margins_pt.left),
        bottom = Float64(margins_pt.bottom),
        right = Float64(margins_pt.right),
        top = Float64(margins_pt.top),
    )
    all(value -> isfinite(value) && value >= 0, margins) ||
        throw(ArgumentError("PDF crop margins must be finite and nonnegative"))
    width = urx - llx + margins.left + margins.right
    height = ury - lly + margins.bottom + margins.top
    width > 0 && height > 0 || error("invalid PDF bounding box")
    pdfcrop = Sys.which("pdfcrop")
    if pdfcrop !== nothing && all(iszero, margins)
        run(Cmd([pdfcrop, "--hires", "--margins", "0", input_path, output_path]))
    else
        gs = Sys.which("gs")
        run(Cmd([
            gs, "-q", "-dNOPAUSE", "-dBATCH", "-sDEVICE=pdfwrite",
            "-dCompatibilityLevel=1.5", "-dFIXEDMEDIA",
            "-dDEVICEWIDTHPOINTS=$(width)", "-dDEVICEHEIGHTPOINTS=$(height)",
            "-sOutputFile=$(output_path)", "-c",
            "<</PageOffset [-$(llx - margins.left) " *
                "-$(lly - margins.bottom)]>> setpagedevice",
            "-f", input_path,
        ]))
    end
    return [width, height]
end

function _cgs_rasterize_pdf(pdf_path::AbstractString, png_path::AbstractString)
    gs = Sys.which("gs")
    gs === nothing && error("PNG export requires Ghostscript (`gs`)")
    run(Cmd([
        gs, "-q", "-dNOPAUSE", "-dBATCH", "-sDEVICE=pngalpha",
        "-r288", "-dTextAlphaBits=4", "-dGraphicsAlphaBits=4",
        "-sOutputFile=$(png_path)", pdf_path,
    ]))
    return png_path
end

function _cgs_write_figure_manifest(path::AbstractString, manifest::AbstractDict)
    output = abspath(path)
    mkpath(dirname(output))
    temporary = output * ".tmp.$(getpid())"
    try
        open(temporary, "w") do io
            TOML.print(io, manifest; sorted = true)
        end
        mv(temporary, output; force = true)
    finally
        isfile(temporary) && rm(temporary; force = true)
    end
    return output
end

function cgs_validate_figure_manifest(path::AbstractString)
    manifest = TOML.parsefile(path)
    payload = cgs_read_plotdata(manifest["plotdata"])
    String.(payload["dataset_keys"]) == CGS_RENDER_KEYS ||
        throw(ArgumentError("unexpected plot-data dataset order"))
    all(isfile, manifest["outputs"]) ||
        throw(ArgumentError("figure output missing"))
    return nothing
end

function _cgs_render_contract(payload::AbstractDict)
        String.(payload["dataset_keys"]) == CGS_RENDER_KEYS ||
            throw(ArgumentError("unexpected plot-data dataset order"))
        return (
            keys = copy(CGS_RENDER_KEYS),
            panel_keys = [CGS_RENDER_KEYS[1:5], CGS_RENDER_KEYS[6:10]],
            families = ["figure1", "figure3"],
            deltas = [Float64(payload["datasets/$key/delta_0_over_2pi_mhz"])
                      for key in (CGS_RENDER_KEYS[1], CGS_RENDER_KEYS[6])],
            rows = 1,
            columns = 2,
            panel_order = ["figure1", "figure3"],
            canvas = CGS_AUTHORING_CANVAS_PT,
            font_scale = CGS_FONT_SCALE,
            crop_margins_pt = CGS_CROP_MARGINS_PT,
            panel_label_position = CGS_PANEL_LABEL_POSITION,
            ylabel = CGS_YLABEL,
            y_axis_label = "SFF_{\\rm fid}(t)",
        )
end

function cgs_render_supplement(plotdata_path::AbstractString,
                               source_manifest_path::AbstractString,
                               output_dir::AbstractString;
                               smoke::Bool = false)
    plotdata = abspath(plotdata_path)
    source_manifest = abspath(source_manifest_path)
    cgs_validate_manifest(source_manifest)
    source = TOML.parsefile(source_manifest)
    source["plotdata"] == plotdata ||
        throw(ArgumentError("source manifest is bound to different plot data"))
    payload = cgs_read_plotdata(plotdata)
    contract = _cgs_render_contract(payload)
    Int(payload["n_orb"]) == 8 && Int(payload["filling"]) == 4 ||
        throw(ArgumentError("renderer requires N8/Q4 plot data"))
    times_full = Float64.(payload["times"])
    !isempty(times_full) && times_full[1] == 0.0 ||
        throw(ArgumentError("plot data must retain t=0"))
    positive_mask = times_full .> 0
    any(positive_mask) || throw(ArgumentError("plot data have no positive times"))
    times = times_full[positive_mask]

    fig1_colors = [cgrad(:viridis)[position]
                   for position in Float64.(payload["figure1_palette_positions"])]
    fig1_control = _cgs_render_rgb(payload["figure1_control_rgb"])
    fig3_colors = _cgs_render_rgb.(payload["figure3_rgb"])
    fig3_control = _cgs_render_rgb(payload["figure3_control_rgb"])
    omitted = Dict{String,Int}()
    legend_labels = String[]
    delta_annotations = ["$(_cgs_number_label(delta0)) MHz"
                         for delta0 in contract.deltas]
    panel_titles = fill("", length(contract.panel_keys))
    legend_inside = [true, true]
    legend_positions = ["right_top", "right_top"]
    fig1_palette = vcat(_cgs_rgb_triplet.(fig1_colors),
                        [_cgs_rgb_triplet(fig1_control)])
    fig3_palette = vcat(_cgs_rgb_triplet.(fig3_colors),
                        [_cgs_rgb_triplet(fig3_control)])
    panel_palettes = [
        deepcopy(family == "figure1" ? fig1_palette : fig3_palette)
        for family in contract.families
    ]

    CairoMakie.activate!()
    figure = with_theme(_cgs_render_theme(
            contract.canvas; font_scale = contract.font_scale)) do
        fig = Figure(backgroundcolor = :white)
        axes = Axis[]
        group_handles = Vector{Vector{Any}}()
        panel_colors = [family == "figure1" ?
                        vcat(fig1_colors, [fig1_control]) :
                        vcat(fig3_colors, [fig3_control])
                        for family in contract.families]
        for panel_index in eachindex(contract.panel_keys)
            row = div(panel_index - 1, contract.columns) + 1
            column = mod(panel_index - 1, contract.columns) + 1
            axis = Axis(
                fig[row, column];
                aspect = 1.35,
                xscale = log10,
                yscale = log10,
                xgridvisible = false,
                ygridvisible = false,
                xticks = _cgs_log_ticks((-3, 0, 3, 6)),
                xlabel = L"tJ",
                ylabel = column == 1 ? contract.ylabel : "",
                title = "",
                xlabelpadding = cgs_source_px(1.5),
                ylabelpadding = cgs_source_px(2.0),
                titlegap = cgs_source_px(1.0),
                xticklabelpad = cgs_source_px(1.2),
                yticklabelpad = cgs_source_px(1.2),
            )
            push!(axes, axis)
            handles = Any[]
            panel_minimum = Inf
            panel_maximum = 0.0
            for (curve_index, key) in enumerate(contract.panel_keys[panel_index])
                control = curve_index == 5
                prefix = "datasets/$key"
                local_payload = Dict{String,Any}(
                    "$prefix/mean" => Float64.(payload["$prefix/mean"])[positive_mask],
                    "$prefix/lower" => Float64.(payload["$prefix/lower"])[positive_mask],
                    "$prefix/upper" => Float64.(payload["$prefix/upper"])[positive_mask],
                )
                for values in values(local_payload)
                    positive = filter(value -> isfinite(value) && value > 0, values)
                    isempty(positive) && continue
                    panel_minimum = min(panel_minimum, minimum(positive))
                    panel_maximum = max(panel_maximum, maximum(positive))
                end
                handle, omitted_count = _cgs_draw_dataset!(
                    axis, times, local_payload, key,
                    panel_colors[panel_index][curve_index]; control = control)
                push!(handles, handle)
                omitted[key] = omitted_count
                push!(legend_labels, String(payload["$prefix/label"]))
            end
            push!(group_handles, handles)
            xlims!(axis, first(times), last(times))
            isfinite(panel_minimum) && panel_maximum > 0 || error(
                "panel $panel_index has no positive survival values")
            ylims!(axis, panel_minimum / 1.25, panel_maximum * 1.4)
            text!(axis, contract.panel_label_position...;
                  text = latexstring("\\mathrm{(", Char('a' + panel_index - 1), ")}"),
                  space = :relative, align = (:left, :top), color = :black)

        end
        legend_panels = [1, 2]
        for panel_index in legend_panels
            label_start = 5 * panel_index - 4
            title = contract.families[panel_index] == "figure1" ?
                L"\eta=" : L"\delta\tilde{\omega}="
            axislegend(
                axes[panel_index], group_handles[panel_index],
                legend_labels[label_start:(label_start + 4)], title;
                position = :rt,
                orientation = :vertical, framevisible = false,
                backgroundcolor = (:white, 0.82),
                labelsize = cgs_source_px(
                    CGS_LEGEND_SIZE_PT * contract.font_scale),
                titlesize = cgs_source_px(
                    CGS_LEGEND_SIZE_PT * contract.font_scale),
                patchsize = cgs_source_px.((9.0, 2.5)),
                padding = cgs_source_px.((1.0, 1.0, 1.0, 1.0)),
                rowgap = cgs_source_px(0.6),
                patchlabelgap = cgs_source_px(0.8))
        end
        colgap!(fig.layout, cgs_source_px(4.0))
        rowgap!(fig.layout, cgs_source_px(3.0))
        for column in 1:contract.columns
            colsize!(fig.layout, column, Relative(1 / contract.columns))
        end
        fig
    end

    output_directory = abspath(output_dir)
    mkpath(output_directory)
    pdf = joinpath(output_directory, "$CGS_OUTPUT_BASENAME.pdf")
    png = joinpath(output_directory, "$CGS_OUTPUT_BASENAME.png")
    crop_size = mktempdir() do temporary_directory
        uncropped = joinpath(temporary_directory, "uncropped.pdf")
        save(uncropped, figure)
        _cgs_crop_pdf(
            uncropped, pdf; margins_pt = contract.crop_margins_pt)
    end
    _cgs_rasterize_pdf(pdf, png)
    caption = "Raw beta-zero coherent-Gibbs-state survival probability at N=8, " *
    "Q=4, averaged over 64 disorder realizations and evolved by full " *
    "exact Liouvillian diagonalization."

    figure_manifest = joinpath(output_directory, "figure.toml")
    manifest = Dict{String,Any}(
        "smoke" => smoke,
        "plotdata" => plotdata,
        "source_manifest" => source_manifest,
        "outputs" => [pdf, png],
        "panel_order" => contract.panel_order,
        "dataset_keys" => contract.keys,
        "curve_counts" => fill(5, length(contract.panel_keys)),
        "band_counts" => fill(5, length(contract.panel_keys)),
        "legend_inside" => legend_inside,
        "panel_palettes" => panel_palettes,
        "figure1_palette_positions" =>
            Float64.(payload["figure1_palette_positions"]),
        "figure1_control_rgb" => Float64.(payload["figure1_control_rgb"]),
        "figure3_rgb" => [Float64.(rgb) for rgb in payload["figure3_rgb"]],
        "figure3_control_rgb" => Float64.(payload["figure3_control_rgb"]),
        "plotted_time_min" => first(times),
        "plotted_time_max" => last(times),
        "omitted_nonpositive" => omitted,
        "n_orb" => Int(payload["n_orb"]),
        "filling" => Int(payload["filling"]),
        "n_seeds" => isempty(payload["seeds"]) ? 0 : length(payload["seeds"]),
        "dataset_seed_counts" => Int.(payload["dataset_seed_counts"]),
        "n_bootstrap" => Int(payload["n_bootstrap"]),
        "authoring_canvas_pt" => collect(contract.canvas),
        "font_scale" => contract.font_scale,
        "crop_margins_pt" => collect(contract.crop_margins_pt),
        "latex_scale" => CGS_LATEX_SCALE,
        "cropped_size_pt" => crop_size,
    )
    _cgs_write_figure_manifest(figure_manifest, manifest)
    cgs_validate_figure_manifest(figure_manifest)
    return (pdf = pdf, png = png, manifest = figure_manifest)
end

function _cgs_parse_render_cli(argv)
    options = Dict{String,String}()
    smoke = false
    index = 1
    while index <= length(argv)
        argument = argv[index]
        if argument == "--smoke"
            smoke = true
        else
            startswith(argument, "--") || error("unexpected positional argument: $argument")
            key = argument[3:end]
            index += 1
            index <= length(argv) || error("--$key requires a value")
            options[key] = argv[index]
        end
        index += 1
    end
    for key in ("plotdata", "manifest", "output-dir")
        haskey(options, key) || error("missing required option --$key")
    end
    options["smoke"] = string(smoke)
    return options
end

function cgs_render_main(argv = ARGS)
    options = _cgs_parse_render_cli(argv)
    return cgs_render_supplement(
        options["plotdata"], options["manifest"], options["output-dir"];
        smoke = parse(Bool, options["smoke"]))
end

if abspath(PROGRAM_FILE) == @__FILE__
    cgs_render_main()
end
