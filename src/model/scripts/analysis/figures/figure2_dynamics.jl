#!/usr/bin/env julia
# figure2_dynamics.jl
#
# Final production Figure 2 for the Lamb-Dicke chaos manuscript.
#
# At Delta_cd/2pi = 1 MHz (the same cut as Figure 1), the top row shows
# seed-averaged orbital-occupation heatmaps at eta = 0.1 and eta = 2.0.
# The bottom row shows the representative-orbital occupation and von Neumann
# entropy side by side, overlaying eta in {0.1, 0.4, 1.0, 2.0}.  All curves
# are raw seed averages of the stored `times`, `entropy_t`, and
# `populations_t` arrays; no display smoothing is applied.
#
# A CSR synthetic-ladder reference can be supplied through `--l2-config`.
# The canonical N=10, filling=3 figure uses the 64-seed L3b target model and
# harvests its stored dynamics directly from the ladder spectra directory.

using CairoMakie
using JLD2
using LaTeXStrings
using Printf
using TOML
include(joinpath(@__DIR__, "..", "..", "..", "DatasetPaths.jl"))

const HERE = @__DIR__
include(joinpath(HERE, "_figure_export.jl"))
const MODULE_DIR = abspath(joinpath(HERE, "..", "..", ".."))       # src/model
const LAMB_DIR   = MODULE_DIR
const REPO_ROOT  = abspath(joinpath(HERE, "..", "..", "..", "..", "..")) # repo root
const DATA_ROOT  = joinpath(REPO_ROOT, "data")
const FIGURE_ROOT = joinpath(REPO_ROOT, "figures")

include(joinpath(HERE, "prl_style.jl"))
include(joinpath(HERE, "MagicLaTeX.jl"))
using .PRLStyle
using .MagicLaTeX

const DEFAULT_CONFIG = joinpath(MODULE_DIR, "configs",
                                 "physical_n10_f3.toml")
const DEFAULT_OUTPUT_DIR =
    joinpath(FIGURE_ROOT)

const MAGIC_PALETTE = palette_magiclatex(:gem_2024)
const FIGURE2_LATEX_SCALE = 0.5
const FIGURE2_AUTHORING_CANVAS_SIZE_PT = (601.0, 540.0)
const FIGURE2_LEGEND_LABEL_PT = 7.2
const FIGURE2_COLORBAR_LABEL_PADDING_PT = -1.0
const FIGURE2_COLUMN_GAP_PT = 5.0
const FIGURE2_ROW_GAP_PT = -12.25
figure2_source_px(points::Real) =
    MagicLaTeX.pt2px(points / FIGURE2_LATEX_SCALE)

function figure2_log_ticks(powers)
    positions = 10.0 .^ collect(powers)
    labels = [latexstring("10^{", power, "}") for power in powers]
    return positions, labels
end

function figure2_colorbar_ticks(heat_lo, heat_hi, n_orb, filling)
    occupation = filling / n_orb
    positions = sort(unique([heat_lo, occupation, 0.4, heat_hi]))
    labels = [value == occupation ?
        latexstring(@sprintf("\\frac{%d}{%d}", filling, n_orb)) :
        latexstring(@sprintf("%g", value)) for value in positions]
    return positions, labels
end

const DEFAULT_L2_CONFIG = joinpath(LAMB_DIR, "configs", "csr_ladder_l3b_n10_f3.toml")
const DEFAULT_L2_ETA    = 2.0
L2_CONFIG_NAME::String  = "synthetic_l3b_bdi_syk4"   # overwritten in main()
const L2_LINE_COLOR     = RGBf(1.0, 0.27, 0.0)    # orangered, matches figure 1
const L2_LINESTYLE      = :dash

# ------------------------------------------------------------------
# Physics constants for this figure
# ------------------------------------------------------------------
const ETAS_PLOT = (0.1, 0.4, 1.0, 2.0)
const FIGURE2_ETA_VIRIDIS_POSITIONS =
    collect(range(0.05, 0.90; length = length(ETAS_PLOT)))
const GAMMA     = 1.0                 # Delta_cd/2pi = 1 MHz
const SVD_TOL   = 1.0e-10

# ------------------------------------------------------------------
# I/O helpers (canonical filename convention, matches sweep_scheduler)
# ------------------------------------------------------------------
_fmt(x::Real)    = string(Float64(x))
_fmt(x::Integer) = string(x)

function _seed_path(spectra_dir, config, eta, gamma, tol, seed)
    fname = @sprintf("%s__eta=%s__gamma=%s__tol=%s__seed=%d.jld2",
                     config, _fmt(eta), _fmt(gamma), _fmt(tol), seed)
    return joinpath(spectra_dir, fname)
end

function load_dynamics_point(path)
    isfile(path) || return nothing
    JLD2.jldopen(path, "r") do f
        (times = Vector{Float64}(f["times"]),
         entropy = Vector{Float64}(f["entropy_t"]),
         populations = Matrix{Float64}(f["populations_t"]))
    end
end

"""
Seed-average `entropy_t` and `populations_t` across the seeds available on
disk.  Skips a seed silently if the JLD2 is missing.  Returns `nothing` if
zero seeds are available for (`eta`, `gamma`).
"""
function seed_average_dynamics(config, eta, gamma, tol, seeds, spectra_dir)
    t_ref = nothing
    s_acc = nothing
    p_acc = nothing
    n_used = 0
    for s in seeds
        pt = load_dynamics_point(_seed_path(spectra_dir, config, eta, gamma, tol, s))
        pt === nothing && continue
        if t_ref === nothing
            t_ref = pt.times
            s_acc = copy(pt.entropy)
            p_acc = copy(pt.populations)
        else
            s_acc .+= pt.entropy
            p_acc .+= pt.populations
        end
        n_used += 1
    end
    n_used == 0 && return nothing
    s_acc ./= n_used
    p_acc ./= n_used
    return (t = t_ref, s = s_acc, pops = p_acc, n_used = n_used)
end

# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
function _parse_cli(argv)
    spectra_dir = ""
    config_path = DEFAULT_CONFIG
    output_dir  = DEFAULT_OUTPUT_DIR
    seeds_str   = ""
    l2_config   = DEFAULT_L2_CONFIG
    l2_eta      = DEFAULT_L2_ETA
    l2_seeds_str = ""    # empty => use 1:l2_cfg.grid.seed_count
    l2_spectra_dir_override = ""  # empty => derive from L2 TOML output.data_subdir
    include_l2  = true
    i = 1
    while i <= length(argv)
        a = argv[i]
        if a == "--config"; i += 1; config_path = argv[i]
        elseif startswith(a, "--config="); config_path = split(a, "=", limit = 2)[2]
        elseif a == "--spectra-dir"
            i += 1
            i <= length(argv) || error("--spectra-dir requires a path")
            spectra_dir = argv[i]
        elseif startswith(a, "--spectra-dir="); spectra_dir = split(a, "=", limit=2)[2]
        elseif a == "--output-dir"; i += 1; output_dir = argv[i]
        elseif startswith(a, "--output-dir="); output_dir = split(a, "=", limit = 2)[2]
        elseif a == "--seeds"; i += 1; seeds_str = argv[i]
        elseif startswith(a, "--seeds="); seeds_str = split(a, "=", limit = 2)[2]
        elseif a == "--l2-config"; i += 1; l2_config = argv[i]
        elseif startswith(a, "--l2-config="); l2_config = split(a, "=", limit = 2)[2]
        elseif a == "--l2-eta"; i += 1; l2_eta = parse(Float64, argv[i])
        elseif startswith(a, "--l2-eta="); l2_eta = parse(Float64, split(a, "=", limit = 2)[2])
        elseif a == "--l2-seeds"; i += 1; l2_seeds_str = argv[i]
        elseif startswith(a, "--l2-seeds="); l2_seeds_str = split(a, "=", limit = 2)[2]
        elseif a == "--l2-spectra-dir"; i += 1; l2_spectra_dir_override = argv[i]
        elseif startswith(a, "--l2-spectra-dir="); l2_spectra_dir_override = split(a, "=", limit = 2)[2]
        elseif a == "--no-l2"; include_l2 = false
        else error("unrecognized argument: $a")
        end
        i += 1
    end
    return (config = abspath(config_path),
             spectra_dir = isempty(spectra_dir) ? "" : abspath(spectra_dir),
             output_dir = abspath(output_dir),
             seeds_str = String(seeds_str),
             l2_config = abspath(l2_config),
             l2_eta = Float64(l2_eta),
             l2_seeds_str = String(l2_seeds_str),
             l2_spectra_dir_override = String(l2_spectra_dir_override),
             include_l2 = include_l2)
end

function _seed_range(str, default_n)
    isempty(str) && return collect(1:default_n)
    occursin(":", str) || error("--seeds must be lo:hi (got $str)")
    parts = split(str, ":")
    return collect(parse(Int, parts[1]):parse(Int, parts[2]))
end

# ------------------------------------------------------------------
# Main driver
# ------------------------------------------------------------------
function main(argv = ARGS)
    cli = _parse_cli(argv)
    cfg = DatasetPaths.read_manifest(cli.config)
    n_orb   = Int(cfg["numerics"]["n_orb"])
    filling = haskey(cfg["numerics"], "filling") ? Int(cfg["numerics"]["filling"]) : div(n_orb, 2)
    grid    = cfg["grid"]
    outcfg  = cfg["output"]
    seed_count   = Int(grid["seed_count"])
    baseline_tol = Float64(grid["baseline_svd_tol"])
    seeds = _seed_range(cli.seeds_str, seed_count)

    orig_data = isabspath(outcfg["data_subdir"]) ?
                    String(outcfg["data_subdir"]) :
                    abspath(joinpath(REPO_ROOT, "data",
                                      String(outcfg["data_subdir"])))
    spectra_dir = isempty(cli.spectra_dir) ? joinpath(orig_data, "spectra") : cli.spectra_dir
    isdir(spectra_dir) || error("spectra directory not found: $spectra_dir")

    # The initial product state is |1,...,1,0,...,0> occupying orbitals 1..filling
    # (see Dynamics.jl `initial_product_state_dm`).  The representative orbital
    # is the last filled one, `j_rep = filling`; the reference density line is
    j_rep = filling
    d_hilbert = binomial(n_orb, filling)
    logD = log(d_hilbert)
    reference_occupation = filling / n_orb

    println("=" ^ 78)
    println("Final Figure 2  (dynamics S(t), n_j(t) at Delta_cd/2pi = $GAMMA MHz)")
    println("config:       $(cli.config)")
    println("spectra dir:  $spectra_dir")
    println("output dir:   $(cli.output_dir)")
    println("etas (plot):  $(collect(ETAS_PLOT))")
    println("seeds:        $(first(seeds)):$(last(seeds)) (n=$(length(seeds)))")
    println("orbital j:    $j_rep  (last filled; D=binomial($n_orb, $filling)=$d_hilbert)")
    println("filling:      $filling / $n_orb = $(round(reference_occupation, digits=3))")
    println("=" ^ 78)

    println("\n--- Seed-averaging entropy and populations ---")
    curves = Vector{NamedTuple}(undef, length(ETAS_PLOT))
    n_used_per_eta = Int[]
    for (i, eta) in enumerate(ETAS_PLOT)
        res = seed_average_dynamics("physical", eta, GAMMA, baseline_tol,
                                     seeds, spectra_dir)
        res === nothing && error("no dynamics seeds available for eta=$eta")
        curves[i] = res
        push!(n_used_per_eta, res.n_used)
        @printf("  eta=%-4g  seeds used=%3d\n", eta, res.n_used)
    end
    times_ref = curves[1].t

    # ---- Ladder overlay: CSR synthetic-ladder dynamics ---------------
    # from the ladder sweep JLD2 files (dynamics=on).  No rebuild, no
    # eigen(L) on the plot-time host.
    l2_dyn = nothing
    l2_manifest = Dict{String,Any}(
        "config"  => cli.include_l2 ? cli.l2_config : "",
    )
    if cli.include_l2
        println("\n--- Ladder overlay (CSR synthetic ladder $(cli.l2_config)) ---")
        isfile(cli.l2_config) ||
            error("ladder config not found: $(cli.l2_config)")
        l2_cfg = DatasetPaths.read_manifest(cli.l2_config)
        l2_grid = l2_cfg["grid"]
        # Bind the ladder config identifier used in JLD2 filenames.
        global L2_CONFIG_NAME = String(l2_grid["configs"][1])
        println("  ladder config name: $L2_CONFIG_NAME")
        l2_seed_count = Int(l2_grid["seed_count"])
        l2_seeds = isempty(cli.l2_seeds_str) ?
            collect(1:l2_seed_count) : _seed_range(cli.l2_seeds_str, l2_seed_count)
        l2_num = l2_cfg["numerics"]
        l2_n_orb = Int(l2_num["n_orb"])
        l2_n_orb == n_orb ||
            error("ladder n_orb=$l2_n_orb ≠ physical n_orb=$n_orb (fig 2 pipeline assumes matched)")
        l2_filling = haskey(l2_num, "filling") ? Int(l2_num["filling"]) : div(l2_n_orb, 2)
        l2_filling == filling ||
            error("ladder filling=$l2_filling ≠ physical filling=$filling (fig 2 pipeline assumes matched)")
        l2_tol = Float64(l2_grid["baseline_svd_tol"])
        l2_spectra_dir = isempty(cli.l2_spectra_dir_override) ?
            joinpath(DATA_ROOT, String(l2_cfg["output"]["data_subdir"]), "spectra") :
            abspath(cli.l2_spectra_dir_override)
        @printf("  ladder config:  %s\n", cli.l2_config)
        @printf("  ladder eta:     %g\n", cli.l2_eta)
        @printf("  ladder seeds:   1:%d (n=%d)\n", last(l2_seeds), length(l2_seeds))
        @printf("  ladder spectra: %s\n", l2_spectra_dir)
        if !isdir(l2_spectra_dir)
            @warn "ladder spectra dir not found; L2 overlay disabled" path=l2_spectra_dir
        else
            l2_avg = seed_average_dynamics(L2_CONFIG_NAME, cli.l2_eta, GAMMA,
                                            l2_tol, l2_seeds, l2_spectra_dir)
            if l2_avg === nothing
                @warn "no ladder dynamics seeds found on disk; L2 overlay disabled" dir=l2_spectra_dir eta=cli.l2_eta gamma=GAMMA tol=l2_tol
            else
                l2_dyn = (times = l2_avg.t, s = l2_avg.s, pops = l2_avg.pops,
                          n_seeds = l2_avg.n_used)
                @printf("  ladder dynamics: n_seeds=%d\n", l2_avg.n_used)
            end
        end
        l2_manifest["eta"]        = cli.l2_eta
        l2_manifest["gamma"]      = GAMMA
        l2_manifest["svd_tol"]    = l2_tol
        l2_manifest["seed_range"] = [first(l2_seeds), last(l2_seeds)]
        l2_manifest["n_seeds"]    = length(l2_seeds)
        l2_manifest["n_seeds_used"] = l2_dyn === nothing ? 0 : l2_dyn.n_seeds
        l2_manifest["spectra_dir"] = l2_spectra_dir
        l2_manifest["config_name"] = L2_CONFIG_NAME
    end

    mkpath(cli.output_dir)
    pdf_path = joinpath(cli.output_dir, "figure_2_dynamics.pdf")
    png_path = joinpath(cli.output_dir, "figure_2_dynamics.png")

    # Two rows on the same authoring-width and half-scale export contract as
    # Figure 1.  All dynamics traces below remain raw seed averages.
    T_MIN_DYN_PLOT = 10.0^(-1.5)
    T_MAX_DYN_PLOT = 10.0^(3.5)
    T_MIN_HEATMAP = T_MIN_DYN_PLOT
    T_MAX_HEATMAP = T_MAX_DYN_PLOT

    idx_eta_left  = findfirst(x -> x == 0.1, collect(ETAS_PLOT))
    idx_eta_right = findfirst(x -> x == 2.0, collect(ETAS_PLOT))
    idx_eta_left  === nothing && error("figure 2 heatmap panel expects eta = 0.1 in ETAS_PLOT")
    idx_eta_right === nothing && error("figure 2 heatmap panel expects eta = 2.0 in ETAS_PLOT")
    times_heat = curves[idx_eta_left].t
    j_axis     = collect(1:n_orb)

    # Preserve the centered blue-white-red occupation scale.
    steady_state_occ = Float64(filling) / Float64(n_orb)
    heat_lo, heat_hi = 0.2, 0.5
    white_pos = (steady_state_occ - heat_lo) / (heat_hi - heat_lo)
    heat_cmap  = cgrad([
        RGBf(5/255,   48/255, 97/255),      # dark blue
        RGBf(1.0,     1.0,    1.0),         # pure white
        RGBf(103/255, 0/255,  31/255),      # dark red
    ], [0.0, white_pos, 1.0])
    pops_left  = collect(transpose(curves[idx_eta_left].pops))
    pops_right = collect(transpose(curves[idx_eta_right].pops))
    eta_colormap = cgrad(:viridis)
    eta_colors = [eta_colormap[position]
                  for position in FIGURE2_ETA_VIRIDIS_POSITIONS]
    target_color = MAGIC_PALETTE[7]
    main_width = figure2_source_px(1.15)
    reference_width = figure2_source_px(1.35)
    base_theme = theme_magiclatex(
        PaletteName = :gem_2024,
        FigureSize = MagicLaTeX.pt2px.(FIGURE2_AUTHORING_CANVAS_SIZE_PT),
    )
    compact_theme = merge(base_theme, Theme(
        figure_padding = figure2_source_px.((2.0, 2.0, 1.5, 1.5)),
    ))

    fig = with_theme(compact_theme) do
        figure = Figure(backgroundcolor = :white)
        top_grid = GridLayout()
        bottom_grid = GridLayout()
        figure[1, 1] = top_grid
        figure[2, 1] = bottom_grid

        axis_common = (
            xscale = log10,
            xgridvisible = false,
            ygridvisible = false,
            xlabelpadding = figure2_source_px(1.5),
            ylabelpadding = figure2_source_px(1.5),
            xticklabelpad = figure2_source_px(1.3),
            yticklabelpad = figure2_source_px(1.3),
        )
        ax_h1 = Axis(top_grid[1, 1];
            axis_common...,
            aspect = 1.05,
            xlabel = "",
            ylabel = L"j",
            xticks = figure2_log_ticks(-1:3),
            xticklabelsvisible = false,
        )
        ax_h2 = Axis(top_grid[1, 2];
            axis_common...,
            aspect = 1.05,
            xlabel = "",
            xticks = figure2_log_ticks(-1:3),
            xticklabelsvisible = false,
            yticklabelsvisible = false,
        )
        ax_n = Axis(bottom_grid[1, 1];
            axis_common...,
            aspect = 0.95,
            xlabel = L"tJ",
            ylabel = latexstring(@sprintf("n_{%d}(t)", j_rep)),
            xticks = figure2_log_ticks(-1:3),
        )
        ax_s = Axis(bottom_grid[1, 2];
            axis_common...,
            aspect = 0.95,
            xlabel = L"tJ",
            ylabel = L"S(t)",
            yaxisposition = :right,
            xticks = figure2_log_ticks(-1:3),
        )

        hm = heatmap!(ax_h1, times_heat, j_axis, pops_left;
            colormap = heat_cmap, colorrange = (heat_lo, heat_hi))
        heatmap!(ax_h2, times_heat, j_axis, pops_right;
            colormap = heat_cmap, colorrange = (heat_lo, heat_hi))
        orbital_ticks = ([1, div(n_orb, 2), n_orb],
                         ["1", string(div(n_orb, 2)), string(n_orb)])
        ax_h1.yticks = orbital_ticks
        ax_h2.yticks = orbital_ticks
        Colorbar(figure[1, 2], hm;
            label = L"n_j(t)",
            ticks = figure2_colorbar_ticks(heat_lo, heat_hi, n_orb, filling),
            flipaxis = true,
            width = figure2_source_px(3.0),
            height = Relative(0.8125),
            valign = :center,
            labelpadding = figure2_source_px(
                FIGURE2_COLORBAR_LABEL_PADDING_PT),
        )

        eta_lines = Any[]
        for (i, eta) in enumerate(ETAS_PLOT)
            curve = curves[i]
            keep = findall(k -> curve.t[k] > 0 && curve.t[k] <= T_MAX_DYN_PLOT,
                           eachindex(curve.t))
            color = eta_colors[i]
            line = lines!(ax_n, curve.t[keep], curve.pops[j_rep, keep];
                color = color, linewidth = main_width)
            lines!(ax_s, curve.t[keep], curve.s[keep];
                color = color, linewidth = main_width)
            push!(eta_lines, line)
        end

        target_n_line = nothing
        target_s_line = nothing
        if l2_dyn !== nothing
            keep = findall(k -> l2_dyn.times[k] > 0 &&
                                l2_dyn.times[k] <= T_MAX_DYN_PLOT,
                           eachindex(l2_dyn.times))
            target_n_line = lines!(ax_n, l2_dyn.times[keep],
                l2_dyn.pops[j_rep, keep]; color = target_color,
                linewidth = reference_width)
            target_s_line = lines!(ax_s, l2_dyn.times[keep], l2_dyn.s[keep];
                color = target_color, linewidth = reference_width)
        end

        line_n_ref = hlines!(ax_n, [reference_occupation];
            color = :gray45, linestyle = :dot,
            linewidth = figure2_source_px(0.7))
        line_s_ref = hlines!(ax_s, [logD];
            color = :gray45, linestyle = :dot,
            linewidth = figure2_source_px(0.7))

        axislegend(
            ax_n,
            eta_lines,
            Any[L"0.1", L"0.4", L"1", L"2"],
            L"\eta=";
            position = (0.96, 0.88),
            orientation = :horizontal,
            titleposition = :left,
            nbanks = 2,
            framevisible = false,
            labelsize = figure2_source_px(FIGURE2_LEGEND_LABEL_PT),
            titlesize = figure2_source_px(FIGURE2_LEGEND_LABEL_PT),
            patchsize = figure2_source_px.((6.0, 2.5)),
            padding = figure2_source_px.((0.5, 0.5, 0.5, 0.5)),
            margin = (0, 0, 0, 0),
            rowgap = figure2_source_px(0.5),
            colgap = figure2_source_px(0.8),
            patchlabelgap = figure2_source_px(0.6),
            titlegap = figure2_source_px(0.8),
        )
        axislegend(
            ax_n,
            [line_n_ref],
            [L"\nu"];
            position = (0.96, 0.68),
            framevisible = false,
            labelsize = figure2_source_px(FIGURE2_LEGEND_LABEL_PT),
            patchsize = figure2_source_px.((8.0, 2.5)),
            padding = figure2_source_px.((0.5, 0.5, 0.5, 0.5)),
            margin = (0, 0, 0, 0),
            patchlabelgap = figure2_source_px(0.8),
        )
        if target_n_line !== nothing
            axislegend(
                ax_n,
                [target_n_line],
                [L"\mathrm{diss.\ SYK}"];
                position = (0.96, 0.56),
                framevisible = false,
                labelsize = figure2_source_px(FIGURE2_LEGEND_LABEL_PT),
                patchsize = figure2_source_px.((10.0, 2.5)),
                padding = figure2_source_px.((0.5, 0.5, 0.5, 0.5)),
                margin = (0, 0, 0, 0),
                patchlabelgap = figure2_source_px(0.8),
            )
        end
        axislegend(
            ax_s,
            [line_s_ref],
            [L"\log D"];
            position = (0.96, 0.22),
            framevisible = false,
            labelsize = figure2_source_px(FIGURE2_LEGEND_LABEL_PT),
            patchsize = figure2_source_px.((8.0, 2.5)),
            padding = figure2_source_px.((0.5, 0.5, 0.5, 0.5)),
            margin = (0, 0, 0, 0),
            patchlabelgap = figure2_source_px(0.8),
        )

        linkxaxes!(ax_h1, ax_h2)
        linkxaxes!(ax_n, ax_s)
        xlims!(ax_h1, T_MIN_HEATMAP, T_MAX_HEATMAP)
        xlims!(ax_n, T_MIN_DYN_PLOT, T_MAX_DYN_PLOT)
        ylims!(ax_h1, 0.5, n_orb + 0.5)
        ylims!(ax_h2, 0.5, n_orb + 0.5)

        for (axis, label, color, y_position) in (
                (ax_h1, L"\mathrm{(a)}", :white, 0.99),
                (ax_h2, L"\mathrm{(b)}", :white, 0.99),
                (ax_n, L"\mathrm{(c)}", :black, 0.93),
                (ax_s, L"\mathrm{(d)}", :black, 0.93))
            text!(axis, 0.012, y_position; text = label, space = :relative,
                align = (:left, :top), color = color)
        end

        colsize!(top_grid, 1, Auto(1.0))
        colsize!(top_grid, 2, Auto(1.0))
        colsize!(bottom_grid, 1, Relative(0.50))
        colsize!(bottom_grid, 2, Relative(0.50))
        colgap!(top_grid, 1, figure2_source_px(FIGURE2_COLUMN_GAP_PT))
        colgap!(bottom_grid, figure2_source_px(FIGURE2_COLUMN_GAP_PT))
        colgap!(figure.layout, 1, figure2_source_px(-14.5))
        rowsize!(figure.layout, 1, Relative(0.47))
        rowsize!(figure.layout, 2, Relative(0.53))
        rowgap!(figure.layout, figure2_source_px(FIGURE2_ROW_GAP_PT))
        figure
    end

    source_size_points = mktempdir() do temporary_dir
        uncropped_pdf = joinpath(temporary_dir, "figure-uncropped.pdf")
        save(uncropped_pdf, fig)
        crop_pdf_hires(uncropped_pdf, pdf_path)
    end
    rasterize_pdf_png(pdf_path, png_path)
    println("wrote $pdf_path")
    println("wrote $png_path")

    # ---- Companion manifest ----
    manifest_path = joinpath(cli.output_dir, "figure_2__manifest.toml")
    open(manifest_path, "w") do io
        DatasetPaths.print_manifest(io, Dict{String,Any}(
            "config"               => cli.config,
            "spectra_dir"          => spectra_dir,
            "n_orb"                => n_orb,
            "filling"              => filling,
            "orbital_index"        => j_rep,
            "hilbert_dim"          => d_hilbert,
            "reference_occupation" => reference_occupation,
            "etas"                 => collect(ETAS_PLOT),
            "delta_cd_over_2pi_mhz"=> GAMMA,
            "svd_tol"              => baseline_tol,
            "seed_range"           => [first(seeds), last(seeds)],
            "n_seeds"              => length(seeds),
            "n_seeds_used"         => n_used_per_eta,
            "authoring_canvas_size_points" =>
                collect(FIGURE2_AUTHORING_CANVAS_SIZE_PT),
            "source_figure_size_points" => source_size_points,
            "final_figure_size_points_at_columnwidth" =>
                source_size_points .* FIGURE2_LATEX_SCALE,
            "intended_latex_scale" => FIGURE2_LATEX_SCALE,
            "curve_palette_positions" => FIGURE2_ETA_VIRIDIS_POSITIONS,
            "legend_label_size_points" => FIGURE2_LEGEND_LABEL_PT,
            "colorbar_height_relative" => 0.8125,
            "colorbar_label_padding_points" =>
                FIGURE2_COLORBAR_LABEL_PADDING_PT,
            "time_plot_limits" => [T_MIN_DYN_PLOT, T_MAX_DYN_PLOT],
            "l2_overlay"           => l2_manifest,
            "outputs"              => [pdf_path, png_path],
        ); path=manifest_path)
    end
    println("wrote $manifest_path")

    return (pdf = pdf_path, png = png_path, manifest = manifest_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
