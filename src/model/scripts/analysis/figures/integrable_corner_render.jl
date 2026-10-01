#!/usr/bin/env julia

using CairoMakie
using JLD2
using LaTeXStrings
using Printf
using Random
using Statistics
using TOML

const HERE = @__DIR__
const LAMB_DIR = abspath(joinpath(HERE, "..", "..", ".."))
include(joinpath(LAMB_DIR, "IntegrableCorner.jl"))
include(joinpath(HERE, "MagicLaTeX.jl"))
include(joinpath(HERE, "_common_sff_helpers.jl"))
include(joinpath(HERE, "_figure1_plot_smoothing.jl"))
using .IntegrableCorner
using .MagicLaTeX
using .Figure1PlotSmoothing: sgolay_log_positive

const AUTHORING_CANVAS_SIZE_PT = (601.0, 360.0)
const LATEX_SCALE = 0.495
const SOURCE_PX = points -> MagicLaTeX.pt2px(points / LATEX_SCALE)
const SIZE_SCHEDULE = ((10, 3, 16), (12, 4, 16), (14, 5, 12), (16, 6, 8))
const BOOTSTRAP_SEED = 20260817
const N_BOOTSTRAP = 500

function parse_cli(argv)
    cache_dir = ""
    panel_a = ""
    output_dir = ""
    i = 1
    while i <= length(argv)
        argv[i] == "--help" && (println("Usage: julia integrable_corner_render.jl --cache-dir DIR --panel-a FILE --output-dir DIR"); exit(0))
        i == length(argv) && error("missing value for $(argv[i])")
        if argv[i] == "--cache-dir"; cache_dir = argv[i + 1]
        elseif argv[i] == "--panel-a"; panel_a = argv[i + 1]
        elseif argv[i] == "--output-dir"; output_dir = argv[i + 1]
        else; error("unknown argument: $(argv[i])")
        end
        i += 2
    end
    isempty(cache_dir) && error("--cache-dir is required")
    isempty(panel_a) && error("--panel-a is required")
    isempty(output_dir) && error("--output-dir is required")
    return abspath(cache_dir), abspath(panel_a), abspath(output_dir)
end

cache_path(root, n, filling, seed) = joinpath(root,
    @sprintf("corner__norb=%d__filling=%d__seed=%d.jld2", n, filling, seed))

function load_panel_a(path)
    isfile(path) || error("panel-a cache not found: $path")
    return JLD2.jldopen(path, "r") do file
        Bool(file["passed"]) || error("panel-a validation did not pass: $path")
        return (
            analytic = Vector{Float64}(file["analytic_sigmas"]),
            numeric = Vector{Float64}(file["numeric_sigmas"]),
            residuals = Vector{Float64}(file["absolute_residuals"]),
            normality = Float64(file["normality_residual"]),
            sigma_relative = Float64(file["max_relative_residual"]),
            center_residual = Float64(file["center_residual"]),
            h_residual = Float64(file["hamiltonian_spectrum_residual"]),
        )
    end
end

function load_size(root, n, filling, n_seeds, size_index)
    taus_ref = Float64[]
    rows = Vector{Vector{ComplexF64}}()
    zero_counts = Int[]
    centers = Float64[]
    d_ref = 0
    for seed in 1:n_seeds
        path = cache_path(root, n, filling, seed)
        isfile(path) || error("missing panel-b cache: $path")
        payload = JLD2.jldopen(path, "r") do file
            Bool(file["complete"]) || error("incomplete cache: $path")
            Int(file["n_orb"]) == n || error("n_orb mismatch: $path")
            Int(file["filling"]) == filling || error("filling mismatch: $path")
            Int(file["seed"]) == seed || error("seed mismatch: $path")
            return (
                taus = Vector{Float64}(file["taus"]),
                Z = complex.(Vector{Float64}(file["Z_re"]),
                            Vector{Float64}(file["Z_im"])),
                d = Int(file["d"]), D_b = Int(file["D_b"]),
                zero_count = Int(file["zero_count_le_1e-8"]),
                center = Float64(file["center_c"]),
            )
        end
        isempty(taus_ref) ? (taus_ref = payload.taus) :
            (payload.taus == taus_ref || error("tau-grid mismatch: $path"))
        d_ref == 0 ? (d_ref = payload.d) :
            (payload.d == d_ref || error("d mismatch: $path"))
        payload.D_b == payload.d^2 || error("D_b mismatch: $path")
        push!(rows, payload.Z)
        push!(zero_counts, payload.zero_count)
        push!(centers, payload.center)
    end
    all(iszero, zero_counts) || error("full-spectrum zero diagnostic failed for N=$n")
    Z = reduce(vcat, permutedims.(rows))
    levels = fill(d_ref^2, n_seeds)
    bootstrap = bootstrap_connected_sigma_sff(Z, levels;
        n_boot = N_BOOTSTRAP,
        rng = MersenneTwister(BOOTSTRAP_SEED + size_index))
    late = max(1, Int(ceil(0.8length(taus_ref)))):length(taus_ref)
    late_samples = vec(mean(@view bootstrap.samples[:, late]; dims = 2))
    plateau = 3 - 2 / d_ref
    interval95 = quantile(late_samples, [0.025, 0.975])
    measured = mean(@view bootstrap.curve[late])
    return (
        n_orb = n, filling = filling, n_seeds = n_seeds,
        d = d_ref, D_b = d_ref^2, taus = taus_ref,
        curve = bootstrap.curve, lower = bootstrap.lower,
        upper = bootstrap.upper, plateau = plateau,
        measured_plateau = measured, plateau_interval95 = interval95,
        plateau_in_interval95 = interval95[1] <= plateau <= interval95[2],
        centers = centers,
    )
end

positive(values) = [isfinite(x) && x > 0 ? x : NaN for x in values]
smooth(values) = sgolay_log_positive(positive(values); window = 11, degree = 3)
function magic_log_ticks(powers)
    positions = 10.0 .^ collect(powers)
    labels = [latexstring("10^{", power, "}") for power in powers]
    return positions, labels
end

function pdf_hires_bbox(path::AbstractString)
    gs = Sys.which("gs"); gs === nothing && error("Ghostscript is required")
    buffer = IOBuffer()
    run(pipeline(Cmd([gs, "-q", "-dNOPAUSE", "-dBATCH", "-sDEVICE=bbox", path]);
                 stdout = devnull, stderr = buffer))
    matched = match(r"%%HiResBoundingBox:\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)",
                    String(take!(buffer)))
    matched === nothing && error("could not determine PDF bounding box")
    return parse.(Float64, matched.captures)
end

function crop_pdf_hires(input_path::AbstractString, output_path::AbstractString)
    llx, lly, urx, ury = pdf_hires_bbox(input_path)
    width, height = urx - llx, ury - lly
    gs = Sys.which("gs"); gs === nothing && error("Ghostscript is required")
    run(Cmd([gs, "-q", "-dNOPAUSE", "-dBATCH", "-sDEVICE=pdfwrite",
             "-dCompatibilityLevel=1.5", "-dFIXEDMEDIA",
             "-dDEVICEWIDTHPOINTS=$width", "-dDEVICEHEIGHTPOINTS=$height",
             "-sOutputFile=$output_path", "-c",
             "<</PageOffset [-$llx -$lly]>> setpagedevice", "-f", input_path]))
    return [width, height]
end

function rasterize_pdf_png(pdf_path::AbstractString, png_path::AbstractString)
    gs = Sys.which("gs"); gs === nothing && error("Ghostscript is required")
    run(Cmd([gs, "-q", "-dNOPAUSE", "-dBATCH", "-sDEVICE=pngalpha", "-r600",
             "-dTextAlphaBits=4", "-dGraphicsAlphaBits=4",
             "-sOutputFile=$png_path", pdf_path]))
end

function write_csv(path, series)
    open(path, "w") do io
        println(io, "n_orb,filling,n_seeds,d,D_b,tau,Kc_over_Kinf,lower68,upper68,predicted_plateau")
        for item in series, index in eachindex(item.taus)
            @printf(io, "%d,%d,%d,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                item.n_orb, item.filling, item.n_seeds, item.d, item.D_b,
                item.taus[index], item.curve[index], item.lower[index],
                item.upper[index], item.plateau)
        end
    end
end

function main(argv = ARGS)
    cache_dir, panel_a_path, output_dir = parse_cli(argv)
    panel_a = load_panel_a(panel_a_path)
    series = [load_size(cache_dir, spec..., index)
              for (index, spec) in enumerate(SIZE_SCHEDULE)]
    mkpath(output_dir)
    pdf_path = joinpath(output_dir, "figure_integrable_corner_sigma_sff.pdf")
    png_path = joinpath(output_dir, "figure_integrable_corner_sigma_sff.png")
    csv_path = joinpath(output_dir, "figure_integrable_corner_sigma_sff.csv")
    manifest_path = joinpath(output_dir, "figure_integrable_corner_sigma_sff__manifest.toml")

    base_theme = theme_magiclatex(PaletteName = :gem_2024,
        FigureSize = MagicLaTeX.pt2px.(AUTHORING_CANVAS_SIZE_PT))
    theme = merge(base_theme, Theme(figure_padding = SOURCE_PX.((2.0, 2.0, 1.5, 1.5))))
    physical_color = RGBf(0.0, 0.447, 0.741)
    main_width, reference_width = SOURCE_PX(1.0), SOURCE_PX(1.15)

    fig = with_theme(theme) do
        figure = Figure(backgroundcolor = :white)
        ax_a = Axis(figure[1, 1], xscale = log10, yscale = log10,
            xlabel = L"|\lambda_j-c|", ylabel = L"\sigma_j^{\mathrm{SVD}}",
            xticks = magic_log_ticks(-2:0), yticks = magic_log_ticks(-2:0),
            height = Relative(0.84), valign = :top,
            xgridvisible = false, ygridvisible = false)
        finite = @. isfinite(panel_a.analytic) & (panel_a.analytic > 0) &
                    isfinite(panel_a.numeric) & (panel_a.numeric > 0)
        scatter!(ax_a, panel_a.analytic[finite], panel_a.numeric[finite];
            color = physical_color, markersize = SOURCE_PX(1.3))
        lo = minimum(panel_a.analytic[finite]); hi = maximum(panel_a.analytic[finite])
        lines!(ax_a, [lo, hi], [lo, hi]; color = :black,
            linewidth = reference_width, linestyle = :dash)
        text!(ax_a, 0.02, 0.98; text = L"\mathrm{(a)}", space = :relative,
            align = (:left, :top))
        inset = Axis(figure[1, 1], tellwidth = false, tellheight = false,
            width = Relative(0.40), height = Relative(0.27), halign = 0.91,
            valign = 0.23, alignmode = Inside(), yscale = log10,
            xlabel = "", ylabel = "", xticklabelsvisible = false,
            xminorticksvisible = false, yminorticksvisible = false,
            xticklabelsize = SOURCE_PX(5.0), yticklabelsize = SOURCE_PX(5.0),
            spinewidth = SOURCE_PX(0.5), backgroundcolor = (:white, 0.94))
        residual_floor = max.(panel_a.residuals, eps(Float64))
        scatter!(inset, eachindex(residual_floor), residual_floor;
            color = (RGBf(0.25, 0.25, 0.25), 0.55),
            markersize = SOURCE_PX(0.45))

        right = figure[1, 2] = GridLayout()
        axes_b = Axis[]
        physical_line = nothing
        goe_line = nothing
        plateau_line = nothing
        for (index, item) in enumerate(series)
            row = index <= 2 ? 1 : 2
            col = isodd(index) ? 1 : 2
            ax = Axis(right[row, col], xscale = log10, yscale = log10,
                xgridvisible = false, ygridvisible = false,
                xticklabelsvisible = row == 2, yticklabelsvisible = col == 1,
                xticks = col == 1 ? magic_log_ticks(-3:0) : magic_log_ticks(-2:1),
                xlabel = "", ylabel = "", aspect = 1.05,
                title = latexstring("N=", item.n_orb, ",\\ Q=", item.filling),
                titlesize = SOURCE_PX(6.2), titlegap = SOURCE_PX(0.5),
                xticklabelsize = SOURCE_PX(5.5), yticklabelsize = SOURCE_PX(5.5))
            lower = positive(item.lower); upper = positive(item.upper)
            valid_band = @. isfinite(lower) & isfinite(upper) & (upper >= lower)
            lower[.!valid_band] .= NaN; upper[.!valid_band] .= NaN
            band!(ax, item.taus, lower, upper; color = (physical_color, 0.16))
            physical_line = lines!(ax, item.taus, smooth(item.curve);
                color = physical_color, linewidth = main_width)
            goe_line = lines!(ax, item.taus, folded_goe_form_factor(item.taus);
                color = :black, linewidth = reference_width, linestyle = :dash)
            d = item.d
            plateau_line = hlines!(ax, [3 - 2 / d]; color = :gray25,
                linewidth = reference_width, linestyle = :dot)
            xlims!(ax, 1e-3, 10.0); ylims!(ax, 1e-2, 20.0)
            push!(axes_b, ax)
        end
        text!(axes_b[1], 0.02, 0.98; text = L"\mathrm{(b)}",
            space = :relative, align = (:left, :top))
        legend = Legend(right[1, 1], [physical_line, goe_line, plateau_line],
            ["corner", "f-GOE", L"3-2/d"]; orientation = :vertical,
            framevisible = false, labelsize = SOURCE_PX(5.6),
            patchsize = SOURCE_PX.((7.0, 2.0)), rowgap = SOURCE_PX(0.4),
            patchlabelgap = SOURCE_PX(0.7), tellwidth = false,
            tellheight = false, halign = 0.96, valign = 0.05,
            alignmode = Inside())
        Label(right[3, 1:2], L"t/t_{\mathrm{Hei}}"; fontsize = SOURCE_PX(7.2),
              padding = (0, 0, 0, 0))
        Label(right[1:2, 0], L"\mathrm{\sigma SFF}";
              rotation = pi / 2, fontsize = SOURCE_PX(7.2),
              padding = (0, 0, 0, 0))
        colgap!(figure.layout, SOURCE_PX(3.0))
        colsize!(figure.layout, 1, Relative(0.38))
        colsize!(figure.layout, 2, Relative(0.62))
        rowgap!(right, SOURCE_PX(1.0)); colgap!(right, SOURCE_PX(1.0))
        figure
    end

    source_size = mktempdir() do temporary_dir
        uncropped = joinpath(temporary_dir, "corner-uncropped.pdf")
        save(uncropped, fig)
        crop_pdf_hires(uncropped, pdf_path)
    end
    rasterize_pdf_png(pdf_path, png_path)
    write_csv(csv_path, series)

    manifest = Dict{String,Any}(
        "authoring_canvas_points" => collect(AUTHORING_CANVAS_SIZE_PT),
        "latex_scale" => LATEX_SCALE,
        "cropped_source_size_points" => source_size,
        "gamma_eff_over_J" => 1.0,
        "kappa_eff_over_J" => 0.2,
        "gamma_tot_over_J" => 1.2,
        "staircase_n_bins" => 40,
        "staircase_degree" => 5,
        "bootstrap_draws" => N_BOOTSTRAP,
        "bootstrap_seed" => BOOTSTRAP_SEED,
        "panel_a" => Dict(
            "n_orb" => 10, "filling" => 3,
            "normality_residual" => panel_a.normality,
            "max_relative_sigma_residual" => panel_a.sigma_relative,
            "center_residual" => panel_a.center_residual,
            "hamiltonian_spectrum_residual" => panel_a.h_residual,
        ),
        "panel_b" => [Dict(
            "n_orb" => item.n_orb, "filling" => item.filling,
            "n_seeds" => item.n_seeds, "d" => item.d, "D_b" => item.D_b,
            "predicted_plateau" => item.plateau,
            "measured_late_plateau" => item.measured_plateau,
            "plateau_interval95" => item.plateau_interval95,
            "plateau_in_interval95" => item.plateau_in_interval95,
        ) for item in series],
    )
    open(manifest_path, "w") do io
        TOML.print(io, manifest; sorted = true)
    end
    println("wrote $pdf_path")
    println("wrote $png_path")
    println("wrote $csv_path")
    println("wrote $manifest_path")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
