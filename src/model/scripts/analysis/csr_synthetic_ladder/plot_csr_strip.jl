#!/usr/bin/env julia
# plot_csr_strip.jl
#
# Compact CSR figure for PRL: four physical panels (a--d) on the top
# row, four multimode panels (e--h) on the bottom row, and the
# real-SYK4 BDI-dagger control (i) spanning the two rows at right.
#
# Each panel shows the CSR density heatmap over the unit disk with the
# panel letter in the top-left. Missing multimode spectra are rendered
#
# Usage:
#     julia --project=src/environment \
#         src/model/scripts/analysis/csr_synthetic_ladder/plot_csr_strip.jl \
#         --n-seeds 128
#
# Outputs:
#     spectral_statistics__csr_strip__physical_eta_L3b.pdf/png
#     spectral_statistics__csr_strip__physical_eta_L3b__plotdata.jld2
#     spectral_statistics__csr_strip__physical_eta_L3b__manifest.toml

if Sys.isapple()
    @eval using AppleAccelerate
end

using CairoMakie
using JLD2
using LaTeXStrings
using LinearAlgebra
using Printf
using Statistics: mean, median, std, quantile
using TOML
include(joinpath(@__DIR__, "..", "..", "..", "DatasetPaths.jl"))

const HERE = @__DIR__
const LAMB_DIR = abspath(joinpath(HERE, "..", "..", ".."))
const REPO_ROOT = abspath(joinpath(LAMB_DIR, "..", ".."))
const FIGURE_ROOT = joinpath(REPO_ROOT, "figures")

include(joinpath(LAMB_DIR, "scripts", "analysis", "figures", "prl_style.jl"))
include(joinpath(HERE, "csr_helpers.jl"))
include(joinpath(LAMB_DIR, "CSRStripPlotData.jl"))
using .PRLStyle
using .CSRStripPlotData

# ------------------------------------------------------------------
# Defaults
# ------------------------------------------------------------------

const DEFAULT_PHYSICAL_DIR =
    "data/physical/n10_f3/merged/spectra"
const DEFAULT_CONTROL_DIR =
    "data/syk4/n10_f3/merged/spectra"
const DEFAULT_OUTPUT_DIR =
    FIGURE_ROOT
const DEFAULT_MULTIMODE_CACHE_DIR =
    "data/multimode/n10_f3"
const DEFAULT_N_SEEDS = 128
const DEFAULT_GAMMA   = 1.0
const DEFAULT_TOL     = 1.0e-10

# Four physical, four multimode, and one real-SYK4 control panel.
const TOP_PANELS = (
    (kind = :physical, label = "physical", tag = "phys_eta0p1", eta = 0.1),
    (kind = :physical, label = "physical", tag = "phys_eta0p4", eta = 0.4),
    (kind = :physical, label = "physical", tag = "phys_eta1p0", eta = 1.0),
    (kind = :physical, label = "physical", tag = "phys_eta2p0", eta = 2.0),
)
const BOTTOM_PANELS = (
    (kind = :multimode, label = "multimode", tag = "mm_dtilde10",   delta_tilde = 10.0),
    (kind = :multimode, label = "multimode", tag = "mm_dtilde1",    delta_tilde = 1.0),
    (kind = :multimode, label = "multimode", tag = "mm_dtilde0p1",  delta_tilde = 0.1),
    (kind = :multimode, label = "multimode", tag = "mm_dtilde0p01", delta_tilde = 0.01),
)
const CONTROL_PANEL =
    (kind = :ladder, label = "synthetic_l3b_bdi_syk4", tag = "l3b", eta = 2.0)
const STRIP_PANELS = (TOP_PANELS..., BOTTOM_PANELS..., CONTROL_PANEL)
const TOP_TAGS = [panel.tag for panel in TOP_PANELS]
const BOTTOM_TAGS = [panel.tag for panel in BOTTOM_PANELS]
const PANEL_ORDER = vcat(TOP_TAGS, BOTTOM_TAGS, [CONTROL_PANEL.tag])
const CSR_STRIP_BASE = "spectral_statistics__csr_strip__physical_eta_L3b"

# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------

_fmt(x::Real)    = string(Float64(x))
_fmt(x::Integer) = string(x)

function physical_seed_path(dir::AbstractString, eta::Real, gamma::Real,
                            tol::Real, seed::Integer)
    fname = @sprintf("physical__eta=%s__gamma=%s__tol=%s__seed=%d.jld2",
                     _fmt(eta), _fmt(gamma), _fmt(tol), seed)
    return joinpath(dir, fname)
end

function ladder_seed_path(dir::AbstractString, label::AbstractString,
                          eta::Real, gamma::Real, tol::Real, seed::Integer)
    fname = @sprintf("%s__eta=%s__gamma=%s__tol=%s__seed=%d.jld2",
                     label, _fmt(eta), _fmt(gamma), _fmt(tol), seed)
    return joinpath(dir, fname)
end

function load_pool_with_seeds(seed_paths::Vector{String})
    seeds = Int[]
    spectra = Vector{Vector{ComplexF64}}()
    for (seed, path) in enumerate(seed_paths)
        isfile(path) || continue
        values = JLD2.jldopen(path, "r") do file
            haskey(file, "L_eigvals") || return nothing
            ComplexF64.(file["L_eigvals"])
        end
        values === nothing && continue
        push!(seeds, seed)
        push!(spectra, values)
    end
    return (seeds = seeds, spectra = spectra)
end

_dtilde_slug(value::Real) = replace(@sprintf("%.6g", value), "." => "p", "-" => "m")

# 1 MHz, i.e. kappa/Delta = 0.2, the same value as the single-mode
# spontaneous-emission column of Table S1.
const MULTIMODE_DELTA_CD = 1.0

function multimode_cache_path(cache_dir::AbstractString, delta_tilde::Real,
                              n_orb::Integer, filling::Integer)
    filling_tag = @sprintf("__f=%d", filling)
    filename = @sprintf(
        "multimode_centered__norb=%d%s__ngrid=160__M=300__deltacd=%s__dtilde=%s.jld2",
        n_orb, filling_tag, _dtilde_slug(MULTIMODE_DELTA_CD),
        _dtilde_slug(delta_tilde))
    return joinpath(cache_dir, filename)
end

function load_multimode_pool(path::AbstractString, n_seeds::Integer)
    isfile(path) || return nothing
    return JLD2.jldopen(path, "r") do file
        haskey(file, "seeds") && haskey(file, "L_eigvals") || return nothing
        seeds = Int.(file["seeds"])
        spectra = [ComplexF64.(values) for values in file["L_eigvals"]]
        length(seeds) == length(spectra) || error(
            "multimode cache seed/spectrum mismatch: $path")
        keep = findall(seed -> seed <= Int(n_seeds), seeds)
        return (seeds = seeds[keep], spectra = spectra[keep])
    end
end

function compute_csr_per_seed(pool_evs::AbstractVector{<:AbstractVector{<:Complex}})
    per_seed = Vector{Vector{ComplexF64}}(undef, length(pool_evs))
    Threads.@threads for index in eachindex(pool_evs)
        filtered = filter(z -> abs(z) > STEADY_TOL, pool_evs[index])
        ratios, _centroid, _rmax, _nbulk, _nfull, _npre =
            complex_spacing_ratios_bulk(filtered)
        per_seed[index] = ComplexF64.(ratios)
    end
    return per_seed
end

function summarize_csr_seed_pools(seed_pools::AbstractVector{<:AbstractVector{<:Complex}})
    n_ratios = sum(length, seed_pools)
    mphi_seed = Float64[]
    r_seed = Float64[]
    pooled_abs_sum = 0.0
    pooled_cos_sum = 0.0
    n_valid = 0
    for seed_ratios in seed_pools
        isempty(seed_ratios) && continue
        zs = ComplexF64.(seed_ratios)
        mask = abs.(zs) .> 0
        any(mask) || continue
        zs_ok = zs[mask]
        push!(mphi_seed, -mean(cos.(angle.(zs_ok))))
        push!(r_seed, mean(abs.(zs_ok)))
        pooled_abs_sum += sum(abs, zs_ok)
        pooled_cos_sum += sum(cos, angle.(zs_ok))
        n_valid += length(zs_ok)
    end
    return (
        n_used_seeds = length(mphi_seed),
        n_ratios = n_ratios,
        mphi_mean = n_valid == 0 ? NaN : -pooled_cos_sum / n_valid,
        mphi_sem = length(mphi_seed) > 1 ?
            std(mphi_seed) / sqrt(length(mphi_seed)) : NaN,
        r_mean = n_valid == 0 ? NaN : pooled_abs_sum / n_valid,
        r_sem = length(r_seed) > 1 ?
            std(r_seed) / sqrt(length(r_seed)) : NaN,
    )
end

# ------------------------------------------------------------------
# Strip plotter
# ------------------------------------------------------------------

function plot_strip(csr_by_tag::AbstractDict{String,<:AbstractVector{<:AbstractVector{<:Complex}}},
                    panel_order::Vector{String},
                    output_dir::AbstractString;
                    suffix::AbstractString = "",
                    n_bins::Integer = 160)
    mkpath(output_dir)
    PRLStyle.apply_prl_theme!()
    panel_order == PANEL_ORDER || error(
        "plot_strip panel order mismatch: expected $(PANEL_ORDER), got $panel_order")
    layout = csr_strip_layout(TOP_TAGS, BOTTOM_TAGS, CONTROL_PANEL.tag)
    fig = Figure(size = (2160, 820),
                 figure_padding = (140, 120, 16, 16))

    edges_x = collect(range(-1.0, 1.0; length = Int(n_bins) + 1))
    edges_y = collect(range(-1.0, 1.0; length = Int(n_bins) + 1))
    dx = 2.0 / Int(n_bins)
    bin_area = dx * dx
    xc = @. 0.5 * (edges_x[1:end-1] + edges_x[2:end])
    yc = @. 0.5 * (edges_y[1:end-1] + edges_y[2:end])

    density_by_tag = Dict{String,Matrix{Float64}}()
    stats_by_tag   = Dict{String,NamedTuple}()

    for tag in keys(csr_by_tag)
        seed_pools = csr_by_tag[tag]

        pool = ComplexF64[]
        for seed_ratios in seed_pools
            append!(pool, seed_ratios)
        end
        xs = Float64.(real.(pool)); ys = Float64.(imag.(pool))
        H = zeros(Float64, Int(n_bins), Int(n_bins))
        N_z = length(xs)
        @inbounds for k in 1:N_z
            x = xs[k]; y = ys[k]
            (abs(x) > 1.0 || abs(y) > 1.0) && continue
            i = clamp(Int(floor((x + 1.0) / dx)) + 1, 1, Int(n_bins))
            j = clamp(Int(floor((y + 1.0) / dx)) + 1, 1, Int(n_bins))
            H[i, j] += 1.0
        end
        density_by_tag[tag] = N_z > 0 ? H ./ (N_z * bin_area) :
                                          zeros(Float64, Int(n_bins), Int(n_bins))

        stats_by_tag[tag] = summarize_csr_seed_pools(seed_pools)
    end

    isempty(density_by_tag) && error("plot_strip has no populated CSR panels")
    cmax = maximum(quantile(vec(density), 0.995) for density in values(density_by_tag))
    if !isfinite(cmax) || cmax <= 0
        cmax = 1.0
        @warn "plot_strip: all densities are zero; using placeholder cmax=1"
    end

    # Font sizes tuned for full-page PRL width once scaled:
    # figure is authored at 1800 pt wide; when PRL scales to 504 pt (7 in),
    # each declared font size gets multiplied by 504/1800 = 0.28.
    #   axis label 28 pt -> ~ 7.8 pt printed
    #   tick label 28 pt -> ~ 7.8 pt printed
    #   panel letter 28 pt -> ~ 7.8 pt printed
    #   stats annotation 22 pt -> ~ 6.2 pt printed
    axis_label_size    = 28
    tick_label_size    = 28
    letter_size        = 28
    stats_size         = 22
    colorbar_label_size = 26
    colorbar_tick_size  = 28

    hm_ref = nothing
    control_ax = nothing
    for panel in layout
        tag = panel.tag
        populated = haskey(density_by_tag, tag)
        ylabel = panel.col == 1 ? L"\mathrm{Im}\,\zeta" : ""
        position = panel.row == 0 ? fig[1:2, panel.col] : fig[panel.row, panel.col]
        ax = Axis(position;
                  xlabel = panel.row == 1 ? "" : L"\mathrm{Re}\,\zeta",
                  ylabel = ylabel,
                  xlabelsize = axis_label_size,
                  ylabelsize = axis_label_size,
                  xticklabelsize = tick_label_size,
                  yticklabelsize = tick_label_size,
                  xticks = ([-1.0, -0.5, 0.0, 0.5, 1.0], ["-1", "", "0", "", "1"]),
                  yticks = ([-1.0, -0.5, 0.0, 0.5, 1.0], ["-1", "", "0", "", "1"]),
                  titlevisible = false,
                  backgroundcolor = :black,
                  aspect = DataAspect())
        if panel.row == 0
            control_ax = ax
        end
        xlims!(ax, -1.05, 1.05); ylims!(ax, -1.05, 1.05)
        if panel.row == 1
            hidexdecorations!(ax; grid = false)
        end
        if panel.col > 1
            hideydecorations!(ax; ticks = false, minorticks = false, grid = false)
        end
        if populated
            hm = heatmap!(ax, xc, yc, density_by_tag[tag];
                          colormap = :inferno,
                          colorrange = (0.0, cmax))
            hm_ref = hm
        end
        θref = range(0, 2π; length = 400)
        lines!(ax, cos.(θref), sin.(θref); color = :white, linewidth = 1.2,
               linestyle = :dash)
        text!(ax, 0.025, 0.985;
              text = "($(panel.letter))",
              space = :relative,
              color = :white, fontsize = letter_size,
              font = :bold,
              align = (:left, :top))
    end

    isnothing(control_ax) && error("plot_strip: centered control axis was not constructed")
    control_height = lift(control_ax.scene.viewport) do viewport
        Float64(widths(viewport)[2])
    end
    Colorbar(fig[1:2, 6], hm_ref;
             label = L"P_{\mathrm{CSR}}(\zeta)",
             labelsize = colorbar_label_size,
             ticklabelsize = colorbar_tick_size,
             height = control_height,
             valign = :center,
             width = 22)
    colsize!(fig.layout, 6, Fixed(22))
    colgap!(fig.layout, 12)
    rowgap!(fig.layout, -20)

    base = CSR_STRIP_BASE
    pdf_path = joinpath(output_dir, "$(base)$(suffix).pdf")
    png_path = joinpath(output_dir, "$(base)$(suffix).png")
    save(pdf_path, fig)
    save(png_path, fig; px_per_unit = 3)
    println("wrote $pdf_path")
    println("wrote $png_path")
    return (pdf = pdf_path, png = png_path,
            stats_by_tag = stats_by_tag, base = base)
end

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

function parse_cli(argv)
    physical_dir = DEFAULT_PHYSICAL_DIR
    control_dir  = DEFAULT_CONTROL_DIR
    multimode_cache_dir = DEFAULT_MULTIMODE_CACHE_DIR
    output_dir   = DEFAULT_OUTPUT_DIR
    n_seeds      = DEFAULT_N_SEEDS
    gamma        = DEFAULT_GAMMA
    tol          = DEFAULT_TOL
    suffix       = ""
    n_orb = 10
    filling::Union{Nothing,Int} = 3
    rebuild_plotdata = false
    update_pending_panels = false
    i = 1
    while i <= length(argv)
        a = argv[i]
        if a == "--physical-dir";  i += 1; physical_dir = argv[i]
        elseif a == "--control-dir"; i += 1; control_dir = argv[i]
        elseif a == "--multimode-cache-dir"; i += 1; multimode_cache_dir = argv[i]
        elseif a == "--output-dir"; i += 1; output_dir = argv[i]
        elseif a == "--n-seeds"; i += 1; n_seeds = parse(Int, argv[i])
        elseif a == "--n-orb"; i += 1; n_orb = parse(Int, argv[i])
        elseif a == "--filling"; i += 1; filling = parse(Int, argv[i])
        elseif a == "--gamma"; i += 1; gamma = parse(Float64, argv[i])
        elseif a == "--tol"; i += 1; tol = parse(Float64, argv[i])
        elseif a == "--suffix"; i += 1; suffix = argv[i]
        elseif a == "--rebuild-plotdata"; rebuild_plotdata = true
        elseif a == "--update-pending-panels"; update_pending_panels = true
        else
            error("unrecognized argument: $a")
        end
        i += 1
    end
    return (physical_dir = physical_dir, control_dir = control_dir,
            multimode_cache_dir = multimode_cache_dir,
            output_dir = output_dir, n_seeds = n_seeds,
            gamma = gamma, tol = tol, suffix = suffix, n_orb = n_orb,
            filling = filling, rebuild_plotdata = rebuild_plotdata,
            update_pending_panels = update_pending_panels)
end

function load_panel_spectra(panel, opts, filling::Integer, panel_meta)
    if panel.kind == :physical
        seed_paths =
            [physical_seed_path(opts.physical_dir, panel.eta, opts.gamma, opts.tol, seed)
             for seed in 1:opts.n_seeds]
        panel_meta["source_path"] = opts.physical_dir
        return load_pool_with_seeds(seed_paths)
    elseif panel.kind == :ladder
        spectra_dir = opts.control_dir
        seed_paths =
            [ladder_seed_path(spectra_dir, panel.label, panel.eta,
                              opts.gamma, opts.tol, seed)
             for seed in 1:opts.n_seeds]
        panel_meta["source_path"] = spectra_dir
        return load_pool_with_seeds(seed_paths)
    else
        source = multimode_cache_path(opts.multimode_cache_dir,
                                      panel.delta_tilde,
                                      opts.n_orb, filling)
        panel_meta["source_path"] = source
        return load_multimode_pool(source, opts.n_seeds)
    end
end

function record_spectrum_metadata!(panel_meta, loaded)
    panel_meta["n_seeds_loaded"] = length(loaded.spectra)
    panel_meta["seeds"] = loaded.seeds
    panel_meta["n_eigenvalues"] = sum(length, loaded.spectra)
    panel_meta["eigenvalues_per_seed"] = sort!(unique(length.(loaded.spectra)))
    return panel_meta
end

function main(argv = ARGS)
    opts = parse_cli(argv)
    filling = opts.filling === nothing ? div(opts.n_orb, 2) : Int(opts.filling)
    println("[csr-strip] physical_dir=$(opts.physical_dir)")
    println("[csr-strip] control_dir=$(opts.control_dir)")
    println("[csr-strip] multimode_cache_dir=$(opts.multimode_cache_dir)")
    println("[csr-strip] output_dir=$(opts.output_dir)")
    println("[csr-strip] n_seeds=$(opts.n_seeds)  gamma=$(opts.gamma)  tol=$(opts.tol)  n_orb=$(opts.n_orb)  filling=$(opts.filling)")
    plotdata_path = joinpath(opts.output_dir,
        "$(CSR_STRIP_BASE)$(opts.suffix)__plotdata.jld2")
    csr_by_tag  = Dict{String,Vector{Vector{ComplexF64}}}()
    per_panel_meta = Dict{String,Any}()
    cache_loaded = false
    expected_spectrum_dim = binomial(opts.n_orb, filling)^2
    constants = Dict{String,Any}(
        "steady_tol" => STEADY_TOL,
        "distance_tol" => DISTANCE_TOL,
        "bulk_fraction" => CSR_BULK_FRACTION,
        "rmax_quantile" => CSR_RMAX_QUANTILE,
        "im_axis_tol" => CSR_IM_AXIS_TOL,
    )
    cache_identity = Dict{String,Any}(
        "n_orb" => opts.n_orb,
        "filling" => filling,
        "n_seeds" => opts.n_seeds,
        "gamma" => opts.gamma,
        "spectrum_tol" => opts.tol,
        "physical_dir" => abspath(opts.physical_dir),
        "control_dir" => abspath(opts.control_dir),
        "multimode_cache_dir" => abspath(opts.multimode_cache_dir),
    )

    if isfile(plotdata_path) && !opts.rebuild_plotdata
        cached = read_csr_strip_plotdata(plotdata_path)
        cached.panel_order == PANEL_ORDER || error(
            "cached CSR panel order does not match the current layout: $plotdata_path")
        cached.constants == constants || error(
            "cached CSR constants do not match current diagnostic constants: $plotdata_path")
        validate_cache_identity(cached.cache_identity, cache_identity)
        csr_by_tag = cached.csr_by_tag
        per_panel_meta = cached.panel_meta
        cache_loaded = true
        println("[csr-strip] loaded plot data: $plotdata_path")
    end

    for panel in STRIP_PANELS
        haskey(per_panel_meta, panel.tag) || begin
            parameter_meta = panel.kind == :multimode ?
                Dict{String,Any}("delta_tilde" => panel.delta_tilde) :
                Dict{String,Any}("eta" => panel.eta)
            per_panel_meta[panel.tag] = merge(parameter_meta, Dict{String,Any}(
                "label" => panel.label,
                "status" => "pending_spectra",
                "n_seeds_requested" => opts.n_seeds,
                "n_seeds_loaded" => 0,
                "seeds" => Int[],
            ))
        end
        should_compute = should_compute_csr_panel(
            cache_loaded, String(per_panel_meta[panel.tag]["status"]),
            haskey(csr_by_tag, panel.tag), opts.update_pending_panels)
        if !should_compute
            continue
        end

        loaded = load_panel_spectra(panel, opts, filling,
                                    per_panel_meta[panel.tag])
        if loaded === nothing || isempty(loaded.spectra)
            println("[csr-strip] $(panel.tag): spectra pending; rendering empty panel")
            continue
        end
        validate_panel_spectra(loaded.seeds, loaded.spectra,
                               opts.n_seeds, expected_spectrum_dim)
        println("[csr-strip] $(panel.tag): n_seeds_loaded=$(length(loaded.spectra))")
        t_csr = time()
        per_seed = compute_csr_per_seed(loaded.spectra)
        @printf("[csr-strip]   CSR compute: %.1fs on %d threads\n",
                time() - t_csr, Threads.nthreads())
        csr_by_tag[panel.tag] = per_seed
        per_panel_meta[panel.tag]["status"] = "complete"
        record_spectrum_metadata!(per_panel_meta[panel.tag], loaded)
    end

    for (tag, seed_pools) in csr_by_tag
        stats = summarize_csr_seed_pools(seed_pools)
        per_panel_meta[tag]["n_used_seeds"] = stats.n_used_seeds
        per_panel_meta[tag]["n_ratios"] = stats.n_ratios
        per_panel_meta[tag]["r_mean"] = stats.r_mean
        per_panel_meta[tag]["r_sem"] = stats.r_sem
        per_panel_meta[tag]["mphi_mean"] = stats.mphi_mean
        per_panel_meta[tag]["mphi_sem"] = stats.mphi_sem
    end
    write_csr_strip_plotdata(plotdata_path, csr_by_tag, PANEL_ORDER,
                             per_panel_meta, constants;
                             cache_identity = cache_identity)
    pending_tags = [tag for tag in PANEL_ORDER
                    if per_panel_meta[tag]["status"] == "pending_spectra"]
    println("[csr-strip] pending panels: $(pending_tags)")
    result = plot_strip(csr_by_tag, PANEL_ORDER, opts.output_dir;
                        suffix = opts.suffix)

    manifest = Dict{String,Any}(
        "gamma"           => opts.gamma,
        "tol"             => opts.tol,
        "n_orb"           => opts.n_orb,
        "filling"         => filling,
        "physical_dir"    => opts.physical_dir,
        "control_dir"     => opts.control_dir,
        "multimode_cache_dir" => opts.multimode_cache_dir,
        "plotdata_jld2"   => plotdata_path,
        "panel_order"     => PANEL_ORDER,
        "pending_tags"    => pending_tags,
        "status"          => isempty(pending_tags) ? "complete" : "pending_spectra",
        "output_pdf"      => result.pdf,
        "output_png"      => result.png,
        "n_seeds_requested" => opts.n_seeds,
        "presentation" => Dict(
            "tick_label_size" => 28,
            "colorbar_tick_size" => 28,
            "panel_gap" => 12,
            "row_gap" => -20,
            "colorbar_column_width" => 22,
            "panel_letter_position" => [0.025, 0.985],
        ),
        "per_panel"       => per_panel_meta,
    )
    for (tag, stats) in result.stats_by_tag
        per_panel_meta[tag]["n_used_seeds"] = stats.n_used_seeds
        per_panel_meta[tag]["n_ratios"]     = stats.n_ratios
        per_panel_meta[tag]["r_mean"]       = isfinite(stats.r_mean) ? stats.r_mean : NaN
        per_panel_meta[tag]["r_sem"]        = isfinite(stats.r_sem)  ? stats.r_sem  : NaN
        per_panel_meta[tag]["mphi_mean"]    = isfinite(stats.mphi_mean) ? stats.mphi_mean : NaN
        per_panel_meta[tag]["mphi_sem"]     = isfinite(stats.mphi_sem)  ? stats.mphi_sem  : NaN
    end

    manifest_path = joinpath(opts.output_dir,
        "$(result.base)$(opts.suffix)__manifest.toml")
    open(manifest_path, "w") do io
        DatasetPaths.print_manifest(io, manifest; sorted=true, path=manifest_path)
    end
    println("wrote $manifest_path")
    return 0
end

# When run as a script (`julia plot_csr_strip.jl ...`) exit with the
# main return code so the shell sees a proper exit status. When the
# file is `include`d from a REPL, just call main() so the caller
# retains an interactive prompt and the in-memory cache stays alive.
if abspath(PROGRAM_FILE) == @__FILE__
    exit(main(ARGS))
end
