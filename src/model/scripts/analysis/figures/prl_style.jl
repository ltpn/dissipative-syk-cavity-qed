# Shared PRL-quality style for the final production figures.
#
# Loaded by `figure1_sigma_sff_dsff.jl`, `figure2_dynamics.jl`, and
# `figure3_multimode_unfolded.jl`.  Supports both PRL layouts:
#                                        with 32 pt axis labels / 26 pt
#                                        tick labels / 2.2 pt lines.
#   `apply_prl_theme!(:one_column)`  -- native 86 mm (~246 pt) wide with
#                                        8 pt axis labels / 7 pt tick
#                                        labels / 0.9 pt lines so the
#                                        figure prints legibly at
#                                        100 % scale without further
#                                        downscaling.
#
# Fonts: Latin Modern Roman (regular / italic / bold) for text; LaTeX
# math is rendered via `LaTeXStrings` -> Makie's MathTeX engine, which
# uses the Computer Modern math families that visually match Latin
# Modern Roman.

module PRLStyle

using CairoMakie
using LaTeXStrings

export prl_theme, apply_prl_theme!, PRL_ETA_COLOR, PRL_GAMMA_COLOR,
       PRL_ETA_CMAP, PRL_GAMMA_CMAP, PRL_ETA_HIGHLIGHT,
       PRL_LINEWIDTH_MAIN, PRL_LINEWIDTH_REF, prl_linewidths,
       lm_reg, lm_bold, lm_italic

# ---------------------------------------------------------------------
# Font handles.  These correspond to CairoMakie / fontconfig look-ups
# on the machine that generated the figures (Latin Modern Roman is
# provided by the `texlive-lm` package under /usr/share/fonts).
# ---------------------------------------------------------------------
const lm_reg    = "Latin Modern Roman"
const lm_italic = "Latin Modern Roman Italic"
const lm_bold   = "Latin Modern Roman Bold"

# Default typographic sizes (in points at the CairoMakie unit level).
# figure remains readable at ~12 pt when embedded in an A4 document at
# 100 % scale.  One-column PRL (86 mm ~= 246 pt) uses proportionally
# smaller fonts and strokes so a natively-sized figure prints legibly
# without further scaling.
const _AXIS_LABEL_SIZE   = 32
const _AXIS_TITLE_SIZE   = 32
const _TICK_LABEL_SIZE   = 26
const _LEGEND_LABEL_SIZE = 30
const _LEGEND_TITLE_SIZE = 32
const _CB_LABEL_SIZE     = 32
const _CB_TICK_SIZE      = 26

const _AXIS_LABEL_SIZE_1COL   = 8
const _AXIS_TITLE_SIZE_1COL   = 8
const _TICK_LABEL_SIZE_1COL   = 7
const _LEGEND_LABEL_SIZE_1COL = 7
const _LEGEND_TITLE_SIZE_1COL = 8
const _CB_LABEL_SIZE_1COL     = 8
const _CB_TICK_SIZE_1COL      = 7

# Line widths (mode-dependent; the constants below hold the two-column
# values used by the older figures; `prl_linewidths(mode)` returns the
# right pair for one- vs two-column layouts).
const PRL_LINEWIDTH_MAIN = 2.2
const PRL_LINEWIDTH_REF  = 2.6
const _LW_MAIN_1COL      = 0.9
const _LW_REF_1COL       = 1.1

# Colormaps used consistently across the three figures.
const PRL_ETA_CMAP   = :viridis
const PRL_GAMMA_CMAP = :plasma

"""
    prl_linewidths(mode::Symbol = :two_column) -> NamedTuple

Return `(main, ref)` line widths appropriate for the requested PRL
layout.  `:two_column` is the historical default (2.2 / 2.6 pt);
`:one_column` returns 0.9 / 1.1 pt so lines stay legible in a native
86 mm figure without being fattened by print-shop shrinkage.
"""
function prl_linewidths(mode::Symbol = :two_column)
    mode === :one_column && return (main = _LW_MAIN_1COL, ref = _LW_REF_1COL)
    mode === :two_column && return (main = PRL_LINEWIDTH_MAIN, ref = PRL_LINEWIDTH_REF)
    error("Unknown PRL layout mode: $mode. Use :one_column or :two_column.")
end

"""
    prl_theme(mode::Symbol = :two_column) -> Theme

Return a `Makie.Theme` bundling Latin Modern fonts, mode-appropriate
sizes, and spare aesthetics suitable for PRL production figures.

`mode = :two_column` (default) — historical settings for a ~7 in wide
figure that will be printed at 100 % scale on A4.
`mode = :one_column` — sizes scaled for a native 86 mm (~246 pt) figure
that prints without further downscaling.
"""
function prl_theme(mode::Symbol = :two_column)
    if mode === :one_column
        axis_label   = _AXIS_LABEL_SIZE_1COL
        axis_title   = _AXIS_TITLE_SIZE_1COL
        tick_label   = _TICK_LABEL_SIZE_1COL
        legend_label = _LEGEND_LABEL_SIZE_1COL
        legend_title = _LEGEND_TITLE_SIZE_1COL
        cb_label     = _CB_LABEL_SIZE_1COL
        cb_tick      = _CB_TICK_SIZE_1COL
        # Extra right-side figure padding so the rightmost log-x tick
        # label (e.g. 10^4) can render outside the axis rectangle
        # without being clipped by the figure boundary.
        fig_pad      = (4, 12, 3, 3)
        spine        = 0.6
        tick_w       = 0.5;   tick_size    = 3
        mtick_w      = 0.4;   mtick_size   = 1.6
        cb_tick_w    = 0.5;   cb_tick_size = 3
        legend_pad   = (3, 3, 2, 2)
    elseif mode === :two_column
        axis_label   = _AXIS_LABEL_SIZE
        axis_title   = _AXIS_TITLE_SIZE
        tick_label   = _TICK_LABEL_SIZE
        legend_label = _LEGEND_LABEL_SIZE
        legend_title = _LEGEND_TITLE_SIZE
        cb_label     = _CB_LABEL_SIZE
        cb_tick      = _CB_TICK_SIZE
        fig_pad      = (12, 12, 6, 6)
        spine        = 1.4
        tick_w       = 1.2;   tick_size    = 8
        mtick_w      = 1.0;   mtick_size   = 4
        cb_tick_w    = 1.2;   cb_tick_size = 8
        legend_pad   = (6, 6, 4, 4)
    else
        error("Unknown PRL layout mode: $mode. Use :one_column or :two_column.")
    end

    return Theme(
        # Fonts (regular / bold / italic / bold_italic).  `Makie` expects
        # a NamedTuple with those four keys.
        fonts = (
            regular = lm_reg,
            bold = lm_bold,
            italic = lm_italic,
            bold_italic = lm_bold,
        ),
        fontsize = axis_label,
        figure_padding = fig_pad,
        Axis = (
            xlabelsize = axis_label,
            ylabelsize = axis_label,
            titlesize  = axis_title,
            xticklabelsize = tick_label,
            yticklabelsize = tick_label,
            xgridvisible = false,
            ygridvisible = false,
            spinewidth = spine,
            xtickwidth = tick_w,
            ytickwidth = tick_w,
            xminortickwidth = mtick_w,
            yminortickwidth = mtick_w,
            xticksize = tick_size,
            yticksize = tick_size,
            xminorticksize = mtick_size,
            yminorticksize = mtick_size,
            xminorticksvisible = true,
            yminorticksvisible = true,
            xtickalign = 1.0,
            ytickalign = 1.0,
            xminortickalign = 1.0,
            yminortickalign = 1.0,
        ),
        Colorbar = (
            labelsize = cb_label,
            ticklabelsize = cb_tick,
            spinewidth = spine,
            tickwidth = cb_tick_w,
            ticksize = cb_tick_size,
            minorticksvisible = true,
        ),
        Legend = (
            labelsize = legend_label,
            titlesize = legend_title,
            framevisible = false,
            padding = legend_pad,
        ),
        Label = (
            fontsize = axis_label,
        ),
    )
end

"""
    apply_prl_theme!(mode::Symbol = :two_column)

Set the current Makie theme to the PRL style.  Called by every driver
script before any `Figure(...)` construction.  Pass `:one_column` for
native 86 mm PRL figures; the default `:two_column` preserves the
sizes used by pre-2026-08 figures.
"""
apply_prl_theme!(mode::Symbol = :two_column) = set_theme!(prl_theme(mode))

# ---------------------------------------------------------------------
# Curve colors keyed by index / grid size.  These share the colormap
# choices of the underlying analysis scripts so plots remain visually
# consistent with older reports.
# ---------------------------------------------------------------------

function PRL_ETA_COLOR(idx::Integer, n::Integer)
    n <= 1 && return RGBf(0.27, 0.0, 0.33)
    cmap = cgrad(PRL_ETA_CMAP)
    return cmap[(idx - 1) / max(n - 1, 1)]
end

function PRL_GAMMA_COLOR(idx::Integer, n::Integer)
    n <= 1 && return RGBf(0.05, 0.03, 0.53)
    cmap = cgrad(PRL_GAMMA_CMAP)
    return cmap[(idx - 1) / max(n - 1, 1)]
end

# For accent panels where only a handful of eta values are shown, we use
# a slightly compressed viridis subrange so the endpoints stay legible.
function PRL_ETA_HIGHLIGHT(idx::Integer, n::Integer)
    n <= 1 && return RGBf(0.27, 0.0, 0.33)
    cmap = cgrad(PRL_ETA_CMAP)
    # avoid the extreme yellow that clashes with axis backgrounds
    t = 0.05 + 0.85 * (idx - 1) / max(n - 1, 1)
    return cmap[t]
end

end # module PRLStyle
