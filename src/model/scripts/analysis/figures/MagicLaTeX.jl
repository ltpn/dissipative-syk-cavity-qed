# Author: Matteo Seclì
# Copyright (c) 2025-2026 Matteo Seclì

module MagicLaTeX
export theme_magiclatex, palette_magiclatex #, register_palette!


using Makie
using MakieThemes: theme_latexfonts
# using MathTeXEngine: texfont


# See:
# - https://mathworks.com/help/matlab/ref/matlab.ui.figure.html
# - https://mathworks.com/help/matlab/ref/matlab.graphics.axis.axes-properties.html
# - https://mathworks.com/help/matlab/ref/matlab.graphics.illustration.colorbar-properties.html
# - https://it.mathworks.com/help/matlab/ref/matlab.graphics.illustration.legend-properties.html
# - My own magicLaTeX.m MATLAB function


# ---------------------------
# MATLAB color palettes
# ---------------------------

const MatlabPalettes = Dict{Symbol,Vector{String}}(
    :gem            => ["#1171BE", "#DD5400", "#EDB120", "#8516D1", "#3BAA32", "#2FBEEF", "#D1048B"],
    :gem12          => ["#1171BE", "#DD5400", "#EDB120", "#8516D1", "#3BAA32", "#2FBEEF", "#D1048B", "#FFD60A", "#6582FD", "#FF453A", "#00A3A3", "#CB845D"],
    :glow           => ["#268CDD", "#F57729", "#FFE864", "#C05CFB", "#49DB40", "#6CF4FF", "#F267C5"],
    :glow12         => ["#268CDD", "#F57729", "#FFE864", "#C05CFB", "#49DB40", "#6CF4FF", "#F267C5", "#FEC04C", "#7DA9FF", "#FF7A74", "#1FCFBE", "#DC996C"],
    :sail           => ["#104280", "#54B6FF", "#FF453A", "#902622", "#1171BE"],
    :reef           => ["#DD5400", "#54B6FF", "#1171BE", "#FE9043", "#74EBDA", "#00A3A3"],
    :meadow         => ["#02580E", "#3AC831", "#FFD60A", "#F57729", "#C04C0B", "#FA8AD4", "#7DA9FF"],
    :dye            => ["#B7312C", "#3BAA32", "#5E2296", "#1171BE", "#DD5400", "#027880", "#E951B8"],
    :earth          => ["#104280", "#B7312C", "#9C7720", "#02580E", "#DC996C", "#5F1B08", "#FFD19E"],
    :gem_2024       => ["#0072BD", "#D95319", "#EDB120", "#7E2F8E", "#77AC30", "#4DBEEE", "#A2142F"],
    :gem12_2024     => ["#0072BD", "#D95319", "#EDB120", "#7E2F8E", "#77AC30", "#4DBEEE", "#A2142F", "#FFD60A", "#6582FD", "#FF453A", "#00A3A3", "#CB845D"],
)

_palette_key(name)::Symbol = name isa Symbol ? name : Symbol(lowercase(String(name)))

# "Register or overwrite a palette at runtime."
# function register_palette!(name::Union{Symbol,String}, colors::AbstractVector{<:AbstractString})
#     MatlabPalettes[_palette_key(name)] = collect(String, colors)
#     return nothing
# end

"""
    palette_magiclatex(PaletteName::Union{Symbol,String})

Get the MATLAB color palette with name `PaletteName`.
"""
function palette_magiclatex(PaletteName::Union{Symbol,String})
    # Julia 1.12 requires an explicit fallback in `get`.
    return get(MatlabPalettes, _palette_key(PaletteName), MatlabPalettes[:gem_2024])
end

# ---------------------------
# Theme function (backend-agnostic)
# ---------------------------


# Note: Makie uses unitless measurements by default, but they're basically
# px (because px_per_units=1 by default). MATLAB, instead, uses pt (1px = 0.75pt).
# So, we have to scale all the sizes by 4/3 to have the same physical size in pt.
# E.g. FigureSize=(560pt, 420pt) becomes FigureSize=(560*4/3, 420*4/3) in Makie units.
# and FontSize=16pt becomes FontSize=16*4/3 in Makie units.
function pt2px(x::Real)
    return x * 4 / 3
end

"""
    theme_magiclatex(; FontSize=16, LabelFontSizeMultiplier=1.1, FigureSize=(560, 420))

Nice, LaTeX-based, publication-quality plotting theme for Makie.jl.
"""
function theme_magiclatex(;
    FontSize::Real=16*4/3,
    LabelFontSizeMultiplier::Real=1.1,
    TitleFontSizeMultiplier::Real=1.1,
    LegendFontSizeMultiplier::Real=0.9,
    FigureSize=pt2px.((560, 420)),
    PaletteName::Union{Symbol,String}=:auto,
    Dark::Bool=false,
)
    Dark == true && @warn("Dark mode is not complete yet!")
    LabelFontSize = FontSize * LabelFontSizeMultiplier
    TitleFontSize = FontSize * TitleFontSizeMultiplier
    LegendFontSize = FontSize * LegendFontSizeMultiplier
    palettename = PaletteName === :auto ? (Dark ? :glow : :gem) : _palette_key(PaletteName)
    colors = get(MatlabPalettes, palettename, :gem_2024)
    return merge(
        theme_latexfonts(),
        Theme(
            # fonts = Attributes(
            #     :bold => texfont(:bold),
            #     :bolditalic => texfont(:bolditalic),
            #     :italic => texfont(:italic),
            #     :regular => texfont(:regular)
            # ),
            size = FigureSize,
            fontsize = FontSize,
            backgroundcolor = Dark ? RGBf(0.15,0.15,0.15) : :transparent,
            # figure_padding = (0.11*FigureSize[1],
            #                   (0.225-0.11)*FigureSize[1],
            #                   0.13*FigureSize[2],
            #                   (0.185-0.13)*FigureSize[2]), #LRBT
            figure_padding = (0.13/2*FigureSize[1],
                              0.13*FigureSize[1],
                              0.11/2*FigureSize[2],
                              0.11*FigureSize[2]), #LRBT
            Axis     = (xlabelsize = LabelFontSize,
                        ylabelsize = LabelFontSize,
                        titlesize  = TitleFontSize,
                        titlefont  = :regular,
                        spinewidth  = pt2px(1.0),
                        xtickalign = 1,
                        ytickalign = 1,
                        xticksize = pt2px(5.0),
                        yticksize = pt2px(5.0),
                        xtickwidth = pt2px(1.0),
                        ytickwidth = pt2px(1.0),
                        xticksmirrored = true,
                        yticksmirrored = true,
                        # xautolimitmargin = (0, 0),
                        # yautolimitmargin = (0, 0),
                        # alignmode = Mixed(
                        #     left = Makie.Protrusion(0),
                        #     right = Makie.Protrusion(0),
                        #     bottom = Makie.Protrusion(0),
                        #     top = Makie.Protrusion(0),
                        # )
                        # bbox = BBox(0.11*FigureSize[1],
                        #             (1-0.11)*FigureSize[1],
                        #             0.13*FigureSize[2],
                        #             (1-0.13)*FigureSize[2])
                        ),
            Axis3    = (xlabelsize = LabelFontSize,
                        ylabelsize = LabelFontSize,
                        titlesize  = TitleFontSize,
                        titlefont  = :regular,
                        spinewidth  = pt2px(1.0),
                        # xtickalign=1,
                        # ytickalign=1,
                        # ztickalign=1,
                        xticksize = pt2px(5.0),
                        yticksize = pt2px(5.0),
                        zticksize = pt2px(5.0),
                        xtickwidth = pt2px(1.0),
                        ytickwidth = pt2px(1.0),
                        ztickwidth = pt2px(1.0),
                        # xticksmirrored = true,
                        # yticksmirrored = true,
                        # zticksmirrored = true,
                        # xautolimitmargin = (0, 0),
                        # yautolimitmargin = (0, 0),
                        # zautolimitmargin = (0, 0),
                        ),
            Lines    = (linewidth = pt2px(2.0),),
            Scatter = (markersize = pt2px(9 + 1/3 + 2.0),),
            ScatterLines = (linewidth = pt2px(2.0),
                            markersize = pt2px(9 + 1/3 + 2.0),),
            Colorbar = (labelsize  = LabelFontSize,),
            Legend   = (fontsize   = LegendFontSize,
                        titlesize  = LegendFontSize,
                        framewidth = pt2px(1.0),
                        titlefont  = :regular,
                        patchsize  = (40, 20),
                        ),
            palette  = (color = colors, patchcolor = colors,)
        )
    )
end

end # module MagicLaTeX
