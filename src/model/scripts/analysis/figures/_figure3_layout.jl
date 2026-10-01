function configure_figure3_layout!(figure, have_dynamics::Bool, source_px)
    if have_dynamics
        colgap!(figure.layout, source_px(-9.0))
        colsize!(figure.layout, 1, Relative(0.50))
        colsize!(figure.layout, 2, Relative(0.50))
    else
        colsize!(figure.layout, 1, Relative(1.0))
    end
    return figure
end

figure3_layout_name(have_dynamics::Bool) = have_dynamics ?
    "one_column_side_by_side_magiclatex" :
    "one_column_sff_only_magiclatex"
