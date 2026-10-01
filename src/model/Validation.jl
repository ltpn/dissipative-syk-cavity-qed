"""
    log_time_grid(t_min, t_max, n)

Return `n` strictly increasing logarithmically spaced positive times,
including both endpoints.   uses this for the default validation plot
and for the additional longer-window `N_orb = 6` DSFF/SFF plot.
"""
function log_time_grid(t_min::Real, t_max::Real, n::Integer)
    lo = Float64(t_min)
    hi = Float64(t_max)
    count = Int(n)
    isfinite(lo) && isfinite(hi) ||
        throw(ArgumentError("time-grid endpoints must be finite"))
    lo > 0.0 || throw(ArgumentError("t_min must be positive, got $t_min"))
    hi > lo || throw(ArgumentError("t_max must be larger than t_min"))
    count >= 2 || throw(ArgumentError("time grid requires at least two points"))
    return collect(10 .^ range(log10(lo), log10(hi), length = count))
end
