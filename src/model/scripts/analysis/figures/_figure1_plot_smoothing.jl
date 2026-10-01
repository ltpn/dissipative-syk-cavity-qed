module Figure1PlotSmoothing

using LinearAlgebra

export sgolay_log_positive

"""
    sgolay_log_positive(values; window=11, degree=3)

Savitzky--Golay smooth each contiguous finite-positive run in `log10(values)`.
Invalid or nonpositive samples are returned as `NaN`. Endpoint samples use a
shifted local window, so the output has the same domain and length as the input.
Runs too short to fit the requested polynomial are copied without smoothing.
"""
function sgolay_log_positive(values::AbstractVector{<:Real};
                             window::Integer = 11,
                             degree::Integer = 3)
    window > 0 || throw(ArgumentError("window must be positive"))
    isodd(window) || throw(ArgumentError("window must be odd"))
    degree >= 0 || throw(ArgumentError("degree must be nonnegative"))
    degree < window || throw(ArgumentError("degree must be smaller than window"))

    output = fill(NaN, length(values))
    valid = map(value -> isfinite(value) && value > 0, values)
    run_start = firstindex(values)
    final_index = lastindex(values)

    while run_start <= final_index
        if !valid[run_start]
            run_start += 1
            continue
        end
        run_stop = run_start
        while run_stop < final_index && valid[run_stop + 1]
            run_stop += 1
        end

        run_length = run_stop - run_start + 1
        if run_length <= degree
            output[run_start:run_stop] .= Float64.(values[run_start:run_stop])
            run_start = run_stop + 1
            continue
        end

        span = min(window, run_length)
        half_span = span ÷ 2
        for index in run_start:run_stop
            left = clamp(index - half_span, run_start, run_stop - span + 1)
            right = left + span - 1
            offsets = Float64.(collect(left:right) .- index)
            design = Matrix{Float64}(undef, span, degree + 1)
            for power in 0:degree
                design[:, power + 1] .= offsets .^ power
            end
            coefficients = design \ log10.(Float64.(values[left:right]))
            filtered = 10.0^coefficients[1]
            isfinite(filtered) && filtered > 0 ||
                error("Savitzky-Golay filtering produced an invalid value at index $index")
            output[index] = filtered
        end
        run_start = run_stop + 1
    end

    return output
end

end # module Figure1PlotSmoothing
