module Figure3SYK4Calibration

using LinearAlgebra
using Statistics

export coherent_superoperator_norm, solve_component_scales,
       relative_error, validate_component_metrics

function coherent_superoperator_norm(H::AbstractMatrix)
    size(H, 1) == size(H, 2) || throw(DimensionMismatch("H must be square"))
    d = size(H, 1)
    Hc = Matrix{ComplexF64}(H)
    shift = tr(Hc) / d
    @inbounds for i in 1:d
        Hc[i, i] -= shift
    end
    return sqrt(2d) * norm(Hc)
end

function solve_component_scales(physical_spans, physical_ld_norms,
                                unit_syk_spans, baseline_ld_norms)
    collections = (physical_spans, physical_ld_norms,
                   unit_syk_spans, baseline_ld_norms)
    all(x -> !isempty(x) && all(isfinite, x) && all(>(0), x), collections) ||
        throw(ArgumentError("calibration samples must be finite and positive"))
    return (
        sigma_syk4 = mean(physical_spans) / mean(unit_syk_spans),
        dissipator_rate_scale =
            mean(physical_ld_norms) / mean(baseline_ld_norms),
    )
end

function relative_error(value::Real, target::Real)
    isfinite(value) && isfinite(target) && target != 0 ||
        throw(ArgumentError("value and nonzero target must be finite"))
    return abs(Float64(value) / Float64(target) - 1)
end

function validate_component_metrics(metrics::AbstractDict;
                                    rtol::Real = 0.02,
                                    expected_n_random_jumps::Integer = 10,
                                    expected_n_cavity_jumps::Integer = 1,
                                    expected_n_orb::Union{Nothing,Integer} = nothing,
                                    expected_filling::Union{Nothing,Integer} = nothing,
                                    expected_target_delta_tilde::Union{Nothing,Real} = nothing)
    expected_n_random_jumps > 0 ||
        throw(ArgumentError("expected_n_random_jumps must be positive"))
    expected_n_cavity_jumps > 0 ||
        throw(ArgumentError("expected_n_cavity_jumps must be positive"))
    metrics["n_random_jumps"] == expected_n_random_jumps ||
        error("expected $expected_n_random_jumps random jumps")
    metrics["n_cavity_jumps"] == expected_n_cavity_jumps ||
        error("expected $expected_n_cavity_jumps cavity jumps")
    if expected_n_orb !== nothing
        metrics["n_orb"] == Int(expected_n_orb) ||
            error("expected n_orb=$(Int(expected_n_orb))")
    end
    if expected_filling !== nothing
        metrics["filling"] == Int(expected_filling) ||
            error("expected filling=$(Int(expected_filling))")
    end
    if expected_target_delta_tilde !== nothing
        metrics["target_delta_tilde"] == Float64(expected_target_delta_tilde) ||
            error("expected target_delta_tilde=$(Float64(expected_target_delta_tilde))")
    end
    relative_error(metrics["synthetic_H_span_mean"],
                   metrics["physical_H_span_mean"]) <= rtol ||
        error("synthetic Hamiltonian span misses calibration target")
    relative_error(metrics["synthetic_LD_norm_mean"],
                   metrics["physical_LD_norm_mean"]) <= rtol ||
        error("synthetic dissipator norm misses calibration target")
    return nothing
end

end
