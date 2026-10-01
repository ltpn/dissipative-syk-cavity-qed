module ComplexDSFFPlotData

using JLD2
using Printf
include("DatasetPaths.jl")

export FixedPowerPolicy,
       CANONICAL_DSFF_RAY_KEY,
       CANONICAL_DSFF_THETA,
       CANONICAL_DSFF_BETA,
       CANONICAL_DSFF_POLICY_KEY,
       TARGET_MODEL_RGB,
       TARGET_MODEL_LINESTYLE,
       beta_label,
       policy_slug,
       positive_or_nan,
       rounded_log_limits,
       validate_plotdata_binding,
       validate_ensemble_compatibility,
       load_policy_curves,
       load_gaussian_ai_dagger_curve,
       load_canonical_fixed_beta_curves

const CANONICAL_DSFF_RAY_KEY = "theta_pi4"
const CANONICAL_DSFF_THETA = pi / 4
const CANONICAL_DSFF_BETA = 1 / 2
const CANONICAL_DSFF_POLICY_KEY = "fixed_beta_1over2"
const TARGET_MODEL_RGB = (0.65, 0.0, 0.0)
const TARGET_MODEL_LINESTYLE = :solid

"A non-adaptive power-map request with a cache-stable beta value."
struct FixedPowerPolicy
    beta::Float64
    function FixedPowerPolicy(beta::Real)
        b = Float64(beta)
        0 < b <= 1 || throw(ArgumentError("beta must lie in (0,1]"))
        return new(b)
    end
end

"Validate that a plot-data file is exactly the production file described by its manifest."
function validate_plotdata_binding(path::AbstractString,
                                   parameters::AbstractDict)
    isfile(path) || throw(ArgumentError("plot data not found: $path"))
    fields = ("n_seeds", "fixed_betas", "ray_keys", "plot_x_min", "plot_x_max", "smoke")
    JLD2.jldopen(path, "r") do file
        for field in fields
            haskey(parameters, field) ||
                throw(ArgumentError("fixed-beta manifest is missing $field"))
            haskey(file, field) ||
                throw(ArgumentError("plot data is missing parameters field $field"))
            file[field] == parameters[field] ||
                throw(ArgumentError("plot-data parameters mismatch for $field"))
        end
    end
    return nothing
end

"Reject attempts to combine sigma-SFF and complex-DSFF data from different ensembles."
function validate_ensemble_compatibility(parameters::AbstractDict,
                                         current::AbstractDict)
    haskey(parameters, "ensemble") ||
        throw(ArgumentError("fixed-beta manifest is missing ensemble parameters"))
    expected = parameters["ensemble"]
    expected isa AbstractDict ||
        throw(ArgumentError("fixed-beta ensemble parameters has an invalid type"))
    fields = ("n_orb", "filling", "physical_etas", "gamma",
              "physical_seed_range", "physical_n_seeds",
              "l3b_config_name", "l3b_eta", "l3b_gamma",
              "l3b_seed_range", "l3b_n_seeds", "l3b_spectra_dir")
    for field in fields
        haskey(expected, field) ||
            throw(ArgumentError("fixed-beta ensemble parameters is missing $field"))
        haskey(current, field) ||
            throw(ArgumentError("current Figure 1 ensemble is missing $field"))
        matches = field == "l3b_spectra_dir" ?
            DatasetPaths.resolve_path(current[field]) == DatasetPaths.resolve_path(expected[field]) :
            current[field] == expected[field]
        matches ||
            throw(ArgumentError("sigma/DSFF ensemble mismatch for $field"))
    end
    return nothing
end

"Compact display label, retaining the named comparison fractions exactly."
function beta_label(beta::Real)
    b = Float64(beta)
    for (value, label) in ((2 / 3, "2/3"), (1 / 2, "1/2"),
                           (1 / 3, "1/3"), (1 / 4, "1/4"))
        b == value && return label
    end
    return @sprintf("%.6g", b)
end

"Filesystem- and JLD2-safe policy identifier."
function policy_slug(policy::FixedPowerPolicy)
    known = Dict("2/3" => "2over3", "1/2" => "1over2",
                 "1/3" => "1over3", "1/4" => "1over4")
    label = beta_label(policy.beta)
    token = get(known, label) do
        bits = reinterpret(UInt64, policy.beta)
        "bits_" * string(bits; base = 16, pad = 16)
    end
    return "fixed_beta_" * token
end

"Replace values that cannot be drawn on a logarithmic axis with NaN."
positive_or_nan(values) = Float64[
    isfinite(value) && value > 0 ? value : NaN for value in values]

"Outward-rounded decade limits over all finite positive input values."
function rounded_log_limits(collections; include::Real = 1.0)
    values = Float64[Float64(value) for collection in collections
                     for value in collection if isfinite(value) && value > 0]
    isempty(values) && throw(ArgumentError("no finite positive plot values"))
    push!(values, Float64(include))
    decade(exponent::Int) = exponent >= 0 ? 10.0^exponent : inv(10.0^(-exponent))
    return decade(floor(Int, log10(minimum(values)))),
           decade(ceil(Int, log10(maximum(values))))
end

"Load and validate curve arrays from one policy/ray plot-data group."
function load_policy_curves(path::AbstractString, policy::AbstractString,
                            ray::AbstractString, dataset_keys)
    isfile(path) || throw(ArgumentError("plot data not found: $path"))
    return JLD2.jldopen(path, "r") do file
        curves = Dict{String,NamedTuple}()
        for key in dataset_keys
            prefix = "$policy/$ray/$key"
            required = ("x", "curve", "lower", "upper", "omitted_nonpositive")
            all(name -> haskey(file, "$prefix/$name"), required) ||
                throw(ArgumentError("incomplete plot-data group: $prefix"))
            x = Float64.(file["$prefix/x"])
            curve = Float64.(file["$prefix/curve"])
            lower = Float64.(file["$prefix/lower"])
            upper = Float64.(file["$prefix/upper"])
            length(x) == length(curve) == length(lower) == length(upper) ||
                throw(DimensionMismatch("curve lengths differ for $prefix"))
            curves[String(key)] = (
                x = x, curve = curve, lower = lower, upper = upper,
                omitted = Int(file["$prefix/omitted_nonpositive"]))
        end
        return curves
    end
end

"Load the independently generated large-N Gaussian AI-dagger theory curve."
function load_gaussian_ai_dagger_curve(path::AbstractString)
    isfile(path) || throw(ArgumentError("plot data not found: $path"))
    return JLD2.jldopen(path, "r") do file
        prefix = "theory/gaussian_ai_dagger"
        required = (
            "x", "curve", "lower", "upper", "finite_size_systematic",
            "chi", "matrix_sizes", "seed_counts", "display_x", "display_curve")
        all(name -> haskey(file, "$prefix/$name"), required) ||
            throw(ArgumentError("Gaussian AI-dagger theory group is incomplete"))
        x = Float64.(file["$prefix/x"])
        curve = Float64.(file["$prefix/curve"])
        lower = Float64.(file["$prefix/lower"])
        upper = Float64.(file["$prefix/upper"])
        systematic = Float64.(file["$prefix/finite_size_systematic"])
        length(x) == length(curve) == length(lower) == length(upper) ==
            length(systematic) ||
            throw(DimensionMismatch("Gaussian AI-dagger curve lengths differ"))
        chi = Float64(file["$prefix/chi"])
        isfinite(chi) && chi > 0 ||
            throw(ArgumentError("Gaussian AI-dagger chi must be positive"))
        display_x = Float64.(file["$prefix/display_x"])
        display_curve = Float64.(file["$prefix/display_curve"])
        length(display_x) == length(display_curve) ||
            throw(DimensionMismatch("Gaussian AI-dagger display curve lengths differ"))
        return (x = x, curve = curve, lower = lower, upper = upper,
                display_x = display_x, display_curve = display_curve,
                finite_size_systematic = systematic, chi = chi,
                matrix_sizes = Int.(file["$prefix/matrix_sizes"]),
                seed_counts = Int.(file["$prefix/seed_counts"]),
        )
    end
end

"Load the locked five-ensemble canonical fixed-beta Figure 1 curves."
function load_canonical_fixed_beta_curves(path::AbstractString;
                                          x_min::Real = 10.0^-2.5,
                                          x_max::Real = 10.0,
                                          n_points::Integer = 400)
    dataset_keys = ("eta_0p1", "eta_0p4", "eta_1p0", "eta_2p0", "l3b")
    curves = load_policy_curves(
        path, CANONICAL_DSFF_POLICY_KEY, CANONICAL_DSFF_RAY_KEY, dataset_keys)
    reference = curves[first(dataset_keys)].x
    length(reference) == n_points ||
        throw(ArgumentError("canonical DSFF grid must contain $n_points points"))
    isapprox(first(reference), Float64(x_min); rtol = 1e-12) ||
        throw(ArgumentError("canonical DSFF grid has the wrong lower limit"))
    isapprox(last(reference), Float64(x_max); rtol = 1e-12) ||
        throw(ArgumentError("canonical DSFF grid has the wrong upper limit"))
    for key in dataset_keys
        curves[key].x == reference ||
            throw(ArgumentError("canonical DSFF x grids differ"))
    end
    return curves
end

end
