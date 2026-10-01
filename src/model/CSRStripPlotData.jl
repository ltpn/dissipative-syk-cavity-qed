module CSRStripPlotData

using JLD2
include("DatasetPaths.jl")

export csr_strip_layout,
       read_csr_strip_plotdata,
       should_compute_csr_panel,
       validate_cache_identity,
       validate_panel_spectra,
       write_csr_strip_plotdata

"""Reject differently configured plot-data caches."""
function validate_cache_identity(cached, current)
    cached_identity = Dict{String,Any}(String(key) => value
                                       for (key, value) in cached)
    current_identity = Dict{String,Any}(String(key) => value
                                        for (key, value) in current)
    DatasetPaths.resolve_metadata(cached_identity,DatasetPaths.dataset_root()) ==
        DatasetPaths.resolve_metadata(current_identity,DatasetPaths.dataset_root()) || throw(ArgumentError(
        "CSR plot-data source/configuration identity does not match this invocation"))
    return nothing
end

"""Decide whether a panel may perform the expensive raw-spectrum CSR pass."""
function should_compute_csr_panel(cache_loaded::Bool, status::AbstractString,
                                  has_ratio_data::Bool, update_pending::Bool)
    has_ratio_data && return false
    cache_loaded || return true
    return String(status) == "pending_spectra" && update_pending
end

"""Require a complete, ordered seed ensemble of the expected spectrum size."""
function validate_panel_spectra(seeds, spectra, n_seeds::Integer,
                                expected_dim::Integer)
    expected_seeds = collect(1:Int(n_seeds))
    Int.(seeds) == expected_seeds || throw(ArgumentError(
        "expected panel seeds 1:$(n_seeds), found $(Int.(seeds))"))
    length(spectra) == Int(n_seeds) || throw(ArgumentError(
        "expected $(n_seeds) panel spectra, found $(length(spectra))"))
    all(length(values) == Int(expected_dim) for values in spectra) ||
        throw(ArgumentError("panel spectrum dimension does not match $expected_dim"))
    return nothing
end

"""Return the fixed two-row panel order plus the vertically centered control."""
function csr_strip_layout(top::AbstractVector{<:AbstractString},
                          bottom::AbstractVector{<:AbstractString},
                          control::AbstractString)
    length(top) == 4 || throw(ArgumentError("CSR top row must have four panels"))
    length(bottom) == 4 || throw(ArgumentError("CSR bottom row must have four panels"))
    tags = vcat(String.(top), String.(bottom), [String(control)])
    length(unique(tags)) == 9 || throw(ArgumentError("CSR panel tags must be unique"))
    panels = NamedTuple[]
    for (idx, tag) in enumerate(top)
        push!(panels, (tag = String(tag), letter = Char('a' + idx - 1),
                       row = 1, col = idx))
    end
    for (idx, tag) in enumerate(bottom)
        push!(panels, (tag = String(tag), letter = Char('e' + idx - 1),
                       row = 2, col = idx))
    end
    push!(panels, (tag = String(control), letter = 'i', row = 0, col = 5))
    return panels
end

function _validate_plotdata(csr_by_tag, panel_order, panel_meta)
    length(unique(panel_order)) == length(panel_order) ||
        throw(ArgumentError("CSR panel_order contains duplicate tags"))
    for tag in panel_order
        haskey(panel_meta, tag) ||
            throw(ArgumentError("CSR panel metadata missing tag '$tag'"))
        status = String(panel_meta[tag]["status"])
        status in ("complete", "pending_spectra") ||
            throw(ArgumentError("unsupported CSR panel status '$status' for '$tag'"))
        if status == "complete"
            haskey(csr_by_tag, tag) ||
                throw(ArgumentError("complete CSR panel '$tag' has no ratio data"))
            seeds = Int.(get(panel_meta[tag], "seeds", Int[]))
            length(csr_by_tag[tag]) == length(seeds) || throw(ArgumentError(
                "complete CSR panel '$tag' ratio/seed counts disagree"))
            if haskey(panel_meta[tag], "n_seeds_loaded")
                Int(panel_meta[tag]["n_seeds_loaded"]) == length(seeds) ||
                    throw(ArgumentError(
                        "complete CSR panel '$tag' metadata seed count disagrees"))
            end
            if haskey(panel_meta[tag], "n_ratios")
                Int(panel_meta[tag]["n_ratios"]) == sum(length, csr_by_tag[tag]) ||
                    throw(ArgumentError(
                        "complete CSR panel '$tag' metadata ratio count disagrees"))
            end
        elseif haskey(csr_by_tag, tag)
            throw(ArgumentError("pending CSR panel '$tag' unexpectedly has ratio data"))
        end
    end
    all(tag -> tag in panel_order, keys(csr_by_tag)) ||
        throw(ArgumentError("CSR ratio data contains an unknown panel tag"))
    return nothing
end

"""Atomically persist seed-resolved complex spacing ratios."""
function write_csr_strip_plotdata(path::AbstractString, csr_by_tag,
                                  panel_order::AbstractVector{<:AbstractString},
                                  panel_meta, constants;
                                  cache_identity = Dict{String,Any}())
    order = String.(panel_order)
    meta = Dict{String,Any}(String(tag) => Dict{String,Any}(
        String(key) => value for (key, value) in values)
        for (tag, values) in panel_meta)
    ratios = Dict{String,Vector{Vector{ComplexF64}}}(
        String(tag) => [ComplexF64.(seed_values) for seed_values in seeds]
        for (tag, seeds) in csr_by_tag)
    consts = Dict{String,Any}(String(key) => value for (key, value) in constants)
    identity = Dict{String,Any}(String(key) => value
                                for (key, value) in cache_identity)
    _validate_plotdata(ratios, order, meta)

    ratios_re = Dict(tag => [Float64.(real.(values)) for values in seeds]
                     for (tag, seeds) in ratios)
    ratios_im = Dict(tag => [Float64.(imag.(values)) for values in seeds]
                     for (tag, seeds) in ratios)
    pending = [tag for tag in order if meta[tag]["status"] == "pending_spectra"]

    mkpath(dirname(abspath(path)))
    tmp = String(path) * ".tmp"
    JLD2.jldopen(tmp, "w"; compress = true) do file
        file["panel_order"] = order
        file["pending_tags"] = pending
        file["panel_meta"] = DatasetPaths.relative_metadata(meta)
        file["csr_constants"] = consts
        file["cache_identity"] = DatasetPaths.relative_metadata(identity)
        file["ratios_re_by_tag"] = ratios_re
        file["ratios_im_by_tag"] = ratios_im
    end
    mv(tmp, path; force = true)
    return String(path)
end

"""Load and validate a plot-ready CSR strip cache."""
function read_csr_strip_plotdata(path::AbstractString)
    isfile(path) || throw(ArgumentError("CSR strip plot-data file not found: $path"))
    return JLD2.jldopen(path, "r") do file
        order = String.(file["panel_order"])
        pending = String.(file["pending_tags"])
        meta = Dict{String,Any}(String(tag) => Dict{String,Any}(
            String(key) => value for (key, value) in values)
            for (tag, values) in DatasetPaths.resolve_metadata(file["panel_meta"],DatasetPaths.dataset_root()))
        constants = Dict{String,Any}(
            String(key) => value for (key, value) in file["csr_constants"])
        identity = Dict{String,Any}(String(key) => value
                                   for (key, value) in DatasetPaths.resolve_metadata(file["cache_identity"],DatasetPaths.dataset_root()))
        ratios_re = file["ratios_re_by_tag"]
        ratios_im = file["ratios_im_by_tag"]
        keys(ratios_re) == keys(ratios_im) ||
            throw(ArgumentError("CSR plot-data real/imaginary panel keys disagree"))
        ratios = Dict{String,Vector{Vector{ComplexF64}}}()
        for tag_raw in keys(ratios_re)
            tag = String(tag_raw)
            re_seeds = ratios_re[tag_raw]
            im_seeds = ratios_im[tag_raw]
            length(re_seeds) == length(im_seeds) || throw(ArgumentError(
                "CSR plot-data real/imaginary seed counts disagree for '$tag'"))
            ratios[tag] = Vector{Vector{ComplexF64}}()
            for (re, im) in zip(re_seeds, im_seeds)
                length(re) == length(im) || throw(ArgumentError(
                    "CSR plot-data real/imaginary ratio counts disagree for '$tag'"))
                push!(ratios[tag], ComplexF64.(complex.(Float64.(re), Float64.(im))))
            end
        end
        _validate_plotdata(ratios, order, meta)
        Set(pending) == Set(tag for tag in order
                            if meta[tag]["status"] == "pending_spectra") ||
            throw(ArgumentError("CSR plot-data pending tag list disagrees with metadata"))
        return (panel_order = order,
                pending_tags = pending, panel_meta = meta,
                constants = constants, cache_identity = identity,
                csr_by_tag = ratios)
    end
end

end
