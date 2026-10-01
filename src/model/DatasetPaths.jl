module DatasetPaths
using TOML

export dataset_root, resolve_path, relative_metadata, read_manifest, write_manifest, print_manifest

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
dataset_root() = abspath(get(ENV, "SYK_DATA_ROOT", REPO_ROOT))
resolve_path(path::AbstractString; root=dataset_root()) = normpath(isabspath(path) ? path : joinpath(root,path))

const PATH_KEYS = Set(["spectra_dir", "l3b_spectra_dir", "cache_dir", "cache_path", "cache_paths",
    "source_manifests", "parameters_manifest", "source_figure1_manifest_input", "plotdata",
    "plotdata_jld2", "physical_dir", "control_dir", "multimode_cache_dir", "source_path",
    "calibration", "calibration_jld2", "validation", "path", "directory", "outputs",
    "output_pdf", "output_png", "output_dir", "baseline_calibration", "baseline_config",
    "physical_cache", "multimode_config", "source_files", "input_paths", "compatibility",
    "reference_jld2", "root"])

function map_strings(f, value)
    value isa AbstractString && return f(value)
    value isa AbstractDict && return Dict{keytype(typeof(value)),Any}(k=>map_strings(f,v) for (k,v) in value)
    value isa NamedTuple && return map(v->map_strings(f,v),value)
    value isa Tuple && return map(v->map_strings(f,v),value)
    if value isa AbstractArray && !(isbitstype(eltype(value)))
        return map(v->map_strings(f,v),value)
    end
    value
end

"Convert absolute provenance paths to dataset-relative strings; leave numerical arrays untouched."
relative_metadata(value; root=dataset_root()) = map_strings(s->isabspath(s) ? relpath(s,root) : s,value)

function resolve_metadata(value, root, key="")
    if value isa AbstractDict
        return Dict{keytype(typeof(value)),Any}(k=>resolve_metadata(v,root,String(k)) for (k,v) in value)
    elseif value isa AbstractArray && !(isbitstype(eltype(value)))
        return map(v->resolve_metadata(v,root,key),value)
    elseif value isa AbstractString
        ispathvalue = key in PATH_KEYS || (key == "config" && endswith(value,".toml"))
        return ispathvalue && value != "current run" && !isempty(value) ? resolve_path(value;root) : value
    end
    value
end

"Read a manifest independently of the caller's working directory."
function read_manifest(path; root=nothing)
    data = TOML.parsefile(path)
    base = if root !== nothing
        abspath(root)
    elseif haskey(ENV,"SYK_DATA_ROOT")
        dataset_root()
    elseif haskey(data,"path_root")
        resolve_path(data["path_root"];root=dirname(abspath(path)))
    else
        pwd() # Legacy manifests used paths relative to the working directory.
    end
    resolve_metadata(data,base)
end

function print_manifest(io, data; root=dataset_root(), path=nothing, sorted=true)
    portable = relative_metadata(data;root)
    path === nothing || (portable["path_root"] = relpath(root,dirname(abspath(path))))
    TOML.print(io,portable;sorted)
end

function write_manifest(path, data; root=dataset_root())
    mkpath(dirname(abspath(path)))
    open(path,"w") do io
        print_manifest(io,data;root,path)
    end
    path
end
end
