module SpectralArtifacts
using JLD2, SHA, TOML, UUIDs, Sockets
include("DatasetPaths.jl")
export identity, publish_artifact, read_artifact, ensure_manifest, read_manifest

function identity(value::AbstractDict)
    io = IOBuffer()
    TOML.print(io, value; sorted=true)
    bytes2hex(SHA.sha256(take!(io)))
end

function validate(file, expected; fields=nothing)
    h = file["header"]
    h["schema_version"] == 1 || error("Unsupported spectral schema")
    for key in ("stage", "point_id", "physics_id", "generation_id", "source_id", "dimension", "artifact_id", "code_revision")
        haskey(h,key) || error("Missing header field: $key")
    end
    for (key,value) in expected
        key == "code_revision" && continue # provenance, not numerical compatibility
        get(h,key,nothing) == value || error("Artifact $key mismatch; preserve this file and use a different output directory")
    end
    stage = h["stage"]
    required = stage == "generate" ? ["L","H","basis_states","metadata"] :
        stage == "eigen" ? ["L_eigvals","L_eigvecs","basis_states","metadata"] :
        stage == "svd" ? ["L_svd_S_centered","L_trace_shift_mu_re","L_trace_shift_mu_im","metadata"] :
        stage == "dynamics" ? ["times","entropy_t","populations_t"] :
        stage == "analysis" ? ["result"] : error("Unknown artifact stage: $stage")
    all(k -> haskey(file,k),required) || error("Incomplete $stage artifact")
    K = Int(h["dimension"])
    K > 0 || error("Invalid dimension")
    for key in required
        fields !== nothing && !(key in fields) && continue
        value = file[key]
        if key in ("L","H","L_eigvals","L_eigvecs","L_svd_S_centered","times","entropy_t","populations_t")
            all(isfinite,value) || error("Nonfinite $key")
        end
        key in ("L","L_eigvecs") && size(value) != (K,K) && error("Invalid $key shape")
        key == "H" && size(value) != (isqrt(K),isqrt(K)) && error("Invalid H shape")
        key in ("L_eigvals","L_svd_S_centered") && length(value) != K && error("Invalid $key length")
        key == "L_svd_S_centered" && any(<(0),value) && error("Negative singular value")
        key == "basis_states" && length(value)^2 != K && error("Invalid basis dimension")
    end
    if stage == "dynamics"
        times = file["times"]
        times isa AbstractVector && !isempty(times) || error("Invalid times shape")
        size(file["entropy_t"]) == size(times) || error("Invalid entropy_t shape")
        populations = file["populations_t"]
        ndims(populations) == 2 && size(populations,1) > 0 && size(populations,2) == length(times) || error("Invalid populations_t shape")
    end
    return h
end

function read_artifact(path::AbstractString; stage::String, expected::AbstractDict=Dict(), fields=nothing)
    isfile(path) || error("Missing $stage artifact: $path; run the producing stage")
    JLD2.jldopen(path,"r") do f
        validate(f,merge(Dict("stage"=>stage),expected); fields=fields)
        selected = fields === nothing ? collect(keys(f)) : unique(vcat(["header"],fields))
        Dict{String,Any}(k=>f[k] for k in selected)
    end
end

"Read authentic matrix identity metadata when a download omits the dense matrix."
function read_generation_header(path; expected=Dict(), header_root=get(ENV,"SYK_HEADER_ROOT",nothing))
    isfile(path) && return read_artifact(path;stage="generate",expected,fields=String[])["header"]
    directory = dirname(abspath(path))
    sidecar = joinpath(directory,"headers.toml")
    if !isfile(sidecar) && header_root !== nothing
        sidecar = joinpath(header_root,relpath(directory,DatasetPaths.dataset_root()),"headers.toml")
    end
    isfile(sidecar) || error("Missing generation artifact and header sidecar: $path")
    headers = TOML.parsefile(sidecar)["headers"]
    haskey(headers,basename(path)) || error("Missing generation header for $path")
    h = headers[basename(path)]
    for key in ("schema_version","stage","point_id","physics_id","generation_id","source_id","dimension","artifact_id","code_revision")
        haskey(h,key) || error("Missing generation header field: $key")
    end
    h["schema_version"] == 1 && h["stage"] == "generate" && h["dimension"] > 0 || error("Invalid generation header: $path")
    for (key,value) in expected
        get(h,key,nothing) == value || error("Generation header $key mismatch: $path")
    end
    h
end

function with_claim(f::Function,path)
    mkpath(dirname(path))
    claim = path * ".claim"
    try
        mkdir(claim)
    catch err
        err isa InterruptException && rethrow()
        error("Output is claimed: $claim. Check its owner is no longer running before removing this stale claim and retrying.")
    end
    try
        write(joinpath(claim,"owner"),"$(gethostname()) pid=$(getpid())\n")
        f()
    finally
        rm(claim; recursive=true,force=true)
    end
end

function atomic_write(writer::Function,path,validator::Function)
    tmp,io = mktemp(dirname(path)); close(io)
    try
        JLD2.jldopen(writer,tmp,"w")
        JLD2.jldopen(validator,tmp,"r")
        ispath(path) && error("Refusing to replace $path")
        Base.Filesystem.rename(tmp,path)
    finally
        isfile(tmp) && rm(tmp)
    end
end

function publish_artifact(writer::Function,path::AbstractString,header::AbstractDict)
    check() = JLD2.jldopen(f -> validate(f,header),path,"r")
    if isfile(path)
        check(); return :skipped
    end
    with_claim(path) do
        if isfile(path)
            check(); return :skipped
        end
        payload = writer()
        h = merge(header,Dict("artifact_id"=>string(uuid4())))
        atomic_write(path, f -> validate(f,header)) do file
            file["header"] = h
            for (key,value) in payload
                key == "header" && error("Payload must not replace header")
                file[key] = value
            end
        end
        :written
    end
end

function read_manifest(path)
    manifest = JLD2.jldopen(f -> f["manifest"],path,"r")
    if haskey(manifest,"resolved_config") && haskey(manifest["resolved_config"],"path_root")
        cfg = manifest["resolved_config"]
        root = haskey(ENV,"SYK_DATA_ROOT") ? DatasetPaths.dataset_root() :
            DatasetPaths.resolve_path(cfg["path_root"];root=dirname(abspath(path)))
        cfg = DatasetPaths.resolve_metadata(cfg,root)
        cfg["path_root"] = root # Runtime root; serialization below stores a relative anchor.
        manifest["resolved_config"] = cfg
    end
    manifest
end

function portable_manifest(manifest,path)
    persisted = deepcopy(manifest)
    if haskey(persisted,"resolved_config")
        cfg = persisted["resolved_config"]
        root = get(cfg,"path_root",DatasetPaths.dataset_root())
        isabspath(root) || (root = DatasetPaths.dataset_root())
        cfg = DatasetPaths.relative_metadata(cfg;root)
        cfg["path_root"] = relpath(root,dirname(abspath(path)))
        persisted["resolved_config"] = cfg
    end
    persisted
end

function ensure_manifest(path::AbstractString,manifest::AbstractDict)
    matches() = begin
        old = read_manifest(path)
        compatible(m) = Dict(k=>v for (k,v) in m if k != "resolved_config")
        identity(compatible(old)) == identity(compatible(manifest)) || error("Manifest mismatch: $path; use a different output directory")
        old
    end
    isfile(path) && return matches()
    # An initializer is a short metadata write, never an expensive matrix calculation.
    try
        return with_claim(path) do
            isfile(path) && return matches()
            persisted = portable_manifest(manifest,path)
            atomic_write(f -> (f["manifest"] = persisted),path,
                f -> identity(f["manifest"]) == identity(persisted) || error("Invalid manifest"))
            Dict{String,Any}(manifest)
        end
    catch err
        err isa InterruptException && rethrow()
        # Another array worker may be publishing the identical manifest.
        if isdir(path * ".claim")
            for _ in 1:100
                isfile(path) && return matches()
                sleep(0.05)
            end
        end
        # The other worker may already have published and released its claim.
        isfile(path) && return matches()
        rethrow(err)
    end
end

function load_pipeline_config(path; root=nothing)
    cfg = TOML.parsefile(path)
    root = root !== nothing ? abspath(root) : haskey(ENV,"SYK_DATA_ROOT") ? DatasetPaths.dataset_root() :
        haskey(cfg,"path_root") ? DatasetPaths.resolve_path(cfg["path_root"];root=dirname(abspath(path))) : DatasetPaths.REPO_ROOT
    model = get(cfg,"model","physical")
    model in ("physical","multimode") || error("Unknown model: $model")
    cfg["model"] = model
    num = cfg["numerics"]
    num["filling"] = get(num,"filling",div(num["n_orb"],2))
    if haskey(num,"calibration_jld2")
        num["calibration_jld2"] = DatasetPaths.resolve_path(num["calibration_jld2"];root)
    end
    cfg["path_root"] = root
    cfg
end

function physics_identity(cfg)
    model = get(cfg,"model","physical")
    num = deepcopy(cfg["numerics"])
    pop!(num,"time_grid",nothing)
    if haskey(num,"calibration_jld2")
        path = pop!(num,"calibration_jld2")
        num["calibration_sha256"] = bytes2hex(open(SHA.sha256,path))
    end
    physical = Dict{String,Any}("schema_version"=>1,"model"=>model,
        "numerics"=>num,"seeds"=>get(cfg,"seeds",Dict()),"loss"=>get(cfg,"loss",Dict()))
    identity(physical)
end
end
