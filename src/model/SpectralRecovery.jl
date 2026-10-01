module SpectralRecovery
using TOML
include("SpectralArtifacts.jl")
using .SpectralArtifacts
import .SpectralArtifacts: identity

"A replacement is a new physical point, never a relabelled old artifact."
function replacement_manifest(base,slot_id,seed::Integer)
    slot = only(filter(p->p["point_id"]==slot_id,base["points"]))
    seed > base["resolved_config"]["grid"]["seed_count"] || error("Replacement seed must be outside the production seed range")
    point = deepcopy(slot)
    point["seed"] = seed
    point["filename"] = replace(slot["filename"],r"__seed=\d+(?=\.jld2$)"=>"__seed=$seed")
    point["point_id"] = point["model"] == "physical" ?
        replace(slot_id,r"\|\d+$"=>"|$seed") : replace(slot_id,r"__seed=\d+$"=>"__seed=$seed")
    point["point_id"] != slot_id && point["filename"] != slot["filename"] || error("Unrecognized point seed encoding")
    generation_id = identity(Dict("parent_generation_id"=>base["generation_id"],
        "replacement_for"=>slot_id,"points"=>[point]))
    merge(base,Dict("stage"=>"generate","points"=>[point],"generation_id"=>generation_id,
        "parent_generation_id"=>base["generation_id"],"replacement_for"=>slot_id,"replacement_seed"=>seed))
end

function load_replacements(path,base; root=get(ENV,"SYK_DATA_ROOT",nothing))
    path === nothing && return Dict{Tuple{String,String},Dict{String,Any}}()
    raw = TOML.parsefile(path)
    data = root === nothing && !haskey(raw,"path_root") ? raw : SpectralArtifacts.DatasetPaths.read_manifest(path;root)
    entries = get(data,"replacements",[])
    result = Dict{Tuple{String,String},Dict{String,Any}}()
    used = Set{Tuple{String,String,Int}}()
    for entry in entries
        stage,slot_id = entry["stage"],entry["slot_id"]
        stage in ("eigen","svd") || error("Replacement stage must be eigen or svd")
        key = (stage,slot_id)
        haskey(result,key) && error("Duplicate replacement for $key")
        seed = Int(entry["seed"])
        replacement = replacement_manifest(base,slot_id,seed)
        point = only(replacement["points"])
        parameters = Dict(k=>v for (k,v) in point if !(k in ("seed","point_id","filename")))
        seedkey = (stage,identity(parameters),seed)
        seedkey in used && error("Duplicate replacement seed within an ensemble")
        push!(used,seedkey)
        directory = abspath(joinpath(dirname(path),entry["directory"]))
        result[key] = Dict("stage"=>stage,"slot_id"=>slot_id,"seed"=>seed,"directory"=>directory)
    end
    result
end

function selected_source(base,point,stage,original_dir,replacements)
    key = (stage,point["point_id"])
    if !haskey(replacements,key)
        return base,point,joinpath(original_dir,point["filename"])
    end
    entry = replacements[key]
    expected = replacement_manifest(base,point["point_id"],Int(entry["seed"]))
    dir = joinpath(entry["directory"],stage)
    actual = read_manifest(joinpath(dir,"manifest.jld2"))
    compatible(m) = Dict(k=>v for (k,v) in m if k != "resolved_config")
    identity(compatible(actual)) == identity(compatible(merge(expected,Dict("stage"=>stage)))) || error("Replacement manifest mismatch")
    p = only(expected["points"])
    expected,p,joinpath(dir,p["filename"])
end

"Validate checkpoint identity and source linkage without loading dense matrices."
function inventory_status(root,directory,slot_id,seed,stage)
    stage in ("generate","eigen","svd") || error("Invalid inventory stage")
    base = read_manifest(joinpath(root,"liouvillians","manifest.jld2"))
    cfg = SpectralArtifacts.load_pipeline_config(joinpath(root,"config.toml"))
    SpectralArtifacts.physics_identity(cfg) == base["physics_id"] || error("Root configuration physics changed")
    identity(cfg["grid"]) == identity(base["resolved_config"]["grid"]) || error("Root configuration grid changed")
    slot = only(filter(p->p["point_id"]==slot_id,base["points"]))
    original = abspath(directory) == abspath(root)
    original && seed != slot["seed"] && error("Original slot seed mismatch")
    m = original ? base : replacement_manifest(base,slot_id,seed)
    p = original ? slot : only(m["points"])
    stage_dir = joinpath(directory,stage == "generate" ? "liouvillians" : stage)
    path = joinpath(stage_dir,p["filename"])
    isfile(path) || return "missing"
    actual = read_manifest(joinpath(stage_dir,"manifest.jld2"))
    compatible(x) = Dict(k=>v for (k,v) in x if k != "resolved_config")
    identity(compatible(actual)) == identity(compatible(merge(m,Dict("stage"=>stage)))) || error("Inventory manifest mismatch")
    n = cfg["numerics"]
    expected = Dict("point_id"=>p["point_id"],"physics_id"=>base["physics_id"],
        "generation_id"=>m["generation_id"],"dimension"=>binomial(n["n_orb"],n["filling"])^2)
    h = read_artifact(path;stage,expected,fields=String[])["header"]
    if stage != "generate"
        source = read_artifact(joinpath(directory,"liouvillians",p["filename"]);
            stage="generate",expected,fields=String[])["header"]
        h["source_id"] == source["artifact_id"] || error("Inventory source mismatch")
    end
    "complete"
end
end
