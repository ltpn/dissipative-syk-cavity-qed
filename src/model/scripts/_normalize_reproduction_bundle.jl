using JLD2, TOML
include(joinpath(@__DIR__, "..", "DatasetPaths.jl"))
include(joinpath(@__DIR__, "..", "SpectralArtifacts.jl"))
include(joinpath(@__DIR__, "..", "SpectralPostprocessing.jl"))

options = TOML.parsefile(only(ARGS))
source, output = options["source_root"], options["output_root"]
mappings = sort(collect(options["path_mapping"]);by=p->-length(first(p)))
config_runs = Dict("csr_ladder_l3b_n10_q4_r16.toml"=>"syk4_control",
    "figure3_syk4_dtilde0p01_m300_n10_q4_r16.toml"=>"fig3_m300",
    "multimode_n10_q4_r16.toml"=>"multimode_n10_q4",
    "physical_n10_q4_r16.toml"=>"physical_n10_q4")

function portable_path(value)
    isabspath(value) || return value
    for (prefix, target) in mappings
        (value == prefix || startswith(value,prefix * "/")) || continue
        relative = normpath(joinpath(target,relpath(value,prefix)))
        relative = replace(relative,r"^source_v\d+/data/"=>"data/",r"^work/"=>"derived/")
        if occursin(r"^source_v\d+/",relative)
            run = get(config_runs,basename(relative),nothing)
            return run === nothing ? replace(relative,r"^source_v\d+/"=>"source_code/") : "runs/$run/config.toml"
        end
        return relative
    end
    error("Unmapped absolute provenance path: $value")
end
normalize(value) = DatasetPaths.map_strings(portable_path,value)

function same_payload(a,b)
    if a isa AbstractDict && b isa AbstractDict
        keys(a)==keys(b) || return false
        return all(same_payload(a[k],b[k]) for k in keys(a))
    elseif a isa AbstractArray && b isa AbstractArray
        size(a)==size(b) || return false
        isbitstype(eltype(a)) && return isequal(a,b)
        return all(same_payload(x,y) for (x,y) in zip(a,b))
    elseif typeof(a)!=typeof(b)
        return false
    elseif a isa Union{Number,AbstractString,Symbol,Nothing}
        return isequal(a,b)
    elseif isstructtype(typeof(a)) && fieldcount(typeof(a))>0
        return all(same_payload(getfield(a,i),getfield(b,i)) for i in 1:fieldcount(typeof(a)))
    end
    isequal(a,b)
end

function leaves(group, prefix="")
    result = Pair{String,Any}[]
    for key in keys(group)
        name = prefix * String(key)
        value = group[key]
        if value isa JLD2.Group
            append!(result,leaves(value,name * "/"))
        else
            push!(result,name=>value)
        end
    end
    result
end

modified = 0
for relative in options["files"]
    destination = joinpath(output,relative)
    original = joinpath(source,relative)
    if endswith(relative,".toml")
        data = normalize(TOML.parsefile(original))
        data["path_root"] = relpath(output,dirname(destination))
        open(io->TOML.print(io,data;sorted=true),destination,"w")
    elseif endswith(relative,".jld2")
        pairs = JLD2.jldopen(leaves,original,"r")
        normalized = [key=>normalize(value) for (key,value) in pairs]
        if any(!same_payload(old,new) for ((_,old),(_,new)) in zip(pairs,normalized))
            temp = destination * ".normalized"
            JLD2.jldopen(temp,"w") do file
                for (key,value) in normalized; file[key]=value; end
            end
            mv(temp,destination;force=true)
            global modified += 1
        end
        # Exact comparison includes all numerical arrays and nonpath metadata.
        JLD2.jldopen(destination,"r") do file
            for (key,expected) in normalized
                same_payload(file[key],expected) || error("Bundle payload mismatch: $relative/$key")
            end
        end
    end
end

headers_saved = 0
for (name, root) in get(options,"generation_roots",Dict())
    directories = [joinpath(root,"liouvillians")]
    attempts = joinpath(root,"recovery/attempts")
    if isdir(attempts)
        append!(directories,[joinpath(attempts,p,"liouvillians") for p in readdir(attempts)])
    end
    for directory in directories
        isdir(directory) || continue
        headers = Dict{String,Any}()
        for file in readdir(directory)
            endswith(file,".jld2") && file != "manifest.jld2" || continue
            headers[file] = SpectralArtifacts.read_artifact(joinpath(directory,file);stage="generate",fields=String[])["header"]
        end
        isempty(headers) && continue
        path = joinpath(output,"runs",name,relpath(directory,root),"headers.toml")
        mkpath(dirname(path))
        open(io->TOML.print(io,Dict("schema_version"=>1,"headers"=>headers);sorted=true),path,"w")
        global headers_saved += length(headers)
    end
end
println("Verified every bundled numerical payload; normalized $modified JLD2 files; saved $headers_saved authentic generation headers")
