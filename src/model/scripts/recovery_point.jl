#!/usr/bin/env julia
using TOML
include(joinpath(@__DIR__,"../SpectralRecovery.jl"))
const R = SpectralRecovery
const A = R.SpectralArtifacts

function main(args=ARGS)
    isempty(args) && error("Usage: recovery_point.jl metadata ROOT | inventory REQUEST OUTPUT | run STAGE ROOT SLOT_ID SEED DIRECTORY")
    if args[1] == "inventory"
        length(args) == 3 || error("inventory requires request and output TOML files")
        records = Dict{String,Any}[]
        for q in get(TOML.parsefile(args[2]),"requests",[])
            record = Dict{String,Any}("key"=>q["key"])
            try
                record["status"] = R.inventory_status(q["root"],q["directory"],q["slot_id"],Int(q["seed"]),q["stage"])
            catch err
                err isa InterruptException && rethrow()
                record["status"] = "invalid"
                record["error"] = sprint(showerror,err)
            end
            push!(records,record)
        end
        open(io->TOML.print(io,Dict("artifacts"=>records);sorted=true),args[3],"w")
        return 0
    end
    args[1] == "metadata" && length(args) == 2 || error("Expected metadata ROOT, inventory REQUEST OUTPUT, or run STAGE ROOT SLOT_ID SEED DIRECTORY")
    include(joinpath(@__DIR__,"../SpectralPipeline.jl"))
    Base.invokelatest() do
        P = SpectralPipeline
        P.LinearAlgebra.BLAS.set_num_threads(parse(Int,get(ENV,"OPENBLAS_THREADS_PER_WORKER","64")))
        if args[1] == "metadata"
            length(args) == 2 || error("metadata requires ROOT")
            root = abspath(args[2])
            cfg = P.load_pipeline_config(joinpath(root,"config.toml"))
            m = P.pipeline_manifest(cfg)
            A.ensure_manifest(joinpath(root,"liouvillians","manifest.jld2"),m)
            mkpath(joinpath(root,"recovery"))
            path=joinpath(root,"recovery","base_manifest.toml")
            A.DatasetPaths.write_manifest(path,m;root=A.DatasetPaths.dataset_root())
            return 0
        end
    end
end

function run_point(args)
    stage,root,slot_id,seed_text,directory = args
    stage in ("generate","eigen","svd") || error("Invalid recovery stage")
    root,directory = abspath(root),abspath(directory)
    seed = parse(Int,seed_text)
    base = A.read_manifest(joinpath(root,"liouvillians","manifest.jld2"))
    cfg = A.load_pipeline_config(joinpath(root,"config.toml"))
    A.physics_identity(cfg) == base["physics_id"] && A.identity(cfg["grid"]) == A.identity(base["resolved_config"]["grid"]) || error("Root configuration changed")
    slot_index = only(findall(p->p["point_id"]==slot_id,base["points"]))
    slot = base["points"][slot_index]
    original = root == directory
    original && seed != slot["seed"] && error("Original point seed mismatch")
    include(joinpath(@__DIR__,"../SpectralPipeline.jl"))
    Base.invokelatest() do
        P = SpectralPipeline
        P.LinearAlgebra.BLAS.set_num_threads(parse(Int,get(ENV,"OPENBLAS_THREADS_PER_WORKER","64")))
        if stage == "generate"
            result = original ? P.run_generation(cfg,joinpath(root,"liouvillians");shard_index=slot_index,shard_count=length(base["points"])) :
                P.run_replacement_generation(base,slot_id,seed,joinpath(directory,"liouvillians"))
        else
            expected = original ? base : R.replacement_manifest(base,slot_id,seed)
            actual = A.read_manifest(joinpath(directory,"liouvillians","manifest.jld2"))
            A.identity(actual) == A.identity(expected) || error("Recovery input manifest mismatch")
            result = P.run_spectral(stage,joinpath(directory,"liouvillians"),joinpath(directory,stage);
                shard_index=original ? slot_index : 1,shard_count=original ? length(base["points"]) : 1)
        end
        result.failed == 0 ? 0 : 1
    end
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    try
        exit(!isempty(ARGS) && ARGS[1] == "run" && length(ARGS) == 6 ? run_point(ARGS[2:end]) : main())
    catch err
        println(stderr,"recovery_point: ",sprint(showerror,err))
        exit(1)
    end
end
