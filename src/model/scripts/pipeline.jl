#!/usr/bin/env julia
# Explicit, resumable numerical stages. Help does not load numerical dependencies.
module PipelineCLI
const MODEL = normpath(joinpath(@__DIR__,".."))
const HELP = """
Usage: julia --project=src/environment src/model/scripts/pipeline.jl STAGE [OPTIONS]

  generate    --config RUN.toml --output MATRICES
  eigen       --input MATRICES --output EIGEN
  svd         --input MATRICES --output SVD
  postprocess --config RUN.toml --eigen EIGEN --svd SVD --output ANALYSIS

Options:
  --tasks sff,csr,dynamics       Postprocessing tasks (default: all).
                                sff = sigma-sff + dsff; tasks can run separately.
  --seeds FIRST:LAST            Explicit postprocessing subset of the production manifest.
  --replacements FILE.toml      Explicit stage-specific timeout replacements for postprocessing.
  --shard-index I --shard-count N  1-based shards for generate/eigen/svd, and for
                                the dynamics points of postprocess.
  --blas-threads N               BLAS threads in this job (default: library setting).
  --help                        Show this help.

All data paths are explicit; relative paths start at the working directory.
Completed artifacts are validated and skipped. Invalid files are preserved.
Eigen and SVD use separate output directories and may run concurrently.
Postprocessing prints its manifest and figure-compatible output paths.
"""

function parse_cli(args)
    isempty(args) && error("A stage is required; use --help")
    "--help" in args && return nothing
    stage = first(args)
    stage in ("generate","eigen","svd","postprocess") || error("Unknown stage: $stage")
    allowed = stage == "generate" ? ["config","output","shard-index","shard-count","blas-threads"] :
        stage == "postprocess" ? ["config","output","eigen","svd","tasks","blas-threads","seeds","replacements","shard-index","shard-count"] :
        ["input","output","shard-index","shard-count","blas-threads"]
    opts = Dict{String,String}(); i=2
    while i <= length(args)
        token = args[i]
        startswith(token,"--") || error("Expected an option: $token")
        pieces = split(token[3:end],'=';limit=2)
        key = first(pieces)
        key in allowed || error("Unknown option for $stage: --$key")
        haskey(opts,key) && error("Repeated option: --$key")
        if length(pieces) == 2
            value = pieces[2]
        else
            i+=1
            i <= length(args) && !startswith(args[i],"--") || error("--$key requires a value")
            value = args[i]
        end
        isempty(value) && error("--$key requires a nonempty value")
        opts[key] = value; i+=1
    end
    for key in (stage in ("generate","postprocess") ? ["config","output"] : ["input","output"])
        haskey(opts,key) || error("$stage requires --$key")
    end
    haskey(opts,"shard-index") == haskey(opts,"shard-count") || error("Provide both shard options")
    shard_index = parse(Int,get(opts,"shard-index","1"))
    shard_count = parse(Int,get(opts,"shard-count","1"))
    1 <= shard_index <= shard_count || error("Require 1 <= shard-index <= shard-count")
    threads = haskey(opts,"blas-threads") ? parse(Int,opts["blas-threads"]) : nothing
    threads === nothing || threads > 0 || error("--blas-threads must be positive")
    seeds = nothing
    if haskey(opts,"seeds")
        bounds = parse.(Int,split(opts["seeds"],':'))
        length(bounds) == 2 && 1 <= bounds[1] <= bounds[2] || error("--seeds requires positive FIRST:LAST")
        seeds = collect(bounds[1]:bounds[2])
    end
    for key in ("config","output","input","eigen","svd","replacements")
        haskey(opts,key) && (opts[key] = abspath(opts[key]))
    end
    (stage=stage,opts=opts,shard_index=shard_index,shard_count=shard_count,threads=threads,seeds=seeds)
end

function main(args=ARGS)
    try
        cli = parse_cli(args)
        cli === nothing && (println(HELP); return 0)
        if cli.stage == "postprocess"
            include(joinpath(MODEL,"SpectralPostprocessing.jl"))
        else
            include(joinpath(MODEL,"SpectralPipeline.jl"))
        end
        # Modules are loaded after parsing so --help stays cheap.
        result = Base.invokelatest() do
            opts = cli.opts
            if cli.stage == "postprocess"
                P = SpectralPostprocessing
                cli.threads === nothing || P.LinearAlgebra.BLAS.set_num_threads(cli.threads)
                cfg = P.SpectralArtifacts.load_pipeline_config(opts["config"])
                P.run_postprocessing(cfg,opts["output"];eigen_dir=get(opts,"eigen",nothing),svd_dir=get(opts,"svd",nothing),
                    tasks=P.parse_tasks(get(opts,"tasks","sff,csr,dynamics")),seeds=cli.seeds,replacements=get(opts,"replacements",nothing),
                    shard_index=cli.shard_index,shard_count=cli.shard_count)
            else
                P = SpectralPipeline
                cli.threads === nothing || P.LinearAlgebra.BLAS.set_num_threads(cli.threads)
                if cli.stage == "generate"
                    cfg = P.load_pipeline_config(opts["config"])
                    P.run_generation(cfg,opts["output"];shard_index=cli.shard_index,shard_count=cli.shard_count)
                else
                    P.run_spectral(cli.stage,opts["input"],opts["output"];shard_index=cli.shard_index,shard_count=cli.shard_count)
                end
            end
        end
        result.failed == 0 ? 0 : 1
    catch err
        println(stderr,"pipeline: ",sprint(showerror,err))
        1
    end
end
end
if abspath(PROGRAM_FILE) == @__FILE__
    exit(PipelineCLI.main())
end
