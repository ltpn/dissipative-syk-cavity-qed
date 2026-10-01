module SpectralPipeline
using JLD2, LinearAlgebra, TOML, SHA
include("SpectralArtifacts.jl")
using .SpectralArtifacts
import .SpectralArtifacts: identity
include("SpectralRecovery.jl")
module Generation
include("scripts/sweeps/spectrum_sweep.jl")
include("MultimodeHamiltonian.jl")
end
export load_pipeline_config, pipeline_manifest, build_point, run_generation, run_spectral, spectral_payload
const REPO = normpath(joinpath(@__DIR__, "..", ".."))

const load_pipeline_config = SpectralArtifacts.load_pipeline_config

function pipeline_manifest(cfg)
    model = get(cfg,"model","physical")
    model in ("physical","multimode") || error("Unknown model: $model")
    points = Dict{String,Any}[]
    if model == "physical"
        for p in Generation.build_sweep_grid(Generation.grid_spec_from_config(cfg))
            push!(points,Dict("point_id"=>Generation.point_key(p),
                "filename"=>Generation.spectrum_filename(p),"model"=>model,
                "seed"=>p.seed,"config"=>p.config,"eta"=>p.eta,"gamma"=>p.gamma,"svd_tol"=>p.tol))
        end
    else
        g = cfg["grid"]
        g["seed_count"] > 0 || error("seed_count must be positive")
        for dt in g["delta_tildes"], seed in 1:g["seed_count"]
            isfinite(dt) && dt >= 0 || error("Invalid delta_tilde")
            id = "multimode__dtilde=$(Float64(dt))__seed=$seed"
            push!(points,Dict("point_id"=>id,"filename"=>id*".jld2",
                "model"=>model,"seed"=>seed,"delta_tilde"=>Float64(dt)))
        end
    end
    isempty(points) && error("Empty generation grid")
    sort!(points;by=p->p["point_id"])
    length(unique(p["point_id"] for p in points)) == length(points) || error("Duplicate grid points")
    physics_id = SpectralArtifacts.physics_identity(cfg)
    generation_id = identity(Dict("physics_id"=>physics_id,"points"=>points))
    Dict{String,Any}("schema_version"=>1,"stage"=>"generate","physics_id"=>physics_id,
        "generation_id"=>generation_id,"resolved_config"=>cfg,"points"=>points)
end

function build_point(cfg,point)
    if point["model"] == "physical"
        p = Generation.SweepPoint(point["config"],point["eta"],point["gamma"],point["svd_tol"],point["seed"])
        ldcfg = Generation.make_liouvillian_config(p,cfg["numerics"],get(cfg,"seeds",Dict()))
        result = Generation.build_lamb_dicke_liouvillian(ldcfg,p.seed,p.eta)
    else
        num = cfg["numerics"]
        params = Dict{Symbol,Any}(Symbol(k)=>v for (k,v) in num if k != "time_grid")
        params[:weight_type] = Symbol(get(params,:weight_type,"speckle"))
        params[:delta_tilde] = point["delta_tilde"]
        p = Generation.MultimodeParams(;params...)
        loss = Generation.MultimodeCavityLossParams(; (Symbol(k)=>v for (k,v) in cfg["loss"])...)
        result = Generation.multimode_open_system_block(p,loss;seed=point["seed"])
    end
    L,H = Matrix(result.L),Matrix(result.H)
    id = Matrix{ComplexF64}(I,size(H)...)
    LH = -1im .* (kron(id,H) .- kron(transpose(H),id))
    h,d = norm(LH),norm(L .- LH)
    energies = real.(eigvals(H))
    mu = tr(L)/size(L,1)
    Dict{String,Any}("L"=>L,"H"=>H,"basis_states"=>result.basis_states,
        "metadata"=>result.metadata,"r_HD_frobenius"=>d > 0 ? h/d : Inf,
        "L_H_frobenius"=>h,"L_D_frobenius"=>d,"H_spectral_span"=>maximum(energies)-minimum(energies),
        "L_trace_shift_mu_re"=>real(mu),"L_trace_shift_mu_im"=>imag(mu),
        "n_jumps"=>length(result.jump_operators),
        "jump_norm_sq"=>[norm(J)^2 for J in result.jump_operators])
end

function header_for(manifest,point,stage; source_id="")
    n = manifest["resolved_config"]["numerics"]
    dim = binomial(n["n_orb"],get(n,"filling",div(n["n_orb"],2)))^2
    revision = try strip(read(`git -C $REPO rev-parse HEAD`,String)) catch; "unknown" end
    Dict{String,Any}("schema_version"=>1,"stage"=>stage,"point_id"=>point["point_id"],
        "physics_id"=>manifest["physics_id"],"generation_id"=>manifest["generation_id"],
        "source_id"=>source_id,"dimension"=>dim,"code_revision"=>revision)
end

function run_points(f,manifest;shard_index=1,shard_count=1)
    Generation.validate_shard_spec(shard_index,shard_count)
    written=skipped=failed=0
    for (i,p) in enumerate(manifest["points"])
        mod(i-1,shard_count)+1 == shard_index || continue
        try
            status = f(p)
            status == :written ? (written+=1) : (skipped+=1)
            println("$status $(p["point_id"])")
        catch err
            err isa InterruptException && rethrow()
            failed+=1
            println(stderr,"FAILED $(p["point_id"]): ",sprint(showerror,err))
        end
        flush(stdout)
    end
    (written=written,skipped=skipped,failed=failed)
end

function run_generation(cfg,output;shard_index=1,shard_count=1)
    manifest = pipeline_manifest(cfg)
    ensure_manifest(joinpath(output,"manifest.jld2"),manifest)
    run_points(manifest;shard_index,shard_count) do point
        publish_artifact(() -> build_point(cfg,point),joinpath(output,point["filename"]),
            header_for(manifest,point,"generate"))
    end
end

function run_replacement_generation(base,slot_id,seed,output)
    manifest = SpectralRecovery.replacement_manifest(base,slot_id,seed)
    ensure_manifest(joinpath(output,"manifest.jld2"),manifest)
    run_points(manifest) do point
        publish_artifact(() -> build_point(base["resolved_config"],point),joinpath(output,point["filename"]),
            header_for(manifest,point,"generate"))
    end
end

function spectral_payload(stage,source)
    small = Dict{String,Any}(k=>v for (k,v) in source if !(k in ("L","H","header")))
    L = source["L"]
    if stage == "eigen"
        info = Dict{String,Any}()
        F = Generation.robust_eigen(L;diagnostics=info)
        merge!(small,Dict("L_eigvals"=>ComplexF64.(F.values),
            "L_eigvecs"=>Matrix{ComplexF64}(F.vectors),"solver_diagnostics"=>info))
    elseif stage == "svd"
        mu = tr(L)/size(L,1)
        centered = copy(L)
        for i in axes(centered,1); centered[i,i] -= mu; end
        merge!(small,Dict("L_svd_S_centered"=>Generation.robust_svdvals(centered),
            "L_trace_shift_mu_re"=>real(mu),"L_trace_shift_mu_im"=>imag(mu)))
    else
        error("Unknown spectral stage: $stage")
    end
    small
end

function run_spectral(stage::String,input::AbstractString,output::AbstractString;shard_index=1,shard_count=1)
    stage in ("eigen","svd") || error("Unknown spectral stage: $stage")
    manifest = read_manifest(joinpath(input,"manifest.jld2"))
    manifest["stage"] == "generate" || error("$stage requires saved Liouvillians")
    mkpath(output)
    realpath(input) != realpath(output) || error("Input and output directories must differ")
    ensure_manifest(joinpath(output,"manifest.jld2"),merge(manifest,Dict("stage"=>stage)))
    run_points(manifest;shard_index,shard_count) do point
        path = joinpath(input,point["filename"])
        expected = Dict("generation_id"=>manifest["generation_id"],"point_id"=>point["point_id"])
        h = read_artifact(path;stage="generate",expected,fields=String[])["header"]
        publish_artifact(joinpath(output,point["filename"]),
            header_for(manifest,point,stage;source_id=h["artifact_id"])) do
            source = read_artifact(path;stage="generate",expected)
            source["header"]["artifact_id"] == h["artifact_id"] || error("Source replaced during computation")
            spectral_payload(stage,source)
        end
    end
end
end
