module SpectralPostprocessing
using JLD2, TOML, Printf, Random, Statistics
include("SpectralArtifacts.jl")
using .SpectralArtifacts
import .SpectralArtifacts: identity
include("SpectralRecovery.jl")
include("Dynamics.jl")
include("ComplexDSFFUnfolding.jl")
using .ComplexDSFFUnfolding
include("scripts/analysis/figures/_common_sff_helpers.jl")
include("scripts/analysis/csr_synthetic_ladder/csr_helpers.jl")
export parse_tasks, run_postprocessing

function parse_tasks(value::AbstractString)
    tasks = String[]
    for task in split(value,',')
        task in ("sff","sigma-sff","dsff","csr","dynamics") || throw(ArgumentError("Unknown task: $task"))
        append!(tasks,task == "sff" ? ["sigma-sff","dsff"] : [task])
    end
    length(unique(tasks)) == length(tasks) || throw(ArgumentError("Repeated task (including sff aliases)"))
    tasks
end

function time_grid(lo,hi,n)
    isfinite(lo) && isfinite(hi) && 0 < lo <= hi && n >= 2 || error("Time grid requires 0 < min <= max and n >= 2")
    exp.(range(log(Float64(lo)),log(Float64(hi));length=Int(n)))
end

const SMALL_FIELDS = ["metadata","basis_states","r_HD_frobenius","L_H_frobenius","L_D_frobenius",
    "H_spectral_span","L_trace_shift_mu_re","L_trace_shift_mu_im","n_jumps","jump_norm_sq"]

function source_data(path,stage;vectors=false)
    fields = vcat(SMALL_FIELDS,stage == "eigen" ? ["L_eigvals"] : ["L_svd_S_centered"])
    vectors && push!(fields,"L_eigvecs")
    read_artifact(path;stage,fields)
end

function checked_sources(cfg,eigen_dir,svd_dir,tasks;seeds=nothing,replacements=nothing)
    need_eigen = any(t -> t in ("dsff","csr","dynamics"),tasks)
    need_svd = "sigma-sff" in tasks
    need_eigen && eigen_dir === nothing && error("Requested tasks require --eigen; run pipeline eigen first")
    need_svd && svd_dir === nothing && error("Requested tasks require --svd; run pipeline svd first")
    sources = Dict{String,Dict{String,String}}()
    manifests = Dict{String,Any}()
    substitutions = nothing
    for (stage,dir) in (("eigen",eigen_dir),("svd",svd_dir))
        dir === nothing && continue
        m = read_manifest(joinpath(dir,"manifest.jld2"))
        m["schema_version"] == 1 && m["stage"] == stage || error("Invalid $stage manifest")
        SpectralArtifacts.physics_identity(cfg) == m["physics_id"] || error("Configuration does not match $stage physics")
        identity(cfg["grid"]) == identity(m["resolved_config"]["grid"]) || error("Configuration grid does not match $stage ensemble")
        stage_substitutions = SpectralRecovery.load_replacements(replacements,m)
        substitutions === nothing && (substitutions = stage_substitutions)
        if seeds !== nothing
            requested = Int.(collect(seeds))
            !isempty(requested) && length(unique(requested)) == length(requested) || error("Seed selection must be nonempty and unique")
            for group in grouped_points(m["points"])
                issubset(Set(requested),Set(p["seed"] for p in group)) || error("Requested seeds are outside the production manifest")
            end
            # Keep production generation_id and artifact headers unchanged.
            m = merge(m,Dict("points"=>filter(p -> p["seed"] in requested,m["points"])))
        end
        manifests[stage] = m
        paths = Dict{String,String}()
        for p in m["points"]
            id = p["point_id"]
            source_manifest,source_point,path = SpectralRecovery.selected_source(m,p,stage,dir,stage_substitutions)
            data = source_data(path,stage)
            h = data["header"]
            h["point_id"] == source_point["point_id"] && h["physics_id"] == m["physics_id"] && h["generation_id"] == source_manifest["generation_id"] || error("Source identity mismatch: $path")
            if haskey(stage_substitutions,(stage,id))
                entry = stage_substitutions[(stage,id)]
                source_header = SpectralArtifacts.read_generation_header(joinpath(entry["directory"],"liouvillians",source_point["filename"]);
                    expected=Dict("point_id"=>source_point["point_id"],"generation_id"=>source_manifest["generation_id"],"physics_id"=>m["physics_id"],"dimension"=>h["dimension"]))
                h["source_id"] == source_header["artifact_id"] || error("Replacement source mismatch")
            elseif any(haskey(stage_substitutions,(s,id)) for s in ("eigen","svd"))
                source_header = SpectralArtifacts.read_generation_header(joinpath(dirname(abspath(dir)),"liouvillians",p["filename"]);
                    expected=Dict("point_id"=>id,"generation_id"=>m["generation_id"],"physics_id"=>m["physics_id"],"dimension"=>h["dimension"]))
                h["source_id"] == source_header["artifact_id"] || error("Retained stage source mismatch")
            end
            num = cfg["numerics"]
            expected_dim = binomial(num["n_orb"],get(num,"filling",div(num["n_orb"],2)))^2
            h["dimension"] == expected_dim || error("Source dimension mismatch: $path")
            paths[id] = abspath(path)
        end
        length(paths) == length(m["points"]) || error("Duplicate points in $stage manifest")
        sources[stage] = paths
    end
    if length(manifests) == 2
        manifests["eigen"]["generation_id"] == manifests["svd"]["generation_id"] || error("Eigen and SVD ensembles differ")
        for id in keys(sources["eigen"])
            eh = read_artifact(sources["eigen"][id];stage="eigen",fields=String[])["header"]
            sh = read_artifact(sources["svd"][id];stage="svd",fields=String[])["header"]
            # Stage-only replacement is explicitly authorized by the selection file.
            # Default ensembles still require the same saved Liouvillian in both stages.
            explicit = haskey(substitutions,("eigen",id)) || haskey(substitutions,("svd",id))
            explicit || eh["source_id"] == sh["source_id"] || error("Eigen and SVD used different Liouvillians for $id")
        end
    end
    isempty(manifests) && error("No spectral input")
    manifest = first(values(manifests))
    selected_ids = Set(p["point_id"] for p in manifest["points"])
    entries = sort([e for e in values(substitutions) if e["slot_id"] in selected_ids];by=e->(e["stage"],e["slot_id"]))
    merge(manifest,Dict("replacements"=>entries)), sources
end

function analysis_header(source,stage,point_id,analysis_id)
    h = copy(source["header"])
    delete!(h,"artifact_id")
    h["stage"] = stage
    h["point_id"] = point_id
    h["source_id"] = source["header"]["artifact_id"]
    h["analysis_id"] = analysis_id
    h
end

function dynamics_payload(ev,cfg)
    tg = cfg["numerics"]["time_grid"]
    times = time_grid(tg["t_min"],tg["t_max"],tg["n"])
    result = dynamics_from_eigen(ev["L_eigvals"],ev["L_eigvecs"],ev["basis_states"],times;
        n_orb=cfg["numerics"]["n_orb"],entropy_eig_tol=get(get(cfg,"dynamics",Dict()),"entropy_eig_tol",1e-12))
    Dict{String,Any}("times"=>result.times,"entropy_t"=>result.entropy_t,"populations_t"=>result.populations_t,
        "dynamics_trace_residual_max"=>result.metadata["trace_residual_max"],
        "dynamics_particle_residual_max"=>result.metadata["particle_residual_max"])
end

function ensemble_analysis(task,data,cfg,group_index)
    options = get(get(cfg,"analysis",Dict()),replace(task,'-'=>'_'),Dict())
    n_boot = Int(get(options,"n_boot",500))
    rng = MersenneTwister(Int(get(options,"rng_seed",20260804))+group_index)
    if task == "sigma-sff"
        sigmas = [d["L_svd_S_centered"] for d in data]
        epsilon = Float64(get(options,"epsilon_zero",1e-8))
        if get(cfg,"model","physical") == "multimode"
            filtered = [sort(filter(>(epsilon),s)) for s in sigmas]
            count = minimum(length,filtered)
            sigmas = [s[1:count] for s in filtered]
        end
        taus = time_grid(get(options,"t_min",1e-3),get(options,"t_max",10.0),get(options,"n_times",400))
        boot = bootstrap_kunf_ham(sigmas,2pi .* taus;
            analysis_window=Tuple(get(options,"window",[0.05,0.95])),epsilon_zero=epsilon,
            n_bins=Int(get(options,"n_bins",40)),degree=Int(get(options,"degree",5)),n_boot,rng)
        return Dict("taus"=>taus,"bootstrap"=>boot)
    end
    spectra = [filter(z -> abs(z) > STEADY_TOL,d["L_eigvals"]) for d in data]
    if task == "csr"
        ratios = [complex_spacing_ratios_bulk(s;
            bulk_fraction=get(options,"bulk_fraction",CSR_BULK_FRACTION),
            rmax_quantile=get(options,"rmax_quantile",CSR_RMAX_QUANTILE),
            im_axis_tol=get(options,"im_axis_tol",CSR_IM_AXIS_TOL),
            distance_tol=get(options,"distance_tol",DISTANCE_TOL)) for s in spectra]
        return Dict("ratios_by_seed"=>[r[1] for r in ratios],"diagnostics_by_seed"=>[r[2:end] for r in ratios])
    elseif task == "dsff"
        beta = Float64(get(options,"beta",0.5))
        fit = fit_fixed_power_policy(spectra,beta;
            n_bins=Int(get(options,"density_bins",256)),smooth_sigma=get(options,"density_sigma",2.0),
            minimum_effective_count=get(options,"minimum_effective_count",500.0))
        mapped = mapped_spectra(spectra,fit)
        times = time_grid(get(options,"t_min",10.0^-2.5),get(options,"t_max",10.0),get(options,"n_times",400))
        theta = Float64(get(options,"theta",pi/4))
        Z,weights = partition_traces(mapped,fit.filter,times,theta)
        boot = bootstrap_connected_dsff(Z,[sum(abs2,w) for w in weights];n_boot,rng)
        spacing = ensemble_weighted_spacing(spectra,fit)
        return Dict("times"=>times,"axis"=>"time conjugate to unfolded spectrum (no reference calibration)",
            "theta"=>theta,"beta"=>beta,"unfolding"=>fit,"spacing"=>spacing,"bootstrap"=>boot)
    end
    error("Unknown ensemble analysis: $task")
end

function grouped_points(points)
    groups = Dict{String,Vector{Dict{String,Any}}}()
    for point in points
        parameters = Dict(k=>v for (k,v) in point if !(k in ("point_id","filename","seed")))
        key = identity(parameters)
        push!(get!(groups,key,Dict{String,Any}[]),point)
    end
    [sort!(groups[k];by=p->p["seed"]) for k in sort!(collect(keys(groups)))]
end

function export_legacy(cfg,directory,manifest,sources,dynamics_paths,analysis_id)
    length(sources) == 2 && length(dynamics_paths) == length(manifest["points"]) || return String[]
    outputs = String[]
    # Export payloads carry the same analysis identity, without large eigenvectors.
    for group in grouped_points(manifest["points"])
        payloads = Dict{String,Any}[]
        for p in group
            ev = source_data(sources["eigen"][p["point_id"]],"eigen")
            sv = source_data(sources["svd"][p["point_id"]],"svd")
            dyn = read_artifact(dynamics_paths[p["point_id"]];stage="dynamics")
            payload = merge(ev,sv,dyn)
            delete!(payload,"header")
            if !isempty(get(manifest,"replacements",[]))
                actual_seed(stage) = get(Dict((e["stage"],e["slot_id"])=>e["seed"] for e in manifest["replacements"]),(stage,p["point_id"]),p["seed"])
                payload["realization_slot"] = p["seed"]
                payload["eigen_seed"] = actual_seed("eigen")
                payload["svd_seed"] = actual_seed("svd")
                payload["eigen_metadata"] = ev["metadata"]
                payload["svd_metadata"] = sv["metadata"]
            end
            push!(payloads,payload)
        end
        if get(cfg,"model","physical") == "physical"
            for (p,payload) in zip(group,payloads)
                path = joinpath(directory,"spectra",p["filename"])
                export_payload(path,payload,analysis_id)
                push!(outputs,path)
            end
        else
            num = cfg["numerics"]; dt = first(group)["delta_tilde"]
            delta = cfg["loss"]["delta_cd_over_2pi_mhz"]
            slug(x) = replace(@sprintf("%.6g",x),"."=>"p","-"=>"m")
            name = "multimode_centered__norb=$(num["n_orb"])__f=$(num["filling"])__ngrid=$(num["n_grid"])__M=$(num["mode_cutoff"])__deltacd=$(slug(delta))__dtilde=$(slug(dt)).jld2"
            payload = Dict{String,Any}("seeds"=>[p["seed"] for p in group],"sigmas"=>[p["L_svd_S_centered"] for p in payloads],
                "mu_re"=>[p["L_trace_shift_mu_re"] for p in payloads],"mu_im"=>[p["L_trace_shift_mu_im"] for p in payloads],
                "r_HD"=>[p["r_HD_frobenius"] for p in payloads],"n_jumps"=>[p["n_jumps"] for p in payloads],
                "jump_norm_sq"=>[p["jump_norm_sq"] for p in payloads],"L_eigvals"=>[p["L_eigvals"] for p in payloads],
                "delta_tilde"=>dt,"delta_cd_over_2pi_mhz"=>delta,"n_orb"=>num["n_orb"],"filling"=>num["filling"],
                "n_grid"=>num["n_grid"],"mode_cutoff"=>num["mode_cutoff"],"n_seeds"=>length(group),"shard_index"=>1,"shard_count"=>1,
                "times"=>first(payloads)["times"])
            if !isempty(get(manifest,"replacements",[]))
                payload["eigen_seeds"] = [p["eigen_seed"] for p in payloads]
                payload["svd_seeds"] = [p["svd_seed"] for p in payloads]
            end
            for key in ("entropy_t","populations_t","dynamics_trace_residual_max","dynamics_particle_residual_max")
                payload[key] = [p[key] for p in payloads]
            end
            path = joinpath(directory,"multimode",name)
            export_payload(path,payload,analysis_id); push!(outputs,path)
        end
    end
    outputs
end

function export_payload(path,payload,analysis_id)
    function check(f)
        f["analysis_id"] == analysis_id || error("Conflicting legacy export: $path")
        all(k -> haskey(f,k),keys(payload)) || error("Incomplete legacy export: $path")
    end
    isfile(path) && return JLD2.jldopen(check,path,"r")
    SpectralArtifacts.with_claim(path) do
        isfile(path) && return JLD2.jldopen(check,path,"r")
        SpectralArtifacts.atomic_write(path,check) do file
            file["analysis_id"] = analysis_id
            for (k,v) in payload; file[k] = v; end
        end
    end
end

# Sharding splits the dynamics points only; it is deliberately absent from settings
# so every shard resolves the same analysis_id, hence the same output directory.
function run_postprocessing(cfg,output;eigen_dir=nothing,svd_dir=nothing,tasks=["sigma-sff","dsff","csr","dynamics"],seeds=nothing,replacements=nothing,shard_index=1,shard_count=1)
    1 <= shard_index <= shard_count || error("Require 1 <= shard-index <= shard-count")
    tasks = parse_tasks(join(tasks,','))
    manifest,sources = checked_sources(cfg,eigen_dir,svd_dir,tasks;seeds,replacements)
    source_ids = Dict(stage=>Dict(id=>read_artifact(path;stage,fields=String[])["header"]["artifact_id"] for (id,path) in paths) for (stage,paths) in sources)
    settings = Dict("tasks"=>sort(tasks),"analysis"=>get(cfg,"analysis",Dict()),
        "time_grid"=>get(cfg["numerics"],"time_grid",Dict()),"dynamics"=>get(cfg,"dynamics",Dict()),"sources"=>source_ids)
    isempty(manifest["replacements"]) || (settings["replacement_export_schema"] = 1)
    analysis_id = identity(settings)
    directory = joinpath(abspath(output),analysis_id)
    ensure_manifest(joinpath(directory,"manifest.jld2"),Dict("schema_version"=>1,"analysis_id"=>analysis_id,"settings"=>settings,"generation_id"=>manifest["generation_id"]))
    dynamics_paths = Dict{String,String}(); outputs = String[]; failed = 0
    if "dynamics" in tasks
        for (i,p) in enumerate(manifest["points"])
            mod(i-1,shard_count)+1 == shard_index || continue
            id = p["point_id"]; path = joinpath(directory,"dynamics",p["filename"])
            source = read_artifact(sources["eigen"][id];stage="eigen",fields=String[])
            try
                publish_artifact(() -> dynamics_payload(source_data(sources["eigen"][id],"eigen";vectors=true),cfg),
                    path,analysis_header(source,"dynamics",id,analysis_id))
                dynamics_paths[id] = path; push!(outputs,path)
            catch err
                err isa InterruptException && rethrow()
                failed += 1; println(stderr,"FAILED dynamics $id: ",sprint(showerror,err))
            end
        end
    end
    # Ensemble tasks need every seed at once and are cheap beside dynamics, so shards skip them.
    ensemble = shard_count == 1 ? filter(!=("dynamics"),tasks) : String[]
    for (index,group) in enumerate(grouped_points(manifest["points"])), task in ensemble
        stage = task == "sigma-sff" ? "svd" : "eigen"
        first_source = read_artifact(sources[stage][first(group)["point_id"]];stage,fields=String[])
        group_id = identity(Dict("points"=>[p["point_id"] for p in group]))
        path = joinpath(directory,task,group_id*".jld2")
        try
            publish_artifact(path,analysis_header(first_source,"analysis",group_id,analysis_id)) do
                data = [source_data(sources[stage][p["point_id"]],stage) for p in group]
                Dict("result"=>ensemble_analysis(task,data,cfg,index),"seeds"=>[p["seed"] for p in group],"task"=>task)
            end
            push!(outputs,path)
        catch err
            err isa InterruptException && rethrow()
            failed += 1; println(stderr,"FAILED $task $group_id: ",sprint(showerror,err))
        end
    end
    compatibility = shard_count == 1 ? export_legacy(cfg,directory,manifest,sources,dynamics_paths,analysis_id) : String[]
    report = Dict{String,Any}("analysis_id"=>analysis_id,"tasks"=>tasks,"failed"=>failed,
        "selected_seeds"=>sort(unique(p["seed"] for p in manifest["points"])),
        "replacements"=>manifest["replacements"],
        "seed_note"=>"selected_seeds are realization slots; replacements record actual seeds independently for eigen and SVD",
        "production_seed_count"=>cfg["grid"]["seed_count"],
        "shard_index"=>shard_index,"shard_count"=>shard_count,
        "outputs"=>outputs,"compatibility"=>compatibility,"settings"=>settings,
        "compatibility_note"=>shard_count > 1 ? "Dynamics shard only; run the unsharded merge for ensemble tasks and exports" :
            isempty(compatibility) ? "Full figure exports require eigen, SVD and completed dynamics" : "complete")
    report_path = joinpath(directory,shard_count == 1 ? "manifest.toml" : @sprintf("manifest.shard-%d-of-%d.toml",shard_index,shard_count))
    # This small status report can change as a failed analysis resumes; results are immutable.
    SpectralArtifacts.with_claim(report_path) do
        temp,io = mktemp(directory)
        try
            SpectralArtifacts.DatasetPaths.print_manifest(io,report;sorted=true,path=report_path); close(io)
            Base.Filesystem.rename(temp,report_path)
        finally
            isopen(io) && close(io); isfile(temp) && rm(temp)
        end
    end
    println("Analysis manifest: $report_path")
    (directory=directory,manifest=report_path,compatibility=compatibility,failed=failed)
end
end
