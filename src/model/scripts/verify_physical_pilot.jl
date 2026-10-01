#!/usr/bin/env julia
using JLD2, TOML, Dates
include(joinpath(@__DIR__,"../SpectralArtifacts.jl"))
using .SpectralArtifacts
include(joinpath(@__DIR__,"../SpectralRecovery.jl"))

function main(args=ARGS)
    1 <= length(args) <= 2 || error("Usage: verify_physical_pilot.jl RUN_ROOT [FIRST:LAST]  (default seeds 1:2, the pilot)")
    root = abspath(first(args))
    bounds = parse.(Int,split(length(args) == 2 ? args[2] : "1:2",':'))
    length(bounds) == 2 && 1 <= bounds[1] <= bounds[2] || error("Seeds must be FIRST:LAST")
    seeds = collect(bounds[1]:bounds[2])
    pilot = seeds == [1,2]
    cfg = TOML.parsefile(joinpath(root,"config.toml"))
    n,q = cfg["numerics"]["n_orb"],cfg["numerics"]["filling"]
    K = binomial(n,q)^2
    manifest = read_manifest(joinpath(root,"liouvillians","manifest.jld2"))
    expected = filter(p -> p["seed"] in seeds,manifest["points"])
    length(expected) == length(cfg["grid"]["etas"]) * length(seeds) || error("Expected $(length(seeds)) seeds at every eta value")
    cfg["grid"]["seed_count"] == 16 || error("Production identity must retain sixteen seeds")
    selection = joinpath(root,"recovery","selection.toml")
    replacements = SpectralRecovery.load_replacements(isfile(selection) ? selection : nothing,manifest)
    records = Dict{String,Any}[]
    for p in expected
        headers = Dict{String,Any}()
        paths = Dict{String,String}()
        for (dir,stage) in (("liouvillians","generate"),("eigen","eigen"),("svd","svd"))
            source_manifest,source_point,path = stage == "generate" ? (manifest,p,joinpath(root,dir,p["filename"])) :
                SpectralRecovery.selected_source(manifest,p,stage,joinpath(root,dir),replacements)
            h = read_artifact(path;stage,fields=String[])["header"]
            h["point_id"] == source_point["point_id"] || error("Wrong point: $path")
            h["generation_id"] == source_manifest["generation_id"] || error("Wrong production identity: $path")
            h["dimension"] == K || error("Wrong dimension: $path")
            paths[stage] = path
            headers[stage] = h
            push!(records,Dict("stage"=>stage,"path"=>path,"point_id"=>p["point_id"],
                "actual_point_id"=>source_point["point_id"],"actual_seed"=>source_point["seed"],
                "artifact_id"=>h["artifact_id"],"bytes"=>filesize(path),"mtime"=>mtime(path)))
            if stage != "generate"
                directory = haskey(replacements,(stage,p["point_id"])) ? replacements[(stage,p["point_id"])]["directory"] : root
                SpectralRecovery.inventory_status(root,directory,p["point_id"],source_point["seed"],stage) == "complete" || error("Invalid spectral source")
            end
        end
        ev = JLD2.load(paths["eigen"],"L_eigvals")
        sv = JLD2.load(paths["svd"],"L_svd_S_centered")
        length(ev) == length(sv) == K || error("Incomplete spectral arrays")
        all(isfinite,ev) && all(isfinite,sv) && all(>=(0),sv) || error("Invalid spectra")
        count(z->abs(z)<=1e-8,ev) == 1 || error("Unexpected steady-mode count for $(p["point_id"])")
        maximum(real,ev) <= 1e-8 || error("Unstable eigenvalues for $(p["point_id"])")
    end
    for dir in ("liouvillians","eigen","svd")
        actual = filter(p->endswith(p,".jld2") && p != "manifest.jld2",readdir(joinpath(root,dir)))
        expected_names = Set(p["filename"] for p in expected)
        issubset(Set(actual),expected_names) || error("Unexpected point coverage in $dir")
        required = Set(p["filename"] for p in expected if dir == "liouvillians" || !haskey(replacements,(dir,p["point_id"])))
        issubset(required,Set(actual)) || error("Missing non-replaced points in $dir")
    end
    analysis = strip(read(joinpath(root,"analysis_directory.txt"),String))
    report = TOML.parsefile(joinpath(analysis,"manifest.toml"))
    report["failed"] == 0 && report["selected_seeds"] == seeds || error("Analysis is incomplete for seeds $(bounds[1]):$(bounds[2])")
    Set((e["stage"],e["slot_id"],e["seed"]) for e in get(report,"replacements",[])) ==
        Set((e["stage"],e["slot_id"],e["seed"]) for e in values(replacements)) || error("Analysis replacement provenance mismatch")
    for record in records
        record["stage"] == "generate" && continue
        report["settings"]["sources"][record["stage"]][record["point_id"]] == record["artifact_id"] || error("Analysis uses stale source artifacts")
    end
    length(report["compatibility"]) == length(expected) || error("Missing figure inputs")
    figures = TOML.parsefile(joinpath(root,"figures","physical_figures_manifest.toml"))
    abspath(figures["analysis_manifest"]) == abspath(joinpath(analysis,"manifest.toml")) || error("Figures refer to stale analysis")
    figures["n_orb"] == n && figures["filling"] == q && figures["n_seeds"] == length(seeds) || error("Figure ensemble mismatch")
    get(figures,"csr_status","") == "complete" && get(figures,"csr_etas",[]) == cfg["grid"]["etas"] || error("Physical CSR panels incomplete")
    for check in values(figures["numerical_checks"])
        check["trace_residual_max"] <= 1e-6 || error("Trace-preservation residual too large")
        check["particle_residual_max"] <= 1e-6 || error("Particle-number residual too large")
    end
    for name in ("figure_1_physical","figure_2_dynamics","spectral_statistics__csr_strip__physical","supplement_raw_sigma_sff_dsff_physical"), ext in ("pdf","png")
        path = joinpath(root,"figures","$name.$ext")
        isfile(path) && filesize(path) > 1000 || error("Missing or empty figure: $path")
    end
    result = Dict("status"=>"passed","checked_at"=>string(now()),"n_orb"=>n,"filling"=>q,
        "dimension"=>K,"seeds"=>seeds,"production_seed_count"=>16,
        "generation_id"=>manifest["generation_id"],"artifacts"=>records,
        "replacements"=>collect(values(replacements)),
        "numerical_checks"=>figures["numerical_checks"],"full_submission_authorized"=>!pilot)
    # The pilot's acceptance report is kept; a wider ensemble gets its own.
    open(joinpath(root,pilot ? "validation.toml" : "validation__seeds=$(bounds[1])-$(bounds[2]).toml"),"w") do io
        TOML.print(io,result;sorted=true)
    end
    println(pilot ? "Pilot validation passed: eight reusable points and four figures; full submission still requires approval." :
        "Validation passed: $(length(expected)) points over seeds $(bounds[1]):$(bounds[2]) and four figures.")
end
abspath(PROGRAM_FILE) == (@__FILE__) && main()
