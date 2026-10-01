#!/usr/bin/env julia
"""Portable canonical figure workflow: cached plotting, final statistics, or decomposition postprocessing."""
module ReproduceFigures
using TOML, Printf
include(joinpath(@__DIR__, "..", "DatasetPaths.jl"))
const REPO = normpath(joinpath(@__DIR__, "..", "..", ".."))
const FIG = joinpath(@__DIR__, "analysis", "figures")
const RUNS = ["physical_n10_q4","syk4_control","physical_n6","physical_n8",
    "multimode_n10_q4","multimode_n6_f3","multimode_n8_f4","fig3_m300"]
const HELP = """
Usage: julia --project=src/environment src/model/scripts/reproduce_figures.jl [OPTIONS]
  --data-root DIR     Dataset root (SYK_DATA_ROOT, otherwise reproduction_data/).
  --manifest FILE    Dataset TOML (otherwise DATA_ROOT/dataset.toml or bundled template).
  --output-dir DIR   PDFs, PNGs and manifests (default: reproduced_figures/).
  --work-dir DIR     Recomputed caches (default: OUTPUT_DIR/work).
  --header-root DIR  Matrix-header sidecars (SYK_HEADER_ROOT, otherwise reproduction_data/).
  --mode plot        Reuse prepared DSFF data; compute plotted statistics (default).
  --mode statistics  Recompute DSFF and all plotted statistics from light spectra.
  --mode postprocess Recompute spectra analysis/dynamics from downloaded eigen/SVD files,
                     then recompute final figure statistics. Requires full eigenvectors.
  --figures LIST     Comma-separated 1,2,3,csr,raw (default: all five).
  --check            Validate the selected inputs without creating outputs.
  --help             Show this help.
Relative command-line roots are relative to your working directory; stored paths
are relative to the dataset root. Saved product paths remain relative.
"""

function parse_cli(argv)
    opts = Dict("data-root"=>get(ENV,"SYK_DATA_ROOT",joinpath(REPO,"reproduction_data")),
        "output-dir"=>joinpath(REPO,"reproduced_figures"),"mode"=>"plot",
        "figures"=>"1,2,3,csr,raw","header-root"=>get(ENV,"SYK_HEADER_ROOT",joinpath(REPO,"reproduction_data")))
    flags = Set{String}()
    supplied = Set{String}()
    i=1
    while i<=length(argv)
        argument=argv[i]
        if argument in ("--help","--check")
            push!(flags,argument[3:end]); i+=1; continue
        end
        startswith(argument,"--") || error("Unexpected argument: $argument")
        parts=split(argument[3:end],'=';limit=2); key=parts[1]
        key in ("data-root","manifest","output-dir","work-dir","header-root","mode","figures") || error("Unknown option: --$key")
        push!(supplied,key)
        if length(parts)==2
            opts[key]=parts[2]
        else
            i+=1; i<=length(argv) || error("--$key needs a value"); opts[key]=argv[i]
        end
        i+=1
    end
    opts["mode"] in ("plot","statistics","postprocess") || error("Invalid --mode")
    figures=split(opts["figures"],',')
    all(f->f in ("1","2","3","csr","raw"),figures) && length(unique(figures))==length(figures) || error("Invalid --figures")
    if "manifest" in supplied && !("data-root" in supplied) && !haskey(ENV,"SYK_DATA_ROOT")
        path=abspath(opts["manifest"])
        metadata=TOML.parsefile(path)
        if haskey(metadata,"path_root")
            opts["data-root"]=DatasetPaths.resolve_path(metadata["path_root"];root=dirname(path))
        end
    end
    for key in ("data-root","output-dir","header-root"); opts[key]=abspath(opts[key]); end
    opts["work-dir"]=abspath(get(opts,"work-dir",joinpath(opts["output-dir"],"work")))
    opts["manifest"]=abspath(get(opts,"manifest",isfile(joinpath(opts["data-root"],"dataset.toml")) ?
        joinpath(opts["data-root"],"dataset.toml") : joinpath(REPO,"reproduction_data/dataset.toml")))
    (opts=opts,figures=figures,check="check" in flags,help="help" in flags)
end

analysis_path(root,manifest,run) = DatasetPaths.resolve_path(manifest["selected_analysis"][run];root)
config_path(root,run) = joinpath(root,"runs",run,"config.toml")
spectra_path(root,manifest,run) = joinpath(analysis_path(root,manifest,run),"spectra")
multimode_path(root,manifest,run) = joinpath(analysis_path(root,manifest,run),"multimode")
slug(x) = replace(@sprintf("%.6g",x),"."=>"p","-"=>"m")

function multimode_file(root,manifest,run,spacing)
    cfg=DatasetPaths.read_manifest(config_path(root,run)); n=cfg["numerics"]
    filename="multimode_centered__norb=$(n["n_orb"])__f=$(n["filling"])__ngrid=$(n["n_grid"])__M=$(n["mode_cutoff"])__deltacd=$(slug(cfg["loss"]["delta_cd_over_2pi_mhz"]))__dtilde=$(slug(spacing)).jld2"
    joinpath(multimode_path(root,manifest,run),filename)
end

function needed_runs(figures)
    needed=Set{String}()
    for figure in figures
        union!(needed,figure=="1" ? ["physical_n10_q4","syk4_control","physical_n6","physical_n8"] :
            figure=="2" ? ["physical_n10_q4","syk4_control"] :
            figure=="3" ? ["multimode_n10_q4","multimode_n6_f3","multimode_n8_f4","fig3_m300"] :
            figure=="csr" ? ["physical_n10_q4","syk4_control","multimode_n10_q4"] :
            ["physical_n10_q4","syk4_control","multimode_n10_q4","fig3_m300"])
    end
    filter(r->r in needed,RUNS)
end

function preflight(root,manifest; figures=["1","2","3","csr","raw"], mode="plot")
    check_figure_metadata(root,figures,mode)
    for run in needed_runs(figures)
        cfg=DatasetPaths.read_manifest(config_path(root,run))
        n=cfg["numerics"]; N=n["n_orb"]; Q=get(n,"filling",div(N,2)); K=binomial(N,Q)^2
        seeds=collect(1:cfg["grid"]["seed_count"])
        if startswith(run,"multimode")
            for spacing in cfg["grid"]["delta_tildes"]
                path=multimode_file(root,manifest,run,spacing)
                isfile(path) || error("Missing multimode export: $path")
                JLD2.jldopen(path,"r") do f
                    f["seeds"]==seeds || error("Incomplete seed set: $path")
                    for key in ("sigmas","L_eigvals","entropy_t","populations_t")
                        haskey(f,key) && length(f[key])==length(seeds) || error("Missing per-seed $key: $path")
                    end
                    T=length(f["times"])
                    all(v->length(v)==K,f["sigmas"]) && all(v->length(v)==K,f["L_eigvals"]) || error("Spectrum dimension mismatch: $path")
                    all(v->size(v)==(N,T),f["populations_t"]) && all(v->length(v)==T,f["entropy_t"]) || error("Dynamics dimension mismatch: $path")
                end
            end
        else
            grid=cfg["grid"]
            for config in grid["configs"], eta in grid["etas"], gamma in grid["gammas"], seed in seeds
                filename="$(config)__eta=$(Float64(eta))__gamma=$(Float64(gamma))__tol=$(Float64(grid["baseline_svd_tol"]))__seed=$seed.jld2"
                path=joinpath(spectra_path(root,manifest,run),filename)
                isfile(path) || error("Missing seed export: $path")
                JLD2.jldopen(path,"r") do f
                    for key in ("L_eigvals","L_svd_S_centered","times","entropy_t","populations_t")
                        haskey(f,key) || error("Missing $key: $path")
                    end
                    length(f["L_eigvals"])==length(f["L_svd_S_centered"])==K || error("Spectrum dimension mismatch: $path")
                    T=length(f["times"])
                    size(f["populations_t"])==(N,T) && length(f["entropy_t"])==T || error("Dynamics dimension mismatch: $path")
                end
            end
        end
    end
    references=String[]
    "1" in figures && mode!="plot" && push!(references,"data/ai_dagger/gaussian_ai_dagger_dsff_reference.jld2")
    "3" in figures && push!(references,"data/syk4_reference/m300/calibration/figure3_syk4_dtilde0p01_m300_n10_q4_r16.jld2")
    for relative in references
        isfile(joinpath(root,relative)) || error("Missing reference/calibration: $relative")
    end
    if "1" in figures && mode=="plot"
        parameters=DatasetPaths.read_manifest(joinpath(root,"figures/figure_1_fixed_beta_dsff__manifest.toml");root)
        ComplexDSFFPlotData.validate_plotdata_binding(joinpath(root,"figures/figure_1_fixed_beta_dsff__plotdata.jld2"),parameters)
        for N in (6,8)
            inset=DatasetPaths.read_manifest(joinpath(root,"figures/source_n$N.toml");root)
            all(isfile,inset["cache_paths"]) || error("Missing Figure 1 inset cache for N=$N")
        end
    end
    if "csr" in figures && mode=="plot"
        cached=CSRStripPlotData.read_csr_strip_plotdata(joinpath(root,"figures/spectral_statistics__csr_strip__physical_eta_L3b__plotdata.jld2"))
        isempty(cached.pending_tags) || error("Prepared CSR cache has incomplete panels")
        CSRStripPlotData.validate_cache_identity(cached.cache_identity,Dict("n_orb"=>10,"filling"=>4,
            "n_seeds"=>manifest["n_seeds"],"gamma"=>1.0,"spectrum_tol"=>1e-10,
            "physical_dir"=>spectra_path(root,manifest,"physical_n10_q4"),
            "control_dir"=>spectra_path(root,manifest,"syk4_control"),
            "multimode_cache_dir"=>multimode_path(root,manifest,"multimode_n10_q4")))
    end
    println("Validated selected ensembles, seed sets, spectra, dynamics and plot inputs")
end

function check_figure_metadata(root,figures,mode)
    required=String[]
    "1" in figures && append!(required,["figures/figure_1_source__manifest.toml",
        "figures/source_n6.toml","figures/source_n8.toml"])
    "1" in figures && mode!="plot" && push!(required,"data/ai_dagger/gaussian_ai_dagger_dsff_reference.jld2")
    "3" in figures && push!(required,"data/syk4_reference/m300/calibration/figure3_syk4_dtilde0p01_m300_n10_q4_r16.jld2")
    "raw" in figures && !("1" in figures) && push!(required,"figures/figure_1__manifest.toml")
    "raw" in figures && !("3" in figures) && push!(required,"figures/figure_3__manifest.toml")
    for relative in required
        isfile(joinpath(root,relative)) || error("Missing figure input: $relative")
    end
end

function source_preflight(root,manifest,figures)
    check_figure_metadata(root,figures,"postprocess")
    for run in needed_runs(figures)
        cfg=SpectralPostprocessing.SpectralArtifacts.load_pipeline_config(config_path(root,run))
        base=joinpath(root,"runs",run)
        for stage in ("eigen","svd")
            isfile(joinpath(base,stage,"manifest.jld2")) || error("--mode postprocess requires downloaded eigen/SVD files; missing $base/$stage/manifest.jld2. Use --mode statistics for bundled light data.")
        end
        selection=joinpath(base,"recovery/selection.toml")
        SpectralPostprocessing.checked_sources(cfg,joinpath(base,"eigen"),joinpath(base,"svd"),["sigma-sff","dynamics"];
            replacements=isfile(selection) ? selection : nothing)
    end
    println("Validated downloaded decomposition identities and recovery source links")
end

function postprocess(root,manifest,work,figures)
    updated=deepcopy(manifest)
    for run in needed_runs(figures)
        cfg=SpectralPostprocessing.SpectralArtifacts.load_pipeline_config(config_path(root,run))
        base=joinpath(root,"runs",run); selection=joinpath(base,"recovery/selection.toml")
        # The smaller insets can have too few eigenvalues for the production DSFF fit.
        tasks=run in ("physical_n6","multimode_n6_f3") ? ["sigma-sff","csr","dynamics"] : ["sigma-sff","dsff","csr","dynamics"]
        result=SpectralPostprocessing.run_postprocessing(cfg,joinpath(work,"analysis",run);
            eigen_dir=joinpath(base,"eigen"),svd_dir=joinpath(base,"svd"),tasks,
            replacements=isfile(selection) ? selection : nothing)
        result.failed==0 && !isempty(result.compatibility) || error("Postprocessing failed for $run")
        updated["selected_analysis"][run]=relpath(result.directory,root)
    end
    updated
end

function render(root,manifest,opts,figures)
    out=opts["output-dir"]; work=opts["work-dir"]; mkpath(out); mkpath(work)
    R=Int(manifest["n_seeds"]); seeds="1:$R"
    phys=spectra_path(root,manifest,"physical_n10_q4"); ctrl=spectra_path(root,manifest,"syk4_control")
    cfg=config_path(root,"physical_n10_q4"); ctrlcfg=config_path(root,"syk4_control")
    # Run the scientific scripts in separate processes to keep their globals isolated.
    function execute(script,args)
        println("Running ",basename(script)); flush(stdout)
        command=Cmd(vcat(collect(Base.julia_cmd()),["--project=$(joinpath(REPO,"src/environment"))",script],String.(args)))
        run(Cmd(command;dir=root))
    end
    sourcepath=joinpath(out,"figure_1_source__manifest.toml")
    if "1" in figures
        source=DatasetPaths.read_manifest(joinpath(root,"figures/figure_1_source__manifest.toml");root)
        source["config"]=cfg; source["spectra_dir"]=phys
        source["l2_overlay"]["config"]=ctrlcfg; source["l2_overlay"]["spectra_dir"]=ctrl
        DatasetPaths.write_manifest(sourcepath,source;root)
    end
    for (N,runname) in ("1" in figures ? ((6,"physical_n6"),(8,"physical_n8")) : ())
        inset=DatasetPaths.read_manifest(joinpath(root,"figures/source_n$N.toml");root)
        inset["config"]=config_path(root,runname); inset["spectra_dir"]=spectra_path(root,manifest,runname)
        if opts["mode"]!="plot"
            numerics=TOML.parsefile(inset["config"])["numerics"]
            q=get(numerics,"filling",div(N,2))
            dir=joinpath(work,"sigma_sff/n$(N)_f$q")
            inset["cache_dir"]=dir
            inset["cache_paths"]=[joinpath(dir,"production_centered__norb=$(N)__f=$(q)__gamma=1__eta=2.jld2")]
        end
        DatasetPaths.write_manifest(joinpath(out,"source_n$N.toml"),inset;root)
    end
    if opts["mode"]!="plot" && "1" in figures
        execute(joinpath(FIG,"figure1b_complex_dsff_unfolding.jl"),["--manifest",sourcepath,
            "--n-seeds",string(R),"--ai-dagger-reference",joinpath(root,"data/ai_dagger/gaussian_ai_dagger_dsff_reference.jld2"),
            "--cache-dir",joinpath(work,"complex_dsff"),"--output-dir",out])
    end
    dsffdir=opts["mode"]=="plot" ? joinpath(root,"figures") : out
    if opts["mode"]=="plot" && "csr" in figures
        filename="spectral_statistics__csr_strip__physical_eta_L3b__plotdata.jld2"
        destination=joinpath(out,filename)
        isfile(destination) || cp(joinpath(root,"figures",filename),destination)
    end
    common=["--config",cfg,"--spectra-dir",phys,"--seeds",seeds,
        "--l2-config",ctrlcfg,"--l2-spectra-dir",ctrl,"--l2-seeds",seeds,"--output-dir",out]
    if "1" in figures
        execute(joinpath(FIG,"figure1_sigma_sff_dsff.jl"),vcat(common,[
            "--cache-dir",opts["mode"]=="plot" ? joinpath(root,"derived/sigma_sff/n10_f4") : joinpath(work,"sigma_sff/n10_f4"),
            "--complex-dsff-plotdata",joinpath(dsffdir,"figure_1_fixed_beta_dsff__plotdata.jld2"),
            "--complex-dsff-manifest",joinpath(dsffdir,"figure_1_fixed_beta_dsff__manifest.toml"),
            "--inset-manifest","6:$(joinpath(out,"source_n6.toml"))",
            "--inset-manifest","8:$(joinpath(out,"source_n8.toml"))"]))
    end
    "2" in figures && execute(joinpath(FIG,"figure2_dynamics.jl"),common)
    mm=multimode_path(root,manifest,"multimode_n10_q4")
    if "3" in figures
        refcfg=config_path(root,"fig3_m300"); refspec=spectra_path(root,manifest,"fig3_m300")
        validation=joinpath(work,"figure3_syk4_reference_validation.toml")
        calibration=joinpath(root,"data/syk4_reference/m300/calibration/figure3_syk4_dtilde0p01_m300_n10_q4_r16.jld2")
        execute(joinpath(FIG,"validate_figure3_syk4_reference.jl"),["--config",refcfg,
            "--spectra-dir",refspec,"--calibration",calibration,"--seed-range",seeds,"--output",validation])
        execute(joinpath(FIG,"figure3_multimode_unfolded.jl"),["--n-orb","10","--filling","4",
            "--n-seeds",string(R),"--delta-tildes","0.01,0.1,1.0,10.0","--delta-cd","1","--cache-dir",mm,
            "--inset-comparison-cache","6:$(multimode_file(root,manifest,"multimode_n6_f3",0.01))",
            "--inset-comparison-cache","8:$(multimode_file(root,manifest,"multimode_n8_f4",0.01))",
            "--syk4-reference-config",refcfg,"--syk4-reference-spectra-dir",refspec,
            "--syk4-reference-validation",validation,"--syk4-reference-eta","2","--syk4-reference-seeds",seeds,"--output-dir",out])
    end
    "csr" in figures && execute(joinpath(@__DIR__,"analysis/csr_synthetic_ladder/plot_csr_strip.jl"),vcat([
        "--n-orb","10","--filling","4","--n-seeds",string(R),"--physical-dir",phys,
        "--control-dir",ctrl,"--multimode-cache-dir",mm,"--output-dir",out],opts["mode"]=="plot" ? String[] : ["--rebuild-plotdata"]))
    if "raw" in figures
        fig1="1" in figures ? joinpath(out,"figure_1__manifest.toml") : joinpath(root,"figures/figure_1__manifest.toml")
        fig3="3" in figures ? joinpath(out,"figure_3__manifest.toml") : joinpath(root,"figures/figure_3__manifest.toml")
        # Override sources when only the raw supplement was selected after fresh postprocessing.
        if opts["mode"]=="postprocess"
            a=DatasetPaths.read_manifest(fig1;root); b=DatasetPaths.read_manifest(fig3;root)
            a["spectra_dir"]=phys; a["l2_overlay"]["spectra_dir"]=ctrl
            b["cache_paths"]=[multimode_file(root,manifest,"multimode_n10_q4",d) for d in (0.01,0.1,1.0,10.0)]
            b["syk4_diss_reference"]["spectra_dir"]=spectra_path(root,manifest,"fig3_m300")
            fig1=joinpath(work,"raw_source_figure1.toml"); fig3=joinpath(work,"raw_source_figure3.toml")
            DatasetPaths.write_manifest(fig1,a;root); DatasetPaths.write_manifest(fig3,b;root)
        end
        execute(joinpath(FIG,"supplement_raw_sigma_sff_dsff.jl"),["--figure1-manifest",fig1,"--figure3-manifest",fig3,"--output-dir",out])
    end
    println("Figures written to $out")
end

function main(argv=ARGS)
    cli=parse_cli(argv); cli.help && (print(HELP);return 0)
    opts=cli.opts; root=opts["data-root"]
    manifest=TOML.parsefile(opts["manifest"])
    withenv("SYK_DATA_ROOT"=>root,"SYK_HEADER_ROOT"=>opts["header-root"]) do
        @eval using JLD2
        @eval include(joinpath(@__DIR__,"..","ComplexDSFFPlotData.jl"))
        @eval include(joinpath(@__DIR__,"..","CSRStripPlotData.jl"))
        Base.invokelatest() do
            if opts["mode"]=="postprocess"
                @eval include(joinpath(@__DIR__,"..","SpectralPostprocessing.jl"))
                Base.invokelatest(source_preflight,root,manifest,cli.figures)
                cli.check && return 0
                manifest=Base.invokelatest(postprocess,root,manifest,opts["work-dir"],cli.figures)
            end
            preflight(root,manifest;figures=cli.figures,mode=opts["mode"])
            cli.check && return 0
            render(root,manifest,opts,cli.figures)
            0
        end
    end
end
end

if abspath(PROGRAM_FILE)==@__FILE__
    exit(ReproduceFigures.main())
end
