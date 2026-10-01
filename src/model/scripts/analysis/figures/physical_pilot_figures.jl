#!/usr/bin/env julia
# Physical-only Figure 1 and raw-form-factor supplement for a staged ensemble.
# Controls and finite-size ensembles are deliberately absent from this pilot.
using JLD2, TOML, CairoMakie, Statistics, Random
const MODEL = normpath(joinpath(@__DIR__,"../../.."))
include(joinpath(MODEL,"SpectralPostprocessing.jl"))
include(joinpath(MODEL,"RawConnectedFormFactors.jl"))
include(joinpath(MODEL,"ComplexDSFFAIDaggerReference.jl"))
const PP = SpectralPostprocessing
const CU = PP.ComplexDSFFUnfolding
const RAW = RawConnectedFormFactors
const REF = ComplexDSFFAIDaggerReference

function draw_curve!(ax,x,boot,color,label)
    positive(y) = [isfinite(v) && v > 0 ? v : NaN for v in y]
    band!(ax,x,positive(boot.lower),positive(boot.upper);color=(color,0.15))
    lines!(ax,x,positive(boot.curve);color,label,linewidth=1.6)
end

function main(args=ARGS)
    length(args) == 4 || error("Usage: physical_pilot_figures.jl CONFIG ANALYSIS_DIR AI_REFERENCE OUTPUT_DIR")
    config,analysis,reference_path,out = abspath.(args)
    cfg = PP.SpectralArtifacts.load_pipeline_config(config)
    report = TOML.parsefile(joinpath(analysis,"manifest.toml"))
    report["failed"] == 0 || error("Analysis has failures")
    seeds = Int.(report["selected_seeds"])
    paths = String.(report["compatibility"])
    length(paths) == length(cfg["grid"]["etas"]) * length(seeds) || error("Incomplete figure inputs")
    mkpath(out)
    reference = REF.load_ai_dagger_reference(reference_path)
    x = 10.0 .^ range(-2.5,1;length=400)
    raw_times = 10.0 .^ range(-3,6;length=400)
    etas = Float64.(cfg["grid"]["etas"])
    colors = cgrad(:viridis,length(etas);categorical=true)
    n,q = cfg["numerics"]["n_orb"],cfg["numerics"]["filling"]
    title = "Single mode spontaneous emission: N=$n, Q=$q, $(length(seeds)) realizations"
    fig = Figure(size=(1050,440))
    Label(fig[0,1:2],title)
    ax1 = Axis(fig[1,1];xlabel="t / tHeis (singular values)",ylabel="Connected σ-SFF",xscale=log10,yscale=log10)
    ax2 = Axis(fig[1,2];xlabel="t / tHeis (AI† calibration)",ylabel="Connected DSFF, β=1/2, θ=π/4",xscale=log10,yscale=log10)
    rawfig = Figure(size=(1050,440))
    Label(rawfig[0,1:2],title * " — raw form factors")
    rawax1 = Axis(rawfig[1,1];xlabel="t",ylabel="Raw connected σ-SFF / plateau",xscale=log10,yscale=log10)
    rawax2 = Axis(rawfig[1,2];xlabel="t",ylabel="Raw connected DSFF / plateau",xscale=log10,yscale=log10)
    csrfig = Figure(size=(1400,390))
    Label(csrfig[0,1:5],title * " — physical CSR panels")
    production = PP.read_manifest(joinpath(dirname(config),"eigen","manifest.jld2"))
    csr_densities = Matrix{Float64}[]
    for eta in etas
        points = sort(filter(p -> p["eta"] == eta && p["seed"] in seeds,production["points"]);by=p->p["seed"])
        group_id = PP.identity(Dict("points"=>[p["point_id"] for p in points]))
        result = JLD2.load(joinpath(analysis,"csr",group_id*".jld2"),"result")
        ratios = reduce(vcat,result["ratios_by_seed"])
        isempty(ratios) && error("No CSR ratios for eta=$eta")
        bins = 160
        density = zeros(bins,bins)
        for z in ratios
            abs(real(z)) <= 1 && abs(imag(z)) <= 1 || continue
            i = clamp(floor(Int,(real(z)+1)*bins/2)+1,1,bins)
            j = clamp(floor(Int,(imag(z)+1)*bins/2)+1,1,bins)
            density[i,j] += 1
        end
        density ./= length(ratios)*(2/bins)^2
        push!(csr_densities,density)
    end
    cmax = max(maximum(quantile(vec(d),0.995) for d in csr_densities),eps())
    centers = collect(range(-1+1/160,1-1/160;length=160))
    for (index,eta) in enumerate(etas)
        ax = Axis(csrfig[1,index];title="η = $eta",xlabel="Re ζ",ylabel=index==1 ? "Im ζ" : "",aspect=DataAspect())
        hm = heatmap!(ax,centers,centers,csr_densities[index];colormap=:inferno,colorrange=(0,cmax))
        angles = range(0,2pi;length=400)
        lines!(ax,cos.(angles),sin.(angles);color=:white,linestyle=:dash)
        index == length(etas) && Colorbar(csrfig[1,index+1],hm;label="P(ζ)")
    end
    plotdata = Dict{String,Any}()
    numerical = Dict{String,Any}()
    for (index,eta) in enumerate(etas)
        files = [only(filter(p -> occursin("__eta=$(eta)__",basename(p)) && endswith(p,"__seed=$s.jld2"),paths)) for s in seeds]
        data = [JLD2.load(p) for p in files]
        K = binomial(n,q)^2
        all(d -> length(d["L_eigvals"]) == K && length(d["L_svd_S_centered"]) == K,data) || error("Wrong matrix dimension")
        sigmas = [d["L_svd_S_centered"] for d in data]
        spectra = [filter(z -> abs(z)>1e-8,d["L_eigvals"]) for d in data]
        options = get(get(cfg,"analysis",Dict()),"dsff",Dict())
        fit = CU.fit_fixed_power_policy(spectra,0.5;
            n_bins=get(options,"density_bins",256),smooth_sigma=2.0,
            minimum_effective_count=get(options,"minimum_effective_count",500.0))
        mapped = CU.mapped_spectra(spectra,fit)
        _,weights = CU.partition_traces(mapped,fit.filter,[1.0],pi/4)
        plateaus = [sum(abs2,w) for w in weights]
        density = REF.gaussian_filter_effective_density(mean(plateaus),fit.filter.alpha_x,fit.filter.alpha_y)
        times = x .* reference.chi .* sqrt(density)
        Z,_ = CU.partition_traces(mapped,fit.filter,times,pi/4)
        dsff = CU.bootstrap_connected_dsff(Z,plateaus;n_boot=500,rng=MersenneTwister(20260804+index))
        sigma = PP.bootstrap_kunf_ham(sigmas,2pi .* x;
            analysis_window=(0.05,0.95),epsilon_zero=1e-8,n_bins=40,degree=5,
            n_boot=500,rng=MersenneTwister(20260804+index))
        rawsigma = RAW.analyze_sigma_sff(sigmas,raw_times;n_boot=500,rng=MersenneTwister(20260804+2index-1))
        rawdsff = RAW.analyze_dsff([d["L_eigvals"] for d in data],raw_times;
            theta=pi/4,n_boot=500,rng=MersenneTwister(20260804+2index))
        label = "η = $eta"
        draw_curve!(ax1,x,sigma,colors[index],label)
        draw_curve!(ax2,x,dsff,colors[index],label)
        draw_curve!(rawax1,raw_times,rawsigma,colors[index],label)
        draw_curve!(rawax2,raw_times,rawdsff,colors[index],label)
        plotdata[string(eta)] = Dict("sigma"=>sigma,"dsff"=>dsff,"raw_sigma"=>rawsigma,
            "raw_dsff"=>rawdsff,"dsff_times"=>times,"effective_density"=>density,"paths"=>files)
        numerical[string(eta)] = Dict(
            "steady_modes"=>[count(z->abs(z)<=1e-8,d["L_eigvals"]) for d in data],
            "max_real_eigenvalue"=>[maximum(real,d["L_eigvals"]) for d in data],
            "trace_residual_max"=>maximum(d["dynamics_trace_residual_max"] for d in data),
            "particle_residual_max"=>maximum(d["dynamics_particle_residual_max"] for d in data))
    end
    lines!(ax2,reference.x,RAW.positive_or_nan(reference.curve);color=:black,linestyle=:dash,label="Gaussian AI†")
    for ax in (ax1,ax2,rawax1,rawax2)
        hlines!(ax,[1.0];color=:gray,linestyle=:dot)
        axislegend(ax;position=:rb,labelsize=11)
    end
    xlims!(ax1,first(x),last(x)); xlims!(ax2,first(x),last(x))
    outputs = String[]
    for (figure,name) in ((fig,"figure_1_physical"),(rawfig,"supplement_raw_sigma_sff_dsff_physical"),
                         (csrfig,"spectral_statistics__csr_strip__physical")), ext in ("pdf","png")
        path = joinpath(out,"$name.$ext")
        save(path,figure); push!(outputs,path)
    end
    JLD2.jldsave(joinpath(out,"physical_plotdata.jld2");plotdata,x,raw_times,seeds,n_orb=n,filling=q,csr_densities,csr_etas=etas)
    manifest = Dict("n_orb"=>n,"filling"=>q,"n_seeds"=>length(seeds),"seeds"=>seeds,
        "replacements"=>get(report,"replacements",[]),
        "production_seed_count"=>cfg["grid"]["seed_count"],"scope"=>"physical single-mode only",
        "analysis_manifest"=>joinpath(analysis,"manifest.toml"),"ai_reference"=>reference_path,
        "ai_reference_token"=>reference.token,"outputs"=>outputs,"numerical_checks"=>numerical,
        "csr_scope"=>"physical only","csr_etas"=>etas,"csr_status"=>"complete")
    open(joinpath(out,"physical_figures_manifest.toml"),"w") do io
        TOML.print(io,manifest;sorted=true)
    end
    println("Physical figures: $out")
end
abspath(PROGRAM_FILE) == (@__FILE__) && main()
