#!/usr/bin/env bash
# Render the canonical paper figures (Figures 1-3, CSR strip, raw sigma-SFF/DSFF
# supplement) from staged analysis exports, following REPRODUCING.md. Every
# input is explicit; nothing falls back to the checked-in paper manifests.
#
# Required environment (absolute paths):
#   REPO_ROOT JULIA OUT WORK N_SEEDS FILLING
#   PHYSICAL_CONFIG PHYSICAL_SPECTRA        physical N=10, four etas
#   CONTROL_CONFIG  CONTROL_SPECTRA         SYK4 control (L3b), eta=2
#   MULTIMODE_DIR   MM_INSET_6 MM_INSET_8   multimode exports; N=6/N=8 dtilde=0.01 caches
#   INSET6_CONFIG INSET6_SPECTRA INSET8_CONFIG INSET8_SPECTRA   physical finite-size inset
#   FIG3_REF_CONFIG FIG3_REF_SPECTRA FIG3_REF_CALIBRATION        m10 SYK4 reference
#   AI_REFERENCE                            Gaussian AI-dagger reference (built if missing)
# Optional:
#   FIG1B_EXTRA_ARGS   extra figure1b arguments; "--smoke" for tiny test ensembles,
#                      whose DSFF fit cannot meet the production effective-count
#                      threshold. Never set for production figures.
set -euo pipefail
for v in REPO_ROOT JULIA OUT WORK N_SEEDS FILLING PHYSICAL_CONFIG PHYSICAL_SPECTRA \
         CONTROL_CONFIG CONTROL_SPECTRA MULTIMODE_DIR MM_INSET_6 MM_INSET_8 \
         INSET6_CONFIG INSET6_SPECTRA INSET8_CONFIG INSET8_SPECTRA \
         FIG3_REF_CONFIG FIG3_REF_SPECTRA FIG3_REF_CALIBRATION AI_REFERENCE; do
    [[ -n ${!v:-} ]] || { echo "missing $v" >&2; exit 2; }
done
cd "$REPO_ROOT"
F=src/model/scripts/analysis/figures
jl=("$JULIA" --project="$REPO_ROOT/src/environment")
seeds="1:$N_SEEDS"
mkdir -p "$OUT" "$WORK"
step() { echo; echo "$(date -Is) ===== $*"; }

step "AI-dagger reference"
[[ -f $AI_REFERENCE ]] || "${jl[@]}" "$F/generate_ai_dagger_dsff_reference.jl" \
    --cache-dir "$WORK/ai_dagger_cache" --output "$AI_REFERENCE"

step "source manifests (Figure 1 inputs, finite-size insets)"
"${jl[@]}" -e '
    using TOML
    out, work, n_seeds, filling = ARGS[1], ARGS[2], parse(Int,ARGS[3]), parse(Int,ARGS[4])
    d = binomial(10, filling)
    control = TOML.parsefile(ARGS[7])
    open(joinpath(out,"figure_1_source__manifest.toml"),"w") do io
        TOML.print(io, Dict(
            "config"=>ARGS[5], "spectra_dir"=>ARGS[6], "n_orb"=>10, "filling"=>filling,
            "hilbert_dim"=>d, "K_liouville"=>d^2, "etas"=>[0.1,0.4,1.0,2.0],
            "delta_cd_over_2pi_mhz"=>1.0, "n_seeds"=>n_seeds, "seed_range"=>[1,n_seeds],
            "steady_tol"=>1e-8, "svd_tol"=>1e-10,
            "l2_overlay"=>Dict("config"=>ARGS[7], "config_name"=>only(control["grid"]["configs"]),
                "spectra_dir"=>ARGS[8], "eta"=>2.0, "gamma"=>1.0, "svd_tol"=>1e-10,
                "n_seeds"=>n_seeds, "seed_range"=>[1,n_seeds])); sorted=true)
    end
    for (n, config, spectra) in ((6, ARGS[9], ARGS[10]), (8, ARGS[11], ARGS[12]))
        num = TOML.parsefile(config)["numerics"]
        q = get(num, "filling", div(n, 2))
        dir = joinpath(work, "sigma_sff", "n$(n)_f$(q)")
        open(joinpath(out,"source_n$(n).toml"),"w") do io
            TOML.print(io, Dict("config"=>config, "spectra_dir"=>spectra, "etas"=>[2.0],
                "n_seeds"=>Int(TOML.parsefile(config)["grid"]["seed_count"]), "svd_tol"=>1e-10,
                "cache_dir"=>dir,
                "cache_paths"=>[joinpath(dir,"production_centered__norb=$(n)__f=$(q)__gamma=1__eta=2.jld2")]); sorted=true)
        end
    end
' "$OUT" "$WORK" "$N_SEEDS" "$FILLING" "$PHYSICAL_CONFIG" "$PHYSICAL_SPECTRA" \
  "$CONTROL_CONFIG" "$CONTROL_SPECTRA" "$INSET6_CONFIG" "$INSET6_SPECTRA" "$INSET8_CONFIG" "$INSET8_SPECTRA"

step "Figure 1b: fixed-beta complex DSFF plot data"
"${jl[@]}" "$F/figure1b_complex_dsff_unfolding.jl" \
    --manifest "$OUT/figure_1_source__manifest.toml" --ai-dagger-reference "$AI_REFERENCE" \
    --n-seeds "$N_SEEDS" --cache-dir "$WORK/complex_dsff" --output-dir "$OUT" ${FIG1B_EXTRA_ARGS:-}

step "Figure 1"
"${jl[@]}" "$F/figure1_sigma_sff_dsff.jl" \
    --config "$PHYSICAL_CONFIG" --spectra-dir "$PHYSICAL_SPECTRA" --seeds "$seeds" \
    --cache-dir "$WORK/sigma_sff/n10_f$FILLING" \
    --l2-config "$CONTROL_CONFIG" --l2-spectra-dir "$CONTROL_SPECTRA" --l2-seeds "$seeds" \
    --complex-dsff-plotdata "$OUT/figure_1_fixed_beta_dsff__plotdata.jld2" \
    --complex-dsff-manifest "$OUT/figure_1_fixed_beta_dsff__manifest.toml" \
    --inset-manifest "6:$OUT/source_n6.toml" --inset-manifest "8:$OUT/source_n8.toml" \
    --output-dir "$OUT"

step "Figure 2"
"${jl[@]}" "$F/figure2_dynamics.jl" \
    --config "$PHYSICAL_CONFIG" --spectra-dir "$PHYSICAL_SPECTRA" --seeds "$seeds" \
    --l2-config "$CONTROL_CONFIG" --l2-spectra-dir "$CONTROL_SPECTRA" --l2-seeds "$seeds" \
    --output-dir "$OUT"

step "Figure 3 SYK4 reference validation"
"${jl[@]}" "$F/validate_figure3_syk4_reference.jl" \
    --config "$FIG3_REF_CONFIG" --spectra-dir "$FIG3_REF_SPECTRA" \
    --calibration "$FIG3_REF_CALIBRATION" --seed-range "$seeds" \
    --output "$WORK/figure3_syk4_reference_validation.toml"

step "Figure 3"
"${jl[@]}" "$F/figure3_multimode_unfolded.jl" \
    --n-orb 10 --filling "$FILLING" --n-seeds "$N_SEEDS" --delta-cd 1 \
    --cache-dir "$MULTIMODE_DIR" \
    --inset-comparison-cache "6:$MM_INSET_6" --inset-comparison-cache "8:$MM_INSET_8" \
    --syk4-reference-config "$FIG3_REF_CONFIG" --syk4-reference-spectra-dir "$FIG3_REF_SPECTRA" \
    --syk4-reference-validation "$WORK/figure3_syk4_reference_validation.toml" \
    --syk4-reference-eta 2 --syk4-reference-seeds "$seeds" --output-dir "$OUT"

step "CSR strip"
"${jl[@]}" src/model/scripts/analysis/csr_synthetic_ladder/plot_csr_strip.jl \
    --n-orb 10 --filling "$FILLING" --n-seeds "$N_SEEDS" \
    --physical-dir "$PHYSICAL_SPECTRA" --control-dir "$CONTROL_SPECTRA" \
    --multimode-cache-dir "$MULTIMODE_DIR" --output-dir "$OUT"

step "Raw sigma-SFF/DSFF supplement"
"${jl[@]}" "$F/supplement_raw_sigma_sff_dsff.jl" \
    --figure1-manifest "$OUT/figure_1__manifest.toml" \
    --figure3-manifest "$OUT/figure_3__manifest.toml" --output-dir "$OUT"

step "done"; ls -la "$OUT"
