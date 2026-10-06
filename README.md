# dissipative-syk-cavity-qed

[![DOI](https://zenodo.org/badge/1400128569.svg)](https://zenodo.org/badge/latestdoi/1400128569)
[![arXiv](https://img.shields.io/badge/arXiv-2608.23557-b31b1b.svg?style=flat)](https://arxiv.org/abs/2608.23557)

Code and figure data for “Dissipation-induced Sachdev-Ye-Kitaev physics in many-body cavity quantum electrodynamics” ([`arXiv:2608.23557`](https://arxiv.org/abs/2608.23557)).

## Reproducing the figures

This repository contains the Julia code for the numerical figures in [the paper](https://arxiv.org/abs/2608.23557) *Dissipation-induced Sachdev-Ye-Kitaev physics in many-body cavity quantum electrodynamics*. The calculations generate ensembles of Liouvillians with different random seeds, compute their spectra, and obtain spectral statistics ($\rm\sigma SFF$, $\rm DSFF$, $\rm CSR$) and dynamical quantities (entropy, occupations). The light dataset in [`reproduction_data/`](reproduction_data/) allows Figures 1–3, the complex-spacing-ratio strip and the raw-form-factor supplement to be replotted without downloading the full dataset or repeating eigendecompositions. The [figure workflow](https://github.com/ltpn/dissipative-syk-cavity-qed/actions/workflows/figures.yml) regenerates the figures in the `figures/` directory on every push and uploads PDFs, PNGs and manifests TOMLs as the `figures` artifact. Published [releases](https://github.com/ltpn/dissipative-syk-cavity-qed/releases) also include `figures.tar.gz`.

## Setup

[Git LFS](https://git-lfs.com/) is required to download the data.

The code can be run with Julia 1.12.6. Ghostscript (`gs`) is used for figure export. The commands below should be run from the repository root. Install the Julia environment with:

```sh
julia --project=src/environment -e 'using Pkg; Pkg.instantiate()'
```

The numerical data is stored in `data/`, which can be a symlink to a larger disk. The following shell shortcuts are used below:

```sh
export STORE="$PWD/data"
jrun() { julia --project=src/environment "$@"; }
pipeline() { jrun src/model/scripts/pipeline.jl "$@"; }
FIG=src/model/scripts/analysis/figures
```

With N orbitals and Q particles, the Hilbert-space dimension is d = binomial(N,Q), and the Liouvillian is d² × d². At N = 10, Q = 4, one complex matrix occupies about 29 GiB; eigendecomposition needs additional memory and saves eigenvectors as well. Larger systems generally require HPC cluster resources.

## Three reproduction levels

After installing the Julia environment and Ghostscript, choose a starting point. The figure data bundles with the repository occupies 451.7 MiB; the dataset on EPFL ACOUA, containing the full diagonalization data, is about 4.54 TiB.

| Starting point | Steps |
| --- | --- |
| Configurations only | Run `generate`, `eigen`, `svd`, `postprocess` as described in the following, then the figure commands below. |
| Downloaded full dataset | Run the command below with `--mode postprocess --data-root /path/to/downloaded_dataset`. This reuses saved decompositions, recomputes dynamics/statistics and renders the five figures. Enough RAM is needed to load the stored eigenvectors |
| Bundled light dataset | Run the command below directly. Add `--mode statistics` to recompute the final DSFF, σ-SFF, CSR and raw-form-factor statistics from the saved spectra. Occupations and entropy as a function of time are already saved. |

```sh
jrun src/model/scripts/reproduce_figures.jl --output-dir reproduced_figures
```

The default reuses the already present DSFF and CSR data and computes the remaining plot statistics. It uses the bundled dataset and writes PDFs, PNGs and manifests to `reproduced_figures/`. `--figures 1,2,3,csr,raw` selects outputs; `--check` validates inputs without plotting. `--data-root` overrides `SYK_DATA_ROOT`, which overrides the bundled default. `SYK_HEADER_ROOT` selects the generation-header overlay used with downloaded decompositions; `--header-root` overrides it. `--work-dir` selects the cache directory, and `--manifest` accepts another dataset map. All paths inside dataset metadata are relative to the dataset root; a manifest's `path_root` locates that root relative to the manifest when no root override is supplied. No cluster paths or symlinks are needed.

```sh
# Recompute the final statistics from the light data, then plot.
JULIA_NUM_THREADS=auto jrun src/model/scripts/reproduce_figures.jl --mode statistics \
  --output-dir reproduced_figures

# Start from downloaded eigen/SVD files, then postprocess and plot.
jrun src/model/scripts/reproduce_figures.jl --mode postprocess \
  --data-root /path/to/downloaded_dataset --output-dir reproduced_figures
```

For the full dataset route [`reproduction_data/dataset.toml`](reproduction_data/dataset.toml) maps each ensemble to its selected analysis directory:

| Light data | Figure role |
| --- | --- |
| `physical_n10_q4` spectra | Figure 1  σ-SFF/DSFF; Figure 2 entropy/occupations; single-mode CSR and raw form factors. |
| `syk4_control` spectra | Figures 1–2 SYK4 comparisons; control CSR and raw form factors. |
| `physical_n6`, `physical_n8` spectra | Figure 1 finite-size inset. |
| `multimode_n10_q4` merged caches | Figure 3 σ-SFF/occupations; multimode CSR and raw form factors. |
| `multimode_n6_f3`, `multimode_n8_f4` merged caches | Figure 3 finite-size inset. |
| `fig3_m300` spectra | Figure 3 dissipative SYK4 comparison and its raw form factors. |
| `figures/*plotdata.jld2`, `derived/sigma_sff/`, `data/` | Prepared plot data, singular-value caches, random-matrix references and calibrations. |

The bundle also retains selected `sigma-sff/`, `dsff/`, `csr/` and `dynamics/` results. Its [inventory](reproduction_data/inventory.json) records file sizes and SHA-256 checksums. Dense matrices and Liouvillian eigenvectors are excluded, so recomputing density-matrix dynamics requires the full dataset. The integrable-corner and coherent-Gibbs-state survival supplements use the separate workflows described below.

## Choosing parameters

Copy a configuration from [`src/model/configs/`](src/model/configs/) and edit it for the desired calculation.

| Calculation | Starting configuration |
| --- | --- |
| Single-mode model, Figures 1 and 2 | [`physical_n10_q4_r16.toml`](src/model/configs/physical_n10_q4_r16.toml) |
| Figure 1 inset comparisons | [`physical_n6_r16.toml`](src/model/configs/physical_n6_r16.toml), [`physical_n8_r16.toml`](src/model/configs/physical_n8_r16.toml) |
| Single-mode SYK4 comparison | [`csr_ladder_l3b_n10_q4_r16.toml`](src/model/configs/csr_ladder_l3b_n10_q4_r16.toml) |
| Multimode model, Figure 3 | [`multimode_n10_q4_r16.toml`](src/model/configs/multimode_n10_q4_r16.toml) |

`[numerics]` specifies `n_orb` (N), `filling` (Q), the spatial grid and disorder parameters. Omitted `filling` means half filling. In `[grid]`, `seed_count` sets the number of disorder realizations; `etas` sets the single-mode sweep and `delta_tildes` the multimode sweep. `[numerics.time_grid]` sets the dynamics time range and number of samples. Multimode cavity and atomic rates are in `[loss]`; frequency entries ending in `_over_2pi_mhz` are in $\rm{MHz}/2\pi$.

The main figures use (N,Q) = (10,4), with (6,3) and (8,4) for the insets smaller size comparisons. For Figure 3, use the main multimode configuration and [`multimode_n6_f3_r16.toml`](src/model/configs/multimode_n6_f3_r16.toml) and [`multimode_n8_f4_r16.toml`](src/model/configs/multimode_n8_f4_r16.toml) for the insets.

The numerical pipeline accepts other sizes, fillings and parameter grids. However, the code retains some paper-specific choices: Figure 1's DSFF requires N = 10, Δ/(2π) = 1 MHz and `etas = [0.1, 0.4, 1.0, 2.0]`, while allowing Q and the ensemble size to change. Changing an ensemble also requires regenerating the calibration of the corresponding SYK4 reference model and derived plot data.

## Generating the data

For each configuration, choose a separate data directory and run, for example:

```sh
CONFIG=src/model/configs/physical_n10_q4_r16.toml
DATA=data/physical_n10_q4_r16

pipeline generate --config "$CONFIG" --output "$DATA/liouvillians"
pipeline eigen --input "$DATA/liouvillians" --output "$DATA/eigen"
pipeline svd --input "$DATA/liouvillians" --output "$DATA/svd"
pipeline postprocess --config "$CONFIG" \
  --eigen "$DATA/eigen" --svd "$DATA/svd" --output "$DATA/analysis"
```

The `generate` step builds one matrix per parameter value and disorder realization. Eigen and SVD calculations can run independently after generation. Postprocessing computes the spectral form factors, complex spacing ratios and dynamical quantities. When it finishes, it prints the analysis directory and its `manifest.toml`. The figure inputs are in that directory's `spectra/` subdirectory for single-mode models and `multimode/` for multimode models. The paths are needed for the plotting steps below.

Resubmitting a stage reuses any completed data, if possible. Analysis settings can be changed without repeating the matrix decompositions. The optional `[analysis.sigma_sff]`, `[analysis.dsff]` and `[analysis.csr]` TOML tables control unfolding, sampling and averaging, while the dynamics time grid remains in `[numerics.time_grid]` and `--blas-threads N` controls solver threads. On Slurm, [`submit_spectral_pipeline.sh`](src/model/scripts/slurm/submit_spectral_pipeline.sh) submits all four stages with `--config`, `--data-root` and separate resource requests for each stage.

The single-mode reference SYK4 comparison needs a calibration against the physical system data before running the same pipeline with its control configuration:

```sh
jrun src/model/scripts/analysis/csr_synthetic_ladder/calibrate_ladder_stats.jl \
  --config "$CONFIG" --eta 2 --n-seeds 16 \
  --output data/syk4/n10_q4_r16/calibration.jld2
```

Use the desired ensemble size by setting `--n-seeds`, and point `numerics.calibration_jld2` in the control configuration to this output. Physical and control configurations must use the same N and Q. Calibration paths in configuration files are relative to the repository root, while paths provided explicitly as command line arguments are relative to the working directory.

## Figures 1 and 2

Set `PHYSICAL` and `CONTROL` to the `spectra/` directories printed by postprocessing. Figure 1 also needs the two smaller size datasets for the inset: copy [`source_n6.toml`](reproduction_data/figures/source_n6.toml) and [`source_n8.toml`](reproduction_data/figures/source_n8.toml) into your output directory, set `path_root = ".."`, then set `config`, `spectra_dir` and `n_seeds` to their corresponding inputs using repository-relative paths. `cache_dir` and `cache_paths` specify where their derived form-factor data are saved.

Some preparation steps for the DSFF are needed before rendering Figure 1. Copy [`figure_1__manifest.toml`](reproduction_data/figures/figure_1__manifest.toml) to `figure1_source.toml`, set `path_root = "."`, and set its `spectra_dir` and `[l2_overlay].spectra_dir` to `PHYSICAL` and `CONTROL`. If changing Q or the ensemble size, also update `filling`, `hilbert_dim = binomial(10,Q)`, `K_liouville = hilbert_dim²`, `n_seeds` and `seed_range`, including the control's seed entries. With `R` set to that ensemble size, run:

```sh
R=16
jrun "$FIG/generate_ai_dagger_dsff_reference.jl"
jrun "$FIG/figure1b_complex_dsff_unfolding.jl" \
  --manifest figure1_source.toml --n-seeds "$R" \
  --ai-dagger-reference data/ai_dagger/gaussian_ai_dagger_dsff_reference.jld2
jrun "$FIG/figure1_sigma_sff_dsff.jl" \
  --config "$CONFIG" --spectra-dir "$PHYSICAL" \
  --l2-config src/model/configs/csr_ladder_l3b_n10_q4_r16.toml \
  --l2-spectra-dir "$CONTROL" --seeds "1:$R" --l2-seeds "1:$R"
jrun "$FIG/figure2_dynamics.jl" \
  --config "$CONFIG" --spectra-dir "$PHYSICAL" \
  --l2-config src/model/configs/csr_ladder_l3b_n10_q4_r16.toml \
  --l2-spectra-dir "$CONTROL" --seeds "1:$R" --l2-seeds "1:$R"
```

Use your control configuration in `--l2-config` if it differs from the example. The unfolding of the DSFF in Figure 1 uses β = 1/2  and projects along the θ = π/4 complex plane axis, as stated in the paper's supplementary material, and it is calibrated against the Gaussian AI† ensemble.

## Figure 3

Set `MULTIMODE` to the main ensemble's `multimode/` export directory and `INSET6` and `INSET8` to the individual JLD2 files for the finite-size inputs at δ̃ = 0.01. Their full filenames are listed in the analysis manifests.

```sh
jrun "$FIG/figure3_multimode_unfolded.jl" \
  --n-orb 10 --filling 4 --n-seeds 16 \
  --delta-tildes 0.01,0.1,1.0,10.0 --delta-cd 1 \
  --cache-dir "$MULTIMODE" \
  --inset-comparison-cache "6:$INSET6" \
  --inset-comparison-cache "8:$INSET8" --output-dir figures
```

Match `--n-orb`, `--filling`, `--n-seeds`, `--delta-tildes` and `--delta-cd` to the generated data. The inset's spacing can be changed with `--inset-comparison-delta-tilde`.

The dissipative SYK4 comparison uses [`figure3_syk4_dtilde0p01_m300_n10_q4_r16.toml`](src/model/configs/figure3_syk4_dtilde0p01_m300_n10_q4_r16.toml): 300 random channels plus one cavity channel. [`calibrate_figure3_syk4_reference.jl`](src/model/scripts/analysis/figures/calibrate_figure3_syk4_reference.jl) matches its Hamiltonian span and dissipator strength to the multimode ensemble at δ̃ = 0.01. Supply `--physical-cache` for that spacing, `--multimode-config`, `--baseline-calibration`, `--baseline-config` for the single-mode SYK4 ensemble, `--n-random-jumps 300`, `--n-seeds` and `--output`. Set the reference configuration's `calibration_jld2` to this output and generate it with the same four-stage pipeline.

The comparison also needs a companion TOML from the [reference metadata script](src/model/scripts/analysis/figures/validate_figure3_syk4_reference.jl), supplied with `--config`, `--spectra-dir`, `--calibration`, `--seed-range 1:R` and `--output`. Add `--syk4-reference-config`, `--syk4-reference-spectra-dir`, `--syk4-reference-validation` (the companion TOML), `--syk4-reference-eta 2` and `--syk4-reference-seeds 1:R` to the Figure 3 command to include the curve.

## Supplementary figures

The supplementary figures can be produced using the same environment. The scripts named below are in `$FIG` unless a full path is given.

| Figure | Inputs and commands |
| --- | --- |
| Complex spacing ratios | `src/model/scripts/analysis/csr_synthetic_ladder/plot_csr_strip.jl` reads `--physical-dir`, `--control-dir` and `--multimode-cache-dir`; supply `--n-orb`, `--filling`, `--n-seeds` and `--output-dir`. |
| Raw σ-SFF and DSFF | `supplement_raw_sigma_sff_dsff.jl` reads `--figure1-manifest` and `--figure3-manifest` from the previous figures, plus `--output-dir`. |
| Integrable corner | `bash src/model/scripts/slurm/submit_integrable_corner.sh` computes both panels and renders them. For individual calculations, `integrable_corner_compute.jl` accepts `--n-orb`, `--filling`, `--seed` and `--output`. `integrable_corner_render.jl` reads `--cache-dir` and `--panel-a`, with `--output-dir`. |

For the coherent Gibbs-state survival figure, use [`cgs_survival_beta0_n8_q4.toml`](src/model/configs/cgs_survival_beta0_n8_q4.toml), which fixes N = 8, Q = 4 and uses seeds 1:64. First obtain the single-mode N = 8 baseline calibration with `calibrate_ladder_stats.jl` as above. Then run `calibrate_cgs_figure3_syk4_reference.jl` with `--config`, `--baseline-calibration`, `--n-seeds 64`, `--n-random-jumps 300`, `--target-delta-tilde 0.01` and `--output` pointing to `$STORE/cgs_survival/calibration/figure3_diss_syk_n8_q4_m300.jld2`.

`cgs_survival_compute.jl` takes `--config`, `--dataset-key`, `--seed`, `--output-root "$STORE/cgs_survival"` and `--figure3-calibration` pointing to that calibration. Compute seeds 1:64 for each key: `fig1_eta_0p1`, `fig1_eta_0p4`, `fig1_eta_1`, `fig1_eta_2`, `fig1_diss_syk`, `fig3_dtilde_0p01`, `fig3_dtilde_0p1`, `fig3_dtilde_1`, `fig3_dtilde_10` and `fig3_diss_syk`. [Assemble the plot data](src/model/scripts/analysis/figures/cgs_survival_merge_validate.jl) with `--config`, the same `--output-root`, `--mode production` and `--expected-seeds 1:64`. Render with:

```sh
jrun "$FIG/cgs_survival_supplement.jl" \
  --plotdata "$STORE/cgs_survival/plotdata/cgs_survival_beta0_n8_q4.jld2" \
  --manifest "$STORE/cgs_survival/manifests/production.toml" --output-dir figures
```

The figure plotting scripts save PDFs and their TOML manifests in `figures/` by default. The manifests record numerical parameters and input paths, and provide the inputs for downstream supplementary plots.

## Citation

If you use this code, please cite the paper:

> P. Pacchioni, F. Ferrari, V. Savona, and M. Seclì, “Dissipation-induced Sachdev-Ye-Kitaev physics in many-body cavity quantum electrodynamics,” arXiv:2608.23557 (2026). https://arxiv.org/abs/2608.23557
