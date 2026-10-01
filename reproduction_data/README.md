# Light canonical figure data

This directory contains 451.7 MiB of selected canonical ensembles for N=10, Q=4 and 16 realization slots per parameter, together with N=6/Q=3 and N=8/Q=4 inset ensembles. Run `src/model/scripts/reproduce_figures.jl` from the repository to render Figures 1–3, CSR and the raw-form-factor supplement. `--mode statistics` repeats the final spectral statistics from these inputs.

`dataset.toml` identifies the selected analysis directory for each ensemble. All metadata paths are relative to this directory. `inventory.json` records checksums; path normalization preserves every numerical payload exactly.

| Directory | Contents |
| --- | --- |
| `runs/*/analysis/*/spectra` | Per-seed eigenvalues, centered singular values, saved entropy/occupations and diagnostics. |
| `runs/*/analysis/*/multimode` | The same inputs merged across seeds for each multimode spacing. |
| `runs/*/analysis/*/{sigma-sff,dsff,csr,dynamics}` | Saved postprocessing results. |
| `runs/*/config.toml` | Matching ensemble configuration. |
| `runs/*/liouvillians/headers.toml`, recovery sidecars | Authentic generation identities for validating decompositions in the full dataset. No matrices are included. |
| `data` | Gaussian AI† reference and calibration payloads. |
| `derived/sigma_sff` | Cached singular-value ensembles for Figure 1. |
| `figures` | Prepared DSFF/CSR/raw-form-factor plot data and numerical manifests. |

Large eigendecompositions are absent. Saved dynamics cannot be recomputed for another initial state or time grid from this bundle. The integrable-corner and coherent-Gibbs-state survival supplements are not included.

| Ensemble or prepared artifact | Role in the figures |
| --- | --- |
| `physical_n10_q4` | Figure 1 physical σ-SFF and complex DSFF at η=0.1, 0.4, 1, 2; Figure 2 entropy and populations; physical CSR panels and raw form factors. |
| `syk4_control` | The single-mode SYK4 comparisons in Figures 1–2, the CSR control panel and raw form factors. |
| `physical_n6`, `physical_n8` | Figure 1 finite-size σ-SFF inset at η=2. |
| `multimode_n10_q4` | Figure 3 σ-SFF and populations at δ̃=0.01, 0.1, 1, 10; multimode CSR panels and raw form factors. |
| `multimode_n6_f3`, `multimode_n8_f4` | Figure 3 finite-size σ-SFF inset at δ̃=0.01. |
| `fig3_m300` | Figure 3 dissipative SYK4 comparison and its raw form factors. |
| `figures/figure_1_fixed_beta_dsff__plotdata.jld2` | Prepared β=1/2, θ=π/4 Figure 1 complex-DSFF curves, bootstrap intervals and Gaussian AI† comparison. |
| `figures/spectral_statistics__csr_strip__physical_eta_L3b__plotdata.jld2` | Seed-resolved complex spacing ratios for all nine CSR panels. |
| `figures/supplement_raw_sigma_sff_dsff__plotdata.jld2` | Prepared raw σ-SFF and DSFF curves for all ten ensembles. The reproduction command recomputes these from saved spectra. |
| `derived/sigma_sff/` | Figure 1 centered singular-value caches, including both finite-size insets. |
| `data/ai_dagger/` | Gaussian AI† reference needed when recomputing Figure 1 complex DSFF. |
| `data/syk4/`, `data/syk4_reference/` | Control calibrations and Figure 3 reference validation metadata. |
