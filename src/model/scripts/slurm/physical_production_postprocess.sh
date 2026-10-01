#!/usr/bin/env bash
# Production analysis driver, for any staged ensemble (physical, SYK4
# control or multimode). Figures are not rendered here: the canonical paper
# renderers need several ensembles at once and run as a separate job.
#
#   dynamics INDEX COUNT   compute one shard of the dynamics points and stop
#   merge                  compute anything still missing, then the ensemble
#                          tasks and the figure-compatible legacy exports
#
# Both modes build the identical run_postprocessing call because the analysis
# directory is the hash of tasks, time grid, dynamics config and source artifact
# ids: a shard invoked with different settings silently writes somewhere else.
# Shards are disjoint and each artifact is claim-protected, so they are safe to
# run concurrently, but never alongside the merge, which walks every point.
set -euo pipefail
mode=${1:?Usage: physical_production_postprocess.sh dynamics INDEX COUNT | merge}
root=${PILOT_ROOT:?}
repo=${REPO_ROOT:?}
seeds=${PRODUCTION_SEEDS:-1:16}
threads=${OPENBLAS_THREADS_PER_WORKER:-64}
selection=${RECOVERY_SELECTION:-$root/recovery/selection.toml}
# Optional per-run task list, kept in the run root so finals submitted later by
# watch jobs see it. Small ensembles omit dsff: its density fit needs more
# eigenvalues than, e.g., 16 seeds at N=6 provide, and no canonical figure reads it.
tasks=$( [[ -f $root/postprocess_tasks ]] && tr -d '[:space:]' < "$root/postprocess_tasks" || echo sigma-sff,dsff,csr,dynamics)
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS="$threads" OMP_NUM_THREADS=1
jl=("${JULIA:?}" --project="$repo/src/environment")

run_analysis() {
    "${jl[@]}" -e '
        include(joinpath(ARGS[1],"src/model/SpectralPostprocessing.jl"))
        P = SpectralPostprocessing
        P.LinearAlgebra.BLAS.set_num_threads(parse(Int,ARGS[6]))
        root, selection = ARGS[2], ARGS[5]
        index, count = parse(Int,ARGS[3]), parse(Int,ARGS[4])
        bounds = parse.(Int,split(ARGS[7],":"))
        cfg = P.SpectralArtifacts.load_pipeline_config(joinpath(root,"config.toml"))
        result = P.run_postprocessing(cfg,joinpath(root,"analysis");
            eigen_dir=joinpath(root,"eigen"),svd_dir=joinpath(root,"svd"),
            seeds=collect(bounds[1]:bounds[2]),
            replacements=isfile(selection) ? selection : nothing,
            shard_index=index,shard_count=count,tasks=split(ARGS[8],","))
        result.failed == 0 || error("Postprocessing failed")
        count == 1 && write(joinpath(root,"analysis_directory.txt"),result.directory*"\n")
    ' "$repo" "$root" "$1" "$2" "$selection" "$threads" "$seeds" "$tasks"
}

case "$mode" in
dynamics)
    index=${2:?shard index}
    count=${3:?shard count}
    echo "$(date -Is) starting dynamics shard=$index/$count seeds=$seeds host=$(hostname) pid=$$"
    awk '/Cpus_allowed_list|Mems_allowed_list/ {print}' /proc/self/status
    run_analysis "$index" "$count"
    echo "$(date -Is) completed dynamics shard=$index/$count"
    ;;
merge)
    echo "$(date -Is) starting merge seeds=$seeds host=$(hostname)"
    run_analysis 1 1
    date -Is > "$root/ANALYSIS_COMPLETED"
    echo "$(date -Is) completed merge: $(cat "$root/analysis_directory.txt")"
    ;;
*)
    echo "Invalid mode: $mode" >&2; exit 2;;
esac
