# Temporary entry point installed only in a preserved legacy run snapshot.
# Already-running eigen processes have loaded their original code. Their next
# Julia invocation must hand SVD work to the separately scheduled SVD array.
using TOML
handoff = TOML.parsefile(joinpath(@__DIR__,"legacy_stage_handoff.toml"))
legacy_array = get(ENV,"SLURM_ARRAY_JOB_ID","") in handoff["eigen_only_arrays"]
if !isempty(ARGS) && first(ARGS) == "svd" && legacy_array
    println("DEFERRED SVD: this legacy allocation now finishes after eigen; SVD runs in a separate job with its own time limit. No SVD was computed here.")
    exit(0)
end
include(joinpath(@__DIR__,"pipeline_before_stage_split.jl"))
exit(PipelineCLI.main(ARGS))
