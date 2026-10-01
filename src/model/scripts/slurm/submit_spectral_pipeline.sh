#!/usr/bin/env bash
set -euo pipefail
repo=${REPO_ROOT:-$(cd "$(dirname "$0")/../../../.." && pwd)}
config= data_root= dry_run=0
tasks=sff,csr,dynamics
generate_shards=1 eigen_shards=1 svd_shards=1
generate_cpus=1 generate_mem=8G generate_time=02:00:00
eigen_cpus=8 eigen_mem=32G eigen_time=24:00:00
svd_cpus=8 svd_mem=32G svd_time=24:00:00
postprocess_cpus=4 postprocess_mem=32G postprocess_time=08:00:00
usage() {
    cat <<'HELP'
Usage: bash submit_spectral_pipeline.sh --config RUN.toml --data-root DATA [OPTIONS]
  --dry-run                     Print commands without submitting or creating files
  --tasks sff,csr,dynamics       Selected postprocessing tasks (default: all)
  --generate-shards N            Generation array size (default: 1)
  --eigen-shards N               Eigen array size (default: 1)
  --svd-shards N                 SVD array size (default: 1)
  --STAGE-cpus N                 Per-job CPU count
  --STAGE-mem SIZE               Per-job Slurm memory request (e.g. 64G)
  --STAGE-time LIMIT             Per-job Slurm wall time (e.g. 24:00:00)
  STAGE is generate, eigen, svd, or postprocess.
Site settings: SBATCH_ACCOUNT, SBATCH_PARTITION, REPO_ROOT, JULIA.
Defaults are examples: size memory and wall time for your matrix dimensions.
HELP
}
while (($#)); do
    option=$1; shift
    case "$option" in
        --help) usage; exit 0;;
        --dry-run) dry_run=1; continue;;
        --config|--data-root|--tasks|--generate-shards|--eigen-shards|--svd-shards|--generate-cpus|--generate-mem|--generate-time|--eigen-cpus|--eigen-mem|--eigen-time|--svd-cpus|--svd-mem|--svd-time|--postprocess-cpus|--postprocess-mem|--postprocess-time)
            (($#)) && [[ -n "$1" && "$1" != --* ]] || { echo "$option requires a value" >&2; exit 2; }
            name=${option#--}; name=${name//-/_}
            printf -v "$name" '%s' "$1"; shift;;
        *) echo "Unknown option: $option" >&2; exit 2;;
    esac
done
[[ -f "$config" && -n "$data_root" ]] || { usage >&2; exit 2; }
for name in generate_shards eigen_shards svd_shards generate_cpus eigen_cpus svd_cpus postprocess_cpus; do
    [[ ${!name} =~ ^[1-9][0-9]*$ ]] || { echo "$name must be positive" >&2; exit 2; }
done
config=$(cd "$(dirname "$config")" && pwd)/$(basename "$config")
[[ "$data_root" == /* ]] || data_root="$PWD/$data_root"
export REPO_ROOT=$repo
wrapper="$repo/src/model/scripts/slurm/pipeline_stage.sbatch"
((dry_run)) || mkdir -p "$data_root/logs"

submit() {
    local stage=$1
    shift
    local cpu_var=${stage}_cpus mem_var=${stage}_mem time_var=${stage}_time
    local command=(sbatch --parsable "--job-name=pipeline-$stage" "--cpus-per-task=${!cpu_var}" "--mem=${!mem_var}" "--time=${!time_var}"
        "--output=$data_root/logs/%x-%A_%a.out" "--error=$data_root/logs/%x-%A_%a.err")
    [[ -z ${SBATCH_ACCOUNT:-} ]] || command+=("--account=$SBATCH_ACCOUNT")
    [[ -z ${SBATCH_PARTITION:-} ]] || command+=("--partition=$SBATCH_PARTITION")
    if [[ "$stage" != postprocess ]]; then
        local shards_var=${stage}_shards
        command+=("--array=1-${!shards_var}")
    fi
    command+=("$@" "$wrapper" "$stage")
    case "$stage" in
        generate) command+=(--config "$config" --output "$data_root/liouvillians");;
        eigen|svd) command+=(--input "$data_root/liouvillians" --output "$data_root/$stage");;
        postprocess) command+=(--config "$config" --eigen "$data_root/eigen" --svd "$data_root/svd" --tasks "$tasks" --output "$data_root/analysis");;
    esac
    printf '%q ' "${command[@]}" >&2; printf '\n' >&2
    if ((dry_run)); then
        printf '%s\n' "${stage}_JOB_ID"
    else
        local reply job
        if ! reply=$("${command[@]}"); then
            echo "Submission failed for $stage; previously printed job IDs remain submitted." >&2
            return 1
        fi
        job=${reply%%;*}
        [[ "$job" =~ ^[0-9]+$ ]] || { echo "Invalid sbatch job ID: $reply" >&2; return 1; }
        echo "$stage job: $job" >&2
        printf '%s\n' "$job"
    fi
}
generate_id=$(submit generate)
eigen_id=$(submit eigen "--dependency=afterok:$generate_id")
svd_id=$(submit svd "--dependency=afterok:$generate_id")
postprocess_id=$(submit postprocess "--dependency=afterok:$eigen_id:$svd_id")
printf 'Pipeline: %s -> (%s, %s) -> %s\n' "$generate_id" "$eigen_id" "$svd_id" "$postprocess_id"
