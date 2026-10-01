#!/bin/bash -l
#SBATCH --job-name=physical_n10q4_pilot
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=128
#SBATCH --hint=multithread
#SBATCH --mem=450G
#SBATCH --time=24:00:00
set -euo pipefail
stage=${1:?Usage: physical_pilot.sbatch STAGE}
case "$stage" in generate|eigen|svd) ;; *) echo "Invalid stage: $stage" >&2; exit 2;; esac
root=${PILOT_ROOT:?}
repo=${REPO_ROOT:?}
row=${SLURM_ARRAY_TASK_ID:?}
read -r eta shard1 shard2 count < <(sed -n "${row}p" "$root/pilot_shards.tsv")
[[ -n $eta && $count -gt 0 ]]
echo "stage=$stage eta=$eta shards=$shard1,$shard2/$count host=$(hostname) job=$SLURM_JOB_ID"
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=${OPENBLAS_THREADS_PER_WORKER:-64} OMP_NUM_THREADS=1
# Slurm places rank 0/1 on distinct sockets in one two-task job step.
srun --nodes=1 --ntasks=2 --ntasks-per-socket=1 --cpus-per-task=128 \
    --distribution=block:block --cpu-bind=verbose,sockets --mem-bind=verbose,local \
    --kill-on-bad-exit=1 --output="$root/logs/%j_${stage}_eta${row}_socket%t.out" \
    --error="$root/logs/%j_${stage}_eta${row}_socket%t.err" \
    bash -c 'shards=("$1" "$2"); exec bash "$REPO_ROOT/src/model/scripts/slurm/physical_pilot_worker.sh" "$4" "${shards[$SLURM_PROCID]}" "$3"' \
    bash "$shard1" "$shard2" "$count" "$stage"
