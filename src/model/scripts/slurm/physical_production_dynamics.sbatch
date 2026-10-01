#!/bin/bash -l
# Parallel dynamics stage for the 16-seed production ensemble: 4 etas x 16 seeds
# = 64 independent points, one 29.7 GB eigenvector matrix each. Submit with an
# explicit array, e.g. --array=1-16 for 32 shards (2 points per shard), and
# throttle nodes with --array=1-16%4 when the queue is busy.
#
# Two shards per node, one per socket: a point holds V plus the LU copy that
# `V \ c` makes, ~60 GB, so 120 GB of the 450 GiB node; and the 540-step
# propagation is memory-bandwidth bound, so the second shard is nearly free on
# its own memory controller. Shards are not killed on a sibling failure --- the
# points are independent and every completed artifact is kept.
#SBATCH --job-name=physical_production_dynamics
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=128
#SBATCH --hint=multithread
#SBATCH --mem=450G
#SBATCH --time=12:00:00
set -euo pipefail
root=${PILOT_ROOT:?}
: "${REPO_ROOT:?}"
: "${SLURM_ARRAY_TASK_ID:?submit with --array}"
count=$(( 2 * ${SLURM_ARRAY_TASK_COUNT:?submit with --array} ))
base=$(( 2 * (SLURM_ARRAY_TASK_ID - ${SLURM_ARRAY_TASK_MIN:-1}) ))
mkdir -p "$root/logs"
echo "array=$SLURM_ARRAY_TASK_ID shards=$((base+1)),$((base+2))/$count host=$(hostname) job=$SLURM_JOB_ID"
export OPENBLAS_THREADS_PER_WORKER=${OPENBLAS_THREADS_PER_WORKER:-64}
srun --nodes=1 --ntasks=2 --ntasks-per-socket=1 --cpus-per-task=128 \
    --distribution=block:block --cpu-bind=verbose,sockets --mem-bind=verbose,local \
    --output="$root/logs/%j_dynamics_shard%t.out" \
    --error="$root/logs/%j_dynamics_shard%t.err" \
    bash -c 'exec bash "$REPO_ROOT/src/model/scripts/slurm/physical_production_postprocess.sh" \
        dynamics "$(( $1 + SLURM_PROCID + 1 ))" "$2"' bash "$base" "$count"
