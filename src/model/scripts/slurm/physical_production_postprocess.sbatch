#!/bin/bash -l
# Final stage, and the --final-script for timeout_recovery.py on the 16-seed
# ensemble. The controller submits it the moment the last spectral artifact
# lands, so it cannot wait on a separate dynamics array; instead it runs two
# dynamics shards here, one per socket (~60 GB each, bandwidth bound), which
# halves the ~14 h serial cost of 64 points, then merges once both have exited.
# Points already published by physical_production_dynamics.sbatch are skipped.
# Analysis only: the canonical figures are rendered by a separate job once every
# ensemble they combine has finished. The same script serves physical, SYK4
# control and multimode runs.
#SBATCH --job-name=production_analysis
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=128
#SBATCH --hint=multithread
#SBATCH --mem=450G
#SBATCH --time=24:00:00
set -euo pipefail
root=${PILOT_ROOT:?}
: "${REPO_ROOT:?}"
driver="$REPO_ROOT/src/model/scripts/slurm/physical_production_postprocess.sh"
mkdir -p "$root/logs"
srun --nodes=1 --ntasks=2 --ntasks-per-socket=1 --cpus-per-task=128 \
    --distribution=block:block --cpu-bind=verbose,sockets --mem-bind=verbose,local \
    --output="$root/logs/%j_final_dynamics_shard%t.out" \
    --error="$root/logs/%j_final_dynamics_shard%t.err" \
    bash -c 'exec bash "$1" dynamics "$(( SLURM_PROCID + 1 ))" 2' bash "$driver"
srun --nodes=1 --ntasks=1 --cpus-per-task=128 --cpu-bind=verbose,sockets \
    bash "$driver" merge
