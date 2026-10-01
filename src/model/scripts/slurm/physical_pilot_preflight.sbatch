#!/bin/bash -l
#SBATCH --job-name=physical_pilot_preflight
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=128
#SBATCH --hint=multithread
#SBATCH --mem=450G
#SBATCH --time=00:30:00
set -euo pipefail
stage=${1:?Submit separate generate, eigen, and svd preflight jobs; specify STAGE}
export PILOT_N_ORB=4 PILOT_FILLING=2
for row in 1 2 3 4; do
    SLURM_ARRAY_TASK_ID=$row bash "$REPO_ROOT/src/model/scripts/slurm/physical_pilot.sbatch" "$stage"
done
