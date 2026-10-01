#!/bin/bash -l
#SBATCH --job-name=physical_pilot_analysis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --hint=multithread
#SBATCH --mem=450G
#SBATCH --time=24:00:00
set -euo pipefail
srun --cpu-bind=verbose,sockets --mem-bind=verbose,local \
    bash "$REPO_ROOT/src/model/scripts/slurm/physical_pilot_postprocess.sh"
