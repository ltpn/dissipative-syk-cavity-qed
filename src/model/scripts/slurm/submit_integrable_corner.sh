#!/bin/bash

set -euo pipefail

repo=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}
export REPO_ROOT="${repo}"
cd "${repo}"
output_root=${CORNER_OUTPUT_ROOT:-${repo}/data/integrable_corner}
figure_dir=${CORNER_FIGURE_DIR:-${repo}/figures}
panel_a_script=${repo}/src/model/scripts/slurm/integrable_corner_panel_a.sbatch
panel_b_script=${repo}/src/model/scripts/slurm/integrable_corner_panel_b.sbatch
render_script=${repo}/src/model/scripts/slurm/integrable_corner_render.sbatch

# Locked schedule: 10:3:16, 12:4:16, 14:5:12, 16:6:8.
schedules=("10:3:16:32:01:00:00" "12:4:16:64:02:00:00" \
           "14:5:12:128:04:00:00" "16:6:8:256:12:00:00")

mkdir -p "${output_root}" "${figure_dir}"
panel_a_job=$(sbatch --parsable --export="ALL,CORNER_OUTPUT_ROOT=${output_root}" \
  "${panel_a_script}")
panel_a_job=${panel_a_job%%;*}
dependencies=("${panel_a_job}")
echo "panel_a=${panel_a_job}"

for schedule in "${schedules[@]}"; do
  IFS=: read -r n_orb filling seeds cpus hours minutes seconds <<< "${schedule}"
  walltime=$(printf '%s:%s:%s' "${hours}" "${minutes}" "${seconds}")
  job=$(sbatch --parsable --array="1-${seeds}" --cpus-per-task="${cpus}" \
    --time="${walltime}" \
    --job-name="corner_n${n_orb}f${filling}" \
    --export="ALL,CORNER_OUTPUT_ROOT=${output_root},CORNER_N_ORB=${n_orb},CORNER_FILLING=${filling}" \
    "${panel_b_script}")
  job=${job%%;*}
  dependencies+=("${job}")
  echo "panel_b_n${n_orb}_f${filling}=${job}"
done

dependency=$(IFS=:; echo "${dependencies[*]}")
render_job=$(sbatch --parsable --dependency="afterok:${dependency}" \
  --export="ALL,CORNER_OUTPUT_ROOT=${output_root},CORNER_FIGURE_DIR=${figure_dir}" \
  "${render_script}")
echo "render=${render_job} dependency=afterok:${dependency}"
