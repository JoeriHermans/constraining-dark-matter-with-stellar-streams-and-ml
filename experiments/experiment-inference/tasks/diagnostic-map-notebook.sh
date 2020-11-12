#!/usr/bin/env bash
#
# Slurm arguments.
#
#SBATCH --cpus-per-task=4
#SBATCH --export=ALL
#SBATCH --gres=gpu:1
#SBATCH --job-name "STREAM_INFERENCE_DIAGNOSTIC_MAP_NOTEBOOK"
#SBATCH --mem-per-cpu=4000
#SBATCH --ntasks=1
#SBATCH --output "logging/diagnostic_map_notebook.log"
#SBATCH --parsable
#SBATCH --requeue
#SBATCH --time="7-00:00:00"
#

out=$BASE/out
mkdir -p $out

if [ ! -f $out/diagnostic-map-convergence.ipynb -o $PROJECT_FORCE_RERUN -ne 0 ]; then
    papermill diagnostic-map-convergence.ipynb $out/diagnostic-map-convergence.ipynb
fi
