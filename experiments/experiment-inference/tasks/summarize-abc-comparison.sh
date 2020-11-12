#!/usr/bin/env bash
#
# Slurm arguments.
#
#SBATCH --cpus-per-task=4
#SBATCH --export=ALL
#SBATCH --job-name "STREAM_INFERENCE_SUMMARIZE_ABC_COMPARISON"
#SBATCH --mem-per-cpu=4000
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output "logging/summarize_abc_comparison.log"
#SBATCH --parsable
#SBATCH --requeue
#SBATCH --time="7-00:00:00"
#

out=$BASE/out
mkdir -p $out

if [ ! -f $out/summary-abc-comparison.ipynb -o $PROJECT_FORCE_RERUN -ne 0 ]; then
    papermill summary-abc-comparison.ipynb $out/summary-abc-comparison.ipynb
fi
