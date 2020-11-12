#!/usr/bin/env bash
#
# Slurm arguments.
#
#SBATCH --cpus-per-task=4
#SBATCH --export=ALL
#SBATCH --gres=gpu:1
#SBATCH --job-name "STREAM_INFERENCE_SUMMARIZE_MOCK_INFERENCE"
#SBATCH --mem-per-cpu=4000
#SBATCH --ntasks=1
#SBATCH --output "logging/summarize_mock_inference.log"
#SBATCH --parsable
#SBATCH --requeue
#SBATCH --time="7-00:00:00"
#

out=$BASE/out
mkdir -p $out

if [ ! -f $out/summary-mock-inference.ipynb -o $PROJECT_FORCE_RERUN -ne 0 ]; then
    papermill summary-mock-inference.ipynb $out/summary-mock-inference.ipynb
fi
