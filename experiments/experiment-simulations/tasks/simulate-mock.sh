#!/usr/bin/env bash
#
# Slurm arguments.
#
#SBATCH --export=ALL
#SBATCH --parsable
#SBATCH --job-name "STREAM_SIMULATE_MOCK"
#SBATCH --output "logging/stream_simulate_mock_%a.log"
#SBATCH --requeue
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2000
#SBATCH --ntasks=1
#SBATCH --time="7-00:00:00"
#


# Fetch the next stream index.
stream_index=$SLURM_ARRAY_TASK_ID
# Prepare the stream data directory.
suffix=$(printf "%05d" $stream_index)
task_identifier="block-"$suffix
out=$DATADIR/mock/$task_identifier
mkdir -p $out
# Check if the simulation has already been completed.
if [ ! -f $out/densities.npy -o $PROJECT_FORCE_RERUN -ne 0 ]; then
    # Compute the number of required simulations.
    python -u simulate.py \
           --mocks $EXPERIMENT_MOCKS \
           --mock-index $SLURM_ARRAY_TASK_ID \
           --out $out \
           --size 1
fi
