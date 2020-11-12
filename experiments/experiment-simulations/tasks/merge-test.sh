#!/usr/bin/env bash
#
# Slurm arguments.
#
#SBATCH --export=ALL
#SBATCH --parsable
#SBATCH --job-name "STREAM_MERGE_TEST"
#SBATCH --output "logging/stream_merge_test.log"
#SBATCH --requeue
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4000
#SBATCH --ntasks=1
#SBATCH --time="7-00:00:00"
#



# Prepare the stream data directory.
data=$DATADIR/test

# Check if the simulation has already been completed.
if [ ! -f $data/densities.npy -o $PROJECT_FORCE_RERUN -ne 0 ]; then
    python -u merge.py --data $data --out $data
fi
