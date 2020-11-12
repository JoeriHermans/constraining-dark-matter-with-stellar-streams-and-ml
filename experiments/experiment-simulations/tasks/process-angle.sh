#!/usr/bin/env bash
#
# Slurm arguments.
#
#SBATCH --export=ALL
#SBATCH --parsable
#SBATCH --job-name "STREAM_PROCESS-ANGLE"
#SBATCH --output "logging/stream_process_angle.log"
#SBATCH --requeue
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4000
#SBATCH --ntasks=1
#SBATCH --time="7-00:00:00"
#



# Check if the simulation has already been completed.
if [ ! -f $DATADIR/phi.npy -o $PROJECT_FORCE_RERUN -ne 0 ]; then
    python angle.py --input $DATADIR/train/block-00000/phi.npy --output $DATADIR/phi.npy
fi
