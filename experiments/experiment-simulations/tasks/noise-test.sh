#!/usr/bin/env bash
#
# Slurm arguments.
#
#SBATCH --export=ALL
#SBATCH --parsable
#SBATCH --job-name "STREAM_NOISE_TEST"
#SBATCH --output "logging/stream_noise_test.log"
#SBATCH --requeue
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4000
#SBATCH --ntasks=1
#SBATCH --time="7-00:00:00"
#



# Prepare the stream data directory.
data=$DATADIR/test

# Check if the simulation has already been completed.
if [ ! -f $data/densities-cut-noised.npy -o $PROJECT_FORCE_RERUN -ne 0 ]; then
    python -u noise.py \
           --data $data/densities-cut-normalized.npy \
           --observed $DATADIR/GD1-stream-track-density.dat \
           --degree $EXPERIMENT_POLYNOMIAL_DEGREE \
           --out $data/densities-cut-noised-normalized.npy \
           --phi $DATADIR/phi-cut.npy
fi

if [ ! -f $data/density-contrasts-cut-noised.npy -o $PROJECT_FORCE_RERUN -ne 0 ]; then
    python -u noise.py \
           --data $data/density-contrasts-cut.npy \
           --observed $DATADIR/GD1-stream-track-density.dat \
           --degree $EXPERIMENT_POLYNOMIAL_DEGREE \
           --out $data/density-contrasts-cut-noised.npy \
           --phi $DATADIR/phi-cut.npy
fi
