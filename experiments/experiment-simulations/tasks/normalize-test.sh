#!/usr/bin/env bash
#
# Slurm arguments.
#
#SBATCH --export=ALL
#SBATCH --parsable
#SBATCH --job-name "STREAM_NORMALIZE_TEST"
#SBATCH --output "logging/stream_normalize_test.log"
#SBATCH --requeue
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4000
#SBATCH --ntasks=1
#SBATCH --time="7-00:00:00"
#



# Prepare the stream data directory.
data=$DATADIR/test

# Process the density
if [ ! -f $data/densities-cut-normalized.npy -o $PROJECT_FORCE_RERUN -ne 0 ]; then
    python -u normalize.py \
           --data $data/densities-cut.npy \
           --density \
           --out $data/densities-cut-normalized.npy
fi

if [ ! -f $data/density-contrasts-cut.npy -o $PROJECT_FORCE_RERUN -ne 0 ]; then
    python -u normalize.py \
           --data $data/densities-cut.npy \
           --degree $EXPERIMENT_POLYNOMIAL_DEGREE \
           --contrast \
           --out $data/density-contrasts-cut.npy \
           --phi $DATADIR/phi-cut.npy
fi
