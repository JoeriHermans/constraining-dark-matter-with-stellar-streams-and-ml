#!/usr/bin/env bash
#
# Slurm arguments.
#
#SBATCH --export=ALL
#SBATCH --parsable
#SBATCH --job-name "STREAM_NORMALIZE_MOCK"
#SBATCH --output "logging/stream_normalize_mock.log"
#SBATCH --requeue
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4000
#SBATCH --ntasks=1
#SBATCH --time="7-00:00:00"
#



# Prepare the stream data directory.
data=$DATADIR/mock

# Process the density
for block_path in $data/*; do
    if [ -f $block_path/densities-cut.npy ]; then
        if [ ! -f $block_path/densities-cut-normalized.npy -o $PROJECT_FORCE_RERUN -ne 0 ]; then
            python -u normalize.py \
                   --data $block_path/densities-cut.npy \
                   --density \
                   --out $block_path/densities-cut-normalized.npy
        fi
    fi
done

# Process the density contrasts
for block_path in $data/*; do
    if [ -f $block_path/densities-cut.npy ]; then
        if [ ! -f $block_path/density-contrasts-cut.npy -o $PROJECT_FORCE_RERUN -ne 0 ]; then
            python -u normalize.py \
                   --data $block_path/densities-cut.npy \
                   --degree $EXPERIMENT_POLYNOMIAL_DEGREE \
                   --contrast \
                   --out $block_path/density-contrasts-cut.npy \
                   --phi $DATADIR/phi-cut.npy
        fi
    fi
done
