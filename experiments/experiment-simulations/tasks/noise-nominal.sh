#!/usr/bin/env bash
#
# Slurm arguments.
#
#SBATCH --export=ALL
#SBATCH --parsable
#SBATCH --job-name "STREAM_NOISE_NOMINAL"
#SBATCH --output "logging/stream_noise_nominal.log"
#SBATCH --requeue
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4000
#SBATCH --ntasks=1
#SBATCH --time="7-00:00:00"
#



# Prepare the stream data directory.
data=$DATADIR/nominal

# Densities
for block_path in $data/*; do
    if [ -f $block_path/densities-cut-normalized.npy ]; then
        if [ ! -f $block_path/densities-cut-noised.npy -o $PROJECT_FORCE_RERUN -ne 0 ]; then
            python -u noise.py \
                   --data $block_path/densities-cut-normalized.npy \
                   --observed $DATADIR/GD1-stream-track-density.dat \
                   --degree $EXPERIMENT_POLYNOMIAL_DEGREE \
                   --out $block_path/densities-cut-noised-normalized.npy \
                   --phi $DATADIR/phi-cut.npy
        fi
    fi
done

# Density contrasts
for block_path in $data/*; do
    if [ -f $block_path/density-contrasts-cut.npy ]; then
        if [ ! -f $block_path/density-contrasts-cut-noised.npy -o $PROJECT_FORCE_RERUN -ne 0 ]; then
            python -u noise.py \
                   --data $block_path/density-contrasts-cut.npy \
                   --observed $DATADIR/GD1-stream-track-density.dat \
                   --degree $EXPERIMENT_POLYNOMIAL_DEGREE \
                   --out $block_path/density-contrasts-cut-noised.npy \
                   --phi $DATADIR/phi-cut.npy
        fi
    fi
done
