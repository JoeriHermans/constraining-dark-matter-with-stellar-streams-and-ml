#!/usr/bin/env bash
#
# Slurm arguments.
#
#SBATCH --export=ALL
#SBATCH --parsable
#SBATCH --job-name "STREAM_NOISE_TRAIN"
#SBATCH --output "logging/stream_noise_train.log"
#SBATCH --requeue
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=15000
#SBATCH --ntasks=1
#SBATCH --time="7-00:00:00"
#



# Prepare the stream data directory.
data=$DATADIR/train

# Check if the simulation has already been completed.
if [ ! -f $data/densities-cut-noised-normalized.npy -o $PROJECT_FORCE_RERUN -ne 0 ]; then
    python -u noise.py \
           --data $data/densities-cut-normalized.npy \
           --observed $DATADIR/GD1-stream-track-density.dat \
           --degree $EXPERIMENT_POLYNOMIAL_DEGREE \
           --out $data/densities-cut-noised-normalized.npy \
           --replicate $EXPERIMENT_NOISY_REPLICATION \
           --phi $DATADIR/phi-cut.npy
    # Replicate the inputs
    python -u stack.py --input $data/masses.npy --output $data/masses-r.npy --replicate $EXPERIMENT_NOISY_REPLICATION
    python -u stack.py --input $data/ages.npy --output $data/ages-r.npy --replicate $EXPERIMENT_NOISY_REPLICATION
    python -u stack.py --input $data/impacts.npy --output $data/impacts-r.npy --replicate $EXPERIMENT_NOISY_REPLICATION
fi

if [ ! -f $data/density-contrasts-cut-noised.npy -o $PROJECT_FORCE_RERUN -ne 0 ]; then
    python -u noise.py \
           --data $data/density-contrasts-cut.npy \
           --observed $DATADIR/GD1-stream-track-density.dat \
           --degree $EXPERIMENT_POLYNOMIAL_DEGREE \
           --out $data/density-contrasts-cut-noised.npy \
           --replicate $EXPERIMENT_NOISY_REPLICATION \
           --phi $DATADIR/phi-cut.npy
fi
