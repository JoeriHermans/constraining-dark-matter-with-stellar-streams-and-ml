#!/usr/bin/env bash
#
# Slurm arguments.
#
#SBATCH --export=ALL
#SBATCH --parsable
#SBATCH --job-name "STREAM_CUT_TRAIN"
#SBATCH --output "logging/stream_cut_train.log"
#SBATCH --requeue
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4000
#SBATCH --ntasks=1
#SBATCH --time="7-00:00:00"
#



# Prepare the stream data directory.
data=$DATADIR/train

# Check if the simulation has already been completed.
if [ ! -f $data/densities-cut.npy -o $PROJECT_FORCE_RERUN -ne 0 ]; then
    python -u cut.py \
           --data $data/densities.npy \
           --observed $DATADIR/GD1-stream-track-density.dat \
           --out $data \
           --low $EXPERIMENT_PHI_LOW \
           --high $EXPERIMENT_PHI_HIGH \
           --phi $DATADIR/phi.npy
    cp $data/phi-cut.npy $DATADIR
fi
