#!/usr/bin/env bash
#
# Slurm arguments.
#
#SBATCH --export=ALL
#SBATCH --parsable
#SBATCH --job-name "STREAM_CUT_NOMINAL"
#SBATCH --output "logging/stream_cut_nominal.log"
#SBATCH --requeue
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4000
#SBATCH --ntasks=1
#SBATCH --time="7-00:00:00"
#



# Prepare the stream data directory.
data=$DATADIR/nominal

for block_path in $data/*; do
    if [ -f $block_path/densities.npy ]; then
        if [ ! -f $block_path/densities-cut.npy -o $PROJECT_FORCE_RERUN -ne 0 ]; then
            python -u cut.py \
                   --data $block_path/densities.npy \
                   --observed $DATADIR/GD1-stream-track-density.dat \
                   --out $block_path \
                   --low $EXPERIMENT_PHI_LOW \
                   --high $EXPERIMENT_PHI_HIGH \
                   --phi $DATADIR/phi.npy
        fi
    fi
done
