#!/usr/bin/env bash
#
# Slurm arguments.
#
#SBATCH --cpus-per-task=1
#SBATCH --export=ALL
#SBATCH --job-name "STREAM_INFERENCE_ABC_NEW"
#SBATCH --mem-per-cpu=4000
#SBATCH --ntasks=1
#SBATCH --output "logging/abc_new.log"
#SBATCH --parsable
#SBATCH --requeue
#SBATCH --time="1-00:00:00"
#

suffix=$(printf "%05d" $EXPERIMENT_MOCK_INDEX)

# Noisy densities
out=$BASE/out/mock/abc-new-$suffix
mkdir -p $out

# Check if the procedure already completed
if [ ! -f $out/samples.npy -o $PROJECT_FORCE_RERUN -ne 0 ]; then
    python -u abc-new.py \
           --average \
           --auto \
           --ages $DATADIR/train/ages-r.npy \
           --masses $DATADIR/train/masses-r.npy \
           --threshold $EXPERIMENT_ABC_THRESHOLD \
           --observations $DATADIR/mock/block-$suffix/density-contrasts-cut-noised.npy \
           --out $out \
           --outputs $DATADIR/train/density-contrasts-cut-noised.npy
fi
