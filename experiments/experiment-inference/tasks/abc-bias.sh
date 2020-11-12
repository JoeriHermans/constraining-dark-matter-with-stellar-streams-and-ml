#!/usr/bin/env bash
#
# Slurm arguments.
#
#SBATCH --cpus-per-task=1
#SBATCH --export=ALL
#SBATCH --job-name "STREAM_INFERENCE_ABC_BIAS"
#SBATCH --mem-per-cpu=4000
#SBATCH --ntasks=1
#SBATCH --output "logging/abc_bias_%a.log"
#SBATCH --parsable
#SBATCH --requeue
#SBATCH --time="1-00:00:00"
#

suffix=$(printf "%05d" $SLURM_ARRAY_TASK_ID)
# Noisy densities
out=$BASE/out/bias/abc-$suffix
mkdir -p $out

# Check if the procedure already completed
if [ ! -f $out/samples.npy -o $PROJECT_FORCE_RERUN -ne 0 ]; then
    python -u abc-new.py \
           --auto \
           --ages $DATADIR/train/ages-r.npy \
           --masses $DATADIR/train/masses-r.npy \
           --threshold $EXPERIMENT_ABC_THRESHOLD \
           --observations $DATADIR/nominal/block-00008/density-contrasts-cut-noised.npy \
           --observation-index $SLURM_ARRAY_TASK_ID \
           --out $out \
           --outputs $DATADIR/train/density-contrasts-cut-noised.npy
fi

# Check if the produced histogram is generated.
if [ ! -f $out/histogram.npy -o $PROJECT_FORCE_RERUN -ne 0 ]; then
    # TODO Implement
    echo "Generating histogram"
fi
