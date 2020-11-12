#!/usr/bin/env bash
#
# Slurm arguments.
#
#SBATCH --cpus-per-task=2
#SBATCH --export=ALL
#SBATCH --gres=gpu:1
#SBATCH --job-name "STREAM_INFERENCE_INTEGRAND_GD1"
#SBATCH --mem-per-cpu=5000
#SBATCH --ntasks=1
#SBATCH --output "logging/integrand_resnet_50_%a.log"
#SBATCH --parsable
#SBATCH --requeue
#SBATCH --time="7-00:00:00"
#


# Marginalized
model_query="$BASE/out/coverage/$EXPERIMENT_BATCH_SIZE/marginalized/$EXPERIMENT_ACTIVATION/ratio-estimator-resnet-50-$EXPERIMENT_EPOCHS-dropout-$EXPERIMENT_DROPOUT-wd-$EXPERIMENT_WEIGHT_DECAY-batchnorm-0-*/best-model.th"
out=$BASE/out/gd1/integrand-resnet-50-marginalized.npy

# Check if the architecture has already been trained.
if [ ! -f $out -o $PROJECT_FORCE_RERUN -ne 0 ]; then
    python -u integrand.py \
           --data $DATADIR/observed-noised.npy \
           --model $model_query \
           --out $out
fi
