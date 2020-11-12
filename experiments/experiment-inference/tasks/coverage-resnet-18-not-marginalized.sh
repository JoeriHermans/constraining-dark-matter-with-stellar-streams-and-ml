#!/usr/bin/env bash
#
# Slurm arguments.
#
#SBATCH --cpus-per-task=2
#SBATCH --export=ALL
#SBATCH --gres=gpu:1
#SBATCH --job-name "STREAM_INFERENCE_COVERAGE_RESNET_18_NOT_MARGINALIZED"
#SBATCH --mem-per-cpu=5000
#SBATCH --ntasks=1
#SBATCH --output "logging/coverage_resnet_18_not_marginalized_%a.log"
#SBATCH --parsable
#SBATCH --requeue
#SBATCH --time="7-00:00:00"
#

model_query="$BASE/out/coverage/$EXPERIMENT_BATCH_SIZE/marginalized/$EXPERIMENT_ACTIVATION/ratio-estimator-resnet-18-$EXPERIMENT_TASK_EPOCHS-dropout-$EXPERIMENT_DROPOUT-wd-$EXPERIMENT_WEIGHT_DECAY-batchnorm-0-*/best-model.th"
suffix=$(printf "%05d" $SLURM_ARRAY_TASK_ID)
out=$BASE/out/coverage/$EXPERIMENT_BATCH_SIZE/not-marginalized/$EXPERIMENT_ACTIVATION/coverage-$EXPERIMENT_TASK_COVERAGE-resnet-18-not-marginalized-$suffix.npy

# Check if the architecture has already been trained.
if [ ! -f $out -o $PROJECT_FORCE_RERUN -ne 0 ]; then
    python -u coverage.py \
           --level $EXPERIMENT_TASK_COVERAGE \
           --data $DATADIR_TEST \
           --model $model_query \
           --out $out
fi
