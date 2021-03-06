#!/usr/bin/env bash
#
# Slurm arguments.
#
#SBATCH --cpus-per-task=2
#SBATCH --export=ALL
#SBATCH --gres=gpu:1
#SBATCH --job-name "STREAM_INFERENCE_COVERAGE_FREQUENTIST_RESNET_50_BN_MARGINALIZED"
#SBATCH --mem-per-cpu=5000
#SBATCH --ntasks=1
#SBATCH --output "logging/coverage_frequentist_resnet_50_bn_marginalized_%a.log"
#SBATCH --parsable
#SBATCH --requeue
#SBATCH --time="7-00:00:00"
#

model_query="$BASE/out/coverage/$EXPERIMENT_BATCH_SIZE/marginalized/$EXPERIMENT_ACTIVATION/ratio-estimator-resnet-50-$EXPERIMENT_TASK_EPOCHS-dropout-$EXPERIMENT_DROPOUT-wd-$EXPERIMENT_WEIGHT_DECAY-batchnorm-1-*/best-model.th"
suffix=$(printf "%05d" $SLURM_ARRAY_TASK_ID)
out=$BASE/out/coverage/$EXPERIMENT_BATCH_SIZE/marginalized/$EXPERIMENT_ACTIVATION/coverage-frequentist-$EXPERIMENT_TASK_COVERAGE-resnet-50-bn-marginalized-$suffix.npy

# Check if the architecture has already been trained.
if [ ! -f $out -o $PROJECT_FORCE_RERUN -ne 0 ]; then
    python -u coverage.py \
           --level $EXPERIMENT_TASK_COVERAGE \
           --frequentist \
           --data $DATADIR_TEST \
           --model $model_query \
           --out $out
fi
