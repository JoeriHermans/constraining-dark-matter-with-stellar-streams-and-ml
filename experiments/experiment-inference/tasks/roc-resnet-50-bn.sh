#!/usr/bin/env bash
#
# Slurm arguments.
#
#SBATCH --cpus-per-task=4
#SBATCH --export=ALL
#SBATCH --gres=gpu:1
#SBATCH --job-name "STREAM_INFERENCE_ROC_RESNET_50_BN"
#SBATCH --mem-per-cpu=5000
#SBATCH --ntasks=1
#SBATCH --output "logging/roc_resnet_50_bn_%a.log"
#SBATCH --parsable
#SBATCH --requeue
#SBATCH --time="7-00:00:00"
#


# Marginalized
model_query="$BASE/out/coverage/$EXPERIMENT_BATCH_SIZE/marginalized/$EXPERIMENT_ACTIVATION/ratio-estimator-resnet-50-$EXPERIMENT_EPOCHS-dropout-$EXPERIMENT_DROPOUT-wd-$EXPERIMENT_WEIGHT_DECAY-batchnorm-1-*/best-model.th"
suffix=$(printf "%05d" $SLURM_ARRAY_TASK_ID)
out=$BASE/out/coverage/$EXPERIMENT_BATCH_SIZE/marginalized/$EXPERIMENT_ACTIVATION/roc-resnet-50-bn-$suffix.pickle

# Check if the architecture has already been trained.
if [ ! -f $out -o $PROJECT_FORCE_RERUN -ne 0 ]; then
    python -u diagnose-ratio.py \
           --model $model_query \
           --experiment $SLURM_ARRAY_TASK_ID \
           --out $out
fi

# # Not marginalized
# model_query="$BASE/out/coverage/$EXPERIMENT_BATCH_SIZE/not-marginalized/$EXPERIMENT_ACTIVATION/ratio-estimator-resnet-50-$EXPERIMENT_TASK_EPOCHS-dropout-$EXPERIMENT_DROPOUT-wd-$EXPERIMENT_WEIGHT_DECAY-batchnorm-1-*/best-model.th"
# suffix=$(printf "%05d" $SLURM_ARRAY_TASK_ID)
# out=$BASE/out/coverage/$EXPERIMENT_BATCH_SIZE/not-marginalized/$EXPERIMENT_ACTIVATION/roc-resnet-50-bn-$suffix.pickle

# # Check if the architecture has already been trained.
# if [ ! -f $out -o $PROJECT_FORCE_RERUN -ne 0 ]; then
#     python -u diagnose-ratio.py \
#            --model $model_query \
#            --experiment $SLURM_ARRAY_TASK_ID \
#            --out $out
# fi
