#!/usr/bin/env bash
#
# Slurm arguments.
#
#SBATCH --cpus-per-task=4
#SBATCH --export=ALL
#SBATCH --gres=gpu:1
#SBATCH --job-name "STREAM_INFERENCE_DIAGNOSTIC_ROC_2D"
#SBATCH --mem-per-cpu=4000
#SBATCH --ntasks=1
#SBATCH --output "logging/diagnostic_roc_2d_%a.log"
#SBATCH --parsable
#SBATCH --requeue
#SBATCH --time="7-00:00:00"
#

suffix=$(printf "%05d" $SLURM_ARRAY_TASK_ID)
marginal_data=$DATADIR/test

# Noisy observations
out=$BASE/out/posterior/noisy/not-marginalized/$EXPERIMENT_ACTIVATION
model=$out/ratio-estimator-depth-$EXPERIMENT_RESNET_DEPTH-$EXPERIMENT_EPOCHS-dropout-$EXPERIMENT_DROPOUT-wd-$EXPERIMENT_WEIGHT_DECAY-batchnorm-$EXPERIMENT_BATCHNORM-\*/best-model.th
if [ ! -f $out/diagnostic-roc.pickle -o $PROJECT_FORCE_RERUN -ne 0 ]; then
    python -u diagnose-ratio.py \
           --likelihood $DATADIR/nominal/block-$suffix \
           --marginal $marginal_data \
           --model $model \
           --out $out/diagnostic-roc-$suffic.pickle
fi

# Clean observations
out=$BASE/out/posterior/clean/not-marginalized/$EXPERIMENT_ACTIVATION
model=$out/ratio-estimator-depth-$EXPERIMENT_RESNET_DEPTH-$EXPERIMENT_EPOCHS-dropout-$EXPERIMENT_DROPOUT-wd-$EXPERIMENT_WEIGHT_DECAY-batchnorm-$EXPERIMENT_BATCHNORM-\*/best-model.th
if [ ! -f $out/diagnostic-roc.pickle -o $PROJECT_FORCE_RERUN -ne 0 ]; then
    python -u diagnose-ratio.py \
           --likelihood $DATADIR/nominal/block-$suffix \
           --marginal $marginal_data \
           --model $model \
           --out $out/diagnostic-roc-$suffic.pickle
fi
