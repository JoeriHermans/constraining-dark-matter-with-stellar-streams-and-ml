#!/usr/bin/env bash
#
# Slurm arguments.
#
#SBATCH --cpus-per-task=4
#SBATCH --export=ALL
#SBATCH --gres=gpu:1
#SBATCH --job-name "STREAM_INFERENCE_TRAIN_ROC_MLP_MARGINALIZED"
#SBATCH --mem-per-cpu=5000
#SBATCH --ntasks=1
#SBATCH --output "logging/train_roc_mlp_marginalized_%a.log"
#SBATCH --parsable
#SBATCH --requeue
#SBATCH --time="7-00:00:00"
#

suffix=$(printf "%05d" $SLURM_ARRAY_TASK_ID)
out=$BASE/out/roc/$EXPERIMENT_BATCH_SIZE/marginalized/$EXPERIMENT_ACTIVATION/ratio-estimator-mlp-$EXPERIMENT_EPOCHS-dropout-$EXPERIMENT_DROPOUT-wd-$EXPERIMENT_WEIGHT_DECAY-batchnorm-$EXPERIMENT_TASK_BATCHNORM-$suffix
mkdir -p $out

# Check if the architecture has already been trained.
if [ ! -f $out/model.th -o $PROJECT_FORCE_RERUN -ne 0 ]; then
    python -u train.py \
           --activation $EXPERIMENT_ACTIVATION \
           --batch-size $EXPERIMENT_BATCH_SIZE \
           --batchnorm $EXPERIMENT_TASK_BATCHNORM \
           --beta $EXPERIMENT_CONSERVATIVE \
           --cut \
           --n-train $EXPERIMENT_TRAIN_SIZE \
           --data-test-masses $DATADIR_TEST/masses.npy \
           --data-test-outputs $DATADIR_TEST/density-contrasts-cut-noised.npy \
           --data-train-masses $DATADIR_TRAIN/masses-r.npy \
           --data-train-outputs $DATADIR_TRAIN/density-contrasts-cut-noised.npy \
           --dropout $EXPERIMENT_DROPOUT \
           --epochs $EXPERIMENT_EPOCHS \
           --lr $EXPERIMENT_LEARNING_RATE \
           --mlp \
           --out $out \
           --resnet-depth $EXPERIMENT_RESNET_DEPTH \
           --weight-decay $EXPERIMENT_WEIGHT_DECAY \
           --workers $EXPERIMENT_WORKERS
fi
