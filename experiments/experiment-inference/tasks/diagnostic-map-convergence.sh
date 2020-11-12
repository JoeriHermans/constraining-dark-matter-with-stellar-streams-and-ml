#!/usr/bin/env bash
#
# Slurm arguments.
#
#SBATCH --cpus-per-task=4
#SBATCH --export=ALL
#SBATCH --gres=gpu:1
#SBATCH --job-name "STREAM_INFERENCE_DIAGNOSTIC_MAP_CONVERGENCE"
#SBATCH --mem-per-cpu=4000
#SBATCH --ntasks=1
#SBATCH --output "logging/diagnostic_diagnostic_map_convergence_%a.log"
#SBATCH --parsable
#SBATCH --requeue
#SBATCH --time="7-00:00:00"
#

suffix=$(printf "%05d" $SLURM_ARRAY_TASK_ID)
data=$DATADIR/nominal/block-$suffix

# Noisy observations, masses as inputs, ages marginalized
out=$BASE/out/posterior/contrasts/noisy/marginalized/$EXPERIMENT_ACTIVATION
model=$out/ratio-estimator-depth-$EXPERIMENT_RESNET_DEPTH-$EXPERIMENT_EPOCHS-dropout-$EXPERIMENT_DROPOUT-wd-$EXPERIMENT_WEIGHT_DECAY-batchnorm-$EXPERIMENT_BATCHNORM-\*best-model.th
if [ ! -f $out/diagnostic-integrand.npy -o $PROJECT_FORCE_RERUN -ne 0 ]; then
    python -u diagnose-map-convergence.py \
           --data $data \
           --model $model \
           --out $out
fi

# Noisy observations, masses and ages as inputs
out=$BASE/out/posterior/contrasts/noisy/not-marginalized/$EXPERIMENT_ACTIVATION
model=$out/ratio-estimator-depth-$EXPERIMENT_RESNET_DEPTH-$EXPERIMENT_EPOCHS-dropout-$EXPERIMENT_DROPOUT-wd-$EXPERIMENT_WEIGHT_DECAY-batchnorm-$EXPERIMENT_BATCHNORM-\*/best-model.th
if [ ! -f $out/diagnostic-integrand.npy -o $PROJECT_FORCE_RERUN -ne 0 ]; then
    python -u diagnose-map-convergence.py \
           --data $data \
           --model $model \
           --out $out
fi
