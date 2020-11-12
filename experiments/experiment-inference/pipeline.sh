#!/usr/bin/env bash -i
#

### Pipeline definitions
################################################################################
export PIPELINE_NAME="Inference"

log_info "Starting experiment: $PIPELINE_NAME\n"

### Script and pipeline initialization
################################################################################
export BASE=$(dirname $BASH_SOURCE) # Set the path of the current directory.

# Create the logging directory.
mkdir -p $BASE/logging
# Create the output directory.
mkdir -p $BASE/out
# Create the plots directory.
mkdir -p $BASE/plots
# Assign the data directory.
export DATADIR=$BASE/../experiment-simulations/data
export DATADIR_TRAIN=$DATADIR/train
export DATADIR_TEST=$DATADIR/test

### Check if the training and testing data is available
################################################################################
if [ ! -f $DATADIR/train/densities.npy ]; then
    log_error "Required data files are not present, please run the simulations first.\n"
    exit
else
    log_info "Data files are present, we can initiate the inference procedure!\n"
fi

### Main experimental variables
################################################################################x
export EXPERIMENT_ABC_THRESHOLD=0.001
export EXPERIMENT_ACTIVATION="selu"
export EXPERIMENT_BATCHNORM=1
export EXPERIMENT_BATCH_SIZE=4096
export EXPERIMENT_CONSERVATIVE=0.0
export EXPERIMENT_DROPOUT=0.0
export EXPERIMENT_EPOCHS=50
export EXPERIMENT_LEARNING_RATE=0.0001
export EXPERIMENT_MOCKS=$(find $DATADIR/mock/block*/densities.npy | wc -l)
export EXPERIMENT_NOMINALS=$(find $DATADIR/nominal/block*/densities.npy | wc -l)
export EXPERIMENT_NUM_RATIO_ESTIMATORS=10
export EXPERIMENT_RESNET_DEPTH=18
export EXPERIMENT_WEIGHT_DECAY=0.1
export EXPERIMENT_WORKERS=4


### Stage 0. Probe bias of ABC
################################################################################
checkpoint=$(sbatch --chdir=$BASE --array=0-999 $BASE/tasks/abc-bias.sh)
exit

### Stage 1. Examine the batch-size effects
################################################################################

# Train the models for the same amount of epochs
export EXPERIMENT_TASK_EPOCHS=50
export EXPERIMENT_TASK_BATCHNORM=1
checkpoint=1

export EXPERIMENT_TASK_BATCH_SIZE=64
task_mlp=$(sbatch --chdir=$BASE --array=0-2 $BASE/tasks/train-batch-size-mlp.sh)
checkpoint=$checkpoint:$task_mlp

export EXPERIMENT_TASK_BATCH_SIZE=256
task_mlp=$(sbatch --chdir=$BASE --array=0-2 $BASE/tasks/train-batch-size-mlp.sh)
checkpoint=$checkpoint:$task_mlp

export EXPERIMENT_TASK_BATCH_SIZE=1024
task_mlp=$(sbatch --chdir=$BASE --array=0-2 $BASE/tasks/train-batch-size-mlp.sh)
checkpoint=$checkpoint:$task_mlp

export EXPERIMENT_TASK_BATCH_SIZE=4096
task_mlp=$(sbatch --chdir=$BASE --array=0-2 $BASE/tasks/train-batch-size-mlp.sh)
checkpoint=$checkpoint:$task_mlp

# Check if the models have coverage at 95% CR.
export EXPERIMENT_TASK_COVERAGE=0.95

export EXPERIMENT_TASK_BATCH_SIZE=64
checkpoint=$checkpoint:$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-batch-size.sh)

export EXPERIMENT_TASK_BATCH_SIZE=256
checkpoint=$checkpoint:$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-batch-size.sh)

export EXPERIMENT_TASK_BATCH_SIZE=1024
checkpoint=$checkpoint:$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-batch-size.sh)

export EXPERIMENT_TASK_BATCH_SIZE=4096
checkpoint=$checkpoint:$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-batch-size.sh)

# Summarize the results.
sbatch --dependency=afterok:$checkpoint --chdir=$BASE $BASE/tasks/summarize-batch-size.sh > /dev/null

### Stage 2. Coverage table for various architectures
################################################################################

### Train the models of th ecoverage table.
export EXPERIMENT_TASK_BATCH_SIZE=$EXPERIMENT_BATCH_SIZE
export EXPERIMENT_TASK_LEARNING_RATE=0.0001

# MLP
export EXPERIMENT_TASK_BATCHNORM=0
marginalized=$(sbatch --chdir=$BASE --array=0-2 $BASE/tasks/train-coverage-mlp-marginalized.sh)
not_marginalized=$(sbatch --chdir=$BASE --array=0-2 $BASE/tasks/train-coverage-mlp-not-marginalized.sh)
task_train_mlp=$marginalized:$not_marginalized

# MLP-BN
export EXPERIMENT_TASK_BATCHNORM=1
marginalized=$(sbatch --chdir=$BASE --array=0-2 $BASE/tasks/train-coverage-mlp-marginalized.sh)
not_marginalized=$(sbatch --chdir=$BASE --array=0-2 $BASE/tasks/train-coverage-mlp-not-marginalized.sh)
task_train_mlp_bn=$marginalized:$not_marginalized

# ResNet-18
export EXPERIMENT_TASK_BATCHNORM=0
marginalized=$(sbatch --chdir=$BASE --array=0-2 $BASE/tasks/train-coverage-resnet-18-marginalized.sh)
not_marginalized=$(sbatch --chdir=$BASE --array=0-2 $BASE/tasks/train-coverage-resnet-18-not-marginalized.sh)
task_train_resnet_18=$marginalized:$not_marginalized

# ResNet-18-BN
export EXPERIMENT_TASK_BATCHNORM=1
marginalized=$(sbatch --chdir=$BASE --array=0-2 $BASE/tasks/train-coverage-resnet-18-marginalized.sh)
not_marginalized=$(sbatch --chdir=$BASE --array=0-2 $BASE/tasks/train-coverage-resnet-18-not-marginalized.sh)
task_train_resnet_18_bn=$marginalized:$not_marginalized

# ResNet-50
export EXPERIMENT_TASK_BATCHNORM=0
marginalized=$(sbatch --partition=quadro --chdir=$BASE --array=0-2 $BASE/tasks/train-coverage-resnet-50-marginalized.sh)
not_marginalized=$(sbatch --partition=quadro --chdir=$BASE --array=0-2 $BASE/tasks/train-coverage-resnet-50-not-marginalized.sh)
task_train_resnet_50=$marginalized:$not_marginalized

# ResNet-50-BN
export EXPERIMENT_TASK_BATCHNORM=1
marginalized=$(sbatch --partition=quadro --chdir=$BASE --array=0-2 $BASE/tasks/train-coverage-resnet-50-marginalized.sh)
not_marginalized=$(sbatch --partition=quadro --chdir=$BASE --array=0-2 $BASE/tasks/train-coverage-resnet-50-not-marginalized.sh)
task_train_resnet_50_bn=$marginalized:$not_marginalized

### Compute the coverage of the models.
checkpoint_training=$task_train_mlp:$task_train_mlp_bn:$task_train_resnet_18:$task_train_resnet_18_bn:$task_train_resnet_50:$task_train_resnet_50_bn
checkpoint=1



# MLP
export EXPERIMENT_TASK_COVERAGE=0.997
# Bayesian
marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-mlp-marginalized.sh)
not_marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-mlp-not-marginalized.sh)
checkpoint=$checkpoint:$marginalized:$not_marginalized
# Frequentist
marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-frequentist-mlp-marginalized.sh)
not_marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-frequentist-mlp-not-marginalized.sh)
checkpoint=$checkpoint:$marginalized:$not_marginalized

export EXPERIMENT_TASK_COVERAGE=0.95
# Bayesian
marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-mlp-marginalized.sh)
not_marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-mlp-not-marginalized.sh)
checkpoint=$checkpoint:$marginalized:$not_marginalized
# Frequentist
marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-frequentist-mlp-marginalized.sh)
not_marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-frequentist-mlp-not-marginalized.sh)
checkpoint=$checkpoint:$marginalized:$not_marginalized

export EXPERIMENT_TASK_COVERAGE=0.68
# Bayesian
marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-mlp-marginalized.sh)
not_marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-mlp-not-marginalized.sh)
checkpoint=$checkpoint:$marginalized:$not_marginalized
# Frequentist
marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-frequentist-mlp-marginalized.sh)
not_marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-frequentist-mlp-not-marginalized.sh)
checkpoint=$checkpoint:$marginalized:$not_marginalized



# MLP-BN
export EXPERIMENT_TASK_COVERAGE=0.997
# Bayesian
marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-mlp-bn-marginalized.sh)
not_marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-mlp-bn-not-marginalized.sh)
checkpoint=$checkpoint:$marginalized:$not_marginalized
# Frequentist
marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-frequentist-mlp-bn-marginalized.sh)
not_marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-frequentist-mlp-bn-not-marginalized.sh)
checkpoint=$checkpoint:$marginalized:$not_marginalized

export EXPERIMENT_TASK_COVERAGE=0.95
# Bayesian
marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-mlp-bn-marginalized.sh)
not_marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-mlp-bn-not-marginalized.sh)
checkpoint=$checkpoint:$marginalized:$not_marginalized
# Frequentist
marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-frequentist-mlp-bn-marginalized.sh)
not_marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-frequentist-mlp-bn-not-marginalized.sh)
checkpoint=$checkpoint:$marginalized:$not_marginalized

export EXPERIMENT_TASK_COVERAGE=0.68
# Bayesian
marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-mlp-bn-marginalized.sh)
not_marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-mlp-bn-not-marginalized.sh)
checkpoint=$checkpoint:$marginalized:$not_marginalized
# Frequentist
marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-frequentist-mlp-bn-marginalized.sh)
not_marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-frequentist-mlp-bn-not-marginalized.sh)
checkpoint=$checkpoint:$marginalized:$not_marginalized



# ResNet-18
export EXPERIMENT_TASK_COVERAGE=0.997
# Bayesian
marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-resnet-18-marginalized.sh)
not_marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-resnet-18-not-marginalized.sh)
checkpoint=$checkpoint:$marginalized:$not_marginalized
# Frequentist
marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-frequentist-resnet-18-marginalized.sh)
not_marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-frequentist-resnet-18-not-marginalized.sh)
checkpoint=$checkpoint:$marginalized:$not_marginalized

export EXPERIMENT_TASK_COVERAGE=0.95
# Bayesian
marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-resnet-18-marginalized.sh)
not_marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-resnet-18-not-marginalized.sh)
checkpoint=$checkpoint:$marginalized:$not_marginalized
# Frequentist
marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-frequentist-resnet-18-marginalized.sh)
not_marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-frequentist-resnet-18-not-marginalized.sh)
checkpoint=$checkpoint:$marginalized:$not_marginalized

export EXPERIMENT_TASK_COVERAGE=0.68
# Bayesian
marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-resnet-18-marginalized.sh)
not_marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-resnet-18-not-marginalized.sh)
checkpoint=$checkpoint:$marginalized:$not_marginalized
# Frequentist
marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-frequentist-resnet-18-marginalized.sh)
not_marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-frequentist-resnet-18-not-marginalized.sh)
checkpoint=$checkpoint:$marginalized:$not_marginalized



# ResNet-18-BN
export EXPERIMENT_TASK_COVERAGE=0.997
# Bayesian
marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-resnet-18-bn-marginalized.sh)
not_marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-resnet-18-bn-not-marginalized.sh)
checkpoint=$checkpoint:$marginalized:$not_marginalized
# Frequentist
marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-frequentist-resnet-18-bn-marginalized.sh)
not_marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-frequentist-resnet-18-bn-not-marginalized.sh)
checkpoint=$checkpoint:$marginalized:$not_marginalized

export EXPERIMENT_TASK_COVERAGE=0.95
# Bayesian
marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-resnet-18-bn-marginalized.sh)
not_marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-resnet-18-bn-not-marginalized.sh)
checkpoint=$checkpoint:$marginalized:$not_marginalized
# Frequentist
marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-frequentist-resnet-18-bn-marginalized.sh)
not_marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-frequentist-resnet-18-bn-not-marginalized.sh)
checkpoint=$checkpoint:$marginalized:$not_marginalized

# Bayesian
export EXPERIMENT_TASK_COVERAGE=0.68
marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-resnet-18-bn-marginalized.sh)
not_marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-resnet-18-bn-not-marginalized.sh)
checkpoint=$checkpoint:$marginalized:$not_marginalized
# Frequentist
marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-frequentist-resnet-18-bn-marginalized.sh)
not_marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-frequentist-resnet-18-bn-not-marginalized.sh)
checkpoint=$checkpoint:$marginalized:$not_marginalized



# ResNet-50
export EXPERIMENT_TASK_COVERAGE=0.997
# Bayesian
marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-resnet-50-marginalized.sh)
not_marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-resnet-50-not-marginalized.sh)
checkpoint=$checkpoint:$marginalized:$not_marginalized
# Frequentist
marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-frequentist-resnet-50-marginalized.sh)
not_marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-frequentist-resnet-50-not-marginalized.sh)
checkpoint=$checkpoint:$marginalized:$not_marginalized

export EXPERIMENT_TASK_COVERAGE=0.95
# Bayesian
marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-resnet-50-marginalized.sh)
not_marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-resnet-50-not-marginalized.sh)
checkpoint=$checkpoint:$marginalized:$not_marginalized
# Frequentist
marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-frequentist-resnet-50-marginalized.sh)
not_marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-frequentist-resnet-50-not-marginalized.sh)
checkpoint=$checkpoint:$marginalized:$not_marginalized

export EXPERIMENT_TASK_COVERAGE=0.68
# Bayesian
marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-resnet-50-marginalized.sh)
not_marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-resnet-50-not-marginalized.sh)
checkpoint=$checkpoint:$marginalized:$not_marginalized
# Frequentist
marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-frequentist-resnet-50-marginalized.sh)
not_marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-frequentist-resnet-50-not-marginalized.sh)
checkpoint=$checkpoint:$marginalized:$not_marginalized



# ResNet-50-BN
export EXPERIMENT_TASK_COVERAGE=0.997
# Bayesian
marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-resnet-50-bn-marginalized.sh)
not_marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-resnet-50-bn-not-marginalized.sh)
checkpoint=$checkpoint:$marginalized:$not_marginalized
# Frequentist
marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-frequentist-resnet-50-bn-marginalized.sh)
not_marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-frequentist-resnet-50-bn-not-marginalized.sh)
checkpoint=$checkpoint:$marginalized:$not_marginalized

export EXPERIMENT_TASK_COVERAGE=0.95
# Bayesian
marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-resnet-50-bn-marginalized.sh)
not_marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-resnet-50-bn-not-marginalized.sh)
checkpoint=$checkpoint:$marginalized:$not_marginalized
# Frequentist
marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-frequentist-resnet-50-bn-marginalized.sh)
not_marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-frequentist-resnet-50-bn-not-marginalized.sh)
checkpoint=$checkpoint:$marginalized:$not_marginalized

export EXPERIMENT_TASK_COVERAGE=0.68
# Bayesian
marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-resnet-50-bn-marginalized.sh)
not_marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-resnet-50-bn-not-marginalized.sh)
checkpoint=$checkpoint:$marginalized:$not_marginalized
# Frequentist
marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-frequentist-resnet-50-bn-marginalized.sh)
not_marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-frequentist-resnet-50-bn-not-marginalized.sh)
checkpoint=$checkpoint:$marginalized:$not_marginalized

# Summary, generate the coverage table.
sbatch --chdir=$BASE --dependency=afterok:$checkpoint $BASE/tasks/summarize-coverage.sh > /dev/null
checkpoint=1

# Compute the coverage for credible regions which includes our bias-term.

# MLP
export EXPERIMENT_TASK_COVERAGE=0.997
export EXPERIMENT_TASK_CR_BIAS=0.002
marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-mlp-marginalized-bias.sh)
not_marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-mlp-not-marginalized-bias.sh)
checkpoint=$checkpoint:$marginalized:$not_marginalized
export EXPERIMENT_TASK_COVERAGE=0.95
export EXPERIMENT_TASK_CR_BIAS=0.02
marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-mlp-marginalized-bias.sh)
not_marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-mlp-not-marginalized-bias.sh)
checkpoint=$checkpoint:$marginalized:$not_marginalized
export EXPERIMENT_TASK_COVERAGE=0.68
export EXPERIMENT_TASK_CR_BIAS=0.02
marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-mlp-marginalized-bias.sh)
not_marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-mlp-not-marginalized-bias.sh)
checkpoint=$checkpoint:$marginalized:$not_marginalized

# MLP-BN
export EXPERIMENT_TASK_COVERAGE=0.997
export EXPERIMENT_TASK_CR_BIAS=0.002
marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-mlp-bn-marginalized-bias.sh)
not_marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-mlp-bn-not-marginalized-bias.sh)
checkpoint=$checkpoint:$marginalized:$not_marginalized
export EXPERIMENT_TASK_COVERAGE=0.95
export EXPERIMENT_TASK_CR_BIAS=0.02
marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-mlp-bn-marginalized-bias.sh)
not_marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-mlp-bn-not-marginalized-bias.sh)
checkpoint=$checkpoint:$marginalized:$not_marginalized
export EXPERIMENT_TASK_COVERAGE=0.68
export EXPERIMENT_TASK_CR_BIAS=0.02
marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-mlp-bn-marginalized-bias.sh)
not_marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-mlp-bn-not-marginalized-bias.sh)
checkpoint=$checkpoint:$marginalized:$not_marginalized

# ResNet-18
export EXPERIMENT_TASK_COVERAGE=0.997
export EXPERIMENT_TASK_CR_BIAS=0.002
marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-resnet-18-marginalized-bias.sh)
not_marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-resnet-18-not-marginalized-bias.sh)
checkpoint=$checkpoint:$marginalized:$not_marginalized
export EXPERIMENT_TASK_COVERAGE=0.95
export EXPERIMENT_TASK_CR_BIAS=0.02
marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-resnet-18-marginalized-bias.sh)
not_marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-resnet-18-not-marginalized-bias.sh)
checkpoint=$checkpoint:$marginalized:$not_marginalized
export EXPERIMENT_TASK_COVERAGE=0.68
export EXPERIMENT_TASK_CR_BIAS=0.02
marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-resnet-18-marginalized-bias.sh)
not_marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-resnet-18-not-marginalized-bias.sh)
checkpoint=$checkpoint:$marginalized:$not_marginalized

# ResNet-18-BN
export EXPERIMENT_TASK_COVERAGE=0.997
export EXPERIMENT_TASK_CR_BIAS=0.002
marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-resnet-18-bn-marginalized-bias.sh)
not_marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-resnet-18-bn-not-marginalized-bias.sh)
checkpoint=$checkpoint:$marginalized:$not_marginalized
export EXPERIMENT_TASK_COVERAGE=0.95
export EXPERIMENT_TASK_CR_BIAS=0.02
marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-resnet-18-bn-marginalized-bias.sh)
not_marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-resnet-18-bn-not-marginalized-bias.sh)
checkpoint=$checkpoint:$marginalized:$not_marginalized
export EXPERIMENT_TASK_COVERAGE=0.68
export EXPERIMENT_TASK_CR_BIAS=0.02
marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-resnet-18-bn-marginalized-bias.sh)
not_marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-resnet-18-bn-not-marginalized-bias.sh)
checkpoint=$checkpoint:$marginalized:$not_marginalized

# ResNet-50
export EXPERIMENT_TASK_COVERAGE=0.997
export EXPERIMENT_TASK_CR_BIAS=0.002
marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-resnet-50-marginalized-bias.sh)
not_marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-resnet-50-not-marginalized-bias.sh)
checkpoint=$checkpoint:$marginalized:$not_marginalized
export EXPERIMENT_TASK_COVERAGE=0.95
export EXPERIMENT_TASK_CR_BIAS=0.02
marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-resnet-50-marginalized-bias.sh)
not_marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-resnet-50-not-marginalized-bias.sh)
checkpoint=$checkpoint:$marginalized:$not_marginalized
export EXPERIMENT_TASK_COVERAGE=0.68
export EXPERIMENT_TASK_CR_BIAS=0.02
marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-resnet-50-marginalized-bias.sh)
not_marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-resnet-50-not-marginalized-bias.sh)
checkpoint=$checkpoint:$marginalized:$not_marginalized

# ResNet-50-BN
export EXPERIMENT_TASK_COVERAGE=0.997
export EXPERIMENT_TASK_CR_BIAS=0.002
marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-resnet-50-bn-marginalized-bias.sh)
not_marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-resnet-50-bn-not-marginalized-bias.sh)
checkpoint=$checkpoint:$marginalized:$not_marginalized
export EXPERIMENT_TASK_COVERAGE=0.95
export EXPERIMENT_TASK_CR_BIAS=0.02
marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-resnet-50-bn-marginalized-bias.sh)
not_marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-resnet-50-bn-not-marginalized-bias.sh)
checkpoint=$checkpoint:$marginalized:$not_marginalized
export EXPERIMENT_TASK_COVERAGE=0.68
export EXPERIMENT_TASK_CR_BIAS=0.02
marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-resnet-50-bn-marginalized-bias.sh)
not_marginalized=$(sbatch --array=0-9 --chdir=$BASE $BASE/tasks/coverage-resnet-50-bn-not-marginalized-bias.sh)
checkpoint=$checkpoint:$marginalized:$not_marginalized


# # Summary, generate the coverage table.
sbatch --chdir=$BASE --dependency=afterok:$checkpoint $BASE/tasks/summarize-coverage-with-cr-bias.sh > /dev/null

### Stage 3. Check whether the architectures model proper probability distributions
################################################################################
# ResNet-18
experiment=$(sbatch --chdir=$BASE --array=0-9 $BASE/tasks/integrand-mlp.sh)
checkpoint=$checkpoint:$experiment

# ResNet-18-BN
experiment=$(sbatch --chdir=$BASE --array=0-9 $BASE/tasks/integrand-mlp-bn.sh)
checkpoint=$checkpoint:$experiment

# ResNet-18
experiment=$(sbatch --chdir=$BASE --array=0-9 $BASE/tasks/integrand-resnet-18.sh)
checkpoint=$checkpoint:$experiment

# ResNet-18-BN
experiment=$(sbatch --chdir=$BASE --array=0-9 $BASE/tasks/integrand-resnet-18-bn.sh)
checkpoint=$checkpoint:$experiment

# ResNet-50
experiment=$(sbatch --chdir=$BASE --array=0-9 $BASE/tasks/integrand-resnet-50.sh)
checkpoint=$checkpoint:$experiment

# ResNet-50-BN
experiment=$(sbatch --chdir=$BASE --array=0-9 $BASE/tasks/integrand-resnet-50-bn.sh)
checkpoint=$checkpoint:$experiment

# Summarize the results
sbatch --chdir=$BASE --dependency=afterok:$checkpoint $BASE/tasks/summarize-integrand.sh > /dev/null

### Stage 4. Mode convergence
################################################################################
sbatch --chdir=$BASE --dependency=afterok:$checkpoint_training $BASE/tasks/diagnostic-map-notebook.sh > /dev/null

### Stage 5. Receiver operating characteristic diagnostic
################################################################################
checkpoint=1

# For 10 million simulations
checkpoint=$checkpoint:$(sbatch --array=0-9 --chdir=$BASE --dependency=afterok:$checkpoint_training $BASE/tasks/roc-mlp.sh)
checkpoint=$checkpoint:$(sbatch --array=0-9 --chdir=$BASE --dependency=afterok:$checkpoint_training $BASE/tasks/roc-mlp-bn.sh)
checkpoint=$checkpoint:$(sbatch --array=0-9 --chdir=$BASE --dependency=afterok:$checkpoint_training $BASE/tasks/roc-resnet-18.sh)
checkpoint=$checkpoint:$(sbatch --array=0-9 --chdir=$BASE --dependency=afterok:$checkpoint_training $BASE/tasks/roc-resnet-18-bn.sh)
checkpoint=$checkpoint:$(sbatch --array=0-9 --chdir=$BASE --dependency=afterok:$checkpoint_training $BASE/tasks/roc-resnet-50.sh)
checkpoint=$checkpoint:$(sbatch --array=0-9 --chdir=$BASE --dependency=afterok:$checkpoint_training $BASE/tasks/roc-resnet-50-bn.sh)

sbatch --chdir=$BASE --dependency=afterok:$checkpoint_training:$checkpoint $BASE/tasks/summarize-roc.sh > /dev/null

### Stage 6. Overview on mock simulations
################################################################################
checkpoint=1

# Note, the thresholds should be tuned depending on the observables that have been simulated.

# Compute the posteriors with ABC
export EXPERIMENT_ABC_THRESHOLD="0.3,0.2,0.1"
export EXPERIMENT_MOCK_INDEX=0
checkpoint=$checkpoint:$(sbatch --chdir=$BASE $BASE/tasks/abc.sh)
export EXPERIMENT_MOCK_INDEX=1
checkpoint=$checkpoint:$(sbatch --chdir=$BASE $BASE/tasks/abc.sh)
export EXPERIMENT_MOCK_INDEX=2
checkpoint=$checkpoint:$(sbatch --chdir=$BASE $BASE/tasks/abc.sh)
export EXPERIMENT_MOCK_INDEX=3
checkpoint=$checkpoint:$(sbatch --chdir=$BASE $BASE/tasks/abc.sh)
export EXPERIMENT_MOCK_INDEX=4
checkpoint=$checkpoint:$(sbatch --chdir=$BASE $BASE/tasks/abc.sh)
export EXPERIMENT_MOCK_INDEX=5
checkpoint=$checkpoint:$(sbatch --chdir=$BASE $BASE/tasks/abc.sh)
export EXPERIMENT_MOCK_INDEX=6
checkpoint=$checkpoint:$(sbatch --chdir=$BASE $BASE/tasks/abc.sh)
export EXPERIMENT_MOCK_INDEX=7
checkpoint=$checkpoint:$(sbatch --chdir=$BASE $BASE/tasks/abc.sh)
export EXPERIMENT_MOCK_INDEX=8
checkpoint=$checkpoint:$(sbatch --chdir=$BASE $BASE/tasks/abc.sh)
export EXPERIMENT_MOCK_INDEX=9
checkpoint=$checkpoint:$(sbatch --chdir=$BASE $BASE/tasks/abc.sh)

# Compute the posteriors with the new ABC analysis
export EXPERIMENT_ABC_THRESHOLD="0.001"
export EXPERIMENT_MOCK_INDEX=0
checkpoint=$checkpoint:$(sbatch --chdir=$BASE $BASE/tasks/abc-new.sh)
export EXPERIMENT_MOCK_INDEX=1
checkpoint=$checkpoint:$(sbatch --chdir=$BASE $BASE/tasks/abc-new.sh)
export EXPERIMENT_MOCK_INDEX=2
checkpoint=$checkpoint:$(sbatch --chdir=$BASE $BASE/tasks/abc-new.sh)
export EXPERIMENT_MOCK_INDEX=3
checkpoint=$checkpoint:$(sbatch --chdir=$BASE $BASE/tasks/abc-new.sh)
export EXPERIMENT_MOCK_INDEX=4
checkpoint=$checkpoint:$(sbatch --chdir=$BASE $BASE/tasks/abc-new.sh)
export EXPERIMENT_MOCK_INDEX=5
checkpoint=$checkpoint:$(sbatch --chdir=$BASE $BASE/tasks/abc-new.sh)
export EXPERIMENT_MOCK_INDEX=6
checkpoint=$checkpoint:$(sbatch --chdir=$BASE $BASE/tasks/abc-new.sh)
export EXPERIMENT_MOCK_INDEX=7
checkpoint=$checkpoint:$(sbatch --chdir=$BASE $BASE/tasks/abc-new.sh)
export EXPERIMENT_MOCK_INDEX=8
checkpoint=$checkpoint:$(sbatch --chdir=$BASE $BASE/tasks/abc-new.sh)
export EXPERIMENT_MOCK_INDEX=9
checkpoint=$checkpoint:$(sbatch --chdir=$BASE $BASE/tasks/abc-new.sh)

# Summarize the comparison of the original ABC formulation
sbatch --chdir=$BASE --dependency=afterok:$checkpoint $BASE/tasks/summarize-abc-comparison.sh > /dev/null

# Summarize the comparison of the original ABC formulation
sbatch --chdir=$BASE --dependency=afterok:$checkpoint $BASE/tasks/summarize-abc-comparison-new.sh > /dev/null

# ### Stage 6. GD-1 Inference summary
# ################################################################################
# Compute the ABC posterior.
export EXPERIMENT_ABC_THRESHOLD="0.3,0.2,0.1"
checkpoint=$(sbatch --array=0-99 --chdir=$BASE $BASE/tasks/abc-gd1.sh)

# Compute the ABC posterior.
export EXPERIMENT_ABC_THRESHOLD="0.001" # 0.1%
checkpoint=$(sbatch --array=0-99 --chdir=$BASE $BASE/tasks/abc-gd1-new.sh)

# Check if the GD-1 posteriors are a proper probability density.
checkpoint=$(sbatch --chdir=$BASE --dependency=afterok:$checkpoint:$checkpoint_training $BASE/tasks/integrand-gd1-resnet-18.sh)
checkpoint=$(sbatch --chdir=$BASE --dependency=afterok:$checkpoint:$checkpoint_training $BASE/tasks/integrand-gd1-resnet-18-bn.sh)
checkpoint=$(sbatch --chdir=$BASE --dependency=afterok:$checkpoint:$checkpoint_training $BASE/tasks/integrand-gd1-resnet-50.sh)
checkpoint=$(sbatch --chdir=$BASE --dependency=afterok:$checkpoint:$checkpoint_training $BASE/tasks/integrand-gd1-resnet-50-bn.sh)

Run the notebook.
sbatch --chdir=$BASE --dependency=afterok:$checkpoint $BASE/tasks/summarize-gd1.sh > /dev/null
