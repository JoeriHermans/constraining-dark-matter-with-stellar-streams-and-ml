#!/usr/bin/env bash -i
#

### Pipeline definitions
################################################################################
export PIPELINE_NAME="Simulations"

log_info "Starting experiment: $PIPELINE_NAME\n"

### Script and pipeline initialization
################################################################################
export BASE=$(dirname $BASH_SOURCE) # Set the path of the current directory.

# Create the logging directory.
mkdir -p $BASE/logging
# Create the data directory.
export DATADIR=$BASE/data
mkdir -p $DATADIR

### Experimental parameters
################################################################################
export EXPERIMENT_POLYNOMIAL_DEGREE=1                # Degree of the polynomial detrending
export EXPERIMENT_MAX_SUBHALO_IMPACTS=64             # Maximum number of subhalo impacts
export EXPERIMENT_STREAMS_TRAIN=20000                # Total number of streams to generate for training
export EXPERIMENT_STREAMS_TEST=1000                  # Total number of streams to generate for testing
export EXPERIMENT_STREAM_SAMPLES_TEST=100            # Number of observations to draw from every stream for testing
export EXPERIMENT_STREAM_SAMPLES_TRAIN=100           # Number of observations to draw from every stream for training
export EXPERIMENT_NOMINALS=10                        # Simulate 10 nominal value blocks
export EXPERIMENT_MOCKS=10                           # Simulate 10 equally spaces nominal values along every axis.
export EXPERIMENT_PHI_LOW=-34                        # Lower angle (phi)
export EXPERIMENT_PHI_HIGH=10                        # Upper angle (phi), should be -4 if you want the small scale data.
export EXPERIMENT_NOISY_REPLICATION=5                # When noising, noise n instances given 1 stream.

## Supported by observation:
# [-90. // 30.]

## Supported by the simulation:
# [-34 // 24.39634139]

### Stage 1. Simulate subhalo tidal effects
############################################################################
# # Training
mkdir -p $DATADIR/train
stage_simulate_train=$(sbatch --chdir=$BASE \
    --array=0-$(($EXPERIMENT_STREAMS_TRAIN - 1)) \
    $BASE/tasks/simulate-train.sh)

# # Testing
mkdir -p $DATADIR/test
stage_simulate_test=$(sbatch --chdir=$BASE \
    --array=0-$(($EXPERIMENT_STREAMS_TEST - 1)) \
    $BASE/tasks/simulate-test.sh)

# Nominal
mkdir -p $DATADIR/nominal
stage_simulate_nominal=$(sbatch --chdir=$BASE \
    --array=0-$(($EXPERIMENT_NOMINALS - 1)) \
    $BASE/tasks/simulate-nominal.sh)

# # Mock
mkdir -p $DATADIR/mock
stage_simulate_mock=$(sbatch --chdir=$BASE \
    --array=0-$(($EXPERIMENT_MOCKS - 1)) \
    $BASE/tasks/simulate-mock.sh)

### Stage 2. Merge simulated blocks
################################################################################
# Training
stage_merge_train=$(sbatch --chdir=$BASE \
    --dependency=afterok:$stage_simulate_train \
    $BASE/tasks/merge-train.sh)

# Testing
stage_merge_test=$(sbatch --chdir=$BASE \
    --dependency=afterok:$stage_simulate_test \
    $BASE/tasks/merge-test.sh)

### Stage 3. Copy and clean angle descriptions
################################################################################
stage_angle=$(sbatch --chdir=$BASE \
    --dependency=afterok:$stage_merge_train:$stage_merge_test \
    $BASE/tasks/process-angle.sh)

### Stage 4. Cut simulated densities
################################################################################
# Training
stage_cut_train=$(sbatch --chdir=$BASE \
    --dependency=afterok:$stage_angle \
    $BASE/tasks/cut-train.sh)

# Testing
stage_cut_test=$(sbatch --chdir=$BASE \
    --dependency=afterok:$stage_angle \
    $BASE/tasks/cut-test.sh)

# Nominal
stage_cut_nominal=$(sbatch --chdir=$BASE \
    --dependency=afterok:$stage_angle:$stage_simulate_nominal \
    $BASE/tasks/cut-nominal.sh)

# Mock
stage_cut_mock=$(sbatch --chdir=$BASE \
    --dependency=afterok:$stage_angle:$stage_simulate_mock \
    $BASE/tasks/cut-mock.sh)

### Stage 5. Normalize the densities
################################################################################
# Training
stage_normalize_train=$(sbatch --chdir=$BASE \
    --dependency=afterok:$stage_cut_train \
    $BASE/tasks/normalize-train.sh)

# Testing
stage_normalize_test=$(sbatch --chdir=$BASE \
    --dependency=afterok:$stage_cut_test \
    $BASE/tasks/normalize-test.sh)

# Nominal
stage_normalize_nominal=$(sbatch --chdir=$BASE \
    --dependency=afterok:$stage_cut_nominal \
    $BASE/tasks/normalize-nominal.sh)

# Mock
stage_normalize_mock=$(sbatch --chdir=$BASE \
    --dependency=afterok:$stage_cut_mock \
    $BASE/tasks/normalize-mock.sh)

### Stage 6. Add noise based on the observed noise levels
################################################################################
# Training
stage_noise_train=$(sbatch --chdir=$BASE \
    --dependency=afterok:$stage_normalize_train \
    $BASE/tasks/noise-train.sh)

# Testing
stage_noise_test=$(sbatch --chdir=$BASE \
    --dependency=afterok:$stage_normalize_test \
    $BASE/tasks/noise-test.sh)

# Nominal
stage_noise_nominal=$(sbatch --chdir=$BASE \
    --dependency=afterok:$stage_normalize_nominal \
    $BASE/tasks/noise-nominal.sh)

# Mock
stage_noise_mock=$(sbatch --chdir=$BASE \
    --dependency=afterok:$stage_normalize_nominal \
    $BASE/tasks/noise-mock.sh)

### Stage 7. Prepare the observed GD-1 stream for inference
################################################################################
stage_prepare_observation=$(sbatch --chdir=$BASE \
    --dependency=afterok:$stage_noise_train:$stage_noise_test \
    $BASE/tasks/prepare-observation.sh)
