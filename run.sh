#!/usr/bin/env -S bash -i
#
# This file:
#
# - Executes my will.
#

### Project parameters
################################################################################
# export SBATCH_PARTITION="" # Set if desired.
export PROJECT_DEBUG=0
export PROJECT_DEPENDENCY=1
export PROJECT_FORCE_RERUN=0
export PROJECT_NAME="stellar-stream-inference"
export PROJECT_NO_PYTHON_BYTECODE=1
export PROJECT_RUN_EXPERIMENTS="all"
export PROJECT_SLURM_USERNAME=$(whoami)

### Argument parsing and script initialization
################################################################################
shopt -s expand_aliases
source lib/bootstrap.sh

# Check if the user changed the project name.
if [ "$PROJECT_NAME" == "changeme" ]; then
    log_emergency "PROJECT_NAME has not been changed, please read and edit run.sh accordingly.\n"
    exit 1
fi

function install_conda_environment {
    log_info "Installing the Anaconda environment.\n"
    conda env create --name "$PROJECT_NAME"
}

function update_conda_environment_requirements {
    conda activate "$PROJECT_NAME"
    conda env update -f environment.yml --prune
    conda deactivate
}

function usage () {
    echo "Options:"
    echo "  -a [slurm job id]           Execute the project after the specified Slurm job id completed."
    echo "  -b                          Enables the generation of Python bytecode."
    echo "  -d                          Enable debugging (default: 0)."
    echo "  -e [experiments]            Run the specified experiments (comma seperated experiment identifiers, default: all)"
    echo "  -f                          Force a complete rerun of the specified experiments, even if completed."
    echo "  -h                          Shows this usage message."
    echo "  -i                          Install or update the Anaconda environment."
    echo "  -p [partition]              Desired Slurm partition to run on (default: all)."
    echo "  -s [slurm username]         Set the Slurm username (if this is different from your accountname)."
    echo "  -u                          Uninstall the Conda environment attached to this project."
}


# Parse the program arguments
while getopts a:bde:fhiup:s:v: opt
do
    case $opt in
        a)
            export PROJECT_DEPENDENCY="$OPTARG"
            ;;
        b)
            export PROJECT_NO_PYTHON_BYTECODE=0
            ;;
        d)
            export PROJECT_DEBUG=1
            ;;
        e)
            export PROJECT_RUN_EXPERIMENTS="$OPTARG"
            ;;
        f)
            export PROJECT_FORCE_RERUN=1
            ;;
        h)
            usage
            exit 0
            ;;
        i)
            eval "$(conda shell.bash hook)"
            environments=$(conda env list | awk '{print $1}')
            if [[ ! $environments == *"$PROJECT_NAME"* ]]; then
                install_conda_environment
            fi
            log_info "Updating the Anaconda environment.\n"
            update_conda_environment_requirements
            exit 0
            ;;
        s)
            export PROJECT_SLURM_USERNAME="$OPTARG"
            ;;
        u)
            eval "$(conda shell.bash hook)"
            log_info "Uninstalling `$PROJECT_NAME` Anaconda environment.\n"
            conda deactivate > /dev/null
            conda remove --name "$PROJECT_NAME" --all --yes
            exit 0
            ;;
        p)
            export SBATCH_PARTITION="$OPTARG"
            ;;
        *)
            echo "Invalid option: -$OPTARG" >&2
            exit 2
            ;;
    esac
done

# Prepare Anaconda.
eval "$(conda shell.bash hook)"

# Check if the Anaconda envirionment has been installed.
environments=$(conda env list | awk '{print $1}')
if [[ ! $environments == *"$PROJECT_NAME"* ]]; then
    install_conda_environment
fi

conda activate "$PROJECT_NAME"

# Perpare the experiments variable.
export PROJECT_RUN_EXPERIMENTS=${PROJECT_RUN_EXPERIMENTS//','/' '}

# Set global variables.
export PROJECT_BASE=$(current_directory)

# Check if Slurm is available.
if [ -z $(command -v sbatch) ]; then
    log_warning "Slurm scheduler not available, using the local compute backend.\n"
    alias sbatch=compute_local
else
    log_info "Slurm detected, submitting parallel jobs.\n"
fi

# Debugging options.
if [ $PROJECT_DEBUG -ne 0 ]; then
    export CUDA_LAUNCH_BLOCKING=1
else
    export CUDA_LAUNCH_BLOCKING=0
fi

# Check if Python bytecode needs to be generated.
if [ $PROJECT_NO_PYTHON_BYTECODE -ne 0 ]; then
    export PYTHONDONTWRITEBYTECODE=1
else
    # Remove all compiled Python bytecode.
    find $PROJECT_BASE -name "*.pyc" -exec rm -f {} \;
    find $PROJECT_BASE -name "__pycache__" -exec rm -rf {} \;
fi

### Experiments
################################################################################
log_info "Starting the experiments.\n"
source $PROJECT_BASE/experiments/run.sh

### Cleaning up
################################################################################
conda deactivate # Deactivate the current Anaconda environment.
