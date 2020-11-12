#!/usr/bin/env bash
#
# This file:
#
#  - Defines the boostrapping procedure of the main pipeline.
#  - Defines all utility functions for the end-user.
#  - Simulates Slurm whenever the Slurm binaries have not been detected.
#
# Based on a template by BASH3 Boilerplate v2.3.0
# http://bash3boilerplate.sh/#authors
#

function __log () {
    local log_level="${1}"
    local message="${2}"
    shift

    # Define the colors, sorted by log-level.
    local color_debug="\x1b[35m"
    local color_info="\x1b[32m"
    local color_notice="\x1b[34m"
    local color_warning="\x1b[33m"
    local color_error="\x1b[31m"
    local color_critical="\x1b[1;31m"
    local color_alert="\x1b[1;33;41m"
    local color_emergency="\x1b[1;4;5;33;41m"

    local colorvar="color_${log_level}"

    local color="${!colorvar:-${color_error}}"
    local color_reset="\x1b[0m"

    # Show the message.
    local log_line=""
    while IFS=$'\n' read -r log_line; do
        echo -en "$(date -u +"%Y-%m-%d %H:%M:%S UTC") ${color}$(echo "[${log_level}]")${color_reset} ${log_line}" 1>&2
    done <<< "${@:-}"
}

function log_debug        () { __log debug "${@}"; true; }
function log_info         () { __log info "${@}"; true; }
function log_notice       () { __log notice "${@}"; true; }
function log_warning      () { __log warning "${@}"; true; }
function log_error        () { __log error "${@}"; true; }
function log_critical     () { __log critical "${@}"; true; }
function log_alert        () { __log alert "${@}"; true; }
function log_emergency    () { __log emergency "${@}"; }

function keyboard_interrupt_control_c {
    for pid in $(pgrep -P $$)
    do
        kill -9 $pid
    done
    kill -9 $$
}

trap keyboard_interrupt_control_c SIGINT

function current_directory {
    echo "$(cd "$(dirname "$0")" ; pwd -P)"
}

function args_get_directory {
    new_directory=$(dirname $BASH_SOURCE) # Defaults to sourced directory.
    for argument in "$@"
    do
        # Check if the argument contains the 'chdir' key.
        if [[ $argument == *"chdir"* ]]; then
            new_directory="$(cut -d'=' -f2 <<< "$argument")"
            break
        fi
    done

    echo $new_directory
}

function args_get_task_array {
    tasks=0
    # Get the number of tasks, if specified.
    for argument in "$@"
    do
        # Check if the arguments contains the 'array' key.
        if [[ $argument == *"array"* ]]; then
            indices="$(cut -d'=' -f2 <<< "$argument")"
            tasks="$(cut -d'-' -f2 <<< "$indices")"
            break
        fi
    done

    echo $tasks
}

function host_available {
    address=$1
    # TODO Fix error in stderr/stdout.
    output=$(ping -c 3 $address | wc -l)
    if [ $output -ge 3 ]; then
        echo 1
    else
        echo 0
    fi
}

function extract_experiment_identifier {
    pipeline=$1
    identifier=${pipeline%/pipeline.sh}
    identifier=${identifier##*/}
    identifier=${identifier#*-}

    echo $identifier
}

function task_extract_logging {
    line=$(cat $1 | grep "SBATCH --output")
    path="$(cut -d' ' -f3 <<< "$line")"

    echo $(echo $path | cut -c2- | rev | cut -c2- | rev)
}

function task_reformat_output_path {
    output_path="$1"
    task_index="$2"
    delimiter="%a"
    # Check if the output path needs to be reworkred.
    if [[ $output_path == *"$delimiter"* ]]; then
        echo $output_path | sed -e "s/$delimiter/$task_index/g"
    else
        echo $output_path
    fi
}

function compute_local {
    calling_directory=$(pwd)
    new_directory=$(args_get_directory "$@")
    tasks=$(args_get_task_array "$@")
    task_script="${@: -1}"
    output_path=$(task_extract_logging $task_script)
    cd $new_directory
    for task_identifier in $(seq 0 $tasks)
    do
        export SLURM_ARRAY_TASK_ID=$((task_identifier))
        task_output_path=$(task_reformat_output_path $output_path $SLURM_ARRAY_TASK_ID)
        log_info "Executing $task_script with task index $SLURM_ARRAY_TASK_ID/$tasks.\n"
        bash $task_script > "$BASE/$task_output_path"
        exit_code=${PIPESTATUS[0]}
        if [ $exit_code -ne 0 ]; then
            log_error "Program terminated with a non-zero exit code.\n"
            exit $exit_code
        fi
    done
    cd $calling_directory
}

function slurm_available {
    if [ -z $(command -v sbatch) ]; then
        echo 1
    else
        echo 0
    fi
}
