#!/usr/bin/env bash
#
# This file:
#
# - Runs the specified experiments and the associated pipelines.
#
# Usage:
#
# ./run.sh
#

# Collect all experiments.
experiments=$(ls $PROJECT_BASE/experiments/experiment-*/pipeline.sh | sort)

for pipeline in $experiments
do
    identifier=$(extract_experiment_identifier $pipeline)
    # Check if the experiment needs to be executed.
    if [[ "$PROJECT_RUN_EXPERIMENTS" == "all" || \
              " ${PROJECT_RUN_EXPERIMENTS[@]} " =~ " ${identifier} " ]]; then
        source $pipeline
    fi
done
