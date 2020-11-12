#!/bin/sh

# A utility script to compress the generated data and results.

mkdir -p experiments/experiment-simulations/data
zip -9 simulations.zip -r experiments/experiment-simulations/data

mkdir -p experiments/experiment-inference/out
zip -9 results.zip -r experiments/experiment-inference/out
