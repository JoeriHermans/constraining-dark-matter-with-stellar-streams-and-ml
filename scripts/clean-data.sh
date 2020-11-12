#!/bin/sh

# A utility script to clean the data generated in post-processing.
# Useful for post-processing development.

DATADIR=experiments/experiment-simulations/data

rm -f $DATADIR/*.npy
rm -f $DATADIR/mock/*/*-cut.npy
rm -f $DATADIR/mock/*/*-noised.npy
rm -f $DATADIR/mock/*/*-normalized.npy
rm -f $DATADIR/nominal/*/*-cut.npy
rm -f $DATADIR/nominal/*/*-noised.npy
rm -f $DATADIR/nominal/*/*-normalized.npy
rm -f $DATADIR/test/*.npy
rm -f $DATADIR/train/*.npy
