#!/bin/sh

# Yes, this is hacky but Anaconda is a shitshow atm.
conda remove --name stellar-stream-inference --all
conda create -n stellar-stream-inference python=3.7
conda activate stellar-stream-inference
conda install -c conda-forge gsl
conda install -c conda-forge galpy=1.5
pip install -r requirements.txt
pip install https://github.com/montefiore-ai/hypothesis.git
