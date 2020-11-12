import argparse
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from util import load_observed_gd1
from scipy import integrate, interpolate



def main():
    phi = np.load("data/phi-cut.npy")
    density, error = load_observed_gd1("data/GD1-stream-track-density.dat", phi, degree=1)
    resolution = len(phi)
    # Process the clean observation
    np.save("data/observed.npy", density)
    # Process the noisy observations
    n = 1000
    noised_densities = np.zeros((n, resolution))
    for index in range(n):
        noised_densities[index, :] = density + np.random.normal(size=resolution) * error
    np.save("data/observed-noised.npy", noised_densities)


if __name__ == "__main__":
    main()
