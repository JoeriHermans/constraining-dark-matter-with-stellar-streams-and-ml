import argparse
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from util import load_observed_gd1
from numpy.polynomial import Polynomial
from scipy import interpolate


def main(arguments):
    phi = np.load(arguments.phi)
    # Load the observed data.
    _, error = load_observed_gd1(arguments.observed, phi, degree=arguments.degree)
    # Load the densities
    densities = np.load(arguments.data)
    resolution = len(densities[0])
    noised_densities = []
    for _ in range(arguments.replicate):
        for index in range(len(densities)):
            density = densities[index]
            density_with_error = density + np.random.normal(size=resolution) * error
            noised_densities.append(density_with_error)
    noised_densities = np.vstack(noised_densities)
    np.save(arguments.out, noised_densities)


def parse_arguments():
    parser = argparse.ArgumentParser("Simulations")
    parser.add_argument("--data", type=str, default=None, help="Path of the data directory (default: none).")
    parser.add_argument("--degree", type=int, default=1, help="Degree of the polynomial to fit (default: 1).")
    parser.add_argument("--observed", type=str, default=None, help="Path to the observed data (default: none).")
    parser.add_argument("--out", type=str, default=None, help="Path of the output files (default: none).")
    parser.add_argument("--phi", type=str, default=None, help="Path to the data fit (default: none).")
    parser.add_argument("--replicate", type=int, default=1, help="Replication factor. Noise n additional observations given a single stream (default: 1).")
    arguments, _ = parser.parse_known_args()

    return arguments


if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments)
