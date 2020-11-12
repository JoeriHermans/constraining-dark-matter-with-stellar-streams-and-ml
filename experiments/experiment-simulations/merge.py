import argparse
import glob
import numpy as np
import os
import torch



def main(arguments):
    merge_ages(arguments)
    merge_densities(arguments)
    merge_impacts(arguments)
    merge_masses(arguments)


def merge_ages(arguments):
    ages = merge(arguments.data + "/block-*/ages.npy")
    np.save(arguments.out + "/ages.npy", ages)


def merge_densities(arguments):
    densities = merge(arguments.data + "/block-*/densities.npy")
    np.save(arguments.out + "/densities.npy", densities)


def merge_impacts(arguments):
    impacts = merge(arguments.data + "/block-*/impacts.npy")
    impacts = impacts.reshape(-1, 1)
    np.save(arguments.out + "/impacts.npy", impacts)


def merge_masses(arguments):
    masses = merge(arguments.data + "/block-*/masses.npy")
    masses = masses.reshape(-1, 1)
    np.save(arguments.out + "/masses.npy", masses)


def merge(query):
    paths = glob.glob(query)
    paths.sort()
    data = []
    for path in paths:
        data.append(np.load(path))
    data = np.vstack(data)

    return data


def parse_arguments():
    parser = argparse.ArgumentParser("Simulations")
    parser.add_argument("--out", type=str, default=None, help="Path of the output files (default: none).")
    parser.add_argument("--data", type=str, default=None, help="Path of the data directory (default: none).")
    arguments, _ = parser.parse_known_args()

    return arguments


if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments)
