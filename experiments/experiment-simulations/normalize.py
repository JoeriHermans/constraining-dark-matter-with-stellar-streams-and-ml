import argparse
import glob
import numpy as np
import os
import torch



def main(arguments):
    if arguments.contrast:
        processed_data = process_contrast(arguments)
    elif arguments.density:
        processed_data = process_density(arguments)
    else:
        processed_data = None
    if processed_data is not None:
        np.save(arguments.out, processed_data)


def process_contrast(arguments):
    data = np.load(arguments.data)
    phi = np.load(arguments.phi)
    for index in range(len(data)):
        original_density = np.array(data[index]) # Clone data
        original_density /= np.mean(original_density) # Center to 1
        trend = np.polyfit(phi, original_density, deg=arguments.degree)
        fitted = np.poly1d(trend)(phi)
        density_contrast = original_density - fitted + 1
        density_contrast /= np.mean(density_contrast)
        data[index] = density_contrast

    return data


def process_density(arguments):
    data = np.load(arguments.data)
    for index in range(len(data)):
        original_density = np.array(data[index]) # Clone data
        data[index] = original_density / np.mean(original_density)

    return data


def parse_arguments():
    parser = argparse.ArgumentParser("Simulations")
    parser.add_argument("--contrast", action="store_true", help="Generate the density contrast (default: none).")
    parser.add_argument("--data", type=str, default=None, help="Path of the data directory (default: none).")
    parser.add_argument("--degree", type=int, default=3, help="Degree of the polynomial which is fit to the density (default: 3).")
    parser.add_argument("--density", action="store_true", help="Generate the normalized and unit-less density (default: none).")
    parser.add_argument("--out", type=str, default=None, help="Path of the output files (default: none).")
    parser.add_argument("--phi", type=str, default=None, help="Path to the linear angle file (default: none).")
    arguments, _ = parser.parse_known_args()

    return arguments


if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments)
