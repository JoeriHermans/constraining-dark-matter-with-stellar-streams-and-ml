import argparse
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import torch



def main(arguments):
    observed = np.genfromtxt(arguments.observed, names=True)
    phi_observed = observed["phi1mid"]
    data = np.load(arguments.data)
    phi = np.load(arguments.phi)
    phi_low = arguments.low
    phi_high = arguments.high
    indices = (phi >= phi_low) & (phi <= phi_high)
    phi_cut = phi[indices]
    data_cut = data[:, indices]
    np.save(arguments.out + "/phi-cut.npy", phi_cut)
    np.save(arguments.out + "/densities-cut.npy", data_cut)


def parse_arguments():
    parser = argparse.ArgumentParser("Simulations")
    parser.add_argument("--low", type=float, default=-34.0, help="Low-end of the linear angle (default: -34).")
    parser.add_argument("--high", type=float, default=-4.0, help="High-end of the linear angle (default: -4).")
    parser.add_argument("--data", type=str, default=None, help="Path of the data directory (default: none).")
    parser.add_argument("--observed", type=str, default=None, help="Path to the observed data (default: none).")
    parser.add_argument("--out", type=str, default=None, help="Path of the output files (default: none).")
    parser.add_argument("--phi", type=str, default=None, help="Path to the file describing the angle (default: none).")
    arguments, _ = parser.parse_known_args()

    return arguments


if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments)
