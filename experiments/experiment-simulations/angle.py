import argparse
import glob
import numpy as np
import os
import torch



def main(arguments):
    original = np.load(arguments.input)
    processed = original[0].reshape(-1)
    np.save(arguments.output, processed)


def parse_arguments():
    parser = argparse.ArgumentParser("Simulations")
    parser.add_argument("--input", type=str, default=None, help="Path of the input angle (default: none).")
    parser.add_argument("--output", type=str, default=None, help="Path of the output angle (default: none).")
    arguments, _ = parser.parse_known_args()

    return arguments


if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments)
