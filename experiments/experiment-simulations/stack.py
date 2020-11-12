import argparse
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import torch



def main(arguments):
    inputs = np.load(arguments.input)
    outputs = []
    for _ in range(arguments.replicate):
        outputs.append(inputs)
    outputs = np.vstack(outputs)
    np.save(arguments.output, outputs)


def parse_arguments():
    parser = argparse.ArgumentParser("Simulations")
    parser.add_argument("--input", type=str, default=None, help="Input file (default: none).")
    parser.add_argument("--output", type=str, default=None, help="Output path (default: none).")
    parser.add_argument("--replicate", type=int, default=1, help="Replication factor. Noise n additional observations given a single stream (default: 1).")
    arguments, _ = parser.parse_known_args()

    return arguments


if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments)
