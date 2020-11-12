import argparse
import numpy as np



def main(arguments):
    impacts = np.load(arguments.input)
    has_impacts = impacts > 0
    np.save(arguments.output, has_impacts)


def parse_arguments():
    parser = argparse.ArgumentParser("Impacts preparation")
    parser.add_argument("--input", type=str, default=None, help="Path to the inputs file (default: none)")
    parser.add_argument("--output", type=str, default=None, help="Path to the outputs file (default: none)")
    arguments, _ = parser.parse_known_args()

    return arguments


if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments)
