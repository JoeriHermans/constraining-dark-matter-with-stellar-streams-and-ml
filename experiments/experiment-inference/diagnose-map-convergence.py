import argparse
import hypothesis
import numpy as np
import torch
import warnings

from util import MarginalizedAgePrior
from util import Prior
from util import load_ratio_estimator


# Disable warnings produced by integrator
warnings.filterwarnings('ignore')


@torch.no_grad()
def main(arguments):
    # Check if the non-marginailized model needs to be evaluated.
    if "not-marginalized" in arguments.model:
        result = diagnose_not_marginalized(arguments)
    else:
        result = diagnose_marginalized(arguments)
    # Check if the results need to be saved.
    if arguments.out is not None:
        np.save(arguments.out + "/diagnostic-integrand.npy", np.array(result))
    else:
        print(result)


def load_densities(arguments):
    if "densities" in arguments.model:
        if "noisy" in arguments.model:
            densities = np.load(arguments.data + "/densities-cut-noised-normalized.npy")
        else:
            densities = np.load(arguments.data + "/densities-cut-normalized.npy")
    else:
        if "noisy" in arguments.model:
            densities = np.load(arguments.data + "/density-contrasts-cut-noised.npy")
        else:
            densities = np.load(arguments.data + "/density-contrasts-cut.npy")

    return densities


def diagnose_not_marginalized(arguments):
    # Load the ratio estimator
    ratio_estimator = load_ratio_estimator(arguments.model)
    # Prepare the diagnostic
    prior = Prior()
    space = [[1, 3], [7, 50]]
    densities = load_densities(arguments)
    densities = torch.from_numpy(densities).float()
    # TODO Implement diagnostic


def diagnose_marginalized(arguments):
    # Load the ratio estimator
    ratio_estimator = load_ratio_estimator(arguments.model)
    # Prepare the diagnostic
    prior = MarginalizedAgePrior()
    space = [[1, 50]]
    densities = load_densities(arguments)
    densities = torch.from_numpy(densities).float()
    # TODO Implement diagnostic


def parse_arguments():
    parser = argparse.ArgumentParser("Integrand diagnostic")
    parser.add_argument("--data", type=str, default=None, help="Path to the data directory (default: none).")
    parser.add_argument("--model", type=str, default=None, help="Path to the weights of the ratio estimator (default: none).")
    parser.add_argument("--out", type=str, default=None, help="Output directory of the diagnostic result (default: false).")
    parser.add_argument("--tests", type=int, default=1000, help="Number of random tests to compute (default: 1000).")
    arguments, _ = parser.parse_known_args()

    return arguments


if __name__ == "__main__":
    arguments = parse_arguments()
    print(arguments.model)
    main(arguments)
