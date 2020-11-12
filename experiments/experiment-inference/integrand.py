import argparse
import hypothesis
import numpy as np
import os
import torch
import warnings

from hypothesis.diagnostic import DensityDiagnostic
from scipy.integrate import quad
from scipy.integrate import nquad
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
    result = np.array(result)
    if arguments.out is not None:
        np.save(arguments.out, result)
    else:
        print(np.mean(result), r"±", np.std(result))


def load_densities(arguments):
    # Check if the specified data path is a directory.
    if os.path.isdir(arguments.data):
        return np.load(arguments.data + "/density-contrasts-cut-noised.npy")
    else:
        return np.load(arguments.data)


def diagnose_not_marginalized(arguments):
    # Load the ratio estimator
    ratio_estimator = load_ratio_estimator(arguments.model, normalize_inputs=False)
    # Prepare the diagnostic
    prior = Prior()
    space = [[1, 50], [3, 7]]
    densities = load_densities(arguments)
    densities = torch.from_numpy(densities).float()
    diagnostic = DensityDiagnostic(space)
    # Iterate through all tests
    for _ in range(arguments.tests):
        # Fetch the density to test
        density = densities[np.random.randint(0, len(densities))].view(1, -1)
        density = density.to(hypothesis.accelerator)
        # Define the pdf function for integration
        def pdf(mass, age):
            mass = torch.tensor(mass).view(1, 1).float()
            age = torch.tensor(age).view(1, 1).float()
            mass = mass.to(hypothesis.accelerator)
            age = age.to(hypothesis.accelerator)
            inputs = torch.cat([mass, age], dim=1)
            log_posterior = prior.log_prob(inputs).sum() + ratio_estimator.log_ratio(inputs=inputs, outputs=density)

            return log_posterior.exp().item()
        # Compute the test
        diagnostic.test(pdf)

    return np.array(diagnostic.areas)



def diagnose_marginalized(arguments):
    # Load the ratio estimator
    ratio_estimator = load_ratio_estimator(arguments.model)
    # Prepare the diagnostic
    prior = MarginalizedAgePrior()
    space = [[1, 50]]
    densities = load_densities(arguments)
    densities = torch.from_numpy(densities).float()
    diagnostic = DensityDiagnostic(space)
    # Iterate through all tests
    for _ in range(arguments.tests):
        # Fetch the density to test
        density = densities[np.random.randint(0, len(densities))].view(1, -1)
        density = density.to(hypothesis.accelerator)
        # Define the pdf function for integration
        def pdf(mass):
            mass = torch.tensor(mass).view(1, 1).float()
            mass = mass.to(hypothesis.accelerator)
            log_posterior = prior.log_prob(mass).item() + ratio_estimator.log_ratio(inputs=mass, outputs=density)

            return log_posterior.exp().item()
        # Compute the test
        diagnostic.test(pdf)

    return np.array(diagnostic.areas)



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
    main(arguments)
