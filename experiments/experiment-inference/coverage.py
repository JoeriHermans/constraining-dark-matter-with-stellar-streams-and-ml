import argparse
import hypothesis
import matplotlib.pyplot as plt
import numpy as np
import torch

from hypothesis.stat import highest_density_level
from util import MarginalizedAgePrior
from util import Prior
from scipy.stats import chi2
from util import load_ratio_estimator



@torch.no_grad()
def main(arguments):
    # Load the ratio estimator
    ratio_estimator = load_ratio_estimator(arguments.model)
    # Load the densities
    densities = torch.from_numpy(np.load(arguments.data + "/density-contrasts-cut-noised.npy")).float()
    # Check if the non-marginalized model has been specified
    resolution = arguments.resolution
    if "not-marginalized" in arguments.model:
        prior = Prior()
        degrees_of_freedom = 2
        masses = torch.from_numpy(np.load(arguments.data + "/masses.npy")).view(-1, 1).float()
        ages = torch.from_numpy(np.load(arguments.data + "/ages.npy")).view(-1, 1).float()
        nominals = torch.cat([masses, ages], dim=1)
        masses = torch.linspace(prior.low[0], prior.high[0] - 0.01, resolution).view(-1, 1)
        masses = masses.to(hypothesis.accelerator)
        ages = torch.linspace(prior.low[1], prior.high[1] - 0.01, resolution).view(-1, 1)
        ages = ages.to(hypothesis.accelerator)
        grid_masses, grid_ages = torch.meshgrid(masses.view(-1), ages.view(-1))
        inputs = torch.cat([grid_masses.reshape(-1,1), grid_ages.reshape(-1, 1)], dim=1)
    else:
        prior = MarginalizedAgePrior()
        degrees_of_freedom = 1
        # Prepare inputs
        nominals = torch.from_numpy(np.load(arguments.data + "/masses.npy")).view(-1, 1).float()
        masses = torch.linspace(prior.low, prior.high - 0.01, resolution).view(-1, 1)
        masses = masses.to(hypothesis.accelerator)
        inputs = masses
    # Prepare the diagnostic
    nominals = nominals.to(hypothesis.accelerator)
    densities = densities.to(hypothesis.accelerator)
    results = []
    indices = np.random.randint(0, len(densities), size=arguments.n)
    for index in indices:
        # Get current density and nominal value
        nominal = nominals[index].view(1, -1)
        density = densities[index].view(1, -1)
        # Prepare the outputs
        outputs = density.repeat(len(inputs), 1)
        # Check if we have to compute Bayesian credible regions
        if not arguments.frequentist:
            # Compute Bayesian credible region
            # Compute the posterior pdf
            log_ratios = ratio_estimator.log_ratio(inputs=inputs, outputs=outputs)
            log_pdf = log_ratios # Uniform prior
            pdf = log_pdf.exp()
            norms = (inputs - nominal).norm(dim=1).cpu().numpy()
            nominal_index = np.argmin(norms)
            nominal_pdf = pdf[nominal_index].item()
            level = highest_density_level(pdf, arguments.level, bias=arguments.bias)
            if nominal_pdf >= level:
                covered = True
            else:
                covered = False
        else:
            # Compute Frequentist confidence interval based on Wilks' theorem.
            # Compute the maximum theta
            log_ratios = ratio_estimator.log_ratio(inputs=inputs, outputs=outputs)
            max_ratio = log_ratios[log_ratios.argmax()]
            test_statistic = -2 * (log_ratios - max_ratio)
            test_statistic -= test_statistic.min()
            x = chi2.isf(1 - arguments.level, df=degrees_of_freedom)
            norms = (inputs - nominal).norm(dim=1).cpu().numpy()
            nominal_index = np.argmin(norms)
            if test_statistic[nominal_index].item() <= x:
                covered = True
            else:
                covered = False
        results.append(covered)
    # Save the results of the diagnostic.
    np.save(arguments.out, results)


def parse_arguments():
    parser = argparse.ArgumentParser("Emperical coverage estimation")
    parser.add_argument("--bias", type=float, default=0.0, help="Bias-term to for high-density-level estimation (default: 0.0)")
    parser.add_argument("--data", type=str, default=None, help="Path of the data directory (default: none).")
    parser.add_argument("--frequentist", action="store_true", help="Flag to compute frequentist confidence intervals instead of Bayesian credible regions (default: false).")
    parser.add_argument("--level", type=float, default=0.95, help="Credible level (default: 0.997 - 3 sigma.)")
    parser.add_argument("--model", type=str, default=None, help="Will load all ratio estimators matching this path query (default: none).")
    parser.add_argument("--n", type=int, default=1000, help="Number of times to repeat the experiment (default: 1000).")
    parser.add_argument("--out", type=str, default=None, help="Path of the output file (default: none).")
    parser.add_argument("--resolution", type=int, default=100, help="Resolution for every variable (default: 100).")
    arguments, _ = parser.parse_known_args()

    return arguments


if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments)
