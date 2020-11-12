import argparse
import hypothesis
import numpy as np
import torch

from simulators import GD1StreamSimulator
from simulators import WDMSubhaloSimulator
from util import allocate_prior_stream_age as PriorAge
from util import allocate_prior_wdm_mass as PriorMass



@torch.no_grad()
def main(arguments):
    hypothesis.disable_gpu()
    prior_age = PriorAge()
    prior_mass = PriorMass()
    if arguments.mocks is not None:
        ages = torch.linspace(prior_age.low, prior_age.high, arguments.mocks)
        masses = torch.linspace(prior_mass.low, prior_mass.high, arguments.mocks)
        ages = ages[arguments.mock_index].view(1, 1)
        masses = masses[arguments.mock_index].view(1, 1)
    elif arguments.nominal:
        ages = torch.linspace(prior_age.low, prior_age.high, 10)
        masses = torch.linspace(prior_mass.low, prior_mass.high, 10)
        ages = ages[arguments.mock_index].view(1, 1)
        ages = np.vstack([ages for _ in range(arguments.size)])
        masses = masses[arguments.mock_index].view(1, 1)
        masses = np.vstack([masses for _ in range(arguments.size)])
    else:
        ages = prior_age.sample().view(1, 1)
        masses = prior_mass.sample((arguments.size,))
    # Simulate the stream
    simulator = GD1StreamSimulator()
    stream = simulator(ages)[0]
    # Simulate the densities with the subhalo impacts
    simulator = WDMSubhaloSimulator(stream, record_impacts=True)
    results = simulator(masses)
    # Prepare the results
    densities = []
    phis = []
    impacts = []
    for result in results:
        impacts.append(np.array(result[0]).reshape(1, 1))
        phis.append(np.array(result[1]).reshape(1, -1))
        densities.append(np.array(result[2]).reshape(1, -1))
    ages = ages.repeat(arguments.size, 1).view(-1, 1).numpy()
    masses = masses.numpy()
    densities = np.vstack(densities).reshape(arguments.size, -1)
    # Store the results
    np.save(arguments.out + "/ages.npy", ages)
    np.save(arguments.out + "/masses.npy", masses)
    np.save(arguments.out + "/phi.npy", phis)
    np.save(arguments.out + "/impacts.npy", impacts)
    np.save(arguments.out + "/densities.npy", densities)


def parse_arguments():
    parser = argparse.ArgumentParser("Simulations")
    parser.add_argument("--mock-index", type=int, default=None, help="Index of the nominal value for the mock simulation (default: none).")
    parser.add_argument("--mocks", type=int, default=None, help="Resolution of the mock simulations (default: none).")
    parser.add_argument("--nominal", action="store_true", help="Use a single random nominal value for the simulations (default: false).")
    parser.add_argument("--out", type=str, default=None, help="Path of the output files (default: none).")
    parser.add_argument("--size", type=int, default=1, help="Number of simulations per simulated stream (default: 1).")
    arguments, _ = parser.parse_known_args()

    return arguments


if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments)
