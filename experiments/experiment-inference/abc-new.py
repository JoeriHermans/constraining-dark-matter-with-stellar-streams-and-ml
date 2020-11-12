import argparse
import corner
import hypothesis
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import torch
import warnings

warnings.filterwarnings("ignore")


phi = np.load("../experiment-simulations/data/phi-cut.npy")


def main(arguments):
    global phi
    hypothesis.disable_gpu()
    observation = load_observation(arguments).reshape(-1)
    inputs, outputs = load_joint(arguments)
    threshold = arguments.threshold
    if arguments.auto:
        print("Tuning acceptance threshold!")
        threshold = tune(inputs, outputs, observation, arguments.threshold)
    print("Threshold found at:", threshold, " Applying full posterior sampling!")
    posterior_samples = sample(inputs, outputs, observation, threshold)
    if arguments.out is not None:
        np.save(arguments.out + "/samples.npy", np.array(posterior_samples))


def sample(inputs, outputs, observation, threshold):
    posterior_samples = []
    for index in range(len(inputs)):
        density = outputs[index].reshape(-1)
        contrast = density / observation
        variance = np.nanvar(contrast)
        if variance < threshold:
            posterior_samples.append(inputs[index])
    posterior_samples = np.array(posterior_samples)

    return posterior_samples


def tune(inputs, outputs, observation, rate, n=10000, epsilon=0.001):
    # Select a random subsample of the data.
    indices = np.random.randint(0, len(inputs), n)
    inputs = inputs[indices, :]
    outputs = outputs[indices, :]
    # Start the tuning procedure
    threshold = 1.0
    error = float("infinity")
    while np.abs(error) > epsilon:
        posterior_samples = sample(inputs, outputs, observation, threshold)
        emperical_rate = len(posterior_samples) / n
        error = rate - emperical_rate
        print(error)
        threshold += error / 10

    return threshold


def load_observation(arguments):
    observations = np.load(arguments.observations)
    if arguments.average:
        return observations[np.random.randint(0, len(observations))]
    else:
        if observations.ndim > 1:
            return observations[arguments.observation_index]
        else:
            return observations


def load_joint(arguments):
    masses = np.load(arguments.masses).reshape(-1, 1)
    if arguments.ages is not None:
        ages = np.load(arguments.ages).reshape(-1, 1)
        inputs = np.hstack([masses, ages])
    else:
        inputs = masses
    outputs = np.load(arguments.outputs)

    return inputs, outputs


def parse_arguments():
    parser = argparse.ArgumentParser("Approximate Bayesian Computation")
    parser.add_argument("--auto", action="store_true", help="Automatically tune the acceptance threshold to the specified level (default: false).")
    parser.add_argument("--ages", type=str, default=None, help="Path to the stream ages (default: none).")
    parser.add_argument("--average", action="store_true", help="Average the observations (default: false).")
    parser.add_argument("--masses", type=str, default=None, help="Path to the WDM masses (default: none).")
    parser.add_argument("--observation-index", type=int, default=0, help="Index of the mocks to use as observed data (default: 0).")
    parser.add_argument("--observations", type=str, default=None, help="Path to the observations to use (default: none).")
    parser.add_argument("--out", type=str, default=None, help="Path to store the posterior samples (default: none).")
    parser.add_argument("--outputs", type=str, default=None, help="Path to the outputs (default: none).")
    parser.add_argument("--threshold", type=float, default=0.5, help="Rejection threshold (default: 0.5).")
    arguments, _ = parser.parse_known_args()

    return arguments


if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments)
