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
    observation = load_observation(arguments)
    summary_observed = summary(observation)
    inputs, outputs = load_joint(arguments)
    observed_length = compute_length(phi, observation) # Also part of the summary statistic.
    posterior_samples = []
    for index in range(len(inputs)):
        density = outputs[index]
        length = compute_length(phi, density)
        if length < observed_length: # Discard shorter streams (implicitely part of the summary).
            continue
        s = summary(density)
        d = distance(s, summary_observed)
        if np.all(d <= arguments.thresholds):
            posterior_samples.append(inputs[index])
    posterior_samples = np.array(posterior_samples)
    np.save(arguments.out + "/samples.npy", np.array(posterior_samples))


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


def distance(summary_a, summary_b):
    a = np.fabs(summary_a[1] - summary_b[1])
    b = np.fabs(summary_a[2] - summary_b[2])
    c = np.fabs(summary_a[3] - summary_b[3])

    return np.array([a, b, c])


def summary(data):
    global phi
    power = scipy.signal.csd(data, data, scaling="spectrum", fs=1./(phi[1] - phi[0]), nperseg=len(phi))[1].real
    power = np.sqrt(power * (phi[-1] - phi[0]))

    return power.reshape(-1)


def compute_length(phi, density, threshold=0.2):
    mean_den = np.mean(density)
    if True in (dd < 0. for dd in density):
        length = 0.
    else:
        den_diff = density - threshold * mean_den
        if True in (d1 < 0. for d1 in den_diff):
            length = np.fabs(phi[np.argmin(np.fabs(den_diff))] - (phi[0]))
        else:
            length = np.abs(max(phi) - min(phi)) # Full length

    return length


def parse_arguments():
    parser = argparse.ArgumentParser("Approximate Bayesian Computation")
    parser.add_argument("--ages", type=str, default=None, help="Path to the stream ages (default: none).")
    parser.add_argument("--average", action="store_true", help="Average the observations (default: false).")
    parser.add_argument("--masses", type=str, default=None, help="Path to the WDM masses (default: none).")
    parser.add_argument("--observation-index", type=int, default=0, help="Index of the mocks to use as observed data (default: 0).")
    parser.add_argument("--observations", type=str, default=None, help="Path to the observations to use (default: none).")
    parser.add_argument("--out", type=str, default=None, help="Path to store the posterior samples (default: none).")
    parser.add_argument("--outputs", type=str, default=None, help="Path to the outputs (default: none).")
    parser.add_argument("--thresholds", type=str, default="0.3,0.2,0.1", help="Rejection thresholds (default: 0.3, 0.2, 0.1).")
    arguments, _ = parser.parse_known_args()
    arguments.thresholds = [float(t) for t in arguments.thresholds.split(',')]

    return arguments


if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments)
