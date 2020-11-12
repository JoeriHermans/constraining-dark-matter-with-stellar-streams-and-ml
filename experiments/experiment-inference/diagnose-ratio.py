import argparse
import hypothesis
import numpy as np
import pickle
import torch

from ratio_estimation import Classifier
from sklearn import svm, datasets
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import label_binarize
from util import load_ratio_estimator


@torch.no_grad()
def main(arguments):
    marginal_samples = np.load("../experiment-simulations/data/train/density-contrasts-cut-noised.npy")
    model = load_ratio_estimator(arguments.model)
    result = {}
    result["auc"] = []
    result["fpr"] = []
    result["tpr"] = []
    for _ in range(arguments.repeat):
        nominal, likelihood_samples = load_experiment(arguments.experiment)
        reweighted_samples = reweigh_samples(marginal_samples, likelihood_samples, nominal, model, batch_size=1024)
        likelihood_samples = torch.tensor(likelihood_samples)
        reweighted_samples = torch.tensor(reweighted_samples)
        x = torch.cat([reweighted_samples, likelihood_samples], dim=0)
        n = len(likelihood_samples)
        ones = torch.ones(n).view(-1, 1)
        zeros = torch.zeros(n).view(-1, 1)
        y = torch.cat([ones, zeros], dim=0)
        x = x.numpy()
        y = y.numpy()
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, shuffle=True)
        classifier = MLPClassifier(early_stopping=True, hidden_layer_sizes=(128, 128,))
        classifier.fit(x_train, y_train.reshape(-1))
        y_score = classifier.predict_proba(x_test)
        fpr, tpr, _ = roc_curve(y_test, y_score[:, 1])
        roc_auc = auc(fpr, tpr)
        result["auc"].append(roc_auc)
        result["fpr"].append(fpr)
        result["tpr"].append(tpr)
    # Save the results.
    if arguments.out is not None:
        with open(arguments.out, "wb") as fd:
            pickle.dump(result, fd)


@torch.no_grad()
def reweigh_samples(marginal_samples, likelihood_samples, nominal, model, batch_size=1):
    weights = np.zeros(len(marginal_samples))
    inputs = torch.from_numpy(nominal).view(1, -1).float()
    ins = inputs.to(hypothesis.accelerator)
    inputs = ins.repeat(batch_size, 1)
    index = 0
    n = len(marginal_samples)
    with tqdm(total=n) as pbar:
        while index < n:
            if (n - index) < batch_size:
                batch_size = n - index
                inputs = ins.repeat(batch_size, 1)
            density = torch.from_numpy(marginal_samples[index:index + batch_size,:]).view(batch_size, -1).float()
            density = density.to(hypothesis.accelerator)
            weight = model.log_ratio(inputs=inputs, outputs=density).exp().view(-1).cpu().numpy()
            weights[index:index + batch_size] = weight
            index += batch_size
            pbar.update(batch_size)
    weights /= np.sum(weights)
    sampled_indices = np.random.choice(np.arange(len(weights)), size=len(likelihood_samples), replace=False, p=weights)
    reweighted_samples = []
    for index in sampled_indices:
        reweighted_samples.append(marginal_samples[index].reshape(1, -1))
    reweighted_samples = np.vstack(reweighted_samples).astype(np.float32)

    return reweighted_samples


def load_experiment(index):
    suffix = str(index).zfill(5)
    base = "../experiment-simulations/data/nominal/block-" + suffix
    likelihood_samples = np.load(base + "/density-contrasts-cut-noised.npy").astype(np.float32)
    nominal = np.array([np.load(base + "/masses.npy")[0]]).reshape(1, -1).astype(np.float32)

    return nominal, likelihood_samples


def parse_arguments():
    parser = argparse.ArgumentParser("Quality of the ratio approximation")
    parser.add_argument("--experiment", type=int, default=0, help="Experiment index (default: 0).")
    parser.add_argument("--model", type=str, default=None, help="Query path to the model weights (default: none).")
    parser.add_argument("--out", type=str, default=None, help="Path of the output file (default: none).")
    parser.add_argument("--repeat", type=int, default=10, help="Repitition of the training and subsampling of the data (default: 10).")
    arguments, _ = parser.parse_known_args()

    return arguments


if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments)
