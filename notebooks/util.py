import glob
import hypothesis
import numpy as np
import os
import requests
import torch


from hypothesis.nn.amortized_ratio_estimation import RatioEstimatorEnsemble
from ratio_estimation import DoubleRatioEstimator
from ratio_estimation import MLPRatioEstimator
from ratio_estimation import RatioEstimator
from ratio_estimation import SingleRatioEstimator
from ratio_estimation import resnet_depth
from torch.distributions.uniform import Uniform



def load(model, marginalized=True):
    model = model.lower()
    if marginalized:
        t = "marginalized"
    else:
        t = "not-marginalized"
    mapping = {
        "all": "models/4096/" + t + "/selu/*-batchnorm-0*/best-model.th",
        "all-bn":  "models/4096/" + t + "/selu/*-batchnorm-1*/best-model.th",
        "mlp": "models/4096/" + t + "/selu/ratio-estimator-mlp-*-batchnorm-0*/best-model.th",
        "mlp-bn": "models/4096/" + t + "/selu/ratio-estimator-mlp-*-batchnorm-1*/best-model.th",
        "resnet-18": "models/4096/" + t + "/selu/ratio-estimator-resnet-18-*-batchnorm-0*/best-model.th",
        "resnet-18-bn": "models/4096/" + t + "/selu/ratio-estimator-resnet-18-*-batchnorm-1*/best-model.th",
        "resnet-50": "models/4096/" + t + "/selu/ratio-estimator-resnet-50-*-batchnorm-0*/best-model.th",
        "resnet-50-bn": "models/4096/" + t + "/selu/ratio-estimator-resnet-50-*-batchnorm-1*/best-model.th"}
    if model not in mapping.keys():
        raise ValueError("Unknown model!")
    
    return load_ratio_estimator(mapping[model])


@torch.no_grad()
def MarginalizedAgePrior():
    lower = torch.tensor(1).float()
    lower = lower.to(hypothesis.accelerator)
    upper = torch.tensor(50.01).float()
    upper = upper.to(hypothesis.accelerator)

    return Uniform(lower, upper)


@torch.no_grad()
def Prior():
    lower = torch.tensor([1,     3]).float()
    lower = lower.to(hypothesis.accelerator)
    upper = torch.tensor([50.01, 7]).float()
    upper = upper.to(hypothesis.accelerator)

    return Uniform(lower, upper)


def load_activation(activation):
    activations = {
        "elu": torch.nn.ELU,
        "leakyrelu": torch.nn.LeakyReLU,
        "prelu": torch.nn.PReLU,
        "relu": torch.nn.ReLU,
        "prelu": torch.nn.PReLU,
        "selu": torch.nn.SELU,
        "tanh": torch.nn.Tanh}
    if activation not in activations.keys():
        raise ValueError("Activation", activation, "is not available.")

    return activations[activation]


def load_ratio_estimator(path, normalize_inputs=False):
    if '*' in path:
        estimator = load_ensemble_ratio_estimator(path, normalize_inputs)
    else:
        estimator = load_single_ratio_estimator(path, normalize_inputs)
    # Move to the default Hypothesis accelerator
    estimator.to(hypothesis.accelerator)
    estimator.eval()

    return estimator


def load_ensemble_ratio_estimator(query, normalize_inputs=False):
    paths = glob.glob(query)
    estimators = []
    for path in paths:
        estimators.append(load_single_ratio_estimator(path, normalize_inputs))
    if(len(estimators) == 0):
        raise ValueError("No ratio estimators found! Verify the specified path.")

    return RatioEstimatorEnsemble(estimators)


def load_single_ratio_estimator(path, normalize_inputs=False):
    # Check if the path exists.
    if not os.path.exists(path):
        raise ValueError("Path " + path + " does not exist.")
    weights = torch.load(path)
    dirname = os.path.dirname(path)
    segments = path.split('/')
    # Check what activation to use
    activation = load_activation(path.split('/')[-3])
    segments = dirname.split('-')
    # Extract the dropout setting
    index = segments.index("dropout")
    dropout = float(segments[index + 1])
    # Extract the batch normalization setting
    index = segments.index("batchnorm")
    batchnorm = bool(int(segments[index + 1]))
    # Check if it's the marginalized model.
    if "not-marginalized" in path:
        inputs_dim = 2
    else:
        inputs_dim = 1
    # Extract the ResNet depth configuration
    try:
        index = segments.index("resnet")
        depth = int(segments[index + 1])
        mlp = False
    except:
        mlp = True
    # Load the MLP
    if not mlp:
        # Allocate the ratio estimator
        ratio_estimator = RatioEstimator(
            activation=activation,
            batchnorm=batchnorm,
            depth=depth,
            dim_inputs=inputs_dim,
            dropout=dropout,
            normalize_inputs=normalize_inputs)
        # Backward compatibility
        if "_normalizer.weight" in weights.keys():
            weights["bn_inputs.weight"] = weights["_normalizer.weight"]
            del weights["_normalizer.weight"]
            weights["bn_inputs.bias"] = weights["_normalizer.bias"]
            del weights["_normalizer.bias"]
            weights["bn_inputs.running_mean"] = weights["_normalizer.running_mean"]
            del weights["_normalizer.running_mean"]
            weights["bn_inputs.running_var"] = weights["_normalizer.running_var"]
            del weights["_normalizer.running_var"]
            weights["bn_inputs.num_batches_tracked"] = weights["_normalizer.num_batches_tracked"]
            del weights["_normalizer.num_batches_tracked"]
    else:
        ratio_estimator = MLPRatioEstimator(
            activation=activation,
            batchnorm=batchnorm,
            dim_inputs=inputs_dim,
            dropout=dropout,
            normalize_inputs=normalize_inputs)
    ratio_estimator.load_state_dict(weights)
    ratio_estimator = ratio_estimator.eval()

    return ratio_estimator


def download(id, destination="."):
    r"""Adapted from:

    https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
    """
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
