r"""Training script for conditional ratio estimators."""

import argparse
import hypothesis
import numpy as np
import torch

from hypothesis.nn.amortized_ratio_estimation import ConservativeLikelihoodToEvidenceCriterion
from hypothesis.nn.amortized_ratio_estimation import LikelihoodToEvidenceCriterion
from ratio_estimation import LSTMRatioEstimator
from ratio_estimation import MLPRatioEstimator
from ratio_estimation import RatioEstimator
from ratio_estimation import Trainer
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset
from util import load_activation



tloss = np.log(4.0)
def main(arguments):
    # Allocate the training and testing dataset.
    dataset_test = allocate_dataset_test(arguments)
    dataset_train = allocate_dataset_train(arguments)
    # Allocate the ratio estimator.
    estimator = allocate_estimator(arguments)
    estimator = estimator.to(hypothesis.accelerator)
    # Allocate the optimizer.
    optimizer = torch.optim.AdamW(
        estimator.parameters(),
        amsgrad=arguments.amsgrad,
        lr=arguments.lr,
        weight_decay=arguments.weight_decay)
    # Prepare the criterion.
    criterion = ConservativeLikelihoodToEvidenceCriterion(
        batch_size=arguments.batch_size,
        beta=arguments.beta,
        estimator=estimator,
        logits=False)
    # Allocate the learning rate scheduler.
    lr_scheduler = None
    # Allocate the trainer.
    trainer = Trainer(
        accelerator=hypothesis.accelerator,
        batch_size=arguments.batch_size,
        criterion=criterion,
        dataset_test=dataset_test,
        dataset_train=dataset_train,
        epochs=arguments.epochs,
        estimator=estimator,
        lr_scheduler_epoch=lr_scheduler,
        optimizer=optimizer,
        workers=arguments.workers)
    # Callbacks
    def report_test_loss(caller):
        trainer = caller
        current_epoch = trainer.current_epoch
        test_loss = trainer.losses_test[-1]
        print("Epoch", current_epoch, ":", test_loss)
    def report_train_loss(caller, index, loss):
        global tloss
        tloss = 0.99 * tloss + 0.01 * loss
        print(tloss)
    trainer.add_event_handler(trainer.events.epoch_complete, report_test_loss)
    # trainer.add_event_handler(trainer.events.batch_complete, report_train_loss)
    # Run the optimization procedure.
    summary = trainer.fit()
    print(summary)
    # Collect the results.
    best_model_weights = summary.best_model()
    final_model_weights = summary.final_model()
    train_losses = summary.train_losses()
    test_losses = summary.test_losses()
    # Save the results.
    np.save(arguments.out + "/losses-train.npy", train_losses)
    np.save(arguments.out + "/losses-test.npy", test_losses)
    torch.save(best_model_weights, arguments.out + "/best-model.th")
    torch.save(final_model_weights, arguments.out + "/model.th")
    summary.save(arguments.out + "/result.summary")


@torch.no_grad()
def allocate_dataset_train(arguments):
    inputs = None

    if arguments.data_train_masses is not None:
        masses = torch.from_numpy(np.load(arguments.data_train_masses)).view(-1, 1).float()
        if arguments.data_train_ages is not None:
            ages = torch.from_numpy(np.load(arguments.data_train_ages)).view(-1, 1).float()
            inputs = torch.cat([masses, ages], dim=1)
        else:
            inputs = masses
    elif arguments.data_train_impacts is not None:
        inputs = torch.from_numpy(np.load(arguments.data_train_impacts)).view(-1, 1).float()

    assert(inputs is not None)
    outputs = torch.from_numpy(np.load(arguments.data_train_outputs)).float()
    if arguments.n_train is not None:
        inputs = inputs[:arguments.n_train]
        outputs = outputs[:arguments.n_train]
    dataset = TensorDataset(inputs, outputs)

    return TensorDataset(inputs, outputs)


@torch.no_grad()
def allocate_dataset_test(arguments):
    inputs = None

    if arguments.data_test_masses is not None:
        masses = torch.from_numpy(np.load(arguments.data_test_masses)).view(-1, 1).float()
        if arguments.data_test_ages is not None:
            ages = torch.from_numpy(np.load(arguments.data_test_ages)).view(-1, 1).float()
            inputs = torch.cat([masses, ages], dim=1)
        else:
            inputs = masses
    elif arguments.data_test_impacts is not None:
        inputs = torch.from_numpy(np.load(arguments.data_test_impacts)).view(-1, 1).float()

    assert(inputs is not None)
    outputs = torch.from_numpy(np.load(arguments.data_test_outputs)).float()
    dataset = TensorDataset(inputs, outputs)

    return TensorDataset(inputs, outputs)


def allocate_estimator(arguments):
    activation = allocate_activation(arguments)
    dropout = arguments.dropout
    normalize_inputs = arguments.normalize_inputs

    # Check if we have 2-dimensional inputs
    if arguments.data_train_ages is not None:
        dim_inputs = 2
    else:
        dim_inputs = 1

    # Check if the MLP ratio estimator needs to be allocated.
    if arguments.mlp:
        estimator = MLPRatioEstimator(
            activation=activation,
            batchnorm=arguments.batchnorm,
            dim_inputs=dim_inputs,
            dropout=dropout,
            normalize_inputs=normalize_inputs)
    elif arguments.lstm:
        estimator = LSTMRatioEstimator(
            activation=activation,
            dim_inputs=dim_inputs,
            dropout=dropout,
            normalize_inputs=normalize_inputs)
    else:
        estimator = RatioEstimator(
            activation=activation,
            batchnorm=arguments.batchnorm,
            depth=arguments.resnet_depth,
            dim_inputs=dim_inputs,
            dropout=dropout,
            normalize_inputs=normalize_inputs)
    # Check if we are able to allocate a data parallel model.
    if torch.cuda.device_count() > 1:
        estimator = torch.nn.DataParallel(estimator)

    return estimator


def allocate_activation(arguments):
    return load_activation(arguments.activation)


def parse_arguments():
    parser = argparse.ArgumentParser("Conditional ratio estimator training")
    parser.add_argument("--activation", type=str, default="relu", help="Activation function (default: relu).")
    parser.add_argument("--amsgrad", action="store_true", help="Use AMSGRAD version of Adam (default: false).")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64).")
    parser.add_argument("--batchnorm", type=int, default=0, help="Batchnorm (default: false).")
    parser.add_argument("--beta", type=float, default=0.0, help="Conservative term (default: 0.0).")
    parser.add_argument("--data-test-ages", type=str, default=None)
    parser.add_argument("--data-test-impacts", type=str, default=None)
    parser.add_argument("--data-test-masses", type=str, default=None)
    parser.add_argument("--data-test-outputs", type=str, default=None)
    parser.add_argument("--data-train-ages", type=str, default=None)
    parser.add_argument("--data-train-impacts", type=str, default=None)
    parser.add_argument("--data-train-masses", type=str, default=None)
    parser.add_argument("--data-train-outputs", type=str, default=None)
    parser.add_argument("--disable-gpu", action="store_true", help="Disable the usage of the GPU, not recommended. (default: false).")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate (default: 0.0).")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs (default: 1).")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001).")
    parser.add_argument("--lstm", action="store_true", help="Train the LSTM-based ratio estimator (default: false).")
    parser.add_argument("--mlp", action="store_true", help="Train the MLP ratio estimator (default: false).")
    parser.add_argument("--n-train", type=int, default=None, help="Number of training samples to select (default: none).")
    parser.add_argument("--normalize-inputs", action="store_true", help="Let the ratio estimator normalize the inputs (default: false).")
    parser.add_argument("--out", type=str, default=None, help="Output directory for the model.")
    parser.add_argument("--resnet-depth", type=int, default=161, help="ResNet depth (default: 161).")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay (default: 0.0).")
    parser.add_argument("--workers", type=int, default=1, help="Number of concurrent data loaders (default: 1).")
    arguments, _ = parser.parse_known_args()
    if arguments.batchnorm > 0:
        arguments.batchnorm = True
    else:
        arguments.batchnorm = False
    if arguments.disable_gpu:
        hypothesis.disable_gpu()

    return arguments


if __name__ == "__main__":
    arguments = parse_arguments()
    print(arguments)
    main(arguments)
