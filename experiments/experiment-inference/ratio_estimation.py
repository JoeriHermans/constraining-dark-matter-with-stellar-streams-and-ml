import hypothesis
import torch

from hypothesis.auto.training import LikelihoodToEvidenceRatioEstimatorTrainer as Trainer
from hypothesis.nn import MultiLayeredPerceptron as MLP
from hypothesis.nn import ResNetHead
from hypothesis.nn.amortized_ratio_estimation import BaseLikelihoodToEvidenceRatioEstimator
from hypothesis.nn.amortized_ratio_estimation import BaseRatioEstimator
from hypothesis.nn.amortized_ratio_estimation import LikelihoodToEvidenceRatioEstimatorMLP
from hypothesis.nn.util import compute_dimensionality

### Defaults ###################################################################

activation = torch.nn.SELU
batchnorm = True
resnet_depth = 101
dropout = float(0.0)
shape_inputs = (1,)
shape_outputs = (62,) # 39 for smaller range, 62 for wide.
trunk = [512] * 3

### Models #####################################################################



class RatioEstimator(BaseRatioEstimator):

    def __init__(self,
        activation=activation,
        batchnorm=True,
        depth=resnet_depth,
        dim_inputs=1,
        dropout=dropout,
        normalize_inputs=False):
        super(RatioEstimator, self).__init__()
        shape = shape_outputs
        self.output_elements = shape[0]
        # Create the ResNet head
        self.head = ResNetHead(
            depth=depth,
            activation=activation,
            batchnorm=batchnorm,
            dilate=True,
            channels=1,
            shape_xs=shape_outputs)
        self.dimensionality = self.head.embedding_dimensionality()
        # Ratio estimator trunk
        dimensionality = self.dimensionality + dim_inputs
        self.trunk = MLP(
            activation=activation,
            dropout=dropout,
            layers=trunk,
            shape_xs=(dimensionality,),
            shape_ys=(1,),
            transform_output=None)
        if normalize_inputs:
            raise NotImplementedError
        else:
            self._normalizer = self._normalize_identity

    def _normalize_identity(self, inputs):
        return inputs

    def forward(self, inputs, outputs):
        log_ratios = self.log_ratio(inputs=inputs, outputs=outputs)

        return log_ratios.sigmoid(), log_ratios

    def log_ratio(self, inputs, outputs):
        inputs = self._normalizer(inputs)
        outputs = outputs.view(-1, 1, self.output_elements) # Reshape outputs
        z = self.head(outputs).view(-1, self.dimensionality)
        features = torch.cat([inputs, z], dim=1)

        return self.trunk(features)



class LSTMRatioEstimator(BaseRatioEstimator):

    def __init__(self,
        activation=activation,
        dim_inputs=1,
        dropout=dropout,
        hidden_size=128,
        normalize_inputs=False):
        super(LSTMRatioEstimator, self).__init__()
        # MLP to initialize the hidden and cell state of the LSTM based on the inputs.
        self.conditioner_mlp_hidden = MLP(
            activation=activation,
            dropout=dropout,
            layers=[hidden_size * 2],
            shape_xs=(dim_inputs,),
            shape_ys=(hidden_size,),
            transform_output=None)
        self.conditioner_mlp_cell = MLP(
            activation=activation,
            dropout=dropout,
            layers=[hidden_size * 2],
            shape_xs=(dim_inputs,),
            shape_ys=(hidden_size,),
            transform_output=None)
        # MLP which transforms the final hidden state into a log-ratio.
        self.mlp = MLP(
            activation=activation,
            dropout=dropout,
            layers=[hidden_size * 2] * 3,
            shape_xs=(hidden_size + dim_inputs,),
            shape_ys=(1,),
            transform_output=None)
        # LSTM which compresses the inputs and outputs to a latent code.
        self.num_stacked = 1
        self.lstm = torch.nn.LSTM(
            input_size=1,                # Expected features of the observable
            hidden_size=hidden_size,     # Hidden state
            num_layers = self.num_stacked,
            batch_first=True,
            dropout=dropout)
        # Check if the inputs have to be normalized.
        if normalize_inputs:
            raise NotImplementedError
        else:
            self._normalizer = self._normalize_identity

    def _normalize_identity(self, inputs):
        return inputs

    def log_ratio(self, inputs, outputs):
        inputs = self._normalizer(inputs)
        outputs = outputs.view(-1, shape_outputs[0], 1)
        # Initialize the hidden state
        initial_hidden_state = self.conditioner_mlp_hidden(inputs).unsqueeze(0).repeat(self.num_stacked, 1, 1)
        initial_cell_state = self.conditioner_mlp_cell(inputs).unsqueeze(0).repeat(self.num_stacked, 1, 1)
        h, _ = self.lstm(outputs, (initial_hidden_state, initial_cell_state))
        h = h[:, 1, :]
        z = torch.cat([h, inputs], dim=1)
        log_ratios = self.mlp(z)

        return log_ratios



class MLPRatioEstimator(LikelihoodToEvidenceRatioEstimatorMLP):

    def __init__(self,
        activation=activation,
        batchnorm=batchnorm,
        dropout=dropout,
        dim_inputs=1,
        normalize_inputs=True):
        super(MLPRatioEstimator, self).__init__(
            shape_inputs=(dim_inputs,),
            shape_outputs=shape_outputs,
            activation=activation,
            layers=trunk,
            dropout=dropout)
        dim = dim_inputs + shape_outputs[0]
        if normalize_inputs:
            self.bn_inputs = torch.nn.BatchNorm1d(dim_inputs)
        if batchnorm:
            self.bn_outputs = torch.nn.BatchNorm1d(shape_outputs[0])
        self.batchnorm = batchnorm
        self.normalize_inputs = normalize_inputs

    def log_ratio(self, inputs, outputs):
        if self.normalize_inputs:
            inputs = self.bn_inputs(inputs)
        if self.batchnorm:
            outputs = self.bn_outputs(outputs)

        return super().log_ratio(inputs=inputs, outputs=outputs)



class SingleRatioEstimator(RatioEstimator):

    def __init__(self,
        activation=activation,
        batchnorm=batchnorm,
        depth=resnet_depth,
        dropout=dropout):
        super(SingleRatioEstimator, self).__init__(
            activation=activation,
            batchnorm=batchnorm,
            depth=depth,
            dim_inputs=1,
            dropout=dropout)



class DoubleRatioEstimator(RatioEstimator):

    def __init__(self,
        activation=activation,
        batchnorm=batchnorm,
        depth=resnet_depth,
        dropout=dropout):
        super(DoubleRatioEstimator, self).__init__(
            activation=activation,
            batchnorm=batchnorm,
            depth=depth,
            dim_inputs=2,
            dropout=dropout)


class Classifier(torch.nn.Module):

    def __init__(self, activation=activation,
        batchnorm=batchnorm,
        depth=resnet_depth,
        dropout=dropout):
        super(Classifier, self).__init__()
        # Create the ResNet head
        self.head = ResNetHead(
            depth=depth,
            activation=activation,
            batchnorm=batchnorm,
            dilate=False,
            channels=1,
            shape_xs=shape_outputs)
        self.dimensionality = self.head.embedding_dimensionality()
        # Ratio estimator trunk
        dimensionality = self.dimensionality
        self.trunk = MLP(
            activation=activation,
            dropout=dropout,
            layers=trunk,
            shape_xs=(dimensionality,),
            shape_ys=(1,),
            transform_output=None)

    def forward(self, x):
        x = x.view(-1, 1, 39)
        z = self.head(x).view(-1, self.dimensionality)
        z = self.trunk(z)

        return z.sigmoid()
