import glob
import hypothesis
import torch

from hypothesis.auto.training import LikelihoodToEvidenceRatioEstimatorTrainer as Trainer
from hypothesis.nn import MultiLayeredPerceptron as MLP
from hypothesis.nn import ResNetHead
from hypothesis.nn.amortized_ratio_estimation import BaseLikelihoodToEvidenceRatioEstimator
from hypothesis.nn.amortized_ratio_estimation import BaseRatioEstimator
from hypothesis.nn.amortized_ratio_estimation import RatioEstimatorEnsemble
from hypothesis.nn.amortized_ratio_estimation import LikelihoodToEvidenceRatioEstimatorMLP
from hypothesis.nn.util import compute_dimensionality

from hypothesis.benchmark.normal import Simulator
from hypothesis.util.data import SimulatorDataset

### Defaults ###################################################################

activation = torch.nn.SELU
batchnorm = True
dropout = float(0.0)
shape_inputs = (1,)
shape_outputs = (1,) # 39 for smaller range, 62 for wide.
trunk = [128] * 3

### Models #####################################################################


def allocate_prior():
    return torch.distributions.uniform.Uniform(-25, 25)

def load_ratio_estimator(path):
    model = RatioEstimator()
    weights = torch.load(path)
    model.load_state_dict(weights)
    
    return model.eval()


def load_estimator(query):
    paths = glob.glob(query)
    if len(paths) == 1:
        model = load_ratio_estimator(paths[0])
    else:
        models = [load_ratio_estimator(path) for path in paths]
        model = RatioEstimatorEnsemble(models)
    
    return model



class RatioEstimator(LikelihoodToEvidenceRatioEstimatorMLP):

    def __init__(self,
        activation=activation,
        batchnorm=batchnorm,
        dropout=dropout,
        dim_inputs=1,
        normalize_inputs=True):
        super(RatioEstimator, self).__init__(
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
    
    
class JointTrain(SimulatorDataset):
    
    def __init__(self):
        simulations = 1000000
        prior = allocate_prior()
        simulator = Simulator()
        super(JointTrain, self).__init__(simulator, prior)
        
        
class JointTest(SimulatorDataset):
    
    def __init__(self):
        simulations = 100000
        prior = allocate_prior()
        simulator = Simulator()
        super(JointTest, self).__init__(simulator, prior)