from typing import Callable

from analysis.api import RegressionModelApi
from analysis.neural_network_regression.NeuralNetworkRegresionModel import NeuralNetworkRegressionModel

NeuralNetworkRegressionModelFactory: Callable[[], RegressionModelApi] = lambda: NeuralNetworkRegressionModel()
