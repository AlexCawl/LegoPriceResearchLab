from typing import Callable

from analysis.api import RegressionModelApi
from analysis.gradient_boosting_regression.GradientBoostingRegressionModel import GradientBoostingRegressionModel

GradientBoostingRegressionModelFactory: Callable[[], RegressionModelApi] = lambda: GradientBoostingRegressionModel()
