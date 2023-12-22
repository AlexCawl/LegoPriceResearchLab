from typing import Callable

from analysis.api import RegressionModelApi
from analysis.random_forest_regression.RandomForestRegressionModel import RandomForestRegressionModel

RandomForestRegressionModelFactory: Callable[[], RegressionModelApi] = lambda: RandomForestRegressionModel()
