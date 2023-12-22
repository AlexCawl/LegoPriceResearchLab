from typing import List, Callable

from analysis.linear_regression.LeastSquaresModel import LeastSquaresLinearRegressionModel
from analysis.linear_regression.RidgeModel import RidgeLinearRegressionModel
from analysis.api import RegressionModelApi

LeastSquaresLinearRegressionModelFactory: Callable[[], RegressionModelApi] = lambda: LeastSquaresLinearRegressionModel()
RidgeLinearRegressionModelFactory: Callable[[], RegressionModelApi] = lambda: RidgeLinearRegressionModel()
