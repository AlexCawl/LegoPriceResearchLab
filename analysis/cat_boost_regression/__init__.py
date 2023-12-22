from typing import Callable

from analysis.api import RegressionModelApi
from analysis.cat_boost_regression.CatBoostRegressionModel import CatBoostRegressionModel

CatBoostRegressionModelFactory: Callable[[], RegressionModelApi] = lambda: CatBoostRegressionModel()
