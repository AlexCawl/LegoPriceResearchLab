from typing import Callable, List, Optional

import pandas as pd

from analysis.SelectiveWrapper import SelectiveWrapper
from analysis.api import RegressionModelApi
from analysis.cat_boost_regression import CatBoostRegressionModelFactory
from analysis.gradient_boosting_regression import GradientBoostingRegressionModelFactory
from analysis.linear_regression import LeastSquaresLinearRegressionModelFactory
from analysis.neural_network_regression import NeuralNetworkRegressionModelFactory
from analysis.preprocess import preprocess, split
from util import logger
from util.constants import LOG_FOLDER_PATH
from util.loader import load_to_script
from util.measurement import measure_execution_time
from util.namespace import get_absolute_path_for_directory_and_make_it


@measure_execution_time
def check(
        x_train: pd.DataFrame, y_train: pd.DataFrame,
        x_test: pd.DataFrame, y_test: pd.DataFrame,
        factory: Callable[[], RegressionModelApi],
        log_path: Optional[str] = None,
        out_path: Optional[str] = None
) -> None:
    selector = SelectiveWrapper("Theme name", factory)
    logger.write_dict(selector.get_info(), path=log_path)
    selector.train(x_train, y_train, out_path)
    selector.test(x_test, y_test, out_path)
    logger.write_dict(selector.get_info()["SOLVERS_INFO"], path=log_path)


@measure_execution_time
def main(
        lego: pd.DataFrame, factories: List[Callable[[], RegressionModelApi]],
        log_path: Optional[str] = None,
        out_path: Optional[str] = None
) -> None:
    transformed, encoders = preprocess(lego)
    x_train, y_train, x_test, y_test = split(transformed)
    for factory in factories:
        check(x_train, y_train, x_test, y_test, factory, log_path, out_path)


if __name__ == "__main__":
    # logging path
    log: str = logger.access_log(path=LOG_FOLDER_PATH)
    logger.init(path=log)
    # output path
    out: str = get_absolute_path_for_directory_and_make_it(LOG_FOLDER_PATH)
    # raw data
    initial: pd.DataFrame = load_to_script()
    # models
    models: List[Callable[[], RegressionModelApi]] = list()
    models.append(LeastSquaresLinearRegressionModelFactory)
    models.append(NeuralNetworkRegressionModelFactory)
    models.append(GradientBoostingRegressionModelFactory)
    models.append(CatBoostRegressionModelFactory)
    main(initial, models, log, out)
