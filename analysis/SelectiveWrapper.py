from typing import Optional, Dict, Any, Callable, Tuple

import pandas as pd

from analysis.api import RegressionModelApi
from util.measurement import measure_execution_time


class SelectiveWrapper(RegressionModelApi):
    # Selective variable
    __selective_variable_name: str

    # Solver models
    __solvers: Dict[Any, RegressionModelApi]

    # Solver model factory
    __factory: Callable[[], RegressionModelApi]

    # Overall report (used to describe model state after training & testing)
    __report: Dict[str, Any]

    def __init__(self, selector_name: str, model: Callable[[], RegressionModelApi]):
        self.__selective_variable_name = selector_name
        self.__factory = model
        self.__solvers = dict()
        self.__report = dict()
        self.__report.update(
            {
                "SELECTOR_NAME": self.__selective_variable_name,
                "SELECTOR_MODEL": self.__factory().__class__.__name__
            }
        )

    @measure_execution_time
    def get_info(self) -> Dict[Any, Any]:
        solvers_info: Dict[Any, Dict[Any, Any]] = dict()
        for key, model in self.__solvers.items():
            solvers_info.update(
                {
                    f"{key}": model.get_info()
                }
            )
        self.__report.update(
            {
                "SOLVERS_INFO": solvers_info
            }
        )
        return self.__report

    @measure_execution_time
    def train(self, x: pd.DataFrame, y: pd.DataFrame, path: Optional[str] = None) -> None:
        for selector, (x_select, y_select) in self.__get_train_test_selective(x, y).items():
            model: RegressionModelApi = self.__factory()
            model.train(x_select, y_select, path)
            self.__solvers[selector] = model

    @measure_execution_time
    def test(self, x: pd.DataFrame, y: pd.DataFrame, path: Optional[str] = None) -> None:
        for selector, (x_select, y_select) in self.__get_train_test_selective(x, y).items():
            model: RegressionModelApi = self.__solvers[selector]
            model.test(x_select, y_select, path)

    def __get_train_test_selective(
            self, x: pd.DataFrame, y: pd.DataFrame
    ) -> Dict[Any, Tuple[pd.DataFrame, pd.DataFrame]]:
        # get factors/targets
        factor_vars = list(x.columns)
        target_vars = list(y.columns)

        # concat to original dataframe
        original_dataframe = pd.concat([x, y], axis=1)

        # get selective values
        selective_values = set(original_dataframe[self.__selective_variable_name].unique())

        # calculate result
        result: Dict[Any, Tuple[pd.DataFrame, pd.DataFrame]] = dict()
        for value in selective_values:
            selected_dataframe = original_dataframe[original_dataframe[self.__selective_variable_name] == value]
            result[value] = (selected_dataframe[factor_vars], selected_dataframe[target_vars])
        return result
