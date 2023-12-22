from typing import Optional, Dict, Any, Callable, Tuple

import pandas as pd

from analysis.api import RegressionModelApi
from util.measurement import measure_execution_time
from util.plots import regression_visualization, regressions_scores


class SelectiveWrapper(RegressionModelApi):
    # Selective variable
    __selective_variable_name: str

    # flag is trained
    __is_trained: bool

    # Solver models
    __solvers: Dict[Any, RegressionModelApi]

    # Solver model factory
    __factory: Callable[[], RegressionModelApi]

    # Overall report (used to describe model state after training & testing)
    __report: Dict[str, Any]

    def __init__(self, selector_name: str, model: Callable[[], RegressionModelApi]):
        self.__selective_variable_name = selector_name
        self.__is_trained = False
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
        if self.__is_trained:
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
        self.__is_trained = True
        for selector, (x_select, y_select) in self.__get_train_test_selective(x, y).items():
            model: RegressionModelApi = self.__factory()
            model.train(x_select, y_select, path)
            self.__solvers[selector] = model

    @measure_execution_time
    def test(self, x: pd.DataFrame, y: pd.DataFrame, path: Optional[str] = None) -> None:
        analytics: pd.DataFrame = pd.DataFrame(columns=['Actual', 'Expected', 'Theme name'])
        values: pd.DataFrame = pd.DataFrame(columns=['Theme name', 'MAE', 'MSE', 'RMSE', 'R2', 'type'], dtype=float)
        for selector, (x_select, y_select) in self.__get_train_test_selective(x, y).items():
            # test model
            model: RegressionModelApi = self.__solvers[selector]
            actual, expected = model.test(x_select, y_select, path)
            # update analytics
            cur_analytics: pd.DataFrame = pd.DataFrame()
            cur_analytics['Actual'] = actual
            cur_analytics['Expected'] = expected
            cur_analytics['Theme name'] = selector
            analytics = pd.concat([analytics, cur_analytics], axis=0)
            # update values
            cur_values = model.get_info()
            cur_values.update({'Theme name': f"{selector}"})
            train_r2 = cur_values.pop('TRAIN_R2')
            test_r2 = cur_values.pop('TEST_R2')
            rowTrain = cur_values.copy()
            rowTrain.update({'R2': train_r2, 'type': 'train'})  # not sorry
            values.loc[len(values)] = pd.Series(rowTrain)
            rowTest = cur_values.copy()
            rowTest.update({'R2': test_r2, 'type': 'test'})  # not sorry
            values.loc[len(values)] = pd.Series(rowTest)
        if path is not None:
            regression_visualization(analytics, path, f"{self.__report['SELECTOR_MODEL']}")
            regressions_scores(values, path, f"{self.__report['SELECTOR_MODEL']}")

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
