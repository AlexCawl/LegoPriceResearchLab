from typing import Optional, Dict, Any, Callable

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

    @measure_execution_time
    def get_info(self) -> Dict[Any, Any]:
        pass

    @measure_execution_time
    def train(self, x: pd.DataFrame, y: pd.DataFrame, path: Optional[str] = None) -> None:
        # get original variables
        factor_vars = list(x.columns.names)
        factor_vars.remove(self.__selective_variable_name)
        target_vars = list(y.columns.names)

        # concat to original dataframe
        original_dataframe = pd.concat([x, y], ignore_index=True)
        selective_variables = set(original_dataframe[self.__selective_variable_name].unique())

        # concat dataframes to original
        # split it by SELECTOR values
        # train each model for split df
        pass

    @measure_execution_time
    def test(self, x: pd.DataFrame, y: pd.DataFrame, path: Optional[str] = None) -> None:
        pass
