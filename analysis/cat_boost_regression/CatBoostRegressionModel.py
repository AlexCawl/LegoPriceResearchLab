from typing import Optional, Tuple, Dict, Any

import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from analysis.api import RegressionModelApi
from util.measurement import measure_execution_time


class CatBoostRegressionModel(RegressionModelApi):
    # Model Name
    __name: str

    # Model state [Trained - True, otherwise - False]
    __state: bool

    # Overall report (used to describe model state after training & testing)
    __report: Dict[str, Any]

    # Load graphics
    __graphics: bool

    # Estimator
    __estimator: CatBoostRegressor

    def __init__(self):
        # init name
        self.__name = "CatBoostRegressionModel"
        # init graphics logging
        self.__graphics = True
        # init state
        self.__state = False
        # init report
        self.__report = dict()
        # setup estimator
        self.__estimator = CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=10, verbose=100)

    def get_info(self) -> Dict[str, Any]:
        return self.__report.copy()

    @measure_execution_time
    def train(self, x: pd.DataFrame, y: pd.DataFrame, path: Optional[str] = None) -> None:
        # change state
        self.__state = True
        # train model
        self.__estimator.fit(X=x, y=y.to_numpy(), verbose=0)
        r2: float = self.__estimator.score(X=x, y=y.to_numpy())
        self.__report.update(
            {
                "TRAIN_R2": r2 if r2 >= -1 else -1
            }
        )

    @measure_execution_time
    def test(self, x: pd.DataFrame, y: pd.DataFrame, path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if not self.__state:
            raise Exception("Model not trained!")

        prediction = self.__estimator.predict(x)
        r2 = r2_score(y, prediction)
        self.__report.update(
            {
                "MAE": mean_absolute_error(y, prediction),
                "MSE": mean_squared_error(y, prediction),
                "RMSE": mean_squared_error(y, prediction, squared=False),
                "TEST_R2": r2 if r2 >= -1 else -1
            }
        )
        return y, prediction
