from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold

from analysis.api import RegressionModelApi
from util.measurement import measure_execution_time


# Base class for Linear Regression Models
# Can (and should) be reused in your models
class BaseRegressionModel(RegressionModelApi):
    # Model Name
    __name: str

    # Model state [Trained - True, otherwise - False]
    __state: bool

    # Model params for GridSearchCV (everything depends on model type API)
    __params: Dict[str, Any]

    # Model bundle from GridSearchCV (matrix of model params)
    __search: GridSearchCV

    # Cross-Validation (same for all models)
    __cv: RepeatedStratifiedKFold = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=36851234)

    # Overall report (used to describe model state after training & testing)
    __report: Dict[str, Any]

    # Load graphics
    __graphics: bool

    def __init__(self, params: Dict[str, Any], estimator: Any, name: str):
        # init name
        self.__name = name
        # init state
        self.__state = False
        # init params
        self.__params = params
        # init cross-validator
        self.__cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=36851234)
        # init logs
        self.__report = dict()
        # init search
        self.__search = GridSearchCV(estimator=estimator, param_grid=self.__params, cv=self.__cv, n_jobs=-1)

    def get_info(self) -> Dict[str, Any]:
        # validation
        if not self.__state:
            raise Exception("Model not trained!")

        # init info report
        report: Dict[str, str] = dict()

        # model params
        report.update(
            {
                "params": self.__search.best_params_
            }
        )

        # model result params
        report.update(self.__report)
        return report

    @measure_execution_time
    def train(self, x: pd.DataFrame, y: pd.DataFrame, path: Optional[str] = None) -> None:
        # change state
        self.__state = True
        # train model
        self.__search.fit(X=x, y=y)
        r2: float = self.__search.score(X=x, y=y)
        self.__report.update(
            {
                "TRAIN_R2": r2 if r2 >= -1 else -1
            }
        )

    @measure_execution_time
    def test(self, x: pd.DataFrame, y: pd.DataFrame, path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if not self.__state:
            raise Exception("Model not trained!")

        prediction = self.__search.best_estimator_.predict(x)
        r2 = metrics.r2_score(y, prediction)
        self.__report.update(
            {
                "MAE": metrics.mean_absolute_error(y, prediction),
                "MSE": metrics.mean_squared_error(y, prediction),
                "RMSE": np.sqrt(metrics.mean_squared_error(y, prediction)),
                "TEST_R2": r2 if r2 >= -1 else -1
            }
        )
        return y, prediction
