from sklearn.linear_model import Ridge

from analysis.BaseRegressionModel import BaseRegressionModel


class RidgeLinearRegressionModel(BaseRegressionModel):
    def __init__(self):
        super().__init__(
            params={
                'alpha': [100.0],
                'solver': ['cholesky'],
                'copy_X': [True],
                'fit_intercept': [True]
            },
            estimator=Ridge(),
            name=f"{self.__class__.__name__}"
        )
