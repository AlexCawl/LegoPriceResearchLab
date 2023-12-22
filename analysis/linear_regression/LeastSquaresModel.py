from sklearn.linear_model import LinearRegression

from analysis.BaseRegressionModel import BaseRegressionModel


class LeastSquaresLinearRegressionModel(BaseRegressionModel):
    def __init__(self):
        super().__init__(
            params={
                'fit_intercept': [False],
                'copy_X': [True]
            },
            estimator=LinearRegression(),
            name=f"{self.__class__.__name__}",
        )
