from sklearn.ensemble import GradientBoostingRegressor

from analysis.BaseRegressionModel import BaseRegressionModel


class GradientBoostingRegressionModel(BaseRegressionModel):
    def __init__(self):
        super().__init__(
            params={
                'max_depth': [5],
                'n_estimators': [30],
                'max_leaf_nodes': [20],
                'learning_rate': [.12]
            },
            estimator=GradientBoostingRegressor(),
            name=f"{self.__class__.__name__}"
        )
