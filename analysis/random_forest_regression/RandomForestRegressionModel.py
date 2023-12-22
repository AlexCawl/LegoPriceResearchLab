from sklearn.ensemble import RandomForestRegressor

from analysis.BaseRegressionModel import BaseRegressionModel


class RandomForestRegressionModel(BaseRegressionModel):
    def __init__(self):
        super().__init__(
            params={
                'n_estimators': [50],
                'max_depth': [15],
                'min_samples_split': [55],
                'max_leaf_nodes': [105]
            },
            estimator=RandomForestRegressor(),
            name=f"{self.__class__.__name__}"
        )
