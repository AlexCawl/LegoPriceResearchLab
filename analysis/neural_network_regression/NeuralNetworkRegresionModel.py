from sklearn.neural_network import MLPRegressor

from analysis.BaseRegressionModel import BaseRegressionModel


class NeuralNetworkRegressionModel(BaseRegressionModel):
    def __init__(self):
        super().__init__(
            params={
                'alpha': [1e-4],
                'hidden_layer_sizes': [(250, 50,)],
                'random_state': [42],
                'max_iter': [2500],
                'solver': ['sgd', 'adam'],
                'activation': ['relu'],
                'early_stopping': [True],
                'learning_rate_init': [0.001],
                'learning_rate': ['adaptive']
            },
            estimator=MLPRegressor(),
            name=f"{self.__class__.__name__}"
        )
