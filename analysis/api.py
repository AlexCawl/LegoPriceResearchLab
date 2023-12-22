from abc import abstractmethod
from typing import Dict, Any, Optional, Tuple

import pandas as pd


class RegressionModelApi:
    @abstractmethod
    def get_info(self) -> Dict[Any, Any]:
        ...

    @abstractmethod
    def train(self, x: pd.DataFrame, y: pd.DataFrame, path: Optional[str] = None) -> None:
        ...

    @abstractmethod
    def test(self, x: pd.DataFrame, y: pd.DataFrame, path: Optional[str] = None) -> None:
        ...
