from typing import Optional, Tuple, Dict, Any

import pandas as pd

from analysis.api import RegressionModelApi
from util.measurement import measure_execution_time


class SelectiveWrapper(RegressionModelApi):
    @measure_execution_time
    def get_info(self) -> Dict[Any, Any]:
        pass

    @measure_execution_time
    def train(self, x: pd.DataFrame, y: pd.DataFrame, path: Optional[str] = None) -> None:
        pass

    @measure_execution_time
    def test(self, x: pd.DataFrame, y: pd.DataFrame, path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        pass
