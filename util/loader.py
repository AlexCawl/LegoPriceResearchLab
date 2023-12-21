import pandas as pd

from util.constants import DATA_FOLDER_PATH, DATA_FILE_NAME
from util.measurement import measure_execution_time


@measure_execution_time
def save_to_csv(path: str, dataframe: pd.DataFrame, separator: str = ";"):
    dataframe.to_csv(path, sep=separator, encoding="utf-8", index=False)


@measure_execution_time
def load_from_csv(path: str, delimiter: str = ";") -> pd.DataFrame:
    return pd.read_csv(path, delimiter=delimiter, encoding="utf-8")


@measure_execution_time
def get_lego() -> pd.DataFrame:
    path = f"{DATA_FOLDER_PATH}/{DATA_FILE_NAME}"
    return load_from_csv(path, delimiter=";")
