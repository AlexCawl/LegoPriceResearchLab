import pandas as pd

from util.constants import DATA_FOLDER_PATH, DATA_FILE_NAME
from util.measurement import measure_execution_time
from util.namespace import get_absolute_path_for_directory_and_make_it


@measure_execution_time
def save_to_csv(path: str, dataframe: pd.DataFrame, separator: str = ";"):
    dataframe.to_csv(path, sep=separator, encoding="utf-8", index=False)


@measure_execution_time
def load_from_csv(path: str, delimiter: str = ";") -> pd.DataFrame:
    return pd.read_csv(path, delimiter=delimiter, encoding="utf-8")


@measure_execution_time
def load_to_notebook() -> pd.DataFrame:
    path = f"{DATA_FOLDER_PATH}/{DATA_FILE_NAME}"
    return load_from_csv(path, delimiter=";")


@measure_execution_time
def load_to_script() -> pd.DataFrame:
    path: str = get_absolute_path_for_directory_and_make_it(DATA_FOLDER_PATH)
    return load_from_csv(f"{path}/{DATA_FILE_NAME}")
