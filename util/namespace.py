import os
import sys

from util.constants import DATA_FOLDER_PATH, DATA_FILE_NAME


def get_absolute_path() -> str:
    return os.path.dirname(os.path.abspath(sys.argv[0]))


def get_absolute_path_for_directory(dir_name: str) -> str:
    return os.path.join(get_absolute_path(), dir_name)


def get_absolute_path_for_directory_and_make_it(local_path: str) -> str:
    abs_path: str = get_absolute_path_for_directory(local_path)
    if not os.path.isdir(abs_path):
        os.makedirs(abs_path)
    return abs_path


def get_data_path() -> str:
    return f"{get_absolute_path_for_directory_and_make_it(DATA_FOLDER_PATH)}/{DATA_FILE_NAME}"
