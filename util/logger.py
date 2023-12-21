from datetime import datetime
from typing import Optional, TextIO, Dict, Any

from util.constants import LOG_COMMON_NAME


def access_log(path: Optional[str]) -> Optional[str]:
    if path is not None:
        return f"{path}/{LOG_COMMON_NAME}"
    else:
        return None


def init(*, path: Optional[str] = None):
    if path is not None:
        file: TextIO = open(path, "w")
        file.write(f"START: {datetime.now()}" + "\n\n")
        file.close()
    else:
        print(f"START: {datetime.now()}")


def write_dict(log: Dict[Any, Any], *, path: Optional[str] = None) -> None:
    if path is not None:
        file: TextIO = open(path, "a")
        for key, value in log.items():
            file.write(f"{key}: {value}" + "\n")
        file.close()
    else:
        for key, value in log.items():
            print(f"{key}: {value}")


def write(log: str, *, path: Optional[str] = None) -> None:
    if path is not None:
        file: TextIO = open(path, "a")
        file.write(f"{log}" + "\n")
        file.close()
    else:
        print(f"{log}")
