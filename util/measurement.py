import time


def measure_execution_time(func):
    def inner1(*args, **kwargs):
        begin: float = time.time()
        result = func(*args, **kwargs)
        end: float = time.time()
        print(f"Execution time for '{func.__name__}' took {end - begin} seconds")
        return result

    return inner1
