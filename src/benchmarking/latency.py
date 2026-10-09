import time


def measure_generation_time(generate_function):

    start = time.perf_counter()

    result = generate_function()

    end = time.perf_counter()

    elapsed = end - start

    return result, elapsed