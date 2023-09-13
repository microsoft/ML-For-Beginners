from concurrent.futures import ThreadPoolExecutor
import matplotlib
import matplotlib.pyplot as plt

processed = []


def double(nmbr):
    doubled = nmbr * 2  # break here
    processed.append(1)
    return doubled


with ThreadPoolExecutor(max_workers=2) as pool:
    futures = []

    for number in range(3):
        future = pool.submit(double, number)
        futures.append(future)

pool.shutdown()
assert len(processed) == 3
print('TEST SUCEEDED!')
