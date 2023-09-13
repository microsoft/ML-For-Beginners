import time
import sys
import itertools

from itertools import groupby
count = itertools.count(0)


def next_val():
    return next(count) % 25


start_time = time.time()
letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

# create an array of random strings of 40 characters each
l = sorted([''.join([letters[next_val()] for _ in range(40)]) for _ in range(10000)])
# group by the first two characters
g = {k: list(v) for k, v in groupby(l, lambda x: x[:2])}

if False:
    pass  # Breakpoint here

print('TotalTime>>%s<<' % (time.time() - start_time,))
print('TEST SUCEEDED')
