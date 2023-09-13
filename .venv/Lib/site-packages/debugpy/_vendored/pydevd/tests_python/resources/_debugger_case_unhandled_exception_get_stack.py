from contextlib import contextmanager


@contextmanager
def something():
    yield


with something():
    raise ValueError('TEST SUCEEDED')  # break line on unhandled exception
    print('a')
    print('b')
