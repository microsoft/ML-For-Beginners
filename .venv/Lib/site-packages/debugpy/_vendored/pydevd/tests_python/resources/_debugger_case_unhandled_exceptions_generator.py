import atexit, sys


def _atexit():
    print('TEST SUCEEDED')
    sys.stderr.write('TEST SUCEEDED\n')
    sys.stderr.flush()
    sys.stdout.flush()


# Register the TEST SUCEEDED msg to the exit of the process.
atexit.register(_atexit)


def f():
    return list(1 / 0 for _ in '123')  # exc line


f()  # call exc
