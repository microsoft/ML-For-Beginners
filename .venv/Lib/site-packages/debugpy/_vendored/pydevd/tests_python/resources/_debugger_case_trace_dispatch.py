import sys


def method():
    a = 10  # add breakpoint
    b = 20
    c = 30
    d = 40
    f_trace = sys._getframe().f_trace
    if sys.version_info[:2] == (2,6) and f_trace.__name__ == 'NO_FTRACE':
        print('TEST SUCEEDED')
    elif f_trace is None:
        print('TEST SUCEEDED')
    else:
        raise AssertionError('frame.f_trace is expected to be None at this point. Found: %s' % (f_trace,))


method()
