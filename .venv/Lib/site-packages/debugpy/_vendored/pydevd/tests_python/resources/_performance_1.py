import time


def method2():
    i = 1


def method():

    for i in range(200000):
        method2()

        if False:
            # Unreachable breakpoint here
            pass


def caller():
    start_time = time.time()
    method()
    print('TotalTime>>%s<<' % (time.time() - start_time,))


if __name__ == '__main__':
    import sys
    if '--regular-trace' in sys.argv:

        def trace_dispatch(frame, event, arg):
            return trace_dispatch

        sys.settrace(trace_dispatch)

    caller()  # Initial breakpoint for a step-over here
    print('TEST SUCEEDED')
