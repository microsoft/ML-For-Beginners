import threading, atexit, sys
import time

try:
    from thread import start_new_thread
except:
    from _thread import start_new_thread


def _atexit():
    print('TEST SUCEEDED')
    sys.stderr.write('TEST SUCEEDED\n')
    sys.stderr.flush()
    sys.stdout.flush()


# Register the TEST SUCEEDED msg to the exit of the process.
atexit.register(_atexit)


def thread_func():
    raise Exception('in thread 1')


start_new_thread(thread_func, ())

# Wait for the first to be handled... otherwise, tests can become flaky if
# both stop at the same time only 1 notification may be given for both, whereas
# the test expects 2 notifications.
time.sleep(.5)


def thread_func2(n):
    raise ValueError('in thread 2')


th = threading.Thread(target=lambda: thread_func2(1))
th.daemon = True
th.start()

th.join()

# This is a bit tricky: although we waited on the event, there's a slight chance
# that we didn't get the notification because the thread could've stopped executing,
# so, sleep a bit so that the test does not become flaky.
time.sleep(.5)
raise IndexError('in main')
