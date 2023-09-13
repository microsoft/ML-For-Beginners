#!/usr/bin/env python
from gevent import monkey, sleep, threading as gevent_threading
import sys

if 'remote' in sys.argv:
    import pydevd
    if '--use-dap-mode' in sys.argv:
        pydevd.config('http_json', 'debugpy-dap')

    port = int(sys.argv[1])
    print('before pydevd.settrace')
    pydevd.settrace(host=('' if 'as-server' in sys.argv else '127.0.0.1'), port=port, suspend=False)
    print('after pydevd.settrace')

monkey.patch_all()
import threading

called = []


class MyGreenThread2(threading.Thread):

    def run(self):
        for _i in range(3):
            sleep()


class MyGreenletThread(threading.Thread):

    def run(self):
        for _i in range(5):
            called.append(self.name)  # break here
            t1 = MyGreenThread2()
            t1.start()
            sleep()


if __name__ == '__main__':
    t1 = MyGreenletThread()
    t1.name = 't1'
    t2 = MyGreenletThread()
    t2.name = 't2'

    if hasattr(gevent_threading, 'Thread'):
        # Only available in newer versions of gevent.
        assert isinstance(t1, gevent_threading.Thread)
        assert isinstance(t2, gevent_threading.Thread)

    t1.start()
    t2.start()

    for t1 in (t1, t2):
        t1.join()

    # With gevent it's always the same (gevent coroutine support makes thread
    # switching serial).
    expected = ['t1', 't2', 't1', 't2', 't1', 't2', 't1', 't2', 't1', 't2']
    if called != expected:
        raise AssertionError("Expected:\n%s\nFound:\n%s" % (expected, called))
    print('TEST SUCEEDED')
