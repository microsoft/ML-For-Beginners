import sys
import threading
from _pydev_bundle import pydev_log


def check():
    with pydev_log.log_context(3, sys.stderr):
        assert hasattr(sys, 'gettotalrefcount')
        import pydevd_tracing

        proceed1 = threading.Event()
        proceed2 = threading.Event()

        class SomeThread(threading.Thread):

            def run(self):
                proceed1.set()
                proceed2.wait()

        t = SomeThread()
        t.start()
        proceed1.wait()
        try:

            def some_func(frame, event, arg):
                return some_func

            pydevd_tracing.set_trace_to_threads(some_func)
        finally:
            proceed2.set()

        lib = pydevd_tracing._load_python_helper_lib()
        assert lib is None
        print('Finished OK')


if __name__ == '__main__':
    check()
