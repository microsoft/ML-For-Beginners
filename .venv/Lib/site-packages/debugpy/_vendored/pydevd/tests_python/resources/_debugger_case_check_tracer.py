import threading, atexit, sys
from collections import namedtuple
import os.path

if sys.version_info[0] >= 3:
    from _thread import start_new_thread
else:
    from thread import start_new_thread

FrameInfo = namedtuple('FrameInfo', 'filename, name, f_trace')


def _atexit():
    sys.stderr.flush()
    sys.stdout.flush()


# Register the TEST SUCEEDED msg to the exit of the process.
atexit.register(_atexit)


def _iter_frame_info(frame):
    while frame is not None:
        yield FrameInfo(
            os.path.basename(frame.f_code.co_filename),
            frame.f_code.co_name,
            frame.f_trace.__name__ if frame.f_trace is not None else "None"
        )
        frame = frame.f_back


def check_frame_info(expected):
    found = list(_iter_frame_info(sys._getframe().f_back))

    def fail():
        raise AssertionError('Expected:\n%s\n\nFound:\n%s\n' % (
            '\n'.join(str(x) for x in expected),
            '\n'.join(str(x) for x in found)))

    for found_info, expected_info in  zip(found, expected):
        if found_info.filename != expected_info.filename or found_info.name != expected_info.name:
            fail()

        for f_trace in expected_info.f_trace.split('|'):
            if f_trace == found_info.f_trace:
                break
        else:
            fail()


def thread_func():
    check_frame_info([
        FrameInfo(filename='_debugger_case_check_tracer.py', name='thread_func', f_trace='trace_exception'),
        FrameInfo(filename='threading.py', name='run', f_trace='None'),
        FrameInfo(filename='threading.py', name='_bootstrap_inner', f_trace='trace_unhandled_exceptions'),
        FrameInfo(filename='threading.py', name='_bootstrap', f_trace='None'),
        FrameInfo(filename='pydev_monkey.py', name='__call__', f_trace='None')
    ])


th = threading.Thread(target=thread_func)
th.daemon = True
th.start()

event = threading.Event()


def thread_func2():
    try:
        check_frame_info([
            FrameInfo(filename='_debugger_case_check_tracer.py', name='thread_func2', f_trace='trace_exception'),
            FrameInfo(filename='pydev_monkey.py', name='__call__', f_trace='trace_unhandled_exceptions')
        ])
    finally:
        event.set()


start_new_thread(thread_func2, ())

event.wait()
th.join()

# This is a bit tricky: although we waited on the event, there's a slight chance
# that we didn't get the notification because the thread could've stopped executing,
# so, sleep a bit so that the test does not become flaky.
import time
time.sleep(.3)

check_frame_info([
    FrameInfo(filename='_debugger_case_check_tracer.py', name='<module>', f_trace='trace_exception'),
    FrameInfo(filename='pydevd_runpy.py', name='_run_code', f_trace='None'),
    FrameInfo(filename='pydevd_runpy.py', name='_run_module_code', f_trace='None'),
    FrameInfo(filename='pydevd_runpy.py', name='run_path', f_trace='None'),
    FrameInfo(filename='pydevd.py', name='_exec', f_trace='trace_unhandled_exceptions'),
    FrameInfo(filename='pydevd.py', name='run', f_trace='trace_dispatch|None'),
    FrameInfo(filename='pydevd.py', name='main', f_trace='trace_dispatch|None'),
    FrameInfo(filename='pydevd.py', name='<module>', f_trace='trace_dispatch|None')
])

print('TEST SUCEEDED')
