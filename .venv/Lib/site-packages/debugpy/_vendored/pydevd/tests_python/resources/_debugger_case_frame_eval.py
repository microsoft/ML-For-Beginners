'''
Things this test checks:

- frame.f_trace is None when there are only regular breakpoints.

- The no-op tracing function is set by default (otherwise when set tracing functions have no effect).

- When stepping in, frame.f_trace must be set by the frame eval.

- When stepping over/return, the frame.f_trace must not be set on intermediate callers.

TODO:

- When frame.f_trace is set to the default tracing function, it'll become None again in frame
  eval mode if not stepping (if breakpoints weren't changed).

- The tracing function in the frames that deal with unhandled exceptions must be set when dealing
  with unhandled exceptions.

- The tracing function in the frames that deal with unhandled exceptions must NOT be set when
  NOT dealing with unhandled exceptions.

- If handled exceptions should be dealt with, the proper tracing should be set in frame.f_trace.
'''

import sys
from _pydevd_frame_eval import pydevd_frame_tracing


def check_with_no_trace():
    if False:
        print('break on check_with_trace')
    frame = sys._getframe()
    if frame.f_trace is not None:
        raise AssertionError('Expected %s to be None' % (frame.f_trace,))

    if sys.gettrace() is not pydevd_frame_tracing.dummy_tracing_holder.dummy_trace_func:
        raise AssertionError('Expected %s to be dummy_trace_func' % (sys.gettrace(),))


def check_step_in_then_step_return():
    frame = sys._getframe()
    f_trace = frame.f_trace
    if f_trace.__class__.__name__ != 'SafeCallWrapper':
        raise AssertionError('Expected %s to be SafeCallWrapper' % (f_trace.__class__.__name__,))

    check_with_no_trace()


def check_revert_to_dummy():
    check_with_no_trace()


if __name__ == '__main__':
    # Check how frame eval works.
    if sys.version_info[0:2] < (3, 6):
        raise AssertionError('Only available for Python 3.6 onwards. Found: %s' % (sys.version_info[0:1],))

    check_with_no_trace()  # break on global (step over)

    check_step_in_then_step_return()

    import pydevd_tracing
    import pydevd

    # This is what a remote attach would do (should revert to the frame eval mode).
    pydevd_tracing.SetTrace(pydevd.get_global_debugger().trace_dispatch)
    check_revert_to_dummy()

    print('TEST SUCEEDED!')
