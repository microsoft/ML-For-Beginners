import pytest
import sys
from _pydevd_bundle.pydevd_constants import NO_FTRACE
from tests_python.debugger_unittest import IS_JYTHON


class Tracer(object):

    def __init__(self):
        self.call_calls = 0
        self.line_calls = 0

    def trace_func(self, frame, event, arg):
        if event == 'call':
            self.call_calls += 1
            return self.on_call()

        elif event == 'line':
            self.line_calls += 1
            # Should disable tracing when None is returned (but doesn't).
            return self.on_line(frame, event, arg)

        else:
            return self.trace_func

    def on_call(self):
        return self.trace_func

    def on_line(self, frame, event, arg):
        return None


class TracerSettingNone(Tracer):

    def on_line(self, frame, event, arg):
        frame.f_trace = NO_FTRACE
        return NO_FTRACE


class TracerChangeToOtherTracing(Tracer):

    def on_line(self, frame, event, arg):
        return self._no_trace

    def _no_trace(self, frame, event, arg):
        return self._no_trace


class TracerDisableOnCall(Tracer):

    def on_call(self):
        return None


def test_tracing_gotchas():
    '''
    Summary of the gotchas tested:

    If 'call' is used, a None return value is used for the tracing. Afterwards the return may or may
    not be ignored depending on the Python version.

    Also, on Python 2.6, the trace function may not be set to None as it'll give an error
    afterwards (it needs to be set to an empty tracing function).

    All of the versions seem to work when the return value is a new callable though.

    Note: according to: https://docs.python.org/3/library/sys.html#sys.settrace the behavior
    does not follow the spec (but we have to work with it nonetheless).

    Note: Jython seems to do what's written in the docs.
    '''

    def method():
        _a = 1
        _b = 1
        _c = 1
        _d = 1

    for tracer, line_events in (
        (Tracer(), 1 if IS_JYTHON else 4),
        (TracerSettingNone(), 1),
        (TracerChangeToOtherTracing(), 1),
        (TracerDisableOnCall(), 0),
        ):
        curr_trace_func = sys.gettrace()
        try:
            sys.settrace(tracer.trace_func)

            method()

            if tracer.call_calls != 1:
                pytest.fail('Expected a single call event. Found: %s' % (tracer.call_calls))

            if tracer.line_calls != line_events:
                pytest.fail('Expected %s line events. Found: %s. Tracer: %s' % (line_events, tracer.line_calls, tracer))
        finally:
            sys.settrace(curr_trace_func)


if __name__ == '__main__':
    pytest.main(['-k', 'test_tracing_gotchas'])
