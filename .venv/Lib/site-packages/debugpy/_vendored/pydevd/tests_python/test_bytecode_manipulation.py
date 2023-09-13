from io import StringIO
import os
import sys
import traceback

import pytest

from tests_python.debug_constants import IS_PY36_OR_GREATER, IS_CPYTHON, TEST_CYTHON, TODO_PY311
import dis

pytestmark = pytest.mark.skipif(
    not IS_PY36_OR_GREATER or
    TODO_PY311 or
    not IS_CPYTHON or
    not TEST_CYTHON, reason='Requires CPython >= 3.6 < 3.11')


class _Tracer(object):

    def __init__(self):
        self.stream = StringIO()
        self._in_print = False
        self.accept_frame = None  # Can be set to a callable
        self.lines_executed = set()

    def tracer_printer(self, frame, event, arg):
        if self._in_print:
            return None
        self._in_print = True
        try:
            if self.accept_frame is None or self.accept_frame(frame):
                if arg is not None:
                    if event == 'exception':
                        arg = arg[0].__name__
                    elif arg is not None:
                        arg = str(arg)
                if arg is None:
                    arg = ''

                self.lines_executed.add(frame.f_lineno)

                s = ' '.join((
                    str(frame.f_lineno),
                    frame.f_code.co_name,
                    os.path.basename(frame.f_code.co_filename),
                    event.upper() if event != 'line' else event,
                    arg,
                ))
                self.writeln(s)
        except:
            traceback.print_exc()
        self._in_print = False
        return self.tracer_printer

    def writeln(self, s):
        self.write(s)
        self.write('\n')

    def write(self, s):
        if isinstance(s, bytes):
            s = s.decode('utf-8')
        self.stream.write(s)

    def call(self, c):
        sys.settrace(self.tracer_printer)
        c()
        sys.settrace(None)
        return self.stream.getvalue()


def check(
    filename,
    method,
    method_kwargs=None,
    skip_breaks_at_lines=None,
    method_to_change=None,
    stop_at_all_lines=False,
    has_line_event_optimized_in_original_case=False,
    ):
    '''
    :param has_line_event_optimized_in_original_case:
        If True, we're handling a case where we have a double jump, i.e.: some case
        where there's a JUMP_FORWARD which points to a JUMP_ABSOLUTE and this is
        optimized so that the JUMP_FORWARD is changed directly to a JUMP_ABSOLUTE and
        we end up skipping one line event which is supposed to be there but isn't in
        the initial case but appears when we run after modifying the bytecode in memory.

        See: https://github.com/microsoft/debugpy/issues/973#issuecomment-1178090731
    '''
    from _pydevd_frame_eval.pydevd_modify_bytecode import _get_code_line_info
    from _pydevd_frame_eval import pydevd_modify_bytecode

    if method_to_change is None:
        method_to_change = method

    if method_kwargs is None:
        method_kwargs = {}
    if skip_breaks_at_lines is None:
        skip_breaks_at_lines = set()

    pydev_break_stops = []

    def _pydev_needs_stop_at_break(line):
        pydev_break_stops.append(line)
        return False

    tracer = _Tracer()

    def accept_frame(f):
        return filename in f.f_code.co_filename

    code = method_to_change.__code__
    code_line_info = _get_code_line_info(code)

    try:
        tracer.accept_frame = accept_frame

        def call():
            method(**method_kwargs)

        tracer.call(call)
        breakpoint_hit_at_least_once = False

        # Ok, we just ran the tracer once without any breakpoints.
        #
        # Gather its tracing profile: this will be our baseline for further tests (it should contain
        # the events and the order in which the were executed).
        #
        # Note: methods cannot have random elements when executing (otherwise
        # the order would be different and the test would be expected to fail).
        baseline = tracer.stream.getvalue()

        for line in sorted(code_line_info.line_to_offset):
            if line in skip_breaks_at_lines:
                continue
            # Now, for each valid line, add a breakpoint and check if the tracing profile is exactly
            # the same (and if the line where we added the breakpoint was executed, see if our
            # callback got called).
            success, new_code = pydevd_modify_bytecode.insert_pydevd_breaks(code, set([line]), _pydev_needs_stop_at_break=_pydev_needs_stop_at_break)

            assert success
            method_to_change.__code__ = new_code

            tracer = _Tracer()
            tracer.accept_frame = accept_frame
            tracer.call(call)
            contents = tracer.stream.getvalue()

            assert tracer.lines_executed
            if has_line_event_optimized_in_original_case:
                lines = sorted(set(x[1] for x in dis.findlinestarts(new_code)))
                new_line_contents = []
                last_line = str(max(lines)) + ' '
                for l in contents.splitlines(keepends=True):
                    if not l.strip().startswith(last_line):
                        new_line_contents.append(l)
                contents = ''.join(new_line_contents)

            if line in tracer.lines_executed:
                assert set([line]) == set(pydev_break_stops)
                breakpoint_hit_at_least_once = True
            else:
                if stop_at_all_lines:
                    raise AssertionError('Expected the debugger to stop at all lines. Did not stop at line: %s' % (line,))
            del pydev_break_stops[:]

            if baseline != contents:
                print('------- replacement at line: %s ---------' % (line,))
                print('------- baseline ---------')
                print(baseline)
                print('------- contents ---------')
                print(contents)
                print('-------- error -----------')
                assert baseline == contents

        # We must have found a break at least once!
        assert breakpoint_hit_at_least_once
    finally:
        method_to_change.__code__ = code


def test_set_pydevd_break_01():
    from tests_python.resources import _bytecode_overflow_example

    check('_bytecode_overflow_example.py', _bytecode_overflow_example.Dummy.fun, method_kwargs={'text': 'ing'}, has_line_event_optimized_in_original_case=True)


def test_set_pydevd_break_01a():
    from tests_python.resources import _bytecode_overflow_example

    check('_bytecode_overflow_example.py', _bytecode_overflow_example.check_backtrack, method_kwargs={'x': 'f'})


def test_set_pydevd_break_02():
    from tests_python.resources import _bytecode_many_names_example

    check('_bytecode_many_names_example.py', _bytecode_many_names_example.foo)


def test_set_pydevd_break_03():
    from tests_python.resources import _bytecode_big_method

    check('_bytecode_big_method.py', _bytecode_big_method.foo)


def test_set_pydevd_break_04():
    from tests_python.resources import _debugger_case_yield_from

    check('_debugger_case_yield_from.py', _debugger_case_yield_from.method)


def test_set_pydevd_break_05():
    from tests_python import debugger_unittest
    add_to_pythonpath = debugger_unittest._get_debugger_test_file('wrong_bytecode')
    sys.path.append(add_to_pythonpath)

    try:
        with open(debugger_unittest._get_debugger_test_file('wrong_bytecode/_debugger_case_wrong_bytecode.py'), 'r') as stream:
            contents = stream.read()

        code = compile(contents, '_my_file_debugger_case_wrong_bytecode.py', 'exec')

        def method():
            pass

        method.__code__ = code

        check('_my_file_debugger_case_wrong_bytecode.py', method, skip_breaks_at_lines=set([1]))
    finally:
        sys.path.remove(add_to_pythonpath)


def test_set_pydevd_break_06(pyfile):
    from tests_python.resources import _bytecode_super

    check('_bytecode_super.py', _bytecode_super.B, method_to_change=_bytecode_super.B.__init__, stop_at_all_lines=True)


def test_set_pydevd_break_07():
    from tests_python.resources import _bytecode_overflow_example

    check('_bytecode_overflow_example.py', _bytecode_overflow_example.offset_overflow, method_kwargs={'stream': StringIO()})


def test_set_pydevd_break_08():
    from tests_python.resources import _bytecode_overflow_example

    check('_bytecode_overflow_example.py', _bytecode_overflow_example.long_lines, stop_at_all_lines=True)


def test_internal_double_linked_list():
    from _pydevd_frame_eval.pydevd_modify_bytecode import _HelperBytecodeList
    lst = _HelperBytecodeList()
    node1 = lst.append(1)
    assert list(lst) == [1]
    node2 = lst.append(2)
    assert list(lst) == [1, 2]

    node15 = node1.append(1.5)
    assert list(lst) == [1, 1.5, 2]

    node12 = node15.prepend(1.2)
    assert list(lst) == [1, 1.2, 1.5, 2]

    node12.prepend(1.1)
    assert list(lst) == [1, 1.1, 1.2, 1.5, 2]

    node1.prepend(0)
    assert list(lst) == [0, 1, 1.1, 1.2, 1.5, 2]

    node2.append(3)
    assert list(lst) == [0, 1, 1.1, 1.2, 1.5, 2, 3]

    assert lst.head.data == 0
    assert lst.tail.data == 3

    lst = _HelperBytecodeList([1, 2, 3, 4, 5])
    assert lst.head.data == 1
    assert lst.tail.data == 5
