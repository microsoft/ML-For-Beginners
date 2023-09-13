import sys
import threading
import pytest
from tests_python.debugger_unittest import IS_PY36_OR_GREATER, IS_CPYTHON
from tests_python.debug_constants import TEST_CYTHON, TODO_PY311

pytestmark = pytest.mark.skipif(not IS_PY36_OR_GREATER or not IS_CPYTHON or not TEST_CYTHON or TODO_PY311, reason='Requires CPython >= 3.6')


def get_foo_frame():
    frame = sys._getframe()
    return frame


class CheckClass(object):

    def collect_info(self):
        from _pydevd_frame_eval import pydevd_frame_evaluator
        thread_info = pydevd_frame_evaluator.get_thread_info_py()
        self.thread_info = thread_info


@pytest.mark.parametrize('_times', range(2))
def test_thread_info(_times):
    obj = CheckClass()
    obj.collect_info()
    assert obj.thread_info.additional_info is not None
    assert not obj.thread_info.is_pydevd_thread
    thread_info = obj.thread_info
    obj.collect_info()
    assert obj.thread_info is thread_info

    obj = CheckClass()
    t = threading.Thread(target=obj.collect_info)
    t.is_pydev_daemon_thread = True
    t.start()
    t.join()

    assert obj.thread_info.additional_info is None
    assert obj.thread_info.is_pydevd_thread


def method():
    return sys._getframe()


@pytest.fixture
def _custom_global_dbg():
    from _pydevd_bundle.pydevd_constants import GlobalDebuggerHolder
    from pydevd import PyDB
    curr = GlobalDebuggerHolder.global_dbg
    PyDB()  # Will make itself current
    yield
    GlobalDebuggerHolder.global_dbg = curr


@pytest.mark.parametrize('_times', range(2))
def test_func_code_info(_times, _custom_global_dbg):
    from _pydevd_frame_eval import pydevd_frame_evaluator
    # Must be called before get_func_code_info_py to initialize the _code_extra_index.
    thread_info = pydevd_frame_evaluator.get_thread_info_py()

    func_info = pydevd_frame_evaluator.get_func_code_info_py(thread_info, method(), method.__code__)
    assert func_info.co_filename is method.__code__.co_filename
    func_info2 = pydevd_frame_evaluator.get_func_code_info_py(thread_info, method(), method.__code__)
    assert func_info is func_info2

    some_func = eval('lambda:sys._getframe()')
    func_info3 = pydevd_frame_evaluator.get_func_code_info_py(thread_info, some_func(), some_func.__code__)
    del some_func
    del func_info3

    some_func = eval('lambda:sys._getframe()')
    pydevd_frame_evaluator.get_func_code_info_py(thread_info, some_func(), some_func.__code__)
    func_info = pydevd_frame_evaluator.get_func_code_info_py(thread_info, some_func(), some_func.__code__)
    assert pydevd_frame_evaluator.get_func_code_info_py(thread_info, some_func(), some_func.__code__) is func_info


def test_generate_code_with_breakpoints():
    from _pydevd_frame_eval.pydevd_frame_evaluator import generate_code_with_breakpoints_py
    from _pydevd_frame_eval.pydevd_frame_evaluator import get_cached_code_obj_info_py

    def create_breakpoints_dict(lines):
        return dict((line, None) for line in lines)

    def method():
        a = 1
        a = 2
        a = 3

    breakpoint_found, new_code = generate_code_with_breakpoints_py(
        method.__code__,
        create_breakpoints_dict([method.__code__.co_firstlineno + 1, method.__code__.co_firstlineno + 2])
    )

    assert breakpoint_found
    with pytest.raises(AssertionError):
        # We must use the cached one directly (in the real-world, this would indicate a reuse
        # of the code object -- which is related to generator handling).
        generate_code_with_breakpoints_py(
            new_code,
            create_breakpoints_dict([method.__code__.co_firstlineno + 1])
        )

    cached_value = get_cached_code_obj_info_py(new_code)
    breakpoint_found, force_stay_in_untraced_mode = cached_value.compute_force_stay_in_untraced_mode(
        create_breakpoints_dict([method.__code__.co_firstlineno + 1]))

    assert breakpoint_found
    assert force_stay_in_untraced_mode

    # i.e.: no breakpoints match (stay in untraced mode)
    breakpoint_found, force_stay_in_untraced_mode = cached_value.compute_force_stay_in_untraced_mode(
        create_breakpoints_dict([method.__code__.co_firstlineno + 10]))

    assert not breakpoint_found
    assert force_stay_in_untraced_mode

    # i.e.: one of the breakpoints match (stay in untraced mode)
    breakpoint_found, force_stay_in_untraced_mode = cached_value.compute_force_stay_in_untraced_mode(
        create_breakpoints_dict([method.__code__.co_firstlineno + 2]))

    assert breakpoint_found
    assert force_stay_in_untraced_mode

    # i.e.: one of the breakpoints doesn't match (leave untraced mode)
    breakpoint_found, force_stay_in_untraced_mode = cached_value.compute_force_stay_in_untraced_mode(
        create_breakpoints_dict([method.__code__.co_firstlineno + 3]))

    assert breakpoint_found
    assert not force_stay_in_untraced_mode
