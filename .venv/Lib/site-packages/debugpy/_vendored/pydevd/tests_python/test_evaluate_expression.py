from _pydevd_bundle.pydevd_constants import IS_PY38_OR_GREATER, NULL
from _pydevd_bundle.pydevd_xml import ExceptionOnEvaluate

import sys
import pytest

SOME_LST = ["foo", "bar"]
BAR = "bar"
FOO = "foo"
global_frame = sys._getframe()


def obtain_frame():
    A = 1
    B = 2
    yield sys._getframe()


@pytest.fixture
def disable_critical_log():
    # We want to hide the logging related to _evaluate_with_timeouts not receiving the py_db.
    from _pydev_bundle.pydev_log import log_context
    import io
    stream = io.StringIO()
    with log_context(0, stream):
        yield


def test_evaluate_expression_basic(disable_critical_log):
    from _pydevd_bundle.pydevd_vars import evaluate_expression

    def check(frame):
        evaluate_expression(None, frame, 'some_var = 1', is_exec=True)

        assert frame.f_locals['some_var'] == 1

    check(next(iter(obtain_frame())))
    assert 'some_var' not in sys._getframe().f_globals

    # as locals == globals, this will also change the current globals
    check(global_frame)
    assert 'some_var' in sys._getframe().f_globals
    del sys._getframe().f_globals['some_var']
    assert 'some_var' not in sys._getframe().f_globals


def test_evaluate_expression_1(disable_critical_log):
    from _pydevd_bundle.pydevd_vars import evaluate_expression

    def check(frame):
        eval_txt = '''
container = ["abc","efg"]
results = []
for s in container:
    result = [s[i] for i in range(3)]
    results.append(result)
'''
        evaluate_expression(None, frame, eval_txt, is_exec=True)
        assert frame.f_locals['results'] == [['a', 'b', 'c'], ['e', 'f', 'g']]
        assert frame.f_locals['s'] == "efg"

    check(next(iter(obtain_frame())))

    for varname in ['container', 'results', 's']:
        assert varname not in sys._getframe().f_globals

    check(global_frame)
    for varname in ['container', 'results', 's']:
        assert varname in sys._getframe().f_globals

    for varname in ['container', 'results', 's']:
        del sys._getframe().f_globals[varname]


def test_evaluate_expression_2(disable_critical_log):
    from _pydevd_bundle.pydevd_vars import evaluate_expression

    def check(frame):
        eval_txt = 'all((x in (BAR, FOO) for x in SOME_LST))'
        assert evaluate_expression(None, frame, eval_txt, is_exec=False)

    check(next(iter(obtain_frame())))
    check(global_frame)


def test_evaluate_expression_3(disable_critical_log):
    if not IS_PY38_OR_GREATER:
        return

    from _pydevd_bundle.pydevd_vars import evaluate_expression

    def check(frame):
        eval_txt = '''11 if (some_var := 22) else 33'''
        assert evaluate_expression(None, frame, eval_txt, is_exec=False) == 11

    check(next(iter(obtain_frame())))
    assert 'some_var' not in sys._getframe().f_globals

    # as locals == globals, this will also change the current globals
    check(global_frame)
    assert 'some_var' in sys._getframe().f_globals
    del sys._getframe().f_globals['some_var']
    assert 'some_var' not in sys._getframe().f_globals


def test_evaluate_expression_4(disable_critical_log):
    from _pydevd_bundle.pydevd_vars import evaluate_expression

    def check(frame):
        eval_txt = '''import email;email.foo_value'''
        with pytest.raises(AttributeError):
            evaluate_expression(None, frame, eval_txt, is_exec=True)
        assert 'email' in frame.f_locals

    check(next(iter(obtain_frame())))
    assert 'email' not in sys._getframe().f_globals

    # as locals == globals, this will also change the current globals
    check(global_frame)
    assert 'email' in sys._getframe().f_globals
    del sys._getframe().f_globals['email']
    assert 'email' not in sys._getframe().f_globals


def test_evaluate_expression_access_globals(disable_critical_log):
    from _pydevd_bundle.pydevd_vars import evaluate_expression

    def check(frame):
        eval_txt = '''globals()['global_variable'] = 22'''
        evaluate_expression(None, frame, eval_txt, is_exec=True)
        assert 'global_variable' not in frame.f_locals
        assert 'global_variable' in frame.f_globals

    check(next(iter(obtain_frame())))
    assert 'global_variable' in sys._getframe().f_globals
    assert 'global_variable' not in sys._getframe().f_locals


def test_evaluate_expression_create_none(disable_critical_log):
    from _pydevd_bundle.pydevd_vars import evaluate_expression

    def check(frame):
        eval_txt = 'x = None'
        evaluate_expression(None, frame, eval_txt, is_exec=True)
        assert 'x' in frame.f_locals
        assert 'x' not in frame.f_globals

    check(next(iter(obtain_frame())))


def test_evaluate_expression_delete_var(disable_critical_log):
    from _pydevd_bundle.pydevd_vars import evaluate_expression

    def check(frame):
        eval_txt = 'x = 22'
        evaluate_expression(None, frame, eval_txt, is_exec=True)
        assert 'x' in frame.f_locals

        eval_txt = 'del x'
        evaluate_expression(None, frame, eval_txt, is_exec=True)
        assert 'x' not in frame.f_locals

    check(next(iter(obtain_frame())))


def test_evaluate_expression_5(disable_critical_log):
    from _pydevd_bundle.pydevd_vars import evaluate_expression

    def check(frame):
        eval_txt = 'A, B = 5, 6'
        evaluate_expression(None, frame, eval_txt, is_exec=True)
        assert frame.f_locals['A'] == 5
        assert frame.f_locals['B'] == 6

    check(next(iter(obtain_frame())))


class _DummyPyDB(object):

    def __init__(self):
        self.created_pydb_daemon_threads = {}
        self.timeout_tracker = NULL
        self.multi_threads_single_notification = False


try:
    from ast import PyCF_ALLOW_TOP_LEVEL_AWAIT  # @UnusedImport
    CAN_EVALUATE_TOP_LEVEL_ASYNC = True
except:
    CAN_EVALUATE_TOP_LEVEL_ASYNC = False


@pytest.mark.skipif(not CAN_EVALUATE_TOP_LEVEL_ASYNC, reason='Requires top-level async evaluation.')
def test_evaluate_expression_async_exec(disable_critical_log):
    py_db = _DummyPyDB()

    async def async_call(a):
        return a

    async def main():
        from _pydevd_bundle.pydevd_vars import evaluate_expression
        a = 10
        assert async_call is not None  # Make sure it's in the locals.
        frame = sys._getframe()
        eval_txt = 'y = await async_call(a)'
        evaluate_expression(py_db, frame, eval_txt, is_exec=True)
        assert frame.f_locals['y'] == a

    import asyncio
    asyncio.run(main())


@pytest.mark.skipif(not CAN_EVALUATE_TOP_LEVEL_ASYNC, reason='Requires top-level async evaluation.')
def test_evaluate_expression_async_exec_as_eval(disable_critical_log):
    py_db = _DummyPyDB()

    async def async_call(a):
        return a

    async def main():
        from _pydevd_bundle.pydevd_vars import evaluate_expression
        assert async_call is not None  # Make sure it's in the locals.
        frame = sys._getframe()
        eval_txt = 'await async_call(10)'
        from io import StringIO
        _original_stdout = sys.stdout
        try:
            stringio = sys.stdout = StringIO()
            evaluate_expression(py_db, frame, eval_txt, is_exec=True)
        finally:
            sys.stdout = _original_stdout

        # I.e.: Check that we printed the value obtained in the exec.
        assert '10\n' in stringio.getvalue()

    import asyncio
    asyncio.run(main())


@pytest.mark.skipif(not CAN_EVALUATE_TOP_LEVEL_ASYNC, reason='Requires top-level async evaluation.')
def test_evaluate_expression_async_exec_error(disable_critical_log):
    py_db = _DummyPyDB()

    async def async_call(a):
        raise RuntimeError('foobar')

    async def main():
        from _pydevd_bundle.pydevd_vars import evaluate_expression
        assert async_call is not None  # Make sure it's in the locals.
        frame = sys._getframe()
        eval_txt = 'y = await async_call(10)'
        with pytest.raises(RuntimeError) as e:
            evaluate_expression(py_db, frame, eval_txt, is_exec=True)
            assert 'foobar' in str(e)
        assert 'y' not in frame.f_locals

    import asyncio
    asyncio.run(main())


@pytest.mark.skipif(not CAN_EVALUATE_TOP_LEVEL_ASYNC, reason='Requires top-level async evaluation.')
def test_evaluate_expression_async_eval(disable_critical_log):
    py_db = _DummyPyDB()

    async def async_call(a):
        return a

    async def main():
        from _pydevd_bundle.pydevd_vars import evaluate_expression
        a = 10
        assert async_call is not None  # Make sure it's in the locals.
        frame = sys._getframe()
        eval_txt = 'await async_call(a)'
        v = evaluate_expression(py_db, frame, eval_txt, is_exec=False)
        if isinstance(v, ExceptionOnEvaluate):
            raise v.result.with_traceback(v.tb)
        assert v == a

    import asyncio
    asyncio.run(main())


@pytest.mark.skipif(not CAN_EVALUATE_TOP_LEVEL_ASYNC, reason='Requires top-level async evaluation.')
def test_evaluate_expression_async_eval_error(disable_critical_log):
    py_db = _DummyPyDB()

    async def async_call(a):
        raise RuntimeError('foobar')

    async def main():
        from _pydevd_bundle.pydevd_vars import evaluate_expression
        a = 10
        assert async_call is not None  # Make sure it's in the locals.
        frame = sys._getframe()
        eval_txt = 'await async_call(a)'
        v = evaluate_expression(py_db, frame, eval_txt, is_exec=False)
        assert isinstance(v, ExceptionOnEvaluate)
        assert 'foobar' in str(v.result)

    import asyncio
    asyncio.run(main())


def test_evaluate_expression_name_mangling(disable_critical_log):
    from _pydevd_bundle.pydevd_vars import evaluate_expression

    class SomeObj(object):

        def __init__(self):
            self.__value = 10
            self.frame = sys._getframe()

    obj = SomeObj()
    frame = obj.frame

    eval_txt = '''self.__value'''
    v = evaluate_expression(None, frame, eval_txt, is_exec=False)
    if isinstance(v, ExceptionOnEvaluate):
        raise v.result.with_traceback(v.tb)

    assert v == 10
