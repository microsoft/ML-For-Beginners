import threading

from _pydevd_bundle.pydevd_utils import convert_dap_log_message_to_expression
from tests_python.debug_constants import TEST_GEVENT, IS_CPYTHON
import sys
from _pydevd_bundle.pydevd_constants import IS_WINDOWS, IS_PYPY, IS_JYTHON
import pytest
import os
from _pydevd_bundle.pydevd_thread_lifecycle import pydevd_find_thread_by_id


def test_expression_to_evaluate():
    from _pydevd_bundle.pydevd_vars import _expression_to_evaluate
    assert _expression_to_evaluate(b'expr') == b'expr'
    assert _expression_to_evaluate(b'  expr') == b'expr'
    assert _expression_to_evaluate(b'for a in b:\n  foo') == b'for a in b:\n  foo'
    assert _expression_to_evaluate(b'  for a in b:\n  foo') == b'for a in b:\nfoo'
    assert _expression_to_evaluate(b'  for a in b:\nfoo') == b'  for a in b:\nfoo'
    assert _expression_to_evaluate(b'\tfor a in b:\n\t\tfoo') == b'for a in b:\n\tfoo'

    assert _expression_to_evaluate(u'  expr') == u'expr'
    assert _expression_to_evaluate(u'  for a in expr:\n  pass') == u'for a in expr:\npass'


@pytest.mark.skipif(IS_WINDOWS, reason='Brittle on Windows.')
def test_is_main_thread():
    '''
    This is now skipped due to it failing sometimes (only on Windows).

    I (fabioz) am not 100% sure on why this happens, but when this happens the initial thread for
    the tests seems to be a non main thread.

    i.e.: With an autouse fixture with a scope='session' with the code and error message below, it's
    possible to see that at even at the `conftest` import (where indent_at_import is assigned) the
    current thread is already not the main thread.

    As far as I know this seems to be an issue in how pytest-xdist is running the tests (i.e.:
    I couldn't reproduce this without running with `python -m pytest -n 0 ...`).

    -------- Code to check error / error output ----------

    from _pydevd_bundle.pydevd_utils import is_current_thread_main_thread
    import threading
    indent_at_import = threading.get_ident()

    @pytest.yield_fixture(autouse=True, scope='session')
    def check_main_thread_session(request):
        if not is_current_thread_main_thread():
            error_msg = 'Current thread does not seem to be a main thread at the start of the session. Details:\n'
            current_thread = threading.current_thread()
            error_msg += 'Current thread: %s\n' % (current_thread,)
            error_msg += 'Current thread ident: %s\n' % (current_thread.ident,)
            error_msg += 'ident at import: %s\n' % (indent_at_import,)
            error_msg += 'curr ident: %s\n' % (threading.get_ident(),)

            if hasattr(threading, 'main_thread'):
                error_msg += 'Main thread found: %s\n' % (threading.main_thread(),)
                error_msg += 'Main thread id: %s\n' % (threading.main_thread().ident,)
            else:
                error_msg += 'Current main thread not instance of: %s (%s)\n' % (
                    threading._MainThread, current_thread.__class__.__mro__,)

>           raise AssertionError(error_msg)
E           AssertionError: Current thread does not seem to be a main thread at the start of the session. Details:
E           Current thread: <_DummyThread(Dummy-2, started daemon 7072)>
E           Current thread ident: 7072
E           ident at import: 7072
E           curr ident: 7072
E           Main thread found: <_MainThread(MainThread, started 5924)>
E           Main thread id: 5924

conftest.py:67: AssertionError
    '''
    from _pydevd_bundle.pydevd_utils import is_current_thread_main_thread
    from _pydevd_bundle.pydevd_utils import dump_threads
    if not is_current_thread_main_thread():
        error_msg = 'Current thread does not seem to be a main thread. Details:\n'
        current_thread = threading.current_thread()
        error_msg += 'Current thread: %s\n' % (current_thread,)

        if hasattr(threading, 'main_thread'):
            error_msg += 'Main thread found: %s\n' % (threading.main_thread(),)
        else:
            error_msg += 'Current main thread not instance of: %s (%s)' % (
                threading._MainThread, current_thread.__class__.__mro__,)

        from io import StringIO

        stream = StringIO()
        dump_threads(stream=stream)
        error_msg += '\n\n' + stream.getvalue()
        raise AssertionError(error_msg)

    class NonMainThread(threading.Thread):

        def run(self):
            self.is_main_thread = is_current_thread_main_thread()

    non_main_thread = NonMainThread()
    non_main_thread.start()
    non_main_thread.join()
    assert not non_main_thread.is_main_thread


def test_find_thread():
    from _pydevd_bundle.pydevd_constants import get_current_thread_id
    assert pydevd_find_thread_by_id('123') is None

    assert pydevd_find_thread_by_id(
        get_current_thread_id(threading.current_thread())) is threading.current_thread()


def check_dap_log_message(log_message, expected, evaluated, eval_locals=None):
    ret = convert_dap_log_message_to_expression(log_message)
    assert ret == expected
    assert (eval(ret, eval_locals)) == evaluated
    return ret


def test_convert_dap_log_message_to_expression():
    assert check_dap_log_message(
        'a',
        "'a'",
        'a',
    )
    assert check_dap_log_message(
        'a {a}',
        "'a %s' % (a,)",
        'a value',
        {'a': 'value'}
    )
    assert check_dap_log_message(
        'a {1}',
        "'a %s' % (1,)",
        'a 1'
    )
    assert check_dap_log_message(
        'a {  }',
        "'a '",
        'a '
    )
    assert check_dap_log_message(
        'a {1} {2}',
        "'a %s %s' % (1, 2,)",
        'a 1 2',
    )
    assert check_dap_log_message(
        'a {{22:22}} {2}',
        "'a %s %s' % ({22:22}, 2,)",
        'a {22: 22} 2'
    )
    assert check_dap_log_message(
        'a {(22,33)}} {2}',
        "'a %s} %s' % ((22,33), 2,)",
        'a (22, 33)} 2'
    )

    assert check_dap_log_message(
        'a {{1: {1}}}',
        "'a %s' % ({1: {1}},)",
        'a {1: {1}}'
    )

    # Error condition.
    assert check_dap_log_message(
        'a {{22:22} {2}',
        "'Unbalanced braces in: a {{22:22} {2}'",
        'Unbalanced braces in: a {{22:22} {2}'
    )


def test_pydevd_log():
    from _pydev_bundle import pydev_log
    import io
    from _pydev_bundle.pydev_log import log_context

    stream = io.StringIO()
    with log_context(0, stream=stream):
        pydev_log.critical('always')
        pydev_log.info('never')

    assert stream.getvalue().endswith('always\n')

    stream = io.StringIO()
    with log_context(1, stream=stream):
        pydev_log.critical('always')
        pydev_log.info('this too')
        assert stream.getvalue().endswith('always\n0.00s - this too\n')

    stream = io.StringIO()
    with log_context(0, stream=stream):
        pydev_log.critical('always %s', 1)

    assert stream.getvalue().endswith('always 1\n')

    stream = io.StringIO()
    with log_context(0, stream=stream):
        pydev_log.critical('always %s %s', 1, 2)

    assert stream.getvalue().endswith('always 1 2\n')

    stream = io.StringIO()
    with log_context(0, stream=stream):
        pydev_log.critical('always %s %s', 1)

    # Even if there's an error in the formatting, don't fail, just print the message and args.
    assert stream.getvalue().endswith('always %s %s - (1,)\n')

    stream = io.StringIO()
    with log_context(0, stream=stream):
        try:
            raise RuntimeError()
        except:
            pydev_log.exception('foo')

        assert 'foo\n' in stream.getvalue()
        assert 'raise RuntimeError()' in stream.getvalue()

    stream = io.StringIO()
    with log_context(0, stream=stream):
        pydev_log.error_once('always %s %s', 1)

    # Even if there's an error in the formatting, don't fail, just print the message and args.
    assert stream.getvalue().endswith('always %s %s - (1,)\n')


def test_pydevd_logging_files(tmpdir):
    from _pydev_bundle import pydev_log
    from _pydevd_bundle.pydevd_constants import DebugInfoHolder
    import os.path
    from _pydev_bundle.pydev_log import _LoggingGlobals

    import io
    from _pydev_bundle.pydev_log import log_context

    stream = io.StringIO()
    with log_context(0, stream=stream):
        d1 = str(tmpdir.join('d1'))
        d2 = str(tmpdir.join('d2'))

        for d in (d1, d2):
            DebugInfoHolder.PYDEVD_DEBUG_FILE = os.path.join(d, 'file.txt')
            pydev_log.initialize_debug_stream(reinitialize=True)

            assert os.path.normpath(_LoggingGlobals._debug_stream_filename) == \
                os.path.normpath(os.path.join(d, 'file.%s.txt' % os.getpid()))

            assert os.path.exists(_LoggingGlobals._debug_stream_filename)

            assert pydev_log.list_log_files(DebugInfoHolder.PYDEVD_DEBUG_FILE) == [
                _LoggingGlobals._debug_stream_filename]


def _check_tracing_other_threads():
    import pydevd_tracing
    import time
    from tests_python.debugger_unittest import wait_for_condition
    import _thread

    # This method is called in a subprocess, so, make sure we exit properly even if we somehow
    # deadlock somewhere else.
    def dump_threads_and_kill_on_timeout():
        time.sleep(10)
        from _pydevd_bundle import pydevd_utils
        pydevd_utils.dump_threads()
        time.sleep(1)
        import os
        os._exit(77)

    _thread.start_new_thread(dump_threads_and_kill_on_timeout, ())

    def method():
        while True:
            trace_func = sys.gettrace()
            if trace_func:
                threading.current_thread().trace_func = trace_func
                break
            time.sleep(.01)

    def dummy_thread_method():
        threads.append(threading.current_thread())
        method()

    threads = []
    threads.append(threading.Thread(target=method))
    threads[-1].daemon = True
    threads[-1].start()
    _thread.start_new_thread(dummy_thread_method, ())

    wait_for_condition(lambda: len(threads) == 2, msg=lambda:'Found threads: %s' % (threads,))

    def tracing_func(frame, event, args):
        return tracing_func

    assert pydevd_tracing.set_trace_to_threads(tracing_func) == 0

    def check_threads_tracing_func():
        for t in threads:
            if getattr(t, 'trace_func', None) != tracing_func:
                return False
        return True

    wait_for_condition(check_threads_tracing_func)

    assert tracing_func == sys.gettrace()


def _build_launch_env():
    import os
    import pydevd

    environ = os.environ.copy()
    cwd = os.path.abspath(os.path.dirname(__file__))
    assert os.path.isdir(cwd)

    resources_dir = os.path.join(os.path.dirname(pydevd.__file__), 'tests_python', 'resources')
    assert os.path.isdir(resources_dir)

    attach_to_process_dir = os.path.join(os.path.dirname(pydevd.__file__), 'pydevd_attach_to_process')
    assert os.path.isdir(attach_to_process_dir)

    pydevd_dir = os.path.dirname(pydevd.__file__)
    assert os.path.isdir(pydevd_dir)

    environ['PYTHONPATH'] = (
            cwd + os.pathsep +
            resources_dir + os.pathsep +
            attach_to_process_dir + os.pathsep +
            pydevd_dir + os.pathsep +
            environ.get('PYTHONPATH', '')
    )
    return cwd, environ


def _check_in_separate_process(method_name, module_name='test_utilities', update_env={}):
    import subprocess
    cwd, environ = _build_launch_env()
    environ.update(update_env)

    subprocess.check_call(
        [sys.executable, '-c', 'import %(module_name)s;%(module_name)s.%(method_name)s()' % dict(
            method_name=method_name, module_name=module_name)],
        env=environ,
        cwd=cwd
    )


@pytest.mark.skipif(not IS_CPYTHON, reason='Functionality to trace other threads requires CPython.')
def test_tracing_other_threads():
    # Note: run this test in a separate process so that it doesn't mess with any current tracing
    # in our current process.
    _check_in_separate_process('_check_tracing_other_threads')


def _check_basic_tracing():
    import pydevd_tracing

    # Note: run this test in a separate process so that it doesn't mess with any current tracing
    # in our current process.
    called = [0]

    def tracing_func(frame, event, args):
        called[0] = called[0] + 1
        return tracing_func

    assert pydevd_tracing.set_trace_to_threads(tracing_func) == 0

    def foo():
        pass

    foo()
    assert called[0] > 2


@pytest.mark.skipif(not IS_CPYTHON, reason='Functionality to trace other threads requires CPython.')
def test_tracing_basic():
    _check_in_separate_process('_check_basic_tracing')


@pytest.mark.skipif(not IS_CPYTHON, reason='Functionality to trace other threads requires CPython.')
def test_find_main_thread_id():
    # Note: run the checks below in a separate process because they rely heavily on what's available
    # in the env (such as threads or having threading imported).
    _check_in_separate_process('check_main_thread_id_simple', '_pydevd_test_find_main_thread_id')
    _check_in_separate_process('check_main_thread_id_multiple_threads', '_pydevd_test_find_main_thread_id')
    _check_in_separate_process('check_win_threads', '_pydevd_test_find_main_thread_id')
    _check_in_separate_process('check_fix_main_thread_id_multiple_threads', '_pydevd_test_find_main_thread_id')

    import subprocess
    import pydevd
    cwd, environ = _build_launch_env()

    subprocess.check_call(
        [sys.executable, '-m', '_pydevd_test_find_main_thread_id'],
        env=environ,
        cwd=cwd
    )

    resources_dir = os.path.join(os.path.dirname(pydevd.__file__), 'tests_python', 'resources')

    subprocess.check_call(
        [sys.executable, os.path.join(resources_dir, '_pydevd_test_find_main_thread_id.py') ],
        env=environ,
        cwd=cwd
    )


@pytest.mark.skipif(not IS_WINDOWS or IS_JYTHON, reason='Windows-only test.')
def test_get_ppid():
    from _pydevd_bundle.pydevd_api import PyDevdAPI
    api = PyDevdAPI()
    # On python 3 we can check that our internal api which is used for Python 2 gives the
    # same result as os.getppid.
    ppid = os.getppid()
    assert api._get_windows_ppid() == ppid


def _check_gevent(expect_msg):
    from _pydevd_bundle.pydevd_utils import notify_about_gevent_if_needed
    assert not notify_about_gevent_if_needed()
    import gevent
    assert not notify_about_gevent_if_needed()
    import gevent.monkey
    assert not notify_about_gevent_if_needed()
    gevent.monkey.patch_all()
    assert notify_about_gevent_if_needed() == expect_msg


def check_notify_on_gevent_loaded():
    _check_gevent(True)


def check_dont_notify_on_gevent_loaded():
    _check_gevent(False)


@pytest.mark.skipif(not TEST_GEVENT, reason='Gevent not installed.')
def test_gevent_notify():
    _check_in_separate_process('check_notify_on_gevent_loaded', update_env={'GEVENT_SUPPORT': ''})
    _check_in_separate_process('check_dont_notify_on_gevent_loaded', update_env={'GEVENT_SUPPORT': 'True'})


@pytest.mark.skipif(True, reason='Skipping because running this test can interrupt the test suite execution.')
def test_interrupt_main_thread():
    from _pydevd_bundle.pydevd_utils import interrupt_main_thread
    import time

    main_thread = threading.current_thread()

    def interrupt():
        # sleep here so that the main thread in the test can get to the sleep too (otherwise
        # if we interrupt too fast we won't really check that the sleep itself
        # got interrupted -- although if that happens on some tests runs it's
        # not really an issue either).
        time.sleep(1)
        interrupt_main_thread(main_thread)

    if IS_PYPY:
        # On PyPy a time.sleep() is not being properly interrupted,
        # so, let's just check that it throws the KeyboardInterrupt in the
        # next instruction.
        timeout = 2
    else:
        timeout = 20
    initial_time = time.time()
    try:
        t = threading.Thread(target=interrupt)
        t.start()
        time.sleep(timeout)
    except KeyboardInterrupt:
        if not IS_PYPY:
            actual_timeout = time.time() - initial_time
            # If this fails it means that although we interrupted Python actually waited for the next
            # instruction to send the event and didn't really interrupt the thread.
            assert actual_timeout < timeout, 'Expected the actual timeout (%s) to be < than the timeout (%s)' % (
                actual_timeout, timeout)
    else:
        raise AssertionError('KeyboardInterrupt not generated in main thread.')


@pytest.mark.skipif(sys.version_info[0] < 3, reason='Only available for Python 3.')
def test_get_smart_step_into_variant_from_frame_offset():
    from _pydevd_bundle.pydevd_bytecode_utils import get_smart_step_into_variant_from_frame_offset
    variants = []
    assert get_smart_step_into_variant_from_frame_offset(0, variants) is None

    class DummyVariant(object):

        def __init__(self, offset):
            self.offset = offset

    variants.append(DummyVariant(1))
    assert get_smart_step_into_variant_from_frame_offset(0, variants) is None
    assert get_smart_step_into_variant_from_frame_offset(1, variants) is variants[0]
    assert get_smart_step_into_variant_from_frame_offset(2, variants) is variants[0]

    variants.append(DummyVariant(3))
    assert get_smart_step_into_variant_from_frame_offset(0, variants) is None
    assert get_smart_step_into_variant_from_frame_offset(1, variants) is variants[0]
    assert get_smart_step_into_variant_from_frame_offset(2, variants) is variants[0]
    assert get_smart_step_into_variant_from_frame_offset(3, variants) is variants[1]
    assert get_smart_step_into_variant_from_frame_offset(4, variants) is variants[1]


def test_threading_hide_pydevd():

    class T(threading.Thread):

        def __init__(self, is_pydev_daemon_thread):
            from _pydevd_bundle.pydevd_daemon_thread import mark_as_pydevd_daemon_thread
            threading.Thread.__init__(self)
            if is_pydev_daemon_thread:
                mark_as_pydevd_daemon_thread(self)
            else:
                self.daemon = True
            self.event = threading.Event()

        def run(self):
            self.event.wait(10)

    current_count = threading.active_count()
    t0 = T(True)
    t1 = T(False)
    t0.start()
    t1.start()

    # i.e.: the patching doesn't work for other implementations.
    if IS_CPYTHON:
        assert threading.active_count() == current_count + 1
        assert t0 not in threading.enumerate()
    else:
        assert threading.active_count() == current_count + 2
        assert t0 in threading.enumerate()

    assert t1 in threading.enumerate()
    t0.event.set()
    t1.event.set()


def test_import_token_from_module():
    from _pydevd_bundle.pydevd_utils import import_attr_from_module

    with pytest.raises(ImportError):
        import_attr_from_module('sys')

    with pytest.raises(ImportError):
        import_attr_from_module('sys.settrace.foo')

    assert import_attr_from_module('sys.settrace') == sys.settrace
    assert import_attr_from_module('threading.Thread.start') == threading.Thread.start
