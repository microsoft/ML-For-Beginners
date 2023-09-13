from _pydev_runfiles import pydev_runfiles_xml_rpc
import pickle
import zlib
import base64
import os
from pydevd_file_utils import canonical_normalized_path
import pytest
import sys
import time
from pathlib import Path

#=========================================================================
# Load filters with tests we should skip
#=========================================================================
py_test_accept_filter = None


def _load_filters():
    global py_test_accept_filter
    if py_test_accept_filter is None:
        py_test_accept_filter = os.environ.get('PYDEV_PYTEST_SKIP')
        if py_test_accept_filter:
            py_test_accept_filter = pickle.loads(
                zlib.decompress(base64.b64decode(py_test_accept_filter)))

            # Newer versions of pytest resolve symlinks, so, we
            # may need to filter with a resolved path too.
            new_dct = {}
            for filename, value in py_test_accept_filter.items():
                new_dct[canonical_normalized_path(str(Path(filename).resolve()))] = value

            py_test_accept_filter.update(new_dct)

        else:
            py_test_accept_filter = {}


def is_in_xdist_node():
    main_pid = os.environ.get('PYDEV_MAIN_PID')
    if main_pid and main_pid != str(os.getpid()):
        return True
    return False


connected = False


def connect_to_server_for_communication_to_xml_rpc_on_xdist():
    global connected
    if connected:
        return
    connected = True
    if is_in_xdist_node():
        port = os.environ.get('PYDEV_PYTEST_SERVER')
        if not port:
            sys.stderr.write(
                'Error: no PYDEV_PYTEST_SERVER environment variable defined.\n')
        else:
            pydev_runfiles_xml_rpc.initialize_server(int(port), daemon=True)


PY2 = sys.version_info[0] <= 2
PY3 = not PY2


class State:
    start_time = time.time()
    buf_err = None
    buf_out = None


def start_redirect():
    if State.buf_out is not None:
        return
    from _pydevd_bundle import pydevd_io
    State.buf_err = pydevd_io.start_redirect(keep_original_redirection=True, std='stderr')
    State.buf_out = pydevd_io.start_redirect(keep_original_redirection=True, std='stdout')


def get_curr_output():
    buf_out = State.buf_out
    buf_err = State.buf_err
    return buf_out.getvalue() if buf_out is not None else '', buf_err.getvalue() if buf_err is not None else ''


def pytest_unconfigure():
    if is_in_xdist_node():
        return
    # Only report that it finished when on the main node (we don't want to report
    # the finish on each separate node).
    pydev_runfiles_xml_rpc.notifyTestRunFinished(
        'Finished in: %.2f secs.' % (time.time() - State.start_time,))


def pytest_collection_modifyitems(session, config, items):
    # A note: in xdist, this is not called on the main process, only in the
    # secondary nodes, so, we'll actually make the filter and report it multiple
    # times.
    connect_to_server_for_communication_to_xml_rpc_on_xdist()

    _load_filters()
    if not py_test_accept_filter:
        pydev_runfiles_xml_rpc.notifyTestsCollected(len(items))
        return  # Keep on going (nothing to filter)

    new_items = []
    for item in items:
        f = canonical_normalized_path(str(item.parent.fspath))
        name = item.name

        if f not in py_test_accept_filter:
            # print('Skip file: %s' % (f,))
            continue  # Skip the file

        i = name.find('[')
        name_without_parametrize = None
        if i > 0:
            name_without_parametrize = name[:i]

        accept_tests = py_test_accept_filter[f]

        if item.cls is not None:
            class_name = item.cls.__name__
        else:
            class_name = None
        for test in accept_tests:
            if test == name:
                # Direct match of the test (just go on with the default
                # loading)
                new_items.append(item)
                break

            if name_without_parametrize is not None and test == name_without_parametrize:
                # This happens when parameterizing pytest tests on older versions
                # of pytest where the test name doesn't include the fixture name
                # in it.
                new_items.append(item)
                break

            if class_name is not None:
                if test == class_name + '.' + name:
                    new_items.append(item)
                    break

                if name_without_parametrize is not None and test == class_name + '.' + name_without_parametrize:
                    new_items.append(item)
                    break

                if class_name == test:
                    new_items.append(item)
                    break
        else:
            pass
            # print('Skip test: %s.%s. Accept: %s' % (class_name, name, accept_tests))

    # Modify the original list
    items[:] = new_items
    pydev_runfiles_xml_rpc.notifyTestsCollected(len(items))


try:
    """
    pytest > 5.4 uses own version of TerminalWriter based on py.io.TerminalWriter
    and assumes there is a specific method TerminalWriter._write_source
    so try load pytest version first or fallback to default one
    """
    from _pytest._io import TerminalWriter
except ImportError:
    from py.io import TerminalWriter


def _get_error_contents_from_report(report):
    if report.longrepr is not None:
        try:
            tw = TerminalWriter(stringio=True)
            stringio = tw.stringio
        except TypeError:
            import io
            stringio = io.StringIO()
            tw = TerminalWriter(file=stringio)
        tw.hasmarkup = False
        report.toterminal(tw)
        exc = stringio.getvalue()
        s = exc.strip()
        if s:
            return s

    return ''


def pytest_collectreport(report):
    error_contents = _get_error_contents_from_report(report)
    if error_contents:
        report_test('fail', '<collect errors>', '<collect errors>', '', error_contents, 0.0)


def append_strings(s1, s2):
    if s1.__class__ == s2.__class__:
        return s1 + s2

    # Prefer str
    if isinstance(s1, bytes):
        s1 = s1.decode('utf-8', 'replace')

    if isinstance(s2, bytes):
        s2 = s2.decode('utf-8', 'replace')

    return s1 + s2


def pytest_runtest_logreport(report):
    if is_in_xdist_node():
        # When running with xdist, we don't want the report to be called from the node, only
        # from the main process.
        return
    report_duration = report.duration
    report_when = report.when
    report_outcome = report.outcome

    if hasattr(report, 'wasxfail'):
        if report_outcome != 'skipped':
            report_outcome = 'passed'

    if report_outcome == 'passed':
        # passed on setup/teardown: no need to report if in setup or teardown
        # (only on the actual test if it passed).
        if report_when in ('setup', 'teardown'):
            return

        status = 'ok'

    elif report_outcome == 'skipped':
        status = 'skip'

    else:
        # It has only passed, skipped and failed (no error), so, let's consider
        # error if not on call.
        if report_when in ('setup', 'teardown'):
            status = 'error'

        else:
            # any error in the call (not in setup or teardown) is considered a
            # regular failure.
            status = 'fail'

    # This will work if pytest is not capturing it, if it is, nothing will
    # come from here...
    captured_output, error_contents = getattr(report, 'pydev_captured_output', ''), getattr(report, 'pydev_error_contents', '')
    for type_section, value in report.sections:
        if value:
            if type_section in ('err', 'stderr', 'Captured stderr call'):
                error_contents = append_strings(error_contents, value)
            else:
                captured_output = append_strings(error_contents, value)

    filename = getattr(report, 'pydev_fspath_strpath', '<unable to get>')
    test = report.location[2]

    if report_outcome != 'skipped':
        # On skipped, we'll have a traceback for the skip, which is not what we
        # want.
        exc = _get_error_contents_from_report(report)
        if exc:
            if error_contents:
                error_contents = append_strings(error_contents, '----------------------------- Exceptions -----------------------------\n')
            error_contents = append_strings(error_contents, exc)

    report_test(status, filename, test, captured_output, error_contents, report_duration)


def report_test(status, filename, test, captured_output, error_contents, duration):
    '''
    @param filename: 'D:\\src\\mod1\\hello.py'
    @param test: 'TestCase.testMet1'
    @param status: fail, error, ok
    '''
    time_str = '%.2f' % (duration,)
    pydev_runfiles_xml_rpc.notifyTest(
        status, captured_output, error_contents, filename, test, time_str)


if not hasattr(pytest, 'hookimpl'):
    raise AssertionError('Please upgrade pytest (the current version of pytest: %s is unsupported)' % (pytest.__version__,))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    report.pydev_fspath_strpath = item.fspath.strpath
    report.pydev_captured_output, report.pydev_error_contents = get_curr_output()


@pytest.mark.tryfirst
def pytest_runtest_setup(item):
    '''
    Note: with xdist will be on a secondary process.
    '''
    # We have our own redirection: if xdist does its redirection, we'll have
    # nothing in our contents (which is OK), but if it does, we'll get nothing
    # from pytest but will get our own here.
    start_redirect()
    filename = item.fspath.strpath
    test = item.location[2]

    pydev_runfiles_xml_rpc.notifyStartTest(filename, test)
