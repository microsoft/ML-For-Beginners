import pytest
import sys
from _pydevd_bundle.pydevd_constants import IS_JYTHON, IS_IRONPYTHON
from tests_python.debug_constants import TEST_CYTHON
from tests_python.debug_constants import PYDEVD_TEST_VM
import site
import os
from _pydev_bundle import pydev_log


def pytest_report_header(config):
    print('PYDEVD_USE_CYTHON: %s' % (TEST_CYTHON,))
    print('PYDEVD_TEST_VM: %s' % (PYDEVD_TEST_VM,))
    try:
        import multiprocessing
    except ImportError:
        pass
    else:
        print('Number of processors: %s' % (multiprocessing.cpu_count(),))

    print('Relevant system paths:')
    print('sys.executable: %s' % (sys.executable,))
    print('sys.prefix: %s' % (sys.prefix,))

    if hasattr(sys, 'base_prefix'):
        print('sys.base_prefix: %s' % (sys.base_prefix,))

    if hasattr(sys, 'real_prefix'):
        print('sys.real_prefix: %s' % (sys.real_prefix,))

    if hasattr(site, 'getusersitepackages'):
        print('site.getusersitepackages(): %s' % (site.getusersitepackages(),))

    if hasattr(site, 'getsitepackages'):
        print('site.getsitepackages(): %s' % (site.getsitepackages(),))

    for path in sys.path:
        if os.path.exists(path) and os.path.basename(path) == 'site-packages':
            print('Folder with "site-packages" in sys.path: %s' % (path,))


_started_monitoring_threads = False


def _start_monitoring_threads():
    # After the session finishes, wait 20 seconds to see if everything finished properly
    # and if it doesn't report an error.
    global _started_monitoring_threads
    if _started_monitoring_threads:
        return

    _started_monitoring_threads = True
    import threading
    if hasattr(sys, '_current_frames') and hasattr(threading, 'enumerate'):
        import time
        import traceback

        class DumpThreads(threading.Thread):

            def run(self):
                time.sleep(20)

                thread_id_to_name = {}
                try:
                    for t in threading.enumerate():
                        thread_id_to_name[t.ident] = '%s  (daemon: %s)' % (t.name, t.daemon)
                except:
                    pass

                stack_trace = [
                    '===============================================================================',
                    'pydev pyunit runner: Threads still found running after tests finished',
                    '================================= Thread Dump =================================']

                for thread_id, stack in sys._current_frames().items():
                    stack_trace.append('\n-------------------------------------------------------------------------------')
                    stack_trace.append(" Thread %s" % thread_id_to_name.get(thread_id, thread_id))
                    stack_trace.append('')

                    if 'self' in stack.f_locals:
                        sys.stderr.write(str(stack.f_locals['self']) + '\n')

                    for filename, lineno, name, line in traceback.extract_stack(stack):
                        stack_trace.append(' File "%s", line %d, in %s' % (filename, lineno, name))
                        if line:
                            stack_trace.append("   %s" % (line.strip()))
                stack_trace.append('\n=============================== END Thread Dump ===============================')
                sys.stderr.write('\n'.join(stack_trace))

                # Force thread run to finish
                import os
                os._exit(123)

        dump_current_frames_thread = DumpThreads()
        dump_current_frames_thread.daemon = True  # Daemon so that this thread doesn't halt it!
        dump_current_frames_thread.start()


def pytest_unconfigure():
    _start_monitoring_threads()


@pytest.fixture(scope="session", autouse=True)
def check_no_threads():
    yield
    _start_monitoring_threads()


# see: http://goo.gl/kTQMs
SYMBOLS = {
    'customary': ('B', 'K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y'),
    'customary_ext': ('byte', 'kilo', 'mega', 'giga', 'tera', 'peta', 'exa',
                       'zetta', 'iotta'),
    'iec': ('Bi', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi', 'Yi'),
    'iec_ext': ('byte', 'kibi', 'mebi', 'gibi', 'tebi', 'pebi', 'exbi',
                       'zebi', 'yobi'),
}


def bytes2human(n, format='%(value).1f %(symbol)s', symbols='customary'):
    """
    Bytes-to-human / human-to-bytes converter.
    Based on: http://goo.gl/kTQMs
    Working with Python 2.x and 3.x.

    Author: Giampaolo Rodola' <g.rodola [AT] gmail [DOT] com>
    License: MIT
    """

    """
    Convert n bytes into a human readable string based on format.
    symbols can be either "customary", "customary_ext", "iec" or "iec_ext",
    see: http://goo.gl/kTQMs

      >>> bytes2human(0)
      '0.0 B'
      >>> bytes2human(0.9)
      '0.0 B'
      >>> bytes2human(1)
      '1.0 B'
      >>> bytes2human(1.9)
      '1.0 B'
      >>> bytes2human(1024)
      '1.0 K'
      >>> bytes2human(1048576)
      '1.0 M'
      >>> bytes2human(1099511627776127398123789121)
      '909.5 Y'

      >>> bytes2human(9856, symbols="customary")
      '9.6 K'
      >>> bytes2human(9856, symbols="customary_ext")
      '9.6 kilo'
      >>> bytes2human(9856, symbols="iec")
      '9.6 Ki'
      >>> bytes2human(9856, symbols="iec_ext")
      '9.6 kibi'

      >>> bytes2human(10000, "%(value).1f %(symbol)s/sec")
      '9.8 K/sec'

      >>> # precision can be adjusted by playing with %f operator
      >>> bytes2human(10000, format="%(value).5f %(symbol)s")
      '9.76562 K'
    """
    n = int(n)
    if n < 0:
        raise ValueError("n < 0")
    symbols = SYMBOLS[symbols]
    prefix = {}
    for i, s in enumerate(symbols[1:]):
        prefix[s] = 1 << (i + 1) * 10
    for symbol in reversed(symbols[1:]):
        if n >= prefix[symbol]:
            value = float(n) / prefix[symbol]
            return format % locals()
    return format % dict(symbol=symbols[0], value=n)


def format_memory_info(memory_info, curr_proc_memory_info):
    return 'Total: %s, Available: %s, Used: %s %%, Curr process: %s' % (
        bytes2human(memory_info.total), bytes2human(memory_info.available), memory_info.percent, format_process_memory_info(curr_proc_memory_info))


def format_process_memory_info(proc_memory_info):
    return bytes2human(proc_memory_info.rss)


DEBUG_MEMORY_INFO = False

_global_collect_info = False

PRINT_MEMORY_BEFORE_AFTER_TEST = False  # This makes running tests slower (but it may be handy to diagnose memory issues).


@pytest.fixture(autouse=PRINT_MEMORY_BEFORE_AFTER_TEST)
def before_after_each_function(request):
    global _global_collect_info

    try:
        import psutil  # Don't fail if not there
    except ImportError:
        yield
        return

    current_pids = set(proc.pid for proc in psutil.process_iter())
    before_curr_proc_memory_info = psutil.Process().memory_info()

    if _global_collect_info and DEBUG_MEMORY_INFO:
        try:
            from pympler import summary, muppy
            sum1 = summary.summarize(muppy.get_objects())
        except:
            pydev_log.exception()

    sys.stdout.write(
'''
===============================================================================
Memory before: %s
%s
===============================================================================
''' % (request.function, format_memory_info(psutil.virtual_memory(), before_curr_proc_memory_info)))
    yield

    processes_info = []
    for proc in psutil.process_iter():
        if proc.pid not in current_pids:
            try:
                try:
                    cmdline = proc.cmdline()
                except:
                    cmdline = '<unable to get>'
                processes_info.append(
                    'New Process: %s(%s - %s) - %s' % (
                        proc.name(),
                        proc.pid,
                        cmdline,
                        format_process_memory_info(proc.memory_info())
                    )
                )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass  # The process could've died in the meanwhile

    after_curr_proc_memory_info = psutil.Process().memory_info()

    if DEBUG_MEMORY_INFO:
        try:
            if after_curr_proc_memory_info.rss - before_curr_proc_memory_info.rss > 10 * 1000 * 1000:
                # 10 MB leak
                if _global_collect_info:
                    sum2 = summary.summarize(muppy.get_objects())
                    diff = summary.get_diff(sum1, sum2)
                    sys.stdout.write('===============================================================================\n')
                    sys.stdout.write('Leak info:\n')
                    sys.stdout.write('===============================================================================\n')
                    summary.print_(diff)
                    sys.stdout.write('===============================================================================\n')

                _global_collect_info = True
                # We'll only really collect the info on the next test (i.e.: if at one test
                # we used too much memory, the next one will start collecting)
            else:
                _global_collect_info = False
        except:
            pydev_log.exception()

    sys.stdout.write(
'''
===============================================================================
Memory after: %s
%s%s
===============================================================================


''' % (
    request.function,
    format_memory_info(psutil.virtual_memory(), after_curr_proc_memory_info),
    '' if not processes_info else '\nLeaked processes:\n' + '\n'.join(processes_info)),
    )


from tests_python.regression_check import data_regression, datadir, original_datadir


@pytest.fixture
def pyfile(request, tmpdir):
    """
    Based on debugpy pyfile fixture (adapter for older versions of Python)

    A fixture providing a factory function that generates .py files.

    The returned factory takes a single function with an empty argument list,
    generates a temporary file that contains the code corresponding to the
    function body, and returns the full path to the generated file. Idiomatic
    use is as a decorator, e.g.:

        @pyfile
        def script_file():
            print('fizz')
            print('buzz')

    will produce a temporary file named script_file.py containing:

        print('fizz')
        print('buzz')

    and the variable script_file will contain the path to that file.

    In order for the factory to be able to extract the function body properly,
    function header ("def") must all be on a single line, with nothing after
    the colon but whitespace.

    Note that because the code is physically in a separate file when it runs,
    it cannot reuse top-level module imports - it must import all the modules
    that it uses locally. When linter complains, use #noqa.

    Returns a py.path.local instance that has the additional attribute "lines".
    After the source is writen to disk, tests.code.get_marked_line_numbers() is
    invoked on the resulting file to compute the value of that attribute.
    """
    import types
    import inspect

    def factory(source):
        assert isinstance(source, types.FunctionType)
        name = source.__name__
        source, _ = inspect.getsourcelines(source)

        # First, find the "def" line.
        def_lineno = 0
        for line in source:
            line = line.strip()
            if line.startswith("def") and line.endswith(":"):
                break
            def_lineno += 1
        else:
            raise ValueError("Failed to locate function header.")

        # Remove everything up to and including "def".
        source = source[def_lineno + 1:]
        assert source

        # Now we need to adjust indentation. Compute how much the first line of
        # the body is indented by, then dedent all lines by that amount. Blank
        # lines don't matter indentation-wise, and might not be indented to begin
        # with, so just replace them with a simple newline.
        for line in source:
            if line.strip():
                break  # i.e.: use first non-empty line
        indent = len(line) - len(line.lstrip())
        source = [l[indent:] if l.strip() else "\n" for l in source]
        source = "".join(source)

        # Write it to file.
        tmpfile = os.path.join(str(tmpdir), name + ".py")
        assert not os.path.exists(tmpfile), '%s already exists.' % (tmpfile,)
        with open(tmpfile, 'w') as stream:
            stream.write(source)

        return tmpfile

    return factory


if IS_JYTHON or IS_IRONPYTHON:

    # On Jython and IronPython, it's a no-op.
    def before_after_each_function():
        pass
