from nose.plugins.multiprocess import MultiProcessTestRunner  # @UnresolvedImport
from nose.plugins.base import Plugin  # @UnresolvedImport
import sys
from _pydev_runfiles import pydev_runfiles_xml_rpc
import time
from _pydev_runfiles.pydev_runfiles_coverage import start_coverage_support
from contextlib import contextmanager
from io import StringIO
import traceback


#=======================================================================================================================
# PydevPlugin
#=======================================================================================================================
class PydevPlugin(Plugin):

    def __init__(self, configuration):
        self.configuration = configuration
        Plugin.__init__(self)

    def begin(self):
        # Called before any test is run (it's always called, with multiprocess or not)
        self.start_time = time.time()
        self.coverage_files, self.coverage = start_coverage_support(self.configuration)

    def finalize(self, result):
        # Called after all tests are run (it's always called, with multiprocess or not)
        self.coverage.stop()
        self.coverage.save()

        pydev_runfiles_xml_rpc.notifyTestRunFinished('Finished in: %.2f secs.' % (time.time() - self.start_time,))

    #===================================================================================================================
    # Methods below are not called with multiprocess (so, we monkey-patch MultiProcessTestRunner.consolidate
    # so that they're called, but unfortunately we loose some info -- i.e.: the time for each test in this
    # process).
    #===================================================================================================================

    class Sentinel(object):
        pass

    @contextmanager
    def _without_user_address(self, test):
        # #PyDev-1095: Conflict between address in test and test.address() in PydevPlugin().report_cond()
        user_test_instance = test.test
        user_address = self.Sentinel
        user_class_address = self.Sentinel
        try:
            if 'address' in user_test_instance.__dict__:
                user_address = user_test_instance.__dict__.pop('address')
        except:
            # Just ignore anything here.
            pass
        try:
            user_class_address = user_test_instance.__class__.address
            del user_test_instance.__class__.address
        except:
            # Just ignore anything here.
            pass

        try:
            yield
        finally:
            if user_address is not self.Sentinel:
                user_test_instance.__dict__['address'] = user_address

            if user_class_address is not self.Sentinel:
                user_test_instance.__class__.address = user_class_address

    def _get_test_address(self, test):
        try:
            if hasattr(test, 'address'):
                with self._without_user_address(test):
                    address = test.address()

                # test.address() is something as:
                # ('D:\\workspaces\\temp\\test_workspace\\pytesting1\\src\\mod1\\hello.py', 'mod1.hello', 'TestCase.testMet1')
                #
                # and we must pass: location, test
                #    E.g.: ['D:\\src\\mod1\\hello.py', 'TestCase.testMet1']
                address = address[0], address[2]
            else:
                # multiprocess
                try:
                    address = test[0], test[1]
                except TypeError:
                    # It may be an error at setup, in which case it's not really a test, but a Context object.
                    f = test.context.__file__
                    if f.endswith('.pyc'):
                        f = f[:-1]
                    elif f.endswith('$py.class'):
                        f = f[:-len('$py.class')] + '.py'
                    address = f, '?'
        except:
            sys.stderr.write("PyDev: Internal pydev error getting test address. Please report at the pydev bug tracker\n")
            traceback.print_exc()
            sys.stderr.write("\n\n\n")
            address = '?', '?'
        return address

    def report_cond(self, cond, test, captured_output, error=''):
        '''
        @param cond: fail, error, ok
        '''

        address = self._get_test_address(test)

        error_contents = self.get_io_from_error(error)
        try:
            time_str = '%.2f' % (time.time() - test._pydev_start_time)
        except:
            time_str = '?'

        pydev_runfiles_xml_rpc.notifyTest(cond, captured_output, error_contents, address[0], address[1], time_str)

    def startTest(self, test):
        test._pydev_start_time = time.time()
        file, test = self._get_test_address(test)
        pydev_runfiles_xml_rpc.notifyStartTest(file, test)

    def get_io_from_error(self, err):
        if type(err) == type(()):
            if len(err) != 3:
                if len(err) == 2:
                    return err[1]  # multiprocess
            s = StringIO()
            etype, value, tb = err
            if isinstance(value, str):
                return value
            traceback.print_exception(etype, value, tb, file=s)
            return s.getvalue()
        return err

    def get_captured_output(self, test):
        if hasattr(test, 'capturedOutput') and test.capturedOutput:
            return test.capturedOutput
        return ''

    def addError(self, test, err):
        self.report_cond(
            'error',
            test,
            self.get_captured_output(test),
            err,
        )

    def addFailure(self, test, err):
        self.report_cond(
            'fail',
            test,
            self.get_captured_output(test),
            err,
        )

    def addSuccess(self, test):
        self.report_cond(
            'ok',
            test,
            self.get_captured_output(test),
            '',
        )


PYDEV_NOSE_PLUGIN_SINGLETON = None


def start_pydev_nose_plugin_singleton(configuration):
    global PYDEV_NOSE_PLUGIN_SINGLETON
    PYDEV_NOSE_PLUGIN_SINGLETON = PydevPlugin(configuration)
    return PYDEV_NOSE_PLUGIN_SINGLETON


original = MultiProcessTestRunner.consolidate


#=======================================================================================================================
# new_consolidate
#=======================================================================================================================
def new_consolidate(self, result, batch_result):
    '''
    Used so that it can work with the multiprocess plugin.
    Monkeypatched because nose seems a bit unsupported at this time (ideally
    the plugin would have this support by default).
    '''
    ret = original(self, result, batch_result)

    parent_frame = sys._getframe().f_back
    # addr is something as D:\pytesting1\src\mod1\hello.py:TestCase.testMet4
    # so, convert it to what report_cond expects
    addr = parent_frame.f_locals['addr']
    i = addr.rindex(':')
    addr = [addr[:i], addr[i + 1:]]

    output, testsRun, failures, errors, errorClasses = batch_result
    if failures or errors:
        for failure in failures:
            PYDEV_NOSE_PLUGIN_SINGLETON.report_cond('fail', addr, output, failure)

        for error in errors:
            PYDEV_NOSE_PLUGIN_SINGLETON.report_cond('error', addr, output, error)
    else:
        PYDEV_NOSE_PLUGIN_SINGLETON.report_cond('ok', addr, output)

    return ret


MultiProcessTestRunner.consolidate = new_consolidate
