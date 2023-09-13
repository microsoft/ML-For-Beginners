import unittest as python_unittest
from _pydev_runfiles import pydev_runfiles_xml_rpc
import time
from _pydevd_bundle import pydevd_io
import traceback
from _pydevd_bundle.pydevd_constants import *  # @UnusedWildImport
from io import StringIO


#=======================================================================================================================
# PydevTextTestRunner
#=======================================================================================================================
class PydevTextTestRunner(python_unittest.TextTestRunner):

    def _makeResult(self):
        return PydevTestResult(self.stream, self.descriptions, self.verbosity)


_PythonTextTestResult = python_unittest.TextTestRunner()._makeResult().__class__


#=======================================================================================================================
# PydevTestResult
#=======================================================================================================================
class PydevTestResult(_PythonTextTestResult):

    def addSubTest(self, test, subtest, err):
        """Called at the end of a subtest.
        'err' is None if the subtest ended successfully, otherwise it's a
        tuple of values as returned by sys.exc_info().
        """
        _PythonTextTestResult.addSubTest(self, test, subtest, err)
        if err is not None:
            subdesc = subtest._subDescription()
            error = (test, self._exc_info_to_string(err, test))
            self._reportErrors([error], [], '', '%s %s' % (self.get_test_name(test), subdesc))

    def startTest(self, test):
        _PythonTextTestResult.startTest(self, test)
        self.buf = pydevd_io.start_redirect(keep_original_redirection=True, std='both')
        self.start_time = time.time()
        self._current_errors_stack = []
        self._current_failures_stack = []

        try:
            test_name = test.__class__.__name__ + "." + test._testMethodName
        except AttributeError:
            # Support for jython 2.1 (__testMethodName is pseudo-private in the test case)
            test_name = test.__class__.__name__ + "." + test._TestCase__testMethodName

        pydev_runfiles_xml_rpc.notifyStartTest(
            test.__pydev_pyfile__, test_name)

    def get_test_name(self, test):
        try:
            try:
                test_name = test.__class__.__name__ + "." + test._testMethodName
            except AttributeError:
                # Support for jython 2.1 (__testMethodName is pseudo-private in the test case)
                try:
                    test_name = test.__class__.__name__ + "." + test._TestCase__testMethodName
                # Support for class/module exceptions (test is instance of _ErrorHolder)
                except:
                    test_name = test.description.split()[1][1:-1] + ' <' + test.description.split()[0] + '>'
        except:
            traceback.print_exc()
            return '<unable to get test name>'
        return test_name

    def stopTest(self, test):
        end_time = time.time()
        pydevd_io.end_redirect(std='both')

        _PythonTextTestResult.stopTest(self, test)

        captured_output = self.buf.getvalue()
        del self.buf
        error_contents = ''
        test_name = self.get_test_name(test)

        diff_time = '%.2f' % (end_time - self.start_time)

        skipped = False
        outcome = getattr(test, '_outcome', None)
        if outcome is not None:
            skipped = bool(getattr(outcome, 'skipped', None))

        if skipped:
            pydev_runfiles_xml_rpc.notifyTest(
                'skip', captured_output, error_contents, test.__pydev_pyfile__, test_name, diff_time)
        elif not self._current_errors_stack and not self._current_failures_stack:
            pydev_runfiles_xml_rpc.notifyTest(
                'ok', captured_output, error_contents, test.__pydev_pyfile__, test_name, diff_time)
        else:
            self._reportErrors(self._current_errors_stack, self._current_failures_stack, captured_output, test_name)

    def _reportErrors(self, errors, failures, captured_output, test_name, diff_time=''):
        error_contents = []
        for test, s in errors + failures:
            if type(s) == type((1,)):  # If it's a tuple (for jython 2.1)
                sio = StringIO()
                traceback.print_exception(s[0], s[1], s[2], file=sio)
                s = sio.getvalue()
            error_contents.append(s)

        sep = '\n' + self.separator1
        error_contents = sep.join(error_contents)

        if errors and not failures:
            try:
                pydev_runfiles_xml_rpc.notifyTest(
                    'error', captured_output, error_contents, test.__pydev_pyfile__, test_name, diff_time)
            except:
                file_start = error_contents.find('File "')
                file_end = error_contents.find('", ', file_start)
                if file_start != -1 and file_end != -1:
                    file = error_contents[file_start + 6:file_end]
                else:
                    file = '<unable to get file>'
                pydev_runfiles_xml_rpc.notifyTest(
                    'error', captured_output, error_contents, file, test_name, diff_time)

        elif failures and not errors:
            pydev_runfiles_xml_rpc.notifyTest(
                'fail', captured_output, error_contents, test.__pydev_pyfile__, test_name, diff_time)

        else:  # Ok, we got both, errors and failures. Let's mark it as an error in the end.
            pydev_runfiles_xml_rpc.notifyTest(
                'error', captured_output, error_contents, test.__pydev_pyfile__, test_name, diff_time)

    def addError(self, test, err):
        _PythonTextTestResult.addError(self, test, err)
        # Support for class/module exceptions (test is instance of _ErrorHolder)
        if not hasattr(self, '_current_errors_stack') or test.__class__.__name__ == '_ErrorHolder':
            # Not in start...end, so, report error now (i.e.: django pre/post-setup)
            self._reportErrors([self.errors[-1]], [], '', self.get_test_name(test))
        else:
            self._current_errors_stack.append(self.errors[-1])

    def addFailure(self, test, err):
        _PythonTextTestResult.addFailure(self, test, err)
        if not hasattr(self, '_current_failures_stack'):
            # Not in start...end, so, report error now (i.e.: django pre/post-setup)
            self._reportErrors([], [self.failures[-1]], '', self.get_test_name(test))
        else:
            self._current_failures_stack.append(self.failures[-1])


class PydevTestSuite(python_unittest.TestSuite):
    pass
