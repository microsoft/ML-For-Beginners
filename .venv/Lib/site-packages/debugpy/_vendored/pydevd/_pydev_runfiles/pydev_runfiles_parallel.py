import unittest
from _pydev_bundle._pydev_saved_modules import thread
import queue as Queue
from _pydev_runfiles import pydev_runfiles_xml_rpc
import time
import os
import threading
import sys


#=======================================================================================================================
# flatten_test_suite
#=======================================================================================================================
def flatten_test_suite(test_suite, ret):
    if isinstance(test_suite, unittest.TestSuite):
        for t in test_suite._tests:
            flatten_test_suite(t, ret)

    elif isinstance(test_suite, unittest.TestCase):
        ret.append(test_suite)


#=======================================================================================================================
# execute_tests_in_parallel
#=======================================================================================================================
def execute_tests_in_parallel(tests, jobs, split, verbosity, coverage_files, coverage_include):
    '''
    @param tests: list(PydevTestSuite)
        A list with the suites to be run

    @param split: str
        Either 'module' or the number of tests that should be run in each batch

    @param coverage_files: list(file)
        A list with the files that should be used for giving coverage information (if empty, coverage information
        should not be gathered).

    @param coverage_include: str
        The pattern that should be included in the coverage.

    @return: bool
        Returns True if the tests were actually executed in parallel. If the tests were not executed because only 1
        should be used (e.g.: 2 jobs were requested for running 1 test), False will be returned and no tests will be
        run.

        It may also return False if in debug mode (in which case, multi-processes are not accepted)
    '''
    try:
        from _pydevd_bundle.pydevd_comm import get_global_debugger
        if get_global_debugger() is not None:
            return False
    except:
        pass  # Ignore any error here.

    # This queue will receive the tests to be run. Each entry in a queue is a list with the tests to be run together When
    # split == 'tests', each list will have a single element, when split == 'module', each list will have all the tests
    # from a given module.
    tests_queue = []

    queue_elements = []
    if split == 'module':
        module_to_tests = {}
        for test in tests:
            lst = []
            flatten_test_suite(test, lst)
            for test in lst:
                key = (test.__pydev_pyfile__, test.__pydev_module_name__)
                module_to_tests.setdefault(key, []).append(test)

        for key, tests in module_to_tests.items():
            queue_elements.append(tests)

        if len(queue_elements) < jobs:
            # Don't create jobs we will never use.
            jobs = len(queue_elements)

    elif split == 'tests':
        for test in tests:
            lst = []
            flatten_test_suite(test, lst)
            for test in lst:
                queue_elements.append([test])

        if len(queue_elements) < jobs:
            # Don't create jobs we will never use.
            jobs = len(queue_elements)

    else:
        raise AssertionError('Do not know how to handle: %s' % (split,))

    for test_cases in queue_elements:
        test_queue_elements = []
        for test_case in test_cases:
            try:
                test_name = test_case.__class__.__name__ + "." + test_case._testMethodName
            except AttributeError:
                # Support for jython 2.1 (__testMethodName is pseudo-private in the test case)
                test_name = test_case.__class__.__name__ + "." + test_case._TestCase__testMethodName

            test_queue_elements.append(test_case.__pydev_pyfile__ + '|' + test_name)

        tests_queue.append(test_queue_elements)

    if jobs < 2:
        return False

    sys.stdout.write('Running tests in parallel with: %s jobs.\n' % (jobs,))

    queue = Queue.Queue()
    for item in tests_queue:
        queue.put(item, block=False)

    providers = []
    clients = []
    for i in range(jobs):
        test_cases_provider = CommunicationThread(queue)
        providers.append(test_cases_provider)

        test_cases_provider.start()
        port = test_cases_provider.port

        if coverage_files:
            clients.append(ClientThread(i, port, verbosity, coverage_files.pop(0), coverage_include))
        else:
            clients.append(ClientThread(i, port, verbosity))

    for client in clients:
        client.start()

    client_alive = True
    while client_alive:
        client_alive = False
        for client in clients:
            # Wait for all the clients to exit.
            if not client.finished:
                client_alive = True
                time.sleep(.2)
                break

    for provider in providers:
        provider.shutdown()

    return True


#=======================================================================================================================
# CommunicationThread
#=======================================================================================================================
class CommunicationThread(threading.Thread):

    def __init__(self, tests_queue):
        threading.Thread.__init__(self)
        self.daemon = True
        self.queue = tests_queue
        self.finished = False
        from _pydev_bundle.pydev_imports import SimpleXMLRPCServer
        from _pydev_bundle import pydev_localhost

        # Create server
        server = SimpleXMLRPCServer((pydev_localhost.get_localhost(), 0), logRequests=False)
        server.register_function(self.GetTestsToRun)
        server.register_function(self.notifyStartTest)
        server.register_function(self.notifyTest)
        server.register_function(self.notifyCommands)
        self.port = server.socket.getsockname()[1]
        self.server = server

    def GetTestsToRun(self, job_id):
        '''
        @param job_id:

        @return: list(str)
            Each entry is a string in the format: filename|Test.testName
        '''
        try:
            ret = self.queue.get(block=False)
            return ret
        except:  # Any exception getting from the queue (empty or not) means we finished our work on providing the tests.
            self.finished = True
            return []

    def notifyCommands(self, job_id, commands):
        # Batch notification.
        for command in commands:
            getattr(self, command[0])(job_id, *command[1], **command[2])

        return True

    def notifyStartTest(self, job_id, *args, **kwargs):
        pydev_runfiles_xml_rpc.notifyStartTest(*args, **kwargs)
        return True

    def notifyTest(self, job_id, *args, **kwargs):
        pydev_runfiles_xml_rpc.notifyTest(*args, **kwargs)
        return True

    def shutdown(self):
        if hasattr(self.server, 'shutdown'):
            self.server.shutdown()
        else:
            self._shutdown = True

    def run(self):
        if hasattr(self.server, 'shutdown'):
            self.server.serve_forever()
        else:
            self._shutdown = False
            while not self._shutdown:
                self.server.handle_request()


#=======================================================================================================================
# Client
#=======================================================================================================================
class ClientThread(threading.Thread):

    def __init__(self, job_id, port, verbosity, coverage_output_file=None, coverage_include=None):
        threading.Thread.__init__(self)
        self.daemon = True
        self.port = port
        self.job_id = job_id
        self.verbosity = verbosity
        self.finished = False
        self.coverage_output_file = coverage_output_file
        self.coverage_include = coverage_include

    def _reader_thread(self, pipe, target):
        while True:
            target.write(pipe.read(1))

    def run(self):
        try:
            from _pydev_runfiles import pydev_runfiles_parallel_client
            # TODO: Support Jython:
            #
            # For jython, instead of using sys.executable, we should use:
            # r'D:\bin\jdk_1_5_09\bin\java.exe',
            # '-classpath',
            # 'D:/bin/jython-2.2.1/jython.jar',
            # 'org.python.util.jython',

            args = [
                sys.executable,
                pydev_runfiles_parallel_client.__file__,
                str(self.job_id),
                str(self.port),
                str(self.verbosity),
            ]

            if self.coverage_output_file and self.coverage_include:
                args.append(self.coverage_output_file)
                args.append(self.coverage_include)

            import subprocess
            if False:
                proc = subprocess.Popen(args, env=os.environ, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                thread.start_new_thread(self._reader_thread, (proc.stdout, sys.stdout))

                thread.start_new_thread(target=self._reader_thread, args=(proc.stderr, sys.stderr))
            else:
                proc = subprocess.Popen(args, env=os.environ, shell=False)
                proc.wait()

        finally:
            self.finished = True

