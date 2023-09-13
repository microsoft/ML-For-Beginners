import sys
import threading
import traceback
import warnings

from _pydev_bundle._pydev_filesystem_encoding import getfilesystemencoding
from _pydev_bundle.pydev_imports import xmlrpclib, _queue
from _pydevd_bundle.pydevd_constants import Null

Queue = _queue.Queue

# This may happen in IronPython (in Python it shouldn't happen as there are
# 'fast' replacements that are used in xmlrpclib.py)
warnings.filterwarnings(
    'ignore', 'The xmllib module is obsolete.*', DeprecationWarning)

file_system_encoding = getfilesystemencoding()


#=======================================================================================================================
# _ServerHolder
#=======================================================================================================================
class _ServerHolder:
    '''
    Helper so that we don't have to use a global here.
    '''
    SERVER = None


#=======================================================================================================================
# set_server
#=======================================================================================================================
def set_server(server):
    _ServerHolder.SERVER = server


#=======================================================================================================================
# ParallelNotification
#=======================================================================================================================
class ParallelNotification(object):

    def __init__(self, method, args):
        self.method = method
        self.args = args

    def to_tuple(self):
        return self.method, self.args


#=======================================================================================================================
# KillServer
#=======================================================================================================================
class KillServer(object):
    pass


#=======================================================================================================================
# ServerFacade
#=======================================================================================================================
class ServerFacade(object):

    def __init__(self, notifications_queue):
        self.notifications_queue = notifications_queue

    def notifyTestsCollected(self, *args):
        self.notifications_queue.put_nowait(ParallelNotification('notifyTestsCollected', args))

    def notifyConnected(self, *args):
        self.notifications_queue.put_nowait(ParallelNotification('notifyConnected', args))

    def notifyTestRunFinished(self, *args):
        self.notifications_queue.put_nowait(ParallelNotification('notifyTestRunFinished', args))

    def notifyStartTest(self, *args):
        self.notifications_queue.put_nowait(ParallelNotification('notifyStartTest', args))

    def notifyTest(self, *args):
        new_args = []
        for arg in args:
            new_args.append(_encode_if_needed(arg))
        args = tuple(new_args)
        self.notifications_queue.put_nowait(ParallelNotification('notifyTest', args))


#=======================================================================================================================
# ServerComm
#=======================================================================================================================
class ServerComm(threading.Thread):

    def __init__(self, notifications_queue, port, daemon=False):
        threading.Thread.__init__(self)
        self.setDaemon(daemon)  # If False, wait for all the notifications to be passed before exiting!
        self.finished = False
        self.notifications_queue = notifications_queue

        from _pydev_bundle import pydev_localhost

        # It is necessary to specify an encoding, that matches
        # the encoding of all bytes-strings passed into an
        # XMLRPC call: "All 8-bit strings in the data structure are assumed to use the
        # packet encoding.  Unicode strings are automatically converted,
        # where necessary."
        # Byte strings most likely come from file names.
        encoding = file_system_encoding
        if encoding == "mbcs":
            # Windos symbolic name for the system encoding CP_ACP.
            # We need to convert it into a encoding that is recognized by Java.
            # Unfortunately this is not always possible. You could use
            # GetCPInfoEx and get a name similar to "windows-1251". Then
            # you need a table to translate on a best effort basis. Much to complicated.
            # ISO-8859-1 is good enough.
            encoding = "ISO-8859-1"

        self.server = xmlrpclib.Server('http://%s:%s' % (pydev_localhost.get_localhost(), port),
                                       encoding=encoding)

    def run(self):
        while True:
            kill_found = False
            commands = []
            command = self.notifications_queue.get(block=True)
            if isinstance(command, KillServer):
                kill_found = True
            else:
                assert isinstance(command, ParallelNotification)
                commands.append(command.to_tuple())

            try:
                while True:
                    command = self.notifications_queue.get(block=False)  # No block to create a batch.
                    if isinstance(command, KillServer):
                        kill_found = True
                    else:
                        assert isinstance(command, ParallelNotification)
                        commands.append(command.to_tuple())
            except:
                pass  # That's OK, we're getting it until it becomes empty so that we notify multiple at once.

            if commands:
                try:
                    self.server.notifyCommands(commands)
                except:
                    traceback.print_exc()

            if kill_found:
                self.finished = True
                return


#=======================================================================================================================
# initialize_server
#=======================================================================================================================
def initialize_server(port, daemon=False):
    if _ServerHolder.SERVER is None:
        if port is not None:
            notifications_queue = Queue()
            _ServerHolder.SERVER = ServerFacade(notifications_queue)
            _ServerHolder.SERVER_COMM = ServerComm(notifications_queue, port, daemon)
            _ServerHolder.SERVER_COMM.start()
        else:
            # Create a null server, so that we keep the interface even without any connection.
            _ServerHolder.SERVER = Null()
            _ServerHolder.SERVER_COMM = Null()

    try:
        _ServerHolder.SERVER.notifyConnected()
    except:
        traceback.print_exc()


#=======================================================================================================================
# notifyTest
#=======================================================================================================================
def notifyTestsCollected(tests_count):
    assert tests_count is not None
    try:
        _ServerHolder.SERVER.notifyTestsCollected(tests_count)
    except:
        traceback.print_exc()


#=======================================================================================================================
# notifyStartTest
#=======================================================================================================================
def notifyStartTest(file, test):
    '''
    @param file: the tests file (c:/temp/test.py)
    @param test: the test ran (i.e.: TestCase.test1)
    '''
    assert file is not None
    if test is None:
        test = ''  # Could happen if we have an import error importing module.

    try:
        _ServerHolder.SERVER.notifyStartTest(file, test)
    except:
        traceback.print_exc()


def _encode_if_needed(obj):
    # In the java side we expect strings to be ISO-8859-1 (org.python.pydev.debug.pyunit.PyUnitServer.initializeDispatches().new Dispatch() {...}.getAsStr(Object))
    if isinstance(obj, str):  # Unicode in py3
        return xmlrpclib.Binary(obj.encode('ISO-8859-1', 'xmlcharrefreplace'))

    elif isinstance(obj, bytes):
        try:
            return xmlrpclib.Binary(obj.decode(sys.stdin.encoding).encode('ISO-8859-1', 'xmlcharrefreplace'))
        except:
            return xmlrpclib.Binary(obj)  # bytes already

    return obj


#=======================================================================================================================
# notifyTest
#=======================================================================================================================
def notifyTest(cond, captured_output, error_contents, file, test, time):
    '''
    @param cond: ok, fail, error
    @param captured_output: output captured from stdout
    @param captured_output: output captured from stderr
    @param file: the tests file (c:/temp/test.py)
    @param test: the test ran (i.e.: TestCase.test1)
    @param time: float with the number of seconds elapsed
    '''
    assert cond is not None
    assert captured_output is not None
    assert error_contents is not None
    assert file is not None
    if test is None:
        test = ''  # Could happen if we have an import error importing module.
    assert time is not None
    try:
        captured_output = _encode_if_needed(captured_output)
        error_contents = _encode_if_needed(error_contents)

        _ServerHolder.SERVER.notifyTest(cond, captured_output, error_contents, file, test, time)
    except:
        traceback.print_exc()


#=======================================================================================================================
# notifyTestRunFinished
#=======================================================================================================================
def notifyTestRunFinished(total_time):
    assert total_time is not None
    try:
        _ServerHolder.SERVER.notifyTestRunFinished(total_time)
    except:
        traceback.print_exc()


#=======================================================================================================================
# force_server_kill
#=======================================================================================================================
def force_server_kill():
    _ServerHolder.SERVER_COMM.notifications_queue.put_nowait(KillServer())
