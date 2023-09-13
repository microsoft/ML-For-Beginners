'''An helper file for the pydev debugger (REPL) console
'''
import sys
import traceback
from _pydevd_bundle.pydevconsole_code import InteractiveConsole, _EvalAwaitInNewEventLoop
from _pydev_bundle import _pydev_completer
from _pydev_bundle.pydev_console_utils import BaseInterpreterInterface, BaseStdIn
from _pydev_bundle.pydev_imports import Exec
from _pydev_bundle.pydev_override import overrides
from _pydevd_bundle import pydevd_save_locals
from _pydevd_bundle.pydevd_io import IOBuf
from pydevd_tracing import get_exception_traceback_str
from _pydevd_bundle.pydevd_xml import make_valid_xml_value
import inspect
from _pydevd_bundle.pydevd_save_locals import update_globals_and_locals

CONSOLE_OUTPUT = "output"
CONSOLE_ERROR = "error"


#=======================================================================================================================
# ConsoleMessage
#=======================================================================================================================
class ConsoleMessage:
    """Console Messages
    """

    def __init__(self):
        self.more = False
        # List of tuple [('error', 'error_message'), ('message_list', 'output_message')]
        self.console_messages = []

    def add_console_message(self, message_type, message):
        """add messages in the console_messages list
        """
        for m in message.split("\n"):
            if m.strip():
                self.console_messages.append((message_type, m))

    def update_more(self, more):
        """more is set to true if further input is required from the user
        else more is set to false
        """
        self.more = more

    def to_xml(self):
        """Create an XML for console message_list, error and more (true/false)
        <xml>
            <message_list>console message_list</message_list>
            <error>console error</error>
            <more>true/false</more>
        </xml>
        """
        makeValid = make_valid_xml_value

        xml = '<xml><more>%s</more>' % (self.more)

        for message_type, message in self.console_messages:
            xml += '<%s message="%s"></%s>' % (message_type, makeValid(message), message_type)

        xml += '</xml>'

        return xml


#=======================================================================================================================
# _DebugConsoleStdIn
#=======================================================================================================================
class _DebugConsoleStdIn(BaseStdIn):

    @overrides(BaseStdIn.readline)
    def readline(self, *args, **kwargs):
        sys.stderr.write('Warning: Reading from stdin is still not supported in this console.\n')
        return '\n'


#=======================================================================================================================
# DebugConsole
#=======================================================================================================================
class DebugConsole(InteractiveConsole, BaseInterpreterInterface):
    """Wrapper around code.InteractiveConsole, in order to send
    errors and outputs to the debug console
    """

    @overrides(BaseInterpreterInterface.create_std_in)
    def create_std_in(self, *args, **kwargs):
        try:
            if not self.__buffer_output:
                return sys.stdin
        except:
            pass

        return _DebugConsoleStdIn()  # If buffered, raw_input is not supported in this console.

    @overrides(InteractiveConsole.push)
    def push(self, line, frame, buffer_output=True):
        """Change built-in stdout and stderr methods by the
        new custom StdMessage.
        execute the InteractiveConsole.push.
        Change the stdout and stderr back be the original built-ins

        :param buffer_output: if False won't redirect the output.

        Return boolean (True if more input is required else False),
        output_messages and input_messages
        """
        self.__buffer_output = buffer_output
        more = False
        if buffer_output:
            original_stdout = sys.stdout
            original_stderr = sys.stderr
        try:
            try:
                self.frame = frame
                if buffer_output:
                    out = sys.stdout = IOBuf()
                    err = sys.stderr = IOBuf()
                more = self.add_exec(line)
            except Exception:
                exc = get_exception_traceback_str()
                if buffer_output:
                    err.buflist.append("Internal Error: %s" % (exc,))
                else:
                    sys.stderr.write("Internal Error: %s\n" % (exc,))
        finally:
            # Remove frame references.
            self.frame = None
            frame = None
            if buffer_output:
                sys.stdout = original_stdout
                sys.stderr = original_stderr

        if buffer_output:
            return more, out.buflist, err.buflist
        else:
            return more, [], []

    @overrides(BaseInterpreterInterface.do_add_exec)
    def do_add_exec(self, line):
        return InteractiveConsole.push(self, line)

    @overrides(InteractiveConsole.runcode)
    def runcode(self, code):
        """Execute a code object.

        When an exception occurs, self.showtraceback() is called to
        display a traceback.  All exceptions are caught except
        SystemExit, which is reraised.

        A note about KeyboardInterrupt: this exception may occur
        elsewhere in this code, and may not always be caught.  The
        caller should be prepared to deal with it.

        """
        try:
            updated_globals = self.get_namespace()
            initial_globals = updated_globals.copy()

            updated_locals = None

            is_async = False
            if hasattr(inspect, 'CO_COROUTINE'):
                is_async = inspect.CO_COROUTINE & code.co_flags == inspect.CO_COROUTINE

            if is_async:
                t = _EvalAwaitInNewEventLoop(code, updated_globals, updated_locals)
                t.start()
                t.join()

                update_globals_and_locals(updated_globals, initial_globals, self.frame)
                if t.exc:
                    raise t.exc[1].with_traceback(t.exc[2])

            else:
                try:
                    exec(code, updated_globals, updated_locals)
                finally:
                    update_globals_and_locals(updated_globals, initial_globals, self.frame)
        except SystemExit:
            raise
        except:
            # In case sys.excepthook called, use original excepthook #PyDev-877: Debug console freezes with Python 3.5+
            # (showtraceback does it on python 3.5 onwards)
            sys.excepthook = sys.__excepthook__
            try:
                self.showtraceback()
            finally:
                sys.__excepthook__ = sys.excepthook

    def get_namespace(self):
        dbg_namespace = {}
        dbg_namespace.update(self.frame.f_globals)
        dbg_namespace.update(self.frame.f_locals)  # locals later because it has precedence over the actual globals
        return dbg_namespace


#=======================================================================================================================
# InteractiveConsoleCache
#=======================================================================================================================
class InteractiveConsoleCache:

    thread_id = None
    frame_id = None
    interactive_console_instance = None


# Note: On Jython 2.1 we can't use classmethod or staticmethod, so, just make the functions below free-functions.
def get_interactive_console(thread_id, frame_id, frame, console_message):
    """returns the global interactive console.
    interactive console should have been initialized by this time
    :rtype: DebugConsole
    """
    if InteractiveConsoleCache.thread_id == thread_id and InteractiveConsoleCache.frame_id == frame_id:
        return InteractiveConsoleCache.interactive_console_instance

    InteractiveConsoleCache.interactive_console_instance = DebugConsole()
    InteractiveConsoleCache.thread_id = thread_id
    InteractiveConsoleCache.frame_id = frame_id

    console_stacktrace = traceback.extract_stack(frame, limit=1)
    if console_stacktrace:
        current_context = console_stacktrace[0]  # top entry from stacktrace
        context_message = 'File "%s", line %s, in %s' % (current_context[0], current_context[1], current_context[2])
        console_message.add_console_message(CONSOLE_OUTPUT, "[Current context]: %s" % (context_message,))
    return InteractiveConsoleCache.interactive_console_instance


def clear_interactive_console():
    InteractiveConsoleCache.thread_id = None
    InteractiveConsoleCache.frame_id = None
    InteractiveConsoleCache.interactive_console_instance = None


def execute_console_command(frame, thread_id, frame_id, line, buffer_output=True):
    """fetch an interactive console instance from the cache and
    push the received command to the console.

    create and return an instance of console_message
    """
    console_message = ConsoleMessage()

    interpreter = get_interactive_console(thread_id, frame_id, frame, console_message)
    more, output_messages, error_messages = interpreter.push(line, frame, buffer_output)
    console_message.update_more(more)

    for message in output_messages:
        console_message.add_console_message(CONSOLE_OUTPUT, message)

    for message in error_messages:
        console_message.add_console_message(CONSOLE_ERROR, message)

    return console_message


def get_description(frame, thread_id, frame_id, expression):
    console_message = ConsoleMessage()
    interpreter = get_interactive_console(thread_id, frame_id, frame, console_message)
    try:
        interpreter.frame = frame
        return interpreter.getDescription(expression)
    finally:
        interpreter.frame = None


def get_completions(frame, act_tok):
    """ fetch all completions, create xml for the same
    return the completions xml
    """
    return _pydev_completer.generate_completions_as_xml(frame, act_tok)

