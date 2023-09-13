'''
Entry point module to start the interactive console.
'''
from _pydev_bundle._pydev_saved_modules import thread, _code
from _pydevd_bundle.pydevd_constants import IS_JYTHON
start_new_thread = thread.start_new_thread

from _pydevd_bundle.pydevconsole_code import InteractiveConsole

compile_command = _code.compile_command
InteractiveInterpreter = _code.InteractiveInterpreter

import os
import sys

from _pydev_bundle._pydev_saved_modules import threading
from _pydevd_bundle.pydevd_constants import INTERACTIVE_MODE_AVAILABLE

import traceback
from _pydev_bundle import pydev_log

from _pydevd_bundle import pydevd_save_locals

from _pydev_bundle.pydev_imports import Exec, _queue

import builtins as __builtin__

from _pydev_bundle.pydev_console_utils import BaseInterpreterInterface, BaseStdIn  # @UnusedImport
from _pydev_bundle.pydev_console_utils import CodeFragment


class Command:

    def __init__(self, interpreter, code_fragment):
        """
        :type code_fragment: CodeFragment
        :type interpreter: InteractiveConsole
        """
        self.interpreter = interpreter
        self.code_fragment = code_fragment
        self.more = None

    def symbol_for_fragment(code_fragment):
        if code_fragment.is_single_line:
            symbol = 'single'
        else:
            if IS_JYTHON:
                symbol = 'single'  # Jython doesn't support exec
            else:
                symbol = 'exec'
        return symbol

    symbol_for_fragment = staticmethod(symbol_for_fragment)

    def run(self):
        text = self.code_fragment.text
        symbol = self.symbol_for_fragment(self.code_fragment)

        self.more = self.interpreter.runsource(text, '<input>', symbol)


try:
    from _pydev_bundle.pydev_imports import execfile

    __builtin__.execfile = execfile
except:
    pass

# Pull in runfile, the interface to UMD that wraps execfile
from _pydev_bundle.pydev_umd import runfile, _set_globals_function
if sys.version_info[0] >= 3:
    __builtin__.runfile = runfile
else:
    __builtin__.runfile = runfile


#=======================================================================================================================
# InterpreterInterface
#=======================================================================================================================
class InterpreterInterface(BaseInterpreterInterface):
    '''
        The methods in this class should be registered in the xml-rpc server.
    '''

    def __init__(self, host, client_port, mainThread, connect_status_queue=None):
        BaseInterpreterInterface.__init__(self, mainThread, connect_status_queue)
        self.client_port = client_port
        self.host = host
        self.namespace = {}
        self.interpreter = InteractiveConsole(self.namespace)
        self._input_error_printed = False

    def do_add_exec(self, codeFragment):
        command = Command(self.interpreter, codeFragment)
        command.run()
        return command.more

    def get_namespace(self):
        return self.namespace

    def getCompletions(self, text, act_tok):
        try:
            from _pydev_bundle._pydev_completer import Completer

            completer = Completer(self.namespace, None)
            return completer.complete(act_tok)
        except:
            pydev_log.exception()
            return []

    def close(self):
        sys.exit(0)

    def get_greeting_msg(self):
        return 'PyDev console: starting.\n'


class _ProcessExecQueueHelper:
    _debug_hook = None
    _return_control_osc = False


def set_debug_hook(debug_hook):
    _ProcessExecQueueHelper._debug_hook = debug_hook


def activate_mpl_if_already_imported(interpreter):
    if interpreter.mpl_modules_for_patching:
        for module in list(interpreter.mpl_modules_for_patching):
            if module in sys.modules:
                activate_function = interpreter.mpl_modules_for_patching.pop(module)
                activate_function()


def init_set_return_control_back(interpreter):
    from pydev_ipython.inputhook import set_return_control_callback

    def return_control():
        ''' A function that the inputhooks can call (via inputhook.stdin_ready()) to find
            out if they should cede control and return '''
        if _ProcessExecQueueHelper._debug_hook:
            # Some of the input hooks check return control without doing
            # a single operation, so we don't return True on every
            # call when the debug hook is in place to allow the GUI to run
            # XXX: Eventually the inputhook code will have diverged enough
            # from the IPython source that it will be worthwhile rewriting
            # it rather than pretending to maintain the old API
            _ProcessExecQueueHelper._return_control_osc = not _ProcessExecQueueHelper._return_control_osc
            if _ProcessExecQueueHelper._return_control_osc:
                return True

        if not interpreter.exec_queue.empty():
            return True
        return False

    set_return_control_callback(return_control)


def init_mpl_in_console(interpreter):
    init_set_return_control_back(interpreter)

    if not INTERACTIVE_MODE_AVAILABLE:
        return

    activate_mpl_if_already_imported(interpreter)
    from _pydev_bundle.pydev_import_hook import import_hook_manager
    for mod in list(interpreter.mpl_modules_for_patching):
        import_hook_manager.add_module_name(mod, interpreter.mpl_modules_for_patching.pop(mod))


if sys.platform != 'win32':

    if not hasattr(os, 'kill'):  # Jython may not have it.

        def pid_exists(pid):
            return True

    else:

        def pid_exists(pid):
            # Note that this function in the face of errors will conservatively consider that
            # the pid is still running (because we'll exit the current process when it's
            # no longer running, so, we need to be 100% sure it actually exited).

            import errno
            if pid == 0:
                # According to "man 2 kill" PID 0 has a special meaning:
                # it refers to <<every process in the process group of the
                # calling process>> so we don't want to go any further.
                # If we get here it means this UNIX platform *does* have
                # a process with id 0.
                return True
            try:
                os.kill(pid, 0)
            except OSError as err:
                if err.errno == errno.ESRCH:
                    # ESRCH == No such process
                    return False
                elif err.errno == errno.EPERM:
                    # EPERM clearly means there's a process to deny access to
                    return True
                else:
                    # According to "man 2 kill" possible error values are
                    # (EINVAL, EPERM, ESRCH) therefore we should never get
                    # here. If we do, although it's an error, consider it
                    # exists (see first comment in this function).
                    return True
            else:
                return True

else:

    def pid_exists(pid):
        # Note that this function in the face of errors will conservatively consider that
        # the pid is still running (because we'll exit the current process when it's
        # no longer running, so, we need to be 100% sure it actually exited).
        import ctypes
        kernel32 = ctypes.windll.kernel32

        PROCESS_QUERY_INFORMATION = 0x0400
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        ERROR_INVALID_PARAMETER = 0x57
        STILL_ACTIVE = 259

        process = kernel32.OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_QUERY_LIMITED_INFORMATION, 0, pid)
        if not process:
            err = kernel32.GetLastError()
            if err == ERROR_INVALID_PARAMETER:
                # Means it doesn't exist (pid parameter is wrong).
                return False

            # There was some unexpected error (such as access denied), so
            # consider it exists (although it could be something else, but we don't want
            # to raise any errors -- so, just consider it exists).
            return True

        try:
            zero = ctypes.c_int(0)
            exit_code = ctypes.pointer(zero)

            exit_code_suceeded = kernel32.GetExitCodeProcess(process, exit_code)
            if not exit_code_suceeded:
                # There was some unexpected error (such as access denied), so
                # consider it exists (although it could be something else, but we don't want
                # to raise any errors -- so, just consider it exists).
                return True

            elif bool(exit_code.contents.value) and int(exit_code.contents.value) != STILL_ACTIVE:
                return False
        finally:
            kernel32.CloseHandle(process)

        return True


def process_exec_queue(interpreter):
    init_mpl_in_console(interpreter)
    from pydev_ipython.inputhook import get_inputhook
    try:
        kill_if_pid_not_alive = int(os.environ.get('PYDEV_ECLIPSE_PID', '-1'))
    except:
        kill_if_pid_not_alive = -1

    while 1:
        if kill_if_pid_not_alive != -1:
            if not pid_exists(kill_if_pid_not_alive):
                exit()

        # Running the request may have changed the inputhook in use
        inputhook = get_inputhook()

        if _ProcessExecQueueHelper._debug_hook:
            _ProcessExecQueueHelper._debug_hook()

        if inputhook:
            try:
                # Note: it'll block here until return_control returns True.
                inputhook()
            except:
                pydev_log.exception()
        try:
            try:
                code_fragment = interpreter.exec_queue.get(block=True, timeout=1 / 20.)  # 20 calls/second
            except _queue.Empty:
                continue

            if callable(code_fragment):
                # It can be a callable (i.e.: something that must run in the main
                # thread can be put in the queue for later execution).
                code_fragment()
            else:
                more = interpreter.add_exec(code_fragment)
        except KeyboardInterrupt:
            interpreter.buffer = None
            continue
        except SystemExit:
            raise
        except:
            pydev_log.exception('Error processing queue on pydevconsole.')
            exit()


if 'IPYTHONENABLE' in os.environ:
    IPYTHON = os.environ['IPYTHONENABLE'] == 'True'
else:
    # By default, don't use IPython because occasionally changes
    # in IPython break pydevd.
    IPYTHON = False

try:
    try:
        exitfunc = sys.exitfunc
    except AttributeError:
        exitfunc = None

    if IPYTHON:
        from _pydev_bundle.pydev_ipython_console import InterpreterInterface
        if exitfunc is not None:
            sys.exitfunc = exitfunc
        else:
            try:
                delattr(sys, 'exitfunc')
            except:
                pass
except:
    IPYTHON = False
    pass


#=======================================================================================================================
# _DoExit
#=======================================================================================================================
def do_exit(*args):
    '''
        We have to override the exit because calling sys.exit will only actually exit the main thread,
        and as we're in a Xml-rpc server, that won't work.
    '''

    try:
        import java.lang.System

        java.lang.System.exit(1)
    except ImportError:
        if len(args) == 1:
            os._exit(args[0])
        else:
            os._exit(0)


#=======================================================================================================================
# start_console_server
#=======================================================================================================================
def start_console_server(host, port, interpreter):
    try:
        if port == 0:
            host = ''

        # I.e.: supporting the internal Jython version in PyDev to create a Jython interactive console inside Eclipse.
        from _pydev_bundle.pydev_imports import SimpleXMLRPCServer as XMLRPCServer  # @Reimport

        try:
            server = XMLRPCServer((host, port), logRequests=False, allow_none=True)

        except:
            sys.stderr.write('Error starting server with host: "%s", port: "%s", client_port: "%s"\n' % (host, port, interpreter.client_port))
            sys.stderr.flush()
            raise

        # Tell UMD the proper default namespace
        _set_globals_function(interpreter.get_namespace)

        server.register_function(interpreter.execLine)
        server.register_function(interpreter.execMultipleLines)
        server.register_function(interpreter.getCompletions)
        server.register_function(interpreter.getFrame)
        server.register_function(interpreter.getVariable)
        server.register_function(interpreter.changeVariable)
        server.register_function(interpreter.getDescription)
        server.register_function(interpreter.close)
        server.register_function(interpreter.interrupt)
        server.register_function(interpreter.handshake)
        server.register_function(interpreter.connectToDebugger)
        server.register_function(interpreter.hello)
        server.register_function(interpreter.getArray)
        server.register_function(interpreter.evaluate)
        server.register_function(interpreter.ShowConsole)
        server.register_function(interpreter.loadFullValue)

        # Functions for GUI main loop integration
        server.register_function(interpreter.enableGui)

        if port == 0:
            (h, port) = server.socket.getsockname()

            print(port)
            print(interpreter.client_port)

        while True:
            try:
                server.serve_forever()
            except:
                # Ugly code to be py2/3 compatible
                # https://sw-brainwy.rhcloud.com/tracker/PyDev/534:
                # Unhandled "interrupted system call" error in the pydevconsol.py
                e = sys.exc_info()[1]
                retry = False
                try:
                    retry = e.args[0] == 4  # errno.EINTR
                except:
                    pass
                if not retry:
                    raise
                    # Otherwise, keep on going
        return server
    except:
        pydev_log.exception()
        # Notify about error to avoid long waiting
        connection_queue = interpreter.get_connect_status_queue()
        if connection_queue is not None:
            connection_queue.put(False)


def start_server(host, port, client_port):
    # replace exit (see comments on method)
    # note that this does not work in jython!!! (sys method can't be replaced).
    sys.exit = do_exit

    interpreter = InterpreterInterface(host, client_port, threading.current_thread())

    start_new_thread(start_console_server, (host, port, interpreter))

    process_exec_queue(interpreter)


def get_ipython_hidden_vars():
    if IPYTHON and hasattr(__builtin__, 'interpreter'):
        interpreter = get_interpreter()
        return interpreter.get_ipython_hidden_vars_dict()


def get_interpreter():
    try:
        interpreterInterface = getattr(__builtin__, 'interpreter')
    except AttributeError:
        interpreterInterface = InterpreterInterface(None, None, threading.current_thread())
        __builtin__.interpreter = interpreterInterface
        sys.stderr.write(interpreterInterface.get_greeting_msg())
        sys.stderr.flush()

    return interpreterInterface


def get_completions(text, token, globals, locals):
    interpreterInterface = get_interpreter()

    interpreterInterface.interpreter.update(globals, locals)

    return interpreterInterface.getCompletions(text, token)

#===============================================================================
# Debugger integration
#===============================================================================


def exec_code(code, globals, locals, debugger):
    interpreterInterface = get_interpreter()
    interpreterInterface.interpreter.update(globals, locals)

    res = interpreterInterface.need_more(code)

    if res:
        return True

    interpreterInterface.add_exec(code, debugger)

    return False


class ConsoleWriter(InteractiveInterpreter):
    skip = 0

    def __init__(self, locals=None):
        InteractiveInterpreter.__init__(self, locals)

    def write(self, data):
        # if (data.find("global_vars") == -1 and data.find("pydevd") == -1):
        if self.skip > 0:
            self.skip -= 1
        else:
            if data == "Traceback (most recent call last):\n":
                self.skip = 1
            sys.stderr.write(data)

    def showsyntaxerror(self, filename=None):
        """Display the syntax error that just occurred."""
        # Override for avoid using sys.excepthook PY-12600
        type, value, tb = sys.exc_info()
        sys.last_type = type
        sys.last_value = value
        sys.last_traceback = tb
        if filename and type is SyntaxError:
            # Work hard to stuff the correct filename in the exception
            try:
                msg, (dummy_filename, lineno, offset, line) = value.args
            except ValueError:
                # Not the format we expect; leave it alone
                pass
            else:
                # Stuff in the right filename
                value = SyntaxError(msg, (filename, lineno, offset, line))
                sys.last_value = value
        list = traceback.format_exception_only(type, value)
        sys.stderr.write(''.join(list))

    def showtraceback(self, *args, **kwargs):
        """Display the exception that just occurred."""
        # Override for avoid using sys.excepthook PY-12600
        try:
            type, value, tb = sys.exc_info()
            sys.last_type = type
            sys.last_value = value
            sys.last_traceback = tb
            tblist = traceback.extract_tb(tb)
            del tblist[:1]
            lines = traceback.format_list(tblist)
            if lines:
                lines.insert(0, "Traceback (most recent call last):\n")
            lines.extend(traceback.format_exception_only(type, value))
        finally:
            tblist = tb = None
        sys.stderr.write(''.join(lines))


def console_exec(thread_id, frame_id, expression, dbg):
    """returns 'False' in case expression is partially correct
    """
    frame = dbg.find_frame(thread_id, frame_id)

    is_multiline = expression.count('@LINE@') > 1
    expression = str(expression.replace('@LINE@', '\n'))

    # Not using frame.f_globals because of https://sourceforge.net/tracker2/?func=detail&aid=2541355&group_id=85796&atid=577329
    # (Names not resolved in generator expression in method)
    # See message: http://mail.python.org/pipermail/python-list/2009-January/526522.html
    updated_globals = {}
    updated_globals.update(frame.f_globals)
    updated_globals.update(frame.f_locals)  # locals later because it has precedence over the actual globals

    if IPYTHON:
        need_more = exec_code(CodeFragment(expression), updated_globals, frame.f_locals, dbg)
        if not need_more:
            pydevd_save_locals.save_locals(frame)
        return need_more

    interpreter = ConsoleWriter()

    if not is_multiline:
        try:
            code = compile_command(expression)
        except (OverflowError, SyntaxError, ValueError):
            # Case 1
            interpreter.showsyntaxerror()
            return False
        if code is None:
            # Case 2
            return True
    else:
        code = expression

    # Case 3

    try:
        Exec(code, updated_globals, frame.f_locals)

    except SystemExit:
        raise
    except:
        interpreter.showtraceback()
    else:
        pydevd_save_locals.save_locals(frame)
    return False


#=======================================================================================================================
# main
#=======================================================================================================================
if __name__ == '__main__':
    # Important: don't use this module directly as the __main__ module, rather, import itself as pydevconsole
    # so that we don't get multiple pydevconsole modules if it's executed directly (otherwise we'd have multiple
    # representations of its classes).
    # See: https://sw-brainwy.rhcloud.com/tracker/PyDev/446:
    # 'Variables' and 'Expressions' views stopped working when debugging interactive console
    import pydevconsole
    sys.stdin = pydevconsole.BaseStdIn(sys.stdin)
    port, client_port = sys.argv[1:3]
    from _pydev_bundle import pydev_localhost

    if int(port) == 0 and int(client_port) == 0:
        (h, p) = pydev_localhost.get_socket_name()

        client_port = p

    pydevconsole.start_server(pydev_localhost.get_localhost(), int(port), int(client_port))
