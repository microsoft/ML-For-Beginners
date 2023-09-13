import os
import sys
import traceback
from _pydev_bundle.pydev_imports import xmlrpclib, _queue, Exec
from  _pydev_bundle._pydev_calltip_util import get_description
from _pydevd_bundle import pydevd_vars
from _pydevd_bundle import pydevd_xml
from _pydevd_bundle.pydevd_constants import (IS_JYTHON, NEXT_VALUE_SEPARATOR, get_global_debugger,
    silence_warnings_decorator)
from contextlib import contextmanager
from _pydev_bundle import pydev_log
from _pydevd_bundle.pydevd_utils import interrupt_main_thread

from io import StringIO


# =======================================================================================================================
# BaseStdIn
# =======================================================================================================================
class BaseStdIn:

    def __init__(self, original_stdin=sys.stdin, *args, **kwargs):
        try:
            self.encoding = sys.stdin.encoding
        except:
            # Not sure if it's available in all Python versions...
            pass
        self.original_stdin = original_stdin

        try:
            self.errors = sys.stdin.errors  # Who knew? sys streams have an errors attribute!
        except:
            # Not sure if it's available in all Python versions...
            pass

    def readline(self, *args, **kwargs):
        # sys.stderr.write('Cannot readline out of the console evaluation\n') -- don't show anything
        # This could happen if the user had done input('enter number).<-- upon entering this, that message would appear,
        # which is not something we want.
        return '\n'

    def write(self, *args, **kwargs):
        pass  # not available StdIn (but it can be expected to be in the stream interface)

    def flush(self, *args, **kwargs):
        pass  # not available StdIn (but it can be expected to be in the stream interface)

    def read(self, *args, **kwargs):
        # in the interactive interpreter, a read and a readline are the same.
        return self.readline()

    def close(self, *args, **kwargs):
        pass  # expected in StdIn

    def __iter__(self):
        # BaseStdIn would not be considered as Iterable in Python 3 without explicit `__iter__` implementation
        return self.original_stdin.__iter__()

    def __getattr__(self, item):
        # it's called if the attribute wasn't found
        if hasattr(self.original_stdin, item):
            return getattr(self.original_stdin, item)
        raise AttributeError("%s has no attribute %s" % (self.original_stdin, item))


# =======================================================================================================================
# StdIn
# =======================================================================================================================
class StdIn(BaseStdIn):
    '''
        Object to be added to stdin (to emulate it as non-blocking while the next line arrives)
    '''

    def __init__(self, interpreter, host, client_port, original_stdin=sys.stdin):
        BaseStdIn.__init__(self, original_stdin)
        self.interpreter = interpreter
        self.client_port = client_port
        self.host = host

    def readline(self, *args, **kwargs):
        # Ok, callback into the client to get the new input
        try:
            server = xmlrpclib.Server('http://%s:%s' % (self.host, self.client_port))
            requested_input = server.RequestInput()
            if not requested_input:
                return '\n'  # Yes, a readline must return something (otherwise we can get an EOFError on the input() call).
            else:
                # readline should end with '\n' (not doing so makes IPython 5 remove the last *valid* character).
                requested_input += '\n'
            return requested_input
        except KeyboardInterrupt:
            raise  # Let KeyboardInterrupt go through -- #PyDev-816: Interrupting infinite loop in the Interactive Console
        except:
            return '\n'

    def close(self, *args, **kwargs):
        pass  # expected in StdIn


#=======================================================================================================================
# DebugConsoleStdIn
#=======================================================================================================================
class DebugConsoleStdIn(BaseStdIn):
    '''
        Object to be added to stdin (to emulate it as non-blocking while the next line arrives)
    '''

    def __init__(self, py_db, original_stdin):
        '''
        :param py_db:
            If None, get_global_debugger() is used.
        '''
        BaseStdIn.__init__(self, original_stdin)
        self._py_db = py_db
        self._in_notification = 0

    def __send_input_requested_message(self, is_started):
        try:
            py_db = self._py_db
            if py_db is None:
                py_db = get_global_debugger()

            if py_db is None:
                return

            cmd = py_db.cmd_factory.make_input_requested_message(is_started)
            py_db.writer.add_command(cmd)
        except Exception:
            pydev_log.exception()

    @contextmanager
    def notify_input_requested(self):
        self._in_notification += 1
        if self._in_notification == 1:
            self.__send_input_requested_message(True)
        try:
            yield
        finally:
            self._in_notification -= 1
            if self._in_notification == 0:
                self.__send_input_requested_message(False)

    def readline(self, *args, **kwargs):
        with self.notify_input_requested():
            return self.original_stdin.readline(*args, **kwargs)

    def read(self, *args, **kwargs):
        with self.notify_input_requested():
            return self.original_stdin.read(*args, **kwargs)


class CodeFragment:

    def __init__(self, text, is_single_line=True):
        self.text = text
        self.is_single_line = is_single_line

    def append(self, code_fragment):
        self.text = self.text + "\n" + code_fragment.text
        if not code_fragment.is_single_line:
            self.is_single_line = False


# =======================================================================================================================
# BaseInterpreterInterface
# =======================================================================================================================
class BaseInterpreterInterface:

    def __init__(self, mainThread, connect_status_queue=None):
        self.mainThread = mainThread
        self.interruptable = False
        self.exec_queue = _queue.Queue(0)
        self.buffer = None
        self.banner_shown = False
        self.connect_status_queue = connect_status_queue
        self.mpl_modules_for_patching = {}
        self.init_mpl_modules_for_patching()

    def build_banner(self):
        return 'print({0})\n'.format(repr(self.get_greeting_msg()))

    def get_greeting_msg(self):
        return 'PyDev console: starting.\n'

    def init_mpl_modules_for_patching(self):
        from pydev_ipython.matplotlibtools import activate_matplotlib, activate_pylab, activate_pyplot
        self.mpl_modules_for_patching = {
            "matplotlib": lambda: activate_matplotlib(self.enableGui),
            "matplotlib.pyplot": activate_pyplot,
            "pylab": activate_pylab
        }

    def need_more_for_code(self, source):
        # PyDev-502: PyDev 3.9 F2 doesn't support backslash continuations

        # Strangely even the IPython console is_complete said it was complete
        # even with a continuation char at the end.
        if source.endswith('\\'):
            return True

        if hasattr(self.interpreter, 'is_complete'):
            return not self.interpreter.is_complete(source)
        try:
            # At this point, it should always be single.
            # If we don't do this, things as:
            #
            #     for i in range(10): print(i)
            #
            # (in a single line) don't work.
            # Note that it won't give an error and code will be None (so, it'll
            # use execMultipleLines in the next call in this case).
            symbol = 'single'
            code = self.interpreter.compile(source, '<input>', symbol)
        except (OverflowError, SyntaxError, ValueError):
            # Case 1
            return False
        if code is None:
            # Case 2
            return True

        # Case 3
        return False

    def need_more(self, code_fragment):
        if self.buffer is None:
            self.buffer = code_fragment
        else:
            self.buffer.append(code_fragment)

        return self.need_more_for_code(self.buffer.text)

    def create_std_in(self, debugger=None, original_std_in=None):
        if debugger is None:
            return StdIn(self, self.host, self.client_port, original_stdin=original_std_in)
        else:
            return DebugConsoleStdIn(py_db=debugger, original_stdin=original_std_in)

    def add_exec(self, code_fragment, debugger=None):
        # In case sys.excepthook called, use original excepthook #PyDev-877: Debug console freezes with Python 3.5+
        # (showtraceback does it on python 3.5 onwards)
        sys.excepthook = sys.__excepthook__
        try:
            original_in = sys.stdin
            try:
                help = None
                if 'pydoc' in sys.modules:
                    pydoc = sys.modules['pydoc']  # Don't import it if it still is not there.

                    if hasattr(pydoc, 'help'):
                        # You never know how will the API be changed, so, let's code defensively here
                        help = pydoc.help
                        if not hasattr(help, 'input'):
                            help = None
            except:
                # Just ignore any error here
                pass

            more = False
            try:
                sys.stdin = self.create_std_in(debugger, original_in)
                try:
                    if help is not None:
                        # This will enable the help() function to work.
                        try:
                            try:
                                help.input = sys.stdin
                            except AttributeError:
                                help._input = sys.stdin
                        except:
                            help = None
                            if not self._input_error_printed:
                                self._input_error_printed = True
                                sys.stderr.write('\nError when trying to update pydoc.help.input\n')
                                sys.stderr.write('(help() may not work -- please report this as a bug in the pydev bugtracker).\n\n')
                                traceback.print_exc()

                    try:
                        self.start_exec()
                        if hasattr(self, 'debugger'):
                            self.debugger.enable_tracing()

                        more = self.do_add_exec(code_fragment)

                        if hasattr(self, 'debugger'):
                            self.debugger.disable_tracing()

                        self.finish_exec(more)
                    finally:
                        if help is not None:
                            try:
                                try:
                                    help.input = original_in
                                except AttributeError:
                                    help._input = original_in
                            except:
                                pass

                finally:
                    sys.stdin = original_in
            except SystemExit:
                raise
            except:
                traceback.print_exc()
        finally:
            sys.__excepthook__ = sys.excepthook

        return more

    def do_add_exec(self, codeFragment):
        '''
        Subclasses should override.

        @return: more (True if more input is needed to complete the statement and False if the statement is complete).
        '''
        raise NotImplementedError()

    def get_namespace(self):
        '''
        Subclasses should override.

        @return: dict with namespace.
        '''
        raise NotImplementedError()

    def __resolve_reference__(self, text):
        """

        :type text: str
        """
        obj = None
        if '.' not in text:
            try:
                obj = self.get_namespace()[text]
            except KeyError:
                pass

            if obj is None:
                try:
                    obj = self.get_namespace()['__builtins__'][text]
                except:
                    pass

            if obj is None:
                try:
                    obj = getattr(self.get_namespace()['__builtins__'], text, None)
                except:
                    pass

        else:
            try:
                last_dot = text.rindex('.')
                parent_context = text[0:last_dot]
                res = pydevd_vars.eval_in_context(parent_context, self.get_namespace(), self.get_namespace())
                obj = getattr(res, text[last_dot + 1:])
            except:
                pass
        return obj

    def getDescription(self, text):
        try:
            obj = self.__resolve_reference__(text)
            if obj is None:
                return ''
            return get_description(obj)
        except:
            return ''

    def do_exec_code(self, code, is_single_line):
        try:
            code_fragment = CodeFragment(code, is_single_line)
            more = self.need_more(code_fragment)
            if not more:
                code_fragment = self.buffer
                self.buffer = None
                self.exec_queue.put(code_fragment)

            return more
        except:
            traceback.print_exc()
            return False

    def execLine(self, line):
        return self.do_exec_code(line, True)

    def execMultipleLines(self, lines):
        if IS_JYTHON:
            more = False
            for line in lines.split('\n'):
                more = self.do_exec_code(line, True)
            return more
        else:
            return self.do_exec_code(lines, False)

    def interrupt(self):
        self.buffer = None  # Also clear the buffer when it's interrupted.
        try:
            if self.interruptable:
                # Fix for #PyDev-500: Console interrupt can't interrupt on sleep
                interrupt_main_thread(self.mainThread)

            self.finish_exec(False)
            return True
        except:
            traceback.print_exc()
            return False

    def close(self):
        sys.exit(0)

    def start_exec(self):
        self.interruptable = True

    def get_server(self):
        if getattr(self, 'host', None) is not None:
            return xmlrpclib.Server('http://%s:%s' % (self.host, self.client_port))
        else:
            return None

    server = property(get_server)

    def ShowConsole(self):
        server = self.get_server()
        if server is not None:
            server.ShowConsole()

    def finish_exec(self, more):
        self.interruptable = False

        server = self.get_server()

        if server is not None:
            return server.NotifyFinished(more)
        else:
            return True

    def getFrame(self):
        xml = StringIO()
        hidden_ns = self.get_ipython_hidden_vars_dict()
        xml.write("<xml>")
        xml.write(pydevd_xml.frame_vars_to_xml(self.get_namespace(), hidden_ns))
        xml.write("</xml>")

        return xml.getvalue()

    @silence_warnings_decorator
    def getVariable(self, attributes):
        xml = StringIO()
        xml.write("<xml>")
        val_dict = pydevd_vars.resolve_compound_var_object_fields(self.get_namespace(), attributes)
        if val_dict is None:
            val_dict = {}

        for k, val in val_dict.items():
            val = val_dict[k]
            evaluate_full_value = pydevd_xml.should_evaluate_full_value(val)
            xml.write(pydevd_vars.var_to_xml(val, k, evaluate_full_value=evaluate_full_value))

        xml.write("</xml>")

        return xml.getvalue()

    def getArray(self, attr, roffset, coffset, rows, cols, format):
        name = attr.split("\t")[-1]
        array = pydevd_vars.eval_in_context(name, self.get_namespace(), self.get_namespace())
        return pydevd_vars.table_like_struct_to_xml(array, name, roffset, coffset, rows, cols, format)

    def evaluate(self, expression):
        xml = StringIO()
        xml.write("<xml>")
        result = pydevd_vars.eval_in_context(expression, self.get_namespace(), self.get_namespace())
        xml.write(pydevd_vars.var_to_xml(result, expression))
        xml.write("</xml>")
        return xml.getvalue()

    @silence_warnings_decorator
    def loadFullValue(self, seq, scope_attrs):
        """
        Evaluate full value for async Console variables in a separate thread and send results to IDE side
        :param seq: id of command
        :param scope_attrs: a sequence of variables with their attributes separated by NEXT_VALUE_SEPARATOR
        (i.e.: obj\tattr1\tattr2NEXT_VALUE_SEPARATORobj2\attr1\tattr2)
        :return:
        """
        frame_variables = self.get_namespace()
        var_objects = []
        vars = scope_attrs.split(NEXT_VALUE_SEPARATOR)
        for var_attrs in vars:
            if '\t' in var_attrs:
                name, attrs = var_attrs.split('\t', 1)

            else:
                name = var_attrs
                attrs = None
            if name in frame_variables:
                var_object = pydevd_vars.resolve_var_object(frame_variables[name], attrs)
                var_objects.append((var_object, name))
            else:
                var_object = pydevd_vars.eval_in_context(name, frame_variables, frame_variables)
                var_objects.append((var_object, name))

        from _pydevd_bundle.pydevd_comm import GetValueAsyncThreadConsole
        py_db = getattr(self, 'debugger', None)

        if py_db is None:
            py_db = get_global_debugger()

        if py_db is None:
            from pydevd import PyDB
            py_db = PyDB()

        t = GetValueAsyncThreadConsole(py_db, self.get_server(), seq, var_objects)
        t.start()

    def changeVariable(self, attr, value):

        def do_change_variable():
            Exec('%s=%s' % (attr, value), self.get_namespace(), self.get_namespace())

        # Important: it has to be really enabled in the main thread, so, schedule
        # it to run in the main thread.
        self.exec_queue.put(do_change_variable)

    def connectToDebugger(self, debuggerPort, debugger_options=None):
        '''
        Used to show console with variables connection.
        Mainly, monkey-patches things in the debugger structure so that the debugger protocol works.
        '''

        if debugger_options is None:
            debugger_options = {}
        env_key = "PYDEVD_EXTRA_ENVS"
        if env_key in debugger_options:
            for (env_name, value) in debugger_options[env_key].items():
                existing_value = os.environ.get(env_name, None)
                if existing_value:
                    os.environ[env_name] = "%s%c%s" % (existing_value, os.path.pathsep, value)
                else:
                    os.environ[env_name] = value
                if env_name == "PYTHONPATH":
                    sys.path.append(value)

            del debugger_options[env_key]

        def do_connect_to_debugger():
            try:
                # Try to import the packages needed to attach the debugger
                import pydevd
                from _pydev_bundle._pydev_saved_modules import threading
            except:
                # This happens on Jython embedded in host eclipse
                traceback.print_exc()
                sys.stderr.write('pydevd is not available, cannot connect\n')

            from _pydevd_bundle.pydevd_constants import set_thread_id
            from _pydev_bundle import pydev_localhost
            set_thread_id(threading.current_thread(), "console_main")

            VIRTUAL_FRAME_ID = "1"  # matches PyStackFrameConsole.java
            VIRTUAL_CONSOLE_ID = "console_main"  # matches PyThreadConsole.java
            f = FakeFrame()
            f.f_back = None
            f.f_globals = {}  # As globals=locals here, let's simply let it empty (and save a bit of network traffic).
            f.f_locals = self.get_namespace()

            self.debugger = pydevd.PyDB()
            self.debugger.add_fake_frame(thread_id=VIRTUAL_CONSOLE_ID, frame_id=VIRTUAL_FRAME_ID, frame=f)
            try:
                pydevd.apply_debugger_options(debugger_options)
                self.debugger.connect(pydev_localhost.get_localhost(), debuggerPort)
                self.debugger.prepare_to_run()
                self.debugger.disable_tracing()
            except:
                traceback.print_exc()
                sys.stderr.write('Failed to connect to target debugger.\n')

            # Register to process commands when idle
            self.debugrunning = False
            try:
                import pydevconsole
                pydevconsole.set_debug_hook(self.debugger.process_internal_commands)
            except:
                traceback.print_exc()
                sys.stderr.write('Version of Python does not support debuggable Interactive Console.\n')

        # Important: it has to be really enabled in the main thread, so, schedule
        # it to run in the main thread.
        self.exec_queue.put(do_connect_to_debugger)

        return ('connect complete',)

    def handshake(self):
        if self.connect_status_queue is not None:
            self.connect_status_queue.put(True)
        return "PyCharm"

    def get_connect_status_queue(self):
        return self.connect_status_queue

    def hello(self, input_str):
        # Don't care what the input string is
        return ("Hello eclipse",)

    def enableGui(self, guiname):
        ''' Enable the GUI specified in guiname (see inputhook for list).
            As with IPython, enabling multiple GUIs isn't an error, but
            only the last one's main loop runs and it may not work
        '''

        def do_enable_gui():
            from _pydev_bundle.pydev_versioncheck import versionok_for_gui
            if versionok_for_gui():
                try:
                    from pydev_ipython.inputhook import enable_gui
                    enable_gui(guiname)
                except:
                    sys.stderr.write("Failed to enable GUI event loop integration for '%s'\n" % guiname)
                    traceback.print_exc()
            elif guiname not in ['none', '', None]:
                # Only print a warning if the guiname was going to do something
                sys.stderr.write("PyDev console: Python version does not support GUI event loop integration for '%s'\n" % guiname)
            # Return value does not matter, so return back what was sent
            return guiname

        # Important: it has to be really enabled in the main thread, so, schedule
        # it to run in the main thread.
        self.exec_queue.put(do_enable_gui)

    def get_ipython_hidden_vars_dict(self):
        return None


# =======================================================================================================================
# FakeFrame
# =======================================================================================================================
class FakeFrame:
    '''
    Used to show console with variables connection.
    A class to be used as a mock of a frame.
    '''
