''' pydevd - a debugging daemon
This is the daemon you launch for python remote debugging.

Protocol:
each command has a format:
    id\tsequence-num\ttext
    id: protocol command number
    sequence-num: each request has a sequence number. Sequence numbers
    originating at the debugger are odd, sequence numbers originating
    at the daemon are even. Every response uses the same sequence number
    as the request.
    payload: it is protocol dependent. When response is a complex structure, it
    is returned as XML. Each attribute value is urlencoded, and then the whole
    payload is urlencoded again to prevent stray characters corrupting protocol/xml encodings

    Commands:

    NUMBER   NAME                     FROM*     ARGUMENTS                     RESPONSE      NOTE
100 series: program execution
    101      RUN                      JAVA      -                             -
    102      LIST_THREADS             JAVA                                    RETURN with XML listing of all threads
    103      THREAD_CREATE            PYDB      -                             XML with thread information
    104      THREAD_KILL              JAVA      id (or * to exit)             kills the thread
                                      PYDB      id                            nofies JAVA that thread was killed
    105      THREAD_SUSPEND           JAVA      XML of the stack,             suspends the thread
                                                reason for suspension
                                      PYDB      id                            notifies JAVA that thread was suspended

    106      CMD_THREAD_RUN           JAVA      id                            resume the thread
                                      PYDB      id \t reason                  notifies JAVA that thread was resumed

    107      STEP_INTO                JAVA      thread_id
    108      STEP_OVER                JAVA      thread_id
    109      STEP_RETURN              JAVA      thread_id

    110      GET_VARIABLE             JAVA      thread_id \t frame_id \t      GET_VARIABLE with XML of var content
                                                FRAME|GLOBAL \t attributes*

    111      SET_BREAK                JAVA      file/line of the breakpoint
    112      REMOVE_BREAK             JAVA      file/line of the return
    113      CMD_EVALUATE_EXPRESSION  JAVA      expression                    result of evaluating the expression
    114      CMD_GET_FRAME            JAVA                                    request for frame contents
    115      CMD_EXEC_EXPRESSION      JAVA
    116      CMD_WRITE_TO_CONSOLE     PYDB
    117      CMD_CHANGE_VARIABLE
    118      CMD_RUN_TO_LINE
    119      CMD_RELOAD_CODE
    120      CMD_GET_COMPLETIONS      JAVA

    200      CMD_REDIRECT_OUTPUT      JAVA      streams to redirect as string -
                                                'STDOUT' (redirect only STDOUT)
                                                'STDERR' (redirect only STDERR)
                                                'STDOUT STDERR' (redirect both streams)

500 series diagnostics/ok
    501      VERSION                  either      Version string (1.0)        Currently just used at startup
    502      RETURN                   either      Depends on caller    -

900 series: errors
    901      ERROR                    either      -                           This is reserved for unexpected errors.

    * JAVA - remote debugger, the java end
    * PYDB - pydevd, the python end
'''

import linecache
import os

from _pydev_bundle.pydev_imports import _queue
from _pydev_bundle._pydev_saved_modules import time
from _pydev_bundle._pydev_saved_modules import threading
from _pydev_bundle._pydev_saved_modules import socket as socket_module
from _pydevd_bundle.pydevd_constants import (DebugInfoHolder, IS_WINDOWS, IS_JYTHON, IS_WASM,
    IS_PY36_OR_GREATER, STATE_RUN, ASYNC_EVAL_TIMEOUT_SEC,
    get_global_debugger, GetGlobalDebugger, set_global_debugger,  # Keep for backward compatibility @UnusedImport
    silence_warnings_decorator, filter_all_warnings, IS_PY311_OR_GREATER)
from _pydev_bundle.pydev_override import overrides
import weakref
from _pydev_bundle._pydev_completer import extract_token_and_qualifier
from _pydevd_bundle._debug_adapter.pydevd_schema import VariablesResponseBody, \
    SetVariableResponseBody, StepInTarget, StepInTargetsResponseBody
from _pydevd_bundle._debug_adapter import pydevd_base_schema, pydevd_schema
from _pydevd_bundle.pydevd_net_command import NetCommand
from _pydevd_bundle.pydevd_xml import ExceptionOnEvaluate
from _pydevd_bundle.pydevd_constants import ForkSafeLock, NULL
from _pydevd_bundle.pydevd_daemon_thread import PyDBDaemonThread
from _pydevd_bundle.pydevd_thread_lifecycle import pydevd_find_thread_by_id, resume_threads
from _pydevd_bundle.pydevd_dont_trace_files import PYDEV_FILE
import dis
import pydevd_file_utils
import itertools
from urllib.parse import quote_plus, unquote_plus
import pydevconsole
from _pydevd_bundle import pydevd_vars, pydevd_io, pydevd_reload
from _pydevd_bundle import pydevd_bytecode_utils
from _pydevd_bundle import pydevd_xml
from _pydevd_bundle import pydevd_vm_type
import sys
import traceback
from _pydevd_bundle.pydevd_utils import quote_smart as quote, compare_object_attrs_key, \
    notify_about_gevent_if_needed, isinstance_checked, ScopeRequest, getattr_checked, Timer
from _pydev_bundle import pydev_log, fsnotify
from _pydev_bundle.pydev_log import exception as pydev_log_exception
from _pydev_bundle import _pydev_completer

from pydevd_tracing import get_exception_traceback_str
from _pydevd_bundle import pydevd_console
from _pydev_bundle.pydev_monkey import disable_trace_thread_modules, enable_trace_thread_modules
from io import StringIO

# CMD_XXX constants imported for backward compatibility
from _pydevd_bundle.pydevd_comm_constants import *  # @UnusedWildImport

# Socket import aliases:
AF_INET, SOCK_STREAM, SHUT_WR, SOL_SOCKET, IPPROTO_TCP, socket = (
    socket_module.AF_INET,
    socket_module.SOCK_STREAM,
    socket_module.SHUT_WR,
    socket_module.SOL_SOCKET,
    socket_module.IPPROTO_TCP,
    socket_module.socket,
)

if IS_WINDOWS and not IS_JYTHON:
    SO_EXCLUSIVEADDRUSE = socket_module.SO_EXCLUSIVEADDRUSE
if not IS_WASM:
    SO_REUSEADDR = socket_module.SO_REUSEADDR


class ReaderThread(PyDBDaemonThread):
    ''' reader thread reads and dispatches commands in an infinite loop '''

    def __init__(self, sock, py_db, PyDevJsonCommandProcessor, process_net_command, terminate_on_socket_close=True):
        assert sock is not None
        PyDBDaemonThread.__init__(self, py_db)
        self.__terminate_on_socket_close = terminate_on_socket_close

        self.sock = sock
        self._buffer = b''
        self.name = "pydevd.Reader"
        self.process_net_command = process_net_command
        self.process_net_command_json = PyDevJsonCommandProcessor(self._from_json).process_net_command_json

    def _from_json(self, json_msg, update_ids_from_dap=False):
        return pydevd_base_schema.from_json(json_msg, update_ids_from_dap, on_dict_loaded=self._on_dict_loaded)

    def _on_dict_loaded(self, dct):
        for listener in self.py_db.dap_messages_listeners:
            listener.after_receive(dct)

    @overrides(PyDBDaemonThread.do_kill_pydev_thread)
    def do_kill_pydev_thread(self):
        PyDBDaemonThread.do_kill_pydev_thread(self)
        # Note that we no longer shutdown the reader, just the writer. The idea is that we shutdown
        # the writer to send that the communication has finished, then, the client will shutdown its
        # own writer when it receives an empty read, at which point this reader will also shutdown.

        # That way, we can *almost* guarantee that all messages have been properly sent -- it's not
        # completely guaranteed because it's possible that the process exits before the whole
        # message was sent as having this thread alive won't stop the process from exiting -- we
        # have a timeout when exiting the process waiting for this thread to finish -- see:
        # PyDB.dispose_and_kill_all_pydevd_threads()).

        # try:
        #    self.sock.shutdown(SHUT_RD)
        # except:
        #    pass
        # try:
        #    self.sock.close()
        # except:
        #    pass

    def _read(self, size):
        while True:
            buffer_len = len(self._buffer)
            if buffer_len == size:
                ret = self._buffer
                self._buffer = b''
                return ret

            if buffer_len > size:
                ret = self._buffer[:size]
                self._buffer = self._buffer[size:]
                return ret

            try:
                r = self.sock.recv(max(size - buffer_len, 1024))
            except OSError:
                return b''
            if not r:
                return b''
            self._buffer += r

    def _read_line(self):
        while True:
            i = self._buffer.find(b'\n')
            if i != -1:
                i += 1  # Add the newline to the return
                ret = self._buffer[:i]
                self._buffer = self._buffer[i:]
                return ret
            else:
                try:
                    r = self.sock.recv(1024)
                except OSError:
                    return b''
                if not r:
                    return b''
                self._buffer += r

    @overrides(PyDBDaemonThread._on_run)
    def _on_run(self):
        try:
            content_len = -1

            while True:
                # i.e.: even if we received a kill, we should only exit the ReaderThread when the
                # client itself closes the connection (although on kill received we stop actually
                # processing anything read).
                try:
                    notify_about_gevent_if_needed()
                    line = self._read_line()

                    if len(line) == 0:
                        pydev_log.debug('ReaderThread: empty contents received (len(line) == 0).')
                        self._terminate_on_socket_close()
                        return  # Finished communication.

                    if self._kill_received:
                        continue

                    if line.startswith(b'Content-Length:'):
                        content_len = int(line.strip().split(b':', 1)[1])
                        continue

                    if content_len != -1:
                        # If we previously received a content length, read until a '\r\n'.
                        if line == b'\r\n':
                            json_contents = self._read(content_len)

                            content_len = -1

                            if len(json_contents) == 0:
                                pydev_log.debug('ReaderThread: empty contents received (len(json_contents) == 0).')
                                self._terminate_on_socket_close()
                                return  # Finished communication.

                            if self._kill_received:
                                continue

                            # We just received a json message, let's process it.
                            self.process_net_command_json(self.py_db, json_contents)

                        continue
                    else:
                        # No content len, regular line-based protocol message (remove trailing new-line).
                        if line.endswith(b'\n\n'):
                            line = line[:-2]

                        elif line.endswith(b'\n'):
                            line = line[:-1]

                        elif line.endswith(b'\r'):
                            line = line[:-1]
                except:
                    if not self._kill_received:
                        pydev_log_exception()
                        self._terminate_on_socket_close()
                    return  # Finished communication.

                # Note: the java backend is always expected to pass utf-8 encoded strings. We now work with str
                # internally and thus, we may need to convert to the actual encoding where needed (i.e.: filenames
                # on python 2 may need to be converted to the filesystem encoding).
                if hasattr(line, 'decode'):
                    line = line.decode('utf-8')

                if DebugInfoHolder.DEBUG_TRACE_LEVEL >= 3:
                    pydev_log.debug('debugger: received >>%s<<\n', line)

                args = line.split('\t', 2)
                try:
                    cmd_id = int(args[0])
                    if DebugInfoHolder.DEBUG_TRACE_LEVEL >= 3:
                        pydev_log.debug('Received command: %s %s\n', ID_TO_MEANING.get(str(cmd_id), '???'), line)
                    self.process_command(cmd_id, int(args[1]), args[2])
                except:
                    if sys is not None and pydev_log_exception is not None:  # Could happen at interpreter shutdown
                        pydev_log_exception("Can't process net command: %s.", line)

        except:
            if not self._kill_received:
                if sys is not None and pydev_log_exception is not None:  # Could happen at interpreter shutdown
                    pydev_log_exception()

            self._terminate_on_socket_close()
        finally:
            pydev_log.debug('ReaderThread: exit')

    def _terminate_on_socket_close(self):
        if self.__terminate_on_socket_close:
            self.py_db.dispose_and_kill_all_pydevd_threads()

    def process_command(self, cmd_id, seq, text):
        self.process_net_command(self.py_db, cmd_id, seq, text)


class FSNotifyThread(PyDBDaemonThread):

    def __init__(self, py_db, api, watch_dirs):
        PyDBDaemonThread.__init__(self, py_db)
        self.api = api
        self.name = "pydevd.FSNotifyThread"
        self.watcher = fsnotify.Watcher()
        self.watch_dirs = watch_dirs

    @overrides(PyDBDaemonThread._on_run)
    def _on_run(self):
        try:
            pydev_log.info('Watching directories for code reload:\n---\n%s\n---' % ('\n'.join(sorted(self.watch_dirs))))

            # i.e.: The first call to set_tracked_paths will do a full scan, so, do it in the thread
            # too (after everything is configured).
            self.watcher.set_tracked_paths(self.watch_dirs)
            while not self._kill_received:
                for change_enum, change_path in self.watcher.iter_changes():
                    # We're only interested in modified events
                    if change_enum == fsnotify.Change.modified:
                        pydev_log.info('Modified: %s', change_path)
                        self.api.request_reload_code(self.py_db, -1, None, change_path)
                    else:
                        pydev_log.info('Ignored (add or remove) change in: %s', change_path)
        except:
            pydev_log.exception('Error when waiting for filesystem changes in FSNotifyThread.')

    @overrides(PyDBDaemonThread.do_kill_pydev_thread)
    def do_kill_pydev_thread(self):
        self.watcher.dispose()
        PyDBDaemonThread.do_kill_pydev_thread(self)


class WriterThread(PyDBDaemonThread):
    ''' writer thread writes out the commands in an infinite loop '''

    def __init__(self, sock, py_db, terminate_on_socket_close=True):
        PyDBDaemonThread.__init__(self, py_db)
        self.sock = sock
        self.__terminate_on_socket_close = terminate_on_socket_close
        self.name = "pydevd.Writer"
        self._cmd_queue = _queue.Queue()
        if pydevd_vm_type.get_vm_type() == 'python':
            self.timeout = 0
        else:
            self.timeout = 0.1

    def add_command(self, cmd):
        ''' cmd is NetCommand '''
        if not self._kill_received:  # we don't take new data after everybody die
            self._cmd_queue.put(cmd, False)

    @overrides(PyDBDaemonThread._on_run)
    def _on_run(self):
        ''' just loop and write responses '''

        try:
            while True:
                try:
                    try:
                        cmd = self._cmd_queue.get(True, 0.1)
                    except _queue.Empty:
                        if self._kill_received:
                            pydev_log.debug('WriterThread: kill_received (sock.shutdown(SHUT_WR))')
                            try:
                                self.sock.shutdown(SHUT_WR)
                            except:
                                pass
                            # Note: don't close the socket, just send the shutdown,
                            # then, when no data is received on the reader, it can close
                            # the socket.
                            # See: https://blog.netherlabs.nl/articles/2009/01/18/the-ultimate-so_linger-page-or-why-is-my-tcp-not-reliable

                            # try:
                            #     self.sock.close()
                            # except:
                            #     pass

                            return  # break if queue is empty and _kill_received
                        else:
                            continue
                except:
                    # pydev_log.info('Finishing debug communication...(1)')
                    # when liberating the thread here, we could have errors because we were shutting down
                    # but the thread was still not liberated
                    return

                if cmd.as_dict is not None:
                    for listener in self.py_db.dap_messages_listeners:
                        listener.before_send(cmd.as_dict)

                notify_about_gevent_if_needed()
                cmd.send(self.sock)

                if cmd.id == CMD_EXIT:
                    pydev_log.debug('WriterThread: CMD_EXIT received')
                    break
                if time is None:
                    break  # interpreter shutdown
                time.sleep(self.timeout)
        except Exception:
            if self.__terminate_on_socket_close:
                self.py_db.dispose_and_kill_all_pydevd_threads()
                if DebugInfoHolder.DEBUG_TRACE_LEVEL > 0:
                    pydev_log_exception()
        finally:
            pydev_log.debug('WriterThread: exit')

    def empty(self):
        return self._cmd_queue.empty()

    @overrides(PyDBDaemonThread.do_kill_pydev_thread)
    def do_kill_pydev_thread(self):
        if not self._kill_received:
            # Add command before setting the kill flag (otherwise the command may not be added).
            exit_cmd = self.py_db.cmd_factory.make_exit_command(self.py_db)
            self.add_command(exit_cmd)

        PyDBDaemonThread.do_kill_pydev_thread(self)


def create_server_socket(host, port):
    try:
        server = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP)
        if IS_WINDOWS and not IS_JYTHON:
            server.setsockopt(SOL_SOCKET, SO_EXCLUSIVEADDRUSE, 1)
        elif not IS_WASM:
            server.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)

        server.bind((host, port))
        server.settimeout(None)
    except Exception:
        server.close()
        raise

    return server


def start_server(port):
    ''' binds to a port, waits for the debugger to connect '''
    s = create_server_socket(host='', port=port)

    try:
        s.listen(1)
        new_socket, _addr = s.accept()
        pydev_log.info("Connection accepted")
        # closing server socket is not necessary but we don't need it
        s.close()
        return new_socket
    except:
        pydev_log.exception("Could not bind to port: %s\n", port)
        raise


def start_client(host, port):
    ''' connects to a host/port '''
    pydev_log.info("Connecting to %s:%s", host, port)

    s = socket(AF_INET, SOCK_STREAM)

    #  Set TCP keepalive on an open socket.
    #  It activates after 1 second (TCP_KEEPIDLE,) of idleness,
    #  then sends a keepalive ping once every 3 seconds (TCP_KEEPINTVL),
    #  and closes the connection after 5 failed ping (TCP_KEEPCNT), or 15 seconds
    try:
        s.setsockopt(SOL_SOCKET, socket_module.SO_KEEPALIVE, 1)
    except (AttributeError, OSError):
        pass  # May not be available everywhere.
    try:
        s.setsockopt(socket_module.IPPROTO_TCP, socket_module.TCP_KEEPIDLE, 1)
    except (AttributeError, OSError):
        pass  # May not be available everywhere.
    try:
        s.setsockopt(socket_module.IPPROTO_TCP, socket_module.TCP_KEEPINTVL, 3)
    except (AttributeError, OSError):
        pass  # May not be available everywhere.
    try:
        s.setsockopt(socket_module.IPPROTO_TCP, socket_module.TCP_KEEPCNT, 5)
    except (AttributeError, OSError):
        pass  # May not be available everywhere.

    try:
        # 10 seconds default timeout
        timeout = int(os.environ.get('PYDEVD_CONNECT_TIMEOUT', 10))
        s.settimeout(timeout)
        s.connect((host, port))
        s.settimeout(None)  # no timeout after connected
        pydev_log.info("Connected.")
        return s
    except:
        pydev_log.exception("Could not connect to %s: %s", host, port)
        raise


INTERNAL_TERMINATE_THREAD = 1
INTERNAL_SUSPEND_THREAD = 2


class InternalThreadCommand(object):
    ''' internal commands are generated/executed by the debugger.

    The reason for their existence is that some commands have to be executed
    on specific threads. These are the InternalThreadCommands that get
    get posted to PyDB.
    '''

    def __init__(self, thread_id, method=None, *args, **kwargs):
        self.thread_id = thread_id
        self.method = method
        self.args = args
        self.kwargs = kwargs

    def can_be_executed_by(self, thread_id):
        '''By default, it must be in the same thread to be executed
        '''
        return self.thread_id == thread_id or self.thread_id.endswith('|' + thread_id)

    def do_it(self, dbg):
        try:
            if self.method is not None:
                self.method(dbg, *self.args, **self.kwargs)
            else:
                raise NotImplementedError("you have to override do_it")
        finally:
            self.args = None
            self.kwargs = None

    def __str__(self):
        return 'InternalThreadCommands(%s, %s, %s)' % (self.method, self.args, self.kwargs)

    __repr__ = __str__


class InternalThreadCommandForAnyThread(InternalThreadCommand):

    def __init__(self, thread_id, method=None, *args, **kwargs):
        assert thread_id == '*'

        InternalThreadCommand.__init__(self, thread_id, method, *args, **kwargs)

        self.executed = False
        self.lock = ForkSafeLock()

    def can_be_executed_by(self, thread_id):
        return True  # Can be executed by any thread.

    def do_it(self, dbg):
        with self.lock:
            if self.executed:
                return
            self.executed = True

        InternalThreadCommand.do_it(self, dbg)


def _send_io_message(py_db, s):
    cmd = py_db.cmd_factory.make_io_message(s, 2)
    if py_db.writer is not None:
        py_db.writer.add_command(cmd)


def internal_reload_code(dbg, seq, module_name, filename):
    try:
        found_module_to_reload = False
        if module_name is not None:
            module_name = module_name
            if module_name not in sys.modules:
                if '.' in module_name:
                    new_module_name = module_name.split('.')[-1]
                    if new_module_name in sys.modules:
                        module_name = new_module_name

        modules_to_reload = {}
        module = sys.modules.get(module_name)
        if module is not None:
            modules_to_reload[id(module)] = (module, module_name)

        if filename:
            filename = pydevd_file_utils.normcase(filename)
            for module_name, module in sys.modules.copy().items():
                f = getattr_checked(module, '__file__')
                if f is not None:
                    if f.endswith(('.pyc', '.pyo')):
                        f = f[:-1]

                    if pydevd_file_utils.normcase(f) == filename:
                        modules_to_reload[id(module)] = (module, module_name)

        if not modules_to_reload:
            if filename and module_name:
                _send_io_message(dbg, 'code reload: Unable to find module %s to reload for path: %s\n' % (module_name, filename))
            elif filename:
                _send_io_message(dbg, 'code reload: Unable to find module to reload for path: %s\n' % (filename,))
            elif module_name:
                _send_io_message(dbg, 'code reload: Unable to find module to reload: %s\n' % (module_name,))

        else:
            # Too much info...
            # _send_io_message(dbg, 'code reload: This usually means you are trying to reload the __main__ module (which cannot be reloaded).\n')
            for module, module_name in modules_to_reload.values():
                _send_io_message(dbg, 'code reload: Start reloading module: "' + module_name + '" ... \n')
                found_module_to_reload = True

                if pydevd_reload.xreload(module):
                    _send_io_message(dbg, 'code reload: reload finished\n')
                else:
                    _send_io_message(dbg, 'code reload: reload finished without applying any change\n')

        cmd = dbg.cmd_factory.make_reloaded_code_message(seq, found_module_to_reload)
        dbg.writer.add_command(cmd)
    except:
        pydev_log.exception('Error reloading code')


class InternalGetThreadStack(InternalThreadCommand):
    '''
    This command will either wait for a given thread to be paused to get its stack or will provide
    it anyways after a timeout (in which case the stack will be gotten but local variables won't
    be available and it'll not be possible to interact with the frame as it's not actually
    stopped in a breakpoint).
    '''

    def __init__(self, seq, thread_id, py_db, set_additional_thread_info, fmt, timeout=.5, start_frame=0, levels=0):
        InternalThreadCommand.__init__(self, thread_id)
        self._py_db = weakref.ref(py_db)
        self._timeout = time.time() + timeout
        self.seq = seq
        self._cmd = None
        self._fmt = fmt
        self._start_frame = start_frame
        self._levels = levels

        # Note: receives set_additional_thread_info to avoid a circular import
        # in this module.
        self._set_additional_thread_info = set_additional_thread_info

    @overrides(InternalThreadCommand.can_be_executed_by)
    def can_be_executed_by(self, _thread_id):
        timed_out = time.time() >= self._timeout

        py_db = self._py_db()
        t = pydevd_find_thread_by_id(self.thread_id)
        frame = None
        if t and not getattr(t, 'pydev_do_not_trace', None):
            additional_info = self._set_additional_thread_info(t)
            frame = additional_info.get_topmost_frame(t)
        try:
            self._cmd = py_db.cmd_factory.make_get_thread_stack_message(
                py_db, self.seq, self.thread_id, frame, self._fmt, must_be_suspended=not timed_out, start_frame=self._start_frame, levels=self._levels)
        finally:
            frame = None
            t = None

        return self._cmd is not None or timed_out

    @overrides(InternalThreadCommand.do_it)
    def do_it(self, dbg):
        if self._cmd is not None:
            dbg.writer.add_command(self._cmd)
            self._cmd = None


def internal_step_in_thread(py_db, thread_id, cmd_id, set_additional_thread_info):
    thread_to_step = pydevd_find_thread_by_id(thread_id)
    if thread_to_step is not None:
        info = set_additional_thread_info(thread_to_step)
        info.pydev_original_step_cmd = cmd_id
        info.pydev_step_cmd = cmd_id
        info.pydev_step_stop = None
        info.pydev_state = STATE_RUN

    if py_db.stepping_resumes_all_threads:
        resume_threads('*', except_thread=thread_to_step)


def internal_smart_step_into(py_db, thread_id, offset, child_offset, set_additional_thread_info):
    thread_to_step = pydevd_find_thread_by_id(thread_id)
    if thread_to_step is not None:
        info = set_additional_thread_info(thread_to_step)
        info.pydev_original_step_cmd = CMD_SMART_STEP_INTO
        info.pydev_step_cmd = CMD_SMART_STEP_INTO
        info.pydev_step_stop = None
        info.pydev_smart_parent_offset = int(offset)
        info.pydev_smart_child_offset = int(child_offset)
        info.pydev_state = STATE_RUN

    if py_db.stepping_resumes_all_threads:
        resume_threads('*', except_thread=thread_to_step)


class InternalSetNextStatementThread(InternalThreadCommand):

    def __init__(self, thread_id, cmd_id, line, func_name, seq=0):
        '''
        cmd_id may actually be one of:

        CMD_RUN_TO_LINE
        CMD_SET_NEXT_STATEMENT
        CMD_SMART_STEP_INTO
        '''
        self.thread_id = thread_id
        self.cmd_id = cmd_id
        self.line = line
        self.seq = seq

        self.func_name = func_name

    def do_it(self, dbg):
        t = pydevd_find_thread_by_id(self.thread_id)
        if t is not None:
            info = t.additional_info
            info.pydev_original_step_cmd = self.cmd_id
            info.pydev_step_cmd = self.cmd_id
            info.pydev_step_stop = None
            info.pydev_next_line = int(self.line)
            info.pydev_func_name = self.func_name
            info.pydev_message = str(self.seq)
            info.pydev_smart_parent_offset = -1
            info.pydev_smart_child_offset = -1
            info.pydev_state = STATE_RUN


@silence_warnings_decorator
def internal_get_variable_json(py_db, request):
    '''
        :param VariablesRequest request:
    '''
    arguments = request.arguments  # : :type arguments: VariablesArguments
    variables_reference = arguments.variablesReference
    scope = None
    if isinstance_checked(variables_reference, ScopeRequest):
        scope = variables_reference
        variables_reference = variables_reference.variable_reference

    fmt = arguments.format
    if hasattr(fmt, 'to_dict'):
        fmt = fmt.to_dict()

    variables = []
    try:
        try:
            variable = py_db.suspended_frames_manager.get_variable(variables_reference)
        except KeyError:
            pass
        else:
            for child_var in variable.get_children_variables(fmt=fmt, scope=scope):
                variables.append(child_var.get_var_data(fmt=fmt))
    except:
        try:
            exc, exc_type, tb = sys.exc_info()
            err = ''.join(traceback.format_exception(exc, exc_type, tb))
            variables = [{
                'name': '<error>',
                'value': err,
                'type': '<error>',
                'variablesReference': 0
            }]
        except:
            err = '<Internal error - unable to get traceback when getting variables>'
            pydev_log.exception(err)
            variables = []

    body = VariablesResponseBody(variables)
    variables_response = pydevd_base_schema.build_response(request, kwargs={'body':body})
    py_db.writer.add_command(NetCommand(CMD_RETURN, 0, variables_response, is_json=True))


class InternalGetVariable(InternalThreadCommand):
    ''' gets the value of a variable '''

    def __init__(self, seq, thread_id, frame_id, scope, attrs):
        self.sequence = seq
        self.thread_id = thread_id
        self.frame_id = frame_id
        self.scope = scope
        self.attributes = attrs

    @silence_warnings_decorator
    def do_it(self, dbg):
        ''' Converts request into python variable '''
        try:
            xml = StringIO()
            xml.write("<xml>")
            type_name, val_dict = pydevd_vars.resolve_compound_variable_fields(
                dbg, self.thread_id, self.frame_id, self.scope, self.attributes)
            if val_dict is None:
                val_dict = {}

            # assume properly ordered if resolver returns 'OrderedDict'
            # check type as string to support OrderedDict backport for older Python
            keys = list(val_dict)
            if not (type_name == "OrderedDict" or val_dict.__class__.__name__ == "OrderedDict" or IS_PY36_OR_GREATER):
                keys = sorted(keys, key=compare_object_attrs_key)

            timer = Timer()
            for k in keys:
                val = val_dict[k]
                evaluate_full_value = pydevd_xml.should_evaluate_full_value(val)
                xml.write(pydevd_xml.var_to_xml(val, k, evaluate_full_value=evaluate_full_value))
                timer.report_if_compute_repr_attr_slow(self.attributes, k, type(val))

            xml.write("</xml>")
            cmd = dbg.cmd_factory.make_get_variable_message(self.sequence, xml.getvalue())
            xml.close()
            dbg.writer.add_command(cmd)
        except Exception:
            cmd = dbg.cmd_factory.make_error_message(
                self.sequence, "Error resolving variables %s" % (get_exception_traceback_str(),))
            dbg.writer.add_command(cmd)


class InternalGetArray(InternalThreadCommand):

    def __init__(self, seq, roffset, coffset, rows, cols, format, thread_id, frame_id, scope, attrs):
        self.sequence = seq
        self.thread_id = thread_id
        self.frame_id = frame_id
        self.scope = scope
        self.name = attrs.split("\t")[-1]
        self.attrs = attrs
        self.roffset = int(roffset)
        self.coffset = int(coffset)
        self.rows = int(rows)
        self.cols = int(cols)
        self.format = format

    def do_it(self, dbg):
        try:
            frame = dbg.find_frame(self.thread_id, self.frame_id)
            var = pydevd_vars.eval_in_context(self.name, frame.f_globals, frame.f_locals, py_db=dbg)
            xml = pydevd_vars.table_like_struct_to_xml(var, self.name, self.roffset, self.coffset, self.rows, self.cols, self.format)
            cmd = dbg.cmd_factory.make_get_array_message(self.sequence, xml)
            dbg.writer.add_command(cmd)
        except:
            cmd = dbg.cmd_factory.make_error_message(self.sequence, "Error resolving array: " + get_exception_traceback_str())
            dbg.writer.add_command(cmd)


def internal_change_variable(dbg, seq, thread_id, frame_id, scope, attr, value):
    ''' Changes the value of a variable '''
    try:
        frame = dbg.find_frame(thread_id, frame_id)
        if frame is not None:
            result = pydevd_vars.change_attr_expression(frame, attr, value, dbg)
        else:
            result = None
        xml = "<xml>"
        xml += pydevd_xml.var_to_xml(result, "")
        xml += "</xml>"
        cmd = dbg.cmd_factory.make_variable_changed_message(seq, xml)
        dbg.writer.add_command(cmd)
    except Exception:
        cmd = dbg.cmd_factory.make_error_message(seq, "Error changing variable attr:%s expression:%s traceback:%s" % (attr, value, get_exception_traceback_str()))
        dbg.writer.add_command(cmd)


def internal_change_variable_json(py_db, request):
    '''
    The pydevd_vars.change_attr_expression(thread_id, frame_id, attr, value, dbg) can only
    deal with changing at a frame level, so, currently changing the contents of something
    in a different scope is currently not supported.

    :param SetVariableRequest request:
    '''
    # : :type arguments: SetVariableArguments
    arguments = request.arguments
    variables_reference = arguments.variablesReference
    scope = None
    if isinstance_checked(variables_reference, ScopeRequest):
        scope = variables_reference
        variables_reference = variables_reference.variable_reference

    fmt = arguments.format
    if hasattr(fmt, 'to_dict'):
        fmt = fmt.to_dict()

    try:
        variable = py_db.suspended_frames_manager.get_variable(variables_reference)
    except KeyError:
        variable = None

    if variable is None:
        _write_variable_response(
            py_db, request, value='', success=False, message='Unable to find variable container to change: %s.' % (variables_reference,))
        return

    child_var = variable.change_variable(arguments.name, arguments.value, py_db, fmt=fmt)

    if child_var is None:
        _write_variable_response(
            py_db, request, value='', success=False, message='Unable to change: %s.' % (arguments.name,))
        return

    var_data = child_var.get_var_data(fmt=fmt)
    body = SetVariableResponseBody(
        value=var_data['value'],
        type=var_data['type'],
        variablesReference=var_data.get('variablesReference'),
        namedVariables=var_data.get('namedVariables'),
        indexedVariables=var_data.get('indexedVariables'),
    )
    variables_response = pydevd_base_schema.build_response(request, kwargs={'body':body})
    py_db.writer.add_command(NetCommand(CMD_RETURN, 0, variables_response, is_json=True))


def _write_variable_response(py_db, request, value, success, message):
    body = SetVariableResponseBody('')
    variables_response = pydevd_base_schema.build_response(
        request,
        kwargs={
            'body':body,
            'success': False,
            'message': message
    })
    cmd = NetCommand(CMD_RETURN, 0, variables_response, is_json=True)
    py_db.writer.add_command(cmd)


@silence_warnings_decorator
def internal_get_frame(dbg, seq, thread_id, frame_id):
    ''' Converts request into python variable '''
    try:
        frame = dbg.find_frame(thread_id, frame_id)
        if frame is not None:
            hidden_ns = pydevconsole.get_ipython_hidden_vars()
            xml = "<xml>"
            xml += pydevd_xml.frame_vars_to_xml(frame.f_locals, hidden_ns)
            del frame
            xml += "</xml>"
            cmd = dbg.cmd_factory.make_get_frame_message(seq, xml)
            dbg.writer.add_command(cmd)
        else:
            # pydevd_vars.dump_frames(thread_id)
            # don't print this error: frame not found: means that the client is not synchronized (but that's ok)
            cmd = dbg.cmd_factory.make_error_message(seq, "Frame not found: %s from thread: %s" % (frame_id, thread_id))
            dbg.writer.add_command(cmd)
    except:
        cmd = dbg.cmd_factory.make_error_message(seq, "Error resolving frame: %s from thread: %s" % (frame_id, thread_id))
        dbg.writer.add_command(cmd)


def internal_get_smart_step_into_variants(dbg, seq, thread_id, frame_id, start_line, end_line, set_additional_thread_info):
    try:
        thread = pydevd_find_thread_by_id(thread_id)
        frame = dbg.find_frame(thread_id, frame_id)

        if thread is None or frame is None:
            cmd = dbg.cmd_factory.make_error_message(seq, "Frame not found: %s from thread: %s" % (frame_id, thread_id))
            dbg.writer.add_command(cmd)
            return

        if pydevd_bytecode_utils is None:
            variants = []
        else:
            variants = pydevd_bytecode_utils.calculate_smart_step_into_variants(frame, int(start_line), int(end_line))

        info = set_additional_thread_info(thread)

        # Store the last request (may be used afterwards when stepping).
        info.pydev_smart_step_into_variants = tuple(variants)
        xml = "<xml>"

        for variant in variants:
            if variant.children_variants:
                for child_variant in variant.children_variants:
                    # If there are child variants, the current one is just an intermediary, so,
                    # just create variants for the child (notifying properly about the parent too).
                    xml += '<variant name="%s" isVisited="%s" line="%s" offset="%s" childOffset="%s" callOrder="%s"/>' % (
                        quote(child_variant.name),
                        str(child_variant.is_visited).lower(),
                        child_variant.line,
                        variant.offset,
                        child_variant.offset,
                        child_variant.call_order,
                    )
            else:
                xml += '<variant name="%s" isVisited="%s" line="%s" offset="%s" childOffset="-1" callOrder="%s"/>' % (
                    quote(variant.name),
                    str(variant.is_visited).lower(),
                    variant.line,
                    variant.offset,
                    variant.call_order,
                )

        xml += "</xml>"
        cmd = NetCommand(CMD_GET_SMART_STEP_INTO_VARIANTS, seq, xml)
        dbg.writer.add_command(cmd)
    except:
        # Error is expected (if `dis` module cannot be used -- i.e.: Jython).
        pydev_log.exception('Error calculating Smart Step Into Variants.')
        cmd = dbg.cmd_factory.make_error_message(
            seq, "Error getting smart step into variants for frame: %s from thread: %s"
            % (frame_id, thread_id))
        dbg.writer.add_command(cmd)


def internal_get_step_in_targets_json(dbg, seq, thread_id, frame_id, request, set_additional_thread_info):
    try:
        thread = pydevd_find_thread_by_id(thread_id)
        frame = dbg.find_frame(thread_id, frame_id)

        if thread is None or frame is None:
            body = StepInTargetsResponseBody([])
            variables_response = pydevd_base_schema.build_response(
                request,
                kwargs={
                    'body': body,
                    'success': False,
                    'message': 'Thread to get step in targets seems to have resumed already.'
                })
            cmd = NetCommand(CMD_RETURN, 0, variables_response, is_json=True)
            dbg.writer.add_command(cmd)
            return

        start_line = 0
        end_line = 99999999
        if pydevd_bytecode_utils is None:
            variants = []
        else:
            variants = pydevd_bytecode_utils.calculate_smart_step_into_variants(frame, start_line, end_line)

        info = set_additional_thread_info(thread)
        targets = []
        counter = itertools.count(0)
        target_id_to_variant = {}
        for variant in variants:
            if not variant.is_visited:
                if variant.children_variants:
                    for child_variant in variant.children_variants:
                        target_id = next(counter)

                        if child_variant.call_order > 1:
                            targets.append(StepInTarget(id=target_id, label='%s (call %s)' % (child_variant.name, child_variant.call_order),))
                        else:
                            targets.append(StepInTarget(id=target_id, label=child_variant.name))
                        target_id_to_variant[target_id] = child_variant

                        if len(targets) >= 15:  # Show at most 15 targets.
                            break
                else:
                    target_id = next(counter)
                    if variant.call_order > 1:
                        targets.append(StepInTarget(id=target_id, label='%s (call %s)' % (variant.name, variant.call_order),))
                    else:
                        targets.append(StepInTarget(id=target_id, label=variant.name))
                    target_id_to_variant[target_id] = variant

                    if len(targets) >= 15:  # Show at most 15 targets.
                        break

        # Store the last request (may be used afterwards when stepping).
        info.pydev_smart_step_into_variants = tuple(variants)
        info.target_id_to_smart_step_into_variant = target_id_to_variant

        body = StepInTargetsResponseBody(targets=targets)
        response = pydevd_base_schema.build_response(request, kwargs={'body': body})
        cmd = NetCommand(CMD_RETURN, 0, response, is_json=True)
        dbg.writer.add_command(cmd)
    except Exception as e:
        # Error is expected (if `dis` module cannot be used -- i.e.: Jython).
        pydev_log.exception('Error calculating Smart Step Into Variants.')
        body = StepInTargetsResponseBody([])
        variables_response = pydevd_base_schema.build_response(
            request,
            kwargs={
                'body': body,
                'success': False,
                'message': str(e)
            })
        cmd = NetCommand(CMD_RETURN, 0, variables_response, is_json=True)
        dbg.writer.add_command(cmd)


def internal_get_next_statement_targets(dbg, seq, thread_id, frame_id):
    ''' gets the valid line numbers for use with set next statement '''
    try:
        frame = dbg.find_frame(thread_id, frame_id)
        if frame is not None:
            code = frame.f_code
            xml = "<xml>"
            try:
                linestarts = dis.findlinestarts(code)
            except:
                # i.e.: jython doesn't provide co_lnotab, so, we can only keep at the current line.
                xml += "<line>%d</line>" % (frame.f_lineno,)
            else:
                for _, line in linestarts:
                    xml += "<line>%d</line>" % (line,)
            del frame
            xml += "</xml>"
            cmd = dbg.cmd_factory.make_get_next_statement_targets_message(seq, xml)
            dbg.writer.add_command(cmd)
        else:
            cmd = dbg.cmd_factory.make_error_message(seq, "Frame not found: %s from thread: %s" % (frame_id, thread_id))
            dbg.writer.add_command(cmd)
    except:
        cmd = dbg.cmd_factory.make_error_message(seq, "Error resolving frame: %s from thread: %s" % (frame_id, thread_id))
        dbg.writer.add_command(cmd)


def _evaluate_response(py_db, request, result, error_message=''):
    is_error = isinstance(result, ExceptionOnEvaluate)
    if is_error:
        result = result.result
    if not error_message:
        body = pydevd_schema.EvaluateResponseBody(result=result, variablesReference=0)
        variables_response = pydevd_base_schema.build_response(request, kwargs={'body':body})
        py_db.writer.add_command(NetCommand(CMD_RETURN, 0, variables_response, is_json=True))
    else:
        body = pydevd_schema.EvaluateResponseBody(result=result, variablesReference=0)
        variables_response = pydevd_base_schema.build_response(request, kwargs={
            'body':body, 'success':False, 'message': error_message})
        py_db.writer.add_command(NetCommand(CMD_RETURN, 0, variables_response, is_json=True))


_global_frame = None


def internal_evaluate_expression_json(py_db, request, thread_id):
    '''
    :param EvaluateRequest request:
    '''
    global _global_frame
    # : :type arguments: EvaluateArguments

    arguments = request.arguments
    expression = arguments.expression
    frame_id = arguments.frameId
    context = arguments.context
    fmt = arguments.format
    if hasattr(fmt, 'to_dict'):
        fmt = fmt.to_dict()

    ctx = NULL
    if context == 'repl':
        if not py_db.is_output_redirected:
            ctx = pydevd_io.redirect_stream_to_pydb_io_messages_context()
    else:
        # If we're not in a repl (watch, hover, ...) don't show warnings.
        ctx = filter_all_warnings()

    with ctx:
        try_exec = False
        if frame_id is None:
            if _global_frame is None:
                # Lazily create a frame to be used for evaluation with no frame id.

                def __create_frame():
                    yield sys._getframe()

                _global_frame = next(__create_frame())

            frame = _global_frame
            try_exec = True  # Always exec in this case
            eval_result = None
        else:
            frame = py_db.find_frame(thread_id, frame_id)

            eval_result = pydevd_vars.evaluate_expression(py_db, frame, expression, is_exec=False)
            is_error = isinstance_checked(eval_result, ExceptionOnEvaluate)
            if is_error:
                if context == 'hover':  # In a hover it doesn't make sense to do an exec.
                    _evaluate_response(py_db, request, result='', error_message='Exception occurred during evaluation.')
                    return
                elif context == 'watch':
                    # If it's a watch, don't show it as an exception object, rather, format
                    # it and show it as a string (with success=False).
                    msg = '%s: %s' % (
                        eval_result.result.__class__.__name__, eval_result.result,)
                    _evaluate_response(py_db, request, result=msg, error_message=msg)
                    return
                else:
                    # We only try the exec if the failure we had was due to not being able
                    # to evaluate the expression.
                    try:
                        pydevd_vars.compile_as_eval(expression)
                    except Exception:
                        try_exec = context == 'repl'
                    else:
                        try_exec = False
                        if context == 'repl':
                            # In the repl we should show the exception to the user.
                            _evaluate_response_return_exception(py_db, request, eval_result.etype, eval_result.result, eval_result.tb)
                            return

        if try_exec:
            try:
                pydevd_vars.evaluate_expression(py_db, frame, expression, is_exec=True)
            except (Exception, KeyboardInterrupt):
                _evaluate_response_return_exception(py_db, request, *sys.exc_info())
                return
            # No result on exec.
            _evaluate_response(py_db, request, result='')
            return

        # Ok, we have the result (could be an error), let's put it into the saved variables.
        frame_tracker = py_db.suspended_frames_manager.get_frame_tracker(thread_id)
        if frame_tracker is None:
            # This is not really expected.
            _evaluate_response(py_db, request, result='', error_message='Thread id: %s is not current thread id.' % (thread_id,))
            return

        safe_repr_custom_attrs = {}
        if context == 'clipboard':
            safe_repr_custom_attrs = dict(
                maxstring_outer=2 ** 64,
                maxstring_inner=2 ** 64,
                maxother_outer=2 ** 64,
                maxother_inner=2 ** 64,
            )

        if context == 'repl' and eval_result is None:
            # We don't want "None" to appear when typing in the repl.
            body = pydevd_schema.EvaluateResponseBody(
                result='',
                variablesReference=0,
            )

        else:
            variable = frame_tracker.obtain_as_variable(expression, eval_result, frame=frame)
            var_data = variable.get_var_data(fmt=fmt, context=context, **safe_repr_custom_attrs)

            body = pydevd_schema.EvaluateResponseBody(
                result=var_data['value'],
                variablesReference=var_data.get('variablesReference', 0),
                type=var_data.get('type'),
                presentationHint=var_data.get('presentationHint'),
                namedVariables=var_data.get('namedVariables'),
                indexedVariables=var_data.get('indexedVariables'),
            )
        variables_response = pydevd_base_schema.build_response(request, kwargs={'body':body})
        py_db.writer.add_command(NetCommand(CMD_RETURN, 0, variables_response, is_json=True))


def _evaluate_response_return_exception(py_db, request, exc_type, exc, initial_tb):
    try:
        tb = initial_tb

        # Show the traceback without pydevd frames.
        temp_tb = tb
        while temp_tb:
            if py_db.get_file_type(temp_tb.tb_frame) == PYDEV_FILE:
                tb = temp_tb.tb_next
            temp_tb = temp_tb.tb_next

        if tb is None:
            tb = initial_tb
        err = ''.join(traceback.format_exception(exc_type, exc, tb))

        # Make sure we don't keep references to them.
        exc = None
        exc_type = None
        tb = None
        temp_tb = None
        initial_tb = None
    except:
        err = '<Internal error - unable to get traceback when evaluating expression>'
        pydev_log.exception(err)

    # Currently there is an issue in VSC where returning success=false for an
    # eval request, in repl context, VSC does not show the error response in
    # the debug console. So return the error message in result as well.
    _evaluate_response(py_db, request, result=err, error_message=err)


@silence_warnings_decorator
def internal_evaluate_expression(dbg, seq, thread_id, frame_id, expression, is_exec, trim_if_too_big, attr_to_set_result):
    ''' gets the value of a variable '''
    try:
        frame = dbg.find_frame(thread_id, frame_id)
        if frame is not None:
            result = pydevd_vars.evaluate_expression(dbg, frame, expression, is_exec)
            if attr_to_set_result != "":
                pydevd_vars.change_attr_expression(frame, attr_to_set_result, expression, dbg, result)
        else:
            result = None

        xml = "<xml>"
        xml += pydevd_xml.var_to_xml(result, expression, trim_if_too_big)
        xml += "</xml>"
        cmd = dbg.cmd_factory.make_evaluate_expression_message(seq, xml)
        dbg.writer.add_command(cmd)
    except:
        exc = get_exception_traceback_str()
        cmd = dbg.cmd_factory.make_error_message(seq, "Error evaluating expression " + exc)
        dbg.writer.add_command(cmd)


def _set_expression_response(py_db, request, result, error_message):
    body = pydevd_schema.SetExpressionResponseBody(result='', variablesReference=0)
    variables_response = pydevd_base_schema.build_response(request, kwargs={
        'body':body, 'success':False, 'message': error_message})
    py_db.writer.add_command(NetCommand(CMD_RETURN, 0, variables_response, is_json=True))


def internal_set_expression_json(py_db, request, thread_id):
    # : :type arguments: SetExpressionArguments

    arguments = request.arguments
    expression = arguments.expression
    frame_id = arguments.frameId
    value = arguments.value
    fmt = arguments.format
    if hasattr(fmt, 'to_dict'):
        fmt = fmt.to_dict()

    frame = py_db.find_frame(thread_id, frame_id)
    exec_code = '%s = (%s)' % (expression, value)
    result = pydevd_vars.evaluate_expression(py_db, frame, exec_code, is_exec=True)
    is_error = isinstance(result, ExceptionOnEvaluate)

    if is_error:
        _set_expression_response(py_db, request, result, error_message='Error executing: %s' % (exec_code,))
        return

    # Ok, we have the result (could be an error), let's put it into the saved variables.
    frame_tracker = py_db.suspended_frames_manager.get_frame_tracker(thread_id)
    if frame_tracker is None:
        # This is not really expected.
        _set_expression_response(py_db, request, result, error_message='Thread id: %s is not current thread id.' % (thread_id,))
        return

    # Now that the exec is done, get the actual value changed to return.
    result = pydevd_vars.evaluate_expression(py_db, frame, expression, is_exec=False)
    variable = frame_tracker.obtain_as_variable(expression, result, frame=frame)
    var_data = variable.get_var_data(fmt=fmt)

    body = pydevd_schema.SetExpressionResponseBody(
        value=var_data['value'],
        variablesReference=var_data.get('variablesReference', 0),
        type=var_data.get('type'),
        presentationHint=var_data.get('presentationHint'),
        namedVariables=var_data.get('namedVariables'),
        indexedVariables=var_data.get('indexedVariables'),
    )
    variables_response = pydevd_base_schema.build_response(request, kwargs={'body':body})
    py_db.writer.add_command(NetCommand(CMD_RETURN, 0, variables_response, is_json=True))


def internal_get_completions(dbg, seq, thread_id, frame_id, act_tok, line=-1, column=-1):
    '''
    Note that if the column is >= 0, the act_tok is considered text and the actual
    activation token/qualifier is computed in this command.
    '''
    try:
        remove_path = None
        try:
            qualifier = ''
            if column >= 0:
                token_and_qualifier = extract_token_and_qualifier(act_tok, line, column)
                act_tok = token_and_qualifier[0]
                if act_tok:
                    act_tok += '.'
                qualifier = token_and_qualifier[1]

            frame = dbg.find_frame(thread_id, frame_id)
            if frame is not None:
                completions = _pydev_completer.generate_completions(frame, act_tok)

                # Note that qualifier and start are only actually valid for the
                # Debug Adapter Protocol (for the line-based protocol, the IDE
                # is required to filter the completions returned).
                cmd = dbg.cmd_factory.make_get_completions_message(
                    seq, completions, qualifier, start=column - len(qualifier))
                dbg.writer.add_command(cmd)
            else:
                cmd = dbg.cmd_factory.make_error_message(seq, "internal_get_completions: Frame not found: %s from thread: %s" % (frame_id, thread_id))
                dbg.writer.add_command(cmd)

        finally:
            if remove_path is not None:
                sys.path.remove(remove_path)

    except:
        exc = get_exception_traceback_str()
        sys.stderr.write('%s\n' % (exc,))
        cmd = dbg.cmd_factory.make_error_message(seq, "Error evaluating expression " + exc)
        dbg.writer.add_command(cmd)


def internal_get_description(dbg, seq, thread_id, frame_id, expression):
    ''' Fetch the variable description stub from the debug console
    '''
    try:
        frame = dbg.find_frame(thread_id, frame_id)
        description = pydevd_console.get_description(frame, thread_id, frame_id, expression)
        description = pydevd_xml.make_valid_xml_value(quote(description, '/>_= \t'))
        description_xml = '<xml><var name="" type="" value="%s"/></xml>' % description
        cmd = dbg.cmd_factory.make_get_description_message(seq, description_xml)
        dbg.writer.add_command(cmd)
    except:
        exc = get_exception_traceback_str()
        cmd = dbg.cmd_factory.make_error_message(seq, "Error in fetching description" + exc)
        dbg.writer.add_command(cmd)


def build_exception_info_response(dbg, thread_id, thread, request_seq, set_additional_thread_info, iter_visible_frames_info, max_frames):
    '''
    :return ExceptionInfoResponse
    '''
    additional_info = set_additional_thread_info(thread)
    topmost_frame = additional_info.get_topmost_frame(thread)

    current_paused_frame_name = ''

    source_path = ''  # This is an extra bit of data used by Visual Studio
    stack_str_lst = []
    name = None
    description = None

    if topmost_frame is not None:
        try:
            try:
                frames_list = dbg.suspended_frames_manager.get_frames_list(thread_id)
                while frames_list is not None and len(frames_list):
                    frames = []

                    frame = None

                    if not name:
                        exc_type = frames_list.exc_type
                        if exc_type is not None:
                            try:
                                name = exc_type.__qualname__
                            except:
                                try:
                                    name = exc_type.__name__
                                except:
                                    try:
                                        name = str(exc_type)
                                    except:
                                        pass

                    if not description:
                        exc_desc = frames_list.exc_desc
                        if exc_desc is not None:
                            try:
                                description = str(exc_desc)
                            except:
                                pass

                    for frame_id, frame, method_name, original_filename, filename_in_utf8, lineno, _applied_mapping, show_as_current_frame, line_col_info in \
                        iter_visible_frames_info(dbg, frames_list):

                        line_text = linecache.getline(original_filename, lineno)

                        # Never filter out plugin frames!
                        if not getattr(frame, 'IS_PLUGIN_FRAME', False):
                            if dbg.is_files_filter_enabled and dbg.apply_files_filter(frame, original_filename, False):
                                continue

                        if show_as_current_frame:
                            current_paused_frame_name = method_name
                            method_name += ' (Current frame)'
                        frames.append((filename_in_utf8, lineno, method_name, line_text, line_col_info))

                    if not source_path and frames:
                        source_path = frames[0][0]

                    if IS_PY311_OR_GREATER:
                        stack_summary = traceback.StackSummary()
                        for filename_in_utf8, lineno, method_name, line_text, line_col_info in frames[-max_frames:]:
                            frame_summary = traceback.FrameSummary(filename_in_utf8, lineno, method_name, line=line_text)
                            if line_col_info is not None:
                                frame_summary.end_lineno = line_col_info.end_lineno
                                frame_summary.colno = line_col_info.colno
                                frame_summary.end_colno = line_col_info.end_colno
                            stack_summary.append(frame_summary)

                        stack_str = ''.join(stack_summary.format())

                    else:
                        # Note: remove col info (just used in 3.11).
                        stack_str = ''.join(traceback.format_list((x[:-1] for x in frames[-max_frames:])))

                    try:
                        stype = frames_list.exc_type.__qualname__
                        smod = frames_list.exc_type.__module__
                        if smod not in ("__main__", "builtins"):
                            if not isinstance(smod, str):
                                smod = "<unknown>"
                            stype = smod + '.' + stype
                    except Exception:
                        stype = '<unable to get exception type>'
                        pydev_log.exception('Error getting exception type.')

                    stack_str += '%s: %s\n' % (stype, frames_list.exc_desc)
                    stack_str += frames_list.exc_context_msg
                    stack_str_lst.append(stack_str)

                    frames_list = frames_list.chained_frames_list
                    if frames_list is None or not frames_list:
                        break

            except:
                pydev_log.exception('Error on build_exception_info_response.')
        finally:
            topmost_frame = None
    full_stack_str = ''.join(reversed(stack_str_lst))

    if not name:
        name = 'exception: type unknown'
    if not description:
        description = 'exception: no description'

    if current_paused_frame_name:
        name += '       (note: full exception trace is shown but execution is paused at: %s)' % (current_paused_frame_name,)

    if thread.stop_reason == CMD_STEP_CAUGHT_EXCEPTION:
        break_mode = pydevd_schema.ExceptionBreakMode.ALWAYS
    else:
        break_mode = pydevd_schema.ExceptionBreakMode.UNHANDLED

    response = pydevd_schema.ExceptionInfoResponse(
        request_seq=request_seq,
        success=True,
        command='exceptionInfo',
        body=pydevd_schema.ExceptionInfoResponseBody(
            exceptionId=name,
            description=description,
            breakMode=break_mode,
            details=pydevd_schema.ExceptionDetails(
                message=description,
                typeName=name,
                stackTrace=full_stack_str,
                source=source_path,
                # Note: ExceptionDetails actually accepts an 'innerException', but
                # when passing it, VSCode is not showing the stack trace at all.
            )
        )
    )
    return response


def internal_get_exception_details_json(dbg, request, thread_id, thread, max_frames, set_additional_thread_info=None, iter_visible_frames_info=None):
    ''' Fetch exception details
    '''
    try:
        response = build_exception_info_response(dbg, thread_id, thread, request.seq, set_additional_thread_info, iter_visible_frames_info, max_frames)
    except:
        exc = get_exception_traceback_str()
        response = pydevd_base_schema.build_response(request, kwargs={
            'success': False,
            'message': exc,
            'body':{}
        })
    dbg.writer.add_command(NetCommand(CMD_RETURN, 0, response, is_json=True))


class InternalGetBreakpointException(InternalThreadCommand):
    ''' Send details of exception raised while evaluating conditional breakpoint '''

    def __init__(self, thread_id, exc_type, stacktrace):
        self.sequence = 0
        self.thread_id = thread_id
        self.stacktrace = stacktrace
        self.exc_type = exc_type

    def do_it(self, dbg):
        try:
            callstack = "<xml>"

            makeValid = pydevd_xml.make_valid_xml_value

            for filename, line, methodname, methodobj in self.stacktrace:
                if not filesystem_encoding_is_utf8 and hasattr(filename, "decode"):
                    # filename is a byte string encoded using the file system encoding
                    # convert it to utf8
                    filename = filename.decode(file_system_encoding).encode("utf-8")

                callstack += '<frame thread_id = "%s" file="%s" line="%s" name="%s" obj="%s" />' \
                                    % (self.thread_id, makeValid(filename), line, makeValid(methodname), makeValid(methodobj))
            callstack += "</xml>"

            cmd = dbg.cmd_factory.make_send_breakpoint_exception_message(self.sequence, self.exc_type + "\t" + callstack)
            dbg.writer.add_command(cmd)
        except:
            exc = get_exception_traceback_str()
            sys.stderr.write('%s\n' % (exc,))
            cmd = dbg.cmd_factory.make_error_message(self.sequence, "Error Sending Exception: " + exc)
            dbg.writer.add_command(cmd)


class InternalSendCurrExceptionTrace(InternalThreadCommand):
    ''' Send details of the exception that was caught and where we've broken in.
    '''

    def __init__(self, thread_id, arg, curr_frame_id):
        '''
        :param arg: exception type, description, traceback object
        '''
        self.sequence = 0
        self.thread_id = thread_id
        self.curr_frame_id = curr_frame_id
        self.arg = arg

    def do_it(self, dbg):
        try:
            cmd = dbg.cmd_factory.make_send_curr_exception_trace_message(dbg, self.sequence, self.thread_id, self.curr_frame_id, *self.arg)
            del self.arg
            dbg.writer.add_command(cmd)
        except:
            exc = get_exception_traceback_str()
            sys.stderr.write('%s\n' % (exc,))
            cmd = dbg.cmd_factory.make_error_message(self.sequence, "Error Sending Current Exception Trace: " + exc)
            dbg.writer.add_command(cmd)


class InternalSendCurrExceptionTraceProceeded(InternalThreadCommand):
    ''' Send details of the exception that was caught and where we've broken in.
    '''

    def __init__(self, thread_id):
        self.sequence = 0
        self.thread_id = thread_id

    def do_it(self, dbg):
        try:
            cmd = dbg.cmd_factory.make_send_curr_exception_trace_proceeded_message(self.sequence, self.thread_id)
            dbg.writer.add_command(cmd)
        except:
            exc = get_exception_traceback_str()
            sys.stderr.write('%s\n' % (exc,))
            cmd = dbg.cmd_factory.make_error_message(self.sequence, "Error Sending Current Exception Trace Proceeded: " + exc)
            dbg.writer.add_command(cmd)


class InternalEvaluateConsoleExpression(InternalThreadCommand):
    ''' Execute the given command in the debug console '''

    def __init__(self, seq, thread_id, frame_id, line, buffer_output=True):
        self.sequence = seq
        self.thread_id = thread_id
        self.frame_id = frame_id
        self.line = line
        self.buffer_output = buffer_output

    def do_it(self, dbg):
        ''' Create an XML for console output, error and more (true/false)
        <xml>
            <output message=output_message></output>
            <error message=error_message></error>
            <more>true/false</more>
        </xml>
        '''
        try:
            frame = dbg.find_frame(self.thread_id, self.frame_id)
            if frame is not None:
                console_message = pydevd_console.execute_console_command(
                    frame, self.thread_id, self.frame_id, self.line, self.buffer_output)

                cmd = dbg.cmd_factory.make_send_console_message(self.sequence, console_message.to_xml())
            else:
                from _pydevd_bundle.pydevd_console import ConsoleMessage
                console_message = ConsoleMessage()
                console_message.add_console_message(
                    pydevd_console.CONSOLE_ERROR,
                    "Select the valid frame in the debug view (thread: %s, frame: %s invalid)" % (self.thread_id, self.frame_id),
                )
                cmd = dbg.cmd_factory.make_error_message(self.sequence, console_message.to_xml())
        except:
            exc = get_exception_traceback_str()
            cmd = dbg.cmd_factory.make_error_message(self.sequence, "Error evaluating expression " + exc)
        dbg.writer.add_command(cmd)


class InternalRunCustomOperation(InternalThreadCommand):
    ''' Run a custom command on an expression
    '''

    def __init__(self, seq, thread_id, frame_id, scope, attrs, style, encoded_code_or_file, fnname):
        self.sequence = seq
        self.thread_id = thread_id
        self.frame_id = frame_id
        self.scope = scope
        self.attrs = attrs
        self.style = style
        self.code_or_file = unquote_plus(encoded_code_or_file)
        self.fnname = fnname

    def do_it(self, dbg):
        try:
            res = pydevd_vars.custom_operation(dbg, self.thread_id, self.frame_id, self.scope, self.attrs,
                                              self.style, self.code_or_file, self.fnname)
            resEncoded = quote_plus(res)
            cmd = dbg.cmd_factory.make_custom_operation_message(self.sequence, resEncoded)
            dbg.writer.add_command(cmd)
        except:
            exc = get_exception_traceback_str()
            cmd = dbg.cmd_factory.make_error_message(self.sequence, "Error in running custom operation" + exc)
            dbg.writer.add_command(cmd)


class InternalConsoleGetCompletions(InternalThreadCommand):
    ''' Fetch the completions in the debug console
    '''

    def __init__(self, seq, thread_id, frame_id, act_tok):
        self.sequence = seq
        self.thread_id = thread_id
        self.frame_id = frame_id
        self.act_tok = act_tok

    def do_it(self, dbg):
        ''' Get completions and write back to the client
        '''
        try:
            frame = dbg.find_frame(self.thread_id, self.frame_id)
            completions_xml = pydevd_console.get_completions(frame, self.act_tok)
            cmd = dbg.cmd_factory.make_send_console_message(self.sequence, completions_xml)
            dbg.writer.add_command(cmd)
        except:
            exc = get_exception_traceback_str()
            cmd = dbg.cmd_factory.make_error_message(self.sequence, "Error in fetching completions" + exc)
            dbg.writer.add_command(cmd)


class InternalConsoleExec(InternalThreadCommand):
    ''' gets the value of a variable '''

    def __init__(self, seq, thread_id, frame_id, expression):
        self.sequence = seq
        self.thread_id = thread_id
        self.frame_id = frame_id
        self.expression = expression

    def do_it(self, dbg):
        ''' Converts request into python variable '''
        try:
            try:
                # don't trace new threads created by console command
                disable_trace_thread_modules()

                result = pydevconsole.console_exec(self.thread_id, self.frame_id, self.expression, dbg)
                xml = "<xml>"
                xml += pydevd_xml.var_to_xml(result, "")
                xml += "</xml>"
                cmd = dbg.cmd_factory.make_evaluate_expression_message(self.sequence, xml)
                dbg.writer.add_command(cmd)
            except:
                exc = get_exception_traceback_str()
                sys.stderr.write('%s\n' % (exc,))
                cmd = dbg.cmd_factory.make_error_message(self.sequence, "Error evaluating console expression " + exc)
                dbg.writer.add_command(cmd)
        finally:
            enable_trace_thread_modules()

            sys.stderr.flush()
            sys.stdout.flush()


class InternalLoadFullValue(InternalThreadCommand):
    '''
    Loads values asynchronously
    '''

    def __init__(self, seq, thread_id, frame_id, vars):
        self.sequence = seq
        self.thread_id = thread_id
        self.frame_id = frame_id
        self.vars = vars

    @silence_warnings_decorator
    def do_it(self, dbg):
        '''Starts a thread that will load values asynchronously'''
        try:
            var_objects = []
            for variable in self.vars:
                variable = variable.strip()
                if len(variable) > 0:
                    if '\t' in variable:  # there are attributes beyond scope
                        scope, attrs = variable.split('\t', 1)
                        name = attrs[0]
                    else:
                        scope, attrs = (variable, None)
                        name = scope
                    var_obj = pydevd_vars.getVariable(dbg, self.thread_id, self.frame_id, scope, attrs)
                    var_objects.append((var_obj, name))

            t = GetValueAsyncThreadDebug(dbg, dbg, self.sequence, var_objects)
            t.start()
        except:
            exc = get_exception_traceback_str()
            sys.stderr.write('%s\n' % (exc,))
            cmd = dbg.cmd_factory.make_error_message(self.sequence, "Error evaluating variable %s " % exc)
            dbg.writer.add_command(cmd)


class AbstractGetValueAsyncThread(PyDBDaemonThread):
    '''
    Abstract class for a thread, which evaluates values for async variables
    '''

    def __init__(self, py_db, frame_accessor, seq, var_objects):
        PyDBDaemonThread.__init__(self, py_db)
        self.frame_accessor = frame_accessor
        self.seq = seq
        self.var_objs = var_objects
        self.cancel_event = threading.Event()

    def send_result(self, xml):
        raise NotImplementedError()

    @overrides(PyDBDaemonThread._on_run)
    def _on_run(self):
        start = time.time()
        xml = StringIO()
        xml.write("<xml>")
        for (var_obj, name) in self.var_objs:
            current_time = time.time()
            if current_time - start > ASYNC_EVAL_TIMEOUT_SEC or self.cancel_event.is_set():
                break
            xml.write(pydevd_xml.var_to_xml(var_obj, name, evaluate_full_value=True))
        xml.write("</xml>")
        self.send_result(xml)
        xml.close()


class GetValueAsyncThreadDebug(AbstractGetValueAsyncThread):
    '''
    A thread for evaluation async values, which returns result for debugger
    Create message and send it via writer thread
    '''

    def send_result(self, xml):
        if self.frame_accessor is not None:
            cmd = self.frame_accessor.cmd_factory.make_load_full_value_message(self.seq, xml.getvalue())
            self.frame_accessor.writer.add_command(cmd)


class GetValueAsyncThreadConsole(AbstractGetValueAsyncThread):
    '''
    A thread for evaluation async values, which returns result for Console
    Send result directly to Console's server
    '''

    def send_result(self, xml):
        if self.frame_accessor is not None:
            self.frame_accessor.ReturnFullValue(self.seq, xml.getvalue())

