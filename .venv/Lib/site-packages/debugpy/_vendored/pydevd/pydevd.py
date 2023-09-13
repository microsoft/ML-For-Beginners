'''
Entry point module (keep at root):

This module starts the debugger.
'''
import sys  # @NoMove
if sys.version_info[:2] < (3, 6):
    raise RuntimeError('The PyDev.Debugger requires Python 3.6 onwards to be run. If you need to use an older Python version, use an older version of the debugger.')
import os

try:
    # Just empty packages to check if they're in the PYTHONPATH.
    import _pydev_bundle
except ImportError:
    # On the first import of a pydevd module, add pydevd itself to the PYTHONPATH
    # if its dependencies cannot be imported.
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import _pydev_bundle

# Import this first as it'll check for shadowed modules and will make sure that we import
# things as needed for gevent.
from _pydevd_bundle import pydevd_constants

import atexit
import dis
import io
from collections import defaultdict
from contextlib import contextmanager
from functools import partial
import itertools
import traceback
import weakref
import getpass as getpass_mod
import functools

import pydevd_file_utils
from _pydev_bundle import pydev_imports, pydev_log
from _pydev_bundle._pydev_filesystem_encoding import getfilesystemencoding
from _pydev_bundle.pydev_is_thread_alive import is_thread_alive
from _pydev_bundle.pydev_override import overrides
from _pydev_bundle._pydev_saved_modules import threading, time, thread
from _pydevd_bundle import pydevd_extension_utils, pydevd_frame_utils
from _pydevd_bundle.pydevd_filtering import FilesFiltering, glob_matches_path
from _pydevd_bundle import pydevd_io, pydevd_vm_type, pydevd_defaults
from _pydevd_bundle import pydevd_utils
from _pydevd_bundle import pydevd_runpy
from _pydev_bundle.pydev_console_utils import DebugConsoleStdIn
from _pydevd_bundle.pydevd_additional_thread_info import set_additional_thread_info
from _pydevd_bundle.pydevd_breakpoints import ExceptionBreakpoint, get_exception_breakpoint
from _pydevd_bundle.pydevd_comm_constants import (CMD_THREAD_SUSPEND, CMD_STEP_INTO, CMD_SET_BREAK,
    CMD_STEP_INTO_MY_CODE, CMD_STEP_OVER, CMD_SMART_STEP_INTO, CMD_RUN_TO_LINE,
    CMD_SET_NEXT_STATEMENT, CMD_STEP_RETURN, CMD_ADD_EXCEPTION_BREAK, CMD_STEP_RETURN_MY_CODE,
    CMD_STEP_OVER_MY_CODE, constant_to_str, CMD_STEP_INTO_COROUTINE)
from _pydevd_bundle.pydevd_constants import (get_thread_id, get_current_thread_id,
    DebugInfoHolder, PYTHON_SUSPEND, STATE_SUSPEND, STATE_RUN, get_frame,
    clear_cached_thread_id, INTERACTIVE_MODE_AVAILABLE, SHOW_DEBUG_INFO_ENV, NULL,
    NO_FTRACE, IS_IRONPYTHON, JSON_PROTOCOL, IS_CPYTHON, HTTP_JSON_PROTOCOL, USE_CUSTOM_SYS_CURRENT_FRAMES_MAP, call_only_once,
    ForkSafeLock, IGNORE_BASENAMES_STARTING_WITH, EXCEPTION_TYPE_UNHANDLED, SUPPORT_GEVENT,
    PYDEVD_IPYTHON_COMPATIBLE_DEBUGGING, PYDEVD_IPYTHON_CONTEXT)
from _pydevd_bundle.pydevd_defaults import PydevdCustomization  # Note: import alias used on pydev_monkey.
from _pydevd_bundle.pydevd_custom_frames import CustomFramesContainer, custom_frames_container_init
from _pydevd_bundle.pydevd_dont_trace_files import DONT_TRACE, PYDEV_FILE, LIB_FILE, DONT_TRACE_DIRS
from _pydevd_bundle.pydevd_extension_api import DebuggerEventHandler
from _pydevd_bundle.pydevd_frame_utils import add_exception_to_frame, remove_exception_from_frame
from _pydevd_bundle.pydevd_net_command_factory_xml import NetCommandFactory
from _pydevd_bundle.pydevd_trace_dispatch import (
    trace_dispatch as _trace_dispatch, global_cache_skips, global_cache_frame_skips, fix_top_level_trace_and_get_trace_func, USING_CYTHON)
from _pydevd_bundle.pydevd_utils import save_main_module, is_current_thread_main_thread, \
    import_attr_from_module
from _pydevd_frame_eval.pydevd_frame_eval_main import (
    frame_eval_func, dummy_trace_dispatch, USING_FRAME_EVAL)
import pydev_ipython  # @UnusedImport
from _pydevd_bundle.pydevd_source_mapping import SourceMapping
from _pydevd_bundle.pydevd_concurrency_analyser.pydevd_concurrency_logger import ThreadingLogger, AsyncioLogger, send_concurrency_message, cur_time
from _pydevd_bundle.pydevd_concurrency_analyser.pydevd_thread_wrappers import wrap_threads
from pydevd_file_utils import get_abs_path_real_path_and_base_from_frame, NORM_PATHS_AND_BASE_CONTAINER
from pydevd_file_utils import get_fullname, get_package_dir
from os.path import abspath as os_path_abspath
import pydevd_tracing
from _pydevd_bundle.pydevd_comm import (InternalThreadCommand, InternalThreadCommandForAnyThread,
    create_server_socket, FSNotifyThread)
from _pydevd_bundle.pydevd_comm import(InternalConsoleExec,
    _queue, ReaderThread, GetGlobalDebugger, get_global_debugger,
    set_global_debugger, WriterThread,
    start_client, start_server, InternalGetBreakpointException, InternalSendCurrExceptionTrace,
    InternalSendCurrExceptionTraceProceeded)
from _pydevd_bundle.pydevd_daemon_thread import PyDBDaemonThread, mark_as_pydevd_daemon_thread
from _pydevd_bundle.pydevd_process_net_command_json import PyDevJsonCommandProcessor
from _pydevd_bundle.pydevd_process_net_command import process_net_command
from _pydevd_bundle.pydevd_net_command import NetCommand, NULL_NET_COMMAND

from _pydevd_bundle.pydevd_breakpoints import stop_on_unhandled_exception
from _pydevd_bundle.pydevd_collect_bytecode_info import collect_try_except_info, collect_return_info, collect_try_except_info_from_source
from _pydevd_bundle.pydevd_suspended_frames import SuspendedFramesManager
from socket import SHUT_RDWR
from _pydevd_bundle.pydevd_api import PyDevdAPI
from _pydevd_bundle.pydevd_timeout import TimeoutTracker
from _pydevd_bundle.pydevd_thread_lifecycle import suspend_all_threads, mark_thread_suspended

pydevd_gevent_integration = None

if SUPPORT_GEVENT:
    try:
        from _pydevd_bundle import pydevd_gevent_integration
    except:
        pydev_log.exception(
            'pydevd: GEVENT_SUPPORT is set but gevent is not available in the environment.\n'
            'Please unset GEVENT_SUPPORT from the environment variables or install gevent.')
    else:
        pydevd_gevent_integration.log_gevent_debug_info()

if USE_CUSTOM_SYS_CURRENT_FRAMES_MAP:
    from _pydevd_bundle.pydevd_constants import constructed_tid_to_last_frame

__version_info__ = (2, 9, 5)
__version_info_str__ = []
for v in __version_info__:
    __version_info_str__.append(str(v))

__version__ = '.'.join(__version_info_str__)

# IMPORTANT: pydevd_constants must be the 1st thing defined because it'll keep a reference to the original sys._getframe


def install_breakpointhook(pydevd_breakpointhook=None):
    if pydevd_breakpointhook is None:

        def pydevd_breakpointhook(*args, **kwargs):
            hookname = os.getenv('PYTHONBREAKPOINT')
            if (
                   hookname is not None
                   and len(hookname) > 0
                   and hasattr(sys, '__breakpointhook__')
                   and sys.__breakpointhook__ != pydevd_breakpointhook
                ):
                sys.__breakpointhook__(*args, **kwargs)
            else:
                settrace(*args, **kwargs)

    if sys.version_info[0:2] >= (3, 7):
        # There are some choices on how to provide the breakpoint hook. Namely, we can provide a
        # PYTHONBREAKPOINT which provides the import path for a method to be executed or we
        # can override sys.breakpointhook.
        # pydevd overrides sys.breakpointhook instead of providing an environment variable because
        # it's possible that the debugger starts the user program but is not available in the
        # PYTHONPATH (and would thus fail to be imported if PYTHONBREAKPOINT was set to pydevd.settrace).
        # Note that the implementation still takes PYTHONBREAKPOINT in account (so, if it was provided
        # by someone else, it'd still work).
        sys.breakpointhook = pydevd_breakpointhook
    else:
        if sys.version_info[0] >= 3:
            import builtins as __builtin__  # Py3 noqa
        else:
            import __builtin__  # noqa

        # In older versions, breakpoint() isn't really available, so, install the hook directly
        # in the builtins.
        __builtin__.breakpoint = pydevd_breakpointhook
        sys.__breakpointhook__ = pydevd_breakpointhook


# Install the breakpoint hook at import time.
install_breakpointhook()

from _pydevd_bundle.pydevd_plugin_utils import PluginManager

threadingEnumerate = threading.enumerate
threadingCurrentThread = threading.current_thread

try:
    'dummy'.encode('utf-8')  # Added because otherwise Jython 2.2.1 wasn't finding the encoding (if it wasn't loaded in the main thread).
except:
    pass

_global_redirect_stdout_to_server = False
_global_redirect_stderr_to_server = False

file_system_encoding = getfilesystemencoding()

_CACHE_FILE_TYPE = {}

pydev_log.debug('Using GEVENT_SUPPORT: %s', pydevd_constants.SUPPORT_GEVENT)
pydev_log.debug('Using GEVENT_SHOW_PAUSED_GREENLETS: %s', pydevd_constants.GEVENT_SHOW_PAUSED_GREENLETS)
pydev_log.debug('pydevd __file__: %s', os.path.abspath(__file__))
pydev_log.debug('Using PYDEVD_IPYTHON_COMPATIBLE_DEBUGGING: %s', pydevd_constants.PYDEVD_IPYTHON_COMPATIBLE_DEBUGGING)
if pydevd_constants.PYDEVD_IPYTHON_COMPATIBLE_DEBUGGING:
    pydev_log.debug('PYDEVD_IPYTHON_CONTEXT: %s', pydevd_constants.PYDEVD_IPYTHON_CONTEXT)


#=======================================================================================================================
# PyDBCommandThread
#=======================================================================================================================
class PyDBCommandThread(PyDBDaemonThread):

    def __init__(self, py_db):
        PyDBDaemonThread.__init__(self, py_db)
        self._py_db_command_thread_event = py_db._py_db_command_thread_event
        self.name = 'pydevd.CommandThread'

    @overrides(PyDBDaemonThread._on_run)
    def _on_run(self):
        # Delay a bit this initialization to wait for the main program to start.
        self._py_db_command_thread_event.wait(0.3)

        if self._kill_received:
            return

        try:
            while not self._kill_received:
                try:
                    self.py_db.process_internal_commands()
                except:
                    pydev_log.info('Finishing debug communication...(2)')
                self._py_db_command_thread_event.clear()
                self._py_db_command_thread_event.wait(0.3)
        except:
            try:
                pydev_log.debug(sys.exc_info()[0])
            except:
                # In interpreter shutdown many things can go wrong (any module variables may
                # be None, streams can be closed, etc).
                pass

            # only got this error in interpreter shutdown
            # pydev_log.info('Finishing debug communication...(3)')

    @overrides(PyDBDaemonThread.do_kill_pydev_thread)
    def do_kill_pydev_thread(self):
        PyDBDaemonThread.do_kill_pydev_thread(self)
        # Set flag so that it can exit before the usual timeout.
        self._py_db_command_thread_event.set()


#=======================================================================================================================
# CheckAliveThread
# Non-daemon thread: guarantees that all data is written even if program is finished
#=======================================================================================================================
class CheckAliveThread(PyDBDaemonThread):

    def __init__(self, py_db):
        PyDBDaemonThread.__init__(self, py_db)
        self.name = 'pydevd.CheckAliveThread'
        self.daemon = False
        self._wait_event = threading.Event()

    @overrides(PyDBDaemonThread._on_run)
    def _on_run(self):
        py_db = self.py_db

        def can_exit():
            with py_db._main_lock:
                # Note: it's important to get the lock besides checking that it's empty (this
                # means that we're not in the middle of some command processing).
                writer = py_db.writer
                writer_empty = writer is not None and writer.empty()

            return not py_db.has_user_threads_alive() and writer_empty

        try:
            while not self._kill_received:
                self._wait_event.wait(0.3)
                if can_exit():
                    break

                py_db.check_output_redirect()

            if can_exit():
                pydev_log.debug("No threads alive, finishing debug session")
                py_db.dispose_and_kill_all_pydevd_threads()
        except:
            pydev_log.exception()

    def join(self, timeout=None):
        # If someone tries to join this thread, mark it to be killed.
        # This is the case for CherryPy when auto-reload is turned on.
        self.do_kill_pydev_thread()
        PyDBDaemonThread.join(self, timeout=timeout)

    @overrides(PyDBDaemonThread.do_kill_pydev_thread)
    def do_kill_pydev_thread(self):
        PyDBDaemonThread.do_kill_pydev_thread(self)
        # Set flag so that it can exit before the usual timeout.
        self._wait_event.set()


class AbstractSingleNotificationBehavior(object):
    '''
    The basic usage should be:

    # Increment the request time for the suspend.
    single_notification_behavior.increment_suspend_time()

    # Notify that this is a pause request (when a pause, not a breakpoint).
    single_notification_behavior.on_pause()

    # Mark threads to be suspended.
    set_suspend(...)

    # On do_wait_suspend, use notify_thread_suspended:
    def do_wait_suspend(...):
        with single_notification_behavior.notify_thread_suspended(thread_id, thread, reason):
            ...
    '''

    __slots__ = [
        '_last_resume_notification_time',
        '_last_suspend_notification_time',
        '_lock',
        '_next_request_time',
        '_suspend_time_request',
        '_suspended_thread_id_to_thread',
        '_pause_requested',
        '_py_db',
    ]

    NOTIFY_OF_PAUSE_TIMEOUT = .5

    def __init__(self, py_db):
        self._py_db = weakref.ref(py_db)
        self._next_request_time = partial(next, itertools.count())
        self._last_suspend_notification_time = -1
        self._last_resume_notification_time = -1
        self._suspend_time_request = self._next_request_time()
        self._lock = thread.allocate_lock()
        self._suspended_thread_id_to_thread = {}
        self._pause_requested = False

    def send_suspend_notification(self, thread_id, thread, stop_reason):
        raise AssertionError('abstract: subclasses must override.')

    def send_resume_notification(self, thread_id):
        raise AssertionError('abstract: subclasses must override.')

    def increment_suspend_time(self):
        with self._lock:
            self._suspend_time_request = self._next_request_time()

    def on_pause(self):
        # Upon a pause, we should force sending new suspend notifications
        # if no notification is sent after some time and there's some thread already stopped.
        with self._lock:
            self._pause_requested = True
            global_suspend_time = self._suspend_time_request
        py_db = self._py_db()
        if py_db is not None:
            py_db.timeout_tracker.call_on_timeout(
                self.NOTIFY_OF_PAUSE_TIMEOUT,
                self._notify_after_timeout,
                kwargs={'global_suspend_time': global_suspend_time}
            )

    def _notify_after_timeout(self, global_suspend_time):
        with self._lock:
            if self._suspended_thread_id_to_thread:
                if global_suspend_time > self._last_suspend_notification_time:
                    self._last_suspend_notification_time = global_suspend_time
                    # Notify about any thread which is currently suspended.
                    pydev_log.info('Sending suspend notification after timeout.')
                    thread_id, thread = next(iter(self._suspended_thread_id_to_thread.items()))
                    self.send_suspend_notification(thread_id, thread, CMD_THREAD_SUSPEND)

    def on_thread_suspend(self, thread_id, thread, stop_reason):
        with self._lock:
            pause_requested = self._pause_requested
            if pause_requested:
                # When a suspend notification is sent, reset the pause flag.
                self._pause_requested = False

            self._suspended_thread_id_to_thread[thread_id] = thread

            # CMD_THREAD_SUSPEND should always be a side-effect of a break, so, only
            # issue for a CMD_THREAD_SUSPEND if a pause is pending.
            if stop_reason != CMD_THREAD_SUSPEND or pause_requested:
                if self._suspend_time_request > self._last_suspend_notification_time:
                    pydev_log.info('Sending suspend notification.')
                    self._last_suspend_notification_time = self._suspend_time_request
                    self.send_suspend_notification(thread_id, thread, stop_reason)
                else:
                    pydev_log.info(
                        'Suspend not sent (it was already sent). Last suspend % <= Last resume %s',
                        self._last_suspend_notification_time,
                        self._last_resume_notification_time,
                    )
            else:
                pydev_log.info(
                    'Suspend not sent because stop reason is thread suspend and pause was not requested.',
                )

    def on_thread_resume(self, thread_id, thread):
        # on resume (step, continue all):
        with self._lock:
            self._suspended_thread_id_to_thread.pop(thread_id)
            if self._last_resume_notification_time < self._last_suspend_notification_time:
                pydev_log.info('Sending resume notification.')
                self._last_resume_notification_time = self._last_suspend_notification_time
                self.send_resume_notification(thread_id)
            else:
                pydev_log.info(
                    'Resume not sent (it was already sent). Last resume %s >= Last suspend %s',
                    self._last_resume_notification_time,
                    self._last_suspend_notification_time,
                )

    @contextmanager
    def notify_thread_suspended(self, thread_id, thread, stop_reason):
        self.on_thread_suspend(thread_id, thread, stop_reason)
        try:
            yield  # At this point the thread must be actually suspended.
        finally:
            self.on_thread_resume(thread_id, thread)


class ThreadsSuspendedSingleNotification(AbstractSingleNotificationBehavior):

    __slots__ = AbstractSingleNotificationBehavior.__slots__ + [
        'multi_threads_single_notification', '_callbacks', '_callbacks_lock']

    def __init__(self, py_db):
        AbstractSingleNotificationBehavior.__init__(self, py_db)
        # If True, pydevd will send a single notification when all threads are suspended/resumed.
        self.multi_threads_single_notification = False
        self._callbacks_lock = threading.Lock()
        self._callbacks = []

    def add_on_resumed_callback(self, callback):
        with self._callbacks_lock:
            self._callbacks.append(callback)

    @overrides(AbstractSingleNotificationBehavior.send_resume_notification)
    def send_resume_notification(self, thread_id):
        py_db = self._py_db()
        if py_db is not None:
            py_db.writer.add_command(py_db.cmd_factory.make_thread_resume_single_notification(thread_id))

            with self._callbacks_lock:
                callbacks = self._callbacks
                self._callbacks = []

            for callback in callbacks:
                callback()

    @overrides(AbstractSingleNotificationBehavior.send_suspend_notification)
    def send_suspend_notification(self, thread_id, thread, stop_reason):
        py_db = self._py_db()
        if py_db is not None:
            py_db.writer.add_command(
                py_db.cmd_factory.make_thread_suspend_single_notification(
                    py_db, thread_id, thread, stop_reason))

    @overrides(AbstractSingleNotificationBehavior.notify_thread_suspended)
    @contextmanager
    def notify_thread_suspended(self, thread_id, thread, stop_reason):
        if self.multi_threads_single_notification:
            with AbstractSingleNotificationBehavior.notify_thread_suspended(self, thread_id, thread, stop_reason):
                yield
        else:
            yield


class _Authentication(object):

    __slots__ = ['access_token', 'client_access_token', '_authenticated', '_wrong_attempts']

    def __init__(self):
        # A token to be send in the command line or through the settrace api -- when such token
        # is given, the first message sent to the IDE must pass the same token to authenticate.
        # Note that if a disconnect is sent, the same message must be resent to authenticate.
        self.access_token = None

        # This token is the one that the client requires to accept a connection from pydevd
        # (it's stored here and just passed back when required, it's not used internally
        # for anything else).
        self.client_access_token = None

        self._authenticated = None

        self._wrong_attempts = 0

    def is_authenticated(self):
        if self._authenticated is None:
            return self.access_token is None
        return self._authenticated

    def login(self, access_token):
        if self._wrong_attempts >= 10:  # A user can fail to authenticate at most 10 times.
            return

        self._authenticated = access_token == self.access_token
        if not self._authenticated:
            self._wrong_attempts += 1
        else:
            self._wrong_attempts = 0

    def logout(self):
        self._authenticated = None
        self._wrong_attempts = 0


class PyDB(object):
    """ Main debugging class
    Lots of stuff going on here:

    PyDB starts two threads on startup that connect to remote debugger (RDB)
    The threads continuously read & write commands to RDB.
    PyDB communicates with these threads through command queues.
       Every RDB command is processed by calling process_net_command.
       Every PyDB net command is sent to the net by posting NetCommand to WriterThread queue

       Some commands need to be executed on the right thread (suspend/resume & friends)
       These are placed on the internal command queue.
    """

    # Direct child pids which should not be terminated when terminating processes.
    # Note: class instance because it should outlive PyDB instances.
    dont_terminate_child_pids = set()

    def __init__(self, set_as_global=True):
        if set_as_global:
            pydevd_tracing.replace_sys_set_trace_func()

        self.authentication = _Authentication()

        self.reader = None
        self.writer = None
        self._fsnotify_thread = None
        self.created_pydb_daemon_threads = {}
        self._waiting_for_connection_thread = None
        self._on_configuration_done_event = threading.Event()
        self.check_alive_thread = None
        self.py_db_command_thread = None
        self.quitting = None
        self.cmd_factory = NetCommandFactory()
        self._cmd_queue = defaultdict(_queue.Queue)  # Key is thread id or '*', value is Queue
        self.suspended_frames_manager = SuspendedFramesManager()
        self._files_filtering = FilesFiltering()
        self.timeout_tracker = TimeoutTracker(self)

        # Note: when the source mapping is changed we also have to clear the file types cache
        # (because if a given file is a part of the project or not may depend on it being
        # defined in the source mapping).
        self.source_mapping = SourceMapping(on_source_mapping_changed=self._clear_filters_caches)

        # Determines whether we should terminate child processes when asked to terminate.
        self.terminate_child_processes = True

        # Determines whether we should try to do a soft terminate (i.e.: interrupt the main
        # thread with a KeyboardInterrupt).
        self.terminate_keyboard_interrupt = False

        # Set to True after a keyboard interrupt is requested the first time.
        self.keyboard_interrupt_requested = False

        # These are the breakpoints received by the PyDevdAPI. They are meant to store
        # the breakpoints in the api -- its actual contents are managed by the api.
        self.api_received_breakpoints = {}

        # These are the breakpoints meant to be consumed during runtime.
        self.breakpoints = {}
        self.function_breakpoint_name_to_breakpoint = {}

        # Set communication protocol
        PyDevdAPI().set_protocol(self, 0, PydevdCustomization.DEFAULT_PROTOCOL)

        self.variable_presentation = PyDevdAPI.VariablePresentation()

        # mtime to be raised when breakpoints change
        self.mtime = 0

        self.file_to_id_to_line_breakpoint = {}
        self.file_to_id_to_plugin_breakpoint = {}

        # Note: breakpoints dict should not be mutated: a copy should be created
        # and later it should be assigned back (to prevent concurrency issues).
        self.break_on_uncaught_exceptions = {}
        self.break_on_caught_exceptions = {}
        self.break_on_user_uncaught_exceptions = {}

        self.ready_to_run = False
        self._main_lock = thread.allocate_lock()
        self._lock_running_thread_ids = thread.allocate_lock()
        self._lock_create_fs_notify = thread.allocate_lock()
        self._py_db_command_thread_event = threading.Event()
        if set_as_global:
            CustomFramesContainer._py_db_command_thread_event = self._py_db_command_thread_event

        self.pydb_disposed = False
        self._wait_for_threads_to_finish_called = False
        self._wait_for_threads_to_finish_called_lock = thread.allocate_lock()
        self._wait_for_threads_to_finish_called_event = threading.Event()

        self.terminate_requested = False
        self._disposed_lock = thread.allocate_lock()
        self.signature_factory = None
        self.SetTrace = pydevd_tracing.SetTrace
        self.skip_on_exceptions_thrown_in_same_context = False
        self.ignore_exceptions_thrown_in_lines_with_ignore_exception = True

        # Suspend debugger even if breakpoint condition raises an exception.
        # May be changed with CMD_PYDEVD_JSON_CONFIG.
        self.skip_suspend_on_breakpoint_exception = ()  # By default suspend on any Exception.
        self.skip_print_breakpoint_exception = ()  # By default print on any Exception.

        # By default user can step into properties getter/setter/deleter methods
        self.disable_property_trace = False
        self.disable_property_getter_trace = False
        self.disable_property_setter_trace = False
        self.disable_property_deleter_trace = False

        # this is a dict of thread ids pointing to thread ids. Whenever a command is passed to the java end that
        # acknowledges that a thread was created, the thread id should be passed here -- and if at some time we do not
        # find that thread alive anymore, we must remove it from this list and make the java side know that the thread
        # was killed.
        self._running_thread_ids = {}
        # Note: also access '_enable_thread_notifications' with '_lock_running_thread_ids'
        self._enable_thread_notifications = False

        self._set_breakpoints_with_id = False

        # This attribute holds the file-> lines which have an @IgnoreException.
        self.filename_to_lines_where_exceptions_are_ignored = {}

        # working with plugins (lazily initialized)
        self.plugin = None
        self.has_plugin_line_breaks = False
        self.has_plugin_exception_breaks = False
        self.thread_analyser = None
        self.asyncio_analyser = None

        # The GUI event loop that's going to run.
        # Possible values:
        # matplotlib - Whatever GUI backend matplotlib is using.
        # 'wx'/'qt'/'none'/... - GUI toolkits that have bulitin support. See pydevd_ipython/inputhook.py:24.
        # Other - A custom function that'll be imported and run.
        self._gui_event_loop = 'matplotlib'
        self._installed_gui_support = False
        self.gui_in_use = False

        # GUI event loop support in debugger
        self.activate_gui_function = None

        # matplotlib support in debugger and debug console
        self.mpl_hooks_in_debug_console = False
        self.mpl_modules_for_patching = {}

        self._filename_to_not_in_scope = {}
        self.first_breakpoint_reached = False
        self._exclude_filters_enabled = self._files_filtering.use_exclude_filters()
        self._is_libraries_filter_enabled = self._files_filtering.use_libraries_filter()
        self.is_files_filter_enabled = self._exclude_filters_enabled or self._is_libraries_filter_enabled
        self.show_return_values = False
        self.remove_return_values_flag = False
        self.redirect_output = False
        # Note that besides the `redirect_output` flag, we also need to consider that someone
        # else is already redirecting (i.e.: debugpy).
        self.is_output_redirected = False

        # this flag disables frame evaluation even if it's available
        self.use_frame_eval = True

        # If True, pydevd will send a single notification when all threads are suspended/resumed.
        self._threads_suspended_single_notification = ThreadsSuspendedSingleNotification(self)

        # If True a step command will do a step in one thread and will also resume all other threads.
        self.stepping_resumes_all_threads = False

        self._local_thread_trace_func = threading.local()

        self._server_socket_ready_event = threading.Event()
        self._server_socket_name = None

        # Bind many locals to the debugger because upon teardown those names may become None
        # in the namespace (and thus can't be relied upon unless the reference was previously
        # saved).
        if IS_IRONPYTHON:

            # A partial() cannot be used in IronPython for sys.settrace.
            def new_trace_dispatch(frame, event, arg):
                return _trace_dispatch(self, frame, event, arg)

            self.trace_dispatch = new_trace_dispatch
        else:
            self.trace_dispatch = partial(_trace_dispatch, self)
        self.fix_top_level_trace_and_get_trace_func = fix_top_level_trace_and_get_trace_func
        self.frame_eval_func = frame_eval_func
        self.dummy_trace_dispatch = dummy_trace_dispatch

        # Note: this is different from pydevd_constants.thread_get_ident because we want Jython
        # to be None here because it also doesn't have threading._active.
        try:
            self.threading_get_ident = threading.get_ident  # Python 3
            self.threading_active = threading._active
        except:
            try:
                self.threading_get_ident = threading._get_ident  # Python 2 noqa
                self.threading_active = threading._active
            except:
                self.threading_get_ident = None  # Jython
                self.threading_active = None
        self.threading_current_thread = threading.currentThread
        self.set_additional_thread_info = set_additional_thread_info
        self.stop_on_unhandled_exception = stop_on_unhandled_exception
        self.collect_return_info = collect_return_info
        self.get_exception_breakpoint = get_exception_breakpoint
        self._dont_trace_get_file_type = DONT_TRACE.get
        self._dont_trace_dirs_get_file_type = DONT_TRACE_DIRS.get
        self.PYDEV_FILE = PYDEV_FILE
        self.LIB_FILE = LIB_FILE

        self._in_project_scope_cache = {}
        self._exclude_by_filter_cache = {}
        self._apply_filter_cache = {}
        self._ignore_system_exit_codes = set()

        # DAP related
        self._dap_messages_listeners = []

        if set_as_global:
            # Set as the global instance only after it's initialized.
            set_global_debugger(self)

        pydevd_defaults.on_pydb_init(self)
        # Stop the tracing as the last thing before the actual shutdown for a clean exit.
        atexit.register(stoptrace)

    def collect_try_except_info(self, code_obj):
        filename = code_obj.co_filename
        try:
            if os.path.exists(filename):
                pydev_log.debug('Collecting try..except info from source for %s', filename)
                try_except_infos = collect_try_except_info_from_source(filename)
                if try_except_infos:
                    # Filter for the current function
                    max_line = -1
                    min_line = sys.maxsize
                    for _, line in dis.findlinestarts(code_obj):

                        if line > max_line:
                            max_line = line

                        if line < min_line:
                            min_line = line

                    try_except_infos = [x for x in try_except_infos if min_line <= x.try_line <= max_line]
                return try_except_infos

        except:
            pydev_log.exception('Error collecting try..except info from source (%s)', filename)

        pydev_log.debug('Collecting try..except info from bytecode for %s', filename)
        return collect_try_except_info(code_obj)

    def setup_auto_reload_watcher(self, enable_auto_reload, watch_dirs, poll_target_time, exclude_patterns, include_patterns):
        try:
            with self._lock_create_fs_notify:

                # When setting up, dispose of the previous one (if any).
                if self._fsnotify_thread is not None:
                    self._fsnotify_thread.do_kill_pydev_thread()
                    self._fsnotify_thread = None

                if not enable_auto_reload:
                    return

                exclude_patterns = tuple(exclude_patterns)
                include_patterns = tuple(include_patterns)

                def accept_directory(absolute_filename, cache={}):
                    try:
                        return cache[absolute_filename]
                    except:
                        if absolute_filename and absolute_filename[-1] not in ('/', '\\'):
                            # I.e.: for directories we always end with '/' or '\\' so that
                            # we match exclusions such as "**/node_modules/**"
                            absolute_filename += os.path.sep

                        # First include what we want
                        for include_pattern in include_patterns:
                            if glob_matches_path(absolute_filename, include_pattern):
                                cache[absolute_filename] = True
                                return True

                        # Then exclude what we don't want
                        for exclude_pattern in exclude_patterns:
                            if glob_matches_path(absolute_filename, exclude_pattern):
                                cache[absolute_filename] = False
                                return False

                        # By default track all directories not excluded.
                        cache[absolute_filename] = True
                        return True

                def accept_file(absolute_filename, cache={}):
                    try:
                        return cache[absolute_filename]
                    except:
                        # First include what we want
                        for include_pattern in include_patterns:
                            if glob_matches_path(absolute_filename, include_pattern):
                                cache[absolute_filename] = True
                                return True

                        # Then exclude what we don't want
                        for exclude_pattern in exclude_patterns:
                            if glob_matches_path(absolute_filename, exclude_pattern):
                                cache[absolute_filename] = False
                                return False

                        # By default don't track files not included.
                        cache[absolute_filename] = False
                        return False

                self._fsnotify_thread = FSNotifyThread(self, PyDevdAPI(), watch_dirs)
                watcher = self._fsnotify_thread.watcher
                watcher.accept_directory = accept_directory
                watcher.accept_file = accept_file

                watcher.target_time_for_single_scan = poll_target_time
                watcher.target_time_for_notification = poll_target_time
                self._fsnotify_thread.start()
        except:
            pydev_log.exception('Error setting up auto-reload.')

    def get_arg_ppid(self):
        try:
            setup = SetupHolder.setup
            if setup:
                return int(setup.get('ppid', 0))
        except:
            pydev_log.exception('Error getting ppid.')

        return 0

    def wait_for_ready_to_run(self):
        while not self.ready_to_run:
            # busy wait until we receive run command
            self.process_internal_commands()
            self._py_db_command_thread_event.clear()
            self._py_db_command_thread_event.wait(0.1)

    def on_initialize(self):
        '''
        Note: only called when using the DAP (Debug Adapter Protocol).
        '''
        self._on_configuration_done_event.clear()

    def on_configuration_done(self):
        '''
        Note: only called when using the DAP (Debug Adapter Protocol).
        '''
        self._on_configuration_done_event.set()
        self._py_db_command_thread_event.set()

    def is_attached(self):
        return self._on_configuration_done_event.is_set()

    def on_disconnect(self):
        '''
        Note: only called when using the DAP (Debug Adapter Protocol).
        '''
        self.authentication.logout()
        self._on_configuration_done_event.clear()

    def set_ignore_system_exit_codes(self, ignore_system_exit_codes):
        assert isinstance(ignore_system_exit_codes, (list, tuple, set))
        self._ignore_system_exit_codes = set(ignore_system_exit_codes)

    def ignore_system_exit_code(self, system_exit_exc):
        if hasattr(system_exit_exc, 'code'):
            return system_exit_exc.code in self._ignore_system_exit_codes
        else:
            return system_exit_exc in self._ignore_system_exit_codes

    def block_until_configuration_done(self, cancel=None):
        if cancel is None:
            cancel = NULL

        while not cancel.is_set():
            if self._on_configuration_done_event.is_set():
                cancel.set()  # Set cancel to prevent reuse
                return

            self.process_internal_commands()
            self._py_db_command_thread_event.clear()
            self._py_db_command_thread_event.wait(1 / 15.)

    def add_fake_frame(self, thread_id, frame_id, frame):
        self.suspended_frames_manager.add_fake_frame(thread_id, frame_id, frame)

    def handle_breakpoint_condition(self, info, pybreakpoint, new_frame):
        condition = pybreakpoint.condition
        try:
            if pybreakpoint.handle_hit_condition(new_frame):
                return True

            if not condition:
                return False

            return eval(condition, new_frame.f_globals, new_frame.f_locals)
        except Exception as e:
            if not isinstance(e, self.skip_print_breakpoint_exception):
                stack_trace = io.StringIO()
                etype, value, tb = sys.exc_info()
                traceback.print_exception(etype, value, tb.tb_next, file=stack_trace)

                msg = 'Error while evaluating expression in conditional breakpoint: %s\n%s' % (
                    condition, stack_trace.getvalue())
                api = PyDevdAPI()
                api.send_error_message(self, msg)

            if not isinstance(e, self.skip_suspend_on_breakpoint_exception):
                try:
                    # add exception_type and stacktrace into thread additional info
                    etype, value, tb = sys.exc_info()
                    error = ''.join(traceback.format_exception_only(etype, value))
                    stack = traceback.extract_stack(f=tb.tb_frame.f_back)

                    # On self.set_suspend(thread, CMD_SET_BREAK) this info will be
                    # sent to the client.
                    info.conditional_breakpoint_exception = \
                        ('Condition:\n' + condition + '\n\nError:\n' + error, stack)
                except:
                    pydev_log.exception()
                return True

            return False

        finally:
            etype, value, tb = None, None, None

    def handle_breakpoint_expression(self, pybreakpoint, info, new_frame):
        try:
            try:
                val = eval(pybreakpoint.expression, new_frame.f_globals, new_frame.f_locals)
            except:
                val = sys.exc_info()[1]
        finally:
            if val is not None:
                info.pydev_message = str(val)

    def _internal_get_file_type(self, abs_real_path_and_basename):
        basename = abs_real_path_and_basename[-1]
        if (
                basename.startswith(IGNORE_BASENAMES_STARTING_WITH) or
                abs_real_path_and_basename[0].startswith(IGNORE_BASENAMES_STARTING_WITH)
            ):
            # Note: these are the files that are completely ignored (they aren't shown to the user
            # as user nor library code as it's usually just noise in the frame stack).
            return self.PYDEV_FILE
        file_type = self._dont_trace_get_file_type(basename)
        if file_type is not None:
            return file_type

        if basename.startswith('__init__.py'):
            # i.e.: ignore the __init__ files inside pydevd (the other
            # files are ignored just by their name).
            abs_path = abs_real_path_and_basename[0]
            i = max(abs_path.rfind('/'), abs_path.rfind('\\'))
            if i:
                abs_path = abs_path[0:i]
                i = max(abs_path.rfind('/'), abs_path.rfind('\\'))
                if i:
                    dirname = abs_path[i + 1:]
                    # At this point, something as:
                    # "my_path\_pydev_runfiles\__init__.py"
                    # is now  "_pydev_runfiles".
                    return self._dont_trace_dirs_get_file_type(dirname)
        return None

    def dont_trace_external_files(self, abs_path):
        '''
        :param abs_path:
            The result from get_abs_path_real_path_and_base_from_file or
            get_abs_path_real_path_and_base_from_frame.

        :return
            True :
                If files should NOT be traced.

            False:
                If files should be traced.
        '''
        # By default all external files are traced. Note: this function is expected to
        # be changed for another function in PyDevdAPI.set_dont_trace_start_end_patterns.
        return False

    def get_file_type(self, frame, abs_real_path_and_basename=None, _cache_file_type=_CACHE_FILE_TYPE):
        '''
        :param abs_real_path_and_basename:
            The result from get_abs_path_real_path_and_base_from_file or
            get_abs_path_real_path_and_base_from_frame.

        :return
            _pydevd_bundle.pydevd_dont_trace_files.PYDEV_FILE:
                If it's a file internal to the debugger which shouldn't be
                traced nor shown to the user.

            _pydevd_bundle.pydevd_dont_trace_files.LIB_FILE:
                If it's a file in a library which shouldn't be traced.

            None:
                If it's a regular user file which should be traced.
        '''
        if abs_real_path_and_basename is None:
            try:
                # Make fast path faster!
                abs_real_path_and_basename = NORM_PATHS_AND_BASE_CONTAINER[frame.f_code.co_filename]
            except:
                abs_real_path_and_basename = get_abs_path_real_path_and_base_from_frame(frame)

        # Note 1: we have to take into account that we may have files as '<string>', and that in
        # this case the cache key can't rely only on the filename. With the current cache, there's
        # still a potential miss if 2 functions which have exactly the same content are compiled
        # with '<string>', but in practice as we only separate the one from python -c from the rest
        # this shouldn't be a problem in practice.

        # Note 2: firstlineno added to make misses faster in the first comparison.

        # Note 3: this cache key is repeated in pydevd_frame_evaluator.pyx:get_func_code_info (for
        # speedups).
        cache_key = (frame.f_code.co_firstlineno, abs_real_path_and_basename[0], frame.f_code)
        try:
            return _cache_file_type[cache_key]
        except:
            if abs_real_path_and_basename[0] == '<string>':

                # Consider it an untraceable file unless there's no back frame (ignoring
                # internal files and runpy.py).
                f = frame.f_back
                while f is not None:
                    if (self.get_file_type(f) != self.PYDEV_FILE and
                            pydevd_file_utils.basename(f.f_code.co_filename) not in ('runpy.py', '<string>')):
                        # We found some back frame that's not internal, which means we must consider
                        # this a library file.
                        # This is done because we only want to trace files as <string> if they don't
                        # have any back frame (which is the case for python -c ...), for all other
                        # cases we don't want to trace them because we can't show the source to the
                        # user (at least for now...).

                        # Note that we return as a LIB_FILE and not PYDEV_FILE because we still want
                        # to show it in the stack.
                        _cache_file_type[cache_key] = LIB_FILE
                        return LIB_FILE
                    f = f.f_back
                else:
                    # This is a top-level file (used in python -c), so, trace it as usual... we
                    # still won't be able to show the sources, but some tests require this to work.
                    _cache_file_type[cache_key] = None
                    return None

            file_type = self._internal_get_file_type(abs_real_path_and_basename)
            if file_type is None:
                if self.dont_trace_external_files(abs_real_path_and_basename[0]):
                    file_type = PYDEV_FILE

            _cache_file_type[cache_key] = file_type
            return file_type

    def is_cache_file_type_empty(self):
        return not _CACHE_FILE_TYPE

    def get_cache_file_type(self, _cache=_CACHE_FILE_TYPE):  # i.e.: Make it local.
        return _cache

    def get_thread_local_trace_func(self):
        try:
            thread_trace_func = self._local_thread_trace_func.thread_trace_func
        except AttributeError:
            thread_trace_func = self.trace_dispatch
        return thread_trace_func

    def enable_tracing(self, thread_trace_func=None, apply_to_all_threads=False):
        '''
        Enables tracing.

        If in regular mode (tracing), will set the tracing function to the tracing
        function for this thread -- by default it's `PyDB.trace_dispatch`, but after
        `PyDB.enable_tracing` is called with a `thread_trace_func`, the given function will
        be the default for the given thread.

        :param bool apply_to_all_threads:
            If True we'll set the tracing function in all threads, not only in the current thread.
            If False only the tracing for the current function should be changed.
            In general apply_to_all_threads should only be true if this is the first time
            this function is called on a multi-threaded program (either programmatically or attach
            to pid).
        '''
        if pydevd_gevent_integration is not None:
            pydevd_gevent_integration.enable_gevent_integration()

        if self.frame_eval_func is not None:
            self.frame_eval_func()
            pydevd_tracing.SetTrace(self.dummy_trace_dispatch)

            if IS_CPYTHON and apply_to_all_threads:
                pydevd_tracing.set_trace_to_threads(self.dummy_trace_dispatch)
            return

        if apply_to_all_threads:
            # If applying to all threads, don't use the local thread trace function.
            assert thread_trace_func is not None
        else:
            if thread_trace_func is None:
                thread_trace_func = self.get_thread_local_trace_func()
            else:
                self._local_thread_trace_func.thread_trace_func = thread_trace_func

        pydevd_tracing.SetTrace(thread_trace_func)
        if IS_CPYTHON and apply_to_all_threads:
            pydevd_tracing.set_trace_to_threads(thread_trace_func)

    def disable_tracing(self):
        pydevd_tracing.SetTrace(None)

    def on_breakpoints_changed(self, removed=False):
        '''
        When breakpoints change, we have to re-evaluate all the assumptions we've made so far.
        '''
        if not self.ready_to_run:
            # No need to do anything if we're still not running.
            return

        self.mtime += 1
        if not removed:
            # When removing breakpoints we can leave tracing as was, but if a breakpoint was added
            # we have to reset the tracing for the existing functions to be re-evaluated.
            self.set_tracing_for_untraced_contexts()

    def set_tracing_for_untraced_contexts(self):
        # Enable the tracing for existing threads (because there may be frames being executed that
        # are currently untraced).

        if IS_CPYTHON:
            # Note: use sys._current_frames instead of threading.enumerate() because this way
            # we also see C/C++ threads, not only the ones visible to the threading module.
            tid_to_frame = sys._current_frames()

            ignore_thread_ids = set(
                t.ident for t in threadingEnumerate()
                if getattr(t, 'is_pydev_daemon_thread', False) or getattr(t, 'pydev_do_not_trace', False)
            )

            for thread_id, frame in tid_to_frame.items():
                if thread_id not in ignore_thread_ids:
                    self.set_trace_for_frame_and_parents(frame)

        else:
            try:
                threads = threadingEnumerate()
                for t in threads:
                    if getattr(t, 'is_pydev_daemon_thread', False) or getattr(t, 'pydev_do_not_trace', False):
                        continue

                    additional_info = set_additional_thread_info(t)
                    frame = additional_info.get_topmost_frame(t)
                    try:
                        if frame is not None:
                            self.set_trace_for_frame_and_parents(frame)
                    finally:
                        frame = None
            finally:
                frame = None
                t = None
                threads = None
                additional_info = None

    @property
    def multi_threads_single_notification(self):
        return self._threads_suspended_single_notification.multi_threads_single_notification

    @multi_threads_single_notification.setter
    def multi_threads_single_notification(self, notify):
        self._threads_suspended_single_notification.multi_threads_single_notification = notify

    @property
    def threads_suspended_single_notification(self):
        return self._threads_suspended_single_notification

    def get_plugin_lazy_init(self):
        if self.plugin is None:
            self.plugin = PluginManager(self)
        return self.plugin

    def in_project_scope(self, frame, absolute_filename=None):
        '''
        Note: in general this method should not be used (apply_files_filter should be used
        in most cases as it also handles the project scope check).

        :param frame:
            The frame we want to check.

        :param absolute_filename:
            Must be the result from get_abs_path_real_path_and_base_from_frame(frame)[0] (can
            be used to speed this function a bit if it's already available to the caller, but
            in general it's not needed).
        '''
        try:
            if absolute_filename is None:
                try:
                    # Make fast path faster!
                    abs_real_path_and_basename = NORM_PATHS_AND_BASE_CONTAINER[frame.f_code.co_filename]
                except:
                    abs_real_path_and_basename = get_abs_path_real_path_and_base_from_frame(frame)

                absolute_filename = abs_real_path_and_basename[0]

            cache_key = (frame.f_code.co_firstlineno, absolute_filename, frame.f_code)

            return self._in_project_scope_cache[cache_key]
        except KeyError:
            cache = self._in_project_scope_cache
            try:
                abs_real_path_and_basename  # If we've gotten it previously, use it again.
            except NameError:
                abs_real_path_and_basename = get_abs_path_real_path_and_base_from_frame(frame)

            # pydevd files are never considered to be in the project scope.
            file_type = self.get_file_type(frame, abs_real_path_and_basename)
            if file_type == self.PYDEV_FILE:
                cache[cache_key] = False

            elif absolute_filename == '<string>':
                # Special handling for '<string>'
                if file_type == self.LIB_FILE:
                    cache[cache_key] = False
                else:
                    cache[cache_key] = True

            elif self.source_mapping.has_mapping_entry(absolute_filename):
                cache[cache_key] = True

            else:
                cache[cache_key] = self._files_filtering.in_project_roots(absolute_filename)

            return cache[cache_key]

    def in_project_roots_filename_uncached(self, absolute_filename):
        return self._files_filtering.in_project_roots(absolute_filename)

    def _clear_filters_caches(self):
        self._in_project_scope_cache.clear()
        self._exclude_by_filter_cache.clear()
        self._apply_filter_cache.clear()
        self._exclude_filters_enabled = self._files_filtering.use_exclude_filters()
        self._is_libraries_filter_enabled = self._files_filtering.use_libraries_filter()
        self.is_files_filter_enabled = self._exclude_filters_enabled or self._is_libraries_filter_enabled

    def clear_dont_trace_start_end_patterns_caches(self):
        # When start/end patterns are changed we must clear all caches which would be
        # affected by a change in get_file_type() and reset the tracing function
        # as places which were traced may no longer need to be traced and vice-versa.
        self.on_breakpoints_changed()
        _CACHE_FILE_TYPE.clear()
        self._clear_filters_caches()
        self._clear_skip_caches()

    def _exclude_by_filter(self, frame, absolute_filename):
        '''
        :return: True if it should be excluded, False if it should be included and None
            if no rule matched the given file.

        :note: it'll be normalized as needed inside of this method.
        '''
        cache_key = (absolute_filename, frame.f_code.co_name, frame.f_code.co_firstlineno)
        try:
            return self._exclude_by_filter_cache[cache_key]
        except KeyError:
            cache = self._exclude_by_filter_cache

            # pydevd files are always filtered out
            if self.get_file_type(frame) == self.PYDEV_FILE:
                cache[cache_key] = True
            else:
                module_name = None
                if self._files_filtering.require_module:
                    module_name = frame.f_globals.get('__name__', '')
                cache[cache_key] = self._files_filtering.exclude_by_filter(absolute_filename, module_name)

            return cache[cache_key]

    def apply_files_filter(self, frame, original_filename, force_check_project_scope):
        '''
        Should only be called if `self.is_files_filter_enabled == True` or `force_check_project_scope == True`.

        Note that it covers both the filter by specific paths includes/excludes as well
        as the check which filters out libraries if not in the project scope.

        :param original_filename:
            Note can either be the original filename or the absolute version of that filename.

        :param force_check_project_scope:
            Check that the file is in the project scope even if the global setting
            is off.

        :return bool:
            True if it should be excluded when stepping and False if it should be
            included.
        '''
        cache_key = (frame.f_code.co_firstlineno, original_filename, force_check_project_scope, frame.f_code)
        try:
            return self._apply_filter_cache[cache_key]
        except KeyError:
            if self.plugin is not None and (self.has_plugin_line_breaks or self.has_plugin_exception_breaks):
                # If it's explicitly needed by some plugin, we can't skip it.
                if not self.plugin.can_skip(self, frame):
                    pydev_log.debug_once('File traced (included by plugins): %s', original_filename)
                    self._apply_filter_cache[cache_key] = False
                    return False

            if self._exclude_filters_enabled:
                absolute_filename = pydevd_file_utils.absolute_path(original_filename)
                exclude_by_filter = self._exclude_by_filter(frame, absolute_filename)
                if exclude_by_filter is not None:
                    if exclude_by_filter:
                        # ignore files matching stepping filters
                        pydev_log.debug_once('File not traced (excluded by filters): %s', original_filename)

                        self._apply_filter_cache[cache_key] = True
                        return True
                    else:
                        pydev_log.debug_once('File traced (explicitly included by filters): %s', original_filename)

                        self._apply_filter_cache[cache_key] = False
                        return False

            if (self._is_libraries_filter_enabled or force_check_project_scope) and not self.in_project_scope(frame):
                # ignore library files while stepping
                self._apply_filter_cache[cache_key] = True
                if force_check_project_scope:
                    pydev_log.debug_once('File not traced (not in project): %s', original_filename)
                else:
                    pydev_log.debug_once('File not traced (not in project - force_check_project_scope): %s', original_filename)

                return True

            if force_check_project_scope:
                pydev_log.debug_once('File traced: %s (force_check_project_scope)', original_filename)
            else:
                pydev_log.debug_once('File traced: %s', original_filename)
            self._apply_filter_cache[cache_key] = False
            return False

    def exclude_exception_by_filter(self, exception_breakpoint, trace):
        if not exception_breakpoint.ignore_libraries and not self._exclude_filters_enabled:
            return False

        if trace is None:
            return True

        ignore_libraries = exception_breakpoint.ignore_libraries
        exclude_filters_enabled = self._exclude_filters_enabled

        if (ignore_libraries and not self.in_project_scope(trace.tb_frame)) \
                or (exclude_filters_enabled and self._exclude_by_filter(
                    trace.tb_frame,
                    pydevd_file_utils.absolute_path(trace.tb_frame.f_code.co_filename))):
            return True

        return False

    def set_project_roots(self, project_roots):
        self._files_filtering.set_project_roots(project_roots)
        self._clear_skip_caches()
        self._clear_filters_caches()

    def set_exclude_filters(self, exclude_filters):
        self._files_filtering.set_exclude_filters(exclude_filters)
        self._clear_skip_caches()
        self._clear_filters_caches()

    def set_use_libraries_filter(self, use_libraries_filter):
        self._files_filtering.set_use_libraries_filter(use_libraries_filter)
        self._clear_skip_caches()
        self._clear_filters_caches()

    def get_use_libraries_filter(self):
        return self._files_filtering.use_libraries_filter()

    def get_require_module_for_filters(self):
        return self._files_filtering.require_module

    def has_user_threads_alive(self):
        for t in pydevd_utils.get_non_pydevd_threads():
            if isinstance(t, PyDBDaemonThread):
                pydev_log.error_once(
                    'Error in debugger: Found PyDBDaemonThread not marked with is_pydev_daemon_thread=True.\n')

            if is_thread_alive(t):
                if not t.daemon or hasattr(t, "__pydevd_main_thread"):
                    return True

        return False

    def initialize_network(self, sock, terminate_on_socket_close=True):
        assert sock is not None
        try:
            sock.settimeout(None)  # infinite, no timeouts from now on - jython does not have it
        except:
            pass
        curr_reader = getattr(self, 'reader', None)
        curr_writer = getattr(self, 'writer', None)
        if curr_reader:
            curr_reader.do_kill_pydev_thread()
        if curr_writer:
            curr_writer.do_kill_pydev_thread()

        self.writer = WriterThread(sock, self, terminate_on_socket_close=terminate_on_socket_close)
        self.reader = ReaderThread(
            sock,
            self,
            PyDevJsonCommandProcessor=PyDevJsonCommandProcessor,
            process_net_command=process_net_command,
            terminate_on_socket_close=terminate_on_socket_close
        )
        self.writer.start()
        self.reader.start()

        time.sleep(0.1)  # give threads time to start

    def connect(self, host, port):
        if host:
            s = start_client(host, port)
        else:
            s = start_server(port)

        self.initialize_network(s)

    def create_wait_for_connection_thread(self):
        if self._waiting_for_connection_thread is not None:
            raise AssertionError('There is already another thread waiting for a connection.')

        self._server_socket_ready_event.clear()
        self._waiting_for_connection_thread = self._WaitForConnectionThread(self)
        self._waiting_for_connection_thread.start()

    def set_server_socket_ready(self):
        self._server_socket_ready_event.set()

    def wait_for_server_socket_ready(self):
        self._server_socket_ready_event.wait()

    @property
    def dap_messages_listeners(self):
        return self._dap_messages_listeners

    def add_dap_messages_listener(self, listener):
        self._dap_messages_listeners.append(listener)

    class _WaitForConnectionThread(PyDBDaemonThread):

        def __init__(self, py_db):
            PyDBDaemonThread.__init__(self, py_db)
            self._server_socket = None

        def run(self):
            host = SetupHolder.setup['client']
            port = SetupHolder.setup['port']

            self._server_socket = create_server_socket(host=host, port=port)
            self.py_db._server_socket_name = self._server_socket.getsockname()
            self.py_db.set_server_socket_ready()

            while not self._kill_received:
                try:
                    s = self._server_socket
                    if s is None:
                        return

                    s.listen(1)
                    new_socket, _addr = s.accept()
                    if self._kill_received:
                        pydev_log.info("Connection (from wait_for_attach) accepted but ignored as kill was already received.")
                        return

                    pydev_log.info("Connection (from wait_for_attach) accepted.")
                    reader = getattr(self.py_db, 'reader', None)
                    if reader is not None:
                        # This is needed if a new connection is done without the client properly
                        # sending a disconnect for the previous connection.
                        api = PyDevdAPI()
                        api.request_disconnect(self.py_db, resume_threads=False)

                    self.py_db.initialize_network(new_socket, terminate_on_socket_close=False)

                except:
                    if DebugInfoHolder.DEBUG_TRACE_LEVEL > 0:
                        pydev_log.exception()
                        pydev_log.debug("Exiting _WaitForConnectionThread: %s\n", port)

        def do_kill_pydev_thread(self):
            PyDBDaemonThread.do_kill_pydev_thread(self)
            s = self._server_socket
            if s is not None:
                try:
                    s.close()
                except:
                    pass
                self._server_socket = None

    def get_internal_queue(self, thread_id):
        """ returns internal command queue for a given thread.
        if new queue is created, notify the RDB about it """
        if thread_id.startswith('__frame__'):
            thread_id = thread_id[thread_id.rfind('|') + 1:]
        return self._cmd_queue[thread_id]

    def post_method_as_internal_command(self, thread_id, method, *args, **kwargs):
        if thread_id == '*':
            internal_cmd = InternalThreadCommandForAnyThread(thread_id, method, *args, **kwargs)
        else:
            internal_cmd = InternalThreadCommand(thread_id, method, *args, **kwargs)
        self.post_internal_command(internal_cmd, thread_id)
        if thread_id == '*':
            # Notify so that the command is handled as soon as possible.
            self._py_db_command_thread_event.set()

    def post_internal_command(self, int_cmd, thread_id):
        """ if thread_id is *, post to the '*' queue"""
        queue = self.get_internal_queue(thread_id)
        queue.put(int_cmd)

    def enable_output_redirection(self, redirect_stdout, redirect_stderr):
        global _global_redirect_stdout_to_server
        global _global_redirect_stderr_to_server

        _global_redirect_stdout_to_server = redirect_stdout
        _global_redirect_stderr_to_server = redirect_stderr
        self.redirect_output = redirect_stdout or redirect_stderr
        if _global_redirect_stdout_to_server:
            _init_stdout_redirect()
        if _global_redirect_stderr_to_server:
            _init_stderr_redirect()

    def check_output_redirect(self):
        global _global_redirect_stdout_to_server
        global _global_redirect_stderr_to_server

        if _global_redirect_stdout_to_server:
            _init_stdout_redirect()

        if _global_redirect_stderr_to_server:
            _init_stderr_redirect()

    def init_matplotlib_in_debug_console(self):
        # import hook and patches for matplotlib support in debug console
        from _pydev_bundle.pydev_import_hook import import_hook_manager
        if is_current_thread_main_thread():
            for module in list(self.mpl_modules_for_patching):
                import_hook_manager.add_module_name(module, self.mpl_modules_for_patching.pop(module))

    def init_gui_support(self):
        if self._installed_gui_support:
            return
        self._installed_gui_support = True

        # enable_gui and enable_gui_function in activate_matplotlib should be called in main thread. Unlike integrated console,
        # in the debug console we have no interpreter instance with exec_queue, but we run this code in the main
        # thread and can call it directly.
        class _ReturnGuiLoopControlHelper:
            _return_control_osc = False

        def return_control():
            # Some of the input hooks (e.g. Qt4Agg) check return control without doing
            # a single operation, so we don't return True on every
            # call when the debug hook is in place to allow the GUI to run
            _ReturnGuiLoopControlHelper._return_control_osc = not _ReturnGuiLoopControlHelper._return_control_osc
            return _ReturnGuiLoopControlHelper._return_control_osc

        from pydev_ipython.inputhook import set_return_control_callback, enable_gui

        set_return_control_callback(return_control)

        if self._gui_event_loop == 'matplotlib':
            # prepare debugger for matplotlib integration with GUI event loop
            from pydev_ipython.matplotlibtools import activate_matplotlib, activate_pylab, activate_pyplot, do_enable_gui

            self.mpl_modules_for_patching = {"matplotlib": lambda: activate_matplotlib(do_enable_gui),
                                "matplotlib.pyplot": activate_pyplot,
                                "pylab": activate_pylab }
        else:
            self.activate_gui_function = enable_gui

    def _activate_gui_if_needed(self):
        if self.gui_in_use:
            return

        if len(self.mpl_modules_for_patching) > 0:
            if is_current_thread_main_thread():  # Note that we call only in the main thread.
                for module in list(self.mpl_modules_for_patching):
                    if module in sys.modules:
                        activate_function = self.mpl_modules_for_patching.pop(module, None)
                        if activate_function is not None:
                            activate_function()
                        self.gui_in_use = True

        if self.activate_gui_function:
            if is_current_thread_main_thread():  # Only call enable_gui in the main thread.
                try:
                    # First try to activate builtin GUI event loops.
                    self.activate_gui_function(self._gui_event_loop)
                    self.activate_gui_function = None
                    self.gui_in_use = True
                except ValueError:
                    # The user requested a custom GUI event loop, try to import it.
                    from pydev_ipython.inputhook import set_inputhook
                    try:
                        inputhook_function = import_attr_from_module(self._gui_event_loop)
                        set_inputhook(inputhook_function)
                        self.gui_in_use = True
                    except Exception as e:
                        pydev_log.debug("Cannot activate custom GUI event loop {}: {}".format(self._gui_event_loop, e))
                    finally:
                        self.activate_gui_function = None

    def _call_input_hook(self):
        try:
            from pydev_ipython.inputhook import get_inputhook
            inputhook = get_inputhook()
            if inputhook:
                inputhook()
        except:
            pass

    def notify_skipped_step_in_because_of_filters(self, frame):
        self.writer.add_command(self.cmd_factory.make_skipped_step_in_because_of_filters(self, frame))

    def notify_thread_created(self, thread_id, thread, use_lock=True):
        if self.writer is None:
            # Protect about threads being created before the communication structure is in place
            # (note that they will appear later on anyways as pydevd does reconcile live/dead threads
            # when processing internal commands, albeit it may take longer and in general this should
            # not be usual as it's expected that the debugger is live before other threads are created).
            return

        with self._lock_running_thread_ids if use_lock else NULL:
            if not self._enable_thread_notifications:
                return

            if thread_id in self._running_thread_ids:
                return

            additional_info = set_additional_thread_info(thread)
            if additional_info.pydev_notify_kill:
                # After we notify it should be killed, make sure we don't notify it's alive (on a racing condition
                # this could happen as we may notify before the thread is stopped internally).
                return

            self._running_thread_ids[thread_id] = thread

        self.writer.add_command(self.cmd_factory.make_thread_created_message(thread))

    def notify_thread_not_alive(self, thread_id, use_lock=True):
        """ if thread is not alive, cancel trace_dispatch processing """
        if self.writer is None:
            return

        with self._lock_running_thread_ids if use_lock else NULL:
            if not self._enable_thread_notifications:
                return

            thread = self._running_thread_ids.pop(thread_id, None)
            if thread is None:
                return

            additional_info = set_additional_thread_info(thread)
            was_notified = additional_info.pydev_notify_kill
            if not was_notified:
                additional_info.pydev_notify_kill = True

        self.writer.add_command(self.cmd_factory.make_thread_killed_message(thread_id))

    def set_enable_thread_notifications(self, enable):
        with self._lock_running_thread_ids:
            if self._enable_thread_notifications != enable:
                self._enable_thread_notifications = enable

                if enable:
                    # As it was previously disabled, we have to notify about existing threads again
                    # (so, clear the cache related to that).
                    self._running_thread_ids = {}

    def process_internal_commands(self):
        '''
        This function processes internal commands.
        '''
        # If this method is being called before the debugger is ready to run we should not notify
        # about threads and should only process commands sent to all threads.
        ready_to_run = self.ready_to_run

        dispose = False
        with self._main_lock:
            program_threads_alive = {}
            if ready_to_run:
                self.check_output_redirect()

                all_threads = threadingEnumerate()
                program_threads_dead = []
                with self._lock_running_thread_ids:
                    reset_cache = not self._running_thread_ids

                    for t in all_threads:
                        if getattr(t, 'is_pydev_daemon_thread', False):
                            pass  # I.e.: skip the DummyThreads created from pydev daemon threads
                        elif isinstance(t, PyDBDaemonThread):
                            pydev_log.error_once('Error in debugger: Found PyDBDaemonThread not marked with is_pydev_daemon_thread=True.')

                        elif is_thread_alive(t):
                            if reset_cache:
                                # Fix multiprocessing debug with breakpoints in both main and child processes
                                # (https://youtrack.jetbrains.com/issue/PY-17092) When the new process is created, the main
                                # thread in the new process already has the attribute 'pydevd_id', so the new thread doesn't
                                # get new id with its process number and the debugger loses access to both threads.
                                # Therefore we should update thread_id for every main thread in the new process.
                                clear_cached_thread_id(t)

                            thread_id = get_thread_id(t)
                            program_threads_alive[thread_id] = t

                            self.notify_thread_created(thread_id, t, use_lock=False)

                    # Compute and notify about threads which are no longer alive.
                    thread_ids = list(self._running_thread_ids.keys())
                    for thread_id in thread_ids:
                        if thread_id not in program_threads_alive:
                            program_threads_dead.append(thread_id)

                    for thread_id in program_threads_dead:
                        self.notify_thread_not_alive(thread_id, use_lock=False)

            cmds_to_execute = []

            # Without self._lock_running_thread_ids
            if len(program_threads_alive) == 0 and ready_to_run:
                dispose = True
            else:
                # Actually process the commands now (make sure we don't have a lock for _lock_running_thread_ids
                # acquired at this point as it could lead to a deadlock if some command evaluated tried to
                # create a thread and wait for it -- which would try to notify about it getting that lock).
                curr_thread_id = get_current_thread_id(threadingCurrentThread())
                if ready_to_run:
                    process_thread_ids = (curr_thread_id, '*')
                else:
                    process_thread_ids = ('*',)

                for thread_id in process_thread_ids:
                    queue = self.get_internal_queue(thread_id)

                    # some commands must be processed by the thread itself... if that's the case,
                    # we will re-add the commands to the queue after executing.
                    cmds_to_add_back = []

                    try:
                        while True:
                            int_cmd = queue.get(False)

                            if not self.mpl_hooks_in_debug_console and isinstance(int_cmd, InternalConsoleExec) and not self.gui_in_use:
                                # add import hooks for matplotlib patches if only debug console was started
                                try:
                                    self.init_matplotlib_in_debug_console()
                                    self.gui_in_use = True
                                except:
                                    pydev_log.debug("Matplotlib support in debug console failed", traceback.format_exc())
                                self.mpl_hooks_in_debug_console = True

                            if int_cmd.can_be_executed_by(curr_thread_id):
                                cmds_to_execute.append(int_cmd)
                            else:
                                pydev_log.verbose("NOT processing internal command: %s ", int_cmd)
                                cmds_to_add_back.append(int_cmd)

                    except _queue.Empty:  # @UndefinedVariable
                        # this is how we exit
                        for int_cmd in cmds_to_add_back:
                            queue.put(int_cmd)

        if dispose:
            # Note: must be called without the main lock to avoid deadlocks.
            self.dispose_and_kill_all_pydevd_threads()
        else:
            # Actually execute the commands without the main lock!
            for int_cmd in cmds_to_execute:
                pydev_log.verbose("processing internal command: %s", int_cmd)
                try:
                    int_cmd.do_it(self)
                except:
                    pydev_log.exception('Error processing internal command.')

    def consolidate_breakpoints(self, canonical_normalized_filename, id_to_breakpoint, file_to_line_to_breakpoints):
        break_dict = {}
        for _breakpoint_id, pybreakpoint in id_to_breakpoint.items():
            break_dict[pybreakpoint.line] = pybreakpoint

        file_to_line_to_breakpoints[canonical_normalized_filename] = break_dict
        self._clear_skip_caches()

    def _clear_skip_caches(self):
        global_cache_skips.clear()
        global_cache_frame_skips.clear()

    def add_break_on_exception(
        self,
        exception,
        condition,
        expression,
        notify_on_handled_exceptions,
        notify_on_unhandled_exceptions,
        notify_on_user_unhandled_exceptions,
        notify_on_first_raise_only,
        ignore_libraries=False
        ):
        try:
            eb = ExceptionBreakpoint(
                exception,
                condition,
                expression,
                notify_on_handled_exceptions,
                notify_on_unhandled_exceptions,
                notify_on_user_unhandled_exceptions,
                notify_on_first_raise_only,
                ignore_libraries
            )
        except ImportError:
            pydev_log.critical("Error unable to add break on exception for: %s (exception could not be imported).", exception)
            return None

        if eb.notify_on_unhandled_exceptions:
            cp = self.break_on_uncaught_exceptions.copy()
            cp[exception] = eb
            pydev_log.info("Exceptions to hook on terminate: %s.", cp)
            self.break_on_uncaught_exceptions = cp

        if eb.notify_on_handled_exceptions:
            cp = self.break_on_caught_exceptions.copy()
            cp[exception] = eb
            pydev_log.info("Exceptions to hook always: %s.", cp)
            self.break_on_caught_exceptions = cp

        if eb.notify_on_user_unhandled_exceptions:
            cp = self.break_on_user_uncaught_exceptions.copy()
            cp[exception] = eb
            pydev_log.info("Exceptions to hook on user uncaught code: %s.", cp)
            self.break_on_user_uncaught_exceptions = cp

        return eb

    def set_suspend(self, thread, stop_reason, suspend_other_threads=False, is_pause=False, original_step_cmd=-1):
        '''
        :param thread:
            The thread which should be suspended.

        :param stop_reason:
            Reason why the thread was suspended.

        :param suspend_other_threads:
            Whether to force other threads to be suspended (i.e.: when hitting a breakpoint
            with a suspend all threads policy).

        :param is_pause:
            If this is a pause to suspend all threads, any thread can be considered as the 'main'
            thread paused.

        :param original_step_cmd:
            If given we may change the stop reason to this.
        '''
        self._threads_suspended_single_notification.increment_suspend_time()
        if is_pause:
            self._threads_suspended_single_notification.on_pause()

        info = mark_thread_suspended(thread, stop_reason, original_step_cmd=original_step_cmd)

        if is_pause:
            # Must set tracing after setting the state to suspend.
            frame = info.get_topmost_frame(thread)
            if frame is not None:
                try:
                    self.set_trace_for_frame_and_parents(frame)
                finally:
                    frame = None

        # If conditional breakpoint raises any exception during evaluation send the details to the client.
        if stop_reason == CMD_SET_BREAK and info.conditional_breakpoint_exception is not None:
            conditional_breakpoint_exception_tuple = info.conditional_breakpoint_exception
            info.conditional_breakpoint_exception = None
            self._send_breakpoint_condition_exception(thread, conditional_breakpoint_exception_tuple)

        if not suspend_other_threads and self.multi_threads_single_notification:
            # In the mode which gives a single notification when all threads are
            # stopped, stop all threads whenever a set_suspend is issued.
            suspend_other_threads = True

        if suspend_other_threads:
            # Suspend all except the current one (which we're currently suspending already).
            suspend_all_threads(self, except_thread=thread)

    def _send_breakpoint_condition_exception(self, thread, conditional_breakpoint_exception_tuple):
        """If conditional breakpoint raises an exception during evaluation
        send exception details to java
        """
        thread_id = get_thread_id(thread)
        # conditional_breakpoint_exception_tuple - should contain 2 values (exception_type, stacktrace)
        if conditional_breakpoint_exception_tuple and len(conditional_breakpoint_exception_tuple) == 2:
            exc_type, stacktrace = conditional_breakpoint_exception_tuple
            int_cmd = InternalGetBreakpointException(thread_id, exc_type, stacktrace)
            self.post_internal_command(int_cmd, thread_id)

    def send_caught_exception_stack(self, thread, arg, curr_frame_id):
        """Sends details on the exception which was caught (and where we stopped) to the java side.

        arg is: exception type, description, traceback object
        """
        thread_id = get_thread_id(thread)
        int_cmd = InternalSendCurrExceptionTrace(thread_id, arg, curr_frame_id)
        self.post_internal_command(int_cmd, thread_id)

    def send_caught_exception_stack_proceeded(self, thread):
        """Sends that some thread was resumed and is no longer showing an exception trace.
        """
        thread_id = get_thread_id(thread)
        int_cmd = InternalSendCurrExceptionTraceProceeded(thread_id)
        self.post_internal_command(int_cmd, thread_id)
        self.process_internal_commands()

    def send_process_created_message(self):
        """Sends a message that a new process has been created.
        """
        if self.writer is None or self.cmd_factory is None:
            return
        cmd = self.cmd_factory.make_process_created_message()
        self.writer.add_command(cmd)

    def send_process_about_to_be_replaced(self):
        """Sends a message that a new process has been created.
        """
        if self.writer is None or self.cmd_factory is None:
            return
        cmd = self.cmd_factory.make_process_about_to_be_replaced_message()
        if cmd is NULL_NET_COMMAND:
            return

        sent = [False]

        def after_sent(*args, **kwargs):
            sent[0] = True

        cmd.call_after_send(after_sent)
        self.writer.add_command(cmd)

        timeout = 5  # Wait up to 5 seconds
        initial_time = time.time()
        while not sent[0]:
            time.sleep(.05)

            if (time.time() - initial_time) > timeout:
                pydev_log.critical('pydevd: Sending message related to process being replaced timed-out after %s seconds', timeout)
                break

    def set_next_statement(self, frame, event, func_name, next_line):
        stop = False
        response_msg = ""
        old_line = frame.f_lineno
        if event == 'line' or event == 'exception':
            # If we're already in the correct context, we have to stop it now, because we can act only on
            # line events -- if a return was the next statement it wouldn't work (so, we have this code
            # repeated at pydevd_frame).

            curr_func_name = frame.f_code.co_name

            # global context is set with an empty name
            if curr_func_name in ('?', '<module>'):
                curr_func_name = ''

            if func_name == '*' or curr_func_name == func_name:
                line = next_line
                frame.f_trace = self.trace_dispatch
                frame.f_lineno = line
                stop = True
            else:
                response_msg = "jump is available only within the bottom frame"
        return stop, old_line, response_msg

    def cancel_async_evaluation(self, thread_id, frame_id):
        with self._main_lock:
            try:
                all_threads = threadingEnumerate()
                for t in all_threads:
                    if getattr(t, 'is_pydev_daemon_thread', False) and hasattr(t, 'cancel_event') and t.thread_id == thread_id and \
                            t.frame_id == frame_id:
                        t.cancel_event.set()
            except:
                pydev_log.exception()

    def find_frame(self, thread_id, frame_id):
        """ returns a frame on the thread that has a given frame_id """
        return self.suspended_frames_manager.find_frame(thread_id, frame_id)

    def do_wait_suspend(self, thread, frame, event, arg, exception_type=None):  # @UnusedVariable
        """ busy waits until the thread state changes to RUN
        it expects thread's state as attributes of the thread.
        Upon running, processes any outstanding Stepping commands.

        :param exception_type:
            If pausing due to an exception, its type.
        """
        if USE_CUSTOM_SYS_CURRENT_FRAMES_MAP:
            constructed_tid_to_last_frame[thread.ident] = sys._getframe()
        self.process_internal_commands()

        thread_id = get_current_thread_id(thread)

        # print('do_wait_suspend %s %s %s %s %s %s (%s)' % (frame.f_lineno, frame.f_code.co_name, frame.f_code.co_filename, event, arg, constant_to_str(thread.additional_info.pydev_step_cmd), constant_to_str(thread.additional_info.pydev_original_step_cmd)))
        # print('--- stack ---')
        # print(traceback.print_stack(file=sys.stdout))
        # print('--- end stack ---')

        # Send the suspend message
        message = thread.additional_info.pydev_message
        suspend_type = thread.additional_info.trace_suspend_type
        thread.additional_info.trace_suspend_type = 'trace'  # Reset to trace mode for next call.
        stop_reason = thread.stop_reason

        frames_list = None

        if arg is not None and event == 'exception':
            # arg must be the exception info (tuple(exc_type, exc, traceback))
            exc_type, exc_desc, trace_obj = arg
            if trace_obj is not None:
                frames_list = pydevd_frame_utils.create_frames_list_from_traceback(trace_obj, frame, exc_type, exc_desc, exception_type=exception_type)

        if frames_list is None:
            frames_list = pydevd_frame_utils.create_frames_list_from_frame(frame)

        if DebugInfoHolder.DEBUG_TRACE_LEVEL > 2:
            pydev_log.debug(
                'PyDB.do_wait_suspend\nname: %s (line: %s)\n file: %s\n event: %s\n arg: %s\n step: %s (original step: %s)\n thread: %s, thread id: %s, id(thread): %s',
                frame.f_code.co_name,
                frame.f_lineno,
                frame.f_code.co_filename,
                event,
                arg,
                constant_to_str(thread.additional_info.pydev_step_cmd),
                constant_to_str(thread.additional_info.pydev_original_step_cmd),
                thread,
                thread_id,
                id(thread),
            )
            for f in frames_list:
                pydev_log.debug('  Stack: %s, %s, %s', f.f_code.co_filename, f.f_code.co_name, f.f_lineno)

        with self.suspended_frames_manager.track_frames(self) as frames_tracker:
            frames_tracker.track(thread_id, frames_list)
            cmd = frames_tracker.create_thread_suspend_command(thread_id, stop_reason, message, suspend_type)
            self.writer.add_command(cmd)

            with CustomFramesContainer.custom_frames_lock:  # @UndefinedVariable
                from_this_thread = []

                for frame_custom_thread_id, custom_frame in CustomFramesContainer.custom_frames.items():
                    if custom_frame.thread_id == thread.ident:
                        frames_tracker.track(thread_id, pydevd_frame_utils.create_frames_list_from_frame(custom_frame.frame), frame_custom_thread_id=frame_custom_thread_id)
                        # print('Frame created as thread: %s' % (frame_custom_thread_id,))

                        self.writer.add_command(self.cmd_factory.make_custom_frame_created_message(
                            frame_custom_thread_id, custom_frame.name))

                        self.writer.add_command(
                            frames_tracker.create_thread_suspend_command(frame_custom_thread_id, CMD_THREAD_SUSPEND, "", suspend_type))

                    from_this_thread.append(frame_custom_thread_id)

            with self._threads_suspended_single_notification.notify_thread_suspended(thread_id, thread, stop_reason):
                keep_suspended = self._do_wait_suspend(thread, frame, event, arg, suspend_type, from_this_thread, frames_tracker)

        frames_list = None

        if keep_suspended:
            # This means that we should pause again after a set next statement.
            self._threads_suspended_single_notification.increment_suspend_time()
            self.do_wait_suspend(thread, frame, event, arg, exception_type)
        if DebugInfoHolder.DEBUG_TRACE_LEVEL > 2:
            pydev_log.debug('Leaving PyDB.do_wait_suspend: %s (%s) %s', thread, thread_id, id(thread))

    def _do_wait_suspend(self, thread, frame, event, arg, suspend_type, from_this_thread, frames_tracker):
        info = thread.additional_info
        info.step_in_initial_location = None
        keep_suspended = False

        with self._main_lock:  # Use lock to check if suspended state changed
            activate_gui = info.pydev_state == STATE_SUSPEND and not self.pydb_disposed

        in_main_thread = is_current_thread_main_thread()
        if activate_gui and in_main_thread:
            # before every stop check if matplotlib modules were imported inside script code
            # or some GUI event loop needs to be activated
            self._activate_gui_if_needed()

        while True:
            with self._main_lock:  # Use lock to check if suspended state changed
                if info.pydev_state != STATE_SUSPEND or (self.pydb_disposed and not self.terminate_requested):
                    # Note: we can't exit here if terminate was requested while a breakpoint was hit.
                    break

            if in_main_thread and self.gui_in_use:
                # call input hooks if only GUI is in use
                self._call_input_hook()

            self.process_internal_commands()
            time.sleep(0.01)

        self.cancel_async_evaluation(get_current_thread_id(thread), str(id(frame)))

        # process any stepping instructions
        if info.pydev_step_cmd in (CMD_STEP_INTO, CMD_STEP_INTO_MY_CODE):
            info.step_in_initial_location = (frame, frame.f_lineno)
            if frame.f_code.co_flags & 0x80:  # CO_COROUTINE = 0x80
                # When in a coroutine we switch to CMD_STEP_INTO_COROUTINE.
                info.pydev_step_cmd = CMD_STEP_INTO_COROUTINE
                info.pydev_step_stop = frame
                self.set_trace_for_frame_and_parents(frame)
            else:
                info.pydev_step_stop = None
                self.set_trace_for_frame_and_parents(frame)

        elif info.pydev_step_cmd in (CMD_STEP_OVER, CMD_STEP_OVER_MY_CODE, CMD_SMART_STEP_INTO):
            info.pydev_step_stop = frame
            self.set_trace_for_frame_and_parents(frame)

        elif info.pydev_step_cmd == CMD_RUN_TO_LINE or info.pydev_step_cmd == CMD_SET_NEXT_STATEMENT:
            info.pydev_step_stop = None
            self.set_trace_for_frame_and_parents(frame)
            stop = False
            response_msg = ""
            try:
                stop, _old_line, response_msg = self.set_next_statement(frame, event, info.pydev_func_name, info.pydev_next_line)
            except ValueError as e:
                response_msg = "%s" % e
            finally:
                seq = info.pydev_message
                cmd = self.cmd_factory.make_set_next_stmnt_status_message(seq, stop, response_msg)
                self.writer.add_command(cmd)
                info.pydev_message = ''

            if stop:
                # Uninstall the current frames tracker before running it.
                frames_tracker.untrack_all()
                cmd = self.cmd_factory.make_thread_run_message(get_current_thread_id(thread), info.pydev_step_cmd)
                self.writer.add_command(cmd)
                info.pydev_state = STATE_SUSPEND
                thread.stop_reason = CMD_SET_NEXT_STATEMENT
                keep_suspended = True

            else:
                # Set next did not work...
                info.pydev_original_step_cmd = -1
                info.pydev_step_cmd = -1
                info.pydev_state = STATE_SUSPEND
                thread.stop_reason = CMD_THREAD_SUSPEND
                # return to the suspend state and wait for other command (without sending any
                # additional notification to the client).
                return self._do_wait_suspend(thread, frame, event, arg, suspend_type, from_this_thread, frames_tracker)

        elif info.pydev_step_cmd in (CMD_STEP_RETURN, CMD_STEP_RETURN_MY_CODE):
            back_frame = frame.f_back
            force_check_project_scope = info.pydev_step_cmd == CMD_STEP_RETURN_MY_CODE

            if force_check_project_scope or self.is_files_filter_enabled:
                while back_frame is not None:
                    if self.apply_files_filter(back_frame, back_frame.f_code.co_filename, force_check_project_scope):
                        frame = back_frame
                        back_frame = back_frame.f_back
                    else:
                        break

            if back_frame is not None:
                # steps back to the same frame (in a return call it will stop in the 'back frame' for the user)
                info.pydev_step_stop = frame
                self.set_trace_for_frame_and_parents(frame)
            else:
                # No back frame?!? -- this happens in jython when we have some frame created from an awt event
                # (the previous frame would be the awt event, but this doesn't make part of 'jython', only 'java')
                # so, if we're doing a step return in this situation, it's the same as just making it run
                info.pydev_step_stop = None
                info.pydev_original_step_cmd = -1
                info.pydev_step_cmd = -1
                info.pydev_state = STATE_RUN

        if PYDEVD_IPYTHON_COMPATIBLE_DEBUGGING:
            info.pydev_use_scoped_step_frame = False
            if info.pydev_step_cmd in (
                    CMD_STEP_OVER, CMD_STEP_OVER_MY_CODE,
                    CMD_STEP_INTO, CMD_STEP_INTO_MY_CODE
                ):
                # i.e.: We're stepping: check if the stepping should be scoped (i.e.: in ipython
                # each line is executed separately in a new frame, in which case we need to consider
                # the next line as if it was still in the same frame).
                f = frame.f_back
                if f and f.f_code.co_name == PYDEVD_IPYTHON_CONTEXT[1]:
                    f = f.f_back
                    if f and f.f_code.co_name == PYDEVD_IPYTHON_CONTEXT[2]:
                        info.pydev_use_scoped_step_frame = True
                        pydev_log.info('Using (ipython) scoped stepping.')
                del f

        del frame
        cmd = self.cmd_factory.make_thread_run_message(get_current_thread_id(thread), info.pydev_step_cmd)
        self.writer.add_command(cmd)

        with CustomFramesContainer.custom_frames_lock:
            # The ones that remained on last_running must now be removed.
            for frame_id in from_this_thread:
                # print('Removing created frame: %s' % (frame_id,))
                self.writer.add_command(self.cmd_factory.make_thread_killed_message(frame_id))

        return keep_suspended

    def do_stop_on_unhandled_exception(self, thread, frame, frames_byid, arg):
        pydev_log.debug("We are stopping in unhandled exception.")
        try:
            add_exception_to_frame(frame, arg)
            self.send_caught_exception_stack(thread, arg, id(frame))
            try:
                self.set_suspend(thread, CMD_ADD_EXCEPTION_BREAK)
                self.do_wait_suspend(thread, frame, 'exception', arg, EXCEPTION_TYPE_UNHANDLED)
            except:
                self.send_caught_exception_stack_proceeded(thread)
        except:
            pydev_log.exception("We've got an error while stopping in unhandled exception: %s.", arg[0])
        finally:
            remove_exception_from_frame(frame)
            frame = None

    def set_trace_for_frame_and_parents(self, frame, **kwargs):
        disable = kwargs.pop('disable', False)
        assert not kwargs

        while frame is not None:
            # Don't change the tracing on debugger-related files
            file_type = self.get_file_type(frame)

            if file_type is None:
                if disable:
                    pydev_log.debug('Disable tracing of frame: %s - %s', frame.f_code.co_filename, frame.f_code.co_name)
                    if frame.f_trace is not None and frame.f_trace is not NO_FTRACE:
                        frame.f_trace = NO_FTRACE

                elif frame.f_trace is not self.trace_dispatch:
                    pydev_log.debug('Set tracing of frame: %s - %s', frame.f_code.co_filename, frame.f_code.co_name)
                    frame.f_trace = self.trace_dispatch
            else:
                pydev_log.debug('SKIP set tracing of frame: %s - %s', frame.f_code.co_filename, frame.f_code.co_name)

            frame = frame.f_back

        del frame

    def _create_pydb_command_thread(self):
        curr_pydb_command_thread = self.py_db_command_thread
        if curr_pydb_command_thread is not None:
            curr_pydb_command_thread.do_kill_pydev_thread()

        new_pydb_command_thread = self.py_db_command_thread = PyDBCommandThread(self)
        new_pydb_command_thread.start()

    def _create_check_output_thread(self):
        curr_output_checker_thread = self.check_alive_thread
        if curr_output_checker_thread is not None:
            curr_output_checker_thread.do_kill_pydev_thread()

        check_alive_thread = self.check_alive_thread = CheckAliveThread(self)
        check_alive_thread.start()

    def start_auxiliary_daemon_threads(self):
        self._create_pydb_command_thread()
        self._create_check_output_thread()

    def __wait_for_threads_to_finish(self, timeout):
        try:
            with self._wait_for_threads_to_finish_called_lock:
                wait_for_threads_to_finish_called = self._wait_for_threads_to_finish_called
                self._wait_for_threads_to_finish_called = True

            if wait_for_threads_to_finish_called:
                # Make sure that we wait for the previous call to be finished.
                self._wait_for_threads_to_finish_called_event.wait(timeout=timeout)
            else:
                try:

                    def get_pydb_daemon_threads_to_wait():
                        pydb_daemon_threads = set(self.created_pydb_daemon_threads)
                        pydb_daemon_threads.discard(self.check_alive_thread)
                        pydb_daemon_threads.discard(threading.current_thread())
                        return pydb_daemon_threads

                    pydev_log.debug("PyDB.dispose_and_kill_all_pydevd_threads waiting for pydb daemon threads to finish")
                    started_at = time.time()
                    # Note: we wait for all except the check_alive_thread (which is not really a daemon
                    # thread and it can call this method itself).
                    while time.time() < started_at + timeout:
                        if len(get_pydb_daemon_threads_to_wait()) == 0:
                            break
                        time.sleep(1 / 10.)
                    else:
                        thread_names = [t.name for t in get_pydb_daemon_threads_to_wait()]
                        if thread_names:
                            pydev_log.debug("The following pydb threads may not have finished correctly: %s",
                                            ', '.join(thread_names))
                finally:
                    self._wait_for_threads_to_finish_called_event.set()
        except:
            pydev_log.exception()

    def dispose_and_kill_all_pydevd_threads(self, wait=True, timeout=.5):
        '''
        When this method is called we finish the debug session, terminate threads
        and if this was registered as the global instance, unregister it -- afterwards
        it should be possible to create a new instance and set as global to start
        a new debug session.

        :param bool wait:
            If True we'll wait for the threads to be actually finished before proceeding
            (based on the available timeout).
            Note that this must be thread-safe and if one thread is waiting the other thread should
            also wait.
        '''
        try:
            back_frame = sys._getframe().f_back
            pydev_log.debug(
                'PyDB.dispose_and_kill_all_pydevd_threads (called from: File "%s", line %s, in %s)',
                back_frame.f_code.co_filename, back_frame.f_lineno, back_frame.f_code.co_name
            )
            back_frame = None
            with self._disposed_lock:
                disposed = self.pydb_disposed
                self.pydb_disposed = True

            if disposed:
                if wait:
                    pydev_log.debug("PyDB.dispose_and_kill_all_pydevd_threads (already disposed - wait)")
                    self.__wait_for_threads_to_finish(timeout)
                else:
                    pydev_log.debug("PyDB.dispose_and_kill_all_pydevd_threads (already disposed - no wait)")
                return

            pydev_log.debug("PyDB.dispose_and_kill_all_pydevd_threads (first call)")

            # Wait until a time when there are no commands being processed to kill the threads.
            started_at = time.time()
            while time.time() < started_at + timeout:
                with self._main_lock:
                    writer = self.writer
                    if writer is None or writer.empty():
                        pydev_log.debug("PyDB.dispose_and_kill_all_pydevd_threads no commands being processed.")
                        break
            else:
                pydev_log.debug("PyDB.dispose_and_kill_all_pydevd_threads timed out waiting for writer to be empty.")

            pydb_daemon_threads = set(self.created_pydb_daemon_threads)
            for t in pydb_daemon_threads:
                if hasattr(t, 'do_kill_pydev_thread'):
                    pydev_log.debug("PyDB.dispose_and_kill_all_pydevd_threads killing thread: %s", t)
                    t.do_kill_pydev_thread()

            if wait:
                self.__wait_for_threads_to_finish(timeout)
            else:
                pydev_log.debug("PyDB.dispose_and_kill_all_pydevd_threads: no wait")

            py_db = get_global_debugger()
            if py_db is self:
                set_global_debugger(None)
        except:
            pydev_log.debug("PyDB.dispose_and_kill_all_pydevd_threads: exception")
            try:
                if DebugInfoHolder.DEBUG_TRACE_LEVEL >= 3:
                    pydev_log.exception()
            except:
                pass
        finally:
            pydev_log.debug("PyDB.dispose_and_kill_all_pydevd_threads: finished")

    def prepare_to_run(self):
        ''' Shared code to prepare debugging by installing traces and registering threads '''
        self.patch_threads()
        self.start_auxiliary_daemon_threads()

    def patch_threads(self):
        try:
            # not available in jython!
            threading.settrace(self.trace_dispatch)  # for all future threads
        except:
            pass

        from _pydev_bundle.pydev_monkey import patch_thread_modules
        patch_thread_modules()

    def run(self, file, globals=None, locals=None, is_module=False, set_trace=True):
        module_name = None
        entry_point_fn = ''
        if is_module:
            # When launching with `python -m <module>`, python automatically adds
            # an empty path to the PYTHONPATH which resolves files in the current
            # directory, so, depending how pydevd itself is launched, we may need
            # to manually add such an entry to properly resolve modules in the
            # current directory (see: https://github.com/Microsoft/ptvsd/issues/1010).
            if '' not in sys.path:
                sys.path.insert(0, '')
            file, _, entry_point_fn = file.partition(':')
            module_name = file
            filename = get_fullname(file)
            if filename is None:
                mod_dir = get_package_dir(module_name)
                if mod_dir is None:
                    sys.stderr.write("No module named %s\n" % file)
                    return
                else:
                    filename = get_fullname("%s.__main__" % module_name)
                    if filename is None:
                        sys.stderr.write("No module named %s\n" % file)
                        return
                    else:
                        file = filename
            else:
                file = filename
                mod_dir = os.path.dirname(filename)
                main_py = os.path.join(mod_dir, '__main__.py')
                main_pyc = os.path.join(mod_dir, '__main__.pyc')
                if filename.endswith('__init__.pyc'):
                    if os.path.exists(main_pyc):
                        filename = main_pyc
                    elif os.path.exists(main_py):
                        filename = main_py
                elif filename.endswith('__init__.py'):
                    if os.path.exists(main_pyc) and not os.path.exists(main_py):
                        filename = main_pyc
                    elif os.path.exists(main_py):
                        filename = main_py

            sys.argv[0] = filename

        if os.path.isdir(file):
            new_target = os.path.join(file, '__main__.py')
            if os.path.isfile(new_target):
                file = new_target

        m = None
        if globals is None:
            m = save_main_module(file, 'pydevd')
            globals = m.__dict__
            try:
                globals['__builtins__'] = __builtins__
            except NameError:
                pass  # Not there on Jython...

        if locals is None:
            locals = globals

        # Predefined (writable) attributes: __name__ is the module's name;
        # __doc__ is the module's documentation string, or None if unavailable;
        # __file__ is the pathname of the file from which the module was loaded,
        # if it was loaded from a file. The __file__ attribute is not present for
        # C modules that are statically linked into the interpreter; for extension modules
        # loaded dynamically from a shared library, it is the pathname of the shared library file.

        # I think this is an ugly hack, bug it works (seems to) for the bug that says that sys.path should be the same in
        # debug and run.
        if sys.path[0] != '' and m is not None and m.__file__.startswith(sys.path[0]):
            # print >> sys.stderr, 'Deleting: ', sys.path[0]
            del sys.path[0]

        if not is_module:
            # now, the local directory has to be added to the pythonpath
            # sys.path.insert(0, os.getcwd())
            # Changed: it's not the local directory, but the directory of the file launched
            # The file being run must be in the pythonpath (even if it was not before)
            sys.path.insert(0, os.path.split(os_path_abspath(file))[0])

        if set_trace:
            self.wait_for_ready_to_run()

            # call prepare_to_run when we already have all information about breakpoints
            self.prepare_to_run()

        t = threadingCurrentThread()
        thread_id = get_current_thread_id(t)

        if self.thread_analyser is not None:
            wrap_threads()
            self.thread_analyser.set_start_time(cur_time())
            send_concurrency_message("threading_event", 0, t.name, thread_id, "thread", "start", file, 1, None, parent=thread_id)

        if self.asyncio_analyser is not None:
            # we don't have main thread in asyncio graph, so we should add a fake event
            send_concurrency_message("asyncio_event", 0, "Task", "Task", "thread", "stop", file, 1, frame=None, parent=None)

        try:
            if INTERACTIVE_MODE_AVAILABLE:
                self.init_gui_support()
        except:
            pydev_log.exception("Matplotlib support in debugger failed")

        if hasattr(sys, 'exc_clear'):
            # we should clean exception information in Python 2, before user's code execution
            sys.exc_clear()

        # Notify that the main thread is created.
        self.notify_thread_created(thread_id, t)

        # Note: important: set the tracing right before calling _exec.
        if set_trace:
            self.enable_tracing()

        return self._exec(is_module, entry_point_fn, module_name, file, globals, locals)

    def _exec(self, is_module, entry_point_fn, module_name, file, globals, locals):
        '''
        This function should have frames tracked by unhandled exceptions (the `_exec` name is important).
        '''
        if not is_module:
            globals = pydevd_runpy.run_path(file, globals, '__main__')
        else:
            # treat ':' as a separator between module and entry point function
            # if there is no entry point we run we same as with -m switch. Otherwise we perform
            # an import and execute the entry point
            if entry_point_fn:
                mod = __import__(module_name, level=0, fromlist=[entry_point_fn], globals=globals, locals=locals)
                func = getattr(mod, entry_point_fn)
                func()
            else:
                # Run with the -m switch
                globals = pydevd_runpy._run_module_as_main(module_name, alter_argv=False)
        return globals

    def wait_for_commands(self, globals):
        self._activate_gui_if_needed()

        thread = threading.current_thread()
        from _pydevd_bundle import pydevd_frame_utils
        frame = pydevd_frame_utils.Frame(None, -1, pydevd_frame_utils.FCode("Console",
                                                                            os.path.abspath(os.path.dirname(__file__))), globals, globals)
        thread_id = get_current_thread_id(thread)
        self.add_fake_frame(thread_id, id(frame), frame)

        cmd = self.cmd_factory.make_show_console_message(self, thread_id, frame)
        if self.writer is not None:
            self.writer.add_command(cmd)

        while True:
            if self.gui_in_use:
                # call input hooks if only GUI is in use
                self._call_input_hook()
            self.process_internal_commands()
            time.sleep(0.01)


class IDAPMessagesListener(object):

    def before_send(self, message_as_dict):
        '''
        Called just before a message is sent to the IDE.

        :type message_as_dict: dict
        '''

    def after_receive(self, message_as_dict):
        '''
        Called just after a message is received from the IDE.

        :type message_as_dict: dict
        '''


def add_dap_messages_listener(dap_messages_listener):
    '''
    Adds a listener for the DAP (debug adapter protocol) messages.

    :type dap_messages_listener: IDAPMessagesListener

    :note: messages from the xml backend are not notified through this API.

    :note: the notifications are sent from threads and they are not synchronized (so,
    it's possible that a message is sent and received from different threads at the same time).
    '''
    py_db = get_global_debugger()
    if py_db is None:
        raise AssertionError('PyDB is still not setup.')

    py_db.add_dap_messages_listener(dap_messages_listener)


def send_json_message(msg):
    '''
    API to send some custom json message.

    :param dict|pydevd_schema.BaseSchema msg:
        The custom message to be sent.

    :return bool:
        True if the message was added to the queue to be sent and False otherwise.
    '''
    py_db = get_global_debugger()
    if py_db is None:
        return False

    writer = py_db.writer
    if writer is None:
        return False

    cmd = NetCommand(-1, 0, msg, is_json=True)
    writer.add_command(cmd)
    return True


def enable_qt_support(qt_support_mode):
    from _pydev_bundle import pydev_monkey_qt
    pydev_monkey_qt.patch_qt(qt_support_mode)


def start_dump_threads_thread(filename_template, timeout, recurrent):
    '''
    Helper to dump threads after a timeout.

    :param filename_template:
        A template filename, such as 'c:/temp/thread_dump_%s.txt', where the %s will
        be replaced by the time for the dump.
    :param timeout:
        The timeout (in seconds) for the dump.
    :param recurrent:
        If True we'll keep on doing thread dumps.
    '''
    assert filename_template.count('%s') == 1, \
        'Expected one %%s to appear in: %s' % (filename_template,)

    def _threads_on_timeout():
        try:
            while True:
                time.sleep(timeout)
                filename = filename_template % (time.time(),)
                try:
                    os.makedirs(os.path.dirname(filename))
                except Exception:
                    pass
                with open(filename, 'w') as stream:
                    dump_threads(stream)
                if not recurrent:
                    return
        except Exception:
            pydev_log.exception()

    t = threading.Thread(target=_threads_on_timeout)
    mark_as_pydevd_daemon_thread(t)
    t.start()


def dump_threads(stream=None):
    '''
    Helper to dump thread info (default is printing to stderr).
    '''
    pydevd_utils.dump_threads(stream)


def usage(doExit=0):
    sys.stdout.write('Usage:\n')
    sys.stdout.write('pydevd.py --port N [(--client hostname) | --server] --file executable [file_options]\n')
    if doExit:
        sys.exit(0)


def _init_stdout_redirect():
    pydevd_io.redirect_stream_to_pydb_io_messages(std='stdout')


def _init_stderr_redirect():
    pydevd_io.redirect_stream_to_pydb_io_messages(std='stderr')


def _enable_attach(
    address,
    dont_trace_start_patterns=(),
    dont_trace_end_patterns=(),
    patch_multiprocessing=False,
    access_token=None,
    client_access_token=None,
    ):
    '''
    Starts accepting connections at the given host/port. The debugger will not be initialized nor
    configured, it'll only start accepting connections (and will have the tracing setup in this
    thread).

    Meant to be used with the DAP (Debug Adapter Protocol) with _wait_for_attach().

    :param address: (host, port)
    :type address: tuple(str, int)
    '''
    host = address[0]
    port = int(address[1])

    if SetupHolder.setup is not None:
        if port != SetupHolder.setup['port']:
            raise AssertionError('Unable to listen in port: %s (already listening in port: %s)' % (port, SetupHolder.setup['port']))
    settrace(
        host=host,
        port=port,
        suspend=False,
        wait_for_ready_to_run=False,
        block_until_connected=False,
        dont_trace_start_patterns=dont_trace_start_patterns,
        dont_trace_end_patterns=dont_trace_end_patterns,
        patch_multiprocessing=patch_multiprocessing,
        access_token=access_token,
        client_access_token=client_access_token,
    )

    py_db = get_global_debugger()
    py_db.wait_for_server_socket_ready()
    return py_db._server_socket_name


def _wait_for_attach(cancel=None):
    '''
    Meant to be called after _enable_attach() -- the current thread will only unblock after a
    connection is in place and the DAP (Debug Adapter Protocol) sends the ConfigurationDone
    request.
    '''
    py_db = get_global_debugger()
    if py_db is None:
        raise AssertionError('Debugger still not created. Please use _enable_attach() before using _wait_for_attach().')

    py_db.block_until_configuration_done(cancel=cancel)


def _is_attached():
    '''
    Can be called any time to check if the connection was established and the DAP (Debug Adapter Protocol) has sent
    the ConfigurationDone request.
    '''
    py_db = get_global_debugger()
    return (py_db is not None) and py_db.is_attached()


#=======================================================================================================================
# settrace
#=======================================================================================================================
def settrace(
    host=None,
    stdout_to_server=False,
    stderr_to_server=False,
    port=5678,
    suspend=True,
    trace_only_current_thread=False,
    overwrite_prev_trace=False,
    patch_multiprocessing=False,
    stop_at_frame=None,
    block_until_connected=True,
    wait_for_ready_to_run=True,
    dont_trace_start_patterns=(),
    dont_trace_end_patterns=(),
    access_token=None,
    client_access_token=None,
    notify_stdin=True,
    **kwargs
    ):
    '''Sets the tracing function with the pydev debug function and initializes needed facilities.

    :param host: the user may specify another host, if the debug server is not in the same machine (default is the local
        host)

    :param stdout_to_server: when this is true, the stdout is passed to the debug server

    :param stderr_to_server: when this is true, the stderr is passed to the debug server
        so that they are printed in its console and not in this process console.

    :param port: specifies which port to use for communicating with the server (note that the server must be started
        in the same port). @note: currently it's hard-coded at 5678 in the client

    :param suspend: whether a breakpoint should be emulated as soon as this function is called.

    :param trace_only_current_thread: determines if only the current thread will be traced or all current and future
        threads will also have the tracing enabled.

    :param overwrite_prev_trace: deprecated

    :param patch_multiprocessing: if True we'll patch the functions which create new processes so that launched
        processes are debugged.

    :param stop_at_frame: if passed it'll stop at the given frame, otherwise it'll stop in the function which
        called this method.

    :param wait_for_ready_to_run: if True settrace will block until the ready_to_run flag is set to True,
        otherwise, it'll set ready_to_run to True and this function won't block.

        Note that if wait_for_ready_to_run == False, there are no guarantees that the debugger is synchronized
        with what's configured in the client (IDE), the only guarantee is that when leaving this function
        the debugger will be already connected.

    :param dont_trace_start_patterns: if set, then any path that starts with one fo the patterns in the collection
        will not be traced

    :param dont_trace_end_patterns: if set, then any path that ends with one fo the patterns in the collection
        will not be traced

    :param access_token: token to be sent from the client (i.e.: IDE) to the debugger when a connection
        is established (verified by the debugger).

    :param client_access_token: token to be sent from the debugger to the client (i.e.: IDE) when
        a connection is established (verified by the client).

    :param notify_stdin:
        If True sys.stdin will be patched to notify the client when a message is requested
        from the IDE. This is done so that when reading the stdin the client is notified.
        Clients may need this to know when something that is being written should be interpreted
        as an input to the process or as a command to be evaluated.
        Note that parallel-python has issues with this (because it tries to assert that sys.stdin
        is of a given type instead of just checking that it has what it needs).
    '''

    stdout_to_server = stdout_to_server or kwargs.get('stdoutToServer', False)  # Backward compatibility
    stderr_to_server = stderr_to_server or kwargs.get('stderrToServer', False)  # Backward compatibility

    # Internal use (may be used to set the setup info directly for subprocesess).
    __setup_holder__ = kwargs.get('__setup_holder__')

    with _set_trace_lock:
        _locked_settrace(
            host,
            stdout_to_server,
            stderr_to_server,
            port,
            suspend,
            trace_only_current_thread,
            patch_multiprocessing,
            stop_at_frame,
            block_until_connected,
            wait_for_ready_to_run,
            dont_trace_start_patterns,
            dont_trace_end_patterns,
            access_token,
            client_access_token,
            __setup_holder__=__setup_holder__,
            notify_stdin=notify_stdin,
        )


_set_trace_lock = ForkSafeLock()


def _locked_settrace(
    host,
    stdout_to_server,
    stderr_to_server,
    port,
    suspend,
    trace_only_current_thread,
    patch_multiprocessing,
    stop_at_frame,
    block_until_connected,
    wait_for_ready_to_run,
    dont_trace_start_patterns,
    dont_trace_end_patterns,
    access_token,
    client_access_token,
    __setup_holder__,
    notify_stdin,
    ):
    if patch_multiprocessing:
        try:
            from _pydev_bundle import pydev_monkey
        except:
            pass
        else:
            pydev_monkey.patch_new_process_functions()

    if host is None:
        from _pydev_bundle import pydev_localhost
        host = pydev_localhost.get_localhost()

    global _global_redirect_stdout_to_server
    global _global_redirect_stderr_to_server

    py_db = get_global_debugger()
    if __setup_holder__:
        SetupHolder.setup = __setup_holder__
    if py_db is None:
        py_db = PyDB()
        pydevd_vm_type.setup_type()

        if SetupHolder.setup is None:
            setup = {
                'client': host,  # dispatch expects client to be set to the host address when server is False
                'server': False,
                'port': int(port),
                'multiprocess': patch_multiprocessing,
                'skip-notify-stdin': not notify_stdin,
            }
            SetupHolder.setup = setup

        if access_token is not None:
            py_db.authentication.access_token = access_token
            SetupHolder.setup['access-token'] = access_token
        if client_access_token is not None:
            py_db.authentication.client_access_token = client_access_token
            SetupHolder.setup['client-access-token'] = client_access_token

        if block_until_connected:
            py_db.connect(host, port)  # Note: connect can raise error.
        else:
            # Create a dummy writer and wait for the real connection.
            py_db.writer = WriterThread(NULL, py_db, terminate_on_socket_close=False)
            py_db.create_wait_for_connection_thread()

        if dont_trace_start_patterns or dont_trace_end_patterns:
            PyDevdAPI().set_dont_trace_start_end_patterns(py_db, dont_trace_start_patterns, dont_trace_end_patterns)

        _global_redirect_stdout_to_server = stdout_to_server
        _global_redirect_stderr_to_server = stderr_to_server

        if _global_redirect_stdout_to_server:
            _init_stdout_redirect()

        if _global_redirect_stderr_to_server:
            _init_stderr_redirect()

        if notify_stdin:
            patch_stdin()

        t = threadingCurrentThread()
        additional_info = set_additional_thread_info(t)

        if not wait_for_ready_to_run:
            py_db.ready_to_run = True

        py_db.wait_for_ready_to_run()
        py_db.start_auxiliary_daemon_threads()

        try:
            if INTERACTIVE_MODE_AVAILABLE:
                py_db.init_gui_support()
        except:
            pydev_log.exception("Matplotlib support in debugger failed")

        if trace_only_current_thread:
            py_db.enable_tracing()
        else:
            # Trace future threads.
            py_db.patch_threads()

            py_db.enable_tracing(py_db.trace_dispatch, apply_to_all_threads=True)

            # As this is the first connection, also set tracing for any untraced threads
            py_db.set_tracing_for_untraced_contexts()

        py_db.set_trace_for_frame_and_parents(get_frame().f_back)

        with CustomFramesContainer.custom_frames_lock:  # @UndefinedVariable
            for _frameId, custom_frame in CustomFramesContainer.custom_frames.items():
                py_db.set_trace_for_frame_and_parents(custom_frame.frame)

    else:
        # ok, we're already in debug mode, with all set, so, let's just set the break
        if access_token is not None:
            py_db.authentication.access_token = access_token
        if client_access_token is not None:
            py_db.authentication.client_access_token = client_access_token

        py_db.set_trace_for_frame_and_parents(get_frame().f_back)

        t = threadingCurrentThread()
        additional_info = set_additional_thread_info(t)

        if trace_only_current_thread:
            py_db.enable_tracing()
        else:
            # Trace future threads.
            py_db.patch_threads()
            py_db.enable_tracing(py_db.trace_dispatch, apply_to_all_threads=True)

    # Suspend as the last thing after all tracing is in place.
    if suspend:
        if stop_at_frame is not None:
            # If the step was set we have to go to run state and
            # set the proper frame for it to stop.
            additional_info.pydev_state = STATE_RUN
            additional_info.pydev_original_step_cmd = CMD_STEP_OVER
            additional_info.pydev_step_cmd = CMD_STEP_OVER
            additional_info.pydev_step_stop = stop_at_frame
            additional_info.suspend_type = PYTHON_SUSPEND
        else:
            # Ask to break as soon as possible.
            py_db.set_suspend(t, CMD_SET_BREAK)


def stoptrace():
    pydev_log.debug("pydevd.stoptrace()")
    pydevd_tracing.restore_sys_set_trace_func()
    sys.settrace(None)
    try:
        # not available in jython!
        threading.settrace(None)  # for all future threads
    except:
        pass

    from _pydev_bundle.pydev_monkey import undo_patch_thread_modules
    undo_patch_thread_modules()

    # Either or both standard streams can be closed at this point,
    # in which case flush() will fail.
    try:
        sys.stdout.flush()
    except:
        pass
    try:
        sys.stderr.flush()
    except:
        pass

    py_db = get_global_debugger()

    if py_db is not None:
        py_db.dispose_and_kill_all_pydevd_threads()


class Dispatcher(object):

    def __init__(self):
        self.port = None

    def connect(self, host, port):
        self.host = host
        self.port = port
        self.client = start_client(self.host, self.port)
        self.reader = DispatchReader(self)
        self.reader.pydev_do_not_trace = False  # we run reader in the same thread so we don't want to loose tracing
        self.reader.run()

    def close(self):
        try:
            self.reader.do_kill_pydev_thread()
        except:
            pass


class DispatchReader(ReaderThread):

    def __init__(self, dispatcher):
        self.dispatcher = dispatcher

        ReaderThread.__init__(
            self,
            get_global_debugger(),
            self.dispatcher.client,
            PyDevJsonCommandProcessor=PyDevJsonCommandProcessor,
            process_net_command=process_net_command,
        )

    @overrides(ReaderThread._on_run)
    def _on_run(self):
        dummy_thread = threading.current_thread()
        dummy_thread.is_pydev_daemon_thread = False
        return ReaderThread._on_run(self)

    @overrides(PyDBDaemonThread.do_kill_pydev_thread)
    def do_kill_pydev_thread(self):
        if not self._kill_received:
            ReaderThread.do_kill_pydev_thread(self)
            try:
                self.sock.shutdown(SHUT_RDWR)
            except:
                pass
            try:
                self.sock.close()
            except:
                pass

    def process_command(self, cmd_id, seq, text):
        if cmd_id == 99:
            self.dispatcher.port = int(text)
            self._kill_received = True


DISPATCH_APPROACH_NEW_CONNECTION = 1  # Used by PyDev
DISPATCH_APPROACH_EXISTING_CONNECTION = 2  # Used by PyCharm
DISPATCH_APPROACH = DISPATCH_APPROACH_NEW_CONNECTION


def dispatch():
    setup = SetupHolder.setup
    host = setup['client']
    port = setup['port']
    if DISPATCH_APPROACH == DISPATCH_APPROACH_EXISTING_CONNECTION:
        dispatcher = Dispatcher()
        try:
            dispatcher.connect(host, port)
            port = dispatcher.port
        finally:
            dispatcher.close()
    return host, port


def settrace_forked(setup_tracing=True):
    '''
    When creating a fork from a process in the debugger, we need to reset the whole debugger environment!
    '''
    from _pydevd_bundle.pydevd_constants import GlobalDebuggerHolder
    py_db = GlobalDebuggerHolder.global_dbg
    if py_db is not None:
        py_db.created_pydb_daemon_threads = {}  # Just making sure we won't touch those (paused) threads.
        py_db = None

    GlobalDebuggerHolder.global_dbg = None
    threading.current_thread().additional_info = None

    # Make sure that we keep the same access tokens for subprocesses started through fork.
    setup = SetupHolder.setup
    if setup is None:
        setup = {}
    else:
        # i.e.: Get the ppid at this point as it just changed.
        # If we later do an exec() it should remain the same ppid.
        setup[pydevd_constants.ARGUMENT_PPID] = PyDevdAPI().get_ppid()
    access_token = setup.get('access-token')
    client_access_token = setup.get('client-access-token')

    if setup_tracing:
        from _pydevd_frame_eval.pydevd_frame_eval_main import clear_thread_local_info
        host, port = dispatch()

    import pydevd_tracing
    pydevd_tracing.restore_sys_set_trace_func()

    if setup_tracing:
        if port is not None:
            custom_frames_container_init()

            if clear_thread_local_info is not None:
                clear_thread_local_info()

            settrace(
                    host,
                    port=port,
                    suspend=False,
                    trace_only_current_thread=False,
                    overwrite_prev_trace=True,
                    patch_multiprocessing=True,
                    access_token=access_token,
                    client_access_token=client_access_token,
            )


@contextmanager
def skip_subprocess_arg_patch():
    '''
    May be used to skip the monkey-patching that pydevd does to
    skip changing arguments to embed the debugger into child processes.

    i.e.:

    with pydevd.skip_subprocess_arg_patch():
        subprocess.call(...)
    '''
    from _pydev_bundle import pydev_monkey
    with pydev_monkey.skip_subprocess_arg_patch():
        yield


def add_dont_terminate_child_pid(pid):
    '''
    May be used to ask pydevd to skip the termination of some process
    when it's asked to terminate (debug adapter protocol only).

    :param int pid:
        The pid to be ignored.

    i.e.:

    process = subprocess.Popen(...)
    pydevd.add_dont_terminate_child_pid(process.pid)
    '''
    py_db = get_global_debugger()
    if py_db is not None:
        py_db.dont_terminate_child_pids.add(pid)


class SetupHolder:

    setup = None


def apply_debugger_options(setup_options):
    """

    :type setup_options: dict[str, bool]
    """
    default_options = {'save-signatures': False, 'qt-support': ''}
    default_options.update(setup_options)
    setup_options = default_options

    debugger = get_global_debugger()
    if setup_options['save-signatures']:
        if pydevd_vm_type.get_vm_type() == pydevd_vm_type.PydevdVmType.JYTHON:
            sys.stderr.write("Collecting run-time type information is not supported for Jython\n")
        else:
            # Only import it if we're going to use it!
            from _pydevd_bundle.pydevd_signature import SignatureFactory
            debugger.signature_factory = SignatureFactory()

    if setup_options['qt-support']:
        enable_qt_support(setup_options['qt-support'])


@call_only_once
def patch_stdin():
    _internal_patch_stdin(None, sys, getpass_mod)


def _internal_patch_stdin(py_db=None, sys=None, getpass_mod=None):
    '''
    Note: don't use this function directly, use `patch_stdin()` instead.
    (this function is only meant to be used on test-cases to avoid patching the actual globals).
    '''
    # Patch stdin so that we notify when readline() is called.
    original_sys_stdin = sys.stdin
    debug_console_stdin = DebugConsoleStdIn(py_db, original_sys_stdin)
    sys.stdin = debug_console_stdin

    _original_getpass = getpass_mod.getpass

    @functools.wraps(_original_getpass)
    def getpass(*args, **kwargs):
        with DebugConsoleStdIn.notify_input_requested(debug_console_stdin):
            try:
                curr_stdin = sys.stdin
                if curr_stdin is debug_console_stdin:
                    sys.stdin = original_sys_stdin
                return _original_getpass(*args, **kwargs)
            finally:
                sys.stdin = curr_stdin

    getpass_mod.getpass = getpass

# Dispatch on_debugger_modules_loaded here, after all primary py_db modules are loaded


for handler in pydevd_extension_utils.extensions_of_type(DebuggerEventHandler):
    handler.on_debugger_modules_loaded(debugger_version=__version__)


def log_to(log_file:str, log_level=3) -> None:
    '''
    In pydevd it's possible to log by setting the following environment variables:

    PYDEVD_DEBUG=1 (sets the default log level to 3 along with other default options)
    PYDEVD_DEBUG_FILE=</path/to/file.log>

    Note that the file will have the pid of the process added to it (so, logging to
    /path/to/file.log would actually start logging to /path/to/file.<pid>.log -- if subprocesses are
    logged, each new subprocess will have the logging set to its own pid).

    Usually setting the environment variable is preferred as it'd log information while
    pydevd is still doing its imports and not just after this method is called, but on
    cases where this is hard to do this function may be called to set the tracing after
    pydevd itself is already imported.
    '''
    pydev_log.log_to(log_file, log_level)


def _log_initial_info():
    pydev_log.debug("Initial arguments: %s", (sys.argv,))
    pydev_log.debug("Current pid: %s", os.getpid())
    pydev_log.debug("Using cython: %s", USING_CYTHON)
    pydev_log.debug("Using frame eval: %s", USING_FRAME_EVAL)
    pydev_log.debug("Using gevent mode: %s / imported gevent module support: %s", SUPPORT_GEVENT, bool(pydevd_gevent_integration))


def config(protocol='', debug_mode='', preimport=''):
    pydev_log.debug('Config: protocol: %s, debug_mode: %s, preimport: %s', protocol, debug_mode, preimport)
    PydevdCustomization.DEFAULT_PROTOCOL = protocol
    PydevdCustomization.DEBUG_MODE = debug_mode
    PydevdCustomization.PREIMPORT = preimport


#=======================================================================================================================
# main
#=======================================================================================================================
def main():

    # parse the command line. --file is our last argument that is required
    _log_initial_info()
    original_argv = sys.argv[:]
    try:
        from _pydevd_bundle.pydevd_command_line_handling import process_command_line
        setup = process_command_line(sys.argv)
        SetupHolder.setup = setup
    except ValueError:
        pydev_log.exception()
        usage(1)

    preimport = setup.get('preimport')
    if preimport:
        pydevd_defaults.PydevdCustomization.PREIMPORT = preimport

    debug_mode = setup.get('debug-mode')
    if debug_mode:
        pydevd_defaults.PydevdCustomization.DEBUG_MODE = debug_mode

    log_trace_level = setup.get('log-level')

    # Note: the logging info could've been changed (this would happen if this is a
    # subprocess and the value in the environment variable does not match the value in the
    # argument because the user used `pydevd.log_to` instead of supplying the environment
    # variable). If this is the case, update the logging info and re-log some information
    # in the new target.
    new_debug_file = setup.get('log-file')
    if new_debug_file and DebugInfoHolder.PYDEVD_DEBUG_FILE != new_debug_file:
        # The debug file can't be set directly, we need to use log_to() so that the a
        # new stream is actually created for the new file.
        log_to(new_debug_file, log_trace_level if log_trace_level is not None else 3)
        _log_initial_info()  # The redirection info just changed, log it again.

    elif log_trace_level is not None:
        # The log file was not specified
        DebugInfoHolder.DEBUG_TRACE_LEVEL = log_trace_level
    pydev_log.debug('Original sys.argv: %s', original_argv)

    if setup['print-in-debugger-startup']:
        try:
            pid = ' (pid: %s)' % os.getpid()
        except:
            pid = ''
        sys.stderr.write("pydev debugger: starting%s\n" % pid)

    pydev_log.debug("Executing file %s", setup['file'])
    pydev_log.debug("arguments: %s", (sys.argv,))

    pydevd_vm_type.setup_type(setup.get('vm_type', None))

    port = setup['port']
    host = setup['client']
    f = setup['file']
    fix_app_engine_debug = False

    debugger = get_global_debugger()
    if debugger is None:
        debugger = PyDB()

    try:
        from _pydev_bundle import pydev_monkey
    except:
        pass  # Not usable on jython 2.1
    else:
        if setup['multiprocess']:  # PyDev
            pydev_monkey.patch_new_process_functions()

        elif setup['multiproc']:  # PyCharm
            pydev_log.debug("Started in multiproc mode\n")
            global DISPATCH_APPROACH
            DISPATCH_APPROACH = DISPATCH_APPROACH_EXISTING_CONNECTION

            dispatcher = Dispatcher()
            try:
                dispatcher.connect(host, port)
                if dispatcher.port is not None:
                    port = dispatcher.port
                    pydev_log.debug("Received port %d\n", port)
                    pydev_log.info("pydev debugger: process %d is connecting\n" % os.getpid())

                    try:
                        pydev_monkey.patch_new_process_functions()
                    except:
                        pydev_log.exception("Error patching process functions.")
                else:
                    pydev_log.critical("pydev debugger: couldn't get port for new debug process.")
            finally:
                dispatcher.close()
        else:
            try:
                pydev_monkey.patch_new_process_functions_with_warning()
            except:
                pydev_log.exception("Error patching process functions.")

            # Only do this patching if we're not running with multiprocess turned on.
            if f.find('dev_appserver.py') != -1:
                if os.path.basename(f).startswith('dev_appserver.py'):
                    appserver_dir = os.path.dirname(f)
                    version_file = os.path.join(appserver_dir, 'VERSION')
                    if os.path.exists(version_file):
                        try:
                            stream = open(version_file, 'r')
                            try:
                                for line in stream.read().splitlines():
                                    line = line.strip()
                                    if line.startswith('release:'):
                                        line = line[8:].strip()
                                        version = line.replace('"', '')
                                        version = version.split('.')
                                        if int(version[0]) > 1:
                                            fix_app_engine_debug = True

                                        elif int(version[0]) == 1:
                                            if int(version[1]) >= 7:
                                                # Only fix from 1.7 onwards
                                                fix_app_engine_debug = True
                                        break
                            finally:
                                stream.close()
                        except:
                            pydev_log.exception()

    try:
        # In the default run (i.e.: run directly on debug mode), we try to patch stackless as soon as possible
        # on a run where we have a remote debug, we may have to be more careful because patching stackless means
        # that if the user already had a stackless.set_schedule_callback installed, he'd loose it and would need
        # to call it again (because stackless provides no way of getting the last function which was registered
        # in set_schedule_callback).
        #
        # So, ideally, if there's an application using stackless and the application wants to use the remote debugger
        # and benefit from stackless debugging, the application itself must call:
        #
        # import pydevd_stackless
        # pydevd_stackless.patch_stackless()
        #
        # itself to be able to benefit from seeing the tasklets created before the remote debugger is attached.
        from _pydevd_bundle import pydevd_stackless
        pydevd_stackless.patch_stackless()
    except:
        # It's ok not having stackless there...
        try:
            if hasattr(sys, 'exc_clear'):
                sys.exc_clear()  # the exception information should be cleaned in Python 2
        except:
            pass

    is_module = setup['module']
    if not setup['skip-notify-stdin']:
        patch_stdin()

    if setup[pydevd_constants.ARGUMENT_JSON_PROTOCOL]:
        PyDevdAPI().set_protocol(debugger, 0, JSON_PROTOCOL)

    elif setup[pydevd_constants.ARGUMENT_HTTP_JSON_PROTOCOL]:
        PyDevdAPI().set_protocol(debugger, 0, HTTP_JSON_PROTOCOL)

    elif setup[pydevd_constants.ARGUMENT_HTTP_PROTOCOL]:
        PyDevdAPI().set_protocol(debugger, 0, pydevd_constants.HTTP_PROTOCOL)

    elif setup[pydevd_constants.ARGUMENT_QUOTED_LINE_PROTOCOL]:
        PyDevdAPI().set_protocol(debugger, 0, pydevd_constants.QUOTED_LINE_PROTOCOL)

    access_token = setup['access-token']
    if access_token:
        debugger.authentication.access_token = access_token

    client_access_token = setup['client-access-token']
    if client_access_token:
        debugger.authentication.client_access_token = client_access_token

    if fix_app_engine_debug:
        sys.stderr.write("pydev debugger: google app engine integration enabled\n")
        curr_dir = os.path.dirname(__file__)
        app_engine_startup_file = os.path.join(curr_dir, 'pydev_app_engine_debug_startup.py')

        sys.argv.insert(1, '--python_startup_script=' + app_engine_startup_file)
        import json
        setup['pydevd'] = __file__
        sys.argv.insert(2, '--python_startup_args=%s' % json.dumps(setup),)
        sys.argv.insert(3, '--automatic_restart=no')
        sys.argv.insert(4, '--max_module_instances=1')

        # Run the dev_appserver
        debugger.run(setup['file'], None, None, is_module, set_trace=False)
    else:
        if setup['save-threading']:
            debugger.thread_analyser = ThreadingLogger()
        if setup['save-asyncio']:
            debugger.asyncio_analyser = AsyncioLogger()

        apply_debugger_options(setup)

        try:
            debugger.connect(host, port)
        except:
            sys.stderr.write("Could not connect to %s: %s\n" % (host, port))
            pydev_log.exception()
            sys.exit(1)

        globals = debugger.run(setup['file'], None, None, is_module)

        if setup['cmd-line']:
            debugger.wait_for_commands(globals)


if __name__ == '__main__':
    main()
