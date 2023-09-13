from _pydev_bundle._pydev_saved_modules import threading
from _pydev_bundle import _pydev_saved_modules
from _pydevd_bundle.pydevd_utils import notify_about_gevent_if_needed
import weakref
from _pydevd_bundle.pydevd_constants import IS_JYTHON, IS_IRONPYTHON, \
    PYDEVD_APPLY_PATCHING_TO_HIDE_PYDEVD_THREADS
from _pydev_bundle.pydev_log import exception as pydev_log_exception
import sys
from _pydev_bundle import pydev_log
import pydevd_tracing
from _pydevd_bundle.pydevd_collect_bytecode_info import iter_instructions

if IS_JYTHON:
    import org.python.core as JyCore  # @UnresolvedImport


class PyDBDaemonThread(threading.Thread):

    def __init__(self, py_db, target_and_args=None):
        '''
        :param target_and_args:
            tuple(func, args, kwargs) if this should be a function and args to run.
            -- Note: use through run_as_pydevd_daemon_thread().
        '''
        threading.Thread.__init__(self)
        notify_about_gevent_if_needed()
        self._py_db = weakref.ref(py_db)
        self._kill_received = False
        mark_as_pydevd_daemon_thread(self)
        self._target_and_args = target_and_args

    @property
    def py_db(self):
        return self._py_db()

    def run(self):
        created_pydb_daemon = self.py_db.created_pydb_daemon_threads
        created_pydb_daemon[self] = 1
        try:
            try:
                if IS_JYTHON and not isinstance(threading.current_thread(), threading._MainThread):
                    # we shouldn't update sys.modules for the main thread, cause it leads to the second importing 'threading'
                    # module, and the new instance of main thread is created
                    ss = JyCore.PySystemState()
                    # Note: Py.setSystemState() affects only the current thread.
                    JyCore.Py.setSystemState(ss)

                self._stop_trace()
                self._on_run()
            except:
                if sys is not None and pydev_log_exception is not None:
                    pydev_log_exception()
        finally:
            del created_pydb_daemon[self]

    def _on_run(self):
        if self._target_and_args is not None:
            target, args, kwargs = self._target_and_args
            target(*args, **kwargs)
        else:
            raise NotImplementedError('Should be reimplemented by: %s' % self.__class__)

    def do_kill_pydev_thread(self):
        if not self._kill_received:
            pydev_log.debug('%s received kill signal', self.name)
            self._kill_received = True

    def _stop_trace(self):
        if self.pydev_do_not_trace:
            pydevd_tracing.SetTrace(None)  # no debugging on this thread


def _collect_load_names(func):
    found_load_names = set()
    for instruction in iter_instructions(func.__code__):
        if instruction.opname in ('LOAD_GLOBAL', 'LOAD_ATTR', 'LOAD_METHOD'):
            found_load_names.add(instruction.argrepr)
    return found_load_names


def _patch_threading_to_hide_pydevd_threads():
    '''
    Patches the needed functions on the `threading` module so that the pydevd threads are hidden.

    Note that we patch the functions __code__ to avoid issues if some code had already imported those
    variables prior to the patching.
    '''
    found_load_names = _collect_load_names(threading.enumerate)
    # i.e.: we'll only apply the patching if the function seems to be what we expect.

    new_threading_enumerate = None

    if found_load_names in (
        {'_active_limbo_lock', '_limbo', '_active', 'values', 'list'},
        {'_active_limbo_lock', '_limbo', '_active', 'values', 'NULL + list'}
        ):
        pydev_log.debug('Applying patching to hide pydevd threads (Py3 version).')

        def new_threading_enumerate():
            with _active_limbo_lock:
                ret = list(_active.values()) + list(_limbo.values())

            return [t for t in ret if not getattr(t, 'is_pydev_daemon_thread', False)]

    elif found_load_names == set(('_active_limbo_lock', '_limbo', '_active', 'values')):
        pydev_log.debug('Applying patching to hide pydevd threads (Py2 version).')

        def new_threading_enumerate():
            with _active_limbo_lock:
                ret = _active.values() + _limbo.values()

            return [t for t in ret if not getattr(t, 'is_pydev_daemon_thread', False)]

    else:
        pydev_log.info('Unable to hide pydevd threads. Found names in threading.enumerate: %s', found_load_names)

    if new_threading_enumerate is not None:

        def pydevd_saved_threading_enumerate():
            with threading._active_limbo_lock:
                return list(threading._active.values()) + list(threading._limbo.values())

        _pydev_saved_modules.pydevd_saved_threading_enumerate = pydevd_saved_threading_enumerate

        threading.enumerate.__code__ = new_threading_enumerate.__code__

        # We also need to patch the active count (to match what we have in the enumerate).
        def new_active_count():
            # Note: as this will be executed in the `threading` module, `enumerate` will
            # actually be threading.enumerate.
            return len(enumerate())

        threading.active_count.__code__ = new_active_count.__code__

        # When shutting down, Python (on some versions) may do something as:
        #
        # def _pickSomeNonDaemonThread():
        #     for t in enumerate():
        #         if not t.daemon and t.is_alive():
        #             return t
        #     return None
        #
        # But in this particular case, we do want threads with `is_pydev_daemon_thread` to appear
        # explicitly due to the pydevd `CheckAliveThread` (because we want the shutdown to wait on it).
        # So, it can't rely on the `enumerate` for that anymore as it's patched to not return pydevd threads.
        if hasattr(threading, '_pickSomeNonDaemonThread'):

            def new_pick_some_non_daemon_thread():
                with _active_limbo_lock:
                    # Ok for py2 and py3.
                    threads = list(_active.values()) + list(_limbo.values())

                for t in threads:
                    if not t.daemon and t.is_alive():
                        return t
                return None

            threading._pickSomeNonDaemonThread.__code__ = new_pick_some_non_daemon_thread.__code__


_patched_threading_to_hide_pydevd_threads = False


def mark_as_pydevd_daemon_thread(thread):
    if not IS_JYTHON and not IS_IRONPYTHON and PYDEVD_APPLY_PATCHING_TO_HIDE_PYDEVD_THREADS:
        global _patched_threading_to_hide_pydevd_threads
        if not _patched_threading_to_hide_pydevd_threads:
            # When we mark the first thread as a pydevd daemon thread, we also change the threading
            # functions to hide pydevd threads.
            # Note: we don't just "hide" the pydevd threads from the threading module by not using it
            # (i.e.: just using the `thread.start_new_thread` instead of `threading.Thread`)
            # because there's 1 thread (the `CheckAliveThread`) which is a pydevd thread but
            # isn't really a daemon thread (so, we need CPython to wait on it for shutdown,
            # in which case it needs to be in `threading` and the patching would be needed anyways).
            _patched_threading_to_hide_pydevd_threads = True
            try:
                _patch_threading_to_hide_pydevd_threads()
            except:
                pydev_log.exception('Error applying patching to hide pydevd threads.')

    thread.pydev_do_not_trace = True
    thread.is_pydev_daemon_thread = True
    thread.daemon = True


def run_as_pydevd_daemon_thread(py_db, func, *args, **kwargs):
    '''
    Runs a function as a pydevd daemon thread (without any tracing in place).
    '''
    t = PyDBDaemonThread(py_db, target_and_args=(func, args, kwargs))
    t.name = '%s (pydevd daemon thread)' % (func.__name__,)
    t.start()
    return t
