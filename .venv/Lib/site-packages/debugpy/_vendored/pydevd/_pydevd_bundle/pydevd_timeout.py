from _pydev_bundle._pydev_saved_modules import threading
from _pydevd_bundle.pydevd_daemon_thread import PyDBDaemonThread
from _pydevd_bundle.pydevd_constants import thread_get_ident, IS_CPYTHON, NULL
import ctypes
import time
from _pydev_bundle import pydev_log
import weakref
from _pydevd_bundle.pydevd_utils import is_current_thread_main_thread
from _pydevd_bundle import pydevd_utils

_DEBUG = False  # Default should be False as this can be very verbose.


class _TimeoutThread(PyDBDaemonThread):
    '''
    The idea in this class is that it should be usually stopped waiting
    for the next event to be called (paused in a threading.Event.wait).

    When a new handle is added it sets the event so that it processes the handles and
    then keeps on waiting as needed again.

    This is done so that it's a bit more optimized than creating many Timer threads.
    '''

    def __init__(self, py_db):
        PyDBDaemonThread.__init__(self, py_db)
        self._event = threading.Event()
        self._handles = []

        # We could probably do things valid without this lock so that it's possible to add
        # handles while processing, but the implementation would also be harder to follow,
        # so, for now, we're either processing or adding handles, not both at the same time.
        self._lock = threading.Lock()

    def _on_run(self):
        wait_time = None
        while not self._kill_received:
            if _DEBUG:
                if wait_time is None:
                    pydev_log.critical('pydevd_timeout: Wait until a new handle is added.')
                else:
                    pydev_log.critical('pydevd_timeout: Next wait time: %s.', wait_time)
            self._event.wait(wait_time)

            if self._kill_received:
                self._handles = []
                return

            wait_time = self.process_handles()

    def process_handles(self):
        '''
        :return int:
            Returns the time we should be waiting for to process the next event properly.
        '''
        with self._lock:
            if _DEBUG:
                pydev_log.critical('pydevd_timeout: Processing handles')
            self._event.clear()
            handles = self._handles
            new_handles = self._handles = []

            # Do all the processing based on this time (we want to consider snapshots
            # of processing time -- anything not processed now may be processed at the
            # next snapshot).
            curtime = time.time()

            min_handle_timeout = None

            for handle in handles:
                if curtime < handle.abs_timeout and not handle.disposed:
                    # It still didn't time out.
                    if _DEBUG:
                        pydev_log.critical('pydevd_timeout: Handle NOT processed: %s', handle)
                    new_handles.append(handle)
                    if min_handle_timeout is None:
                        min_handle_timeout = handle.abs_timeout

                    elif handle.abs_timeout < min_handle_timeout:
                        min_handle_timeout = handle.abs_timeout

                else:
                    if _DEBUG:
                        pydev_log.critical('pydevd_timeout: Handle processed: %s', handle)
                    # Timed out (or disposed), so, let's execute it (should be no-op if disposed).
                    handle.exec_on_timeout()

            if min_handle_timeout is None:
                return None
            else:
                timeout = min_handle_timeout - curtime
                if timeout <= 0:
                    pydev_log.critical('pydevd_timeout: Expected timeout to be > 0. Found: %s', timeout)

                return timeout

    def do_kill_pydev_thread(self):
        PyDBDaemonThread.do_kill_pydev_thread(self)
        with self._lock:
            self._event.set()

    def add_on_timeout_handle(self, handle):
        with self._lock:
            self._handles.append(handle)
            self._event.set()


class _OnTimeoutHandle(object):

    def __init__(self, tracker, abs_timeout, on_timeout, kwargs):
        self._str = '_OnTimeoutHandle(%s)' % (on_timeout,)

        self._tracker = weakref.ref(tracker)
        self.abs_timeout = abs_timeout
        self.on_timeout = on_timeout
        if kwargs is None:
            kwargs = {}
        self.kwargs = kwargs
        self.disposed = False

    def exec_on_timeout(self):
        # Note: lock should already be obtained when executing this function.
        kwargs = self.kwargs
        on_timeout = self.on_timeout

        if not self.disposed:
            self.disposed = True
            self.kwargs = None
            self.on_timeout = None

            try:
                if _DEBUG:
                    pydev_log.critical('pydevd_timeout: Calling on timeout: %s with kwargs: %s', on_timeout, kwargs)

                on_timeout(**kwargs)
            except Exception:
                pydev_log.exception('pydevd_timeout: Exception on callback timeout.')

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        tracker = self._tracker()

        if tracker is None:
            lock = NULL
        else:
            lock = tracker._lock

        with lock:
            self.disposed = True
            self.kwargs = None
            self.on_timeout = None

    def __str__(self):
        return self._str

    __repr__ = __str__


class TimeoutTracker(object):
    '''
    This is a helper class to track the timeout of something.
    '''

    def __init__(self, py_db):
        self._thread = None
        self._lock = threading.Lock()
        self._py_db = weakref.ref(py_db)

    def call_on_timeout(self, timeout, on_timeout, kwargs=None):
        '''
        This can be called regularly to always execute the given function after a given timeout:

        call_on_timeout(py_db, 10, on_timeout)


        Or as a context manager to stop the method from being called if it finishes before the timeout
        elapses:

        with call_on_timeout(py_db, 10, on_timeout):
            ...

        Note: the callback will be called from a PyDBDaemonThread.
        '''
        with self._lock:
            if self._thread is None:
                if _DEBUG:
                    pydev_log.critical('pydevd_timeout: Created _TimeoutThread.')

                self._thread = _TimeoutThread(self._py_db())
                self._thread.start()

            curtime = time.time()
            handle = _OnTimeoutHandle(self, curtime + timeout, on_timeout, kwargs)
            if _DEBUG:
                pydev_log.critical('pydevd_timeout: Added handle: %s.', handle)
            self._thread.add_on_timeout_handle(handle)
            return handle


def create_interrupt_this_thread_callback():
    '''
    The idea here is returning a callback that when called will generate a KeyboardInterrupt
    in the thread that called this function.

    If this is the main thread, this means that it'll emulate a Ctrl+C (which may stop I/O
    and sleep operations).

    For other threads, this will call PyThreadState_SetAsyncExc to raise
    a KeyboardInterrupt before the next instruction (so, it won't really interrupt I/O or
    sleep operations).

    :return callable:
        Returns a callback that will interrupt the current thread (this may be called
        from an auxiliary thread).
    '''
    tid = thread_get_ident()

    if is_current_thread_main_thread():
        main_thread = threading.current_thread()

        def raise_on_this_thread():
            pydev_log.debug('Callback to interrupt main thread.')
            pydevd_utils.interrupt_main_thread(main_thread)

    else:

        # Note: this works in the sense that it can stop some cpu-intensive slow operation,
        # but we can't really interrupt the thread out of some sleep or I/O operation
        # (this will only be raised when Python is about to execute the next instruction).
        def raise_on_this_thread():
            if IS_CPYTHON:
                pydev_log.debug('Interrupt thread: %s', tid)
                ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), ctypes.py_object(KeyboardInterrupt))
            else:
                pydev_log.debug('It is only possible to interrupt non-main threads in CPython.')

    return raise_on_this_thread
