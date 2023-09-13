from functools import partial
import itertools
from pydevd import AbstractSingleNotificationBehavior
import time

import pytest

from _pydevd_bundle.pydevd_daemon_thread import run_as_pydevd_daemon_thread
from tests_python.debugger_unittest import CMD_THREAD_SUSPEND, CMD_STEP_OVER, CMD_SET_BREAK
from _pydev_bundle.pydev_override import overrides
import threading

STATE_RUN = 1
STATE_SUSPEND = 2


class _ThreadInfo(object):

    next_thread_id = partial(next, itertools.count())

    def __init__(self):
        self.state = STATE_RUN
        self.thread = threading.Thread()
        self.thread_id = self.next_thread_id()


class _CustomSingleNotificationBehavior(AbstractSingleNotificationBehavior):

    NOTIFY_OF_PAUSE_TIMEOUT = .01

    __slots__ = AbstractSingleNotificationBehavior.__slots__ + ['notification_queue']

    def __init__(self, py_db):
        try:
            from queue import Queue
        except ImportError:
            from Queue import Queue
        super(_CustomSingleNotificationBehavior, self).__init__(py_db)
        self.notification_queue = Queue()

    @overrides(AbstractSingleNotificationBehavior.send_resume_notification)
    def send_resume_notification(self, *args, **kwargs):
        # print('put resume', threading.current_thread())
        self.notification_queue.put('resume')

    @overrides(AbstractSingleNotificationBehavior.send_suspend_notification)
    def send_suspend_notification(self, *args, **kwargs):
        # print('put suspend', threading.current_thread())
        self.notification_queue.put('suspend')

    def do_wait_suspend(self, thread_info, stop_reason):
        with self.notify_thread_suspended(thread_info.thread_id, thread_info.thread, stop_reason=stop_reason):
            while thread_info.state == STATE_SUSPEND:
                time.sleep(.1)


@pytest.fixture
def _dummy_pydb():
    return _DummyPyDB()


@pytest.fixture(
    name='single_notification_behavior',
#     params=range(50)  #  uncomment to run the tests many times.
)
def _single_notification_behavior(_dummy_pydb):
    single_notification_behavior = _CustomSingleNotificationBehavior(_dummy_pydb)
    return single_notification_behavior


@pytest.fixture(name='notification_queue')
def _notification_queue(single_notification_behavior):
    return single_notification_behavior.notification_queue


def wait_for_notification(notification_queue, msg):
    __tracebackhide__ = True
    try:
        from Queue import Empty
    except ImportError:
        from queue import Empty
    try:
        found = notification_queue.get(timeout=2)
        assert found == msg
    except Empty:
        raise AssertionError('Timed out while waiting for %s notification.' % (msg,))


def join_thread(t):
    __tracebackhide__ = True
    t.join(2)
    assert not t.is_alive(), 'Thread still alive after timeout.s'


class _DummyPyDB(object):

    def __init__(self):
        from _pydevd_bundle.pydevd_timeout import TimeoutTracker
        self.created_pydb_daemon_threads = {}
        self.timeout_tracker = TimeoutTracker(self)


def test_single_notification_1(single_notification_behavior, notification_queue):
    '''
    1. Resume before pausing 2nd thread

    - user pauses all (2) threads
    - break first -> send notification
    - user presses continue all before second is paused
      - 2nd should not pause nor send notification
      - resume all notification should be sent
    '''
    thread_info1 = _ThreadInfo()
    thread_info2 = _ThreadInfo()

    # pause all = set_suspend both
    single_notification_behavior.increment_suspend_time()
    single_notification_behavior.on_pause()
    thread_info1.state = STATE_SUSPEND
    thread_info2.state = STATE_SUSPEND

    dummy_py_db = _DummyPyDB()

    t = run_as_pydevd_daemon_thread(dummy_py_db, single_notification_behavior.do_wait_suspend, thread_info1, CMD_THREAD_SUSPEND)
    thread_info1.state = STATE_RUN
    # Set 2 to run before it starts (should not send additional message).
    thread_info2.state = STATE_RUN
    t.join()

    assert notification_queue.qsize() == 2
    assert notification_queue.get() == 'suspend'
    assert notification_queue.get() == 'resume'
    assert notification_queue.qsize() == 0

    # Run thread 2 only now (no additional notification).
    t = run_as_pydevd_daemon_thread(dummy_py_db, single_notification_behavior.do_wait_suspend, thread_info1, CMD_THREAD_SUSPEND)
    t.join()

    assert notification_queue.qsize() == 0


def test_single_notification_2(single_notification_behavior, notification_queue):
    '''
    2. Pausing all then stepping

    - user pauses all (2) threads
    - break first -> send notification
    - break second (no notification)
    - user steps in second
    - suspend in second -> send resume/pause notification on step
    '''
    thread_info1 = _ThreadInfo()
    thread_info2 = _ThreadInfo()

    dummy_py_db = _DummyPyDB()

    # pause all = set_suspend both
    single_notification_behavior.increment_suspend_time()
    single_notification_behavior.on_pause()
    thread_info1.state = STATE_SUSPEND
    thread_info2.state = STATE_SUSPEND

    # Leave both in break mode
    t1 = run_as_pydevd_daemon_thread(dummy_py_db, single_notification_behavior.do_wait_suspend, thread_info1, CMD_THREAD_SUSPEND)
    wait_for_notification(notification_queue, 'suspend')

    t2 = run_as_pydevd_daemon_thread(dummy_py_db, single_notification_behavior.do_wait_suspend, thread_info2, CMD_THREAD_SUSPEND)

    # Step would actually be set state to STEP, which would result in resuming
    # and then stopping again as if it was a SUSPEND (which calls a set_supend again with
    # the step mode).
    thread_info2.state = STATE_RUN
    join_thread(t2)
    wait_for_notification(notification_queue, 'resume')

    single_notification_behavior.increment_suspend_time()
    thread_info2.state = STATE_SUSPEND
    t2 = run_as_pydevd_daemon_thread(dummy_py_db, single_notification_behavior.do_wait_suspend, thread_info2, CMD_STEP_OVER)
    wait_for_notification(notification_queue, 'suspend')

    thread_info1.state = STATE_RUN
    thread_info2.state = STATE_RUN
    # First does a resume notification, the other remains quiet.
    wait_for_notification(notification_queue, 'resume')

    join_thread(t2)
    join_thread(t1)
    assert notification_queue.qsize() == 0


def test_single_notification_3(single_notification_behavior, notification_queue, _dummy_pydb):
    '''
    3. Deadlocked thread

    - user adds breakpoint in thread.join() -- just before threads becomes deadlocked
    - breakpoint hits -> send notification
      - pauses 2nd thread (no notification)
    - user steps over thead.join() -> never completes
    - user presses pause
      - second thread is already stopped
        - send notification regarding 2nd thread (still stopped).
    - leave both threads running: no suspend should be shown as there are no stopped threads
    - when thread is paused, show suspend notification
    '''

    # i.e.: stopping at breakpoint
    thread_info1 = _ThreadInfo()
    single_notification_behavior.increment_suspend_time()
    thread_info1.state = STATE_SUSPEND
    t1 = run_as_pydevd_daemon_thread(_dummy_pydb, single_notification_behavior.do_wait_suspend, thread_info1, CMD_SET_BREAK)

    # i.e.: stop because of breakpoint
    thread_info2 = _ThreadInfo()
    thread_info2.state = STATE_SUSPEND
    t2 = run_as_pydevd_daemon_thread(_dummy_pydb, single_notification_behavior.do_wait_suspend, thread_info2, CMD_SET_BREAK)

    wait_for_notification(notification_queue, 'suspend')

    # i.e.: step over (thread 2 is still suspended and this one never completes)
    thread_info1.state = STATE_RUN
    wait_for_notification(notification_queue, 'resume')

    join_thread(t1)

    # On pause we should notify that the thread 2 is suspended (after timeout if no other thread suspends first).
    single_notification_behavior.increment_suspend_time()
    single_notification_behavior.on_pause()
    thread_info1.state = STATE_SUSPEND
    thread_info2.state = STATE_SUSPEND
    wait_for_notification(notification_queue, 'suspend')

    thread_info2.state = STATE_RUN
    wait_for_notification(notification_queue, 'resume')

    join_thread(t2)
    assert notification_queue.qsize() == 0
    assert not single_notification_behavior._suspended_thread_id_to_thread

    # Now, no threads are running and pause is pressed
    # (maybe we could do a thread dump in this case as this
    # means nothing is stopped after pause is requested and
    # the timeout elapses).
    single_notification_behavior.increment_suspend_time()
    single_notification_behavior.on_pause()
    thread_info1.state = STATE_SUSPEND
    thread_info2.state = STATE_SUSPEND

    time.sleep(single_notification_behavior.NOTIFY_OF_PAUSE_TIMEOUT * 2)
    assert notification_queue.qsize() == 0

    t1 = run_as_pydevd_daemon_thread(_dummy_pydb, single_notification_behavior.do_wait_suspend, thread_info1, CMD_THREAD_SUSPEND)
    wait_for_notification(notification_queue, 'suspend')
    thread_info1.state = STATE_RUN
    wait_for_notification(notification_queue, 'resume')
    join_thread(t1)
    assert notification_queue.qsize() == 0


def test_single_notification_4(single_notification_behavior, notification_queue, _dummy_pydb):
    '''
    4. Delayed stop

    - user presses pause
    - stops first (2nd keeps running)
    - user steps on first
    - 2nd hits before first ends step (should not send any notification)
    - when step finishes send notification
    '''
    thread_info1 = _ThreadInfo()
    thread_info2 = _ThreadInfo()

    single_notification_behavior.increment_suspend_time()
    single_notification_behavior.on_pause()
    thread_info1.state = STATE_SUSPEND
    thread_info2.state = STATE_SUSPEND

    t1 = run_as_pydevd_daemon_thread(_dummy_pydb, single_notification_behavior.do_wait_suspend, thread_info1, CMD_THREAD_SUSPEND)
    wait_for_notification(notification_queue, 'suspend')
    thread_info1.state = STATE_RUN
    wait_for_notification(notification_queue, 'resume')
    join_thread(t1)

    t2 = run_as_pydevd_daemon_thread(_dummy_pydb, single_notification_behavior.do_wait_suspend, thread_info1, CMD_THREAD_SUSPEND)
    time.sleep(.1)
    assert notification_queue.qsize() == 0

    single_notification_behavior.increment_suspend_time()
    thread_info1.state = STATE_SUSPEND
    t1 = run_as_pydevd_daemon_thread(_dummy_pydb, single_notification_behavior.do_wait_suspend, thread_info1, CMD_STEP_OVER)
    wait_for_notification(notification_queue, 'suspend')
    thread_info2.state = STATE_RUN
    thread_info1.state = STATE_RUN
    join_thread(t1)
    join_thread(t2)
    wait_for_notification(notification_queue, 'resume')
    assert notification_queue.qsize() == 0

