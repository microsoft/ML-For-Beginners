from _pydevd_bundle import pydevd_utils
from _pydevd_bundle.pydevd_additional_thread_info import set_additional_thread_info
from _pydevd_bundle.pydevd_comm_constants import CMD_STEP_INTO, CMD_THREAD_SUSPEND
from _pydevd_bundle.pydevd_constants import PYTHON_SUSPEND, STATE_SUSPEND, get_thread_id, STATE_RUN
from _pydev_bundle._pydev_saved_modules import threading
from _pydev_bundle import pydev_log


def pydevd_find_thread_by_id(thread_id):
    try:
        threads = threading.enumerate()
        for i in threads:
            tid = get_thread_id(i)
            if thread_id == tid or thread_id.endswith('|' + tid):
                return i

        # This can happen when a request comes for a thread which was previously removed.
        pydev_log.info("Could not find thread %s.", thread_id)
        pydev_log.info("Available: %s.", ([get_thread_id(t) for t in threads],))
    except:
        pydev_log.exception()

    return None


def mark_thread_suspended(thread, stop_reason, original_step_cmd=-1):
    info = set_additional_thread_info(thread)
    info.suspend_type = PYTHON_SUSPEND
    if original_step_cmd != -1:
        stop_reason = original_step_cmd
    thread.stop_reason = stop_reason

    # Note: don't set the 'pydev_original_step_cmd' here if unset.

    if info.pydev_step_cmd == -1:
        # If the step command is not specified, set it to step into
        # to make sure it'll break as soon as possible.
        info.pydev_step_cmd = CMD_STEP_INTO
        info.pydev_step_stop = None

    # Mark as suspended as the last thing.
    info.pydev_state = STATE_SUSPEND

    return info


def internal_run_thread(thread, set_additional_thread_info):
    info = set_additional_thread_info(thread)
    info.pydev_original_step_cmd = -1
    info.pydev_step_cmd = -1
    info.pydev_step_stop = None
    info.pydev_state = STATE_RUN


def resume_threads(thread_id, except_thread=None):
    pydev_log.info('Resuming threads: %s (except thread: %s)', thread_id, except_thread)
    threads = []
    if thread_id == '*':
        threads = pydevd_utils.get_non_pydevd_threads()

    elif thread_id.startswith('__frame__:'):
        pydev_log.critical("Can't make tasklet run: %s", thread_id)

    else:
        threads = [pydevd_find_thread_by_id(thread_id)]

    for t in threads:
        if t is None or t is except_thread:
            pydev_log.info('Skipped resuming thread: %s', t)
            continue

        internal_run_thread(t, set_additional_thread_info=set_additional_thread_info)


def suspend_all_threads(py_db, except_thread):
    '''
    Suspend all except the one passed as a parameter.
    :param except_thread:
    '''
    pydev_log.info('Suspending all threads except: %s', except_thread)
    all_threads = pydevd_utils.get_non_pydevd_threads()
    for t in all_threads:
        if getattr(t, 'pydev_do_not_trace', None):
            pass  # skip some other threads, i.e. ipython history saving thread from debug console
        else:
            if t is except_thread:
                continue
            info = mark_thread_suspended(t, CMD_THREAD_SUSPEND)
            frame = info.get_topmost_frame(t)

            # Reset the tracing as in this case as it could've set scopes to be untraced.
            if frame is not None:
                try:
                    py_db.set_trace_for_frame_and_parents(frame)
                finally:
                    frame = None
