from _pydevd_bundle.pydevd_constants import (STATE_RUN, PYTHON_SUSPEND, SUPPORT_GEVENT, ForkSafeLock,
    _current_frames)
from _pydev_bundle import pydev_log
# IFDEF CYTHON
# pydev_log.debug("Using Cython speedups")
# ELSE
from _pydevd_bundle.pydevd_frame import PyDBFrame
# ENDIF

version = 11


#=======================================================================================================================
# PyDBAdditionalThreadInfo
#=======================================================================================================================
# IFDEF CYTHON
# cdef class PyDBAdditionalThreadInfo:
# ELSE
class PyDBAdditionalThreadInfo(object):
# ENDIF

    # Note: the params in cython are declared in pydevd_cython.pxd.
    # IFDEF CYTHON
    # ELSE
    __slots__ = [
        'pydev_state',
        'pydev_step_stop',
        'pydev_original_step_cmd',
        'pydev_step_cmd',
        'pydev_notify_kill',
        'pydev_django_resolve_frame',
        'pydev_call_from_jinja2',
        'pydev_call_inside_jinja2',
        'is_tracing',
        'conditional_breakpoint_exception',
        'pydev_message',
        'suspend_type',
        'pydev_next_line',
        'pydev_func_name',
        'suspended_at_unhandled',
        'trace_suspend_type',
        'top_level_thread_tracer_no_back_frames',
        'top_level_thread_tracer_unhandled',
        'thread_tracer',
        'step_in_initial_location',

        # Used for CMD_SMART_STEP_INTO (to know which smart step into variant to use)
        'pydev_smart_parent_offset',
        'pydev_smart_child_offset',

        # Used for CMD_SMART_STEP_INTO (list[_pydevd_bundle.pydevd_bytecode_utils.Variant])
        # Filled when the cmd_get_smart_step_into_variants is requested (so, this is a copy
        # of the last request for a given thread and pydev_smart_parent_offset/pydev_smart_child_offset relies on it).
        'pydev_smart_step_into_variants',
        'target_id_to_smart_step_into_variant',

        'pydev_use_scoped_step_frame',
    ]
    # ENDIF

    def __init__(self):
        self.pydev_state = STATE_RUN  # STATE_RUN or STATE_SUSPEND
        self.pydev_step_stop = None

        # Note: we have `pydev_original_step_cmd` and `pydev_step_cmd` because the original is to
        # say the action that started it and the other is to say what's the current tracing behavior
        # (because it's possible that we start with a step over but may have to switch to a
        # different step strategy -- for instance, if a step over is done and we return the current
        # method the strategy is changed to a step in).

        self.pydev_original_step_cmd = -1  # Something as CMD_STEP_INTO, CMD_STEP_OVER, etc.
        self.pydev_step_cmd = -1  # Something as CMD_STEP_INTO, CMD_STEP_OVER, etc.

        self.pydev_notify_kill = False
        self.pydev_django_resolve_frame = False
        self.pydev_call_from_jinja2 = None
        self.pydev_call_inside_jinja2 = None
        self.is_tracing = 0
        self.conditional_breakpoint_exception = None
        self.pydev_message = ''
        self.suspend_type = PYTHON_SUSPEND
        self.pydev_next_line = -1
        self.pydev_func_name = '.invalid.'  # Must match the type in cython
        self.suspended_at_unhandled = False
        self.trace_suspend_type = 'trace'  # 'trace' or 'frame_eval'
        self.top_level_thread_tracer_no_back_frames = []
        self.top_level_thread_tracer_unhandled = None
        self.thread_tracer = None
        self.step_in_initial_location = None
        self.pydev_smart_parent_offset = -1
        self.pydev_smart_child_offset = -1
        self.pydev_smart_step_into_variants = ()
        self.target_id_to_smart_step_into_variant = {}

        # Flag to indicate ipython use-case where each line will be executed as a call/line/return
        # in a new new frame but in practice we want to consider each new frame as if it was all
        # part of the same frame.
        #
        # In practice this means that a step over shouldn't revert to a step in and we need some
        # special logic to know when we should stop in a step over as we need to consider 2
        # different frames as being equal if they're logically the continuation of a frame
        # being executed by ipython line by line.
        #
        # See: https://github.com/microsoft/debugpy/issues/869#issuecomment-1132141003
        self.pydev_use_scoped_step_frame = False

    def get_topmost_frame(self, thread):
        '''
        Gets the topmost frame for the given thread. Note that it may be None
        and callers should remove the reference to the frame as soon as possible
        to avoid disturbing user code.
        '''
        # sys._current_frames(): dictionary with thread id -> topmost frame
        current_frames = _current_frames()
        topmost_frame = current_frames.get(thread.ident)
        if topmost_frame is None:
            # Note: this is expected for dummy threads (so, getting the topmost frame should be
            # treated as optional).
            pydev_log.info(
                'Unable to get topmost frame for thread: %s, thread.ident: %s, id(thread): %s\nCurrent frames: %s.\n'
                'GEVENT_SUPPORT: %s',
                thread,
                thread.ident,
                id(thread),
                current_frames,
                SUPPORT_GEVENT,
            )

        return topmost_frame

    def __str__(self):
        return 'State:%s Stop:%s Cmd: %s Kill:%s' % (
            self.pydev_state, self.pydev_step_stop, self.pydev_step_cmd, self.pydev_notify_kill)


_set_additional_thread_info_lock = ForkSafeLock()


def set_additional_thread_info(thread):
    try:
        additional_info = thread.additional_info
        if additional_info is None:
            raise AttributeError()
    except:
        with _set_additional_thread_info_lock:
            # If it's not there, set it within a lock to avoid any racing
            # conditions.
            additional_info = getattr(thread, 'additional_info', None)
            if additional_info is None:
                additional_info = PyDBAdditionalThreadInfo()
            thread.additional_info = additional_info

    return additional_info
