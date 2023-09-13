from _pydevd_bundle.pydevd_constants import get_current_thread_id, Null, ForkSafeLock
from pydevd_file_utils import get_abs_path_real_path_and_base_from_frame
from _pydev_bundle._pydev_saved_modules import thread, threading
import sys
from _pydev_bundle import pydev_log

DEBUG = False


class CustomFramesContainer:

    # Actual Values initialized later on.
    custom_frames_lock = None  # : :type custom_frames_lock: threading.Lock

    custom_frames = None

    _next_frame_id = None

    _py_db_command_thread_event = None


def custom_frames_container_init():  # Note: no staticmethod on jython 2.1 (so, use free-function)

    CustomFramesContainer.custom_frames_lock = ForkSafeLock()

    # custom_frames can only be accessed if properly locked with custom_frames_lock!
    # Key is a string identifying the frame (as well as the thread it belongs to).
    # Value is a CustomFrame.
    #
    CustomFramesContainer.custom_frames = {}

    # Only to be used in this module
    CustomFramesContainer._next_frame_id = 0

    # This is the event we must set to release an internal process events. It's later set by the actual debugger
    # when we do create the debugger.
    CustomFramesContainer._py_db_command_thread_event = Null()


# Initialize it the first time (it may be reinitialized later on when dealing with a fork).
custom_frames_container_init()


class CustomFrame:

    def __init__(self, name, frame, thread_id):
        # 0 = string with the representation of that frame
        self.name = name

        # 1 = the frame to show
        self.frame = frame

        # 2 = an integer identifying the last time the frame was changed.
        self.mod_time = 0

        # 3 = the thread id of the given frame
        self.thread_id = thread_id


def add_custom_frame(frame, name, thread_id):
    '''
    It's possible to show paused frames by adding a custom frame through this API (it's
    intended to be used for coroutines, but could potentially be used for generators too).

    :param frame:
        The topmost frame to be shown paused when a thread with thread.ident == thread_id is paused.

    :param name:
        The name to be shown for the custom thread in the UI.

    :param thread_id:
        The thread id to which this frame is related (must match thread.ident).

    :return: str
        Returns the custom thread id which will be used to show the given frame paused.
    '''
    with CustomFramesContainer.custom_frames_lock:
        curr_thread_id = get_current_thread_id(threading.current_thread())
        next_id = CustomFramesContainer._next_frame_id = CustomFramesContainer._next_frame_id + 1

        # Note: the frame id kept contains an id and thread information on the thread where the frame was added
        # so that later on we can check if the frame is from the current thread by doing frame_id.endswith('|'+thread_id).
        frame_custom_thread_id = '__frame__:%s|%s' % (next_id, curr_thread_id)
        if DEBUG:
            sys.stderr.write('add_custom_frame: %s (%s) %s %s\n' % (
                frame_custom_thread_id, get_abs_path_real_path_and_base_from_frame(frame)[-1], frame.f_lineno, frame.f_code.co_name))

        CustomFramesContainer.custom_frames[frame_custom_thread_id] = CustomFrame(name, frame, thread_id)
        CustomFramesContainer._py_db_command_thread_event.set()
        return frame_custom_thread_id


def update_custom_frame(frame_custom_thread_id, frame, thread_id, name=None):
    with CustomFramesContainer.custom_frames_lock:
        if DEBUG:
            sys.stderr.write('update_custom_frame: %s\n' % frame_custom_thread_id)
        try:
            old = CustomFramesContainer.custom_frames[frame_custom_thread_id]
            if name is not None:
                old.name = name
            old.mod_time += 1
            old.thread_id = thread_id
        except:
            sys.stderr.write('Unable to get frame to replace: %s\n' % (frame_custom_thread_id,))
            pydev_log.exception()

        CustomFramesContainer._py_db_command_thread_event.set()


def remove_custom_frame(frame_custom_thread_id):
    with CustomFramesContainer.custom_frames_lock:
        if DEBUG:
            sys.stderr.write('remove_custom_frame: %s\n' % frame_custom_thread_id)
        CustomFramesContainer.custom_frames.pop(frame_custom_thread_id, None)
        CustomFramesContainer._py_db_command_thread_event.set()

