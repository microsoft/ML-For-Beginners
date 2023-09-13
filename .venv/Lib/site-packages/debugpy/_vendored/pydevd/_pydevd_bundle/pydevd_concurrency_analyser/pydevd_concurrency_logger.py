import time

from _pydev_bundle._pydev_filesystem_encoding import getfilesystemencoding
from _pydev_bundle._pydev_saved_modules import threading
from _pydevd_bundle import pydevd_xml
from _pydevd_bundle.pydevd_constants import GlobalDebuggerHolder
from _pydevd_bundle.pydevd_constants import get_thread_id
from _pydevd_bundle.pydevd_net_command import NetCommand
from _pydevd_bundle.pydevd_concurrency_analyser.pydevd_thread_wrappers import ObjectWrapper, wrap_attr
import pydevd_file_utils
from _pydev_bundle import pydev_log
import sys

file_system_encoding = getfilesystemencoding()

from urllib.parse import quote

threadingCurrentThread = threading.current_thread

DONT_TRACE_THREADING = ['threading.py', 'pydevd.py']
INNER_METHODS = ['_stop']
INNER_FILES = ['threading.py']
THREAD_METHODS = ['start', '_stop', 'join']
LOCK_METHODS = ['__init__', 'acquire', 'release', '__enter__', '__exit__']
QUEUE_METHODS = ['put', 'get']

# return time since epoch in milliseconds
cur_time = lambda: int(round(time.time() * 1000000))


def get_text_list_for_frame(frame):
    # partial copy-paste from make_thread_suspend_str
    curFrame = frame
    cmdTextList = []
    try:
        while curFrame:
            # print cmdText
            myId = str(id(curFrame))
            # print "id is ", myId

            if curFrame.f_code is None:
                break  # Iron Python sometimes does not have it!

            myName = curFrame.f_code.co_name  # method name (if in method) or ? if global
            if myName is None:
                break  # Iron Python sometimes does not have it!

            # print "name is ", myName

            absolute_filename = pydevd_file_utils.get_abs_path_real_path_and_base_from_frame(curFrame)[0]

            my_file, _applied_mapping = pydevd_file_utils.map_file_to_client(absolute_filename)

            # print "file is ", my_file
            # my_file = inspect.getsourcefile(curFrame) or inspect.getfile(frame)

            myLine = str(curFrame.f_lineno)
            # print "line is ", myLine

            # the variables are all gotten 'on-demand'
            # variables = pydevd_xml.frame_vars_to_xml(curFrame.f_locals)

            variables = ''
            cmdTextList.append('<frame id="%s" name="%s" ' % (myId , pydevd_xml.make_valid_xml_value(myName)))
            cmdTextList.append('file="%s" line="%s">' % (quote(my_file, '/>_= \t'), myLine))
            cmdTextList.append(variables)
            cmdTextList.append("</frame>")
            curFrame = curFrame.f_back
    except:
        pydev_log.exception()

    return cmdTextList


def send_concurrency_message(event_class, time, name, thread_id, type, event, file, line, frame, lock_id=0, parent=None):
    dbg = GlobalDebuggerHolder.global_dbg
    if dbg is None:
        return
    cmdTextList = ['<xml>']

    cmdTextList.append('<' + event_class)
    cmdTextList.append(' time="%s"' % pydevd_xml.make_valid_xml_value(str(time)))
    cmdTextList.append(' name="%s"' % pydevd_xml.make_valid_xml_value(name))
    cmdTextList.append(' thread_id="%s"' % pydevd_xml.make_valid_xml_value(thread_id))
    cmdTextList.append(' type="%s"' % pydevd_xml.make_valid_xml_value(type))
    if type == "lock":
        cmdTextList.append(' lock_id="%s"' % pydevd_xml.make_valid_xml_value(str(lock_id)))
    if parent is not None:
        cmdTextList.append(' parent="%s"' % pydevd_xml.make_valid_xml_value(parent))
    cmdTextList.append(' event="%s"' % pydevd_xml.make_valid_xml_value(event))
    cmdTextList.append(' file="%s"' % pydevd_xml.make_valid_xml_value(file))
    cmdTextList.append(' line="%s"' % pydevd_xml.make_valid_xml_value(str(line)))
    cmdTextList.append('></' + event_class + '>')

    cmdTextList += get_text_list_for_frame(frame)
    cmdTextList.append('</xml>')

    text = ''.join(cmdTextList)
    if dbg.writer is not None:
        dbg.writer.add_command(NetCommand(145, 0, text))


def log_new_thread(global_debugger, t):
    event_time = cur_time() - global_debugger.thread_analyser.start_time
    send_concurrency_message("threading_event", event_time, t.name, get_thread_id(t), "thread",
             "start", "code_name", 0, None, parent=get_thread_id(t))


class ThreadingLogger:

    def __init__(self):
        self.start_time = cur_time()

    def set_start_time(self, time):
        self.start_time = time

    def log_event(self, frame):
        write_log = False
        self_obj = None
        if "self" in frame.f_locals:
            self_obj = frame.f_locals["self"]
            if isinstance(self_obj, threading.Thread) or self_obj.__class__ == ObjectWrapper:
                write_log = True
        if hasattr(frame, "f_back") and frame.f_back is not None:
            back = frame.f_back
            if hasattr(back, "f_back") and back.f_back is not None:
                back = back.f_back
                if "self" in back.f_locals:
                    if isinstance(back.f_locals["self"], threading.Thread):
                        write_log = True
        try:
            if write_log:
                t = threadingCurrentThread()
                back = frame.f_back
                if not back:
                    return
                name, _, back_base = pydevd_file_utils.get_abs_path_real_path_and_base_from_frame(back)
                event_time = cur_time() - self.start_time
                method_name = frame.f_code.co_name

                if isinstance(self_obj, threading.Thread):
                    if not hasattr(self_obj, "_pydev_run_patched"):
                        wrap_attr(self_obj, "run")
                    if (method_name in THREAD_METHODS) and (back_base not in DONT_TRACE_THREADING or \
                            (method_name in INNER_METHODS and back_base in INNER_FILES)):
                        thread_id = get_thread_id(self_obj)
                        name = self_obj.getName()
                        real_method = frame.f_code.co_name
                        parent = None
                        if real_method == "_stop":
                            if back_base in INNER_FILES and \
                                            back.f_code.co_name == "_wait_for_tstate_lock":
                                back = back.f_back.f_back
                            real_method = "stop"
                            if hasattr(self_obj, "_pydev_join_called"):
                                parent = get_thread_id(t)
                        elif real_method == "join":
                            # join called in the current thread, not in self object
                            if not self_obj.is_alive():
                                return
                            thread_id = get_thread_id(t)
                            name = t.name
                            self_obj._pydev_join_called = True

                        if real_method == "start":
                            parent = get_thread_id(t)
                        send_concurrency_message("threading_event", event_time, name, thread_id, "thread",
                        real_method, back.f_code.co_filename, back.f_lineno, back, parent=parent)
                        # print(event_time, self_obj.getName(), thread_id, "thread",
                        #       real_method, back.f_code.co_filename, back.f_lineno)

                if method_name == "pydev_after_run_call":
                    if hasattr(frame, "f_back") and frame.f_back is not None:
                        back = frame.f_back
                        if hasattr(back, "f_back") and back.f_back is not None:
                            back = back.f_back
                        if "self" in back.f_locals:
                            if isinstance(back.f_locals["self"], threading.Thread):
                                my_self_obj = frame.f_back.f_back.f_locals["self"]
                                my_back = frame.f_back.f_back
                                my_thread_id = get_thread_id(my_self_obj)
                                send_massage = True
                                if hasattr(my_self_obj, "_pydev_join_called"):
                                    send_massage = False
                                    # we can't detect stop after join in Python 2 yet
                                if send_massage:
                                    send_concurrency_message("threading_event", event_time, "Thread", my_thread_id, "thread",
                                                 "stop", my_back.f_code.co_filename, my_back.f_lineno, my_back, parent=None)

                if self_obj.__class__ == ObjectWrapper:
                    if back_base in DONT_TRACE_THREADING:
                        # do not trace methods called from threading
                        return
                    back_back_base = pydevd_file_utils.get_abs_path_real_path_and_base_from_frame(back.f_back)[2]
                    back = back.f_back
                    if back_back_base in DONT_TRACE_THREADING:
                        # back_back_base is the file, where the method was called froms
                        return
                    if method_name == "__init__":
                        send_concurrency_message("threading_event", event_time, t.name, get_thread_id(t), "lock",
                                     method_name, back.f_code.co_filename, back.f_lineno, back, lock_id=str(id(frame.f_locals["self"])))
                    if "attr" in frame.f_locals and \
                            (frame.f_locals["attr"] in LOCK_METHODS or
                            frame.f_locals["attr"] in QUEUE_METHODS):
                        real_method = frame.f_locals["attr"]
                        if method_name == "call_begin":
                            real_method += "_begin"
                        elif method_name == "call_end":
                            real_method += "_end"
                        else:
                            return
                        if real_method == "release_end":
                            # do not log release end. Maybe use it later
                            return
                        send_concurrency_message("threading_event", event_time, t.name, get_thread_id(t), "lock",
                        real_method, back.f_code.co_filename, back.f_lineno, back, lock_id=str(id(self_obj)))

                        if real_method in ("put_end", "get_end"):
                            # fake release for queue, cause we don't call it directly
                            send_concurrency_message("threading_event", event_time, t.name, get_thread_id(t), "lock",
                                         "release", back.f_code.co_filename, back.f_lineno, back, lock_id=str(id(self_obj)))
                        # print(event_time, t.name, get_thread_id(t), "lock",
                        #       real_method, back.f_code.co_filename, back.f_lineno)

        except Exception:
            pydev_log.exception()


class NameManager:

    def __init__(self, name_prefix):
        self.tasks = {}
        self.last = 0
        self.prefix = name_prefix

    def get(self, id):
        if id not in self.tasks:
            self.last += 1
            self.tasks[id] = self.prefix + "-" + str(self.last)
        return self.tasks[id]


class AsyncioLogger:

    def __init__(self):
        self.task_mgr = NameManager("Task")
        self.coro_mgr = NameManager("Coro")
        self.start_time = cur_time()

    def get_task_id(self, frame):
        asyncio = sys.modules.get('asyncio')
        if asyncio is None:
            # If asyncio was not imported, there's nothing to be done
            # (also fixes issue where multiprocessing is imported due
            # to asyncio).
            return None
        while frame is not None:
            if "self" in frame.f_locals:
                self_obj = frame.f_locals["self"]
                if isinstance(self_obj, asyncio.Task):
                    method_name = frame.f_code.co_name
                    if method_name == "_step":
                        return id(self_obj)
            frame = frame.f_back
        return None

    def log_event(self, frame):
        event_time = cur_time() - self.start_time

        # Debug loop iterations
        # if isinstance(self_obj, asyncio.base_events.BaseEventLoop):
        #     if method_name == "_run_once":
        #         print("Loop iteration")

        if not hasattr(frame, "f_back") or frame.f_back is None:
            return

        asyncio = sys.modules.get('asyncio')
        if asyncio is None:
            # If asyncio was not imported, there's nothing to be done
            # (also fixes issue where multiprocessing is imported due
            # to asyncio).
            return

        back = frame.f_back

        if "self" in frame.f_locals:
            self_obj = frame.f_locals["self"]
            if isinstance(self_obj, asyncio.Task):
                method_name = frame.f_code.co_name
                if method_name == "set_result":
                    task_id = id(self_obj)
                    task_name = self.task_mgr.get(str(task_id))
                    send_concurrency_message("asyncio_event", event_time, task_name, task_name, "thread", "stop", frame.f_code.co_filename,
                                 frame.f_lineno, frame)

                method_name = back.f_code.co_name
                if method_name == "__init__":
                    task_id = id(self_obj)
                    task_name = self.task_mgr.get(str(task_id))
                    send_concurrency_message("asyncio_event", event_time, task_name, task_name, "thread", "start", frame.f_code.co_filename,
                                 frame.f_lineno, frame)

            method_name = frame.f_code.co_name
            if isinstance(self_obj, asyncio.Lock):
                if method_name in ("acquire", "release"):
                    task_id = self.get_task_id(frame)
                    task_name = self.task_mgr.get(str(task_id))

                    if method_name == "acquire":
                        if not self_obj._waiters and not self_obj.locked():
                            send_concurrency_message("asyncio_event", event_time, task_name, task_name, "lock",
                                         method_name + "_begin", frame.f_code.co_filename, frame.f_lineno, frame, lock_id=str(id(self_obj)))
                        if self_obj.locked():
                            method_name += "_begin"
                        else:
                            method_name += "_end"
                    elif method_name == "release":
                        method_name += "_end"

                    send_concurrency_message("asyncio_event", event_time, task_name, task_name, "lock",
                                 method_name, frame.f_code.co_filename, frame.f_lineno, frame, lock_id=str(id(self_obj)))

            if isinstance(self_obj, asyncio.Queue):
                if method_name in ("put", "get", "_put", "_get"):
                    task_id = self.get_task_id(frame)
                    task_name = self.task_mgr.get(str(task_id))

                    if method_name == "put":
                        send_concurrency_message("asyncio_event", event_time, task_name, task_name, "lock",
                                     "acquire_begin", frame.f_code.co_filename, frame.f_lineno, frame, lock_id=str(id(self_obj)))
                    elif method_name == "_put":
                        send_concurrency_message("asyncio_event", event_time, task_name, task_name, "lock",
                                     "acquire_end", frame.f_code.co_filename, frame.f_lineno, frame, lock_id=str(id(self_obj)))
                        send_concurrency_message("asyncio_event", event_time, task_name, task_name, "lock",
                                     "release", frame.f_code.co_filename, frame.f_lineno, frame, lock_id=str(id(self_obj)))
                    elif method_name == "get":
                        back = frame.f_back
                        if back.f_code.co_name != "send":
                            send_concurrency_message("asyncio_event", event_time, task_name, task_name, "lock",
                                         "acquire_begin", frame.f_code.co_filename, frame.f_lineno, frame, lock_id=str(id(self_obj)))
                        else:
                            send_concurrency_message("asyncio_event", event_time, task_name, task_name, "lock",
                                         "acquire_end", frame.f_code.co_filename, frame.f_lineno, frame, lock_id=str(id(self_obj)))
                            send_concurrency_message("asyncio_event", event_time, task_name, task_name, "lock",
                                         "release", frame.f_code.co_filename, frame.f_lineno, frame, lock_id=str(id(self_obj)))
