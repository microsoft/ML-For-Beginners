from _pydev_bundle import pydev_log
from _pydevd_bundle import pydevd_import_class
from _pydevd_bundle.pydevd_frame_utils import add_exception_to_frame
from _pydev_bundle._pydev_saved_modules import threading


class ExceptionBreakpoint(object):

    def __init__(
        self,
        qname,
        condition,
        expression,
        notify_on_handled_exceptions,
        notify_on_unhandled_exceptions,
        notify_on_user_unhandled_exceptions,
        notify_on_first_raise_only,
        ignore_libraries
        ):
        exctype = get_exception_class(qname)
        self.qname = qname
        if exctype is not None:
            self.name = exctype.__name__
        else:
            self.name = None

        self.condition = condition
        self.expression = expression
        self.notify_on_unhandled_exceptions = notify_on_unhandled_exceptions
        self.notify_on_handled_exceptions = notify_on_handled_exceptions
        self.notify_on_first_raise_only = notify_on_first_raise_only
        self.notify_on_user_unhandled_exceptions = notify_on_user_unhandled_exceptions
        self.ignore_libraries = ignore_libraries

        self.type = exctype

    def __str__(self):
        return self.qname

    @property
    def has_condition(self):
        return self.condition is not None

    def handle_hit_condition(self, frame):
        return False


class LineBreakpoint(object):

    def __init__(self, breakpoint_id, line, condition, func_name, expression, suspend_policy="NONE", hit_condition=None, is_logpoint=False):
        self.breakpoint_id = breakpoint_id
        self.line = line
        self.condition = condition
        self.func_name = func_name
        self.expression = expression
        self.suspend_policy = suspend_policy
        self.hit_condition = hit_condition
        self._hit_count = 0
        self._hit_condition_lock = threading.Lock()
        self.is_logpoint = is_logpoint

    @property
    def has_condition(self):
        return bool(self.condition) or bool(self.hit_condition)

    def handle_hit_condition(self, frame):
        if not self.hit_condition:
            return False
        ret = False
        with self._hit_condition_lock:
            self._hit_count += 1
            expr = self.hit_condition.replace('@HIT@', str(self._hit_count))
            try:
                ret = bool(eval(expr, frame.f_globals, frame.f_locals))
            except Exception:
                ret = False
        return ret


class FunctionBreakpoint(object):

    def __init__(self, func_name, condition, expression, suspend_policy="NONE", hit_condition=None, is_logpoint=False):
        self.condition = condition
        self.func_name = func_name
        self.expression = expression
        self.suspend_policy = suspend_policy
        self.hit_condition = hit_condition
        self._hit_count = 0
        self._hit_condition_lock = threading.Lock()
        self.is_logpoint = is_logpoint

    @property
    def has_condition(self):
        return bool(self.condition) or bool(self.hit_condition)

    def handle_hit_condition(self, frame):
        if not self.hit_condition:
            return False
        ret = False
        with self._hit_condition_lock:
            self._hit_count += 1
            expr = self.hit_condition.replace('@HIT@', str(self._hit_count))
            try:
                ret = bool(eval(expr, frame.f_globals, frame.f_locals))
            except Exception:
                ret = False
        return ret


def get_exception_breakpoint(exctype, exceptions):
    if not exctype:
        exception_full_qname = None
    else:
        exception_full_qname = str(exctype.__module__) + '.' + exctype.__name__

    exc = None
    if exceptions is not None:
        try:
            return exceptions[exception_full_qname]
        except KeyError:
            for exception_breakpoint in exceptions.values():
                if exception_breakpoint.type is not None and issubclass(exctype, exception_breakpoint.type):
                    if exc is None or issubclass(exception_breakpoint.type, exc.type):
                        exc = exception_breakpoint
    return exc


def stop_on_unhandled_exception(py_db, thread, additional_info, arg):
    exctype, value, tb = arg
    break_on_uncaught_exceptions = py_db.break_on_uncaught_exceptions
    if break_on_uncaught_exceptions:
        exception_breakpoint = py_db.get_exception_breakpoint(exctype, break_on_uncaught_exceptions)
    else:
        exception_breakpoint = None

    if not exception_breakpoint:
        return

    if tb is None:  # sometimes it can be None, e.g. with GTK
        return

    if exctype is KeyboardInterrupt:
        return

    if exctype is SystemExit and py_db.ignore_system_exit_code(value):
        return

    frames = []
    user_frame = None

    while tb is not None:
        if not py_db.exclude_exception_by_filter(exception_breakpoint, tb):
            user_frame = tb.tb_frame
        frames.append(tb.tb_frame)
        tb = tb.tb_next

    if user_frame is None:
        return

    frames_byid = dict([(id(frame), frame) for frame in frames])
    add_exception_to_frame(user_frame, arg)
    if exception_breakpoint.condition is not None:
        eval_result = py_db.handle_breakpoint_condition(additional_info, exception_breakpoint, user_frame)
        if not eval_result:
            return

    if exception_breakpoint.expression is not None:
        py_db.handle_breakpoint_expression(exception_breakpoint, additional_info, user_frame)

    try:
        additional_info.pydev_message = exception_breakpoint.qname
    except:
        additional_info.pydev_message = exception_breakpoint.qname.encode('utf-8')

    pydev_log.debug('Handling post-mortem stop on exception breakpoint %s' % (exception_breakpoint.qname,))

    py_db.do_stop_on_unhandled_exception(thread, user_frame, frames_byid, arg)


def get_exception_class(kls):
    try:
        return eval(kls)
    except:
        return pydevd_import_class.import_name(kls)
