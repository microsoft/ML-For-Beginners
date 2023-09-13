def add_line_breakpoint(plugin, pydb, type, canonical_normalized_filename, breakpoint_id, line, condition, expression, func_name, hit_condition=None, is_logpoint=False, add_breakpoint_result=None, on_changed_breakpoint_state=None):
    return None


def after_breakpoints_consolidated(py_db, canonical_normalized_filename, id_to_pybreakpoint, file_to_line_to_breakpoints):
    return None


def add_exception_breakpoint(plugin, pydb, type, exception):
    return False


def remove_exception_breakpoint(plugin, pydb, type, exception):
    return False


def remove_all_exception_breakpoints(plugin, pydb):
    return False


def get_breakpoints(plugin, pydb):
    return None


def can_skip(plugin, pydb, frame):
    return True


def has_exception_breaks(plugin):
    return False


def has_line_breaks(plugin):
    return False


def cmd_step_into(plugin, pydb, frame, event, args, stop_info, stop):
    return False


def cmd_step_over(plugin, pydb, frame, event, args, stop_info, stop):
    return False


def stop(plugin, pydb, frame, event, args, stop_info, arg, step_cmd):
    return False


def get_breakpoint(plugin, pydb, pydb_frame, frame, event, args):
    return None


def suspend(plugin, pydb, thread, frame):
    return None


def exception_break(plugin, pydb, pydb_frame, frame, args, arg):
    return None


def change_variable(plugin, frame, attr, expression):
    return False
