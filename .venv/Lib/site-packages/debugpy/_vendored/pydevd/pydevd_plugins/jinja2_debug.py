from _pydevd_bundle.pydevd_constants import STATE_SUSPEND, JINJA2_SUSPEND
from _pydevd_bundle.pydevd_comm import CMD_SET_BREAK, CMD_ADD_EXCEPTION_BREAK
from pydevd_file_utils import canonical_normalized_path
from _pydevd_bundle.pydevd_frame_utils import add_exception_to_frame, FCode
from _pydev_bundle import pydev_log
from pydevd_plugins.pydevd_line_validation import LineBreakpointWithLazyValidation, ValidationInfo
from _pydev_bundle.pydev_override import overrides
from _pydevd_bundle.pydevd_api import PyDevdAPI


class Jinja2LineBreakpoint(LineBreakpointWithLazyValidation):

    def __init__(self, canonical_normalized_filename, breakpoint_id, line, condition, func_name, expression, hit_condition=None, is_logpoint=False):
        self.canonical_normalized_filename = canonical_normalized_filename
        LineBreakpointWithLazyValidation.__init__(self, breakpoint_id, line, condition, func_name, expression, hit_condition=hit_condition, is_logpoint=is_logpoint)

    def __str__(self):
        return "Jinja2LineBreakpoint: %s-%d" % (self.canonical_normalized_filename, self.line)


class _Jinja2ValidationInfo(ValidationInfo):

    @overrides(ValidationInfo._collect_valid_lines_in_template_uncached)
    def _collect_valid_lines_in_template_uncached(self, template):
        lineno_mapping = _get_frame_lineno_mapping(template)
        if not lineno_mapping:
            return set()

        return set(x[0] for x in lineno_mapping)


def add_line_breakpoint(plugin, pydb, type, canonical_normalized_filename, breakpoint_id, line, condition, expression, func_name, hit_condition=None, is_logpoint=False, add_breakpoint_result=None, on_changed_breakpoint_state=None):
    if type == 'jinja2-line':
        jinja2_line_breakpoint = Jinja2LineBreakpoint(canonical_normalized_filename, breakpoint_id, line, condition, func_name, expression, hit_condition=hit_condition, is_logpoint=is_logpoint)
        if not hasattr(pydb, 'jinja2_breakpoints'):
            _init_plugin_breaks(pydb)

        add_breakpoint_result.error_code = PyDevdAPI.ADD_BREAKPOINT_LAZY_VALIDATION
        jinja2_line_breakpoint.add_breakpoint_result = add_breakpoint_result
        jinja2_line_breakpoint.on_changed_breakpoint_state = on_changed_breakpoint_state

        return jinja2_line_breakpoint, pydb.jinja2_breakpoints
    return None


def after_breakpoints_consolidated(plugin, py_db, canonical_normalized_filename, id_to_pybreakpoint, file_to_line_to_breakpoints):
    jinja2_breakpoints_for_file = file_to_line_to_breakpoints.get(canonical_normalized_filename)
    if not jinja2_breakpoints_for_file:
        return

    if not hasattr(py_db, 'jinja2_validation_info'):
        _init_plugin_breaks(py_db)

    # In general we validate the breakpoints only when the template is loaded, but if the template
    # was already loaded, we can validate the breakpoints based on the last loaded value.
    py_db.jinja2_validation_info.verify_breakpoints_from_template_cached_lines(
        py_db, canonical_normalized_filename, jinja2_breakpoints_for_file)


def add_exception_breakpoint(plugin, pydb, type, exception):
    if type == 'jinja2':
        if not hasattr(pydb, 'jinja2_exception_break'):
            _init_plugin_breaks(pydb)
        pydb.jinja2_exception_break[exception] = True
        return True
    return False


def _init_plugin_breaks(pydb):
    pydb.jinja2_exception_break = {}
    pydb.jinja2_breakpoints = {}

    pydb.jinja2_validation_info = _Jinja2ValidationInfo()


def remove_all_exception_breakpoints(plugin, pydb):
    if hasattr(pydb, 'jinja2_exception_break'):
        pydb.jinja2_exception_break = {}
        return True
    return False


def remove_exception_breakpoint(plugin, pydb, type, exception):
    if type == 'jinja2':
        try:
            del pydb.jinja2_exception_break[exception]
            return True
        except:
            pass
    return False


def get_breakpoints(plugin, pydb, type):
    if type == 'jinja2-line':
        return pydb.jinja2_breakpoints
    return None


def _is_jinja2_render_call(frame):
    try:
        name = frame.f_code.co_name
        if "__jinja_template__" in frame.f_globals and name in ("root", "loop", "macro") or name.startswith("block_"):
            return True
        return False
    except:
        pydev_log.exception()
        return False


def _suspend_jinja2(pydb, thread, frame, cmd=CMD_SET_BREAK, message=None):
    frame = Jinja2TemplateFrame(frame)

    if frame.f_lineno is None:
        return None

    pydb.set_suspend(thread, cmd)

    thread.additional_info.suspend_type = JINJA2_SUSPEND
    if cmd == CMD_ADD_EXCEPTION_BREAK:
        # send exception name as message
        if message:
            message = str(message)
        thread.additional_info.pydev_message = message

    return frame


def _is_jinja2_suspended(thread):
    return thread.additional_info.suspend_type == JINJA2_SUSPEND


def _is_jinja2_context_call(frame):
    return "_Context__obj" in frame.f_locals


def _is_jinja2_internal_function(frame):
    return 'self' in frame.f_locals and frame.f_locals['self'].__class__.__name__ in \
        ('LoopContext', 'TemplateReference', 'Macro', 'BlockReference')


def _find_jinja2_render_frame(frame):
    while frame is not None and not _is_jinja2_render_call(frame):
        frame = frame.f_back

    return frame

#=======================================================================================================================
# Jinja2 Frame
#=======================================================================================================================


class Jinja2TemplateFrame(object):

    IS_PLUGIN_FRAME = True

    def __init__(self, frame, original_filename=None, template_lineno=None):

        if original_filename is None:
            original_filename = _get_jinja2_template_original_filename(frame)

        if template_lineno is None:
            template_lineno = _get_jinja2_template_line(frame)

        self.back_context = None
        if 'context' in frame.f_locals:
            # sometimes we don't have 'context', e.g. in macros
            self.back_context = frame.f_locals['context']
        self.f_code = FCode('template', original_filename)
        self.f_lineno = template_lineno
        self.f_back = frame
        self.f_globals = {}
        self.f_locals = self.collect_context(frame)
        self.f_trace = None

    def _get_real_var_name(self, orig_name):
        # replace leading number for local variables
        parts = orig_name.split('_')
        if len(parts) > 1 and parts[0].isdigit():
            return parts[1]
        return orig_name

    def collect_context(self, frame):
        res = {}
        for k, v in frame.f_locals.items():
            if not k.startswith('l_'):
                res[k] = v
            elif v and not _is_missing(v):
                res[self._get_real_var_name(k[2:])] = v
        if self.back_context is not None:
            for k, v in self.back_context.items():
                res[k] = v
        return res

    def _change_variable(self, frame, name, value):
        in_vars_or_parents = False
        if 'context' in frame.f_locals:
            if name in frame.f_locals['context'].parent:
                self.back_context.parent[name] = value
                in_vars_or_parents = True
            if name in frame.f_locals['context'].vars:
                self.back_context.vars[name] = value
                in_vars_or_parents = True

        l_name = 'l_' + name
        if l_name in frame.f_locals:
            if in_vars_or_parents:
                frame.f_locals[l_name] = self.back_context.resolve(name)
            else:
                frame.f_locals[l_name] = value


class Jinja2TemplateSyntaxErrorFrame(object):

    IS_PLUGIN_FRAME = True

    def __init__(self, frame, exception_cls_name, filename, lineno, f_locals):
        self.f_code = FCode('Jinja2 %s' % (exception_cls_name,), filename)
        self.f_lineno = lineno
        self.f_back = frame
        self.f_globals = {}
        self.f_locals = f_locals
        self.f_trace = None


def change_variable(plugin, frame, attr, expression):
    if isinstance(frame, Jinja2TemplateFrame):
        result = eval(expression, frame.f_globals, frame.f_locals)
        frame._change_variable(frame.f_back, attr, result)
        return result
    return False


def _is_missing(item):
    if item.__class__.__name__ == 'MissingType':
        return True
    return False


def _find_render_function_frame(frame):
    # in order to hide internal rendering functions
    old_frame = frame
    try:
        while not ('self' in frame.f_locals and frame.f_locals['self'].__class__.__name__ == 'Template' and \
                               frame.f_code.co_name == 'render'):
            frame = frame.f_back
            if frame is None:
                return old_frame
        return frame
    except:
        return old_frame


def _get_jinja2_template_debug_info(frame):
    frame_globals = frame.f_globals

    jinja_template = frame_globals.get('__jinja_template__')

    if jinja_template is None:
        return None

    return _get_frame_lineno_mapping(jinja_template)


def _get_frame_lineno_mapping(jinja_template):
    '''
    :rtype: list(tuple(int,int))
    :return: list((original_line, line_in_frame))
    '''
    # _debug_info is a string with the mapping from frame line to actual line
    # i.e.: "5=13&8=14"
    _debug_info = jinja_template._debug_info
    if not _debug_info:
        # Sometimes template contains only plain text.
        return None

    # debug_info is a list with the mapping from frame line to actual line
    # i.e.: [(5, 13), (8, 14)]
    return jinja_template.debug_info


def _get_jinja2_template_line(frame):
    debug_info = _get_jinja2_template_debug_info(frame)
    if debug_info is None:
        return None

    lineno = frame.f_lineno

    for pair in debug_info:
        if pair[1] == lineno:
            return pair[0]

    return None


def _convert_to_str(s):
    return s


def _get_jinja2_template_original_filename(frame):
    if '__jinja_template__' in frame.f_globals:
        return _convert_to_str(frame.f_globals['__jinja_template__'].filename)

    return None

#=======================================================================================================================
# Jinja2 Step Commands
#=======================================================================================================================


def has_exception_breaks(plugin):
    if len(plugin.main_debugger.jinja2_exception_break) > 0:
        return True
    return False


def has_line_breaks(plugin):
    for _canonical_normalized_filename, breakpoints in plugin.main_debugger.jinja2_breakpoints.items():
        if len(breakpoints) > 0:
            return True
    return False


def can_skip(plugin, pydb, frame):
    if pydb.jinja2_breakpoints and _is_jinja2_render_call(frame):
        filename = _get_jinja2_template_original_filename(frame)
        if filename is not None:
            canonical_normalized_filename = canonical_normalized_path(filename)
            jinja2_breakpoints_for_file = pydb.jinja2_breakpoints.get(canonical_normalized_filename)
            if jinja2_breakpoints_for_file:
                return False

    if pydb.jinja2_exception_break:
        name = frame.f_code.co_name

        # errors in compile time
        if name in ('template', 'top-level template code', '<module>') or name.startswith('block '):
            f_back = frame.f_back
            module_name = ''
            if f_back is not None:
                module_name = f_back.f_globals.get('__name__', '')
            if module_name.startswith('jinja2.'):
                return False

    return True


def cmd_step_into(plugin, pydb, frame, event, args, stop_info, stop):
    info = args[2]
    thread = args[3]
    plugin_stop = False
    stop_info['jinja2_stop'] = False
    if _is_jinja2_suspended(thread):
        stop_info['jinja2_stop'] = event in ('call', 'line') and _is_jinja2_render_call(frame)
        plugin_stop = stop_info['jinja2_stop']
        stop = False
        if info.pydev_call_from_jinja2 is not None:
            if _is_jinja2_internal_function(frame):
                # if internal Jinja2 function was called, we sould continue debugging inside template
                info.pydev_call_from_jinja2 = None
            else:
                # we go into python code from Jinja2 rendering frame
                stop = True

        if event == 'call' and _is_jinja2_context_call(frame.f_back):
            # we called function from context, the next step will be in function
            info.pydev_call_from_jinja2 = 1

    if event == 'return' and _is_jinja2_context_call(frame.f_back):
        # we return from python code to Jinja2 rendering frame
        info.pydev_step_stop = info.pydev_call_from_jinja2
        info.pydev_call_from_jinja2 = None
        thread.additional_info.suspend_type = JINJA2_SUSPEND
        stop = False

        # print "info.pydev_call_from_jinja2", info.pydev_call_from_jinja2, "stop_info", stop_info, \
        #    "thread.additional_info.suspend_type", thread.additional_info.suspend_type
        # print "event", event, "farme.locals", frame.f_locals
    return stop, plugin_stop


def cmd_step_over(plugin, pydb, frame, event, args, stop_info, stop):
    info = args[2]
    thread = args[3]
    plugin_stop = False
    stop_info['jinja2_stop'] = False
    if _is_jinja2_suspended(thread):
        stop = False

        if info.pydev_call_inside_jinja2 is None:
            if _is_jinja2_render_call(frame):
                if event == 'call':
                    info.pydev_call_inside_jinja2 = frame.f_back
                if event in ('line', 'return'):
                    info.pydev_call_inside_jinja2 = frame
        else:
            if event == 'line':
                if _is_jinja2_render_call(frame) and info.pydev_call_inside_jinja2 is frame:
                    stop_info['jinja2_stop'] = True
                    plugin_stop = stop_info['jinja2_stop']
            if event == 'return':
                if frame is info.pydev_call_inside_jinja2 and 'event' not in frame.f_back.f_locals:
                    info.pydev_call_inside_jinja2 = _find_jinja2_render_frame(frame.f_back)
        return stop, plugin_stop
    else:
        if event == 'return' and _is_jinja2_context_call(frame.f_back):
            # we return from python code to Jinja2 rendering frame
            info.pydev_call_from_jinja2 = None
            info.pydev_call_inside_jinja2 = _find_jinja2_render_frame(frame)
            thread.additional_info.suspend_type = JINJA2_SUSPEND
            stop = False
            return stop, plugin_stop
    # print "info.pydev_call_from_jinja2", info.pydev_call_from_jinja2, "stop", stop, "jinja_stop", jinja2_stop, \
    #    "thread.additional_info.suspend_type", thread.additional_info.suspend_type
    # print "event", event, "info.pydev_call_inside_jinja2", info.pydev_call_inside_jinja2
    # print "frame", frame, "frame.f_back", frame.f_back, "step_stop", info.pydev_step_stop
    # print "is_context_call", _is_jinja2_context_call(frame)
    # print "render", _is_jinja2_render_call(frame)
    # print "-------------"
    return stop, plugin_stop


def stop(plugin, pydb, frame, event, args, stop_info, arg, step_cmd):
    pydb = args[0]
    thread = args[3]
    if 'jinja2_stop' in stop_info and stop_info['jinja2_stop']:
        frame = _suspend_jinja2(pydb, thread, frame, step_cmd)
        if frame:
            pydb.do_wait_suspend(thread, frame, event, arg)
            return True
    return False


def get_breakpoint(plugin, py_db, pydb_frame, frame, event, args):
    py_db = args[0]
    _filename = args[1]
    info = args[2]
    break_type = 'jinja2'

    if event == 'line' and info.pydev_state != STATE_SUSPEND and py_db.jinja2_breakpoints and _is_jinja2_render_call(frame):

        jinja_template = frame.f_globals.get('__jinja_template__')
        if jinja_template is None:
            return False, None, None, break_type

        original_filename = _get_jinja2_template_original_filename(frame)
        if original_filename is not None:
            pydev_log.debug("Jinja2 is rendering a template: %s", original_filename)
            canonical_normalized_filename = canonical_normalized_path(original_filename)
            jinja2_breakpoints_for_file = py_db.jinja2_breakpoints.get(canonical_normalized_filename)

            if jinja2_breakpoints_for_file:

                jinja2_validation_info = py_db.jinja2_validation_info
                jinja2_validation_info.verify_breakpoints(py_db, canonical_normalized_filename, jinja2_breakpoints_for_file, jinja_template)

                template_lineno = _get_jinja2_template_line(frame)
                if template_lineno is not None:
                    jinja2_breakpoint = jinja2_breakpoints_for_file.get(template_lineno)
                    if jinja2_breakpoint is not None:
                        new_frame = Jinja2TemplateFrame(frame, original_filename, template_lineno)
                        return True, jinja2_breakpoint, new_frame, break_type

    return False, None, None, break_type


def suspend(plugin, pydb, thread, frame, bp_type):
    if bp_type == 'jinja2':
        return _suspend_jinja2(pydb, thread, frame)
    return None


def exception_break(plugin, pydb, pydb_frame, frame, args, arg):
    pydb = args[0]
    thread = args[3]
    exception, value, trace = arg
    if pydb.jinja2_exception_break and exception is not None:
        exception_type = list(pydb.jinja2_exception_break.keys())[0]
        if exception.__name__ in ('UndefinedError', 'TemplateNotFound', 'TemplatesNotFound'):
            # errors in rendering
            render_frame = _find_jinja2_render_frame(frame)
            if render_frame:
                suspend_frame = _suspend_jinja2(pydb, thread, render_frame, CMD_ADD_EXCEPTION_BREAK, message=exception_type)
                if suspend_frame:
                    add_exception_to_frame(suspend_frame, (exception, value, trace))
                    suspend_frame.f_back = frame
                    frame = suspend_frame
                    return True, frame

        elif exception.__name__ in ('TemplateSyntaxError', 'TemplateAssertionError'):
            name = frame.f_code.co_name

            # errors in compile time
            if name in ('template', 'top-level template code', '<module>') or name.startswith('block '):

                f_back = frame.f_back
                if f_back is not None:
                    module_name = f_back.f_globals.get('__name__', '')

                if module_name.startswith('jinja2.'):
                    # Jinja2 translates exception info and creates fake frame on his own
                    pydb_frame.set_suspend(thread, CMD_ADD_EXCEPTION_BREAK)
                    add_exception_to_frame(frame, (exception, value, trace))
                    thread.additional_info.suspend_type = JINJA2_SUSPEND
                    thread.additional_info.pydev_message = str(exception_type)
                    return True, frame
    return None
