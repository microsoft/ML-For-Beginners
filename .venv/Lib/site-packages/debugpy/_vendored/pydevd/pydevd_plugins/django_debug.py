import inspect

from _pydev_bundle import pydev_log
from _pydevd_bundle.pydevd_comm import CMD_SET_BREAK, CMD_ADD_EXCEPTION_BREAK
from _pydevd_bundle.pydevd_constants import STATE_SUSPEND, DJANGO_SUSPEND, \
    DebugInfoHolder
from _pydevd_bundle.pydevd_frame_utils import add_exception_to_frame, FCode, just_raised, ignore_exception_trace
from pydevd_file_utils import canonical_normalized_path, absolute_path
from _pydevd_bundle.pydevd_api import PyDevdAPI
from pydevd_plugins.pydevd_line_validation import LineBreakpointWithLazyValidation, ValidationInfo
from _pydev_bundle.pydev_override import overrides

IS_DJANGO18 = False
IS_DJANGO19 = False
IS_DJANGO19_OR_HIGHER = False
try:
    import django
    version = django.VERSION
    IS_DJANGO18 = version[0] == 1 and version[1] == 8
    IS_DJANGO19 = version[0] == 1 and version[1] == 9
    IS_DJANGO19_OR_HIGHER = ((version[0] == 1 and version[1] >= 9) or version[0] > 1)
except:
    pass


class DjangoLineBreakpoint(LineBreakpointWithLazyValidation):

    def __init__(self, canonical_normalized_filename, breakpoint_id, line, condition, func_name, expression, hit_condition=None, is_logpoint=False):
        self.canonical_normalized_filename = canonical_normalized_filename
        LineBreakpointWithLazyValidation.__init__(self, breakpoint_id, line, condition, func_name, expression, hit_condition=hit_condition, is_logpoint=is_logpoint)

    def __str__(self):
        return "DjangoLineBreakpoint: %s-%d" % (self.canonical_normalized_filename, self.line)


class _DjangoValidationInfo(ValidationInfo):

    @overrides(ValidationInfo._collect_valid_lines_in_template_uncached)
    def _collect_valid_lines_in_template_uncached(self, template):
        lines = set()
        for node in self._iternodes(template.nodelist):
            if node.__class__.__name__ in _IGNORE_RENDER_OF_CLASSES:
                continue
            lineno = self._get_lineno(node)
            if lineno is not None:
                lines.add(lineno)
        return lines

    def _get_lineno(self, node):
        if hasattr(node, 'token') and hasattr(node.token, 'lineno'):
            return node.token.lineno
        return None

    def _iternodes(self, nodelist):
        for node in nodelist:
            yield node

            try:
                children = node.child_nodelists
            except:
                pass
            else:
                for attr in children:
                    nodelist = getattr(node, attr, None)
                    if nodelist:
                        # i.e.: yield from _iternodes(nodelist)
                        for node in self._iternodes(nodelist):
                            yield node


def add_line_breakpoint(plugin, pydb, type, canonical_normalized_filename, breakpoint_id, line, condition, expression, func_name, hit_condition=None, is_logpoint=False, add_breakpoint_result=None, on_changed_breakpoint_state=None):
    if type == 'django-line':
        django_line_breakpoint = DjangoLineBreakpoint(canonical_normalized_filename, breakpoint_id, line, condition, func_name, expression, hit_condition=hit_condition, is_logpoint=is_logpoint)
        if not hasattr(pydb, 'django_breakpoints'):
            _init_plugin_breaks(pydb)

        if IS_DJANGO19_OR_HIGHER:
            add_breakpoint_result.error_code = PyDevdAPI.ADD_BREAKPOINT_LAZY_VALIDATION
            django_line_breakpoint.add_breakpoint_result = add_breakpoint_result
            django_line_breakpoint.on_changed_breakpoint_state = on_changed_breakpoint_state
        else:
            add_breakpoint_result.error_code = PyDevdAPI.ADD_BREAKPOINT_NO_ERROR

        return django_line_breakpoint, pydb.django_breakpoints
    return None


def after_breakpoints_consolidated(plugin, py_db, canonical_normalized_filename, id_to_pybreakpoint, file_to_line_to_breakpoints):
    if IS_DJANGO19_OR_HIGHER:
        django_breakpoints_for_file = file_to_line_to_breakpoints.get(canonical_normalized_filename)
        if not django_breakpoints_for_file:
            return

        if not hasattr(py_db, 'django_validation_info'):
            _init_plugin_breaks(py_db)

        # In general we validate the breakpoints only when the template is loaded, but if the template
        # was already loaded, we can validate the breakpoints based on the last loaded value.
        py_db.django_validation_info.verify_breakpoints_from_template_cached_lines(
            py_db, canonical_normalized_filename, django_breakpoints_for_file)


def add_exception_breakpoint(plugin, pydb, type, exception):
    if type == 'django':
        if not hasattr(pydb, 'django_exception_break'):
            _init_plugin_breaks(pydb)
        pydb.django_exception_break[exception] = True
        return True
    return False


def _init_plugin_breaks(pydb):
    pydb.django_exception_break = {}
    pydb.django_breakpoints = {}

    pydb.django_validation_info = _DjangoValidationInfo()


def remove_exception_breakpoint(plugin, pydb, type, exception):
    if type == 'django':
        try:
            del pydb.django_exception_break[exception]
            return True
        except:
            pass
    return False


def remove_all_exception_breakpoints(plugin, pydb):
    if hasattr(pydb, 'django_exception_break'):
        pydb.django_exception_break = {}
        return True
    return False


def get_breakpoints(plugin, pydb, type):
    if type == 'django-line':
        return pydb.django_breakpoints
    return None


def _inherits(cls, *names):
    if cls.__name__ in names:
        return True
    inherits_node = False
    for base in inspect.getmro(cls):
        if base.__name__ in names:
            inherits_node = True
            break
    return inherits_node


_IGNORE_RENDER_OF_CLASSES = ('TextNode', 'NodeList')


def _is_django_render_call(frame, debug=False):
    try:
        name = frame.f_code.co_name
        if name != 'render':
            return False

        if 'self' not in frame.f_locals:
            return False

        cls = frame.f_locals['self'].__class__

        inherits_node = _inherits(cls, 'Node')

        if not inherits_node:
            return False

        clsname = cls.__name__
        if IS_DJANGO19:
            # in Django 1.9 we need to save the flag that there is included template
            if clsname == 'IncludeNode':
                if 'context' in frame.f_locals:
                    context = frame.f_locals['context']
                    context._has_included_template = True

        return clsname not in _IGNORE_RENDER_OF_CLASSES
    except:
        pydev_log.exception()
        return False


def _is_django_context_get_call(frame):
    try:
        if 'self' not in frame.f_locals:
            return False

        cls = frame.f_locals['self'].__class__

        return _inherits(cls, 'BaseContext')
    except:
        pydev_log.exception()
        return False


def _is_django_resolve_call(frame):
    try:
        name = frame.f_code.co_name
        if name != '_resolve_lookup':
            return False

        if 'self' not in frame.f_locals:
            return False

        cls = frame.f_locals['self'].__class__

        clsname = cls.__name__
        return clsname == 'Variable'
    except:
        pydev_log.exception()
        return False


def _is_django_suspended(thread):
    return thread.additional_info.suspend_type == DJANGO_SUSPEND


def suspend_django(main_debugger, thread, frame, cmd=CMD_SET_BREAK):
    if frame.f_lineno is None:
        return None

    main_debugger.set_suspend(thread, cmd)
    thread.additional_info.suspend_type = DJANGO_SUSPEND

    return frame


def _find_django_render_frame(frame):
    while frame is not None and not _is_django_render_call(frame):
        frame = frame.f_back

    return frame

#=======================================================================================================================
# Django Frame
#=======================================================================================================================


def _read_file(filename):
    # type: (str) -> str
    f = open(filename, 'r', encoding='utf-8', errors='replace')
    s = f.read()
    f.close()
    return s


def _offset_to_line_number(text, offset):
    curLine = 1
    curOffset = 0
    while curOffset < offset:
        if curOffset == len(text):
            return -1
        c = text[curOffset]
        if c == '\n':
            curLine += 1
        elif c == '\r':
            curLine += 1
            if curOffset < len(text) and text[curOffset + 1] == '\n':
                curOffset += 1

        curOffset += 1

    return curLine


def _get_source_django_18_or_lower(frame):
    # This method is usable only for the Django <= 1.8
    try:
        node = frame.f_locals['self']
        if hasattr(node, 'source'):
            return node.source
        else:
            if IS_DJANGO18:
                # The debug setting was changed since Django 1.8
                pydev_log.error_once("WARNING: Template path is not available. Set the 'debug' option in the OPTIONS of a DjangoTemplates "
                                     "backend.")
            else:
                # The debug setting for Django < 1.8
                pydev_log.error_once("WARNING: Template path is not available. Please set TEMPLATE_DEBUG=True in your settings.py to make "
                                     "django template breakpoints working")
            return None

    except:
        pydev_log.exception()
        return None


def _convert_to_str(s):
    return s


def _get_template_original_file_name_from_frame(frame):
    try:
        if IS_DJANGO19:
            # The Node source was removed since Django 1.9
            if 'context' in frame.f_locals:
                context = frame.f_locals['context']
                if hasattr(context, '_has_included_template'):
                    # if there was included template we need to inspect the previous frames and find its name
                    back = frame.f_back
                    while back is not None and frame.f_code.co_name in ('render', '_render'):
                        locals = back.f_locals
                        if 'self' in locals:
                            self = locals['self']
                            if self.__class__.__name__ == 'Template' and hasattr(self, 'origin') and \
                                    hasattr(self.origin, 'name'):
                                return _convert_to_str(self.origin.name)
                        back = back.f_back
                else:
                    if hasattr(context, 'template') and hasattr(context.template, 'origin') and \
                            hasattr(context.template.origin, 'name'):
                        return _convert_to_str(context.template.origin.name)
            return None
        elif IS_DJANGO19_OR_HIGHER:
            # For Django 1.10 and later there is much simpler way to get template name
            if 'self' in frame.f_locals:
                self = frame.f_locals['self']
                if hasattr(self, 'origin') and hasattr(self.origin, 'name'):
                    return _convert_to_str(self.origin.name)
            return None

        source = _get_source_django_18_or_lower(frame)
        if source is None:
            pydev_log.debug("Source is None\n")
            return None
        fname = _convert_to_str(source[0].name)

        if fname == '<unknown source>':
            pydev_log.debug("Source name is %s\n" % fname)
            return None
        else:
            return fname
    except:
        if DebugInfoHolder.DEBUG_TRACE_LEVEL >= 2:
            pydev_log.exception('Error getting django template filename.')
        return None


def _get_template_line(frame):
    if IS_DJANGO19_OR_HIGHER:
        node = frame.f_locals['self']
        if hasattr(node, 'token') and hasattr(node.token, 'lineno'):
            return node.token.lineno
        else:
            return None

    source = _get_source_django_18_or_lower(frame)
    original_filename = _get_template_original_file_name_from_frame(frame)
    if original_filename is not None:
        try:
            absolute_filename = absolute_path(original_filename)
            return _offset_to_line_number(_read_file(absolute_filename), source[1][0])
        except:
            return None
    return None


class DjangoTemplateFrame(object):

    IS_PLUGIN_FRAME = True

    def __init__(self, frame):
        original_filename = _get_template_original_file_name_from_frame(frame)
        self._back_context = frame.f_locals['context']
        self.f_code = FCode('Django Template', original_filename)
        self.f_lineno = _get_template_line(frame)
        self.f_back = frame
        self.f_globals = {}
        self.f_locals = self._collect_context(self._back_context)
        self.f_trace = None

    def _collect_context(self, context):
        res = {}
        try:
            for d in context.dicts:
                for k, v in d.items():
                    res[k] = v
        except  AttributeError:
            pass
        return res

    def _change_variable(self, name, value):
        for d in self._back_context.dicts:
            for k, v in d.items():
                if k == name:
                    d[k] = value


class DjangoTemplateSyntaxErrorFrame(object):

    IS_PLUGIN_FRAME = True

    def __init__(self, frame, original_filename, lineno, f_locals):
        self.f_code = FCode('Django TemplateSyntaxError', original_filename)
        self.f_lineno = lineno
        self.f_back = frame
        self.f_globals = {}
        self.f_locals = f_locals
        self.f_trace = None


def change_variable(plugin, frame, attr, expression):
    if isinstance(frame, DjangoTemplateFrame):
        result = eval(expression, frame.f_globals, frame.f_locals)
        frame._change_variable(attr, result)
        return result
    return False


def _is_django_variable_does_not_exist_exception_break_context(frame):
    try:
        name = frame.f_code.co_name
    except:
        name = None
    return name in ('_resolve_lookup', 'find_template')


def _is_ignoring_failures(frame):
    while frame is not None:
        if frame.f_code.co_name == 'resolve':
            ignore_failures = frame.f_locals.get('ignore_failures')
            if ignore_failures:
                return True
        frame = frame.f_back

    return False

#=======================================================================================================================
# Django Step Commands
#=======================================================================================================================


def can_skip(plugin, main_debugger, frame):
    if main_debugger.django_breakpoints:
        if _is_django_render_call(frame):
            return False

    if main_debugger.django_exception_break:
        module_name = frame.f_globals.get('__name__', '')

        if module_name == 'django.template.base':
            # Exceptions raised at django.template.base must be checked.
            return False

    return True


def has_exception_breaks(plugin):
    if len(plugin.main_debugger.django_exception_break) > 0:
        return True
    return False


def has_line_breaks(plugin):
    for _canonical_normalized_filename, breakpoints in plugin.main_debugger.django_breakpoints.items():
        if len(breakpoints) > 0:
            return True
    return False


def cmd_step_into(plugin, main_debugger, frame, event, args, stop_info, stop):
    info = args[2]
    thread = args[3]
    plugin_stop = False
    if _is_django_suspended(thread):
        stop_info['django_stop'] = event == 'call' and _is_django_render_call(frame)
        plugin_stop = stop_info['django_stop']
        stop = stop and _is_django_resolve_call(frame.f_back) and not _is_django_context_get_call(frame)
        if stop:
            info.pydev_django_resolve_frame = True  # we remember that we've go into python code from django rendering frame
    return stop, plugin_stop


def cmd_step_over(plugin, main_debugger, frame, event, args, stop_info, stop):
    info = args[2]
    thread = args[3]
    plugin_stop = False
    if _is_django_suspended(thread):
        stop_info['django_stop'] = event == 'call' and _is_django_render_call(frame)
        plugin_stop = stop_info['django_stop']
        stop = False
        return stop, plugin_stop
    else:
        if event == 'return' and info.pydev_django_resolve_frame and _is_django_resolve_call(frame.f_back):
            # we return to Django suspend mode and should not stop before django rendering frame
            info.pydev_step_stop = frame.f_back
            info.pydev_django_resolve_frame = False
            thread.additional_info.suspend_type = DJANGO_SUSPEND
        stop = info.pydev_step_stop is frame and event in ('line', 'return')
    return stop, plugin_stop


def stop(plugin, main_debugger, frame, event, args, stop_info, arg, step_cmd):
    main_debugger = args[0]
    thread = args[3]
    if 'django_stop' in stop_info and stop_info['django_stop']:
        frame = suspend_django(main_debugger, thread, DjangoTemplateFrame(frame), step_cmd)
        if frame:
            main_debugger.do_wait_suspend(thread, frame, event, arg)
            return True
    return False


def get_breakpoint(plugin, py_db, pydb_frame, frame, event, args):
    py_db = args[0]
    _filename = args[1]
    info = args[2]
    breakpoint_type = 'django'

    if event == 'call' and info.pydev_state != STATE_SUSPEND and py_db.django_breakpoints and _is_django_render_call(frame):
        original_filename = _get_template_original_file_name_from_frame(frame)
        pydev_log.debug("Django is rendering a template: %s", original_filename)

        canonical_normalized_filename = canonical_normalized_path(original_filename)
        django_breakpoints_for_file = py_db.django_breakpoints.get(canonical_normalized_filename)

        if django_breakpoints_for_file:

            # At this point, let's validate whether template lines are correct.
            if IS_DJANGO19_OR_HIGHER:
                django_validation_info = py_db.django_validation_info
                context = frame.f_locals['context']
                django_template = context.template
                django_validation_info.verify_breakpoints(py_db, canonical_normalized_filename, django_breakpoints_for_file, django_template)

            pydev_log.debug("Breakpoints for that file: %s", django_breakpoints_for_file)
            template_line = _get_template_line(frame)
            pydev_log.debug("Tracing template line: %s", template_line)

            if template_line in django_breakpoints_for_file:
                django_breakpoint = django_breakpoints_for_file[template_line]
                new_frame = DjangoTemplateFrame(frame)
                return True, django_breakpoint, new_frame, breakpoint_type

    return False, None, None, breakpoint_type


def suspend(plugin, main_debugger, thread, frame, bp_type):
    if bp_type == 'django':
        return suspend_django(main_debugger, thread, DjangoTemplateFrame(frame))
    return None


def _get_original_filename_from_origin_in_parent_frame_locals(frame, parent_frame_name):
    filename = None
    parent_frame = frame
    while parent_frame.f_code.co_name != parent_frame_name:
        parent_frame = parent_frame.f_back

    origin = None
    if parent_frame is not None:
        origin = parent_frame.f_locals.get('origin')

    if hasattr(origin, 'name') and origin.name is not None:
        filename = _convert_to_str(origin.name)
    return filename


def exception_break(plugin, main_debugger, pydb_frame, frame, args, arg):
    main_debugger = args[0]
    thread = args[3]
    exception, value, trace = arg

    if main_debugger.django_exception_break and exception is not None:
        if exception.__name__ in ['VariableDoesNotExist', 'TemplateDoesNotExist', 'TemplateSyntaxError'] and \
                just_raised(trace) and not ignore_exception_trace(trace):

            if exception.__name__ == 'TemplateSyntaxError':
                # In this case we don't actually have a regular render frame with the context
                # (we didn't really get to that point).
                token = getattr(value, 'token', None)

                if token is None:
                    # Django 1.7 does not have token in exception. Try to get it from locals.
                    token = frame.f_locals.get('token')

                lineno = getattr(token, 'lineno', None)

                original_filename = None
                if lineno is not None:
                    original_filename = _get_original_filename_from_origin_in_parent_frame_locals(frame, 'get_template')

                    if original_filename is None:
                        # Django 1.7 does not have origin in get_template. Try to get it from
                        # load_template.
                        original_filename = _get_original_filename_from_origin_in_parent_frame_locals(frame, 'load_template')

                if original_filename is not None and lineno is not None:
                    syntax_error_frame = DjangoTemplateSyntaxErrorFrame(
                        frame, original_filename, lineno, {'token': token, 'exception': exception})

                    suspend_frame = suspend_django(
                        main_debugger, thread, syntax_error_frame, CMD_ADD_EXCEPTION_BREAK)
                    return True, suspend_frame

            elif exception.__name__ == 'VariableDoesNotExist':
                if _is_django_variable_does_not_exist_exception_break_context(frame):
                    if not getattr(exception, 'silent_variable_failure', False) and not _is_ignoring_failures(frame):
                        render_frame = _find_django_render_frame(frame)
                        if render_frame:
                            suspend_frame = suspend_django(
                                main_debugger, thread, DjangoTemplateFrame(render_frame), CMD_ADD_EXCEPTION_BREAK)
                            if suspend_frame:
                                add_exception_to_frame(suspend_frame, (exception, value, trace))
                                thread.additional_info.pydev_message = 'VariableDoesNotExist'
                                suspend_frame.f_back = frame
                                frame = suspend_frame
                                return True, frame

    return None
