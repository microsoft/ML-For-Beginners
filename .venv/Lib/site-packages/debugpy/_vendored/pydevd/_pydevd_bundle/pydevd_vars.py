""" pydevd_vars deals with variables:
    resolution/conversion to XML.
"""
import pickle
from _pydevd_bundle.pydevd_constants import get_frame, get_current_thread_id, \
    iter_chars, silence_warnings_decorator, get_global_debugger

from _pydevd_bundle.pydevd_xml import ExceptionOnEvaluate, get_type, var_to_xml
from _pydev_bundle import pydev_log
import functools
from _pydevd_bundle.pydevd_thread_lifecycle import resume_threads, mark_thread_suspended, suspend_all_threads
from _pydevd_bundle.pydevd_comm_constants import CMD_SET_BREAK

import sys  # @Reimport

from _pydev_bundle._pydev_saved_modules import threading
from _pydevd_bundle import pydevd_save_locals, pydevd_timeout, pydevd_constants
from _pydev_bundle.pydev_imports import Exec, execfile
from _pydevd_bundle.pydevd_utils import to_string
import inspect
from _pydevd_bundle.pydevd_daemon_thread import PyDBDaemonThread
from _pydevd_bundle.pydevd_save_locals import update_globals_and_locals
from functools import lru_cache

SENTINEL_VALUE = []


class VariableError(RuntimeError):
    pass


def iter_frames(frame):
    while frame is not None:
        yield frame
        frame = frame.f_back
    frame = None


def dump_frames(thread_id):
    sys.stdout.write('dumping frames\n')
    if thread_id != get_current_thread_id(threading.current_thread()):
        raise VariableError("find_frame: must execute on same thread")

    frame = get_frame()
    for frame in iter_frames(frame):
        sys.stdout.write('%s\n' % pickle.dumps(frame))


@silence_warnings_decorator
def getVariable(dbg, thread_id, frame_id, scope, locator):
    """
    returns the value of a variable

    :scope: can be BY_ID, EXPRESSION, GLOBAL, LOCAL, FRAME

    BY_ID means we'll traverse the list of all objects alive to get the object.

    :locator: after reaching the proper scope, we have to get the attributes until we find
            the proper location (i.e.: obj\tattr1\tattr2)

    :note: when BY_ID is used, the frame_id is considered the id of the object to find and
           not the frame (as we don't care about the frame in this case).
    """
    if scope == 'BY_ID':
        if thread_id != get_current_thread_id(threading.current_thread()):
            raise VariableError("getVariable: must execute on same thread")

        try:
            import gc
            objects = gc.get_objects()
        except:
            pass  # Not all python variants have it.
        else:
            frame_id = int(frame_id)
            for var in objects:
                if id(var) == frame_id:
                    if locator is not None:
                        locator_parts = locator.split('\t')
                        for k in locator_parts:
                            _type, _type_name, resolver = get_type(var)
                            var = resolver.resolve(var, k)

                    return var

        # If it didn't return previously, we coudn't find it by id (i.e.: already garbage collected).
        sys.stderr.write('Unable to find object with id: %s\n' % (frame_id,))
        return None

    frame = dbg.find_frame(thread_id, frame_id)
    if frame is None:
        return {}

    if locator is not None:
        locator_parts = locator.split('\t')
    else:
        locator_parts = []

    for attr in locator_parts:
        attr.replace("@_@TAB_CHAR@_@", '\t')

    if scope == 'EXPRESSION':
        for count in range(len(locator_parts)):
            if count == 0:
                # An Expression can be in any scope (globals/locals), therefore it needs to evaluated as an expression
                var = evaluate_expression(dbg, frame, locator_parts[count], False)
            else:
                _type, _type_name, resolver = get_type(var)
                var = resolver.resolve(var, locator_parts[count])
    else:
        if scope == "GLOBAL":
            var = frame.f_globals
            del locator_parts[0]  # globals are special, and they get a single dummy unused attribute
        else:
            # in a frame access both locals and globals as Python does
            var = {}
            var.update(frame.f_globals)
            var.update(frame.f_locals)

        for k in locator_parts:
            _type, _type_name, resolver = get_type(var)
            var = resolver.resolve(var, k)

    return var


def resolve_compound_variable_fields(dbg, thread_id, frame_id, scope, attrs):
    """
    Resolve compound variable in debugger scopes by its name and attributes

    :param thread_id: id of the variable's thread
    :param frame_id: id of the variable's frame
    :param scope: can be BY_ID, EXPRESSION, GLOBAL, LOCAL, FRAME
    :param attrs: after reaching the proper scope, we have to get the attributes until we find
            the proper location (i.e.: obj\tattr1\tattr2)
    :return: a dictionary of variables's fields
    """

    var = getVariable(dbg, thread_id, frame_id, scope, attrs)

    try:
        _type, type_name, resolver = get_type(var)
        return type_name, resolver.get_dictionary(var)
    except:
        pydev_log.exception('Error evaluating: thread_id: %s\nframe_id: %s\nscope: %s\nattrs: %s.',
            thread_id, frame_id, scope, attrs)


def resolve_var_object(var, attrs):
    """
    Resolve variable's attribute

    :param var: an object of variable
    :param attrs: a sequence of variable's attributes separated by \t (i.e.: obj\tattr1\tattr2)
    :return: a value of resolved variable's attribute
    """
    if attrs is not None:
        attr_list = attrs.split('\t')
    else:
        attr_list = []
    for k in attr_list:
        type, _type_name, resolver = get_type(var)
        var = resolver.resolve(var, k)
    return var


def resolve_compound_var_object_fields(var, attrs):
    """
    Resolve compound variable by its object and attributes

    :param var: an object of variable
    :param attrs: a sequence of variable's attributes separated by \t (i.e.: obj\tattr1\tattr2)
    :return: a dictionary of variables's fields
    """
    attr_list = attrs.split('\t')

    for k in attr_list:
        type, _type_name, resolver = get_type(var)
        var = resolver.resolve(var, k)

    try:
        type, _type_name, resolver = get_type(var)
        return resolver.get_dictionary(var)
    except:
        pydev_log.exception()


def custom_operation(dbg, thread_id, frame_id, scope, attrs, style, code_or_file, operation_fn_name):
    """
    We'll execute the code_or_file and then search in the namespace the operation_fn_name to execute with the given var.

    code_or_file: either some code (i.e.: from pprint import pprint) or a file to be executed.
    operation_fn_name: the name of the operation to execute after the exec (i.e.: pprint)
    """
    expressionValue = getVariable(dbg, thread_id, frame_id, scope, attrs)

    try:
        namespace = {'__name__': '<custom_operation>'}
        if style == "EXECFILE":
            namespace['__file__'] = code_or_file
            execfile(code_or_file, namespace, namespace)
        else:  # style == EXEC
            namespace['__file__'] = '<customOperationCode>'
            Exec(code_or_file, namespace, namespace)

        return str(namespace[operation_fn_name](expressionValue))
    except:
        pydev_log.exception()


@lru_cache(3)
def _expression_to_evaluate(expression):
    keepends = True
    lines = expression.splitlines(keepends)
    # find first non-empty line
    chars_to_strip = 0
    for line in lines:
        if line.strip():  # i.e.: check first non-empty line
            for c in iter_chars(line):
                if c.isspace():
                    chars_to_strip += 1
                else:
                    break
            break

    if chars_to_strip:
        # I.e.: check that the chars we'll remove are really only whitespaces.
        proceed = True
        new_lines = []
        for line in lines:
            if not proceed:
                break
            for c in iter_chars(line[:chars_to_strip]):
                if not c.isspace():
                    proceed = False
                    break

            new_lines.append(line[chars_to_strip:])

        if proceed:
            if isinstance(expression, bytes):
                expression = b''.join(new_lines)
            else:
                expression = u''.join(new_lines)

    return expression


def eval_in_context(expression, global_vars, local_vars, py_db=None):
    result = None
    try:
        compiled = compile_as_eval(expression)
        is_async = inspect.CO_COROUTINE & compiled.co_flags == inspect.CO_COROUTINE

        if is_async:
            if py_db is None:
                py_db = get_global_debugger()
                if py_db is None:
                    raise RuntimeError('Cannot evaluate async without py_db.')
            t = _EvalAwaitInNewEventLoop(py_db, compiled, global_vars, local_vars)
            t.start()
            t.join()

            if t.exc:
                raise t.exc[1].with_traceback(t.exc[2])
            else:
                result = t.evaluated_value
        else:
            result = eval(compiled, global_vars, local_vars)
    except (Exception, KeyboardInterrupt):
        etype, result, tb = sys.exc_info()
        result = ExceptionOnEvaluate(result, etype, tb)

        # Ok, we have the initial error message, but let's see if we're dealing with a name mangling error...
        try:
            if '.__' in expression:
                # Try to handle '__' name mangling (for simple cases such as self.__variable.__another_var).
                split = expression.split('.')
                entry = split[0]

                if local_vars is None:
                    local_vars = global_vars
                curr = local_vars[entry]  # Note: we want the KeyError if it's not there.
                for entry in split[1:]:
                    if entry.startswith('__') and not hasattr(curr, entry):
                        entry = '_%s%s' % (curr.__class__.__name__, entry)
                    curr = getattr(curr, entry)

                result = curr
        except:
            pass
    return result


def _run_with_interrupt_thread(original_func, py_db, curr_thread, frame, expression, is_exec):
    on_interrupt_threads = None
    timeout_tracker = py_db.timeout_tracker  # : :type timeout_tracker: TimeoutTracker

    interrupt_thread_timeout = pydevd_constants.PYDEVD_INTERRUPT_THREAD_TIMEOUT

    if interrupt_thread_timeout > 0:
        on_interrupt_threads = pydevd_timeout.create_interrupt_this_thread_callback()
        pydev_log.info('Doing evaluate with interrupt threads timeout: %s.', interrupt_thread_timeout)

    if on_interrupt_threads is None:
        return original_func(py_db, frame, expression, is_exec)
    else:
        with timeout_tracker.call_on_timeout(interrupt_thread_timeout, on_interrupt_threads):
            return original_func(py_db, frame, expression, is_exec)


def _run_with_unblock_threads(original_func, py_db, curr_thread, frame, expression, is_exec):
    on_timeout_unblock_threads = None
    timeout_tracker = py_db.timeout_tracker  # : :type timeout_tracker: TimeoutTracker

    if py_db.multi_threads_single_notification:
        unblock_threads_timeout = pydevd_constants.PYDEVD_UNBLOCK_THREADS_TIMEOUT
    else:
        unblock_threads_timeout = -1  # Don't use this if threads are managed individually.

    if unblock_threads_timeout >= 0:
        pydev_log.info('Doing evaluate with unblock threads timeout: %s.', unblock_threads_timeout)
        tid = get_current_thread_id(curr_thread)

        def on_timeout_unblock_threads():
            on_timeout_unblock_threads.called = True
            pydev_log.info('Resuming threads after evaluate timeout.')
            resume_threads('*', except_thread=curr_thread)
            py_db.threads_suspended_single_notification.on_thread_resume(tid, curr_thread)

        on_timeout_unblock_threads.called = False

    try:
        if on_timeout_unblock_threads is None:
            return _run_with_interrupt_thread(original_func, py_db, curr_thread, frame, expression, is_exec)
        else:
            with timeout_tracker.call_on_timeout(unblock_threads_timeout, on_timeout_unblock_threads):
                return _run_with_interrupt_thread(original_func, py_db, curr_thread, frame, expression, is_exec)

    finally:
        if on_timeout_unblock_threads is not None and on_timeout_unblock_threads.called:
            mark_thread_suspended(curr_thread, CMD_SET_BREAK)
            py_db.threads_suspended_single_notification.increment_suspend_time()
            suspend_all_threads(py_db, except_thread=curr_thread)
            py_db.threads_suspended_single_notification.on_thread_suspend(tid, curr_thread, CMD_SET_BREAK)


def _evaluate_with_timeouts(original_func):
    '''
    Provides a decorator that wraps the original evaluate to deal with slow evaluates.

    If some evaluation is too slow, we may show a message, resume threads or interrupt them
    as needed (based on the related configurations).
    '''

    @functools.wraps(original_func)
    def new_func(py_db, frame, expression, is_exec):
        if py_db is None:
            # Only for testing...
            pydev_log.critical('_evaluate_with_timeouts called without py_db!')
            return original_func(py_db, frame, expression, is_exec)
        warn_evaluation_timeout = pydevd_constants.PYDEVD_WARN_EVALUATION_TIMEOUT
        curr_thread = threading.current_thread()

        def on_warn_evaluation_timeout():
            py_db.writer.add_command(py_db.cmd_factory.make_evaluation_timeout_msg(
                py_db, expression, curr_thread))

        timeout_tracker = py_db.timeout_tracker  # : :type timeout_tracker: TimeoutTracker
        with timeout_tracker.call_on_timeout(warn_evaluation_timeout, on_warn_evaluation_timeout):
            return _run_with_unblock_threads(original_func, py_db, curr_thread, frame, expression, is_exec)

    return new_func


_ASYNC_COMPILE_FLAGS = None
try:
    from ast import PyCF_ALLOW_TOP_LEVEL_AWAIT
    _ASYNC_COMPILE_FLAGS = PyCF_ALLOW_TOP_LEVEL_AWAIT
except:
    pass


def compile_as_eval(expression):
    '''

    :param expression:
        The expression to be _compiled.

    :return: code object

    :raises Exception if the expression cannot be evaluated.
    '''
    expression_to_evaluate = _expression_to_evaluate(expression)
    if _ASYNC_COMPILE_FLAGS is not None:
        return compile(expression_to_evaluate, '<string>', 'eval', _ASYNC_COMPILE_FLAGS)
    else:
        return compile(expression_to_evaluate, '<string>', 'eval')


def _compile_as_exec(expression):
    '''

    :param expression:
        The expression to be _compiled.

    :return: code object

    :raises Exception if the expression cannot be evaluated.
    '''
    expression_to_evaluate = _expression_to_evaluate(expression)
    if _ASYNC_COMPILE_FLAGS is not None:
        return compile(expression_to_evaluate, '<string>', 'exec', _ASYNC_COMPILE_FLAGS)
    else:
        return compile(expression_to_evaluate, '<string>', 'exec')


class _EvalAwaitInNewEventLoop(PyDBDaemonThread):

    def __init__(self, py_db, compiled, updated_globals, updated_locals):
        PyDBDaemonThread.__init__(self, py_db)
        self._compiled = compiled
        self._updated_globals = updated_globals
        self._updated_locals = updated_locals

        # Output
        self.evaluated_value = None
        self.exc = None

    async def _async_func(self):
        return await eval(self._compiled, self._updated_locals, self._updated_globals)

    def _on_run(self):
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self.evaluated_value = asyncio.run(self._async_func())
        except:
            self.exc = sys.exc_info()


@_evaluate_with_timeouts
def evaluate_expression(py_db, frame, expression, is_exec):
    '''
    :param str expression:
        The expression to be evaluated.

        Note that if the expression is indented it's automatically dedented (based on the indentation
        found on the first non-empty line).

        i.e.: something as:

        `
            def method():
                a = 1
        `

        becomes:

        `
        def method():
            a = 1
        `

        Also, it's possible to evaluate calls with a top-level await (currently this is done by
        creating a new event loop in a new thread and making the evaluate at that thread -- note
        that this is still done synchronously so the evaluation has to finish before this
        function returns).

    :param is_exec: determines if we should do an exec or an eval.
        There are some changes in this function depending on whether it's an exec or an eval.

        When it's an exec (i.e.: is_exec==True):
            This function returns None.
            Any exception that happens during the evaluation is reraised.
            If the expression could actually be evaluated, the variable is printed to the console if not None.

        When it's an eval (i.e.: is_exec==False):
            This function returns the result from the evaluation.
            If some exception happens in this case, the exception is caught and a ExceptionOnEvaluate is returned.
            Also, in this case we try to resolve name-mangling (i.e.: to be able to add a self.__my_var watch).

    :param py_db:
        The debugger. Only needed if some top-level await is detected (for creating a
        PyDBDaemonThread).
    '''
    if frame is None:
        return

    # This is very tricky. Some statements can change locals and use them in the same
    # call (see https://github.com/microsoft/debugpy/issues/815), also, if locals and globals are
    # passed separately, it's possible that one gets updated but apparently Python will still
    # try to load from the other, so, what's done is that we merge all in a single dict and
    # then go on and update the frame with the results afterwards.

    # -- see tests in test_evaluate_expression.py

    # This doesn't work because the variables aren't updated in the locals in case the
    # evaluation tries to set a variable and use it in the same expression.
    # updated_globals = frame.f_globals
    # updated_locals = frame.f_locals

    # This doesn't work because the variables aren't updated in the locals in case the
    # evaluation tries to set a variable and use it in the same expression.
    # updated_globals = {}
    # updated_globals.update(frame.f_globals)
    # updated_globals.update(frame.f_locals)
    #
    # updated_locals = frame.f_locals

    # This doesn't work either in the case where the evaluation tries to set a variable and use
    # it in the same expression (I really don't know why as it seems like this *should* work
    # in theory but doesn't in practice).
    # updated_globals = {}
    # updated_globals.update(frame.f_globals)
    #
    # updated_locals = {}
    # updated_globals.update(frame.f_locals)

    # This is the only case that worked consistently to run the tests in test_evaluate_expression.py
    # It's a bit unfortunate because although the exec works in this case, we have to manually
    # put the updates in the frame locals afterwards.
    updated_globals = {}
    updated_globals.update(frame.f_globals)
    updated_globals.update(frame.f_locals)
    if 'globals' not in updated_globals:
        # If the user explicitly uses 'globals()' then we provide the
        # frame globals (unless he has shadowed it already).
        updated_globals['globals'] = lambda: frame.f_globals

    initial_globals = updated_globals.copy()

    updated_locals = None

    try:
        expression = expression.replace('@LINE@', '\n')

        if is_exec:
            try:
                # Try to make it an eval (if it is an eval we can print it, otherwise we'll exec it and
                # it will have whatever the user actually did)
                compiled = compile_as_eval(expression)
            except Exception:
                compiled = None

            if compiled is None:
                try:
                    compiled = _compile_as_exec(expression)
                    is_async = inspect.CO_COROUTINE & compiled.co_flags == inspect.CO_COROUTINE
                    if is_async:
                        t = _EvalAwaitInNewEventLoop(py_db, compiled, updated_globals, updated_locals)
                        t.start()
                        t.join()

                        if t.exc:
                            raise t.exc[1].with_traceback(t.exc[2])
                    else:
                        Exec(compiled, updated_globals, updated_locals)
                finally:
                    # Update the globals even if it errored as it may have partially worked.
                    update_globals_and_locals(updated_globals, initial_globals, frame)
            else:
                is_async = inspect.CO_COROUTINE & compiled.co_flags == inspect.CO_COROUTINE
                if is_async:
                    t = _EvalAwaitInNewEventLoop(py_db, compiled, updated_globals, updated_locals)
                    t.start()
                    t.join()

                    if t.exc:
                        raise t.exc[1].with_traceback(t.exc[2])
                    else:
                        result = t.evaluated_value
                else:
                    result = eval(compiled, updated_globals, updated_locals)
                if result is not None:  # Only print if it's not None (as python does)
                    sys.stdout.write('%s\n' % (result,))
            return

        else:
            ret = eval_in_context(expression, updated_globals, updated_locals, py_db)
            try:
                is_exception_returned = ret.__class__ == ExceptionOnEvaluate
            except:
                pass
            else:
                if not is_exception_returned:
                    # i.e.: by using a walrus assignment (:=), expressions can change the locals,
                    # so, make sure that we save the locals back to the frame.
                    update_globals_and_locals(updated_globals, initial_globals, frame)
            return ret
    finally:
        # Should not be kept alive if an exception happens and this frame is kept in the stack.
        del updated_globals
        del updated_locals
        del initial_globals
        del frame


def change_attr_expression(frame, attr, expression, dbg, value=SENTINEL_VALUE):
    '''Changes some attribute in a given frame.
    '''
    if frame is None:
        return

    try:
        expression = expression.replace('@LINE@', '\n')

        if dbg.plugin and value is SENTINEL_VALUE:
            result = dbg.plugin.change_variable(frame, attr, expression)
            if result:
                return result

        if attr[:7] == "Globals":
            attr = attr[8:]
            if attr in frame.f_globals:
                if value is SENTINEL_VALUE:
                    value = eval(expression, frame.f_globals, frame.f_locals)
                frame.f_globals[attr] = value
                return frame.f_globals[attr]
        else:
            if '.' not in attr:  # i.e.: if we have a '.', we're changing some attribute of a local var.
                if pydevd_save_locals.is_save_locals_available():
                    if value is SENTINEL_VALUE:
                        value = eval(expression, frame.f_globals, frame.f_locals)
                    frame.f_locals[attr] = value
                    pydevd_save_locals.save_locals(frame)
                    return frame.f_locals[attr]

            # i.e.: case with '.' or save locals not available (just exec the assignment in the frame).
            if value is SENTINEL_VALUE:
                value = eval(expression, frame.f_globals, frame.f_locals)
            result = value
            Exec('%s=%s' % (attr, expression), frame.f_globals, frame.f_locals)
            return result

    except Exception:
        pydev_log.exception()


MAXIMUM_ARRAY_SIZE = 100
MAX_SLICE_SIZE = 1000


def table_like_struct_to_xml(array, name, roffset, coffset, rows, cols, format):
    _, type_name, _ = get_type(array)
    if type_name == 'ndarray':
        array, metaxml, r, c, f = array_to_meta_xml(array, name, format)
        xml = metaxml
        format = '%' + f
        if rows == -1 and cols == -1:
            rows = r
            cols = c
        xml += array_to_xml(array, roffset, coffset, rows, cols, format)
    elif type_name == 'DataFrame':
        xml = dataframe_to_xml(array, name, roffset, coffset, rows, cols, format)
    else:
        raise VariableError("Do not know how to convert type %s to table" % (type_name))

    return "<xml>%s</xml>" % xml


def array_to_xml(array, roffset, coffset, rows, cols, format):
    xml = ""
    rows = min(rows, MAXIMUM_ARRAY_SIZE)
    cols = min(cols, MAXIMUM_ARRAY_SIZE)

    # there is no obvious rule for slicing (at least 5 choices)
    if len(array) == 1 and (rows > 1 or cols > 1):
        array = array[0]
    if array.size > len(array):
        array = array[roffset:, coffset:]
        rows = min(rows, len(array))
        cols = min(cols, len(array[0]))
        if len(array) == 1:
            array = array[0]
    elif array.size == len(array):
        if roffset == 0 and rows == 1:
            array = array[coffset:]
            cols = min(cols, len(array))
        elif coffset == 0 and cols == 1:
            array = array[roffset:]
            rows = min(rows, len(array))

    xml += "<arraydata rows=\"%s\" cols=\"%s\"/>" % (rows, cols)
    for row in range(rows):
        xml += "<row index=\"%s\"/>" % to_string(row)
        for col in range(cols):
            value = array
            if rows == 1 or cols == 1:
                if rows == 1 and cols == 1:
                    value = array[0]
                else:
                    if rows == 1:
                        dim = col
                    else:
                        dim = row
                    value = array[dim]
                    if "ndarray" in str(type(value)):
                        value = value[0]
            else:
                value = array[row][col]
            value = format % value
            xml += var_to_xml(value, '')
    return xml


def array_to_meta_xml(array, name, format):
    type = array.dtype.kind
    slice = name
    l = len(array.shape)

    # initial load, compute slice
    if format == '%':
        if l > 2:
            slice += '[0]' * (l - 2)
            for r in range(l - 2):
                array = array[0]
        if type == 'f':
            format = '.5f'
        elif type == 'i' or type == 'u':
            format = 'd'
        else:
            format = 's'
    else:
        format = format.replace('%', '')

    l = len(array.shape)
    reslice = ""
    if l > 2:
        raise Exception("%s has more than 2 dimensions." % slice)
    elif l == 1:
        # special case with 1D arrays arr[i, :] - row, but arr[:, i] - column with equal shape and ndim
        # http://stackoverflow.com/questions/16837946/numpy-a-2-rows-1-column-file-loadtxt-returns-1row-2-columns
        # explanation: http://stackoverflow.com/questions/15165170/how-do-i-maintain-row-column-orientation-of-vectors-in-numpy?rq=1
        # we use kind of a hack - get information about memory from C_CONTIGUOUS
        is_row = array.flags['C_CONTIGUOUS']

        if is_row:
            rows = 1
            cols = min(len(array), MAX_SLICE_SIZE)
            if cols < len(array):
                reslice = '[0:%s]' % (cols)
            array = array[0:cols]
        else:
            cols = 1
            rows = min(len(array), MAX_SLICE_SIZE)
            if rows < len(array):
                reslice = '[0:%s]' % (rows)
            array = array[0:rows]
    elif l == 2:
        rows = min(array.shape[-2], MAX_SLICE_SIZE)
        cols = min(array.shape[-1], MAX_SLICE_SIZE)
        if cols < array.shape[-1] or rows < array.shape[-2]:
            reslice = '[0:%s, 0:%s]' % (rows, cols)
        array = array[0:rows, 0:cols]

    # avoid slice duplication
    if not slice.endswith(reslice):
        slice += reslice

    bounds = (0, 0)
    if type in "biufc":
        bounds = (array.min(), array.max())
    xml = '<array slice=\"%s\" rows=\"%s\" cols=\"%s\" format=\"%s\" type=\"%s\" max=\"%s\" min=\"%s\"/>' % \
          (slice, rows, cols, format, type, bounds[1], bounds[0])
    return array, xml, rows, cols, format


def dataframe_to_xml(df, name, roffset, coffset, rows, cols, format):
    """
    :type df: pandas.core.frame.DataFrame
    :type name: str
    :type coffset: int
    :type roffset: int
    :type rows: int
    :type cols: int
    :type format: str


    """
    num_rows = min(df.shape[0], MAX_SLICE_SIZE)
    num_cols = min(df.shape[1], MAX_SLICE_SIZE)
    if (num_rows, num_cols) != df.shape:
        df = df.iloc[0:num_rows, 0: num_cols]
        slice = '.iloc[0:%s, 0:%s]' % (num_rows, num_cols)
    else:
        slice = ''
    slice = name + slice
    xml = '<array slice=\"%s\" rows=\"%s\" cols=\"%s\" format=\"\" type=\"\" max=\"0\" min=\"0\"/>\n' % \
          (slice, num_rows, num_cols)

    if (rows, cols) == (-1, -1):
        rows, cols = num_rows, num_cols

    rows = min(rows, MAXIMUM_ARRAY_SIZE)
    cols = min(min(cols, MAXIMUM_ARRAY_SIZE), num_cols)
    # need to precompute column bounds here before slicing!
    col_bounds = [None] * cols
    for col in range(cols):
        dtype = df.dtypes.iloc[coffset + col].kind
        if dtype in "biufc":
            cvalues = df.iloc[:, coffset + col]
            bounds = (cvalues.min(), cvalues.max())
        else:
            bounds = (0, 0)
        col_bounds[col] = bounds

    df = df.iloc[roffset: roffset + rows, coffset: coffset + cols]
    rows, cols = df.shape

    xml += "<headerdata rows=\"%s\" cols=\"%s\">\n" % (rows, cols)
    format = format.replace('%', '')
    col_formats = []

    get_label = lambda label: str(label) if not isinstance(label, tuple) else '/'.join(map(str, label))

    for col in range(cols):
        dtype = df.dtypes.iloc[col].kind
        if dtype == 'f' and format:
            fmt = format
        elif dtype == 'f':
            fmt = '.5f'
        elif dtype == 'i' or dtype == 'u':
            fmt = 'd'
        else:
            fmt = 's'
        col_formats.append('%' + fmt)
        bounds = col_bounds[col]

        xml += '<colheader index=\"%s\" label=\"%s\" type=\"%s\" format=\"%s\" max=\"%s\" min=\"%s\" />\n' % \
               (str(col), get_label(df.axes[1].values[col]), dtype, fmt, bounds[1], bounds[0])
    for row, label in enumerate(iter(df.axes[0])):
        xml += "<rowheader index=\"%s\" label = \"%s\"/>\n" % \
               (str(row), get_label(label))
    xml += "</headerdata>\n"
    xml += "<arraydata rows=\"%s\" cols=\"%s\"/>\n" % (rows, cols)
    for row in range(rows):
        xml += "<row index=\"%s\"/>\n" % str(row)
        for col in range(cols):
            value = df.iat[row, col]
            value = col_formats[col] % value
            xml += var_to_xml(value, '')
    return xml
