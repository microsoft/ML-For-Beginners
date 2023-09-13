# License: EPL
import os
import re
import sys
from _pydev_bundle._pydev_saved_modules import threading
from _pydevd_bundle.pydevd_constants import get_global_debugger, IS_WINDOWS, IS_JYTHON, get_current_thread_id, \
    sorted_dict_repr, set_global_debugger, DebugInfoHolder
from _pydev_bundle import pydev_log
from contextlib import contextmanager
from _pydevd_bundle import pydevd_constants, pydevd_defaults
from _pydevd_bundle.pydevd_defaults import PydevdCustomization
import ast

try:
    from pathlib import Path
except ImportError:
    Path = None

#===============================================================================
# Things that are dependent on having the pydevd debugger
#===============================================================================

pydev_src_dir = os.path.dirname(os.path.dirname(__file__))

_arg_patch = threading.local()


@contextmanager
def skip_subprocess_arg_patch():
    _arg_patch.apply_arg_patching = False
    try:
        yield
    finally:
        _arg_patch.apply_arg_patching = True


def _get_apply_arg_patching():
    return getattr(_arg_patch, 'apply_arg_patching', True)


def _get_setup_updated_with_protocol_and_ppid(setup, is_exec=False):
    if setup is None:
        setup = {}
    setup = setup.copy()
    # Discard anything related to the protocol (we'll set the the protocol based on the one
    # currently set).
    setup.pop(pydevd_constants.ARGUMENT_HTTP_JSON_PROTOCOL, None)
    setup.pop(pydevd_constants.ARGUMENT_JSON_PROTOCOL, None)
    setup.pop(pydevd_constants.ARGUMENT_QUOTED_LINE_PROTOCOL, None)

    if not is_exec:
        # i.e.: The ppid for the subprocess is the current pid.
        # If it's an exec, keep it what it was.
        setup[pydevd_constants.ARGUMENT_PPID] = os.getpid()

    protocol = pydevd_constants.get_protocol()
    if protocol == pydevd_constants.HTTP_JSON_PROTOCOL:
        setup[pydevd_constants.ARGUMENT_HTTP_JSON_PROTOCOL] = True

    elif protocol == pydevd_constants.JSON_PROTOCOL:
        setup[pydevd_constants.ARGUMENT_JSON_PROTOCOL] = True

    elif protocol == pydevd_constants.QUOTED_LINE_PROTOCOL:
        setup[pydevd_constants.ARGUMENT_QUOTED_LINE_PROTOCOL] = True

    elif protocol == pydevd_constants.HTTP_PROTOCOL:
        setup[pydevd_constants.ARGUMENT_HTTP_PROTOCOL] = True

    else:
        pydev_log.debug('Unexpected protocol: %s', protocol)

    mode = pydevd_defaults.PydevdCustomization.DEBUG_MODE
    if mode:
        setup['debug-mode'] = mode

    preimport = pydevd_defaults.PydevdCustomization.PREIMPORT
    if preimport:
        setup['preimport'] = preimport

    if DebugInfoHolder.PYDEVD_DEBUG_FILE:
        setup['log-file'] = DebugInfoHolder.PYDEVD_DEBUG_FILE

    if DebugInfoHolder.DEBUG_TRACE_LEVEL:
        setup['log-level'] = DebugInfoHolder.DEBUG_TRACE_LEVEL

    return setup


class _LastFutureImportFinder(ast.NodeVisitor):

    def __init__(self):
        self.last_future_import_found = None

    def visit_ImportFrom(self, node):
        if node.module == '__future__':
            self.last_future_import_found = node


def _get_offset_from_line_col(code, line, col):
    offset = 0
    for i, line_contents in enumerate(code.splitlines(True)):
        if i == line:
            offset += col
            return offset
        else:
            offset += len(line_contents)

    return -1


def _separate_future_imports(code):
    '''
    :param code:
        The code from where we want to get the __future__ imports (note that it's possible that
        there's no such entry).

    :return tuple(str, str):
        The return is a tuple(future_import, code).

        If the future import is not available a return such as ('', code) is given, otherwise, the
        future import will end with a ';' (so that it can be put right before the pydevd attach
        code).
    '''
    try:
        node = ast.parse(code, '<string>', 'exec')
        visitor = _LastFutureImportFinder()
        visitor.visit(node)

        if visitor.last_future_import_found is None:
            return '', code

        node = visitor.last_future_import_found
        offset = -1
        if hasattr(node, 'end_lineno') and hasattr(node, 'end_col_offset'):
            # Python 3.8 onwards has these (so, use when possible).
            line, col = node.end_lineno, node.end_col_offset
            offset = _get_offset_from_line_col(code, line - 1, col)  # ast lines are 1-based, make it 0-based.

        else:
            # end line/col not available, let's just find the offset and then search
            # for the alias from there.
            line, col = node.lineno, node.col_offset
            offset = _get_offset_from_line_col(code, line - 1, col)  # ast lines are 1-based, make it 0-based.
            if offset >= 0 and node.names:
                from_future_import_name = node.names[-1].name
                i = code.find(from_future_import_name, offset)
                if i < 0:
                    offset = -1
                else:
                    offset = i + len(from_future_import_name)

        if offset >= 0:
            for i in range(offset, len(code)):
                if code[i] in (' ', '\t', ';', ')', '\n'):
                    offset += 1
                else:
                    break

            future_import = code[:offset]
            code_remainder = code[offset:]

            # Now, put '\n' lines back into the code remainder (we had to search for
            # `\n)`, but in case we just got the `\n`, it should be at the remainder,
            # not at the future import.
            while future_import.endswith('\n'):
                future_import = future_import[:-1]
                code_remainder = '\n' + code_remainder

            if not future_import.endswith(';'):
                future_import += ';'
            return future_import, code_remainder

        # This shouldn't happen...
        pydev_log.info('Unable to find line %s in code:\n%r', line, code)
        return '', code

    except:
        pydev_log.exception('Error getting from __future__ imports from: %r', code)
        return '', code


def _get_python_c_args(host, port, code, args, setup):
    setup = _get_setup_updated_with_protocol_and_ppid(setup)

    # i.e.: We want to make the repr sorted so that it works in tests.
    setup_repr = setup if setup is None else (sorted_dict_repr(setup))

    future_imports = ''
    if '__future__' in code:
        # If the code has a __future__ import, we need to be able to strip the __future__
        # imports from the code and add them to the start of our code snippet.
        future_imports, code = _separate_future_imports(code)

    return ("%simport sys; sys.path.insert(0, r'%s'); import pydevd; pydevd.config(%r, %r); "
            "pydevd.settrace(host=%r, port=%s, suspend=False, trace_only_current_thread=False, patch_multiprocessing=True, access_token=%r, client_access_token=%r, __setup_holder__=%s); "
            "%s"
            ) % (
               future_imports,
               pydev_src_dir,
               pydevd_constants.get_protocol(),
               PydevdCustomization.DEBUG_MODE,
               host,
               port,
               setup.get('access-token'),
               setup.get('client-access-token'),
               setup_repr,
               code)


def _get_host_port():
    import pydevd
    host, port = pydevd.dispatch()
    return host, port


def _is_managed_arg(arg):
    pydevd_py = _get_str_type_compatible(arg, 'pydevd.py')
    if arg.endswith(pydevd_py):
        return True
    return False


def _on_forked_process(setup_tracing=True):
    pydevd_constants.after_fork()
    pydev_log.initialize_debug_stream(reinitialize=True)

    if setup_tracing:
        pydev_log.debug('pydevd on forked process: %s', os.getpid())

    import pydevd
    pydevd.threadingCurrentThread().__pydevd_main_thread = True
    pydevd.settrace_forked(setup_tracing=setup_tracing)


def _on_set_trace_for_new_thread(global_debugger):
    if global_debugger is not None:
        global_debugger.enable_tracing()


def _get_str_type_compatible(s, args):
    '''
    This method converts `args` to byte/unicode based on the `s' type.
    '''
    if isinstance(args, (list, tuple)):
        ret = []
        for arg in args:
            if type(s) == type(arg):
                ret.append(arg)
            else:
                if isinstance(s, bytes):
                    ret.append(arg.encode('utf-8'))
                else:
                    ret.append(arg.decode('utf-8'))
        return ret
    else:
        if type(s) == type(args):
            return args
        else:
            if isinstance(s, bytes):
                return args.encode('utf-8')
            else:
                return args.decode('utf-8')


#===============================================================================
# Things related to monkey-patching
#===============================================================================
def is_python(path):
    single_quote, double_quote = _get_str_type_compatible(path, ["'", '"'])

    if path.endswith(single_quote) or path.endswith(double_quote):
        path = path[1:len(path) - 1]
    filename = os.path.basename(path).lower()
    for name in _get_str_type_compatible(filename, ['python', 'jython', 'pypy']):
        if filename.find(name) != -1:
            return True

    return False


class InvalidTypeInArgsException(Exception):
    pass


def remove_quotes_from_args(args):
    if sys.platform == "win32":
        new_args = []

        for x in args:
            if Path is not None and isinstance(x, Path):
                x = str(x)
            else:
                if not isinstance(x, (bytes, str)):
                    raise InvalidTypeInArgsException(str(type(x)))

            double_quote, two_double_quotes = _get_str_type_compatible(x, ['"', '""'])

            if x != two_double_quotes:
                if len(x) > 1 and x.startswith(double_quote) and x.endswith(double_quote):
                    x = x[1:-1]

            new_args.append(x)
        return new_args
    else:
        new_args = []
        for x in args:
            if Path is not None and isinstance(x, Path):
                x = x.as_posix()
            else:
                if not isinstance(x, (bytes, str)):
                    raise InvalidTypeInArgsException(str(type(x)))
            new_args.append(x)

        return new_args


def quote_arg_win32(arg):
    fix_type = lambda x: _get_str_type_compatible(arg, x)

    # See if we need to quote at all - empty strings need quoting, as do strings
    # with whitespace or quotes in them. Backslashes do not need quoting.
    if arg and not set(arg).intersection(fix_type(' "\t\n\v')):
        return arg

    # Per https://docs.microsoft.com/en-us/windows/desktop/api/shellapi/nf-shellapi-commandlinetoargvw,
    # the standard way to interpret arguments in double quotes is as follows:
    #
    #       2N backslashes followed by a quotation mark produce N backslashes followed by
    #       begin/end quote. This does not become part of the parsed argument, but toggles
    #       the "in quotes" mode.
    #
    #       2N+1 backslashes followed by a quotation mark again produce N backslashes followed
    #       by a quotation mark literal ("). This does not toggle the "in quotes" mode.
    #
    #       N backslashes not followed by a quotation mark simply produce N backslashes.
    #
    # This code needs to do the reverse transformation, thus:
    #
    #       N backslashes followed by " produce 2N+1 backslashes followed by "
    #
    #       N backslashes at the end (i.e. where the closing " goes) produce 2N backslashes.
    #
    #       N backslashes in any other position remain as is.

    arg = re.sub(fix_type(r'(\\*)\"'), fix_type(r'\1\1\\"'), arg)
    arg = re.sub(fix_type(r'(\\*)$'), fix_type(r'\1\1'), arg)
    return fix_type('"') + arg + fix_type('"')


def quote_args(args):
    if sys.platform == "win32":
        return list(map(quote_arg_win32, args))
    else:
        return args


def patch_args(args, is_exec=False):
    '''
    :param list args:
        Arguments to patch.

    :param bool is_exec:
        If it's an exec, the current process will be replaced (this means we have
        to keep the same ppid).
    '''
    try:
        pydev_log.debug("Patching args: %s", args)
        original_args = args
        try:
            unquoted_args = remove_quotes_from_args(args)
        except InvalidTypeInArgsException as e:
            pydev_log.info('Unable to monkey-patch subprocess arguments because a type found in the args is invalid: %s', e)
            return original_args

        # Internally we should reference original_args (if we want to return them) or unquoted_args
        # to add to the list which will be then quoted in the end.
        del args

        from pydevd import SetupHolder
        if not unquoted_args:
            return original_args

        if not is_python(unquoted_args[0]):
            pydev_log.debug("Process is not python, returning.")
            return original_args

        # Note: we create a copy as string to help with analyzing the arguments, but
        # the final list should have items from the unquoted_args as they were initially.
        args_as_str = _get_str_type_compatible('', unquoted_args)

        params_with_value_in_separate_arg = (
            '--check-hash-based-pycs',
            '--jit'  # pypy option
        )

        # All short switches may be combined together. The ones below require a value and the
        # value itself may be embedded in the arg.
        #
        # i.e.: Python accepts things as:
        #
        # python -OQold -qmtest
        #
        # Which is the same as:
        #
        # python -O -Q old -q -m test
        #
        # or even:
        #
        # python -OQold "-vcimport sys;print(sys)"
        #
        # Which is the same as:
        #
        # python -O -Q old -v -c "import sys;print(sys)"

        params_with_combinable_arg = set(('W', 'X', 'Q', 'c', 'm'))

        module_name = None
        before_module_flag = ''
        module_name_i_start = -1
        module_name_i_end = -1

        code = None
        code_i = -1
        code_i_end = -1
        code_flag = ''

        filename = None
        filename_i = -1

        ignore_next = True  # start ignoring the first (the first entry is the python executable)
        for i, arg_as_str in enumerate(args_as_str):
            if ignore_next:
                ignore_next = False
                continue

            if arg_as_str.startswith('-'):
                if arg_as_str == '-':
                    # Contents will be read from the stdin. This is not currently handled.
                    pydev_log.debug('Unable to fix arguments to attach debugger on subprocess when reading from stdin ("python ... -").')
                    return original_args

                if arg_as_str.startswith(params_with_value_in_separate_arg):
                    if arg_as_str in params_with_value_in_separate_arg:
                        ignore_next = True
                    continue

                break_out = False
                for j, c in enumerate(arg_as_str):

                    # i.e.: Python supports -X faulthandler as well as -Xfaulthandler
                    # (in one case we have to ignore the next and in the other we don't
                    # have to ignore it).
                    if c in params_with_combinable_arg:
                        remainder = arg_as_str[j + 1:]
                        if not remainder:
                            ignore_next = True

                        if c == 'm':
                            # i.e.: Something as
                            # python -qm test
                            # python -m test
                            # python -qmtest
                            before_module_flag = arg_as_str[:j]  # before_module_flag would then be "-q"
                            if before_module_flag == '-':
                                before_module_flag = ''
                            module_name_i_start = i
                            if not remainder:
                                module_name = unquoted_args[i + 1]
                                module_name_i_end = i + 1
                            else:
                                # i.e.: python -qmtest should provide 'test' as the module_name
                                module_name = unquoted_args[i][j + 1:]
                                module_name_i_end = module_name_i_start
                            break_out = True
                            break

                        elif c == 'c':
                            # i.e.: Something as
                            # python -qc "import sys"
                            # python -c "import sys"
                            # python "-qcimport sys"
                            code_flag = arg_as_str[:j + 1]  # code_flag would then be "-qc"

                            if not remainder:
                                # arg_as_str is something as "-qc", "import sys"
                                code = unquoted_args[i + 1]
                                code_i_end = i + 2
                            else:
                                # if arg_as_str is something as "-qcimport sys"
                                code = remainder  # code would be "import sys"
                                code_i_end = i + 1
                            code_i = i
                            break_out = True
                            break

                        else:
                            break

                if break_out:
                    break

            else:
                # It doesn't start with '-' and we didn't ignore this entry:
                # this means that this is the file to be executed.
                filename = unquoted_args[i]

                # Note that the filename is not validated here.
                # There are cases where even a .exe is valid (xonsh.exe):
                # https://github.com/microsoft/debugpy/issues/945
                # So, we should support whatever runpy.run_path
                # supports in this case.

                filename_i = i

                if _is_managed_arg(filename):  # no need to add pydevd twice
                    pydev_log.debug('Skipped monkey-patching as pydevd.py is in args already.')
                    return original_args

                break
        else:
            # We didn't find the filename (something is unexpected).
            pydev_log.debug('Unable to fix arguments to attach debugger on subprocess (filename not found).')
            return original_args

        if code_i != -1:
            host, port = _get_host_port()

            if port is not None:
                new_args = []
                new_args.extend(unquoted_args[:code_i])
                new_args.append(code_flag)
                new_args.append(_get_python_c_args(host, port, code, unquoted_args, SetupHolder.setup))
                new_args.extend(unquoted_args[code_i_end:])

                return quote_args(new_args)

        first_non_vm_index = max(filename_i, module_name_i_start)
        if first_non_vm_index == -1:
            pydev_log.debug('Unable to fix arguments to attach debugger on subprocess (could not resolve filename nor module name).')
            return original_args

        # Original args should be something as:
        # ['X:\\pysrc\\pydevd.py', '--multiprocess', '--print-in-debugger-startup',
        #  '--vm_type', 'python', '--client', '127.0.0.1', '--port', '56352', '--file', 'x:\\snippet1.py']
        from _pydevd_bundle.pydevd_command_line_handling import setup_to_argv
        new_args = []
        new_args.extend(unquoted_args[:first_non_vm_index])
        if before_module_flag:
            new_args.append(before_module_flag)

        add_module_at = len(new_args) + 1

        new_args.extend(setup_to_argv(
            _get_setup_updated_with_protocol_and_ppid(SetupHolder.setup, is_exec=is_exec),
            skip_names=set(('module', 'cmd-line'))
        ))
        new_args.append('--file')

        if module_name is not None:
            assert module_name_i_start != -1
            assert module_name_i_end != -1
            # Always after 'pydevd' (i.e.: pydevd "--module" --multiprocess ...)
            new_args.insert(add_module_at, '--module')
            new_args.append(module_name)
            new_args.extend(unquoted_args[module_name_i_end + 1:])

        elif filename is not None:
            assert filename_i != -1
            new_args.append(filename)
            new_args.extend(unquoted_args[filename_i + 1:])

        else:
            raise AssertionError('Internal error (unexpected condition)')

        return quote_args(new_args)
    except:
        pydev_log.exception('Error patching args (debugger not attached to subprocess).')
        return original_args


def str_to_args_windows(args):
    # See https://docs.microsoft.com/en-us/cpp/c-language/parsing-c-command-line-arguments.
    #
    # Implemetation ported from DebugPlugin.parseArgumentsWindows:
    # https://github.com/eclipse/eclipse.platform.debug/blob/master/org.eclipse.debug.core/core/org/eclipse/debug/core/DebugPlugin.java

    result = []

    DEFAULT = 0
    ARG = 1
    IN_DOUBLE_QUOTE = 2

    state = DEFAULT
    backslashes = 0
    buf = ''

    args_len = len(args)
    for i in range(args_len):
        ch = args[i]
        if (ch == '\\'):
            backslashes += 1
            continue
        elif (backslashes != 0):
            if ch == '"':
                while backslashes >= 2:
                    backslashes -= 2
                    buf += '\\'
                if (backslashes == 1):
                    if (state == DEFAULT):
                        state = ARG

                    buf += '"'
                    backslashes = 0
                    continue
                # else fall through to switch
            else:
                # false alarm, treat passed backslashes literally...
                if (state == DEFAULT):
                    state = ARG

                while backslashes > 0:
                    backslashes -= 1
                    buf += '\\'
                # fall through to switch
        if ch in (' ', '\t'):
            if (state == DEFAULT):
                # skip
                continue
            elif (state == ARG):
                state = DEFAULT
                result.append(buf)
                buf = ''
                continue

        if state in (DEFAULT, ARG):
            if ch == '"':
                state = IN_DOUBLE_QUOTE
            else:
                state = ARG
                buf += ch

        elif state == IN_DOUBLE_QUOTE:
            if ch == '"':
                if (i + 1 < args_len and args[i + 1] == '"'):
                    # Undocumented feature in Windows:
                    # Two consecutive double quotes inside a double-quoted argument are interpreted as
                    # a single double quote.
                    buf += '"'
                    i += 1
                else:
                    state = ARG
            else:
                buf += ch

        else:
            raise RuntimeError('Illegal condition')

    if len(buf) > 0 or state != DEFAULT:
        result.append(buf)

    return result


def patch_arg_str_win(arg_str):
    args = str_to_args_windows(arg_str)
    # Fix https://youtrack.jetbrains.com/issue/PY-9767 (args may be empty)
    if not args or not is_python(args[0]):
        return arg_str
    arg_str = ' '.join(patch_args(args))
    pydev_log.debug("New args: %s", arg_str)
    return arg_str


def monkey_patch_module(module, funcname, create_func):
    if hasattr(module, funcname):
        original_name = 'original_' + funcname
        if not hasattr(module, original_name):
            setattr(module, original_name, getattr(module, funcname))
            setattr(module, funcname, create_func(original_name))


def monkey_patch_os(funcname, create_func):
    monkey_patch_module(os, funcname, create_func)


def warn_multiproc():
    pass  # TODO: Provide logging as messages to the IDE.
    # pydev_log.error_once(
    #     "pydev debugger: New process is launching (breakpoints won't work in the new process).\n"
    #     "pydev debugger: To debug that process please enable 'Attach to subprocess automatically while debugging?' option in the debugger settings.\n")
    #


def create_warn_multiproc(original_name):

    def new_warn_multiproc(*args, **kwargs):
        import os

        warn_multiproc()

        return getattr(os, original_name)(*args, **kwargs)

    return new_warn_multiproc


def create_execl(original_name):

    def new_execl(path, *args):
        """
        os.execl(path, arg0, arg1, ...)
        os.execle(path, arg0, arg1, ..., env)
        os.execlp(file, arg0, arg1, ...)
        os.execlpe(file, arg0, arg1, ..., env)
        """
        if _get_apply_arg_patching():
            args = patch_args(args, is_exec=True)
            send_process_created_message()
            send_process_about_to_be_replaced()

        return getattr(os, original_name)(path, *args)

    return new_execl


def create_execv(original_name):

    def new_execv(path, args):
        """
        os.execv(path, args)
        os.execvp(file, args)
        """
        if _get_apply_arg_patching():
            args = patch_args(args, is_exec=True)
            send_process_created_message()
            send_process_about_to_be_replaced()

        return getattr(os, original_name)(path, args)

    return new_execv


def create_execve(original_name):
    """
    os.execve(path, args, env)
    os.execvpe(file, args, env)
    """

    def new_execve(path, args, env):
        if _get_apply_arg_patching():
            args = patch_args(args, is_exec=True)
            send_process_created_message()
            send_process_about_to_be_replaced()

        return getattr(os, original_name)(path, args, env)

    return new_execve


def create_spawnl(original_name):

    def new_spawnl(mode, path, *args):
        """
        os.spawnl(mode, path, arg0, arg1, ...)
        os.spawnlp(mode, file, arg0, arg1, ...)
        """
        if _get_apply_arg_patching():
            args = patch_args(args)
            send_process_created_message()

        return getattr(os, original_name)(mode, path, *args)

    return new_spawnl


def create_spawnv(original_name):

    def new_spawnv(mode, path, args):
        """
        os.spawnv(mode, path, args)
        os.spawnvp(mode, file, args)
        """
        if _get_apply_arg_patching():
            args = patch_args(args)
            send_process_created_message()

        return getattr(os, original_name)(mode, path, args)

    return new_spawnv


def create_spawnve(original_name):
    """
    os.spawnve(mode, path, args, env)
    os.spawnvpe(mode, file, args, env)
    """

    def new_spawnve(mode, path, args, env):
        if _get_apply_arg_patching():
            args = patch_args(args)
            send_process_created_message()

        return getattr(os, original_name)(mode, path, args, env)

    return new_spawnve


def create_posix_spawn(original_name):
    """
    os.posix_spawn(executable, args, env, **kwargs)
    """

    def new_posix_spawn(executable, args, env, **kwargs):
        if _get_apply_arg_patching():
            args = patch_args(args)
            send_process_created_message()

        return getattr(os, original_name)(executable, args, env, **kwargs)

    return new_posix_spawn


def create_fork_exec(original_name):
    """
    _posixsubprocess.fork_exec(args, executable_list, close_fds, ... (13 more))
    """

    def new_fork_exec(args, *other_args):
        import _posixsubprocess  # @UnresolvedImport
        if _get_apply_arg_patching():
            args = patch_args(args)
            send_process_created_message()

        return getattr(_posixsubprocess, original_name)(args, *other_args)

    return new_fork_exec


def create_warn_fork_exec(original_name):
    """
    _posixsubprocess.fork_exec(args, executable_list, close_fds, ... (13 more))
    """

    def new_warn_fork_exec(*args):
        try:
            import _posixsubprocess
            warn_multiproc()
            return getattr(_posixsubprocess, original_name)(*args)
        except:
            pass

    return new_warn_fork_exec


def create_subprocess_fork_exec(original_name):
    """
    subprocess._fork_exec(args, executable_list, close_fds, ... (13 more))
    """

    def new_fork_exec(args, *other_args):
        import subprocess
        if _get_apply_arg_patching():
            args = patch_args(args)
            send_process_created_message()

        return getattr(subprocess, original_name)(args, *other_args)

    return new_fork_exec


def create_subprocess_warn_fork_exec(original_name):
    """
    subprocess._fork_exec(args, executable_list, close_fds, ... (13 more))
    """

    def new_warn_fork_exec(*args):
        try:
            import subprocess
            warn_multiproc()
            return getattr(subprocess, original_name)(*args)
        except:
            pass

    return new_warn_fork_exec


def create_CreateProcess(original_name):
    """
    CreateProcess(*args, **kwargs)
    """

    def new_CreateProcess(app_name, cmd_line, *args):
        try:
            import _subprocess
        except ImportError:
            import _winapi as _subprocess

        if _get_apply_arg_patching():
            cmd_line = patch_arg_str_win(cmd_line)
            send_process_created_message()

        return getattr(_subprocess, original_name)(app_name, cmd_line, *args)

    return new_CreateProcess


def create_CreateProcessWarnMultiproc(original_name):
    """
    CreateProcess(*args, **kwargs)
    """

    def new_CreateProcess(*args):
        try:
            import _subprocess
        except ImportError:
            import _winapi as _subprocess
        warn_multiproc()
        return getattr(_subprocess, original_name)(*args)

    return new_CreateProcess


def create_fork(original_name):

    def new_fork():
        # A simple fork will result in a new python process
        is_new_python_process = True
        frame = sys._getframe()

        apply_arg_patch = _get_apply_arg_patching()

        is_subprocess_fork = False
        while frame is not None:
            if frame.f_code.co_name == '_execute_child' and 'subprocess' in frame.f_code.co_filename:
                is_subprocess_fork = True
                # If we're actually in subprocess.Popen creating a child, it may
                # result in something which is not a Python process, (so, we
                # don't want to connect with it in the forked version).
                executable = frame.f_locals.get('executable')
                if executable is not None:
                    is_new_python_process = False
                    if is_python(executable):
                        is_new_python_process = True
                break

            frame = frame.f_back
        frame = None  # Just make sure we don't hold on to it.

        protocol = pydevd_constants.get_protocol()
        debug_mode = PydevdCustomization.DEBUG_MODE

        child_process = getattr(os, original_name)()  # fork
        if not child_process:
            if is_new_python_process:
                PydevdCustomization.DEFAULT_PROTOCOL = protocol
                PydevdCustomization.DEBUG_MODE = debug_mode
                _on_forked_process(setup_tracing=apply_arg_patch and not is_subprocess_fork)
            else:
                set_global_debugger(None)
        else:
            if is_new_python_process:
                send_process_created_message()
        return child_process

    return new_fork


def send_process_created_message():
    py_db = get_global_debugger()
    if py_db is not None:
        py_db.send_process_created_message()


def send_process_about_to_be_replaced():
    py_db = get_global_debugger()
    if py_db is not None:
        py_db.send_process_about_to_be_replaced()


def patch_new_process_functions():
    # os.execl(path, arg0, arg1, ...)
    # os.execle(path, arg0, arg1, ..., env)
    # os.execlp(file, arg0, arg1, ...)
    # os.execlpe(file, arg0, arg1, ..., env)
    # os.execv(path, args)
    # os.execve(path, args, env)
    # os.execvp(file, args)
    # os.execvpe(file, args, env)
    monkey_patch_os('execl', create_execl)
    monkey_patch_os('execle', create_execl)
    monkey_patch_os('execlp', create_execl)
    monkey_patch_os('execlpe', create_execl)
    monkey_patch_os('execv', create_execv)
    monkey_patch_os('execve', create_execve)
    monkey_patch_os('execvp', create_execv)
    monkey_patch_os('execvpe', create_execve)

    # os.spawnl(mode, path, ...)
    # os.spawnle(mode, path, ..., env)
    # os.spawnlp(mode, file, ...)
    # os.spawnlpe(mode, file, ..., env)
    # os.spawnv(mode, path, args)
    # os.spawnve(mode, path, args, env)
    # os.spawnvp(mode, file, args)
    # os.spawnvpe(mode, file, args, env)

    monkey_patch_os('spawnl', create_spawnl)
    monkey_patch_os('spawnle', create_spawnl)
    monkey_patch_os('spawnlp', create_spawnl)
    monkey_patch_os('spawnlpe', create_spawnl)
    monkey_patch_os('spawnv', create_spawnv)
    monkey_patch_os('spawnve', create_spawnve)
    monkey_patch_os('spawnvp', create_spawnv)
    monkey_patch_os('spawnvpe', create_spawnve)
    monkey_patch_os('posix_spawn', create_posix_spawn)

    if not IS_JYTHON:
        if not IS_WINDOWS:
            monkey_patch_os('fork', create_fork)
            try:
                import _posixsubprocess
                monkey_patch_module(_posixsubprocess, 'fork_exec', create_fork_exec)
            except ImportError:
                pass

            try:
                import subprocess
                monkey_patch_module(subprocess, '_fork_exec', create_subprocess_fork_exec)
            except AttributeError:
                pass
        else:
            # Windows
            try:
                import _subprocess
            except ImportError:
                import _winapi as _subprocess
            monkey_patch_module(_subprocess, 'CreateProcess', create_CreateProcess)


def patch_new_process_functions_with_warning():
    monkey_patch_os('execl', create_warn_multiproc)
    monkey_patch_os('execle', create_warn_multiproc)
    monkey_patch_os('execlp', create_warn_multiproc)
    monkey_patch_os('execlpe', create_warn_multiproc)
    monkey_patch_os('execv', create_warn_multiproc)
    monkey_patch_os('execve', create_warn_multiproc)
    monkey_patch_os('execvp', create_warn_multiproc)
    monkey_patch_os('execvpe', create_warn_multiproc)
    monkey_patch_os('spawnl', create_warn_multiproc)
    monkey_patch_os('spawnle', create_warn_multiproc)
    monkey_patch_os('spawnlp', create_warn_multiproc)
    monkey_patch_os('spawnlpe', create_warn_multiproc)
    monkey_patch_os('spawnv', create_warn_multiproc)
    monkey_patch_os('spawnve', create_warn_multiproc)
    monkey_patch_os('spawnvp', create_warn_multiproc)
    monkey_patch_os('spawnvpe', create_warn_multiproc)
    monkey_patch_os('posix_spawn', create_warn_multiproc)

    if not IS_JYTHON:
        if not IS_WINDOWS:
            monkey_patch_os('fork', create_warn_multiproc)
            try:
                import _posixsubprocess
                monkey_patch_module(_posixsubprocess, 'fork_exec', create_warn_fork_exec)
            except ImportError:
                pass

            try:
                import subprocess
                monkey_patch_module(subprocess, '_fork_exec', create_subprocess_warn_fork_exec)
            except AttributeError:
                pass

        else:
            # Windows
            try:
                import _subprocess
            except ImportError:
                import _winapi as _subprocess
            monkey_patch_module(_subprocess, 'CreateProcess', create_CreateProcessWarnMultiproc)


class _NewThreadStartupWithTrace:

    def __init__(self, original_func, args, kwargs):
        self.original_func = original_func
        self.args = args
        self.kwargs = kwargs

    def __call__(self):
        # We monkey-patch the thread creation so that this function is called in the new thread. At this point
        # we notify of its creation and start tracing it.
        py_db = get_global_debugger()

        thread_id = None
        if py_db is not None:
            # Note: if this is a thread from threading.py, we're too early in the boostrap process (because we mocked
            # the start_new_thread internal machinery and thread._bootstrap has not finished), so, the code below needs
            # to make sure that we use the current thread bound to the original function and not use
            # threading.current_thread() unless we're sure it's a dummy thread.
            t = getattr(self.original_func, '__self__', getattr(self.original_func, 'im_self', None))
            if not isinstance(t, threading.Thread):
                # This is not a threading.Thread but a Dummy thread (so, get it as a dummy thread using
                # currentThread).
                t = threading.current_thread()

            if not getattr(t, 'is_pydev_daemon_thread', False):
                thread_id = get_current_thread_id(t)
                py_db.notify_thread_created(thread_id, t)
                _on_set_trace_for_new_thread(py_db)

            if getattr(py_db, 'thread_analyser', None) is not None:
                try:
                    from _pydevd_bundle.pydevd_concurrency_analyser.pydevd_concurrency_logger import log_new_thread
                    log_new_thread(py_db, t)
                except:
                    sys.stderr.write("Failed to detect new thread for visualization")
        try:
            ret = self.original_func(*self.args, **self.kwargs)
        finally:
            if thread_id is not None:
                if py_db is not None:
                    # At thread shutdown we only have pydevd-related code running (which shouldn't
                    # be tracked).
                    py_db.disable_tracing()
                    py_db.notify_thread_not_alive(thread_id)

        return ret


class _NewThreadStartupWithoutTrace:

    def __init__(self, original_func, args, kwargs):
        self.original_func = original_func
        self.args = args
        self.kwargs = kwargs

    def __call__(self):
        return self.original_func(*self.args, **self.kwargs)


_UseNewThreadStartup = _NewThreadStartupWithTrace


def _get_threading_modules_to_patch():
    threading_modules_to_patch = []

    try:
        import thread as _thread
    except:
        import _thread
    threading_modules_to_patch.append(_thread)
    threading_modules_to_patch.append(threading)

    return threading_modules_to_patch


threading_modules_to_patch = _get_threading_modules_to_patch()


def patch_thread_module(thread_module):

    if getattr(thread_module, '_original_start_new_thread', None) is None:
        if thread_module is threading:
            if not hasattr(thread_module, '_start_new_thread'):
                return  # Jython doesn't have it.
            _original_start_new_thread = thread_module._original_start_new_thread = thread_module._start_new_thread
        else:
            _original_start_new_thread = thread_module._original_start_new_thread = thread_module.start_new_thread
    else:
        _original_start_new_thread = thread_module._original_start_new_thread

    class ClassWithPydevStartNewThread:

        def pydev_start_new_thread(self, function, args=(), kwargs={}):
            '''
            We need to replace the original thread_module.start_new_thread with this function so that threads started
            through it and not through the threading module are properly traced.
            '''
            return _original_start_new_thread(_UseNewThreadStartup(function, args, kwargs), ())

    # This is a hack for the situation where the thread_module.start_new_thread is declared inside a class, such as the one below
    # class F(object):
    #    start_new_thread = thread_module.start_new_thread
    #
    #    def start_it(self):
    #        self.start_new_thread(self.function, args, kwargs)
    # So, if it's an already bound method, calling self.start_new_thread won't really receive a different 'self' -- it
    # does work in the default case because in builtins self isn't passed either.
    pydev_start_new_thread = ClassWithPydevStartNewThread().pydev_start_new_thread

    try:
        # We need to replace the original thread_module.start_new_thread with this function so that threads started through
        # it and not through the threading module are properly traced.
        if thread_module is threading:
            thread_module._start_new_thread = pydev_start_new_thread
        else:
            thread_module.start_new_thread = pydev_start_new_thread
            thread_module.start_new = pydev_start_new_thread
    except:
        pass


def patch_thread_modules():
    for t in threading_modules_to_patch:
        patch_thread_module(t)


def undo_patch_thread_modules():
    for t in threading_modules_to_patch:
        try:
            t.start_new_thread = t._original_start_new_thread
        except:
            pass

        try:
            t.start_new = t._original_start_new_thread
        except:
            pass

        try:
            t._start_new_thread = t._original_start_new_thread
        except:
            pass


def disable_trace_thread_modules():
    '''
    Can be used to temporarily stop tracing threads created with thread.start_new_thread.
    '''
    global _UseNewThreadStartup
    _UseNewThreadStartup = _NewThreadStartupWithoutTrace


def enable_trace_thread_modules():
    '''
    Can be used to start tracing threads created with thread.start_new_thread again.
    '''
    global _UseNewThreadStartup
    _UseNewThreadStartup = _NewThreadStartupWithTrace


def get_original_start_new_thread(threading_module):
    try:
        return threading_module._original_start_new_thread
    except:
        return threading_module.start_new_thread
