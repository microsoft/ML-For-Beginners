import json
import urllib.parse as urllib_parse


class DebugOptions(object):

    __slots__ = [
        'just_my_code',
        'redirect_output',
        'show_return_value',
        'break_system_exit_zero',
        'django_debug',
        'flask_debug',
        'stop_on_entry',
        'max_exception_stack_frames',
        'gui_event_loop',
        'client_os',
    ]

    def __init__(self):
        self.just_my_code = True
        self.redirect_output = False
        self.show_return_value = False
        self.break_system_exit_zero = False
        self.django_debug = False
        self.flask_debug = False
        self.stop_on_entry = False
        self.max_exception_stack_frames = 0
        self.gui_event_loop = 'matplotlib'
        self.client_os = None

    def to_json(self):
        dct = {}
        for s in self.__slots__:
            dct[s] = getattr(self, s)
        return json.dumps(dct)

    def update_fom_debug_options(self, debug_options):
        if 'DEBUG_STDLIB' in debug_options:
            self.just_my_code = not debug_options.get('DEBUG_STDLIB')

        if 'REDIRECT_OUTPUT' in debug_options:
            self.redirect_output = debug_options.get('REDIRECT_OUTPUT')

        if 'SHOW_RETURN_VALUE' in debug_options:
            self.show_return_value = debug_options.get('SHOW_RETURN_VALUE')

        if 'BREAK_SYSTEMEXIT_ZERO' in debug_options:
            self.break_system_exit_zero = debug_options.get('BREAK_SYSTEMEXIT_ZERO')

        if 'DJANGO_DEBUG' in debug_options:
            self.django_debug = debug_options.get('DJANGO_DEBUG')

        if 'FLASK_DEBUG' in debug_options:
            self.flask_debug = debug_options.get('FLASK_DEBUG')

        if 'STOP_ON_ENTRY' in debug_options:
            self.stop_on_entry = debug_options.get('STOP_ON_ENTRY')

        if 'CLIENT_OS_TYPE' in debug_options:
            self.client_os = debug_options.get('CLIENT_OS_TYPE')

        # Note: _max_exception_stack_frames cannot be set by debug options.

    def update_from_args(self, args):
        if 'justMyCode' in args:
            self.just_my_code = bool_parser(args['justMyCode'])
        else:
            # i.e.: if justMyCode is provided, don't check the deprecated value
            if 'debugStdLib' in args:
                self.just_my_code = not bool_parser(args['debugStdLib'])

        if 'redirectOutput' in args:
            self.redirect_output = bool_parser(args['redirectOutput'])

        if 'showReturnValue' in args:
            self.show_return_value = bool_parser(args['showReturnValue'])

        if 'breakOnSystemExitZero' in args:
            self.break_system_exit_zero = bool_parser(args['breakOnSystemExitZero'])

        if 'django' in args:
            self.django_debug = bool_parser(args['django'])

        if 'flask' in args:
            self.flask_debug = bool_parser(args['flask'])

        if 'jinja' in args:
            self.flask_debug = bool_parser(args['jinja'])

        if 'stopOnEntry' in args:
            self.stop_on_entry = bool_parser(args['stopOnEntry'])

        self.max_exception_stack_frames = int_parser(args.get('maxExceptionStackFrames', 0))

        if 'guiEventLoop' in args:
            self.gui_event_loop = str(args['guiEventLoop'])

        if 'clientOS' in args:
            self.client_os = str(args['clientOS']).upper()


def int_parser(s, default_value=0):
    try:
        return int(s)
    except Exception:
        return default_value


def bool_parser(s):
    return s in ("True", "true", "1", True, 1)


def unquote(s):
    return None if s is None else urllib_parse.unquote(s)


DEBUG_OPTIONS_PARSER = {
    'WAIT_ON_ABNORMAL_EXIT': bool_parser,
    'WAIT_ON_NORMAL_EXIT': bool_parser,
    'BREAK_SYSTEMEXIT_ZERO': bool_parser,
    'REDIRECT_OUTPUT': bool_parser,
    'DJANGO_DEBUG': bool_parser,
    'FLASK_DEBUG': bool_parser,
    'FIX_FILE_PATH_CASE': bool_parser,
    'CLIENT_OS_TYPE': unquote,
    'DEBUG_STDLIB': bool_parser,
    'STOP_ON_ENTRY': bool_parser,
    'SHOW_RETURN_VALUE': bool_parser,
    'MULTIPROCESS': bool_parser,
}

DEBUG_OPTIONS_BY_FLAG = {
    'RedirectOutput': 'REDIRECT_OUTPUT=True',
    'WaitOnNormalExit': 'WAIT_ON_NORMAL_EXIT=True',
    'WaitOnAbnormalExit': 'WAIT_ON_ABNORMAL_EXIT=True',
    'BreakOnSystemExitZero': 'BREAK_SYSTEMEXIT_ZERO=True',
    'Django': 'DJANGO_DEBUG=True',
    'Flask': 'FLASK_DEBUG=True',
    'Jinja': 'FLASK_DEBUG=True',
    'FixFilePathCase': 'FIX_FILE_PATH_CASE=True',
    'DebugStdLib': 'DEBUG_STDLIB=True',
    'WindowsClient': 'CLIENT_OS_TYPE=WINDOWS',
    'UnixClient': 'CLIENT_OS_TYPE=UNIX',
    'StopOnEntry': 'STOP_ON_ENTRY=True',
    'ShowReturnValue': 'SHOW_RETURN_VALUE=True',
    'Multiprocess': 'MULTIPROCESS=True',
}


def _build_debug_options(flags):
    """Build string representation of debug options from the launch config."""
    return ';'.join(DEBUG_OPTIONS_BY_FLAG[flag]
                    for flag in flags or []
                    if flag in DEBUG_OPTIONS_BY_FLAG)


def _parse_debug_options(opts):
    """Debug options are semicolon separated key=value pairs
    """
    options = {}
    if not opts:
        return options

    for opt in opts.split(';'):
        try:
            key, value = opt.split('=')
        except ValueError:
            continue
        try:
            options[key] = DEBUG_OPTIONS_PARSER[key](value)
        except KeyError:
            continue

    return options


def _extract_debug_options(opts, flags=None):
    """Return the debug options encoded in the given value.

    "opts" is a semicolon-separated string of "key=value" pairs.
    "flags" is a list of strings.

    If flags is provided then it is used as a fallback.

    The values come from the launch config:

     {
         type:'python',
         request:'launch'|'attach',
         name:'friendly name for debug config',
         debugOptions:[
             'RedirectOutput', 'Django'
         ],
         options:'REDIRECT_OUTPUT=True;DJANGO_DEBUG=True'
     }

    Further information can be found here:

    https://code.visualstudio.com/docs/editor/debugging#_launchjson-attributes
    """
    if not opts:
        opts = _build_debug_options(flags)
    return _parse_debug_options(opts)
