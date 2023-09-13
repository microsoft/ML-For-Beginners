from _pydevd_bundle.pydevd_constants import EXCEPTION_TYPE_USER_UNHANDLED, EXCEPTION_TYPE_UNHANDLED, \
    IS_PY311_OR_GREATER
from _pydev_bundle import pydev_log
import itertools
from typing import Any, Dict


class Frame(object):

    def __init__(
            self,
            f_back,
            f_fileno,
            f_code,
            f_locals,
            f_globals=None,
            f_trace=None):
        self.f_back = f_back
        self.f_lineno = f_fileno
        self.f_code = f_code
        self.f_locals = f_locals
        self.f_globals = f_globals
        self.f_trace = f_trace

        if self.f_globals is None:
            self.f_globals = {}


class FCode(object):

    def __init__(self, name, filename):
        self.co_name = name
        self.co_filename = filename
        self.co_firstlineno = 1
        self.co_flags = 0


def add_exception_to_frame(frame, exception_info):
    frame.f_locals['__exception__'] = exception_info


def remove_exception_from_frame(frame):
    frame.f_locals.pop('__exception__', None)


FILES_WITH_IMPORT_HOOKS = ['pydev_monkey_qt.py', 'pydev_import_hook.py']


def just_raised(trace):
    if trace is None:
        return False
    return trace.tb_next is None


def ignore_exception_trace(trace):
    while trace is not None:
        filename = trace.tb_frame.f_code.co_filename
        if filename in (
            '<frozen importlib._bootstrap>', '<frozen importlib._bootstrap_external>'):
            # Do not stop on inner exceptions in py3 while importing
            return True

        # ImportError should appear in a user's code, not inside debugger
        for file in FILES_WITH_IMPORT_HOOKS:
            if filename.endswith(file):
                return True

        trace = trace.tb_next

    return False


def cached_call(obj, func, *args):
    cached_name = '_cached_' + func.__name__
    if not hasattr(obj, cached_name):
        setattr(obj, cached_name, func(*args))

    return getattr(obj, cached_name)


class _LineColInfo:

    def __init__(self, lineno, end_lineno, colno, end_colno):
        self.lineno = lineno
        self.end_lineno = end_lineno
        self.colno = colno
        self.end_colno = end_colno

    def map_columns_to_line(self, original_line: str):
        '''
        The columns internally are actually based on bytes.

        Also, the position isn't always the ideal one as the start may not be
        what we want (if the user has many subscripts in the line the start
        will always be the same and only the end would change).
        For more details see:
        https://github.com/microsoft/debugpy/issues/1099#issuecomment-1303403995

        So, this function maps the start/end columns to the position to be shown in the editor.
        '''
        colno = _utf8_byte_offset_to_character_offset(original_line, self.colno)
        end_colno = _utf8_byte_offset_to_character_offset(original_line, self.end_colno)

        if self.lineno == self.end_lineno:
            try:
                ret = _extract_caret_anchors_in_bytes_from_line_segment(
                    original_line[colno:end_colno]
                )
                if ret is not None:
                    return (
                        _utf8_byte_offset_to_character_offset(original_line, ret[0] + self.colno),
                        _utf8_byte_offset_to_character_offset(original_line, ret[1] + self.colno)
                    )
            except Exception:
                pass  # Suppress exception

        return colno, end_colno


_utf8_with_2_bytes = 0x80
_utf8_with_3_bytes = 0x800
_utf8_with_4_bytes = 0x10000


def _utf8_byte_offset_to_character_offset(s: str, offset: int):
    byte_offset = 0
    char_offset = 0

    for char_offset, character in enumerate(s):
        byte_offset += 1

        codepoint = ord(character)

        if codepoint >= _utf8_with_4_bytes:
            byte_offset += 3

        elif codepoint >= _utf8_with_3_bytes:
            byte_offset += 2

        elif codepoint >= _utf8_with_2_bytes:
            byte_offset += 1

        if byte_offset > offset:
            break
    else:
        char_offset += 1

    return char_offset


# Based on traceback._extract_caret_anchors_in_bytes_from_line_segment (Python 3.11.0)
def _extract_caret_anchors_in_bytes_from_line_segment(segment: str):
    import ast

    try:
        segment = segment.encode('utf-8')
    except UnicodeEncodeError:
        return None
    try:
        tree = ast.parse(segment)
    except SyntaxError:
        return None

    if len(tree.body) != 1:
        return None

    statement = tree.body[0]
    if isinstance(statement, ast.Expr):
        expr = statement.value
        if isinstance(expr, ast.BinOp):
            operator_str = segment[expr.left.end_col_offset:expr.right.col_offset]
            operator_offset = len(operator_str) - len(operator_str.lstrip())

            left_anchor = expr.left.end_col_offset + operator_offset
            right_anchor = left_anchor + 1
            if (
                operator_offset + 1 < len(operator_str)
                and not operator_str[operator_offset + 1] == ord(b' ')
            ):
                right_anchor += 1
            return left_anchor, right_anchor
        if isinstance(expr, ast.Subscript):
            return expr.value.end_col_offset, expr.slice.end_col_offset + 1

    return None


class FramesList(object):

    def __init__(self):
        self._frames = []

        # If available, the line number for the frame will be gotten from this dict,
        # otherwise frame.f_lineno will be used (needed for unhandled exceptions as
        # the place where we report may be different from the place where it's raised).
        self.frame_id_to_lineno = {}
        self.frame_id_to_line_col_info: Dict[Any, _LineColInfo] = {}

        self.exc_type = None
        self.exc_desc = None
        self.trace_obj = None

        # This may be set to set the current frame (for the case where we have
        # an unhandled exception where we want to show the root bu we have a different
        # executing frame).
        self.current_frame = None

        # This is to know whether an exception was extracted from a __cause__ or __context__.
        self.exc_context_msg = ''

        self.chained_frames_list = None

    def append(self, frame):
        self._frames.append(frame)

    def last_frame(self):
        return self._frames[-1]

    def __len__(self):
        return len(self._frames)

    def __iter__(self):
        return iter(self._frames)

    def __repr__(self):
        lst = ['FramesList(']

        lst.append('\n    exc_type: ')
        lst.append(str(self.exc_type))

        lst.append('\n    exc_desc: ')
        lst.append(str(self.exc_desc))

        lst.append('\n    trace_obj: ')
        lst.append(str(self.trace_obj))

        lst.append('\n    current_frame: ')
        lst.append(str(self.current_frame))

        for frame in self._frames:
            lst.append('\n    ')
            lst.append(repr(frame))
            lst.append(',')

        if self.chained_frames_list is not None:
            lst.append('\n--- Chained ---\n')
            lst.append(str(self.chained_frames_list))

        lst.append('\n)')

        return ''.join(lst)

    __str__ = __repr__


class _DummyFrameWrapper(object):

    def __init__(self, frame, f_lineno, f_back):
        self._base_frame = frame
        self.f_lineno = f_lineno
        self.f_back = f_back
        self.f_trace = None
        original_code = frame.f_code
        name = original_code.co_name
        self.f_code = FCode(name, original_code.co_filename)

    @property
    def f_locals(self):
        return self._base_frame.f_locals

    @property
    def f_globals(self):
        return self._base_frame.f_globals

    def __str__(self):
        return "<_DummyFrameWrapper, file '%s', line %s, %s" % (self.f_code.co_filename, self.f_lineno, self.f_code.co_name)

    __repr__ = __str__


_cause_message = (
    "\nThe above exception was the direct cause "
    "of the following exception:\n\n")

_context_message = (
    "\nDuring handling of the above exception, "
    "another exception occurred:\n\n")


def create_frames_list_from_exception_cause(trace_obj, frame, exc_type, exc_desc, memo):
    lst = []
    msg = '<Unknown context>'
    try:
        exc_cause = getattr(exc_desc, '__cause__', None)
        msg = _cause_message
    except Exception:
        exc_cause = None

    if exc_cause is None:
        try:
            exc_cause = getattr(exc_desc, '__context__', None)
            msg = _context_message
        except Exception:
            exc_cause = None

    if exc_cause is None or id(exc_cause) in memo:
        return None

    # The traceback module does this, so, let's play safe here too...
    memo.add(id(exc_cause))

    tb = exc_cause.__traceback__
    frames_list = FramesList()
    frames_list.exc_type = type(exc_cause)
    frames_list.exc_desc = exc_cause
    frames_list.trace_obj = tb
    frames_list.exc_context_msg = msg

    while tb is not None:
        # Note: we don't use the actual tb.tb_frame because if the cause of the exception
        # uses the same frame object, the id(frame) would be the same and the frame_id_to_lineno
        # would be wrong as the same frame needs to appear with 2 different lines.
        lst.append((_DummyFrameWrapper(tb.tb_frame, tb.tb_lineno, None), tb.tb_lineno, _get_line_col_info_from_tb(tb)))
        tb = tb.tb_next

    for tb_frame, tb_lineno, line_col_info in lst:
        frames_list.append(tb_frame)
        frames_list.frame_id_to_lineno[id(tb_frame)] = tb_lineno
        frames_list.frame_id_to_line_col_info[id(tb_frame)] = line_col_info

    return frames_list


if IS_PY311_OR_GREATER:

    def _get_code_position(code, instruction_index):
        if instruction_index < 0:
            return (None, None, None, None)
        positions_gen = code.co_positions()
        # Note: some or all of the tuple elements can be None...
        return next(itertools.islice(positions_gen, instruction_index // 2, None))

    def _get_line_col_info_from_tb(tb):
        positions = _get_code_position(tb.tb_frame.f_code, tb.tb_lasti)
        if positions[0] is None:
            return _LineColInfo(tb.tb_lineno, *positions[1:])
        else:
            return _LineColInfo(*positions)

else:

    def _get_line_col_info_from_tb(tb):
        # Not available on older versions of Python.
        return None


def create_frames_list_from_traceback(trace_obj, frame, exc_type, exc_desc, exception_type=None):
    '''
    :param trace_obj:
        This is the traceback from which the list should be created.

    :param frame:
        This is the first frame to be considered (i.e.: topmost frame). If None is passed, all
        the frames from the traceback are shown (so, None should be passed for unhandled exceptions).

    :param exception_type:
        If this is an unhandled exception or user unhandled exception, we'll not trim the stack to create from the passed
        frame, rather, we'll just mark the frame in the frames list.
    '''
    lst = []

    tb = trace_obj
    if tb is not None and tb.tb_frame is not None:
        f = tb.tb_frame.f_back
        while f is not None:
            lst.insert(0, (f, f.f_lineno, None))
            f = f.f_back

    while tb is not None:
        lst.append((tb.tb_frame, tb.tb_lineno, _get_line_col_info_from_tb(tb)))
        tb = tb.tb_next

    frames_list = None

    for tb_frame, tb_lineno, line_col_info in reversed(lst):
        if frames_list is None and (
                (frame is tb_frame) or
                (frame is None) or
                (exception_type == EXCEPTION_TYPE_USER_UNHANDLED)
            ):
            frames_list = FramesList()

        if frames_list is not None:
            frames_list.append(tb_frame)
            frames_list.frame_id_to_lineno[id(tb_frame)] = tb_lineno
            frames_list.frame_id_to_line_col_info[id(tb_frame)] = line_col_info

    if frames_list is None and frame is not None:
        # Fallback (shouldn't happen in practice).
        pydev_log.info('create_frames_list_from_traceback did not find topmost frame in list.')
        frames_list = create_frames_list_from_frame(frame)

    frames_list.exc_type = exc_type
    frames_list.exc_desc = exc_desc
    frames_list.trace_obj = trace_obj

    if exception_type == EXCEPTION_TYPE_USER_UNHANDLED:
        frames_list.current_frame = frame
    elif exception_type == EXCEPTION_TYPE_UNHANDLED:
        if len(frames_list) > 0:
            frames_list.current_frame = frames_list.last_frame()

    curr = frames_list
    memo = set()
    memo.add(id(exc_desc))

    while True:
        chained = create_frames_list_from_exception_cause(None, None, None, curr.exc_desc, memo)
        if chained is None:
            break
        else:
            curr.chained_frames_list = chained
            curr = chained

    return frames_list


def create_frames_list_from_frame(frame):
    lst = FramesList()
    while frame is not None:
        lst.append(frame)
        frame = frame.f_back

    return lst
