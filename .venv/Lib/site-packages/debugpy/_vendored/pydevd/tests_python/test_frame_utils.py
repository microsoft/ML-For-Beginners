# coding: utf-8
import sys
from _pydevd_bundle.pydevd_constants import EXCEPTION_TYPE_USER_UNHANDLED
import pytest
from tests_python.debug_constants import IS_PY311_OR_GREATER


def test_create_frames_list_from_traceback():

    def method():
        raise RuntimeError('first')

    def method1():
        try:
            method()
        except Exception as e:
            raise RuntimeError('second') from e

    def method2():
        try:
            method1()
        except Exception as e:
            raise RuntimeError('third') from e

    try:
        method2()
    except Exception as e:
        exc_type, exc_desc, trace_obj = sys.exc_info()
        frame = sys._getframe()

        from _pydevd_bundle.pydevd_frame_utils import create_frames_list_from_traceback
        frames_list = create_frames_list_from_traceback(trace_obj, frame, exc_type, exc_desc, exception_type=EXCEPTION_TYPE_USER_UNHANDLED)
        assert str(frames_list.exc_desc) == 'third'
        assert str(frames_list.chained_frames_list.exc_desc) == 'second'
        assert str(frames_list.chained_frames_list.chained_frames_list.exc_desc) == 'first'
        assert frames_list.chained_frames_list.chained_frames_list.chained_frames_list is None


if IS_PY311_OR_GREATER:
    import traceback
    _byte_offset_to_character_offset = getattr(traceback, '_byte_offset_to_character_offset', None)
    if _byte_offset_to_character_offset is not None:
        _original = traceback._byte_offset_to_character_offset

        def _byte_offset_to_character_offset(*args, **kwargs):
            try:
                return _original(*args, **kwargs)
            except:

                # Replacement to deal with the buggy version released on Python 3.11.0.
                def replacement(str, offset):
                    as_utf8 = str.encode('utf-8')
                    if offset > len(as_utf8):
                        offset = len(as_utf8)

                    return len(as_utf8[:offset + 1].decode("utf-8", 'replace'))

                return replacement(*args , **kwargs)

        traceback._byte_offset_to_character_offset = _byte_offset_to_character_offset

_USE_UNICODE = [False, True]


@pytest.mark.parametrize('use_unicode', _USE_UNICODE)
@pytest.mark.skipif(not IS_PY311_OR_GREATER, reason='Python 3.11 required.')
def test_collect_anchors_subscript(use_unicode):
    from _pydevd_bundle.pydevd_frame_utils import create_frames_list_from_traceback

    if use_unicode:

        def method():
            d = {
                "x": {
                    "á": {
                        "í": {
                            "theta": 1
                        }
                    }
                }
            }

            result = d["x"]["á"]["í"]["beta"]

    else:

        def method():
            d = {
                "x": {
                    "y": {
                        "i": {
                            "theta": 1
                        }
                    }
                }
            }

            result = d["x"]["y"]["i"]["beta"]

    try:
        method()
    except:
        exc_type, exc_desc, trace_obj = sys.exc_info()
        memo = {}
        frame = None
        frames_list = create_frames_list_from_traceback(trace_obj, frame, exc_type, exc_desc, memo)
        iter_in = iter(frames_list)
        f = next(iter_in)
        assert f.f_code.co_name == 'method'
        line_col_info = frames_list.frame_id_to_line_col_info[id(f)]

        if use_unicode:
            line = '            result = d["x"]["á"]["í"]["beta"]'
        else:
            line = '            result = d["x"]["y"]["i"]["beta"]'

        # Ok, so, the range that we we have covers >>d["x"]["á"]["í"]["beta"]<<
        # the problem here is that ideally we'd like to present to the user that
        # the current key is "beta", so, we need to do some additional computation
        # to find out the proper column to show to the user.
        # (see https://github.com/microsoft/debugpy/issues/1099
        # for more information).
        assert line_col_info.colno == line.index('d["x"]')

        # It's +1 here due to the í unicode char (we need to convert from the bytes
        # index to the actual character in the string to get the actual col).
        if use_unicode:
            assert line_col_info.end_colno == len(line) + 2
        else:
            assert line_col_info.end_colno == len(line)
        original_line = line

        col, endcol = line_col_info.map_columns_to_line(original_line)
        assert col == line.index('["beta"]')
        assert endcol == len(line)


@pytest.mark.parametrize('use_unicode', _USE_UNICODE)
@pytest.mark.skipif(not IS_PY311_OR_GREATER, reason='Python 3.11 required.')
def test_collect_anchors_binop_1(use_unicode):
    from _pydevd_bundle.pydevd_frame_utils import create_frames_list_from_traceback

    if use_unicode:

        def method():
            á = 1
            í = 2
            c = tuple

            result = á + í + c

    else:

        def method():
            a = 1
            b = 2
            c = tuple

            result = a + b + c

    try:
        method()
    except:
        exc_type, exc_desc, trace_obj = sys.exc_info()
        memo = {}
        frame = None
        frames_list = create_frames_list_from_traceback(trace_obj, frame, exc_type, exc_desc, memo)
        iter_in = iter(frames_list)
        f = next(iter_in)
        assert f.f_code.co_name == 'method'
        line_col_info = frames_list.frame_id_to_line_col_info[id(f)]

        if use_unicode:
            line = '            result = á + í + c'
            expected_index = line.index('á + í')
        else:
            line = '            result = a + b + c'
            expected_index = line.index('a + b')

        assert line_col_info.colno == expected_index

        # It's +2 here due to the á and í unicode chars (we need to convert from the bytes
        # index to the actual character in the string to get the actual col).
        if use_unicode:
            assert line_col_info.end_colno == len(line) + 2
        else:
            assert line_col_info.end_colno == len(line)
        original_line = line

        col, endcol = line_col_info.map_columns_to_line(original_line)
        assert col == line.index('+ c')
        assert endcol == col + 1


@pytest.mark.parametrize('use_unicode', _USE_UNICODE)
@pytest.mark.skipif(not IS_PY311_OR_GREATER, reason='Python 3.11 required.')
def test_collect_anchors_binop_2(use_unicode):
    from _pydevd_bundle.pydevd_frame_utils import create_frames_list_from_traceback

    if use_unicode:

        def method():
            á = 1
            í = 2
            c = tuple

            result = á + c + í

    else:

        def method():
            a = 1
            b = 2
            c = tuple

            result = a + c + b

    try:
        method()
    except:
        exc_type, exc_desc, trace_obj = sys.exc_info()
        memo = {}
        frame = None
        frames_list = create_frames_list_from_traceback(trace_obj, frame, exc_type, exc_desc, memo)
        iter_in = iter(frames_list)
        f = next(iter_in)
        assert f.f_code.co_name == 'method'
        line_col_info = frames_list.frame_id_to_line_col_info[id(f)]

        if use_unicode:
            line = '            result = á + c + í'
            expected_index = line.index('á + c')
        else:
            line = '            result = a + c + b'
            expected_index = line.index('a + c')

        assert line_col_info.colno == expected_index

        # It's +2 here due to the á and í unicode chars (we need to convert from the bytes
        # index to the actual character in the string to get the actual col).
        if use_unicode:
            assert line_col_info.end_colno == line.index('c + í') + 2
        else:
            assert line_col_info.end_colno == line.index('c + b') + 1
        original_line = line

        col, endcol = line_col_info.map_columns_to_line(original_line)
        assert col == 23
        assert endcol == 24
        assert col == line.index('+ c')
        assert endcol == col + 1
