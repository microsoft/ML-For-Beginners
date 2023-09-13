import os
from textwrap import dedent

import numpy as np
import pytest

from pandas.compat import (
    is_ci_environment,
    is_platform_mac,
)
from pandas.errors import (
    PyperclipException,
    PyperclipWindowsException,
)

import pandas as pd
from pandas import (
    NA,
    DataFrame,
    Series,
    get_option,
    read_clipboard,
)
import pandas._testing as tm
from pandas.core.arrays import (
    ArrowStringArray,
    StringArray,
)

from pandas.io.clipboard import (
    CheckedCall,
    _stringifyText,
    clipboard_get,
    clipboard_set,
)


def build_kwargs(sep, excel):
    kwargs = {}
    if excel != "default":
        kwargs["excel"] = excel
    if sep != "default":
        kwargs["sep"] = sep
    return kwargs


@pytest.fixture(
    params=[
        "delims",
        "utf8",
        "utf16",
        "string",
        "long",
        "nonascii",
        "colwidth",
        "mixed",
        "float",
        "int",
    ]
)
def df(request):
    data_type = request.param

    if data_type == "delims":
        return DataFrame({"a": ['"a,\t"b|c', "d\tef`"], "b": ["hi'j", "k''lm"]})
    elif data_type == "utf8":
        return DataFrame({"a": ["µasd", "Ωœ∑`"], "b": ["øπ∆˚¬", "œ∑`®"]})
    elif data_type == "utf16":
        return DataFrame(
            {"a": ["\U0001f44d\U0001f44d", "\U0001f44d\U0001f44d"], "b": ["abc", "def"]}
        )
    elif data_type == "string":
        return tm.makeCustomDataframe(
            5, 3, c_idx_type="s", r_idx_type="i", c_idx_names=[None], r_idx_names=[None]
        )
    elif data_type == "long":
        max_rows = get_option("display.max_rows")
        return tm.makeCustomDataframe(
            max_rows + 1,
            3,
            data_gen_f=lambda *args: np.random.default_rng(2).integers(2),
            c_idx_type="s",
            r_idx_type="i",
            c_idx_names=[None],
            r_idx_names=[None],
        )
    elif data_type == "nonascii":
        return DataFrame({"en": "in English".split(), "es": "en español".split()})
    elif data_type == "colwidth":
        _cw = get_option("display.max_colwidth") + 1
        return tm.makeCustomDataframe(
            5,
            3,
            data_gen_f=lambda *args: "x" * _cw,
            c_idx_type="s",
            r_idx_type="i",
            c_idx_names=[None],
            r_idx_names=[None],
        )
    elif data_type == "mixed":
        return DataFrame(
            {
                "a": np.arange(1.0, 6.0) + 0.01,
                "b": np.arange(1, 6).astype(np.int64),
                "c": list("abcde"),
            }
        )
    elif data_type == "float":
        return tm.makeCustomDataframe(
            5,
            3,
            data_gen_f=lambda r, c: float(r) + 0.01,
            c_idx_type="s",
            r_idx_type="i",
            c_idx_names=[None],
            r_idx_names=[None],
        )
    elif data_type == "int":
        return tm.makeCustomDataframe(
            5,
            3,
            data_gen_f=lambda *args: np.random.default_rng(2).integers(2),
            c_idx_type="s",
            r_idx_type="i",
            c_idx_names=[None],
            r_idx_names=[None],
        )
    else:
        raise ValueError


@pytest.fixture
def mock_ctypes(monkeypatch):
    """
    Mocks WinError to help with testing the clipboard.
    """

    def _mock_win_error():
        return "Window Error"

    # Set raising to False because WinError won't exist on non-windows platforms
    with monkeypatch.context() as m:
        m.setattr("ctypes.WinError", _mock_win_error, raising=False)
        yield


@pytest.mark.usefixtures("mock_ctypes")
def test_checked_call_with_bad_call(monkeypatch):
    """
    Give CheckCall a function that returns a falsey value and
    mock get_errno so it returns false so an exception is raised.
    """

    def _return_false():
        return False

    monkeypatch.setattr("pandas.io.clipboard.get_errno", lambda: True)
    msg = f"Error calling {_return_false.__name__} \\(Window Error\\)"

    with pytest.raises(PyperclipWindowsException, match=msg):
        CheckedCall(_return_false)()


@pytest.mark.usefixtures("mock_ctypes")
def test_checked_call_with_valid_call(monkeypatch):
    """
    Give CheckCall a function that returns a truthy value and
    mock get_errno so it returns true so an exception is not raised.
    The function should return the results from _return_true.
    """

    def _return_true():
        return True

    monkeypatch.setattr("pandas.io.clipboard.get_errno", lambda: False)

    # Give CheckedCall a callable that returns a truthy value s
    checked_call = CheckedCall(_return_true)
    assert checked_call() is True


@pytest.mark.parametrize(
    "text",
    [
        "String_test",
        True,
        1,
        1.0,
        1j,
    ],
)
def test_stringify_text(text):
    valid_types = (str, int, float, bool)

    if isinstance(text, valid_types):
        result = _stringifyText(text)
        assert result == str(text)
    else:
        msg = (
            "only str, int, float, and bool values "
            f"can be copied to the clipboard, not {type(text).__name__}"
        )
        with pytest.raises(PyperclipException, match=msg):
            _stringifyText(text)


@pytest.fixture
def mock_clipboard(monkeypatch, request):
    """Fixture mocking clipboard IO.

    This mocks pandas.io.clipboard.clipboard_get and
    pandas.io.clipboard.clipboard_set.

    This uses a local dict for storing data. The dictionary
    key used is the test ID, available with ``request.node.name``.

    This returns the local dictionary, for direct manipulation by
    tests.
    """
    # our local clipboard for tests
    _mock_data = {}

    def _mock_set(data):
        _mock_data[request.node.name] = data

    def _mock_get():
        return _mock_data[request.node.name]

    monkeypatch.setattr("pandas.io.clipboard.clipboard_set", _mock_set)
    monkeypatch.setattr("pandas.io.clipboard.clipboard_get", _mock_get)

    yield _mock_data


@pytest.mark.clipboard
def test_mock_clipboard(mock_clipboard):
    import pandas.io.clipboard

    pandas.io.clipboard.clipboard_set("abc")
    assert "abc" in set(mock_clipboard.values())
    result = pandas.io.clipboard.clipboard_get()
    assert result == "abc"


@pytest.mark.single_cpu
@pytest.mark.clipboard
@pytest.mark.usefixtures("mock_clipboard")
class TestClipboard:
    def check_round_trip_frame(self, data, excel=None, sep=None, encoding=None):
        data.to_clipboard(excel=excel, sep=sep, encoding=encoding)
        result = read_clipboard(sep=sep or "\t", index_col=0, encoding=encoding)
        tm.assert_frame_equal(data, result)

    # Test that default arguments copy as tab delimited
    def test_round_trip_frame(self, df):
        self.check_round_trip_frame(df)

    # Test that explicit delimiters are respected
    @pytest.mark.parametrize("sep", ["\t", ",", "|"])
    def test_round_trip_frame_sep(self, df, sep):
        self.check_round_trip_frame(df, sep=sep)

    # Test white space separator
    def test_round_trip_frame_string(self, df):
        df.to_clipboard(excel=False, sep=None)
        result = read_clipboard()
        assert df.to_string() == result.to_string()
        assert df.shape == result.shape

    # Two character separator is not supported in to_clipboard
    # Test that multi-character separators are not silently passed
    def test_excel_sep_warning(self, df):
        with tm.assert_produces_warning(
            UserWarning,
            match="to_clipboard in excel mode requires a single character separator.",
            check_stacklevel=False,
        ):
            df.to_clipboard(excel=True, sep=r"\t")

    # Separator is ignored when excel=False and should produce a warning
    def test_copy_delim_warning(self, df):
        with tm.assert_produces_warning():
            df.to_clipboard(excel=False, sep="\t")

    # Tests that the default behavior of to_clipboard is tab
    # delimited and excel="True"
    @pytest.mark.parametrize("sep", ["\t", None, "default"])
    @pytest.mark.parametrize("excel", [True, None, "default"])
    def test_clipboard_copy_tabs_default(self, sep, excel, df, request, mock_clipboard):
        kwargs = build_kwargs(sep, excel)
        df.to_clipboard(**kwargs)
        assert mock_clipboard[request.node.name] == df.to_csv(sep="\t")

    # Tests reading of white space separated tables
    @pytest.mark.parametrize("sep", [None, "default"])
    @pytest.mark.parametrize("excel", [False])
    def test_clipboard_copy_strings(self, sep, excel, df):
        kwargs = build_kwargs(sep, excel)
        df.to_clipboard(**kwargs)
        result = read_clipboard(sep=r"\s+")
        assert result.to_string() == df.to_string()
        assert df.shape == result.shape

    def test_read_clipboard_infer_excel(self, request, mock_clipboard):
        # gh-19010: avoid warnings
        clip_kwargs = {"engine": "python"}

        text = dedent(
            """
            John James\tCharlie Mingus
            1\t2
            4\tHarry Carney
            """.strip()
        )
        mock_clipboard[request.node.name] = text
        df = read_clipboard(**clip_kwargs)

        # excel data is parsed correctly
        assert df.iloc[1, 1] == "Harry Carney"

        # having diff tab counts doesn't trigger it
        text = dedent(
            """
            a\t b
            1  2
            3  4
            """.strip()
        )
        mock_clipboard[request.node.name] = text
        res = read_clipboard(**clip_kwargs)

        text = dedent(
            """
            a  b
            1  2
            3  4
            """.strip()
        )
        mock_clipboard[request.node.name] = text
        exp = read_clipboard(**clip_kwargs)

        tm.assert_frame_equal(res, exp)

    def test_infer_excel_with_nulls(self, request, mock_clipboard):
        # GH41108
        text = "col1\tcol2\n1\tred\n\tblue\n2\tgreen"

        mock_clipboard[request.node.name] = text
        df = read_clipboard()
        df_expected = DataFrame(
            data={"col1": [1, None, 2], "col2": ["red", "blue", "green"]}
        )

        # excel data is parsed correctly
        tm.assert_frame_equal(df, df_expected)

    @pytest.mark.parametrize(
        "multiindex",
        [
            (  # Can't use `dedent` here as it will remove the leading `\t`
                "\n".join(
                    [
                        "\t\t\tcol1\tcol2",
                        "A\t0\tTrue\t1\tred",
                        "A\t1\tTrue\t\tblue",
                        "B\t0\tFalse\t2\tgreen",
                    ]
                ),
                [["A", "A", "B"], [0, 1, 0], [True, True, False]],
            ),
            (
                "\n".join(
                    ["\t\tcol1\tcol2", "A\t0\t1\tred", "A\t1\t\tblue", "B\t0\t2\tgreen"]
                ),
                [["A", "A", "B"], [0, 1, 0]],
            ),
        ],
    )
    def test_infer_excel_with_multiindex(self, request, mock_clipboard, multiindex):
        # GH41108

        mock_clipboard[request.node.name] = multiindex[0]
        df = read_clipboard()
        df_expected = DataFrame(
            data={"col1": [1, None, 2], "col2": ["red", "blue", "green"]},
            index=multiindex[1],
        )

        # excel data is parsed correctly
        tm.assert_frame_equal(df, df_expected)

    def test_invalid_encoding(self, df):
        msg = "clipboard only supports utf-8 encoding"
        # test case for testing invalid encoding
        with pytest.raises(ValueError, match=msg):
            df.to_clipboard(encoding="ascii")
        with pytest.raises(NotImplementedError, match=msg):
            read_clipboard(encoding="ascii")

    @pytest.mark.parametrize("enc", ["UTF-8", "utf-8", "utf8"])
    def test_round_trip_valid_encodings(self, enc, df):
        self.check_round_trip_frame(df, encoding=enc)

    @pytest.mark.single_cpu
    @pytest.mark.parametrize("data", ["\U0001f44d...", "Ωœ∑`...", "abcd..."])
    @pytest.mark.xfail(
        (os.environ.get("DISPLAY") is None and not is_platform_mac())
        or is_ci_environment(),
        reason="Cannot pass if a headless system is not put in place with Xvfb",
        strict=not is_ci_environment(),  # Flaky failures in the CI
    )
    def test_raw_roundtrip(self, data):
        # PR #25040 wide unicode wasn't copied correctly on PY3 on windows
        clipboard_set(data)
        assert data == clipboard_get()

    @pytest.mark.parametrize("engine", ["c", "python"])
    def test_read_clipboard_dtype_backend(
        self, request, mock_clipboard, string_storage, dtype_backend, engine
    ):
        # GH#50502
        if string_storage == "pyarrow" or dtype_backend == "pyarrow":
            pa = pytest.importorskip("pyarrow")

        if string_storage == "python":
            string_array = StringArray(np.array(["x", "y"], dtype=np.object_))
            string_array_na = StringArray(np.array(["x", NA], dtype=np.object_))

        else:
            string_array = ArrowStringArray(pa.array(["x", "y"]))
            string_array_na = ArrowStringArray(pa.array(["x", None]))

        text = """a,b,c,d,e,f,g,h,i
x,1,4.0,x,2,4.0,,True,False
y,2,5.0,,,,,False,"""
        mock_clipboard[request.node.name] = text

        with pd.option_context("mode.string_storage", string_storage):
            result = read_clipboard(sep=",", dtype_backend=dtype_backend, engine=engine)

        expected = DataFrame(
            {
                "a": string_array,
                "b": Series([1, 2], dtype="Int64"),
                "c": Series([4.0, 5.0], dtype="Float64"),
                "d": string_array_na,
                "e": Series([2, NA], dtype="Int64"),
                "f": Series([4.0, NA], dtype="Float64"),
                "g": Series([NA, NA], dtype="Int64"),
                "h": Series([True, False], dtype="boolean"),
                "i": Series([False, NA], dtype="boolean"),
            }
        )
        if dtype_backend == "pyarrow":
            from pandas.arrays import ArrowExtensionArray

            expected = DataFrame(
                {
                    col: ArrowExtensionArray(pa.array(expected[col], from_pandas=True))
                    for col in expected.columns
                }
            )
            expected["g"] = ArrowExtensionArray(pa.array([None, None]))

        tm.assert_frame_equal(result, expected)

    def test_invalid_dtype_backend(self):
        msg = (
            "dtype_backend numpy is invalid, only 'numpy_nullable' and "
            "'pyarrow' are allowed."
        )
        with pytest.raises(ValueError, match=msg):
            read_clipboard(dtype_backend="numpy")
