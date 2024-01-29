""" test feather-format compat """
import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import (
    ArrowStringArray,
    StringArray,
)

from pandas.io.feather_format import read_feather, to_feather  # isort:skip

pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)

pa = pytest.importorskip("pyarrow")


@pytest.mark.single_cpu
class TestFeather:
    def check_error_on_write(self, df, exc, err_msg):
        # check that we are raising the exception
        # on writing

        with pytest.raises(exc, match=err_msg):
            with tm.ensure_clean() as path:
                to_feather(df, path)

    def check_external_error_on_write(self, df):
        # check that we are raising the exception
        # on writing

        with tm.external_error_raised(Exception):
            with tm.ensure_clean() as path:
                to_feather(df, path)

    def check_round_trip(self, df, expected=None, write_kwargs={}, **read_kwargs):
        if expected is None:
            expected = df.copy()

        with tm.ensure_clean() as path:
            to_feather(df, path, **write_kwargs)

            result = read_feather(path, **read_kwargs)

            tm.assert_frame_equal(result, expected)

    def test_error(self):
        msg = "feather only support IO with DataFrames"
        for obj in [
            pd.Series([1, 2, 3]),
            1,
            "foo",
            pd.Timestamp("20130101"),
            np.array([1, 2, 3]),
        ]:
            self.check_error_on_write(obj, ValueError, msg)

    def test_basic(self):
        df = pd.DataFrame(
            {
                "string": list("abc"),
                "int": list(range(1, 4)),
                "uint": np.arange(3, 6).astype("u1"),
                "float": np.arange(4.0, 7.0, dtype="float64"),
                "float_with_null": [1.0, np.nan, 3],
                "bool": [True, False, True],
                "bool_with_null": [True, np.nan, False],
                "cat": pd.Categorical(list("abc")),
                "dt": pd.DatetimeIndex(
                    list(pd.date_range("20130101", periods=3)), freq=None
                ),
                "dttz": pd.DatetimeIndex(
                    list(pd.date_range("20130101", periods=3, tz="US/Eastern")),
                    freq=None,
                ),
                "dt_with_null": [
                    pd.Timestamp("20130101"),
                    pd.NaT,
                    pd.Timestamp("20130103"),
                ],
                "dtns": pd.DatetimeIndex(
                    list(pd.date_range("20130101", periods=3, freq="ns")), freq=None
                ),
            }
        )
        df["periods"] = pd.period_range("2013", freq="M", periods=3)
        df["timedeltas"] = pd.timedelta_range("1 day", periods=3)
        df["intervals"] = pd.interval_range(0, 3, 3)

        assert df.dttz.dtype.tz.zone == "US/Eastern"

        expected = df.copy()
        expected.loc[1, "bool_with_null"] = None
        self.check_round_trip(df, expected=expected)

    def test_duplicate_columns(self):
        # https://github.com/wesm/feather/issues/53
        # not currently able to handle duplicate columns
        df = pd.DataFrame(np.arange(12).reshape(4, 3), columns=list("aaa")).copy()
        self.check_external_error_on_write(df)

    def test_read_columns(self):
        # GH 24025
        df = pd.DataFrame(
            {
                "col1": list("abc"),
                "col2": list(range(1, 4)),
                "col3": list("xyz"),
                "col4": list(range(4, 7)),
            }
        )
        columns = ["col1", "col3"]
        self.check_round_trip(df, expected=df[columns], columns=columns)

    def test_read_columns_different_order(self):
        # GH 33878
        df = pd.DataFrame({"A": [1, 2], "B": ["x", "y"], "C": [True, False]})
        expected = df[["B", "A"]]
        self.check_round_trip(df, expected, columns=["B", "A"])

    def test_unsupported_other(self):
        # mixed python objects
        df = pd.DataFrame({"a": ["a", 1, 2.0]})
        self.check_external_error_on_write(df)

    def test_rw_use_threads(self):
        df = pd.DataFrame({"A": np.arange(100000)})
        self.check_round_trip(df, use_threads=True)
        self.check_round_trip(df, use_threads=False)

    def test_path_pathlib(self):
        df = pd.DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),
            columns=pd.Index(list("ABCD"), dtype=object),
            index=pd.Index([f"i-{i}" for i in range(30)], dtype=object),
        ).reset_index()
        result = tm.round_trip_pathlib(df.to_feather, read_feather)
        tm.assert_frame_equal(df, result)

    def test_path_localpath(self):
        df = pd.DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),
            columns=pd.Index(list("ABCD"), dtype=object),
            index=pd.Index([f"i-{i}" for i in range(30)], dtype=object),
        ).reset_index()
        result = tm.round_trip_localpath(df.to_feather, read_feather)
        tm.assert_frame_equal(df, result)

    def test_passthrough_keywords(self):
        df = pd.DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),
            columns=pd.Index(list("ABCD"), dtype=object),
            index=pd.Index([f"i-{i}" for i in range(30)], dtype=object),
        ).reset_index()
        self.check_round_trip(df, write_kwargs={"version": 1})

    @pytest.mark.network
    @pytest.mark.single_cpu
    def test_http_path(self, feather_file, httpserver):
        # GH 29055
        expected = read_feather(feather_file)
        with open(feather_file, "rb") as f:
            httpserver.serve_content(content=f.read())
            res = read_feather(httpserver.url)
        tm.assert_frame_equal(expected, res)

    def test_read_feather_dtype_backend(self, string_storage, dtype_backend):
        # GH#50765
        df = pd.DataFrame(
            {
                "a": pd.Series([1, np.nan, 3], dtype="Int64"),
                "b": pd.Series([1, 2, 3], dtype="Int64"),
                "c": pd.Series([1.5, np.nan, 2.5], dtype="Float64"),
                "d": pd.Series([1.5, 2.0, 2.5], dtype="Float64"),
                "e": [True, False, None],
                "f": [True, False, True],
                "g": ["a", "b", "c"],
                "h": ["a", "b", None],
            }
        )

        if string_storage == "python":
            string_array = StringArray(np.array(["a", "b", "c"], dtype=np.object_))
            string_array_na = StringArray(np.array(["a", "b", pd.NA], dtype=np.object_))

        elif dtype_backend == "pyarrow":
            from pandas.arrays import ArrowExtensionArray

            string_array = ArrowExtensionArray(pa.array(["a", "b", "c"]))
            string_array_na = ArrowExtensionArray(pa.array(["a", "b", None]))

        else:
            string_array = ArrowStringArray(pa.array(["a", "b", "c"]))
            string_array_na = ArrowStringArray(pa.array(["a", "b", None]))

        with tm.ensure_clean() as path:
            to_feather(df, path)
            with pd.option_context("mode.string_storage", string_storage):
                result = read_feather(path, dtype_backend=dtype_backend)

        expected = pd.DataFrame(
            {
                "a": pd.Series([1, np.nan, 3], dtype="Int64"),
                "b": pd.Series([1, 2, 3], dtype="Int64"),
                "c": pd.Series([1.5, np.nan, 2.5], dtype="Float64"),
                "d": pd.Series([1.5, 2.0, 2.5], dtype="Float64"),
                "e": pd.Series([True, False, pd.NA], dtype="boolean"),
                "f": pd.Series([True, False, True], dtype="boolean"),
                "g": string_array,
                "h": string_array_na,
            }
        )

        if dtype_backend == "pyarrow":
            from pandas.arrays import ArrowExtensionArray

            expected = pd.DataFrame(
                {
                    col: ArrowExtensionArray(pa.array(expected[col], from_pandas=True))
                    for col in expected.columns
                }
            )

        tm.assert_frame_equal(result, expected)

    def test_int_columns_and_index(self):
        df = pd.DataFrame({"a": [1, 2, 3]}, index=pd.Index([3, 4, 5], name="test"))
        self.check_round_trip(df)

    def test_invalid_dtype_backend(self):
        msg = (
            "dtype_backend numpy is invalid, only 'numpy_nullable' and "
            "'pyarrow' are allowed."
        )
        df = pd.DataFrame({"int": list(range(1, 4))})
        with tm.ensure_clean("tmp.feather") as path:
            df.to_feather(path)
            with pytest.raises(ValueError, match=msg):
                read_feather(path, dtype_backend="numpy")

    def test_string_inference(self, tmp_path):
        # GH#54431
        path = tmp_path / "test_string_inference.p"
        df = pd.DataFrame(data={"a": ["x", "y"]})
        df.to_feather(path)
        with pd.option_context("future.infer_string", True):
            result = read_feather(path)
        expected = pd.DataFrame(data={"a": ["x", "y"]}, dtype="string[pyarrow_numpy]")
        tm.assert_frame_equal(result, expected)
