"""
This file contains a minimal set of tests for compliance with the extension
array interface test suite, and should contain no other tests.
The test suite for the full functionality of the array is located in
`pandas/tests/arrays/`.

The tests in this file are inherited from the BaseExtensionTests, and only
minimal tweaks should be applied to get the tests passing (by overwriting a
parent method).

Additional tests should either be added to one of the BaseExtensionTests
classes (if they are relevant for the extension interface for all dtypes), or
be added to the array-specific tests in `pandas/tests/arrays/`.

"""
import string

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.api.types import is_string_dtype
from pandas.core.arrays import ArrowStringArray
from pandas.core.arrays.string_ import StringDtype
from pandas.tests.extension import base


def split_array(arr):
    if arr.dtype.storage != "pyarrow":
        pytest.skip("only applicable for pyarrow chunked array n/a")

    def _split_array(arr):
        import pyarrow as pa

        arrow_array = arr._pa_array
        split = len(arrow_array) // 2
        arrow_array = pa.chunked_array(
            [*arrow_array[:split].chunks, *arrow_array[split:].chunks]
        )
        assert arrow_array.num_chunks == 2
        return type(arr)(arrow_array)

    return _split_array(arr)


@pytest.fixture(params=[True, False])
def chunked(request):
    return request.param


@pytest.fixture
def dtype(string_storage):
    return StringDtype(storage=string_storage)


@pytest.fixture
def data(dtype, chunked):
    strings = np.random.default_rng(2).choice(list(string.ascii_letters), size=100)
    while strings[0] == strings[1]:
        strings = np.random.default_rng(2).choice(list(string.ascii_letters), size=100)

    arr = dtype.construct_array_type()._from_sequence(strings)
    return split_array(arr) if chunked else arr


@pytest.fixture
def data_missing(dtype, chunked):
    """Length 2 array with [NA, Valid]"""
    arr = dtype.construct_array_type()._from_sequence([pd.NA, "A"])
    return split_array(arr) if chunked else arr


@pytest.fixture
def data_for_sorting(dtype, chunked):
    arr = dtype.construct_array_type()._from_sequence(["B", "C", "A"])
    return split_array(arr) if chunked else arr


@pytest.fixture
def data_missing_for_sorting(dtype, chunked):
    arr = dtype.construct_array_type()._from_sequence(["B", pd.NA, "A"])
    return split_array(arr) if chunked else arr


@pytest.fixture
def data_for_grouping(dtype, chunked):
    arr = dtype.construct_array_type()._from_sequence(
        ["B", "B", pd.NA, pd.NA, "A", "A", "B", "C"]
    )
    return split_array(arr) if chunked else arr


class TestDtype(base.BaseDtypeTests):
    def test_eq_with_str(self, dtype):
        assert dtype == f"string[{dtype.storage}]"
        super().test_eq_with_str(dtype)

    def test_is_not_string_type(self, dtype):
        # Different from BaseDtypeTests.test_is_not_string_type
        # because StringDtype is a string type
        assert is_string_dtype(dtype)


class TestInterface(base.BaseInterfaceTests):
    def test_view(self, data, request, arrow_string_storage):
        if data.dtype.storage in arrow_string_storage:
            pytest.skip(reason="2D support not implemented for ArrowStringArray")
        super().test_view(data)


class TestConstructors(base.BaseConstructorsTests):
    def test_from_dtype(self, data):
        # base test uses string representation of dtype
        pass


class TestReshaping(base.BaseReshapingTests):
    def test_transpose(self, data, request, arrow_string_storage):
        if data.dtype.storage in arrow_string_storage:
            pytest.skip(reason="2D support not implemented for ArrowStringArray")
        super().test_transpose(data)


class TestGetitem(base.BaseGetitemTests):
    pass


class TestSetitem(base.BaseSetitemTests):
    def test_setitem_preserves_views(self, data, request, arrow_string_storage):
        if data.dtype.storage in arrow_string_storage:
            pytest.skip(reason="2D support not implemented for ArrowStringArray")
        super().test_setitem_preserves_views(data)


class TestIndex(base.BaseIndexTests):
    pass


class TestMissing(base.BaseMissingTests):
    def test_dropna_array(self, data_missing):
        result = data_missing.dropna()
        expected = data_missing[[1]]
        tm.assert_extension_array_equal(result, expected)

    def test_fillna_no_op_returns_copy(self, data):
        data = data[~data.isna()]

        valid = data[0]
        result = data.fillna(valid)
        assert result is not data
        tm.assert_extension_array_equal(result, data)

        result = data.fillna(method="backfill")
        assert result is not data
        tm.assert_extension_array_equal(result, data)


class TestReduce(base.BaseReduceTests):
    def _supports_reduction(self, ser: pd.Series, op_name: str) -> bool:
        return (
            ser.dtype.storage == "pyarrow_numpy"  # type: ignore[union-attr]
            and op_name in ("any", "all")
        )

    @pytest.mark.parametrize("skipna", [True, False])
    def test_reduce_series_numeric(self, data, all_numeric_reductions, skipna):
        op_name = all_numeric_reductions

        if op_name in ["min", "max"]:
            return None

        ser = pd.Series(data)
        with pytest.raises(TypeError):
            getattr(ser, op_name)(skipna=skipna)


class TestMethods(base.BaseMethodsTests):
    pass


class TestCasting(base.BaseCastingTests):
    pass


class TestComparisonOps(base.BaseComparisonOpsTests):
    def _cast_pointwise_result(self, op_name: str, obj, other, pointwise_result):
        dtype = tm.get_dtype(obj)
        # error: Item "dtype[Any]" of "dtype[Any] | ExtensionDtype" has no
        # attribute "storage"
        if dtype.storage == "pyarrow":  # type: ignore[union-attr]
            cast_to = "boolean[pyarrow]"
        elif dtype.storage == "pyarrow_numpy":  # type: ignore[union-attr]
            cast_to = np.bool_  # type: ignore[assignment]
        else:
            cast_to = "boolean"
        return pointwise_result.astype(cast_to)

    def test_compare_scalar(self, data, comparison_op):
        ser = pd.Series(data)
        self._compare_other(ser, data, comparison_op, "abc")


class TestParsing(base.BaseParsingTests):
    pass


class TestPrinting(base.BasePrintingTests):
    pass


class TestGroupBy(base.BaseGroupbyTests):
    @pytest.mark.filterwarnings("ignore:Falling back:pandas.errors.PerformanceWarning")
    def test_groupby_extension_apply(self, data_for_grouping, groupby_apply_op):
        super().test_groupby_extension_apply(data_for_grouping, groupby_apply_op)


class Test2DCompat(base.Dim2CompatTests):
    @pytest.fixture(autouse=True)
    def arrow_not_supported(self, data, request):
        if isinstance(data, ArrowStringArray):
            pytest.skip(reason="2D support not implemented for ArrowStringArray")


def test_searchsorted_with_na_raises(data_for_sorting, as_series):
    # GH50447
    b, c, a = data_for_sorting
    arr = data_for_sorting.take([2, 0, 1])  # to get [a, b, c]
    arr[-1] = pd.NA

    if as_series:
        arr = pd.Series(arr)

    msg = (
        "searchsorted requires array to be sorted, "
        "which is impossible with NAs present."
    )
    with pytest.raises(ValueError, match=msg):
        arr.searchsorted(b)
