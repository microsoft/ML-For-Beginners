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

Note: we do not bother with base.BaseIndexTests because NumpyExtensionArray
will never be held in an Index.
"""
import numpy as np
import pytest

from pandas.core.dtypes.cast import can_hold_element
from pandas.core.dtypes.dtypes import NumpyEADtype

import pandas as pd
import pandas._testing as tm
from pandas.api.types import is_object_dtype
from pandas.core.arrays.numpy_ import NumpyExtensionArray
from pandas.core.internals import blocks
from pandas.tests.extension import base


def _can_hold_element_patched(obj, element) -> bool:
    if isinstance(element, NumpyExtensionArray):
        element = element.to_numpy()
    return can_hold_element(obj, element)


orig_assert_attr_equal = tm.assert_attr_equal


def _assert_attr_equal(attr: str, left, right, obj: str = "Attributes"):
    """
    patch tm.assert_attr_equal so NumpyEADtype("object") is closed enough to
    np.dtype("object")
    """
    if attr == "dtype":
        lattr = getattr(left, "dtype", None)
        rattr = getattr(right, "dtype", None)
        if isinstance(lattr, NumpyEADtype) and not isinstance(rattr, NumpyEADtype):
            left = left.astype(lattr.numpy_dtype)
        elif isinstance(rattr, NumpyEADtype) and not isinstance(lattr, NumpyEADtype):
            right = right.astype(rattr.numpy_dtype)

    orig_assert_attr_equal(attr, left, right, obj)


@pytest.fixture(params=["float", "object"])
def dtype(request):
    return NumpyEADtype(np.dtype(request.param))


@pytest.fixture
def allow_in_pandas(monkeypatch):
    """
    A monkeypatch to tells pandas to let us in.

    By default, passing a NumpyExtensionArray to an index / series / frame
    constructor will unbox that NumpyExtensionArray to an ndarray, and treat
    it as a non-EA column. We don't want people using EAs without
    reason.

    The mechanism for this is a check against ABCNumpyExtensionArray
    in each constructor.

    But, for testing, we need to allow them in pandas. So we patch
    the _typ of NumpyExtensionArray, so that we evade the ABCNumpyExtensionArray
    check.
    """
    with monkeypatch.context() as m:
        m.setattr(NumpyExtensionArray, "_typ", "extension")
        m.setattr(blocks, "can_hold_element", _can_hold_element_patched)
        m.setattr(tm.asserters, "assert_attr_equal", _assert_attr_equal)
        yield


@pytest.fixture
def data(allow_in_pandas, dtype):
    if dtype.numpy_dtype == "object":
        return pd.Series([(i,) for i in range(100)]).array
    return NumpyExtensionArray(np.arange(1, 101, dtype=dtype._dtype))


@pytest.fixture
def data_missing(allow_in_pandas, dtype):
    if dtype.numpy_dtype == "object":
        return NumpyExtensionArray(np.array([np.nan, (1,)], dtype=object))
    return NumpyExtensionArray(np.array([np.nan, 1.0]))


@pytest.fixture
def na_cmp():
    def cmp(a, b):
        return np.isnan(a) and np.isnan(b)

    return cmp


@pytest.fixture
def data_for_sorting(allow_in_pandas, dtype):
    """Length-3 array with a known sort order.

    This should be three items [B, C, A] with
    A < B < C
    """
    if dtype.numpy_dtype == "object":
        # Use an empty tuple for first element, then remove,
        # to disable np.array's shape inference.
        return NumpyExtensionArray(np.array([(), (2,), (3,), (1,)], dtype=object)[1:])
    return NumpyExtensionArray(np.array([1, 2, 0]))


@pytest.fixture
def data_missing_for_sorting(allow_in_pandas, dtype):
    """Length-3 array with a known sort order.

    This should be three items [B, NA, A] with
    A < B and NA missing.
    """
    if dtype.numpy_dtype == "object":
        return NumpyExtensionArray(np.array([(1,), np.nan, (0,)], dtype=object))
    return NumpyExtensionArray(np.array([1, np.nan, 0]))


@pytest.fixture
def data_for_grouping(allow_in_pandas, dtype):
    """Data for factorization, grouping, and unique tests.

    Expected to be like [B, B, NA, NA, A, A, B, C]

    Where A < B < C and NA is missing
    """
    if dtype.numpy_dtype == "object":
        a, b, c = (1,), (2,), (3,)
    else:
        a, b, c = np.arange(3)
    return NumpyExtensionArray(
        np.array([b, b, np.nan, np.nan, a, a, b, c], dtype=dtype.numpy_dtype)
    )


@pytest.fixture
def data_for_twos(dtype):
    if dtype.kind == "O":
        pytest.skip("Not a numeric dtype")
    arr = np.ones(100) * 2
    return NumpyExtensionArray._from_sequence(arr, dtype=dtype)


@pytest.fixture
def skip_numpy_object(dtype, request):
    """
    Tests for NumpyExtensionArray with nested data. Users typically won't create
    these objects via `pd.array`, but they can show up through `.array`
    on a Series with nested data. Many of the base tests fail, as they aren't
    appropriate for nested data.

    This fixture allows these tests to be skipped when used as a usefixtures
    marker to either an individual test or a test class.
    """
    if dtype == "object":
        mark = pytest.mark.xfail(reason="Fails for object dtype")
        request.node.add_marker(mark)


skip_nested = pytest.mark.usefixtures("skip_numpy_object")


class BaseNumPyTests:
    pass


class TestCasting(BaseNumPyTests, base.BaseCastingTests):
    pass


class TestConstructors(BaseNumPyTests, base.BaseConstructorsTests):
    @pytest.mark.skip(reason="We don't register our dtype")
    # We don't want to register. This test should probably be split in two.
    def test_from_dtype(self, data):
        pass

    @skip_nested
    def test_series_constructor_scalar_with_index(self, data, dtype):
        # ValueError: Length of passed values is 1, index implies 3.
        super().test_series_constructor_scalar_with_index(data, dtype)


class TestDtype(BaseNumPyTests, base.BaseDtypeTests):
    def test_check_dtype(self, data, request):
        if data.dtype.numpy_dtype == "object":
            request.node.add_marker(
                pytest.mark.xfail(
                    reason=f"NumpyExtensionArray expectedly clashes with a "
                    f"NumPy name: {data.dtype.numpy_dtype}"
                )
            )
        super().test_check_dtype(data)

    def test_is_not_object_type(self, dtype, request):
        if dtype.numpy_dtype == "object":
            # Different from BaseDtypeTests.test_is_not_object_type
            # because NumpyEADtype(object) is an object type
            assert is_object_dtype(dtype)
        else:
            super().test_is_not_object_type(dtype)


class TestGetitem(BaseNumPyTests, base.BaseGetitemTests):
    @skip_nested
    def test_getitem_scalar(self, data):
        # AssertionError
        super().test_getitem_scalar(data)


class TestGroupby(BaseNumPyTests, base.BaseGroupbyTests):
    pass


class TestInterface(BaseNumPyTests, base.BaseInterfaceTests):
    @skip_nested
    def test_array_interface(self, data):
        # NumPy array shape inference
        super().test_array_interface(data)


class TestMethods(BaseNumPyTests, base.BaseMethodsTests):
    @skip_nested
    def test_shift_fill_value(self, data):
        # np.array shape inference. Shift implementation fails.
        super().test_shift_fill_value(data)

    @skip_nested
    def test_fillna_copy_frame(self, data_missing):
        # The "scalar" for this array isn't a scalar.
        super().test_fillna_copy_frame(data_missing)

    @skip_nested
    def test_fillna_copy_series(self, data_missing):
        # The "scalar" for this array isn't a scalar.
        super().test_fillna_copy_series(data_missing)

    @skip_nested
    def test_searchsorted(self, data_for_sorting, as_series):
        # Test setup fails.
        super().test_searchsorted(data_for_sorting, as_series)

    @pytest.mark.xfail(reason="NumpyExtensionArray.diff may fail on dtype")
    def test_diff(self, data, periods):
        return super().test_diff(data, periods)

    def test_insert(self, data, request):
        if data.dtype.numpy_dtype == object:
            mark = pytest.mark.xfail(reason="Dimension mismatch in np.concatenate")
            request.node.add_marker(mark)

        super().test_insert(data)

    @skip_nested
    def test_insert_invalid(self, data, invalid_scalar):
        # NumpyExtensionArray[object] can hold anything, so skip
        super().test_insert_invalid(data, invalid_scalar)


class TestArithmetics(BaseNumPyTests, base.BaseArithmeticOpsTests):
    divmod_exc = None
    series_scalar_exc = None
    frame_scalar_exc = None
    series_array_exc = None

    @skip_nested
    def test_divmod(self, data):
        super().test_divmod(data)

    @skip_nested
    def test_arith_series_with_scalar(self, data, all_arithmetic_operators):
        super().test_arith_series_with_scalar(data, all_arithmetic_operators)

    def test_arith_series_with_array(self, data, all_arithmetic_operators, request):
        opname = all_arithmetic_operators
        if data.dtype.numpy_dtype == object and opname not in ["__add__", "__radd__"]:
            mark = pytest.mark.xfail(reason="Fails for object dtype")
            request.node.add_marker(mark)
        super().test_arith_series_with_array(data, all_arithmetic_operators)

    @skip_nested
    def test_arith_frame_with_scalar(self, data, all_arithmetic_operators):
        super().test_arith_frame_with_scalar(data, all_arithmetic_operators)


class TestPrinting(BaseNumPyTests, base.BasePrintingTests):
    pass


class TestReduce(BaseNumPyTests, base.BaseReduceTests):
    def _supports_reduction(self, obj, op_name: str) -> bool:
        if tm.get_dtype(obj).kind == "O":
            return op_name in ["sum", "min", "max", "any", "all"]
        return True

    def check_reduce(self, s, op_name, skipna):
        res_op = getattr(s, op_name)
        # avoid coercing int -> float. Just cast to the actual numpy type.
        exp_op = getattr(s.astype(s.dtype._dtype), op_name)
        if op_name == "count":
            result = res_op()
            expected = exp_op()
        else:
            result = res_op(skipna=skipna)
            expected = exp_op(skipna=skipna)
        tm.assert_almost_equal(result, expected)

    @pytest.mark.skip("tests not written yet")
    @pytest.mark.parametrize("skipna", [True, False])
    def test_reduce_frame(self, data, all_numeric_reductions, skipna):
        pass


class TestMissing(BaseNumPyTests, base.BaseMissingTests):
    @skip_nested
    def test_fillna_series(self, data_missing):
        # Non-scalar "scalar" values.
        super().test_fillna_series(data_missing)

    @skip_nested
    def test_fillna_frame(self, data_missing):
        # Non-scalar "scalar" values.
        super().test_fillna_frame(data_missing)


class TestReshaping(BaseNumPyTests, base.BaseReshapingTests):
    pass


class TestSetitem(BaseNumPyTests, base.BaseSetitemTests):
    @skip_nested
    def test_setitem_invalid(self, data, invalid_scalar):
        # object dtype can hold anything, so doesn't raise
        super().test_setitem_invalid(data, invalid_scalar)

    @skip_nested
    def test_setitem_sequence_broadcasts(self, data, box_in_series):
        # ValueError: cannot set using a list-like indexer with a different
        # length than the value
        super().test_setitem_sequence_broadcasts(data, box_in_series)

    @skip_nested
    @pytest.mark.parametrize("setter", ["loc", None])
    def test_setitem_mask_broadcast(self, data, setter):
        # ValueError: cannot set using a list-like indexer with a different
        # length than the value
        super().test_setitem_mask_broadcast(data, setter)

    @skip_nested
    def test_setitem_scalar_key_sequence_raise(self, data):
        # Failed: DID NOT RAISE <class 'ValueError'>
        super().test_setitem_scalar_key_sequence_raise(data)

    # TODO: there is some issue with NumpyExtensionArray, therefore,
    #   skip the setitem test for now, and fix it later (GH 31446)

    @skip_nested
    @pytest.mark.parametrize(
        "mask",
        [
            np.array([True, True, True, False, False]),
            pd.array([True, True, True, False, False], dtype="boolean"),
        ],
        ids=["numpy-array", "boolean-array"],
    )
    def test_setitem_mask(self, data, mask, box_in_series):
        super().test_setitem_mask(data, mask, box_in_series)

    @skip_nested
    @pytest.mark.parametrize(
        "idx",
        [[0, 1, 2], pd.array([0, 1, 2], dtype="Int64"), np.array([0, 1, 2])],
        ids=["list", "integer-array", "numpy-array"],
    )
    def test_setitem_integer_array(self, data, idx, box_in_series):
        super().test_setitem_integer_array(data, idx, box_in_series)

    @pytest.mark.parametrize(
        "idx, box_in_series",
        [
            ([0, 1, 2, pd.NA], False),
            pytest.param([0, 1, 2, pd.NA], True, marks=pytest.mark.xfail),
            (pd.array([0, 1, 2, pd.NA], dtype="Int64"), False),
            (pd.array([0, 1, 2, pd.NA], dtype="Int64"), False),
        ],
        ids=["list-False", "list-True", "integer-array-False", "integer-array-True"],
    )
    def test_setitem_integer_with_missing_raises(self, data, idx, box_in_series):
        super().test_setitem_integer_with_missing_raises(data, idx, box_in_series)

    @skip_nested
    def test_setitem_slice(self, data, box_in_series):
        super().test_setitem_slice(data, box_in_series)

    @skip_nested
    def test_setitem_loc_iloc_slice(self, data):
        super().test_setitem_loc_iloc_slice(data)

    def test_setitem_with_expansion_dataframe_column(self, data, full_indexer):
        # https://github.com/pandas-dev/pandas/issues/32395
        df = expected = pd.DataFrame({"data": pd.Series(data)})
        result = pd.DataFrame(index=df.index)

        # because result has object dtype, the attempt to do setting inplace
        #  is successful, and object dtype is retained
        key = full_indexer(df)
        result.loc[key, "data"] = df["data"]

        # base class method has expected = df; NumpyExtensionArray behaves oddly because
        #  we patch _typ for these tests.
        if data.dtype.numpy_dtype != object:
            if not isinstance(key, slice) or key != slice(None):
                expected = pd.DataFrame({"data": data.to_numpy()})
        tm.assert_frame_equal(result, expected)


@skip_nested
class TestParsing(BaseNumPyTests, base.BaseParsingTests):
    pass


class Test2DCompat(BaseNumPyTests, base.NDArrayBacked2DTests):
    pass
