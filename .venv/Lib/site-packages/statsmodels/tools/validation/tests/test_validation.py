from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest

from statsmodels.tools.validation import (
    array_like,
    PandasWrapper,
    bool_like,
    dict_like,
    float_like,
    int_like,
    string_like,
)

from statsmodels.tools.validation.validation import _right_squeeze


@pytest.fixture(params=[True, False])
def use_pandas(request):
    return request.param


def gen_data(dim, use_pandas):
    if dim == 1:
        out = np.empty(10,)
        if use_pandas:
            out = pd.Series(out)
    elif dim == 2:
        out = np.empty((20, 10))
        if use_pandas:
            out = pd.DataFrame(out)
    else:
        out = np.empty(np.arange(5, 5 + dim))

    return out


class TestArrayLike:
    def test_1d(self, use_pandas):
        data = gen_data(1, use_pandas)
        a = array_like(data, "a")
        assert a.ndim == 1
        assert a.shape == (10,)
        assert type(a) is np.ndarray

        a = array_like(data, "a", ndim=1)
        assert a.ndim == 1
        a = array_like(data, "a", shape=(10,))
        assert a.shape == (10,)
        a = array_like(data, "a", ndim=1, shape=(None,))
        assert a.ndim == 1
        a = array_like(data, "a", ndim=2, shape=(10, 1))
        assert a.ndim == 2
        assert a.shape == (10, 1)

        with pytest.raises(ValueError, match="a is required to have shape"):
            array_like(data, "a", shape=(5,))

    def test_2d(self, use_pandas):
        data = gen_data(2, use_pandas)
        a = array_like(data, "a", ndim=2)
        assert a.ndim == 2
        assert a.shape == (20, 10)
        assert type(a) is np.ndarray

        a = array_like(data, "a", ndim=2)
        assert a.ndim == 2
        a = array_like(data, "a", ndim=2, shape=(20, None))
        assert a.shape == (20, 10)
        a = array_like(data, "a", ndim=2, shape=(20,))
        assert a.shape == (20, 10)
        a = array_like(data, "a", ndim=2, shape=(None, 10))
        assert a.shape == (20, 10)

        a = array_like(data, "a", ndim=2, shape=(None, None))
        assert a.ndim == 2
        a = array_like(data, "a", ndim=3)
        assert a.ndim == 3
        assert a.shape == (20, 10, 1)

        with pytest.raises(ValueError, match="a is required to have shape"):
            array_like(data, "a", ndim=2, shape=(10,))
        with pytest.raises(ValueError, match="a is required to have shape"):
            array_like(data, "a", ndim=2, shape=(20, 20))
        with pytest.raises(ValueError, match="a is required to have shape"):
            array_like(data, "a", ndim=2, shape=(None, 20))
        match = "a is required to have ndim 1 but has ndim 2"
        with pytest.raises(ValueError, match=match):
            array_like(data, "a", ndim=1)
        match = "a must have ndim <= 1"
        with pytest.raises(ValueError, match=match):
            array_like(data, "a", maxdim=1)

    def test_3d(self):
        data = gen_data(3, False)
        a = array_like(data, "a", ndim=3)
        assert a.shape == (5, 6, 7)
        assert a.ndim == 3
        assert type(a) is np.ndarray

        a = array_like(data, "a", ndim=3, shape=(5, None, 7))
        assert a.shape == (5, 6, 7)
        a = array_like(data, "a", ndim=3, shape=(None, None, 7))
        assert a.shape == (5, 6, 7)
        a = array_like(data, "a", ndim=5)
        assert a.shape == (5, 6, 7, 1, 1)
        with pytest.raises(ValueError, match="a is required to have shape"):
            array_like(data, "a", ndim=3, shape=(10,))
        with pytest.raises(ValueError, match="a is required to have shape"):
            array_like(data, "a", ndim=3, shape=(None, None, 5))
        match = "a is required to have ndim 2 but has ndim 3"
        with pytest.raises(ValueError, match=match):
            array_like(data, "a", ndim=2)
        match = "a must have ndim <= 1"
        with pytest.raises(ValueError, match=match):
            array_like(data, "a", maxdim=1)
        match = "a must have ndim <= 2"
        with pytest.raises(ValueError, match=match):
            array_like(data, "a", maxdim=2)

    def test_right_squeeze_and_pad(self):
        data = np.empty((2, 1, 2))
        a = array_like(data, "a", ndim=3)
        assert a.shape == (2, 1, 2)
        data = np.empty((2))
        a = array_like(data, "a", ndim=3)
        assert a.shape == (2, 1, 1)
        data = np.empty((2, 1))
        a = array_like(data, "a", ndim=3)
        assert a.shape == (2, 1, 1)

        data = np.empty((2, 1, 1, 1))
        a = array_like(data, "a", ndim=3)
        assert a.shape == (2, 1, 1)

        data = np.empty((2, 1, 1, 2, 1, 1))
        with pytest.raises(ValueError):
            array_like(data, "a", ndim=3)

    def test_contiguous(self):
        x = np.arange(10)
        y = x[::2]
        a = array_like(y, "a", contiguous=True)
        assert not y.flags["C_CONTIGUOUS"]
        assert a.flags["C_CONTIGUOUS"]

    def test_dtype(self):
        x = np.arange(10)
        a = array_like(x, "a", dtype=np.float32)
        assert a.dtype == np.float32

        a = array_like(x, "a", dtype=np.uint8)
        assert a.dtype == np.uint8

    @pytest.mark.xfail(reason="Failing for now")
    def test_dot(self, use_pandas):
        data = gen_data(2, use_pandas)
        a = array_like(data, "a")
        assert not isinstance(a.T.dot(data), array_like)
        assert not isinstance(a.T.dot(a), array_like)

    def test_slice(self, use_pandas):
        data = gen_data(2, use_pandas)
        a = array_like(data, "a", ndim=2)
        assert type(a[1:]) is np.ndarray


def test_right_squeeze():
    x = np.empty((10, 1, 10))
    y = _right_squeeze(x)
    assert y.shape == (10, 1, 10)

    x = np.empty((10, 10, 1))
    y = _right_squeeze(x)
    assert y.shape == (10, 10)

    x = np.empty((10, 10, 1, 1, 1, 1, 1))
    y = _right_squeeze(x)
    assert y.shape == (10, 10)

    x = np.empty((10, 1, 10, 1, 1, 1, 1, 1))
    y = _right_squeeze(x)
    assert y.shape == (10, 1, 10)


def test_wrap_pandas(use_pandas):
    a = gen_data(1, use_pandas)
    b = gen_data(1, False)

    wrapped = PandasWrapper(a).wrap(b)
    expected_type = pd.Series if use_pandas else np.ndarray
    assert isinstance(wrapped, expected_type)
    assert not use_pandas or wrapped.name is None

    wrapped = PandasWrapper(a).wrap(b, columns="name")
    assert isinstance(wrapped, expected_type)
    assert not use_pandas or wrapped.name == "name"

    wrapped = PandasWrapper(a).wrap(b, columns=["name"])
    assert isinstance(wrapped, expected_type)
    assert not use_pandas or wrapped.name == "name"

    expected_type = pd.DataFrame if use_pandas else np.ndarray
    wrapped = PandasWrapper(a).wrap(b[:, None])
    assert isinstance(wrapped, expected_type)
    assert not use_pandas or wrapped.columns[0] == 0

    wrapped = PandasWrapper(a).wrap(b[:, None], columns=["name"])
    assert isinstance(wrapped, expected_type)
    assert not use_pandas or wrapped.columns == ["name"]

    if use_pandas:
        match = "Can only wrap 1 or 2-d array_like"
        with pytest.raises(ValueError, match=match):
            PandasWrapper(a).wrap(b[:, None, None])

        match = "obj must have the same number of elements in axis 0 as"
        with pytest.raises(ValueError, match=match):
            PandasWrapper(a).wrap(b[: b.shape[0] // 2])


def test_wrap_pandas_append():
    a = gen_data(1, True)
    a.name = "apple"
    b = gen_data(1, False)
    wrapped = PandasWrapper(a).wrap(b, append="appended")
    expected = "apple_appended"
    assert wrapped.name == expected

    a = gen_data(2, True)
    a.columns = ["apple_" + str(i) for i in range(a.shape[1])]
    b = gen_data(2, False)
    wrapped = PandasWrapper(a).wrap(b, append="appended")
    expected = [c + "_appended" for c in a.columns]
    assert list(wrapped.columns) == expected


def test_wrap_pandas_append_non_string():
    # GH 6826
    a = gen_data(1, True)
    a.name = 7
    b = gen_data(1, False)
    wrapped = PandasWrapper(a).wrap(b, append="appended")
    expected = "7_appended"
    assert wrapped.name == expected

    a = gen_data(2, True)
    a.columns = [i for i in range(a.shape[1])]
    b = gen_data(2, False)
    wrapped = PandasWrapper(a).wrap(b, append="appended")
    expected = [f"{c}_appended" for c in a.columns]
    assert list(wrapped.columns) == expected


class CustomDict(dict):
    pass


@pytest.fixture(params=(dict, OrderedDict, CustomDict, None))
def dict_type(request):
    return request.param


def test_optional_dict_like(dict_type):
    val = dict_type() if dict_type is not None else dict_type
    out = dict_like(val, "value", optional=True)
    assert isinstance(out, type(val))


def test_optional_dict_like_error():
    match = r"value must be a dict or dict_like \(i.e., a Mapping\)"
    with pytest.raises(TypeError, match=match):
        dict_like([], "value", optional=True)
    with pytest.raises(TypeError, match=match):
        dict_like({"a"}, "value", optional=True)
    with pytest.raises(TypeError, match=match):
        dict_like("a", "value", optional=True)


def test_string():
    out = string_like("apple", "value")
    assert out == "apple"

    out = string_like("apple", "value", options=("apple", "banana", "cherry"))
    assert out == "apple"

    with pytest.raises(TypeError, match="value must be a string"):
        string_like(1, "value")
    with pytest.raises(TypeError, match="value must be a string"):
        string_like(b"4", "value")
    with pytest.raises(
        ValueError,
        match="value must be one of: 'apple'," " 'banana', 'cherry'",
    ):
        string_like("date", "value", options=("apple", "banana", "cherry"))


def test_optional_string():
    out = string_like("apple", "value")
    assert out == "apple"

    out = string_like("apple", "value", options=("apple", "banana", "cherry"))
    assert out == "apple"

    out = string_like(None, "value", optional=True)
    assert out is None

    out = string_like(
        None, "value", optional=True, options=("apple", "banana", "cherry")
    )
    assert out is None

    with pytest.raises(TypeError, match="value must be a string"):
        string_like(1, "value", optional=True)
    with pytest.raises(TypeError, match="value must be a string"):
        string_like(b"4", "value", optional=True)


@pytest.fixture(params=(1.0, 1.1, np.float32(1.2), np.array([1.2]), 1.2 + 0j))
def floating(request):
    return request.param


@pytest.fixture(params=(np.empty(2), 1.2 + 1j, True, "3.2", None))
def not_floating(request):
    return request.param


def test_float_like(floating):
    assert isinstance(float_like(floating, "floating"), float)
    assert isinstance(float_like(floating, "floating", optional=True), float)
    assert float_like(None, "floating", optional=True) is None
    if isinstance(floating, (int, np.integer, float, np.inexact)):
        assert isinstance(float_like(floating, "floating", strict=True), float)
        assert float_like(None, "floating", optional=True, strict=True) is None


def test_not_float_like(not_floating):
    with pytest.raises(TypeError):
        float_like(not_floating, "floating")


@pytest.fixture(params=(1.0, 2, np.float32(3.0), np.array([4.0])))
def integer(request):
    return request.param


@pytest.fixture(
    params=(
        3.2,
        np.float32(3.2),
        3 + 2j,
        complex(2.3 + 0j),
        "apple",
        1.0 + 0j,
        np.timedelta64(2),
    )
)
def not_integer(request):
    return request.param


def test_int_like(integer):
    assert isinstance(int_like(integer, "integer"), int)
    assert isinstance(int_like(integer, "integer", optional=True), int)
    assert int_like(None, "floating", optional=True) is None
    if isinstance(integer, (int, np.integer)):
        assert isinstance(int_like(integer, "integer", strict=True), int)
        assert int_like(None, "floating", optional=True, strict=True) is None


def test_not_int_like(not_integer):
    with pytest.raises(TypeError):
        int_like(not_integer, "integer")


@pytest.fixture(params=[True, False, 1, 1.2, "a", ""])
def boolean(request):
    return request.param


def test_bool_like(boolean):
    assert isinstance(bool_like(boolean, "boolean"), bool)
    assert bool_like(None, "boolean", optional=True) is None
    if isinstance(boolean, bool):
        assert isinstance(bool_like(boolean, "boolean", strict=True), bool)
    else:
        with pytest.raises(TypeError):
            bool_like(boolean, "boolean", strict=True)


def test_not_bool_like():
    with pytest.raises(TypeError):
        bool_like(np.array([True, True]), boolean)
