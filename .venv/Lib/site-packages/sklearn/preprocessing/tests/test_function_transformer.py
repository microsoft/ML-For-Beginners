import warnings

import numpy as np
import pytest
from scipy import sparse

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils import _safe_indexing
from sklearn.utils._testing import (
    _convert_container,
    assert_allclose_dense_sparse,
    assert_array_equal,
)


def _make_func(args_store, kwargs_store, func=lambda X, *a, **k: X):
    def _func(X, *args, **kwargs):
        args_store.append(X)
        args_store.extend(args)
        kwargs_store.update(kwargs)
        return func(X)

    return _func


def test_delegate_to_func():
    # (args|kwargs)_store will hold the positional and keyword arguments
    # passed to the function inside the FunctionTransformer.
    args_store = []
    kwargs_store = {}
    X = np.arange(10).reshape((5, 2))
    assert_array_equal(
        FunctionTransformer(_make_func(args_store, kwargs_store)).transform(X),
        X,
        "transform should have returned X unchanged",
    )

    # The function should only have received X.
    assert args_store == [
        X
    ], "Incorrect positional arguments passed to func: {args}".format(args=args_store)

    assert (
        not kwargs_store
    ), "Unexpected keyword arguments passed to func: {args}".format(args=kwargs_store)

    # reset the argument stores.
    args_store[:] = []
    kwargs_store.clear()
    transformed = FunctionTransformer(
        _make_func(args_store, kwargs_store),
    ).transform(X)

    assert_array_equal(
        transformed, X, err_msg="transform should have returned X unchanged"
    )

    # The function should have received X
    assert args_store == [
        X
    ], "Incorrect positional arguments passed to func: {args}".format(args=args_store)

    assert (
        not kwargs_store
    ), "Unexpected keyword arguments passed to func: {args}".format(args=kwargs_store)


def test_np_log():
    X = np.arange(10).reshape((5, 2))

    # Test that the numpy.log example still works.
    assert_array_equal(
        FunctionTransformer(np.log1p).transform(X),
        np.log1p(X),
    )


def test_kw_arg():
    X = np.linspace(0, 1, num=10).reshape((5, 2))

    F = FunctionTransformer(np.around, kw_args=dict(decimals=3))

    # Test that rounding is correct
    assert_array_equal(F.transform(X), np.around(X, decimals=3))


def test_kw_arg_update():
    X = np.linspace(0, 1, num=10).reshape((5, 2))

    F = FunctionTransformer(np.around, kw_args=dict(decimals=3))

    F.kw_args["decimals"] = 1

    # Test that rounding is correct
    assert_array_equal(F.transform(X), np.around(X, decimals=1))


def test_kw_arg_reset():
    X = np.linspace(0, 1, num=10).reshape((5, 2))

    F = FunctionTransformer(np.around, kw_args=dict(decimals=3))

    F.kw_args = dict(decimals=1)

    # Test that rounding is correct
    assert_array_equal(F.transform(X), np.around(X, decimals=1))


def test_inverse_transform():
    X = np.array([1, 4, 9, 16]).reshape((2, 2))

    # Test that inverse_transform works correctly
    F = FunctionTransformer(
        func=np.sqrt,
        inverse_func=np.around,
        inv_kw_args=dict(decimals=3),
    )
    assert_array_equal(
        F.inverse_transform(F.transform(X)),
        np.around(np.sqrt(X), decimals=3),
    )


def test_check_inverse():
    X_dense = np.array([1, 4, 9, 16], dtype=np.float64).reshape((2, 2))

    X_list = [X_dense, sparse.csr_matrix(X_dense), sparse.csc_matrix(X_dense)]

    for X in X_list:
        if sparse.issparse(X):
            accept_sparse = True
        else:
            accept_sparse = False
        trans = FunctionTransformer(
            func=np.sqrt,
            inverse_func=np.around,
            accept_sparse=accept_sparse,
            check_inverse=True,
            validate=True,
        )
        warning_message = (
            "The provided functions are not strictly"
            " inverse of each other. If you are sure you"
            " want to proceed regardless, set"
            " 'check_inverse=False'."
        )
        with pytest.warns(UserWarning, match=warning_message):
            trans.fit(X)

        trans = FunctionTransformer(
            func=np.expm1,
            inverse_func=np.log1p,
            accept_sparse=accept_sparse,
            check_inverse=True,
            validate=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            Xt = trans.fit_transform(X)

        assert_allclose_dense_sparse(X, trans.inverse_transform(Xt))

    # check that we don't check inverse when one of the func or inverse is not
    # provided.
    trans = FunctionTransformer(
        func=np.expm1, inverse_func=None, check_inverse=True, validate=True
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        trans.fit(X_dense)
    trans = FunctionTransformer(
        func=None, inverse_func=np.expm1, check_inverse=True, validate=True
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        trans.fit(X_dense)


def test_function_transformer_frame():
    pd = pytest.importorskip("pandas")
    X_df = pd.DataFrame(np.random.randn(100, 10))
    transformer = FunctionTransformer()
    X_df_trans = transformer.fit_transform(X_df)
    assert hasattr(X_df_trans, "loc")


@pytest.mark.parametrize("X_type", ["array", "series"])
def test_function_transformer_raise_error_with_mixed_dtype(X_type):
    """Check that `FunctionTransformer.check_inverse` raises error on mixed dtype."""
    mapping = {"one": 1, "two": 2, "three": 3, 5: "five", 6: "six"}
    inverse_mapping = {value: key for key, value in mapping.items()}
    dtype = "object"

    data = ["one", "two", "three", "one", "one", 5, 6]
    data = _convert_container(data, X_type, columns_name=["value"], dtype=dtype)

    def func(X):
        return np.array(
            [mapping[_safe_indexing(X, i)] for i in range(X.size)], dtype=object
        )

    def inverse_func(X):
        return _convert_container(
            [inverse_mapping[x] for x in X],
            X_type,
            columns_name=["value"],
            dtype=dtype,
        )

    transformer = FunctionTransformer(
        func=func, inverse_func=inverse_func, validate=False, check_inverse=True
    )

    msg = "'check_inverse' is only supported when all the elements in `X` is numerical."
    with pytest.raises(ValueError, match=msg):
        transformer.fit(data)


def test_function_transformer_support_all_nummerical_dataframes_check_inverse_True():
    """Check support for dataframes with only numerical values."""
    pd = pytest.importorskip("pandas")

    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    transformer = FunctionTransformer(
        func=lambda x: x + 2, inverse_func=lambda x: x - 2, check_inverse=True
    )

    # Does not raise an error
    df_out = transformer.fit_transform(df)
    assert_allclose_dense_sparse(df_out, df + 2)


def test_function_transformer_with_dataframe_and_check_inverse_True():
    """Check error is raised when check_inverse=True.

    Non-regresion test for gh-25261.
    """
    pd = pytest.importorskip("pandas")
    transformer = FunctionTransformer(
        func=lambda x: x, inverse_func=lambda x: x, check_inverse=True
    )

    df_mixed = pd.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]})
    msg = "'check_inverse' is only supported when all the elements in `X` is numerical."
    with pytest.raises(ValueError, match=msg):
        transformer.fit(df_mixed)


@pytest.mark.parametrize(
    "X, feature_names_out, input_features, expected",
    [
        (
            # NumPy inputs, default behavior: generate names
            np.random.rand(100, 3),
            "one-to-one",
            None,
            ("x0", "x1", "x2"),
        ),
        (
            # Pandas input, default behavior: use input feature names
            {"a": np.random.rand(100), "b": np.random.rand(100)},
            "one-to-one",
            None,
            ("a", "b"),
        ),
        (
            # NumPy input, feature_names_out=callable
            np.random.rand(100, 3),
            lambda transformer, input_features: ("a", "b"),
            None,
            ("a", "b"),
        ),
        (
            # Pandas input, feature_names_out=callable
            {"a": np.random.rand(100), "b": np.random.rand(100)},
            lambda transformer, input_features: ("c", "d", "e"),
            None,
            ("c", "d", "e"),
        ),
        (
            # NumPy input, feature_names_out=callable – default input_features
            np.random.rand(100, 3),
            lambda transformer, input_features: tuple(input_features) + ("a",),
            None,
            ("x0", "x1", "x2", "a"),
        ),
        (
            # Pandas input, feature_names_out=callable – default input_features
            {"a": np.random.rand(100), "b": np.random.rand(100)},
            lambda transformer, input_features: tuple(input_features) + ("c",),
            None,
            ("a", "b", "c"),
        ),
        (
            # NumPy input, input_features=list of names
            np.random.rand(100, 3),
            "one-to-one",
            ("a", "b", "c"),
            ("a", "b", "c"),
        ),
        (
            # Pandas input, input_features=list of names
            {"a": np.random.rand(100), "b": np.random.rand(100)},
            "one-to-one",
            ("a", "b"),  # must match feature_names_in_
            ("a", "b"),
        ),
        (
            # NumPy input, feature_names_out=callable, input_features=list
            np.random.rand(100, 3),
            lambda transformer, input_features: tuple(input_features) + ("d",),
            ("a", "b", "c"),
            ("a", "b", "c", "d"),
        ),
        (
            # Pandas input, feature_names_out=callable, input_features=list
            {"a": np.random.rand(100), "b": np.random.rand(100)},
            lambda transformer, input_features: tuple(input_features) + ("c",),
            ("a", "b"),  # must match feature_names_in_
            ("a", "b", "c"),
        ),
    ],
)
@pytest.mark.parametrize("validate", [True, False])
def test_function_transformer_get_feature_names_out(
    X, feature_names_out, input_features, expected, validate
):
    if isinstance(X, dict):
        pd = pytest.importorskip("pandas")
        X = pd.DataFrame(X)

    transformer = FunctionTransformer(
        feature_names_out=feature_names_out, validate=validate
    )
    transformer.fit_transform(X)
    names = transformer.get_feature_names_out(input_features)
    assert isinstance(names, np.ndarray)
    assert names.dtype == object
    assert_array_equal(names, expected)


def test_function_transformer_get_feature_names_out_without_validation():
    transformer = FunctionTransformer(feature_names_out="one-to-one", validate=False)
    X = np.random.rand(100, 2)
    transformer.fit_transform(X)

    names = transformer.get_feature_names_out(("a", "b"))
    assert isinstance(names, np.ndarray)
    assert names.dtype == object
    assert_array_equal(names, ("a", "b"))


def test_function_transformer_feature_names_out_is_None():
    transformer = FunctionTransformer()
    X = np.random.rand(100, 2)
    transformer.fit_transform(X)

    msg = "This 'FunctionTransformer' has no attribute 'get_feature_names_out'"
    with pytest.raises(AttributeError, match=msg):
        transformer.get_feature_names_out()


def test_function_transformer_feature_names_out_uses_estimator():
    def add_n_random_features(X, n):
        return np.concatenate([X, np.random.rand(len(X), n)], axis=1)

    def feature_names_out(transformer, input_features):
        n = transformer.kw_args["n"]
        return list(input_features) + [f"rnd{i}" for i in range(n)]

    transformer = FunctionTransformer(
        func=add_n_random_features,
        feature_names_out=feature_names_out,
        kw_args=dict(n=3),
        validate=True,
    )
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"a": np.random.rand(100), "b": np.random.rand(100)})
    transformer.fit_transform(df)
    names = transformer.get_feature_names_out()

    assert isinstance(names, np.ndarray)
    assert names.dtype == object
    assert_array_equal(names, ("a", "b", "rnd0", "rnd1", "rnd2"))


def test_function_transformer_validate_inverse():
    """Test that function transformer does not reset estimator in
    `inverse_transform`."""

    def add_constant_feature(X):
        X_one = np.ones((X.shape[0], 1))
        return np.concatenate((X, X_one), axis=1)

    def inverse_add_constant(X):
        return X[:, :-1]

    X = np.array([[1, 2], [3, 4], [3, 4]])
    trans = FunctionTransformer(
        func=add_constant_feature,
        inverse_func=inverse_add_constant,
        validate=True,
    )
    X_trans = trans.fit_transform(X)
    assert trans.n_features_in_ == X.shape[1]

    trans.inverse_transform(X_trans)
    assert trans.n_features_in_ == X.shape[1]


@pytest.mark.parametrize(
    "feature_names_out, expected",
    [
        ("one-to-one", ["pet", "color"]),
        [lambda est, names: [f"{n}_out" for n in names], ["pet_out", "color_out"]],
    ],
)
@pytest.mark.parametrize("in_pipeline", [True, False])
def test_get_feature_names_out_dataframe_with_string_data(
    feature_names_out, expected, in_pipeline
):
    """Check that get_feature_names_out works with DataFrames with string data."""
    pd = pytest.importorskip("pandas")
    X = pd.DataFrame({"pet": ["dog", "cat"], "color": ["red", "green"]})

    transformer = FunctionTransformer(feature_names_out=feature_names_out)
    if in_pipeline:
        transformer = make_pipeline(transformer)

    X_trans = transformer.fit_transform(X)
    assert isinstance(X_trans, pd.DataFrame)

    names = transformer.get_feature_names_out()
    assert isinstance(names, np.ndarray)
    assert names.dtype == object
    assert_array_equal(names, expected)


def test_set_output_func():
    """Check behavior of set_output with different settings."""
    pd = pytest.importorskip("pandas")

    X = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 100]})

    ft = FunctionTransformer(np.log, feature_names_out="one-to-one")

    # no warning is raised when feature_names_out is defined
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        ft.set_output(transform="pandas")

    X_trans = ft.fit_transform(X)
    assert isinstance(X_trans, pd.DataFrame)
    assert_array_equal(X_trans.columns, ["a", "b"])

    # If feature_names_out is not defined, then a warning is raised in
    # `set_output`
    ft = FunctionTransformer(lambda x: 2 * x)
    msg = "should return a DataFrame to follow the set_output API"
    with pytest.warns(UserWarning, match=msg):
        ft.set_output(transform="pandas")

    X_trans = ft.fit_transform(X)
    assert isinstance(X_trans, pd.DataFrame)
    assert_array_equal(X_trans.columns, ["a", "b"])
