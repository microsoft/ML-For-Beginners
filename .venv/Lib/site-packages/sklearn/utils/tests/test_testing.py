import atexit
import os
import unittest
import warnings

import numpy as np
import pytest
from scipy import sparse

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import _IS_WASM
from sklearn.utils._testing import (
    TempMemmap,
    _convert_container,
    _delete_folder,
    assert_allclose,
    assert_allclose_dense_sparse,
    assert_no_warnings,
    assert_raise_message,
    assert_raises,
    assert_raises_regex,
    check_docstring_parameters,
    create_memmap_backed_data,
    ignore_warnings,
    raises,
    set_random_state,
)
from sklearn.utils.deprecation import deprecated
from sklearn.utils.fixes import (
    CSC_CONTAINERS,
    CSR_CONTAINERS,
    parse_version,
    sp_version,
)
from sklearn.utils.metaestimators import available_if


def test_set_random_state():
    lda = LinearDiscriminantAnalysis()
    tree = DecisionTreeClassifier()
    # Linear Discriminant Analysis doesn't have random state: smoke test
    set_random_state(lda, 3)
    set_random_state(tree, 3)
    assert tree.random_state == 3


@pytest.mark.parametrize("csr_container", CSC_CONTAINERS)
def test_assert_allclose_dense_sparse(csr_container):
    x = np.arange(9).reshape(3, 3)
    msg = "Not equal to tolerance "
    y = csr_container(x)
    for X in [x, y]:
        # basic compare
        with pytest.raises(AssertionError, match=msg):
            assert_allclose_dense_sparse(X, X * 2)
        assert_allclose_dense_sparse(X, X)

    with pytest.raises(ValueError, match="Can only compare two sparse"):
        assert_allclose_dense_sparse(x, y)

    A = sparse.diags(np.ones(5), offsets=0).tocsr()
    B = csr_container(np.ones((1, 5)))
    with pytest.raises(AssertionError, match="Arrays are not equal"):
        assert_allclose_dense_sparse(B, A)


def test_assert_raises_msg():
    with assert_raises_regex(AssertionError, "Hello world"):
        with assert_raises(ValueError, msg="Hello world"):
            pass


def test_assert_raise_message():
    def _raise_ValueError(message):
        raise ValueError(message)

    def _no_raise():
        pass

    assert_raise_message(ValueError, "test", _raise_ValueError, "test")

    assert_raises(
        AssertionError,
        assert_raise_message,
        ValueError,
        "something else",
        _raise_ValueError,
        "test",
    )

    assert_raises(
        ValueError,
        assert_raise_message,
        TypeError,
        "something else",
        _raise_ValueError,
        "test",
    )

    assert_raises(AssertionError, assert_raise_message, ValueError, "test", _no_raise)

    # multiple exceptions in a tuple
    assert_raises(
        AssertionError,
        assert_raise_message,
        (ValueError, AttributeError),
        "test",
        _no_raise,
    )


def test_ignore_warning():
    # This check that ignore_warning decorator and context manager are working
    # as expected
    def _warning_function():
        warnings.warn("deprecation warning", DeprecationWarning)

    def _multiple_warning_function():
        warnings.warn("deprecation warning", DeprecationWarning)
        warnings.warn("deprecation warning")

    # Check the function directly
    assert_no_warnings(ignore_warnings(_warning_function))
    assert_no_warnings(ignore_warnings(_warning_function, category=DeprecationWarning))
    with pytest.warns(DeprecationWarning):
        ignore_warnings(_warning_function, category=UserWarning)()
    with pytest.warns(UserWarning):
        ignore_warnings(_multiple_warning_function, category=FutureWarning)()
    with pytest.warns(DeprecationWarning):
        ignore_warnings(_multiple_warning_function, category=UserWarning)()
    assert_no_warnings(
        ignore_warnings(_warning_function, category=(DeprecationWarning, UserWarning))
    )

    # Check the decorator
    @ignore_warnings
    def decorator_no_warning():
        _warning_function()
        _multiple_warning_function()

    @ignore_warnings(category=(DeprecationWarning, UserWarning))
    def decorator_no_warning_multiple():
        _multiple_warning_function()

    @ignore_warnings(category=DeprecationWarning)
    def decorator_no_deprecation_warning():
        _warning_function()

    @ignore_warnings(category=UserWarning)
    def decorator_no_user_warning():
        _warning_function()

    @ignore_warnings(category=DeprecationWarning)
    def decorator_no_deprecation_multiple_warning():
        _multiple_warning_function()

    @ignore_warnings(category=UserWarning)
    def decorator_no_user_multiple_warning():
        _multiple_warning_function()

    assert_no_warnings(decorator_no_warning)
    assert_no_warnings(decorator_no_warning_multiple)
    assert_no_warnings(decorator_no_deprecation_warning)
    with pytest.warns(DeprecationWarning):
        decorator_no_user_warning()
    with pytest.warns(UserWarning):
        decorator_no_deprecation_multiple_warning()
    with pytest.warns(DeprecationWarning):
        decorator_no_user_multiple_warning()

    # Check the context manager
    def context_manager_no_warning():
        with ignore_warnings():
            _warning_function()

    def context_manager_no_warning_multiple():
        with ignore_warnings(category=(DeprecationWarning, UserWarning)):
            _multiple_warning_function()

    def context_manager_no_deprecation_warning():
        with ignore_warnings(category=DeprecationWarning):
            _warning_function()

    def context_manager_no_user_warning():
        with ignore_warnings(category=UserWarning):
            _warning_function()

    def context_manager_no_deprecation_multiple_warning():
        with ignore_warnings(category=DeprecationWarning):
            _multiple_warning_function()

    def context_manager_no_user_multiple_warning():
        with ignore_warnings(category=UserWarning):
            _multiple_warning_function()

    assert_no_warnings(context_manager_no_warning)
    assert_no_warnings(context_manager_no_warning_multiple)
    assert_no_warnings(context_manager_no_deprecation_warning)
    with pytest.warns(DeprecationWarning):
        context_manager_no_user_warning()
    with pytest.warns(UserWarning):
        context_manager_no_deprecation_multiple_warning()
    with pytest.warns(DeprecationWarning):
        context_manager_no_user_multiple_warning()

    # Check that passing warning class as first positional argument
    warning_class = UserWarning
    match = "'obj' should be a callable.+you should use 'category=UserWarning'"

    with pytest.raises(ValueError, match=match):
        silence_warnings_func = ignore_warnings(warning_class)(_warning_function)
        silence_warnings_func()

    with pytest.raises(ValueError, match=match):

        @ignore_warnings(warning_class)
        def test():
            pass


class TestWarns(unittest.TestCase):
    def test_warn(self):
        def f():
            warnings.warn("yo")
            return 3

        with pytest.raises(AssertionError):
            assert_no_warnings(f)
        assert assert_no_warnings(lambda x: x, 1) == 1


# Tests for docstrings:


def f_ok(a, b):
    """Function f

    Parameters
    ----------
    a : int
        Parameter a
    b : float
        Parameter b

    Returns
    -------
    c : list
        Parameter c
    """
    c = a + b
    return c


def f_bad_sections(a, b):
    """Function f

    Parameters
    ----------
    a : int
        Parameter a
    b : float
        Parameter b

    Results
    -------
    c : list
        Parameter c
    """
    c = a + b
    return c


def f_bad_order(b, a):
    """Function f

    Parameters
    ----------
    a : int
        Parameter a
    b : float
        Parameter b

    Returns
    -------
    c : list
        Parameter c
    """
    c = a + b
    return c


def f_too_many_param_docstring(a, b):
    """Function f

    Parameters
    ----------
    a : int
        Parameter a
    b : int
        Parameter b
    c : int
        Parameter c

    Returns
    -------
    d : list
        Parameter c
    """
    d = a + b
    return d


def f_missing(a, b):
    """Function f

    Parameters
    ----------
    a : int
        Parameter a

    Returns
    -------
    c : list
        Parameter c
    """
    c = a + b
    return c


def f_check_param_definition(a, b, c, d, e):
    """Function f

    Parameters
    ----------
    a: int
        Parameter a
    b:
        Parameter b
    c :
        This is parsed correctly in numpydoc 1.2
    d:int
        Parameter d
    e
        No typespec is allowed without colon
    """
    return a + b + c + d


class Klass:
    def f_missing(self, X, y):
        pass

    def f_bad_sections(self, X, y):
        """Function f

        Parameter
        ---------
        a : int
            Parameter a
        b : float
            Parameter b

        Results
        -------
        c : list
            Parameter c
        """
        pass


class MockEst:
    def __init__(self):
        """MockEstimator"""

    def fit(self, X, y):
        return X

    def predict(self, X):
        return X

    def predict_proba(self, X):
        return X

    def score(self, X):
        return 1.0


class MockMetaEstimator:
    def __init__(self, delegate):
        """MetaEstimator to check if doctest on delegated methods work.

        Parameters
        ---------
        delegate : estimator
            Delegated estimator.
        """
        self.delegate = delegate

    @available_if(lambda self: hasattr(self.delegate, "predict"))
    def predict(self, X):
        """This is available only if delegate has predict.

        Parameters
        ----------
        y : ndarray
            Parameter y
        """
        return self.delegate.predict(X)

    @available_if(lambda self: hasattr(self.delegate, "score"))
    @deprecated("Testing a deprecated delegated method")
    def score(self, X):
        """This is available only if delegate has score.

        Parameters
        ---------
        y : ndarray
            Parameter y
        """

    @available_if(lambda self: hasattr(self.delegate, "predict_proba"))
    def predict_proba(self, X):
        """This is available only if delegate has predict_proba.

        Parameters
        ---------
        X : ndarray
            Parameter X
        """
        return X

    @deprecated("Testing deprecated function with wrong params")
    def fit(self, X, y):
        """Incorrect docstring but should not be tested"""


def test_check_docstring_parameters():
    pytest.importorskip(
        "numpydoc",
        reason="numpydoc is required to test the docstrings",
        minversion="1.2.0",
    )

    incorrect = check_docstring_parameters(f_ok)
    assert incorrect == []
    incorrect = check_docstring_parameters(f_ok, ignore=["b"])
    assert incorrect == []
    incorrect = check_docstring_parameters(f_missing, ignore=["b"])
    assert incorrect == []
    with pytest.raises(RuntimeError, match="Unknown section Results"):
        check_docstring_parameters(f_bad_sections)
    with pytest.raises(RuntimeError, match="Unknown section Parameter"):
        check_docstring_parameters(Klass.f_bad_sections)

    incorrect = check_docstring_parameters(f_check_param_definition)
    mock_meta = MockMetaEstimator(delegate=MockEst())
    mock_meta_name = mock_meta.__class__.__name__
    assert incorrect == [
        (
            "sklearn.utils.tests.test_testing.f_check_param_definition There "
            "was no space between the param name and colon ('a: int')"
        ),
        (
            "sklearn.utils.tests.test_testing.f_check_param_definition There "
            "was no space between the param name and colon ('b:')"
        ),
        (
            "sklearn.utils.tests.test_testing.f_check_param_definition There "
            "was no space between the param name and colon ('d:int')"
        ),
    ]

    messages = [
        [
            "In function: sklearn.utils.tests.test_testing.f_bad_order",
            (
                "There's a parameter name mismatch in function docstring w.r.t."
                " function signature, at index 0 diff: 'b' != 'a'"
            ),
            "Full diff:",
            "- ['b', 'a']",
            "+ ['a', 'b']",
        ],
        [
            "In function: "
            + "sklearn.utils.tests.test_testing.f_too_many_param_docstring",
            (
                "Parameters in function docstring have more items w.r.t. function"
                " signature, first extra item: c"
            ),
            "Full diff:",
            "- ['a', 'b']",
            "+ ['a', 'b', 'c']",
            "?          +++++",
        ],
        [
            "In function: sklearn.utils.tests.test_testing.f_missing",
            (
                "Parameters in function docstring have less items w.r.t. function"
                " signature, first missing item: b"
            ),
            "Full diff:",
            "- ['a', 'b']",
            "+ ['a']",
        ],
        [
            "In function: sklearn.utils.tests.test_testing.Klass.f_missing",
            (
                "Parameters in function docstring have less items w.r.t. function"
                " signature, first missing item: X"
            ),
            "Full diff:",
            "- ['X', 'y']",
            "+ []",
        ],
        [
            "In function: "
            + f"sklearn.utils.tests.test_testing.{mock_meta_name}.predict",
            (
                "There's a parameter name mismatch in function docstring w.r.t."
                " function signature, at index 0 diff: 'X' != 'y'"
            ),
            "Full diff:",
            "- ['X']",
            "?   ^",
            "+ ['y']",
            "?   ^",
        ],
        [
            "In function: "
            + f"sklearn.utils.tests.test_testing.{mock_meta_name}."
            + "predict_proba",
            "potentially wrong underline length... ",
            "Parameters ",
            "--------- in ",
        ],
        [
            "In function: "
            + f"sklearn.utils.tests.test_testing.{mock_meta_name}.score",
            "potentially wrong underline length... ",
            "Parameters ",
            "--------- in ",
        ],
        [
            "In function: " + f"sklearn.utils.tests.test_testing.{mock_meta_name}.fit",
            (
                "Parameters in function docstring have less items w.r.t. function"
                " signature, first missing item: X"
            ),
            "Full diff:",
            "- ['X', 'y']",
            "+ []",
        ],
    ]

    for msg, f in zip(
        messages,
        [
            f_bad_order,
            f_too_many_param_docstring,
            f_missing,
            Klass.f_missing,
            mock_meta.predict,
            mock_meta.predict_proba,
            mock_meta.score,
            mock_meta.fit,
        ],
    ):
        incorrect = check_docstring_parameters(f)
        assert msg == incorrect, '\n"%s"\n not in \n"%s"' % (msg, incorrect)


class RegistrationCounter:
    def __init__(self):
        self.nb_calls = 0

    def __call__(self, to_register_func):
        self.nb_calls += 1
        assert to_register_func.func is _delete_folder


def check_memmap(input_array, mmap_data, mmap_mode="r"):
    assert isinstance(mmap_data, np.memmap)
    writeable = mmap_mode != "r"
    assert mmap_data.flags.writeable is writeable
    np.testing.assert_array_equal(input_array, mmap_data)


def test_tempmemmap(monkeypatch):
    registration_counter = RegistrationCounter()
    monkeypatch.setattr(atexit, "register", registration_counter)

    input_array = np.ones(3)
    with TempMemmap(input_array) as data:
        check_memmap(input_array, data)
        temp_folder = os.path.dirname(data.filename)
    if os.name != "nt":
        assert not os.path.exists(temp_folder)
    assert registration_counter.nb_calls == 1

    mmap_mode = "r+"
    with TempMemmap(input_array, mmap_mode=mmap_mode) as data:
        check_memmap(input_array, data, mmap_mode=mmap_mode)
        temp_folder = os.path.dirname(data.filename)
    if os.name != "nt":
        assert not os.path.exists(temp_folder)
    assert registration_counter.nb_calls == 2


@pytest.mark.xfail(_IS_WASM, reason="memmap not fully supported")
def test_create_memmap_backed_data(monkeypatch):
    registration_counter = RegistrationCounter()
    monkeypatch.setattr(atexit, "register", registration_counter)

    input_array = np.ones(3)
    data = create_memmap_backed_data(input_array)
    check_memmap(input_array, data)
    assert registration_counter.nb_calls == 1

    data, folder = create_memmap_backed_data(input_array, return_folder=True)
    check_memmap(input_array, data)
    assert folder == os.path.dirname(data.filename)
    assert registration_counter.nb_calls == 2

    mmap_mode = "r+"
    data = create_memmap_backed_data(input_array, mmap_mode=mmap_mode)
    check_memmap(input_array, data, mmap_mode)
    assert registration_counter.nb_calls == 3

    input_list = [input_array, input_array + 1, input_array + 2]
    mmap_data_list = create_memmap_backed_data(input_list)
    for input_array, data in zip(input_list, mmap_data_list):
        check_memmap(input_array, data)
    assert registration_counter.nb_calls == 4

    output_data, other = create_memmap_backed_data([input_array, "not-an-array"])
    check_memmap(input_array, output_data)
    assert other == "not-an-array"


@pytest.mark.parametrize(
    "constructor_name, container_type",
    [
        ("list", list),
        ("tuple", tuple),
        ("array", np.ndarray),
        ("sparse", sparse.csr_matrix),
        # using `zip` will only keep the available sparse containers
        # depending of the installed SciPy version
        *zip(["sparse_csr", "sparse_csr_array"], CSR_CONTAINERS),
        *zip(["sparse_csc", "sparse_csc_array"], CSC_CONTAINERS),
        ("dataframe", lambda: pytest.importorskip("pandas").DataFrame),
        ("series", lambda: pytest.importorskip("pandas").Series),
        ("index", lambda: pytest.importorskip("pandas").Index),
        ("slice", slice),
    ],
)
@pytest.mark.parametrize(
    "dtype, superdtype",
    [
        (np.int32, np.integer),
        (np.int64, np.integer),
        (np.float32, np.floating),
        (np.float64, np.floating),
    ],
)
def test_convert_container(
    constructor_name,
    container_type,
    dtype,
    superdtype,
):
    """Check that we convert the container to the right type of array with the
    right data type."""
    if constructor_name in ("dataframe", "series", "index"):
        # delay the import of pandas within the function to only skip this test
        # instead of the whole file
        container_type = container_type()
    container = [0, 1]

    container_converted = _convert_container(
        container,
        constructor_name,
        dtype=dtype,
    )
    assert isinstance(container_converted, container_type)

    if constructor_name in ("list", "tuple", "index"):
        # list and tuple will use Python class dtype: int, float
        # pandas index will always use high precision: np.int64 and np.float64
        assert np.issubdtype(type(container_converted[0]), superdtype)
    elif hasattr(container_converted, "dtype"):
        assert container_converted.dtype == dtype
    elif hasattr(container_converted, "dtypes"):
        assert container_converted.dtypes[0] == dtype


def test_convert_container_categories_pandas():
    pytest.importorskip("pandas")
    df = _convert_container(
        [["x"]], "dataframe", ["A"], categorical_feature_names=["A"]
    )
    assert df.dtypes.iloc[0] == "category"


def test_convert_container_categories_polars():
    pl = pytest.importorskip("polars")
    df = _convert_container([["x"]], "polars", ["A"], categorical_feature_names=["A"])
    assert df.schema["A"] == pl.Categorical()


def test_convert_container_categories_pyarrow():
    pa = pytest.importorskip("pyarrow")
    df = _convert_container([["x"]], "pyarrow", ["A"], categorical_feature_names=["A"])
    assert type(df.schema[0].type) is pa.DictionaryType


@pytest.mark.skipif(
    sp_version >= parse_version("1.8"),
    reason="sparse arrays are available as of scipy 1.8.0",
)
@pytest.mark.parametrize("constructor_name", ["sparse_csr_array", "sparse_csc_array"])
@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])
def test_convert_container_raise_when_sparray_not_available(constructor_name, dtype):
    """Check that if we convert to sparse array but sparse array are not supported
    (scipy<1.8.0), we should raise an explicit error."""
    container = [0, 1]

    with pytest.raises(
        ValueError,
        match=f"only available with scipy>=1.8.0, got {sp_version}",
    ):
        _convert_container(container, constructor_name, dtype=dtype)


def test_raises():
    # Tests for the raises context manager

    # Proper type, no match
    with raises(TypeError):
        raise TypeError()

    # Proper type, proper match
    with raises(TypeError, match="how are you") as cm:
        raise TypeError("hello how are you")
    assert cm.raised_and_matched

    # Proper type, proper match with multiple patterns
    with raises(TypeError, match=["not this one", "how are you"]) as cm:
        raise TypeError("hello how are you")
    assert cm.raised_and_matched

    # bad type, no match
    with pytest.raises(ValueError, match="this will be raised"):
        with raises(TypeError) as cm:
            raise ValueError("this will be raised")
    assert not cm.raised_and_matched

    # Bad type, no match, with a err_msg
    with pytest.raises(AssertionError, match="the failure message"):
        with raises(TypeError, err_msg="the failure message") as cm:
            raise ValueError()
    assert not cm.raised_and_matched

    # bad type, with match (is ignored anyway)
    with pytest.raises(ValueError, match="this will be raised"):
        with raises(TypeError, match="this is ignored") as cm:
            raise ValueError("this will be raised")
    assert not cm.raised_and_matched

    # proper type but bad match
    with pytest.raises(
        AssertionError, match="should contain one of the following patterns"
    ):
        with raises(TypeError, match="hello") as cm:
            raise TypeError("Bad message")
    assert not cm.raised_and_matched

    # proper type but bad match, with err_msg
    with pytest.raises(AssertionError, match="the failure message"):
        with raises(TypeError, match="hello", err_msg="the failure message") as cm:
            raise TypeError("Bad message")
    assert not cm.raised_and_matched

    # no raise with default may_pass=False
    with pytest.raises(AssertionError, match="Did not raise"):
        with raises(TypeError) as cm:
            pass
    assert not cm.raised_and_matched

    # no raise with may_pass=True
    with raises(TypeError, match="hello", may_pass=True) as cm:
        pass  # still OK
    assert not cm.raised_and_matched

    # Multiple exception types:
    with raises((TypeError, ValueError)):
        raise TypeError()
    with raises((TypeError, ValueError)):
        raise ValueError()
    with pytest.raises(AssertionError):
        with raises((TypeError, ValueError)):
            pass


def test_float32_aware_assert_allclose():
    # The relative tolerance for float32 inputs is 1e-4
    assert_allclose(np.array([1.0 + 2e-5], dtype=np.float32), 1.0)
    with pytest.raises(AssertionError):
        assert_allclose(np.array([1.0 + 2e-4], dtype=np.float32), 1.0)

    # The relative tolerance for other inputs is left to 1e-7 as in
    # the original numpy version.
    assert_allclose(np.array([1.0 + 2e-8], dtype=np.float64), 1.0)
    with pytest.raises(AssertionError):
        assert_allclose(np.array([1.0 + 2e-7], dtype=np.float64), 1.0)

    # atol is left to 0.0 by default, even for float32
    with pytest.raises(AssertionError):
        assert_allclose(np.array([1e-5], dtype=np.float32), 0.0)
    assert_allclose(np.array([1e-5], dtype=np.float32), 0.0, atol=2e-5)
