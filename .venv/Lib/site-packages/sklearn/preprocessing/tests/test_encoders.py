import re

import numpy as np
import pytest
from scipy import sparse

from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
    _convert_container,
    assert_allclose,
    assert_array_equal,
)
from sklearn.utils.fixes import CSR_CONTAINERS


def test_one_hot_encoder_sparse_dense():
    # check that sparse and dense will give the same results

    X = np.array([[3, 2, 1], [0, 1, 1]])
    enc_sparse = OneHotEncoder()
    enc_dense = OneHotEncoder(sparse_output=False)

    X_trans_sparse = enc_sparse.fit_transform(X)
    X_trans_dense = enc_dense.fit_transform(X)

    assert X_trans_sparse.shape == (2, 5)
    assert X_trans_dense.shape == (2, 5)

    assert sparse.issparse(X_trans_sparse)
    assert not sparse.issparse(X_trans_dense)

    # check outcome
    assert_array_equal(
        X_trans_sparse.toarray(), [[0.0, 1.0, 0.0, 1.0, 1.0], [1.0, 0.0, 1.0, 0.0, 1.0]]
    )
    assert_array_equal(X_trans_sparse.toarray(), X_trans_dense)


@pytest.mark.parametrize("handle_unknown", ["ignore", "infrequent_if_exist"])
def test_one_hot_encoder_handle_unknown(handle_unknown):
    X = np.array([[0, 2, 1], [1, 0, 3], [1, 0, 2]])
    X2 = np.array([[4, 1, 1]])

    # Test that one hot encoder raises error for unknown features
    # present during transform.
    oh = OneHotEncoder(handle_unknown="error")
    oh.fit(X)
    with pytest.raises(ValueError, match="Found unknown categories"):
        oh.transform(X2)

    # Test the ignore option, ignores unknown features (giving all 0's)
    oh = OneHotEncoder(handle_unknown=handle_unknown)
    oh.fit(X)
    X2_passed = X2.copy()
    assert_array_equal(
        oh.transform(X2_passed).toarray(),
        np.array([[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]),
    )
    # ensure transformed data was not modified in place
    assert_allclose(X2, X2_passed)


@pytest.mark.parametrize("handle_unknown", ["ignore", "infrequent_if_exist"])
def test_one_hot_encoder_handle_unknown_strings(handle_unknown):
    X = np.array(["11111111", "22", "333", "4444"]).reshape((-1, 1))
    X2 = np.array(["55555", "22"]).reshape((-1, 1))
    # Non Regression test for the issue #12470
    # Test the ignore option, when categories are numpy string dtype
    # particularly when the known category strings are larger
    # than the unknown category strings
    oh = OneHotEncoder(handle_unknown=handle_unknown)
    oh.fit(X)
    X2_passed = X2.copy()
    assert_array_equal(
        oh.transform(X2_passed).toarray(),
        np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
    )
    # ensure transformed data was not modified in place
    assert_array_equal(X2, X2_passed)


@pytest.mark.parametrize("output_dtype", [np.int32, np.float32, np.float64])
@pytest.mark.parametrize("input_dtype", [np.int32, np.float32, np.float64])
def test_one_hot_encoder_dtype(input_dtype, output_dtype):
    X = np.asarray([[0, 1]], dtype=input_dtype).T
    X_expected = np.asarray([[1, 0], [0, 1]], dtype=output_dtype)

    oh = OneHotEncoder(categories="auto", dtype=output_dtype)
    assert_array_equal(oh.fit_transform(X).toarray(), X_expected)
    assert_array_equal(oh.fit(X).transform(X).toarray(), X_expected)

    oh = OneHotEncoder(categories="auto", dtype=output_dtype, sparse_output=False)
    assert_array_equal(oh.fit_transform(X), X_expected)
    assert_array_equal(oh.fit(X).transform(X), X_expected)


@pytest.mark.parametrize("output_dtype", [np.int32, np.float32, np.float64])
def test_one_hot_encoder_dtype_pandas(output_dtype):
    pd = pytest.importorskip("pandas")

    X_df = pd.DataFrame({"A": ["a", "b"], "B": [1, 2]})
    X_expected = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=output_dtype)

    oh = OneHotEncoder(dtype=output_dtype)
    assert_array_equal(oh.fit_transform(X_df).toarray(), X_expected)
    assert_array_equal(oh.fit(X_df).transform(X_df).toarray(), X_expected)

    oh = OneHotEncoder(dtype=output_dtype, sparse_output=False)
    assert_array_equal(oh.fit_transform(X_df), X_expected)
    assert_array_equal(oh.fit(X_df).transform(X_df), X_expected)


def test_one_hot_encoder_feature_names():
    enc = OneHotEncoder()
    X = [
        ["Male", 1, "girl", 2, 3],
        ["Female", 41, "girl", 1, 10],
        ["Male", 51, "boy", 12, 3],
        ["Male", 91, "girl", 21, 30],
    ]

    enc.fit(X)
    feature_names = enc.get_feature_names_out()

    assert_array_equal(
        [
            "x0_Female",
            "x0_Male",
            "x1_1",
            "x1_41",
            "x1_51",
            "x1_91",
            "x2_boy",
            "x2_girl",
            "x3_1",
            "x3_2",
            "x3_12",
            "x3_21",
            "x4_3",
            "x4_10",
            "x4_30",
        ],
        feature_names,
    )

    feature_names2 = enc.get_feature_names_out(["one", "two", "three", "four", "five"])

    assert_array_equal(
        [
            "one_Female",
            "one_Male",
            "two_1",
            "two_41",
            "two_51",
            "two_91",
            "three_boy",
            "three_girl",
            "four_1",
            "four_2",
            "four_12",
            "four_21",
            "five_3",
            "five_10",
            "five_30",
        ],
        feature_names2,
    )

    with pytest.raises(ValueError, match="input_features should have length"):
        enc.get_feature_names_out(["one", "two"])


def test_one_hot_encoder_feature_names_unicode():
    enc = OneHotEncoder()
    X = np.array([["c❤t1", "dat2"]], dtype=object).T
    enc.fit(X)
    feature_names = enc.get_feature_names_out()
    assert_array_equal(["x0_c❤t1", "x0_dat2"], feature_names)
    feature_names = enc.get_feature_names_out(input_features=["n👍me"])
    assert_array_equal(["n👍me_c❤t1", "n👍me_dat2"], feature_names)


def test_one_hot_encoder_custom_feature_name_combiner():
    """Check the behaviour of `feature_name_combiner` as a callable."""

    def name_combiner(feature, category):
        return feature + "_" + repr(category)

    enc = OneHotEncoder(feature_name_combiner=name_combiner)
    X = np.array([["None", None]], dtype=object).T
    enc.fit(X)
    feature_names = enc.get_feature_names_out()
    assert_array_equal(["x0_'None'", "x0_None"], feature_names)
    feature_names = enc.get_feature_names_out(input_features=["a"])
    assert_array_equal(["a_'None'", "a_None"], feature_names)

    def wrong_combiner(feature, category):
        # we should be returning a Python string
        return 0

    enc = OneHotEncoder(feature_name_combiner=wrong_combiner).fit(X)
    err_msg = (
        "When `feature_name_combiner` is a callable, it should return a Python string."
    )
    with pytest.raises(TypeError, match=err_msg):
        enc.get_feature_names_out()


def test_one_hot_encoder_set_params():
    X = np.array([[1, 2]]).T
    oh = OneHotEncoder()
    # set params on not yet fitted object
    oh.set_params(categories=[[0, 1, 2, 3]])
    assert oh.get_params()["categories"] == [[0, 1, 2, 3]]
    assert oh.fit_transform(X).toarray().shape == (2, 4)
    # set params on already fitted object
    oh.set_params(categories=[[0, 1, 2, 3, 4]])
    assert oh.fit_transform(X).toarray().shape == (2, 5)


def check_categorical_onehot(X):
    enc = OneHotEncoder(categories="auto")
    Xtr1 = enc.fit_transform(X)

    enc = OneHotEncoder(categories="auto", sparse_output=False)
    Xtr2 = enc.fit_transform(X)

    assert_allclose(Xtr1.toarray(), Xtr2)

    assert sparse.issparse(Xtr1) and Xtr1.format == "csr"
    return Xtr1.toarray()


@pytest.mark.parametrize(
    "X",
    [
        [["def", 1, 55], ["abc", 2, 55]],
        np.array([[10, 1, 55], [5, 2, 55]]),
        np.array([["b", "A", "cat"], ["a", "B", "cat"]], dtype=object),
        np.array([["b", 1, "cat"], ["a", np.nan, "cat"]], dtype=object),
        np.array([["b", 1, "cat"], ["a", float("nan"), "cat"]], dtype=object),
        np.array([[None, 1, "cat"], ["a", 2, "cat"]], dtype=object),
        np.array([[None, 1, None], ["a", np.nan, None]], dtype=object),
        np.array([[None, 1, None], ["a", float("nan"), None]], dtype=object),
    ],
    ids=[
        "mixed",
        "numeric",
        "object",
        "mixed-nan",
        "mixed-float-nan",
        "mixed-None",
        "mixed-None-nan",
        "mixed-None-float-nan",
    ],
)
def test_one_hot_encoder(X):
    Xtr = check_categorical_onehot(np.array(X)[:, [0]])
    assert_allclose(Xtr, [[0, 1], [1, 0]])

    Xtr = check_categorical_onehot(np.array(X)[:, [0, 1]])
    assert_allclose(Xtr, [[0, 1, 1, 0], [1, 0, 0, 1]])

    Xtr = OneHotEncoder(categories="auto").fit_transform(X)
    assert_allclose(Xtr.toarray(), [[0, 1, 1, 0, 1], [1, 0, 0, 1, 1]])


@pytest.mark.parametrize("handle_unknown", ["ignore", "infrequent_if_exist"])
@pytest.mark.parametrize("sparse_", [False, True])
@pytest.mark.parametrize("drop", [None, "first"])
def test_one_hot_encoder_inverse(handle_unknown, sparse_, drop):
    X = [["abc", 2, 55], ["def", 1, 55], ["abc", 3, 55]]
    enc = OneHotEncoder(sparse_output=sparse_, drop=drop)
    X_tr = enc.fit_transform(X)
    exp = np.array(X, dtype=object)
    assert_array_equal(enc.inverse_transform(X_tr), exp)

    X = [[2, 55], [1, 55], [3, 55]]
    enc = OneHotEncoder(sparse_output=sparse_, categories="auto", drop=drop)
    X_tr = enc.fit_transform(X)
    exp = np.array(X)
    assert_array_equal(enc.inverse_transform(X_tr), exp)

    if drop is None:
        # with unknown categories
        # drop is incompatible with handle_unknown=ignore
        X = [["abc", 2, 55], ["def", 1, 55], ["abc", 3, 55]]
        enc = OneHotEncoder(
            sparse_output=sparse_,
            handle_unknown=handle_unknown,
            categories=[["abc", "def"], [1, 2], [54, 55, 56]],
        )
        X_tr = enc.fit_transform(X)
        exp = np.array(X, dtype=object)
        exp[2, 1] = None
        assert_array_equal(enc.inverse_transform(X_tr), exp)

        # with an otherwise numerical output, still object if unknown
        X = [[2, 55], [1, 55], [3, 55]]
        enc = OneHotEncoder(
            sparse_output=sparse_,
            categories=[[1, 2], [54, 56]],
            handle_unknown=handle_unknown,
        )
        X_tr = enc.fit_transform(X)
        exp = np.array(X, dtype=object)
        exp[2, 0] = None
        exp[:, 1] = None
        assert_array_equal(enc.inverse_transform(X_tr), exp)

    # incorrect shape raises
    X_tr = np.array([[0, 1, 1], [1, 0, 1]])
    msg = re.escape("Shape of the passed X data is not correct")
    with pytest.raises(ValueError, match=msg):
        enc.inverse_transform(X_tr)


@pytest.mark.parametrize("sparse_", [False, True])
@pytest.mark.parametrize(
    "X, X_trans",
    [
        ([[2, 55], [1, 55], [2, 55]], [[0, 1, 1], [0, 0, 0], [0, 1, 1]]),
        (
            [["one", "a"], ["two", "a"], ["three", "b"], ["two", "a"]],
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 1, 0, 0, 0]],
        ),
    ],
)
def test_one_hot_encoder_inverse_transform_raise_error_with_unknown(
    X, X_trans, sparse_
):
    """Check that `inverse_transform` raise an error with unknown samples, no
    dropped feature, and `handle_unknow="error`.
    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/14934
    """
    enc = OneHotEncoder(sparse_output=sparse_).fit(X)
    msg = (
        r"Samples \[(\d )*\d\] can not be inverted when drop=None and "
        r"handle_unknown='error' because they contain all zeros"
    )

    if sparse_:
        # emulate sparse data transform by a one-hot encoder sparse.
        X_trans = _convert_container(X_trans, "sparse")
    with pytest.raises(ValueError, match=msg):
        enc.inverse_transform(X_trans)


def test_one_hot_encoder_inverse_if_binary():
    X = np.array([["Male", 1], ["Female", 3], ["Female", 2]], dtype=object)
    ohe = OneHotEncoder(drop="if_binary", sparse_output=False)
    X_tr = ohe.fit_transform(X)
    assert_array_equal(ohe.inverse_transform(X_tr), X)


@pytest.mark.parametrize("drop", ["if_binary", "first", None])
@pytest.mark.parametrize("reset_drop", ["if_binary", "first", None])
def test_one_hot_encoder_drop_reset(drop, reset_drop):
    # check that resetting drop option without refitting does not throw an error
    X = np.array([["Male", 1], ["Female", 3], ["Female", 2]], dtype=object)
    ohe = OneHotEncoder(drop=drop, sparse_output=False)
    ohe.fit(X)
    X_tr = ohe.transform(X)
    feature_names = ohe.get_feature_names_out()
    ohe.set_params(drop=reset_drop)
    assert_array_equal(ohe.inverse_transform(X_tr), X)
    assert_allclose(ohe.transform(X), X_tr)
    assert_array_equal(ohe.get_feature_names_out(), feature_names)


@pytest.mark.parametrize("method", ["fit", "fit_transform"])
@pytest.mark.parametrize("X", [[1, 2], np.array([3.0, 4.0])])
def test_X_is_not_1D(X, method):
    oh = OneHotEncoder()

    msg = "Expected 2D array, got 1D array instead"
    with pytest.raises(ValueError, match=msg):
        getattr(oh, method)(X)


@pytest.mark.parametrize("method", ["fit", "fit_transform"])
def test_X_is_not_1D_pandas(method):
    pd = pytest.importorskip("pandas")
    X = pd.Series([6, 3, 4, 6])
    oh = OneHotEncoder()

    msg = f"Expected a 2-dimensional container but got {type(X)} instead."
    with pytest.raises(ValueError, match=msg):
        getattr(oh, method)(X)


@pytest.mark.parametrize(
    "X, cat_exp, cat_dtype",
    [
        ([["abc", 55], ["def", 55]], [["abc", "def"], [55]], np.object_),
        (np.array([[1, 2], [3, 2]]), [[1, 3], [2]], np.integer),
        (
            np.array([["A", "cat"], ["B", "cat"]], dtype=object),
            [["A", "B"], ["cat"]],
            np.object_,
        ),
        (np.array([["A", "cat"], ["B", "cat"]]), [["A", "B"], ["cat"]], np.str_),
        (np.array([[1, 2], [np.nan, 2]]), [[1, np.nan], [2]], np.float64),
        (
            np.array([["A", np.nan], [None, np.nan]], dtype=object),
            [["A", None], [np.nan]],
            np.object_,
        ),
        (
            np.array([["A", float("nan")], [None, float("nan")]], dtype=object),
            [["A", None], [float("nan")]],
            np.object_,
        ),
    ],
    ids=[
        "mixed",
        "numeric",
        "object",
        "string",
        "missing-float",
        "missing-np.nan-object",
        "missing-float-nan-object",
    ],
)
def test_one_hot_encoder_categories(X, cat_exp, cat_dtype):
    # order of categories should not depend on order of samples
    for Xi in [X, X[::-1]]:
        enc = OneHotEncoder(categories="auto")
        enc.fit(Xi)
        # assert enc.categories == 'auto'
        assert isinstance(enc.categories_, list)
        for res, exp in zip(enc.categories_, cat_exp):
            res_list = res.tolist()
            if is_scalar_nan(exp[-1]):
                assert is_scalar_nan(res_list[-1])
                assert res_list[:-1] == exp[:-1]
            else:
                assert res.tolist() == exp
            assert np.issubdtype(res.dtype, cat_dtype)


@pytest.mark.parametrize("handle_unknown", ["ignore", "infrequent_if_exist"])
@pytest.mark.parametrize(
    "X, X2, cats, cat_dtype",
    [
        (
            np.array([["a", "b"]], dtype=object).T,
            np.array([["a", "d"]], dtype=object).T,
            [["a", "b", "c"]],
            np.object_,
        ),
        (
            np.array([[1, 2]], dtype="int64").T,
            np.array([[1, 4]], dtype="int64").T,
            [[1, 2, 3]],
            np.int64,
        ),
        (
            np.array([["a", "b"]], dtype=object).T,
            np.array([["a", "d"]], dtype=object).T,
            [np.array(["a", "b", "c"])],
            np.object_,
        ),
        (
            np.array([[None, "a"]], dtype=object).T,
            np.array([[None, "b"]], dtype=object).T,
            [[None, "a", "z"]],
            object,
        ),
        (
            np.array([["a", "b"]], dtype=object).T,
            np.array([["a", np.nan]], dtype=object).T,
            [["a", "b", "z"]],
            object,
        ),
        (
            np.array([["a", None]], dtype=object).T,
            np.array([["a", np.nan]], dtype=object).T,
            [["a", None, "z"]],
            object,
        ),
    ],
    ids=[
        "object",
        "numeric",
        "object-string",
        "object-string-none",
        "object-string-nan",
        "object-None-and-nan",
    ],
)
def test_one_hot_encoder_specified_categories(X, X2, cats, cat_dtype, handle_unknown):
    enc = OneHotEncoder(categories=cats)
    exp = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert_array_equal(enc.fit_transform(X).toarray(), exp)
    assert list(enc.categories[0]) == list(cats[0])
    assert enc.categories_[0].tolist() == list(cats[0])
    # manually specified categories should have same dtype as
    # the data when coerced from lists
    assert enc.categories_[0].dtype == cat_dtype

    # when specifying categories manually, unknown categories should already
    # raise when fitting
    enc = OneHotEncoder(categories=cats)
    with pytest.raises(ValueError, match="Found unknown categories"):
        enc.fit(X2)
    enc = OneHotEncoder(categories=cats, handle_unknown=handle_unknown)
    exp = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    assert_array_equal(enc.fit(X2).transform(X2).toarray(), exp)


def test_one_hot_encoder_unsorted_categories():
    X = np.array([["a", "b"]], dtype=object).T

    enc = OneHotEncoder(categories=[["b", "a", "c"]])
    exp = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    assert_array_equal(enc.fit(X).transform(X).toarray(), exp)
    assert_array_equal(enc.fit_transform(X).toarray(), exp)
    assert enc.categories_[0].tolist() == ["b", "a", "c"]
    assert np.issubdtype(enc.categories_[0].dtype, np.object_)

    # unsorted passed categories still raise for numerical values
    X = np.array([[1, 2]]).T
    enc = OneHotEncoder(categories=[[2, 1, 3]])
    msg = "Unsorted categories are not supported"
    with pytest.raises(ValueError, match=msg):
        enc.fit_transform(X)


@pytest.mark.parametrize("Encoder", [OneHotEncoder, OrdinalEncoder])
def test_encoder_nan_ending_specified_categories(Encoder):
    """Test encoder for specified categories that nan is at the end.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/27088
    """
    cats = [np.array([0, np.nan, 1])]
    enc = Encoder(categories=cats)
    X = np.array([[0, 1]], dtype=object).T
    with pytest.raises(ValueError, match="Nan should be the last element"):
        enc.fit(X)


def test_one_hot_encoder_specified_categories_mixed_columns():
    # multiple columns
    X = np.array([["a", "b"], [0, 2]], dtype=object).T
    enc = OneHotEncoder(categories=[["a", "b", "c"], [0, 1, 2]])
    exp = np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 1.0]])
    assert_array_equal(enc.fit_transform(X).toarray(), exp)
    assert enc.categories_[0].tolist() == ["a", "b", "c"]
    assert np.issubdtype(enc.categories_[0].dtype, np.object_)
    assert enc.categories_[1].tolist() == [0, 1, 2]
    # integer categories but from object dtype data
    assert np.issubdtype(enc.categories_[1].dtype, np.object_)


def test_one_hot_encoder_pandas():
    pd = pytest.importorskip("pandas")

    X_df = pd.DataFrame({"A": ["a", "b"], "B": [1, 2]})

    Xtr = check_categorical_onehot(X_df)
    assert_allclose(Xtr, [[1, 0, 1, 0], [0, 1, 0, 1]])


@pytest.mark.parametrize(
    "drop, expected_names",
    [
        ("first", ["x0_c", "x2_b"]),
        ("if_binary", ["x0_c", "x1_2", "x2_b"]),
        (["c", 2, "b"], ["x0_b", "x2_a"]),
    ],
    ids=["first", "binary", "manual"],
)
def test_one_hot_encoder_feature_names_drop(drop, expected_names):
    X = [["c", 2, "a"], ["b", 2, "b"]]

    ohe = OneHotEncoder(drop=drop)
    ohe.fit(X)
    feature_names = ohe.get_feature_names_out()
    assert_array_equal(expected_names, feature_names)


def test_one_hot_encoder_drop_equals_if_binary():
    # Canonical case
    X = [[10, "yes"], [20, "no"], [30, "yes"]]
    expected = np.array(
        [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]
    )
    expected_drop_idx = np.array([None, 0])

    ohe = OneHotEncoder(drop="if_binary", sparse_output=False)
    result = ohe.fit_transform(X)
    assert_array_equal(ohe.drop_idx_, expected_drop_idx)
    assert_allclose(result, expected)

    # with only one cat, the behaviour is equivalent to drop=None
    X = [["true", "a"], ["false", "a"], ["false", "a"]]
    expected = np.array([[1.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    expected_drop_idx = np.array([0, None])

    ohe = OneHotEncoder(drop="if_binary", sparse_output=False)
    result = ohe.fit_transform(X)
    assert_array_equal(ohe.drop_idx_, expected_drop_idx)
    assert_allclose(result, expected)


@pytest.mark.parametrize(
    "X",
    [
        [["abc", 2, 55], ["def", 1, 55]],
        np.array([[10, 2, 55], [20, 1, 55]]),
        np.array([["a", "B", "cat"], ["b", "A", "cat"]], dtype=object),
    ],
    ids=["mixed", "numeric", "object"],
)
def test_ordinal_encoder(X):
    enc = OrdinalEncoder()
    exp = np.array([[0, 1, 0], [1, 0, 0]], dtype="int64")
    assert_array_equal(enc.fit_transform(X), exp.astype("float64"))
    enc = OrdinalEncoder(dtype="int64")
    assert_array_equal(enc.fit_transform(X), exp)


@pytest.mark.parametrize(
    "X, X2, cats, cat_dtype",
    [
        (
            np.array([["a", "b"]], dtype=object).T,
            np.array([["a", "d"]], dtype=object).T,
            [["a", "b", "c"]],
            np.object_,
        ),
        (
            np.array([[1, 2]], dtype="int64").T,
            np.array([[1, 4]], dtype="int64").T,
            [[1, 2, 3]],
            np.int64,
        ),
        (
            np.array([["a", "b"]], dtype=object).T,
            np.array([["a", "d"]], dtype=object).T,
            [np.array(["a", "b", "c"])],
            np.object_,
        ),
    ],
    ids=["object", "numeric", "object-string-cat"],
)
def test_ordinal_encoder_specified_categories(X, X2, cats, cat_dtype):
    enc = OrdinalEncoder(categories=cats)
    exp = np.array([[0.0], [1.0]])
    assert_array_equal(enc.fit_transform(X), exp)
    assert list(enc.categories[0]) == list(cats[0])
    assert enc.categories_[0].tolist() == list(cats[0])
    # manually specified categories should have same dtype as
    # the data when coerced from lists
    assert enc.categories_[0].dtype == cat_dtype

    # when specifying categories manually, unknown categories should already
    # raise when fitting
    enc = OrdinalEncoder(categories=cats)
    with pytest.raises(ValueError, match="Found unknown categories"):
        enc.fit(X2)


def test_ordinal_encoder_inverse():
    X = [["abc", 2, 55], ["def", 1, 55]]
    enc = OrdinalEncoder()
    X_tr = enc.fit_transform(X)
    exp = np.array(X, dtype=object)
    assert_array_equal(enc.inverse_transform(X_tr), exp)

    # incorrect shape raises
    X_tr = np.array([[0, 1, 1, 2], [1, 0, 1, 0]])
    msg = re.escape("Shape of the passed X data is not correct")
    with pytest.raises(ValueError, match=msg):
        enc.inverse_transform(X_tr)


def test_ordinal_encoder_handle_unknowns_string():
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-2)
    X_fit = np.array([["a", "x"], ["b", "y"], ["c", "z"]], dtype=object)
    X_trans = np.array([["c", "xy"], ["bla", "y"], ["a", "x"]], dtype=object)
    enc.fit(X_fit)

    X_trans_enc = enc.transform(X_trans)
    exp = np.array([[2, -2], [-2, 1], [0, 0]], dtype="int64")
    assert_array_equal(X_trans_enc, exp)

    X_trans_inv = enc.inverse_transform(X_trans_enc)
    inv_exp = np.array([["c", None], [None, "y"], ["a", "x"]], dtype=object)
    assert_array_equal(X_trans_inv, inv_exp)


@pytest.mark.parametrize("dtype", [float, int])
def test_ordinal_encoder_handle_unknowns_numeric(dtype):
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-999)
    X_fit = np.array([[1, 7], [2, 8], [3, 9]], dtype=dtype)
    X_trans = np.array([[3, 12], [23, 8], [1, 7]], dtype=dtype)
    enc.fit(X_fit)

    X_trans_enc = enc.transform(X_trans)
    exp = np.array([[2, -999], [-999, 1], [0, 0]], dtype="int64")
    assert_array_equal(X_trans_enc, exp)

    X_trans_inv = enc.inverse_transform(X_trans_enc)
    inv_exp = np.array([[3, None], [None, 8], [1, 7]], dtype=object)
    assert_array_equal(X_trans_inv, inv_exp)


def test_ordinal_encoder_handle_unknowns_nan():
    # Make sure unknown_value=np.nan properly works

    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)

    X_fit = np.array([[1], [2], [3]])
    enc.fit(X_fit)
    X_trans = enc.transform([[1], [2], [4]])
    assert_array_equal(X_trans, [[0], [1], [np.nan]])


def test_ordinal_encoder_handle_unknowns_nan_non_float_dtype():
    # Make sure an error is raised when unknown_value=np.nan and the dtype
    # isn't a float dtype
    enc = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=np.nan, dtype=int
    )

    X_fit = np.array([[1], [2], [3]])
    with pytest.raises(ValueError, match="dtype parameter should be a float dtype"):
        enc.fit(X_fit)


def test_ordinal_encoder_raise_categories_shape():
    X = np.array([["Low", "Medium", "High", "Medium", "Low"]], dtype=object).T
    cats = ["Low", "Medium", "High"]
    enc = OrdinalEncoder(categories=cats)
    msg = "Shape mismatch: if categories is an array,"

    with pytest.raises(ValueError, match=msg):
        enc.fit(X)


def test_encoder_dtypes():
    # check that dtypes are preserved when determining categories
    enc = OneHotEncoder(categories="auto")
    exp = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]], dtype="float64")

    for X in [
        np.array([[1, 2], [3, 4]], dtype="int64"),
        np.array([[1, 2], [3, 4]], dtype="float64"),
        np.array([["a", "b"], ["c", "d"]]),  # str dtype
        np.array([[b"a", b"b"], [b"c", b"d"]]),  # bytes dtype
        np.array([[1, "a"], [3, "b"]], dtype="object"),
    ]:
        enc.fit(X)
        assert all([enc.categories_[i].dtype == X.dtype for i in range(2)])
        assert_array_equal(enc.transform(X).toarray(), exp)

    X = [[1, 2], [3, 4]]
    enc.fit(X)
    assert all([np.issubdtype(enc.categories_[i].dtype, np.integer) for i in range(2)])
    assert_array_equal(enc.transform(X).toarray(), exp)

    X = [[1, "a"], [3, "b"]]
    enc.fit(X)
    assert all([enc.categories_[i].dtype == "object" for i in range(2)])
    assert_array_equal(enc.transform(X).toarray(), exp)


def test_encoder_dtypes_pandas():
    # check dtype (similar to test_categorical_encoder_dtypes for dataframes)
    pd = pytest.importorskip("pandas")

    enc = OneHotEncoder(categories="auto")
    exp = np.array(
        [[1.0, 0.0, 1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]],
        dtype="float64",
    )

    X = pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]}, dtype="int64")
    enc.fit(X)
    assert all([enc.categories_[i].dtype == "int64" for i in range(2)])
    assert_array_equal(enc.transform(X).toarray(), exp)

    X = pd.DataFrame({"A": [1, 2], "B": ["a", "b"], "C": [3.0, 4.0]})
    X_type = [X["A"].dtype, X["B"].dtype, X["C"].dtype]
    enc.fit(X)
    assert all([enc.categories_[i].dtype == X_type[i] for i in range(3)])
    assert_array_equal(enc.transform(X).toarray(), exp)


def test_one_hot_encoder_warning():
    enc = OneHotEncoder()
    X = [["Male", 1], ["Female", 3]]
    np.testing.assert_no_warnings(enc.fit_transform, X)


@pytest.mark.parametrize("missing_value", [np.nan, None, float("nan")])
def test_one_hot_encoder_drop_manual(missing_value):
    cats_to_drop = ["def", 12, 3, 56, missing_value]
    enc = OneHotEncoder(drop=cats_to_drop)
    X = [
        ["abc", 12, 2, 55, "a"],
        ["def", 12, 1, 55, "a"],
        ["def", 12, 3, 56, missing_value],
    ]
    trans = enc.fit_transform(X).toarray()
    exp = [[1, 0, 1, 1, 1], [0, 1, 0, 1, 1], [0, 0, 0, 0, 0]]
    assert_array_equal(trans, exp)
    assert enc.drop is cats_to_drop

    dropped_cats = [
        cat[feature] for cat, feature in zip(enc.categories_, enc.drop_idx_)
    ]
    X_inv_trans = enc.inverse_transform(trans)
    X_array = np.array(X, dtype=object)

    # last value is np.nan
    if is_scalar_nan(cats_to_drop[-1]):
        assert_array_equal(dropped_cats[:-1], cats_to_drop[:-1])
        assert is_scalar_nan(dropped_cats[-1])
        assert is_scalar_nan(cats_to_drop[-1])
        # do not include the last column which includes missing values
        assert_array_equal(X_array[:, :-1], X_inv_trans[:, :-1])

        # check last column is the missing value
        assert_array_equal(X_array[-1, :-1], X_inv_trans[-1, :-1])
        assert is_scalar_nan(X_array[-1, -1])
        assert is_scalar_nan(X_inv_trans[-1, -1])
    else:
        assert_array_equal(dropped_cats, cats_to_drop)
        assert_array_equal(X_array, X_inv_trans)


@pytest.mark.parametrize("drop", [["abc", 3], ["abc", 3, 41, "a"]])
def test_invalid_drop_length(drop):
    enc = OneHotEncoder(drop=drop)
    err_msg = "`drop` should have length equal to the number"
    with pytest.raises(ValueError, match=err_msg):
        enc.fit([["abc", 2, 55], ["def", 1, 55], ["def", 3, 59]])


@pytest.mark.parametrize("density", [True, False], ids=["sparse", "dense"])
@pytest.mark.parametrize("drop", ["first", ["a", 2, "b"]], ids=["first", "manual"])
def test_categories(density, drop):
    ohe_base = OneHotEncoder(sparse_output=density)
    ohe_test = OneHotEncoder(sparse_output=density, drop=drop)
    X = [["c", 1, "a"], ["a", 2, "b"]]
    ohe_base.fit(X)
    ohe_test.fit(X)
    assert_array_equal(ohe_base.categories_, ohe_test.categories_)
    if drop == "first":
        assert_array_equal(ohe_test.drop_idx_, 0)
    else:
        for drop_cat, drop_idx, cat_list in zip(
            drop, ohe_test.drop_idx_, ohe_test.categories_
        ):
            assert cat_list[int(drop_idx)] == drop_cat
    assert isinstance(ohe_test.drop_idx_, np.ndarray)
    assert ohe_test.drop_idx_.dtype == object


@pytest.mark.parametrize("Encoder", [OneHotEncoder, OrdinalEncoder])
def test_encoders_has_categorical_tags(Encoder):
    assert "categorical" in Encoder()._get_tags()["X_types"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_categories": 2},
        {"min_frequency": 11},
        {"min_frequency": 0.29},
        {"max_categories": 2, "min_frequency": 6},
        {"max_categories": 4, "min_frequency": 12},
    ],
)
@pytest.mark.parametrize("categories", ["auto", [["a", "b", "c", "d"]]])
def test_ohe_infrequent_two_levels(kwargs, categories):
    """Test that different parameters for combine 'a', 'c', and 'd' into
    the infrequent category works as expected."""

    X_train = np.array([["a"] * 5 + ["b"] * 20 + ["c"] * 10 + ["d"] * 3]).T
    ohe = OneHotEncoder(
        categories=categories,
        handle_unknown="infrequent_if_exist",
        sparse_output=False,
        **kwargs,
    ).fit(X_train)
    assert_array_equal(ohe.infrequent_categories_, [["a", "c", "d"]])

    X_test = [["b"], ["a"], ["c"], ["d"], ["e"]]
    expected = np.array([[1, 0], [0, 1], [0, 1], [0, 1], [0, 1]])

    X_trans = ohe.transform(X_test)
    assert_allclose(expected, X_trans)

    expected_inv = [[col] for col in ["b"] + ["infrequent_sklearn"] * 4]
    X_inv = ohe.inverse_transform(X_trans)
    assert_array_equal(expected_inv, X_inv)

    feature_names = ohe.get_feature_names_out()
    assert_array_equal(["x0_b", "x0_infrequent_sklearn"], feature_names)


@pytest.mark.parametrize("drop", ["if_binary", "first", ["b"]])
def test_ohe_infrequent_two_levels_drop_frequent(drop):
    """Test two levels and dropping the frequent category."""

    X_train = np.array([["a"] * 5 + ["b"] * 20 + ["c"] * 10 + ["d"] * 3]).T
    ohe = OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        sparse_output=False,
        max_categories=2,
        drop=drop,
    ).fit(X_train)
    assert ohe.categories_[0][ohe.drop_idx_[0]] == "b"

    X_test = np.array([["b"], ["c"]])
    X_trans = ohe.transform(X_test)
    assert_allclose([[0], [1]], X_trans)

    feature_names = ohe.get_feature_names_out()
    assert_array_equal(["x0_infrequent_sklearn"], feature_names)

    X_inverse = ohe.inverse_transform(X_trans)
    assert_array_equal([["b"], ["infrequent_sklearn"]], X_inverse)


@pytest.mark.parametrize("drop", [["a"], ["d"]])
def test_ohe_infrequent_two_levels_drop_infrequent_errors(drop):
    """Test two levels and dropping any infrequent category removes the
    whole infrequent category."""

    X_train = np.array([["a"] * 5 + ["b"] * 20 + ["c"] * 10 + ["d"] * 3]).T
    ohe = OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        sparse_output=False,
        max_categories=2,
        drop=drop,
    )

    msg = f"Unable to drop category {drop[0]!r} from feature 0 because it is infrequent"
    with pytest.raises(ValueError, match=msg):
        ohe.fit(X_train)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_categories": 3},
        {"min_frequency": 6},
        {"min_frequency": 9},
        {"min_frequency": 0.24},
        {"min_frequency": 0.16},
        {"max_categories": 3, "min_frequency": 8},
        {"max_categories": 4, "min_frequency": 6},
    ],
)
def test_ohe_infrequent_three_levels(kwargs):
    """Test that different parameters for combing 'a', and 'd' into
    the infrequent category works as expected."""

    X_train = np.array([["a"] * 5 + ["b"] * 20 + ["c"] * 10 + ["d"] * 3]).T
    ohe = OneHotEncoder(
        handle_unknown="infrequent_if_exist", sparse_output=False, **kwargs
    ).fit(X_train)
    assert_array_equal(ohe.infrequent_categories_, [["a", "d"]])

    X_test = [["b"], ["a"], ["c"], ["d"], ["e"]]
    expected = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1], [0, 0, 1]])

    X_trans = ohe.transform(X_test)
    assert_allclose(expected, X_trans)

    expected_inv = [
        ["b"],
        ["infrequent_sklearn"],
        ["c"],
        ["infrequent_sklearn"],
        ["infrequent_sklearn"],
    ]
    X_inv = ohe.inverse_transform(X_trans)
    assert_array_equal(expected_inv, X_inv)

    feature_names = ohe.get_feature_names_out()
    assert_array_equal(["x0_b", "x0_c", "x0_infrequent_sklearn"], feature_names)


@pytest.mark.parametrize("drop", ["first", ["b"]])
def test_ohe_infrequent_three_levels_drop_frequent(drop):
    """Test three levels and dropping the frequent category."""

    X_train = np.array([["a"] * 5 + ["b"] * 20 + ["c"] * 10 + ["d"] * 3]).T
    ohe = OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        sparse_output=False,
        max_categories=3,
        drop=drop,
    ).fit(X_train)

    X_test = np.array([["b"], ["c"], ["d"]])
    assert_allclose([[0, 0], [1, 0], [0, 1]], ohe.transform(X_test))

    # Check handle_unknown="ignore"
    ohe.set_params(handle_unknown="ignore").fit(X_train)
    msg = "Found unknown categories"
    with pytest.warns(UserWarning, match=msg):
        X_trans = ohe.transform([["b"], ["e"]])

    assert_allclose([[0, 0], [0, 0]], X_trans)


@pytest.mark.parametrize("drop", [["a"], ["d"]])
def test_ohe_infrequent_three_levels_drop_infrequent_errors(drop):
    """Test three levels and dropping the infrequent category."""
    X_train = np.array([["a"] * 5 + ["b"] * 20 + ["c"] * 10 + ["d"] * 3]).T
    ohe = OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        sparse_output=False,
        max_categories=3,
        drop=drop,
    )

    msg = f"Unable to drop category {drop[0]!r} from feature 0 because it is infrequent"
    with pytest.raises(ValueError, match=msg):
        ohe.fit(X_train)


def test_ohe_infrequent_handle_unknown_error():
    """Test that different parameters for combining 'a', and 'd' into
    the infrequent category works as expected."""

    X_train = np.array([["a"] * 5 + ["b"] * 20 + ["c"] * 10 + ["d"] * 3]).T
    ohe = OneHotEncoder(
        handle_unknown="error", sparse_output=False, max_categories=3
    ).fit(X_train)
    assert_array_equal(ohe.infrequent_categories_, [["a", "d"]])

    # all categories are known
    X_test = [["b"], ["a"], ["c"], ["d"]]
    expected = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1]])

    X_trans = ohe.transform(X_test)
    assert_allclose(expected, X_trans)

    # 'bad' is not known and will error
    X_test = [["bad"]]
    msg = r"Found unknown categories \['bad'\] in column 0"
    with pytest.raises(ValueError, match=msg):
        ohe.transform(X_test)


@pytest.mark.parametrize(
    "kwargs", [{"max_categories": 3, "min_frequency": 1}, {"min_frequency": 4}]
)
def test_ohe_infrequent_two_levels_user_cats_one_frequent(kwargs):
    """'a' is the only frequent category, all other categories are infrequent."""

    X_train = np.array([["a"] * 5 + ["e"] * 30], dtype=object).T
    ohe = OneHotEncoder(
        categories=[["c", "d", "a", "b"]],
        sparse_output=False,
        handle_unknown="infrequent_if_exist",
        **kwargs,
    ).fit(X_train)

    X_test = [["a"], ["b"], ["c"], ["d"], ["e"]]
    expected = np.array([[1, 0], [0, 1], [0, 1], [0, 1], [0, 1]])

    X_trans = ohe.transform(X_test)
    assert_allclose(expected, X_trans)

    # 'a' is dropped
    drops = ["first", "if_binary", ["a"]]
    X_test = [["a"], ["c"]]
    for drop in drops:
        ohe.set_params(drop=drop).fit(X_train)
        assert_allclose([[0], [1]], ohe.transform(X_test))


def test_ohe_infrequent_two_levels_user_cats():
    """Test that the order of the categories provided by a user is respected."""
    X_train = np.array(
        [["a"] * 5 + ["b"] * 20 + ["c"] * 10 + ["d"] * 3], dtype=object
    ).T
    ohe = OneHotEncoder(
        categories=[["c", "d", "a", "b"]],
        sparse_output=False,
        handle_unknown="infrequent_if_exist",
        max_categories=2,
    ).fit(X_train)

    assert_array_equal(ohe.infrequent_categories_, [["c", "d", "a"]])

    X_test = [["b"], ["a"], ["c"], ["d"], ["e"]]
    expected = np.array([[1, 0], [0, 1], [0, 1], [0, 1], [0, 1]])

    X_trans = ohe.transform(X_test)
    assert_allclose(expected, X_trans)

    # 'infrequent' is used to denote the infrequent categories for
    # `inverse_transform`
    expected_inv = [[col] for col in ["b"] + ["infrequent_sklearn"] * 4]
    X_inv = ohe.inverse_transform(X_trans)
    assert_array_equal(expected_inv, X_inv)


def test_ohe_infrequent_three_levels_user_cats():
    """Test that the order of the categories provided by a user is respected.
    In this case 'c' is encoded as the first category and 'b' is encoded
    as the second one."""

    X_train = np.array(
        [["a"] * 5 + ["b"] * 20 + ["c"] * 10 + ["d"] * 3], dtype=object
    ).T
    ohe = OneHotEncoder(
        categories=[["c", "d", "b", "a"]],
        sparse_output=False,
        handle_unknown="infrequent_if_exist",
        max_categories=3,
    ).fit(X_train)

    assert_array_equal(ohe.infrequent_categories_, [["d", "a"]])

    X_test = [["b"], ["a"], ["c"], ["d"], ["e"]]
    expected = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 0, 1], [0, 0, 1]])

    X_trans = ohe.transform(X_test)
    assert_allclose(expected, X_trans)

    # 'infrequent' is used to denote the infrequent categories for
    # `inverse_transform`
    expected_inv = [
        ["b"],
        ["infrequent_sklearn"],
        ["c"],
        ["infrequent_sklearn"],
        ["infrequent_sklearn"],
    ]
    X_inv = ohe.inverse_transform(X_trans)
    assert_array_equal(expected_inv, X_inv)


def test_ohe_infrequent_mixed():
    """Test infrequent categories where feature 0 has infrequent categories,
    and feature 1 does not."""

    # X[:, 0] 1 and 2 are infrequent
    # X[:, 1] nothing is infrequent
    X = np.c_[[0, 1, 3, 3, 3, 3, 2, 0, 3], [0, 0, 0, 0, 1, 1, 1, 1, 1]]

    ohe = OneHotEncoder(max_categories=3, drop="if_binary", sparse_output=False)
    ohe.fit(X)

    X_test = [[3, 0], [1, 1]]
    X_trans = ohe.transform(X_test)

    # feature 1 is binary so it drops a category 0
    assert_allclose(X_trans, [[0, 1, 0, 0], [0, 0, 1, 1]])


def test_ohe_infrequent_multiple_categories():
    """Test infrequent categories with feature matrix with 3 features."""

    X = np.c_[
        [0, 1, 3, 3, 3, 3, 2, 0, 3],
        [0, 0, 5, 1, 1, 10, 5, 5, 0],
        [1, 0, 1, 0, 1, 0, 1, 0, 1],
    ]

    ohe = OneHotEncoder(
        categories="auto", max_categories=3, handle_unknown="infrequent_if_exist"
    )
    # X[:, 0] 1 and 2 are infrequent
    # X[:, 1] 1 and 10 are infrequent
    # X[:, 2] nothing is infrequent

    X_trans = ohe.fit_transform(X).toarray()
    assert_array_equal(ohe.infrequent_categories_[0], [1, 2])
    assert_array_equal(ohe.infrequent_categories_[1], [1, 10])
    assert_array_equal(ohe.infrequent_categories_[2], None)

    # 'infrequent' is used to denote the infrequent categories
    # For the first column, 1 and 2 have the same frequency. In this case,
    # 1 will be chosen to be the feature name because is smaller lexiconically
    feature_names = ohe.get_feature_names_out()
    assert_array_equal(
        [
            "x0_0",
            "x0_3",
            "x0_infrequent_sklearn",
            "x1_0",
            "x1_5",
            "x1_infrequent_sklearn",
            "x2_0",
            "x2_1",
        ],
        feature_names,
    )

    expected = [
        [1, 0, 0, 1, 0, 0, 0, 1],
        [0, 0, 1, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 1, 0, 0, 1],
        [0, 1, 0, 0, 0, 1, 1, 0],
        [0, 1, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 1, 0],
        [0, 0, 1, 0, 1, 0, 0, 1],
        [1, 0, 0, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 0, 0, 1],
    ]

    assert_allclose(expected, X_trans)

    X_test = [[3, 1, 2], [4, 0, 3]]

    X_test_trans = ohe.transform(X_test)

    # X[:, 2] does not have an infrequent category, thus it is encoded as all
    # zeros
    expected = [[0, 1, 0, 0, 0, 1, 0, 0], [0, 0, 1, 1, 0, 0, 0, 0]]
    assert_allclose(expected, X_test_trans.toarray())

    X_inv = ohe.inverse_transform(X_test_trans)
    expected_inv = np.array(
        [[3, "infrequent_sklearn", None], ["infrequent_sklearn", 0, None]], dtype=object
    )
    assert_array_equal(expected_inv, X_inv)

    # error for unknown categories
    ohe = OneHotEncoder(
        categories="auto", max_categories=3, handle_unknown="error"
    ).fit(X)
    with pytest.raises(ValueError, match="Found unknown categories"):
        ohe.transform(X_test)

    # only infrequent or known categories
    X_test = [[1, 1, 1], [3, 10, 0]]
    X_test_trans = ohe.transform(X_test)

    expected = [[0, 0, 1, 0, 0, 1, 0, 1], [0, 1, 0, 0, 0, 1, 1, 0]]
    assert_allclose(expected, X_test_trans.toarray())

    X_inv = ohe.inverse_transform(X_test_trans)

    expected_inv = np.array(
        [["infrequent_sklearn", "infrequent_sklearn", 1], [3, "infrequent_sklearn", 0]],
        dtype=object,
    )
    assert_array_equal(expected_inv, X_inv)


def test_ohe_infrequent_multiple_categories_dtypes():
    """Test infrequent categories with a pandas dataframe with multiple dtypes."""

    pd = pytest.importorskip("pandas")
    X = pd.DataFrame(
        {
            "str": ["a", "f", "c", "f", "f", "a", "c", "b", "b"],
            "int": [5, 3, 0, 10, 10, 12, 0, 3, 5],
        },
        columns=["str", "int"],
    )

    ohe = OneHotEncoder(
        categories="auto", max_categories=3, handle_unknown="infrequent_if_exist"
    )
    # X[:, 0] 'a', 'b', 'c' have the same frequency. 'a' and 'b' will be
    # considered infrequent because they are greater

    # X[:, 1] 0, 3, 5, 10 has frequency 2 and 12 has frequency 1.
    # 0, 3, 12 will be considered infrequent

    X_trans = ohe.fit_transform(X).toarray()
    assert_array_equal(ohe.infrequent_categories_[0], ["a", "b"])
    assert_array_equal(ohe.infrequent_categories_[1], [0, 3, 12])

    expected = [
        [0, 0, 1, 1, 0, 0],
        [0, 1, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1],
        [1, 0, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 1],
        [0, 0, 1, 1, 0, 0],
    ]

    assert_allclose(expected, X_trans)

    X_test = pd.DataFrame({"str": ["b", "f"], "int": [14, 12]}, columns=["str", "int"])

    expected = [[0, 0, 1, 0, 0, 1], [0, 1, 0, 0, 0, 1]]
    X_test_trans = ohe.transform(X_test)
    assert_allclose(expected, X_test_trans.toarray())

    X_inv = ohe.inverse_transform(X_test_trans)
    expected_inv = np.array(
        [["infrequent_sklearn", "infrequent_sklearn"], ["f", "infrequent_sklearn"]],
        dtype=object,
    )
    assert_array_equal(expected_inv, X_inv)

    # only infrequent or known categories
    X_test = pd.DataFrame({"str": ["c", "b"], "int": [12, 5]}, columns=["str", "int"])
    X_test_trans = ohe.transform(X_test).toarray()
    expected = [[1, 0, 0, 0, 0, 1], [0, 0, 1, 1, 0, 0]]
    assert_allclose(expected, X_test_trans)

    X_inv = ohe.inverse_transform(X_test_trans)
    expected_inv = np.array(
        [["c", "infrequent_sklearn"], ["infrequent_sklearn", 5]], dtype=object
    )
    assert_array_equal(expected_inv, X_inv)


@pytest.mark.parametrize("kwargs", [{"min_frequency": 21, "max_categories": 1}])
def test_ohe_infrequent_one_level_errors(kwargs):
    """All user provided categories are infrequent."""
    X_train = np.array([["a"] * 5 + ["b"] * 20 + ["c"] * 10 + ["d"] * 2]).T

    ohe = OneHotEncoder(
        handle_unknown="infrequent_if_exist", sparse_output=False, **kwargs
    )
    ohe.fit(X_train)

    X_trans = ohe.transform([["a"]])
    assert_allclose(X_trans, [[1]])


@pytest.mark.parametrize("kwargs", [{"min_frequency": 2, "max_categories": 3}])
def test_ohe_infrequent_user_cats_unknown_training_errors(kwargs):
    """All user provided categories are infrequent."""

    X_train = np.array([["e"] * 3], dtype=object).T
    ohe = OneHotEncoder(
        categories=[["c", "d", "a", "b"]],
        sparse_output=False,
        handle_unknown="infrequent_if_exist",
        **kwargs,
    ).fit(X_train)

    X_trans = ohe.transform([["a"], ["e"]])
    assert_allclose(X_trans, [[1], [1]])


# deliberately omit 'OS' as an invalid combo
@pytest.mark.parametrize(
    "input_dtype, category_dtype", ["OO", "OU", "UO", "UU", "SO", "SU", "SS"]
)
@pytest.mark.parametrize("array_type", ["list", "array", "dataframe"])
def test_encoders_string_categories(input_dtype, category_dtype, array_type):
    """Check that encoding work with object, unicode, and byte string dtypes.
    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/15616
    https://github.com/scikit-learn/scikit-learn/issues/15726
    https://github.com/scikit-learn/scikit-learn/issues/19677
    """

    X = np.array([["b"], ["a"]], dtype=input_dtype)
    categories = [np.array(["b", "a"], dtype=category_dtype)]
    ohe = OneHotEncoder(categories=categories, sparse_output=False).fit(X)

    X_test = _convert_container(
        [["a"], ["a"], ["b"], ["a"]], array_type, dtype=input_dtype
    )
    X_trans = ohe.transform(X_test)

    expected = np.array([[0, 1], [0, 1], [1, 0], [0, 1]])
    assert_allclose(X_trans, expected)

    oe = OrdinalEncoder(categories=categories).fit(X)
    X_trans = oe.transform(X_test)

    expected = np.array([[1], [1], [0], [1]])
    assert_array_equal(X_trans, expected)


def test_mixed_string_bytes_categoricals():
    """Check that this mixture of predefined categories and X raises an error.

    Categories defined as bytes can not easily be compared to data that is
    a string.
    """
    # data as unicode
    X = np.array([["b"], ["a"]], dtype="U")
    # predefined categories as bytes
    categories = [np.array(["b", "a"], dtype="S")]
    ohe = OneHotEncoder(categories=categories, sparse_output=False)

    msg = re.escape(
        "In column 0, the predefined categories have type 'bytes' which is incompatible"
        " with values of type 'str_'."
    )

    with pytest.raises(ValueError, match=msg):
        ohe.fit(X)


@pytest.mark.parametrize("missing_value", [np.nan, None])
def test_ohe_missing_values_get_feature_names(missing_value):
    # encoder with missing values with object dtypes
    X = np.array([["a", "b", missing_value, "a", missing_value]], dtype=object).T
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(X)
    names = ohe.get_feature_names_out()
    assert_array_equal(names, ["x0_a", "x0_b", f"x0_{missing_value}"])


def test_ohe_missing_value_support_pandas():
    # check support for pandas with mixed dtypes and missing values
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(
        {
            "col1": ["dog", "cat", None, "cat"],
            "col2": np.array([3, 0, 4, np.nan], dtype=float),
        },
        columns=["col1", "col2"],
    )
    expected_df_trans = np.array(
        [
            [0, 1, 0, 0, 1, 0, 0],
            [1, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 0, 1],
        ]
    )

    Xtr = check_categorical_onehot(df)
    assert_allclose(Xtr, expected_df_trans)


@pytest.mark.parametrize("handle_unknown", ["infrequent_if_exist", "ignore"])
@pytest.mark.parametrize("pd_nan_type", ["pd.NA", "np.nan"])
def test_ohe_missing_value_support_pandas_categorical(pd_nan_type, handle_unknown):
    # checks pandas dataframe with categorical features
    pd = pytest.importorskip("pandas")

    pd_missing_value = pd.NA if pd_nan_type == "pd.NA" else np.nan

    df = pd.DataFrame(
        {
            "col1": pd.Series(["c", "a", pd_missing_value, "b", "a"], dtype="category"),
        }
    )
    expected_df_trans = np.array(
        [
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
        ]
    )

    ohe = OneHotEncoder(sparse_output=False, handle_unknown=handle_unknown)
    df_trans = ohe.fit_transform(df)
    assert_allclose(expected_df_trans, df_trans)

    assert len(ohe.categories_) == 1
    assert_array_equal(ohe.categories_[0][:-1], ["a", "b", "c"])
    assert np.isnan(ohe.categories_[0][-1])


@pytest.mark.parametrize("handle_unknown", ["ignore", "infrequent_if_exist"])
def test_ohe_drop_first_handle_unknown_ignore_warns(handle_unknown):
    """Check drop='first' and handle_unknown='ignore'/'infrequent_if_exist'
    during transform."""
    X = [["a", 0], ["b", 2], ["b", 1]]

    ohe = OneHotEncoder(
        drop="first", sparse_output=False, handle_unknown=handle_unknown
    )
    X_trans = ohe.fit_transform(X)

    X_expected = np.array(
        [
            [0, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
        ]
    )
    assert_allclose(X_trans, X_expected)

    # Both categories are unknown
    X_test = [["c", 3]]
    X_expected = np.array([[0, 0, 0]])

    warn_msg = (
        r"Found unknown categories in columns \[0, 1\] during "
        "transform. These unknown categories will be encoded as all "
        "zeros"
    )
    with pytest.warns(UserWarning, match=warn_msg):
        X_trans = ohe.transform(X_test)
    assert_allclose(X_trans, X_expected)

    # inverse_transform maps to None
    X_inv = ohe.inverse_transform(X_expected)
    assert_array_equal(X_inv, np.array([["a", 0]], dtype=object))


@pytest.mark.parametrize("handle_unknown", ["ignore", "infrequent_if_exist"])
def test_ohe_drop_if_binary_handle_unknown_ignore_warns(handle_unknown):
    """Check drop='if_binary' and handle_unknown='ignore' during transform."""
    X = [["a", 0], ["b", 2], ["b", 1]]

    ohe = OneHotEncoder(
        drop="if_binary", sparse_output=False, handle_unknown=handle_unknown
    )
    X_trans = ohe.fit_transform(X)

    X_expected = np.array(
        [
            [0, 1, 0, 0],
            [1, 0, 0, 1],
            [1, 0, 1, 0],
        ]
    )
    assert_allclose(X_trans, X_expected)

    # Both categories are unknown
    X_test = [["c", 3]]
    X_expected = np.array([[0, 0, 0, 0]])

    warn_msg = (
        r"Found unknown categories in columns \[0, 1\] during "
        "transform. These unknown categories will be encoded as all "
        "zeros"
    )
    with pytest.warns(UserWarning, match=warn_msg):
        X_trans = ohe.transform(X_test)
    assert_allclose(X_trans, X_expected)

    # inverse_transform maps to None
    X_inv = ohe.inverse_transform(X_expected)
    assert_array_equal(X_inv, np.array([["a", None]], dtype=object))


@pytest.mark.parametrize("handle_unknown", ["ignore", "infrequent_if_exist"])
def test_ohe_drop_first_explicit_categories(handle_unknown):
    """Check drop='first' and handle_unknown='ignore'/'infrequent_if_exist'
    during fit with categories passed in."""

    X = [["a", 0], ["b", 2], ["b", 1]]

    ohe = OneHotEncoder(
        drop="first",
        sparse_output=False,
        handle_unknown=handle_unknown,
        categories=[["b", "a"], [1, 2]],
    )
    ohe.fit(X)

    X_test = [["c", 1]]
    X_expected = np.array([[0, 0]])

    warn_msg = (
        r"Found unknown categories in columns \[0\] during transform. "
        r"These unknown categories will be encoded as all zeros"
    )
    with pytest.warns(UserWarning, match=warn_msg):
        X_trans = ohe.transform(X_test)
    assert_allclose(X_trans, X_expected)


def test_ohe_more_informative_error_message():
    """Raise informative error message when pandas output and sparse_output=True."""
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["z", "b", "b"]}, columns=["a", "b"])

    ohe = OneHotEncoder(sparse_output=True)
    ohe.set_output(transform="pandas")

    msg = (
        "Pandas output does not support sparse data. Set "
        "sparse_output=False to output pandas dataframes or disable Pandas output"
    )
    with pytest.raises(ValueError, match=msg):
        ohe.fit_transform(df)

    ohe.fit(df)
    with pytest.raises(ValueError, match=msg):
        ohe.transform(df)


def test_ordinal_encoder_passthrough_missing_values_float_errors_dtype():
    """Test ordinal encoder with nan passthrough fails when dtype=np.int32."""

    X = np.array([[np.nan, 3.0, 1.0, 3.0]]).T
    oe = OrdinalEncoder(dtype=np.int32)

    msg = (
        r"There are missing values in features \[0\]. For OrdinalEncoder "
        f"to encode missing values with dtype: {np.int32}"
    )
    with pytest.raises(ValueError, match=msg):
        oe.fit(X)


@pytest.mark.parametrize("encoded_missing_value", [np.nan, -2])
def test_ordinal_encoder_passthrough_missing_values_float(encoded_missing_value):
    """Test ordinal encoder with nan on float dtypes."""

    X = np.array([[np.nan, 3.0, 1.0, 3.0]], dtype=np.float64).T
    oe = OrdinalEncoder(encoded_missing_value=encoded_missing_value).fit(X)

    assert len(oe.categories_) == 1

    assert_allclose(oe.categories_[0], [1.0, 3.0, np.nan])

    X_trans = oe.transform(X)
    assert_allclose(X_trans, [[encoded_missing_value], [1.0], [0.0], [1.0]])

    X_inverse = oe.inverse_transform(X_trans)
    assert_allclose(X_inverse, X)


@pytest.mark.parametrize("pd_nan_type", ["pd.NA", "np.nan"])
@pytest.mark.parametrize("encoded_missing_value", [np.nan, -2])
def test_ordinal_encoder_missing_value_support_pandas_categorical(
    pd_nan_type, encoded_missing_value
):
    """Check ordinal encoder is compatible with pandas."""
    # checks pandas dataframe with categorical features
    pd = pytest.importorskip("pandas")

    pd_missing_value = pd.NA if pd_nan_type == "pd.NA" else np.nan

    df = pd.DataFrame(
        {
            "col1": pd.Series(["c", "a", pd_missing_value, "b", "a"], dtype="category"),
        }
    )

    oe = OrdinalEncoder(encoded_missing_value=encoded_missing_value).fit(df)
    assert len(oe.categories_) == 1
    assert_array_equal(oe.categories_[0][:3], ["a", "b", "c"])
    assert np.isnan(oe.categories_[0][-1])

    df_trans = oe.transform(df)

    assert_allclose(df_trans, [[2.0], [0.0], [encoded_missing_value], [1.0], [0.0]])

    X_inverse = oe.inverse_transform(df_trans)
    assert X_inverse.shape == (5, 1)
    assert_array_equal(X_inverse[:2, 0], ["c", "a"])
    assert_array_equal(X_inverse[3:, 0], ["b", "a"])
    assert np.isnan(X_inverse[2, 0])


@pytest.mark.parametrize(
    "X, X2, cats, cat_dtype",
    [
        (
            (
                np.array([["a", np.nan]], dtype=object).T,
                np.array([["a", "b"]], dtype=object).T,
                [np.array(["a", "d", np.nan], dtype=object)],
                np.object_,
            )
        ),
        (
            (
                np.array([["a", np.nan]], dtype=object).T,
                np.array([["a", "b"]], dtype=object).T,
                [np.array(["a", "d", np.nan], dtype=object)],
                np.object_,
            )
        ),
        (
            (
                np.array([[2.0, np.nan]], dtype=np.float64).T,
                np.array([[3.0]], dtype=np.float64).T,
                [np.array([2.0, 4.0, np.nan])],
                np.float64,
            )
        ),
    ],
    ids=[
        "object-None-missing-value",
        "object-nan-missing_value",
        "numeric-missing-value",
    ],
)
def test_ordinal_encoder_specified_categories_missing_passthrough(
    X, X2, cats, cat_dtype
):
    """Test ordinal encoder for specified categories."""
    oe = OrdinalEncoder(categories=cats)
    exp = np.array([[0.0], [np.nan]])
    assert_array_equal(oe.fit_transform(X), exp)
    # manually specified categories should have same dtype as
    # the data when coerced from lists
    assert oe.categories_[0].dtype == cat_dtype

    # when specifying categories manually, unknown categories should already
    # raise when fitting
    oe = OrdinalEncoder(categories=cats)
    with pytest.raises(ValueError, match="Found unknown categories"):
        oe.fit(X2)


@pytest.mark.parametrize("Encoder", [OneHotEncoder, OrdinalEncoder])
def test_encoder_duplicate_specified_categories(Encoder):
    """Test encoder for specified categories have duplicate values.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/27088
    """
    cats = [np.array(["a", "b", "a"], dtype=object)]
    enc = Encoder(categories=cats)
    X = np.array([["a", "b"]], dtype=object).T
    with pytest.raises(
        ValueError, match="the predefined categories contain duplicate elements."
    ):
        enc.fit(X)


@pytest.mark.parametrize(
    "X, expected_X_trans, X_test",
    [
        (
            np.array([[1.0, np.nan, 3.0]]).T,
            np.array([[0.0, np.nan, 1.0]]).T,
            np.array([[4.0]]),
        ),
        (
            np.array([[1.0, 4.0, 3.0]]).T,
            np.array([[0.0, 2.0, 1.0]]).T,
            np.array([[np.nan]]),
        ),
        (
            np.array([["c", np.nan, "b"]], dtype=object).T,
            np.array([[1.0, np.nan, 0.0]]).T,
            np.array([["d"]], dtype=object),
        ),
        (
            np.array([["c", "a", "b"]], dtype=object).T,
            np.array([[2.0, 0.0, 1.0]]).T,
            np.array([[np.nan]], dtype=object),
        ),
    ],
)
def test_ordinal_encoder_handle_missing_and_unknown(X, expected_X_trans, X_test):
    """Test the interaction between missing values and handle_unknown"""

    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    X_trans = oe.fit_transform(X)
    assert_allclose(X_trans, expected_X_trans)

    assert_allclose(oe.transform(X_test), [[-1.0]])


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_ordinal_encoder_sparse(csr_container):
    """Check that we raise proper error with sparse input in OrdinalEncoder.
    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/19878
    """
    X = np.array([[3, 2, 1], [0, 1, 1]])
    X_sparse = csr_container(X)

    encoder = OrdinalEncoder()

    err_msg = "Sparse data was passed, but dense data is required"
    with pytest.raises(TypeError, match=err_msg):
        encoder.fit(X_sparse)
    with pytest.raises(TypeError, match=err_msg):
        encoder.fit_transform(X_sparse)

    X_trans = encoder.fit_transform(X)
    X_trans_sparse = csr_container(X_trans)
    with pytest.raises(TypeError, match=err_msg):
        encoder.inverse_transform(X_trans_sparse)


def test_ordinal_encoder_fit_with_unseen_category():
    """Check OrdinalEncoder.fit works with unseen category when
    `handle_unknown="use_encoded_value"`.
    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/19872
    """
    X = np.array([0, 0, 1, 0, 2, 5])[:, np.newaxis]
    oe = OrdinalEncoder(
        categories=[[-1, 0, 1]], handle_unknown="use_encoded_value", unknown_value=-999
    )
    oe.fit(X)

    oe = OrdinalEncoder(categories=[[-1, 0, 1]], handle_unknown="error")
    with pytest.raises(ValueError, match="Found unknown categories"):
        oe.fit(X)


@pytest.mark.parametrize(
    "X_train",
    [
        [["AA", "B"]],
        np.array([["AA", "B"]], dtype="O"),
        np.array([["AA", "B"]], dtype="U"),
    ],
)
@pytest.mark.parametrize(
    "X_test",
    [
        [["A", "B"]],
        np.array([["A", "B"]], dtype="O"),
        np.array([["A", "B"]], dtype="U"),
    ],
)
def test_ordinal_encoder_handle_unknown_string_dtypes(X_train, X_test):
    """Checks that `OrdinalEncoder` transforms string dtypes.
    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/19872
    """
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-9)
    enc.fit(X_train)

    X_trans = enc.transform(X_test)
    assert_allclose(X_trans, [[-9, 0]])


def test_ordinal_encoder_python_integer():
    """Check that `OrdinalEncoder` accepts Python integers that are potentially
    larger than 64 bits.
    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/20721
    """
    X = np.array(
        [
            44253463435747313673,
            9867966753463435747313673,
            44253462342215747313673,
            442534634357764313673,
        ]
    ).reshape(-1, 1)
    encoder = OrdinalEncoder().fit(X)
    assert_array_equal(encoder.categories_, np.sort(X, axis=0).T)
    X_trans = encoder.transform(X)
    assert_array_equal(X_trans, [[0], [3], [2], [1]])


def test_ordinal_encoder_features_names_out_pandas():
    """Check feature names out is same as the input."""
    pd = pytest.importorskip("pandas")

    names = ["b", "c", "a"]
    X = pd.DataFrame([[1, 2, 3]], columns=names)
    enc = OrdinalEncoder().fit(X)

    feature_names_out = enc.get_feature_names_out()
    assert_array_equal(names, feature_names_out)


def test_ordinal_encoder_unknown_missing_interaction():
    """Check interactions between encode_unknown and missing value encoding."""

    X = np.array([["a"], ["b"], [np.nan]], dtype=object)

    oe = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=np.nan,
        encoded_missing_value=-3,
    ).fit(X)

    X_trans = oe.transform(X)
    assert_allclose(X_trans, [[0], [1], [-3]])

    # "c" is unknown and is mapped to np.nan
    # "None" is a missing value and is set to -3
    X_test = np.array([["c"], [np.nan]], dtype=object)
    X_test_trans = oe.transform(X_test)
    assert_allclose(X_test_trans, [[np.nan], [-3]])

    # Non-regression test for #24082
    X_roundtrip = oe.inverse_transform(X_test_trans)

    # np.nan is unknown so it maps to None
    assert X_roundtrip[0][0] is None

    # -3 is the encoded missing value so it maps back to nan
    assert np.isnan(X_roundtrip[1][0])


@pytest.mark.parametrize("with_pandas", [True, False])
def test_ordinal_encoder_encoded_missing_value_error(with_pandas):
    """Check OrdinalEncoder errors when encoded_missing_value is used by
    an known category."""
    X = np.array([["a", "dog"], ["b", "cat"], ["c", np.nan]], dtype=object)

    # The 0-th feature has no missing values so it is not included in the list of
    # features
    error_msg = (
        r"encoded_missing_value \(1\) is already used to encode a known category "
        r"in features: "
    )

    if with_pandas:
        pd = pytest.importorskip("pandas")
        X = pd.DataFrame(X, columns=["letter", "pet"])
        error_msg = error_msg + r"\['pet'\]"
    else:
        error_msg = error_msg + r"\[1\]"

    oe = OrdinalEncoder(encoded_missing_value=1)

    with pytest.raises(ValueError, match=error_msg):
        oe.fit(X)


@pytest.mark.parametrize(
    "X_train, X_test_trans_expected, X_roundtrip_expected",
    [
        (
            # missing value is not in training set
            # inverse transform will considering encoded nan as unknown
            np.array([["a"], ["1"]], dtype=object),
            [[0], [np.nan], [np.nan]],
            np.asarray([["1"], [None], [None]], dtype=object),
        ),
        (
            # missing value in training set,
            # inverse transform will considering encoded nan as missing
            np.array([[np.nan], ["1"], ["a"]], dtype=object),
            [[0], [np.nan], [np.nan]],
            np.asarray([["1"], [np.nan], [np.nan]], dtype=object),
        ),
    ],
)
def test_ordinal_encoder_unknown_missing_interaction_both_nan(
    X_train, X_test_trans_expected, X_roundtrip_expected
):
    """Check transform when unknown_value and encoded_missing_value is nan.

    Non-regression test for #24082.
    """
    oe = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=np.nan,
        encoded_missing_value=np.nan,
    ).fit(X_train)

    X_test = np.array([["1"], [np.nan], ["b"]])
    X_test_trans = oe.transform(X_test)

    # both nan and unknown are encoded as nan
    assert_allclose(X_test_trans, X_test_trans_expected)
    X_roundtrip = oe.inverse_transform(X_test_trans)

    n_samples = X_roundtrip_expected.shape[0]
    for i in range(n_samples):
        expected_val = X_roundtrip_expected[i, 0]
        val = X_roundtrip[i, 0]

        if expected_val is None:
            assert val is None
        elif is_scalar_nan(expected_val):
            assert np.isnan(val)
        else:
            assert val == expected_val


def test_one_hot_encoder_set_output():
    """Check OneHotEncoder works with set_output."""
    pd = pytest.importorskip("pandas")

    X_df = pd.DataFrame({"A": ["a", "b"], "B": [1, 2]})
    ohe = OneHotEncoder()

    ohe.set_output(transform="pandas")

    match = "Pandas output does not support sparse data. Set sparse_output=False"
    with pytest.raises(ValueError, match=match):
        ohe.fit_transform(X_df)

    ohe_default = OneHotEncoder(sparse_output=False).set_output(transform="default")
    ohe_pandas = OneHotEncoder(sparse_output=False).set_output(transform="pandas")

    X_default = ohe_default.fit_transform(X_df)
    X_pandas = ohe_pandas.fit_transform(X_df)

    assert_allclose(X_pandas.to_numpy(), X_default)
    assert_array_equal(ohe_pandas.get_feature_names_out(), X_pandas.columns)


def test_ordinal_set_output():
    """Check OrdinalEncoder works with set_output."""
    pd = pytest.importorskip("pandas")

    X_df = pd.DataFrame({"A": ["a", "b"], "B": [1, 2]})

    ord_default = OrdinalEncoder().set_output(transform="default")
    ord_pandas = OrdinalEncoder().set_output(transform="pandas")

    X_default = ord_default.fit_transform(X_df)
    X_pandas = ord_pandas.fit_transform(X_df)

    assert_allclose(X_pandas.to_numpy(), X_default)
    assert_array_equal(ord_pandas.get_feature_names_out(), X_pandas.columns)


def test_predefined_categories_dtype():
    """Check that the categories_ dtype is `object` for string categories

    Regression test for gh-25171.
    """
    categories = [["as", "mmas", "eas", "ras", "acs"], ["1", "2"]]

    enc = OneHotEncoder(categories=categories)

    enc.fit([["as", "1"]])

    assert len(categories) == len(enc.categories_)
    for n, cat in enumerate(enc.categories_):
        assert cat.dtype == object
        assert_array_equal(categories[n], cat)


def test_ordinal_encoder_missing_unknown_encoding_max():
    """Check missing value or unknown encoding can equal the cardinality."""
    X = np.array([["dog"], ["cat"], [np.nan]], dtype=object)
    X_trans = OrdinalEncoder(encoded_missing_value=2).fit_transform(X)
    assert_allclose(X_trans, [[1], [0], [2]])

    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=2).fit(X)
    X_test = np.array([["snake"]])
    X_trans = enc.transform(X_test)
    assert_allclose(X_trans, [[2]])


def test_drop_idx_infrequent_categories():
    """Check drop_idx is defined correctly with infrequent categories.

    Non-regression test for gh-25550.
    """
    X = np.array(
        [["a"] * 2 + ["b"] * 4 + ["c"] * 4 + ["d"] * 4 + ["e"] * 4], dtype=object
    ).T
    ohe = OneHotEncoder(min_frequency=4, sparse_output=False, drop="first").fit(X)
    assert_array_equal(
        ohe.get_feature_names_out(), ["x0_c", "x0_d", "x0_e", "x0_infrequent_sklearn"]
    )
    assert ohe.categories_[0][ohe.drop_idx_[0]] == "b"

    X = np.array([["a"] * 2 + ["b"] * 2 + ["c"] * 10], dtype=object).T
    ohe = OneHotEncoder(min_frequency=4, sparse_output=False, drop="if_binary").fit(X)
    assert_array_equal(ohe.get_feature_names_out(), ["x0_infrequent_sklearn"])
    assert ohe.categories_[0][ohe.drop_idx_[0]] == "c"

    X = np.array(
        [["a"] * 2 + ["b"] * 4 + ["c"] * 4 + ["d"] * 4 + ["e"] * 4], dtype=object
    ).T
    ohe = OneHotEncoder(min_frequency=4, sparse_output=False, drop=["d"]).fit(X)
    assert_array_equal(
        ohe.get_feature_names_out(), ["x0_b", "x0_c", "x0_e", "x0_infrequent_sklearn"]
    )
    assert ohe.categories_[0][ohe.drop_idx_[0]] == "d"

    ohe = OneHotEncoder(min_frequency=4, sparse_output=False, drop=None).fit(X)
    assert_array_equal(
        ohe.get_feature_names_out(),
        ["x0_b", "x0_c", "x0_d", "x0_e", "x0_infrequent_sklearn"],
    )
    assert ohe.drop_idx_ is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_categories": 3},
        {"min_frequency": 6},
        {"min_frequency": 9},
        {"min_frequency": 0.24},
        {"min_frequency": 0.16},
        {"max_categories": 3, "min_frequency": 8},
        {"max_categories": 4, "min_frequency": 6},
    ],
)
def test_ordinal_encoder_infrequent_three_levels(kwargs):
    """Test parameters for grouping 'a', and 'd' into the infrequent category."""

    X_train = np.array([["a"] * 5 + ["b"] * 20 + ["c"] * 10 + ["d"] * 3]).T
    ordinal = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1, **kwargs
    ).fit(X_train)
    assert_array_equal(ordinal.categories_, [["a", "b", "c", "d"]])
    assert_array_equal(ordinal.infrequent_categories_, [["a", "d"]])

    X_test = [["a"], ["b"], ["c"], ["d"], ["z"]]
    expected_trans = [[2], [0], [1], [2], [-1]]

    X_trans = ordinal.transform(X_test)
    assert_allclose(X_trans, expected_trans)

    X_inverse = ordinal.inverse_transform(X_trans)
    expected_inverse = [
        ["infrequent_sklearn"],
        ["b"],
        ["c"],
        ["infrequent_sklearn"],
        [None],
    ]
    assert_array_equal(X_inverse, expected_inverse)


def test_ordinal_encoder_infrequent_three_levels_user_cats():
    """Test that the order of the categories provided by a user is respected.

    In this case 'c' is encoded as the first category and 'b' is encoded
    as the second one.
    """

    X_train = np.array(
        [["a"] * 5 + ["b"] * 20 + ["c"] * 10 + ["d"] * 3], dtype=object
    ).T
    ordinal = OrdinalEncoder(
        categories=[["c", "d", "b", "a"]],
        max_categories=3,
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    ).fit(X_train)
    assert_array_equal(ordinal.categories_, [["c", "d", "b", "a"]])
    assert_array_equal(ordinal.infrequent_categories_, [["d", "a"]])

    X_test = [["a"], ["b"], ["c"], ["d"], ["z"]]
    expected_trans = [[2], [1], [0], [2], [-1]]

    X_trans = ordinal.transform(X_test)
    assert_allclose(X_trans, expected_trans)

    X_inverse = ordinal.inverse_transform(X_trans)
    expected_inverse = [
        ["infrequent_sklearn"],
        ["b"],
        ["c"],
        ["infrequent_sklearn"],
        [None],
    ]
    assert_array_equal(X_inverse, expected_inverse)


def test_ordinal_encoder_infrequent_mixed():
    """Test when feature 0 has infrequent categories and feature 1 does not."""

    X = np.column_stack(([0, 1, 3, 3, 3, 3, 2, 0, 3], [0, 0, 0, 0, 1, 1, 1, 1, 1]))

    ordinal = OrdinalEncoder(max_categories=3).fit(X)

    assert_array_equal(ordinal.infrequent_categories_[0], [1, 2])
    assert ordinal.infrequent_categories_[1] is None

    X_test = [[3, 0], [1, 1]]
    expected_trans = [[1, 0], [2, 1]]

    X_trans = ordinal.transform(X_test)
    assert_allclose(X_trans, expected_trans)

    X_inverse = ordinal.inverse_transform(X_trans)
    expected_inverse = np.array([[3, 0], ["infrequent_sklearn", 1]], dtype=object)
    assert_array_equal(X_inverse, expected_inverse)


def test_ordinal_encoder_infrequent_multiple_categories_dtypes():
    """Test infrequent categories with a pandas DataFrame with multiple dtypes."""

    pd = pytest.importorskip("pandas")
    categorical_dtype = pd.CategoricalDtype(["bird", "cat", "dog", "snake"])
    X = pd.DataFrame(
        {
            "str": ["a", "f", "c", "f", "f", "a", "c", "b", "b"],
            "int": [5, 3, 0, 10, 10, 12, 0, 3, 5],
            "categorical": pd.Series(
                ["dog"] * 4 + ["cat"] * 3 + ["snake"] + ["bird"],
                dtype=categorical_dtype,
            ),
        },
        columns=["str", "int", "categorical"],
    )

    ordinal = OrdinalEncoder(max_categories=3).fit(X)
    # X[:, 0] 'a', 'b', 'c' have the same frequency. 'a' and 'b' will be
    # considered infrequent because they appear first when sorted

    # X[:, 1] 0, 3, 5, 10 has frequency 2 and 12 has frequency 1.
    # 0, 3, 12 will be considered infrequent because they appear first when
    # sorted.

    # X[:, 2] "snake" and "bird" or infrequent

    assert_array_equal(ordinal.infrequent_categories_[0], ["a", "b"])
    assert_array_equal(ordinal.infrequent_categories_[1], [0, 3, 12])
    assert_array_equal(ordinal.infrequent_categories_[2], ["bird", "snake"])

    X_test = pd.DataFrame(
        {
            "str": ["a", "b", "f", "c"],
            "int": [12, 0, 10, 5],
            "categorical": pd.Series(
                ["cat"] + ["snake"] + ["bird"] + ["dog"],
                dtype=categorical_dtype,
            ),
        },
        columns=["str", "int", "categorical"],
    )
    expected_trans = [[2, 2, 0], [2, 2, 2], [1, 1, 2], [0, 0, 1]]

    X_trans = ordinal.transform(X_test)
    assert_allclose(X_trans, expected_trans)


def test_ordinal_encoder_infrequent_custom_mapping():
    """Check behavior of unknown_value and encoded_missing_value with infrequent."""
    X_train = np.array(
        [["a"] * 5 + ["b"] * 20 + ["c"] * 10 + ["d"] * 3 + [np.nan]], dtype=object
    ).T

    ordinal = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=2,
        max_categories=2,
        encoded_missing_value=3,
    ).fit(X_train)
    assert_array_equal(ordinal.infrequent_categories_, [["a", "c", "d"]])

    X_test = np.array([["a"], ["b"], ["c"], ["d"], ["e"], [np.nan]], dtype=object)
    expected_trans = [[1], [0], [1], [1], [2], [3]]

    X_trans = ordinal.transform(X_test)
    assert_allclose(X_trans, expected_trans)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_categories": 6},
        {"min_frequency": 2},
    ],
)
def test_ordinal_encoder_all_frequent(kwargs):
    """All categories are considered frequent have same encoding as default encoder."""
    X_train = np.array(
        [["a"] * 5 + ["b"] * 20 + ["c"] * 10 + ["d"] * 3], dtype=object
    ).T

    adjusted_encoder = OrdinalEncoder(
        **kwargs, handle_unknown="use_encoded_value", unknown_value=-1
    ).fit(X_train)
    default_encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1
    ).fit(X_train)

    X_test = [["a"], ["b"], ["c"], ["d"], ["e"]]

    assert_allclose(
        adjusted_encoder.transform(X_test), default_encoder.transform(X_test)
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_categories": 1},
        {"min_frequency": 100},
    ],
)
def test_ordinal_encoder_all_infrequent(kwargs):
    """When all categories are infrequent, they are all encoded as zero."""
    X_train = np.array(
        [["a"] * 5 + ["b"] * 20 + ["c"] * 10 + ["d"] * 3], dtype=object
    ).T
    encoder = OrdinalEncoder(
        **kwargs, handle_unknown="use_encoded_value", unknown_value=-1
    ).fit(X_train)

    X_test = [["a"], ["b"], ["c"], ["d"], ["e"]]
    assert_allclose(encoder.transform(X_test), [[0], [0], [0], [0], [-1]])


def test_ordinal_encoder_missing_appears_frequent():
    """Check behavior when missing value appears frequently."""
    X = np.array(
        [[np.nan] * 20 + ["dog"] * 10 + ["cat"] * 5 + ["snake"] + ["deer"]],
        dtype=object,
    ).T
    ordinal = OrdinalEncoder(max_categories=3).fit(X)

    X_test = np.array([["snake", "cat", "dog", np.nan]], dtype=object).T
    X_trans = ordinal.transform(X_test)
    assert_allclose(X_trans, [[2], [0], [1], [np.nan]])


def test_ordinal_encoder_missing_appears_infrequent():
    """Check behavior when missing value appears infrequently."""

    # feature 0 has infrequent categories
    # feature 1 has no infrequent categories
    X = np.array(
        [
            [np.nan] + ["dog"] * 10 + ["cat"] * 5 + ["snake"] + ["deer"],
            ["red"] * 9 + ["green"] * 9,
        ],
        dtype=object,
    ).T
    ordinal = OrdinalEncoder(min_frequency=4).fit(X)

    X_test = np.array(
        [
            ["snake", "red"],
            ["deer", "green"],
            [np.nan, "green"],
            ["dog", "green"],
            ["cat", "red"],
        ],
        dtype=object,
    )
    X_trans = ordinal.transform(X_test)
    assert_allclose(X_trans, [[2, 1], [2, 0], [np.nan, 0], [1, 0], [0, 1]])


@pytest.mark.parametrize("Encoder", [OneHotEncoder, OrdinalEncoder])
def test_encoder_not_fitted(Encoder):
    """Check that we raise a `NotFittedError` by calling transform before fit with
    the encoders.

    One could expect that the passing the `categories` argument to the encoder
    would make it stateless. However, `fit` is making a couple of check, such as the
    position of `np.nan`.
    """
    X = np.array([["A"], ["B"], ["C"]], dtype=object)
    encoder = Encoder(categories=[["A", "B", "C"]])
    with pytest.raises(NotFittedError):
        encoder.transform(X)
