# Authors: Lars Buitinck
#          Dan Blanchard <dblanchard@ets.org>
# License: BSD 3 clause

from random import Random

import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose, assert_array_equal

from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest, chi2


@pytest.mark.parametrize("sparse", (True, False))
@pytest.mark.parametrize("dtype", (int, np.float32, np.int16))
@pytest.mark.parametrize("sort", (True, False))
@pytest.mark.parametrize("iterable", (True, False))
def test_dictvectorizer(sparse, dtype, sort, iterable):
    D = [{"foo": 1, "bar": 3}, {"bar": 4, "baz": 2}, {"bar": 1, "quux": 1, "quuux": 2}]

    v = DictVectorizer(sparse=sparse, dtype=dtype, sort=sort)
    X = v.fit_transform(iter(D) if iterable else D)

    assert sp.issparse(X) == sparse
    assert X.shape == (3, 5)
    assert X.sum() == 14
    assert v.inverse_transform(X) == D

    if sparse:
        # CSR matrices can't be compared for equality
        assert_array_equal(
            X.toarray(), v.transform(iter(D) if iterable else D).toarray()
        )
    else:
        assert_array_equal(X, v.transform(iter(D) if iterable else D))

    if sort:
        assert v.feature_names_ == sorted(v.feature_names_)


def test_feature_selection():
    # make two feature dicts with two useful features and a bunch of useless
    # ones, in terms of chi2
    d1 = dict([("useless%d" % i, 10) for i in range(20)], useful1=1, useful2=20)
    d2 = dict([("useless%d" % i, 10) for i in range(20)], useful1=20, useful2=1)

    for indices in (True, False):
        v = DictVectorizer().fit([d1, d2])
        X = v.transform([d1, d2])
        sel = SelectKBest(chi2, k=2).fit(X, [0, 1])

        v.restrict(sel.get_support(indices=indices), indices=indices)
        assert_array_equal(v.get_feature_names_out(), ["useful1", "useful2"])


def test_one_of_k():
    D_in = [
        {"version": "1", "ham": 2},
        {"version": "2", "spam": 0.3},
        {"version=3": True, "spam": -1},
    ]
    v = DictVectorizer()
    X = v.fit_transform(D_in)
    assert X.shape == (3, 5)

    D_out = v.inverse_transform(X)
    assert D_out[0] == {"version=1": 1, "ham": 2}

    names = v.get_feature_names_out()
    assert "version=2" in names
    assert "version" not in names


def test_iterable_value():
    D_names = ["ham", "spam", "version=1", "version=2", "version=3"]
    X_expected = [
        [2.0, 0.0, 2.0, 1.0, 0.0],
        [0.0, 0.3, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0, 1.0],
    ]
    D_in = [
        {"version": ["1", "2", "1"], "ham": 2},
        {"version": "2", "spam": 0.3},
        {"version=3": True, "spam": -1},
    ]
    v = DictVectorizer()
    X = v.fit_transform(D_in)
    X = X.toarray()
    assert_array_equal(X, X_expected)

    D_out = v.inverse_transform(X)
    assert D_out[0] == {"version=1": 2, "version=2": 1, "ham": 2}

    names = v.get_feature_names_out()

    assert_array_equal(names, D_names)


def test_iterable_not_string_error():
    error_value = (
        "Unsupported type <class 'int'> in iterable value. "
        "Only iterables of string are supported."
    )
    D2 = [{"foo": "1", "bar": "2"}, {"foo": "3", "baz": "1"}, {"foo": [1, "three"]}]
    v = DictVectorizer(sparse=False)
    with pytest.raises(TypeError) as error:
        v.fit(D2)
    assert str(error.value) == error_value


def test_mapping_error():
    error_value = (
        "Unsupported value type <class 'dict'> "
        "for foo: {'one': 1, 'three': 3}.\n"
        "Mapping objects are not supported."
    )
    D2 = [
        {"foo": "1", "bar": "2"},
        {"foo": "3", "baz": "1"},
        {"foo": {"one": 1, "three": 3}},
    ]
    v = DictVectorizer(sparse=False)
    with pytest.raises(TypeError) as error:
        v.fit(D2)
    assert str(error.value) == error_value


def test_unseen_or_no_features():
    D = [{"camelot": 0, "spamalot": 1}]
    for sparse in [True, False]:
        v = DictVectorizer(sparse=sparse).fit(D)

        X = v.transform({"push the pram a lot": 2})
        if sparse:
            X = X.toarray()
        assert_array_equal(X, np.zeros((1, 2)))

        X = v.transform({})
        if sparse:
            X = X.toarray()
        assert_array_equal(X, np.zeros((1, 2)))

        with pytest.raises(ValueError, match="empty"):
            v.transform([])


def test_deterministic_vocabulary(global_random_seed):
    # Generate equal dictionaries with different memory layouts
    items = [("%03d" % i, i) for i in range(1000)]
    rng = Random(global_random_seed)
    d_sorted = dict(items)
    rng.shuffle(items)
    d_shuffled = dict(items)

    # check that the memory layout does not impact the resulting vocabulary
    v_1 = DictVectorizer().fit([d_sorted])
    v_2 = DictVectorizer().fit([d_shuffled])

    assert v_1.vocabulary_ == v_2.vocabulary_


def test_n_features_in():
    # For vectorizers, n_features_in_ does not make sense and does not exist.
    dv = DictVectorizer()
    assert not hasattr(dv, "n_features_in_")
    d = [{"foo": 1, "bar": 2}, {"foo": 3, "baz": 1}]
    dv.fit(d)
    assert not hasattr(dv, "n_features_in_")


def test_dictvectorizer_dense_sparse_equivalence():
    """Check the equivalence between between sparse and dense DictVectorizer.
    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/19978
    """
    movie_entry_fit = [
        {"category": ["thriller", "drama"], "year": 2003},
        {"category": ["animation", "family"], "year": 2011},
        {"year": 1974},
    ]
    movie_entry_transform = [{"category": ["thriller"], "unseen_feature": "3"}]
    dense_vectorizer = DictVectorizer(sparse=False)
    sparse_vectorizer = DictVectorizer(sparse=True)

    dense_vector_fit = dense_vectorizer.fit_transform(movie_entry_fit)
    sparse_vector_fit = sparse_vectorizer.fit_transform(movie_entry_fit)

    assert not sp.issparse(dense_vector_fit)
    assert sp.issparse(sparse_vector_fit)

    assert_allclose(dense_vector_fit, sparse_vector_fit.toarray())

    dense_vector_transform = dense_vectorizer.transform(movie_entry_transform)
    sparse_vector_transform = sparse_vectorizer.transform(movie_entry_transform)

    assert not sp.issparse(dense_vector_transform)
    assert sp.issparse(sparse_vector_transform)

    assert_allclose(dense_vector_transform, sparse_vector_transform.toarray())

    dense_inverse_transform = dense_vectorizer.inverse_transform(dense_vector_transform)
    sparse_inverse_transform = sparse_vectorizer.inverse_transform(
        sparse_vector_transform
    )

    expected_inverse = [{"category=thriller": 1.0}]
    assert dense_inverse_transform == expected_inverse
    assert sparse_inverse_transform == expected_inverse


def test_dict_vectorizer_unsupported_value_type():
    """Check that we raise an error when the value associated to a feature
    is not supported.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/19489
    """

    class A:
        pass

    vectorizer = DictVectorizer(sparse=True)
    X = [{"foo": A()}]
    err_msg = "Unsupported value Type"
    with pytest.raises(TypeError, match=err_msg):
        vectorizer.fit_transform(X)


def test_dict_vectorizer_get_feature_names_out():
    """Check that integer feature names are converted to strings in
    feature_names_out."""

    X = [{1: 2, 3: 4}, {2: 4}]
    dv = DictVectorizer(sparse=False).fit(X)

    feature_names = dv.get_feature_names_out()
    assert isinstance(feature_names, np.ndarray)
    assert feature_names.dtype == object
    assert_array_equal(feature_names, ["1", "2", "3"])


@pytest.mark.parametrize(
    "method, input",
    [
        ("transform", [{1: 2, 3: 4}, {2: 4}]),
        ("inverse_transform", [{1: 2, 3: 4}, {2: 4}]),
        ("restrict", [True, False, True]),
    ],
)
def test_dict_vectorizer_not_fitted_error(method, input):
    """Check that unfitted DictVectorizer instance raises NotFittedError.

    This should be part of the common test but currently they test estimator accepting
    text input.
    """
    dv = DictVectorizer(sparse=False)

    with pytest.raises(NotFittedError):
        getattr(dv, method)(input)
