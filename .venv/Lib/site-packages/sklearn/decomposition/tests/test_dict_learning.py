import itertools
import warnings
from functools import partial

import numpy as np
import pytest

import sklearn
from sklearn.base import clone
from sklearn.decomposition import (
    DictionaryLearning,
    MiniBatchDictionaryLearning,
    SparseCoder,
    dict_learning,
    dict_learning_online,
    sparse_encode,
)
from sklearn.decomposition._dict_learning import _update_dict
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_array
from sklearn.utils._testing import (
    TempMemmap,
    assert_allclose,
    assert_array_almost_equal,
    assert_array_equal,
    ignore_warnings,
)
from sklearn.utils.estimator_checks import (
    check_transformer_data_not_an_array,
    check_transformer_general,
    check_transformers_unfitted,
)
from sklearn.utils.parallel import Parallel

rng_global = np.random.RandomState(0)
n_samples, n_features = 10, 8
X = rng_global.randn(n_samples, n_features)


def test_sparse_encode_shapes_omp():
    rng = np.random.RandomState(0)
    algorithms = ["omp", "lasso_lars", "lasso_cd", "lars", "threshold"]
    for n_components, n_samples in itertools.product([1, 5], [1, 9]):
        X_ = rng.randn(n_samples, n_features)
        dictionary = rng.randn(n_components, n_features)
        for algorithm, n_jobs in itertools.product(algorithms, [1, 3]):
            code = sparse_encode(X_, dictionary, algorithm=algorithm, n_jobs=n_jobs)
            assert code.shape == (n_samples, n_components)


def test_dict_learning_shapes():
    n_components = 5
    dico = DictionaryLearning(n_components, random_state=0).fit(X)
    assert dico.components_.shape == (n_components, n_features)

    n_components = 1
    dico = DictionaryLearning(n_components, random_state=0).fit(X)
    assert dico.components_.shape == (n_components, n_features)
    assert dico.transform(X).shape == (X.shape[0], n_components)


def test_dict_learning_overcomplete():
    n_components = 12
    dico = DictionaryLearning(n_components, random_state=0).fit(X)
    assert dico.components_.shape == (n_components, n_features)


def test_max_iter():
    def ricker_function(resolution, center, width):
        """Discrete sub-sampled Ricker (Mexican hat) wavelet"""
        x = np.linspace(0, resolution - 1, resolution)
        x = (
            (2 / (np.sqrt(3 * width) * np.pi**0.25))
            * (1 - (x - center) ** 2 / width**2)
            * np.exp(-((x - center) ** 2) / (2 * width**2))
        )
        return x

    def ricker_matrix(width, resolution, n_components):
        """Dictionary of Ricker (Mexican hat) wavelets"""
        centers = np.linspace(0, resolution - 1, n_components)
        D = np.empty((n_components, resolution))
        for i, center in enumerate(centers):
            D[i] = ricker_function(resolution, center, width)
        D /= np.sqrt(np.sum(D**2, axis=1))[:, np.newaxis]
        return D

    transform_algorithm = "lasso_cd"
    resolution = 1024
    subsampling = 3  # subsampling factor
    n_components = resolution // subsampling

    # Compute a wavelet dictionary
    D_multi = np.r_[
        tuple(
            ricker_matrix(
                width=w, resolution=resolution, n_components=n_components // 5
            )
            for w in (10, 50, 100, 500, 1000)
        )
    ]

    X = np.linspace(0, resolution - 1, resolution)
    first_quarter = X < resolution / 4
    X[first_quarter] = 3.0
    X[np.logical_not(first_quarter)] = -1.0
    X = X.reshape(1, -1)

    # check that the underlying model fails to converge
    with pytest.warns(ConvergenceWarning):
        model = SparseCoder(
            D_multi, transform_algorithm=transform_algorithm, transform_max_iter=1
        )
        model.fit_transform(X)

    # check that the underlying model converges w/o warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model = SparseCoder(
            D_multi, transform_algorithm=transform_algorithm, transform_max_iter=2000
        )
        model.fit_transform(X)


def test_dict_learning_lars_positive_parameter():
    n_components = 5
    alpha = 1
    err_msg = "Positive constraint not supported for 'lars' coding method."
    with pytest.raises(ValueError, match=err_msg):
        dict_learning(X, n_components, alpha=alpha, positive_code=True)


@pytest.mark.parametrize(
    "transform_algorithm",
    [
        "lasso_lars",
        "lasso_cd",
        "threshold",
    ],
)
@pytest.mark.parametrize("positive_code", [False, True])
@pytest.mark.parametrize("positive_dict", [False, True])
def test_dict_learning_positivity(transform_algorithm, positive_code, positive_dict):
    n_components = 5
    dico = DictionaryLearning(
        n_components,
        transform_algorithm=transform_algorithm,
        random_state=0,
        positive_code=positive_code,
        positive_dict=positive_dict,
        fit_algorithm="cd",
    ).fit(X)

    code = dico.transform(X)
    if positive_dict:
        assert (dico.components_ >= 0).all()
    else:
        assert (dico.components_ < 0).any()
    if positive_code:
        assert (code >= 0).all()
    else:
        assert (code < 0).any()


@pytest.mark.parametrize("positive_dict", [False, True])
def test_dict_learning_lars_dict_positivity(positive_dict):
    n_components = 5
    dico = DictionaryLearning(
        n_components,
        transform_algorithm="lars",
        random_state=0,
        positive_dict=positive_dict,
        fit_algorithm="cd",
    ).fit(X)

    if positive_dict:
        assert (dico.components_ >= 0).all()
    else:
        assert (dico.components_ < 0).any()


def test_dict_learning_lars_code_positivity():
    n_components = 5
    dico = DictionaryLearning(
        n_components,
        transform_algorithm="lars",
        random_state=0,
        positive_code=True,
        fit_algorithm="cd",
    ).fit(X)

    err_msg = "Positive constraint not supported for '{}' coding method."
    err_msg = err_msg.format("lars")
    with pytest.raises(ValueError, match=err_msg):
        dico.transform(X)


def test_dict_learning_reconstruction():
    n_components = 12
    dico = DictionaryLearning(
        n_components, transform_algorithm="omp", transform_alpha=0.001, random_state=0
    )
    code = dico.fit(X).transform(X)
    assert_array_almost_equal(np.dot(code, dico.components_), X)

    dico.set_params(transform_algorithm="lasso_lars")
    code = dico.transform(X)
    assert_array_almost_equal(np.dot(code, dico.components_), X, decimal=2)

    # used to test lars here too, but there's no guarantee the number of
    # nonzero atoms is right.


def test_dict_learning_reconstruction_parallel():
    # regression test that parallel reconstruction works with n_jobs>1
    n_components = 12
    dico = DictionaryLearning(
        n_components,
        transform_algorithm="omp",
        transform_alpha=0.001,
        random_state=0,
        n_jobs=4,
    )
    code = dico.fit(X).transform(X)
    assert_array_almost_equal(np.dot(code, dico.components_), X)

    dico.set_params(transform_algorithm="lasso_lars")
    code = dico.transform(X)
    assert_array_almost_equal(np.dot(code, dico.components_), X, decimal=2)


def test_dict_learning_lassocd_readonly_data():
    n_components = 12
    with TempMemmap(X) as X_read_only:
        dico = DictionaryLearning(
            n_components,
            transform_algorithm="lasso_cd",
            transform_alpha=0.001,
            random_state=0,
            n_jobs=4,
        )
        with ignore_warnings(category=ConvergenceWarning):
            code = dico.fit(X_read_only).transform(X_read_only)
        assert_array_almost_equal(
            np.dot(code, dico.components_), X_read_only, decimal=2
        )


def test_dict_learning_nonzero_coefs():
    n_components = 4
    dico = DictionaryLearning(
        n_components,
        transform_algorithm="lars",
        transform_n_nonzero_coefs=3,
        random_state=0,
    )
    code = dico.fit(X).transform(X[np.newaxis, 1])
    assert len(np.flatnonzero(code)) == 3

    dico.set_params(transform_algorithm="omp")
    code = dico.transform(X[np.newaxis, 1])
    assert len(np.flatnonzero(code)) == 3


def test_dict_learning_split():
    n_components = 5
    dico = DictionaryLearning(
        n_components, transform_algorithm="threshold", random_state=0
    )
    code = dico.fit(X).transform(X)
    dico.split_sign = True
    split_code = dico.transform(X)

    assert_array_almost_equal(
        split_code[:, :n_components] - split_code[:, n_components:], code
    )


def test_dict_learning_online_shapes():
    rng = np.random.RandomState(0)
    n_components = 8

    code, dictionary = dict_learning_online(
        X,
        n_components=n_components,
        batch_size=4,
        max_iter=10,
        method="cd",
        random_state=rng,
        return_code=True,
    )
    assert code.shape == (n_samples, n_components)
    assert dictionary.shape == (n_components, n_features)
    assert np.dot(code, dictionary).shape == X.shape

    dictionary = dict_learning_online(
        X,
        n_components=n_components,
        batch_size=4,
        max_iter=10,
        method="cd",
        random_state=rng,
        return_code=False,
    )
    assert dictionary.shape == (n_components, n_features)


def test_dict_learning_online_lars_positive_parameter():
    err_msg = "Positive constraint not supported for 'lars' coding method."
    with pytest.raises(ValueError, match=err_msg):
        dict_learning_online(X, batch_size=4, max_iter=10, positive_code=True)


@pytest.mark.parametrize(
    "transform_algorithm",
    [
        "lasso_lars",
        "lasso_cd",
        "threshold",
    ],
)
@pytest.mark.parametrize("positive_code", [False, True])
@pytest.mark.parametrize("positive_dict", [False, True])
def test_minibatch_dictionary_learning_positivity(
    transform_algorithm, positive_code, positive_dict
):
    n_components = 8
    dico = MiniBatchDictionaryLearning(
        n_components,
        batch_size=4,
        max_iter=10,
        transform_algorithm=transform_algorithm,
        random_state=0,
        positive_code=positive_code,
        positive_dict=positive_dict,
        fit_algorithm="cd",
    ).fit(X)

    code = dico.transform(X)
    if positive_dict:
        assert (dico.components_ >= 0).all()
    else:
        assert (dico.components_ < 0).any()
    if positive_code:
        assert (code >= 0).all()
    else:
        assert (code < 0).any()


@pytest.mark.parametrize("positive_dict", [False, True])
def test_minibatch_dictionary_learning_lars(positive_dict):
    n_components = 8

    dico = MiniBatchDictionaryLearning(
        n_components,
        batch_size=4,
        max_iter=10,
        transform_algorithm="lars",
        random_state=0,
        positive_dict=positive_dict,
        fit_algorithm="cd",
    ).fit(X)

    if positive_dict:
        assert (dico.components_ >= 0).all()
    else:
        assert (dico.components_ < 0).any()


@pytest.mark.parametrize("positive_code", [False, True])
@pytest.mark.parametrize("positive_dict", [False, True])
def test_dict_learning_online_positivity(positive_code, positive_dict):
    rng = np.random.RandomState(0)
    n_components = 8

    code, dictionary = dict_learning_online(
        X,
        n_components=n_components,
        batch_size=4,
        method="cd",
        alpha=1,
        random_state=rng,
        positive_dict=positive_dict,
        positive_code=positive_code,
    )
    if positive_dict:
        assert (dictionary >= 0).all()
    else:
        assert (dictionary < 0).any()
    if positive_code:
        assert (code >= 0).all()
    else:
        assert (code < 0).any()


def test_dict_learning_online_verbosity():
    # test verbosity for better coverage
    n_components = 5
    import sys
    from io import StringIO

    old_stdout = sys.stdout
    try:
        sys.stdout = StringIO()

        # convergence monitoring verbosity
        dico = MiniBatchDictionaryLearning(
            n_components, batch_size=4, max_iter=5, verbose=1, tol=0.1, random_state=0
        )
        dico.fit(X)
        dico = MiniBatchDictionaryLearning(
            n_components,
            batch_size=4,
            max_iter=5,
            verbose=1,
            max_no_improvement=2,
            random_state=0,
        )
        dico.fit(X)
        # higher verbosity level
        dico = MiniBatchDictionaryLearning(
            n_components, batch_size=4, max_iter=5, verbose=2, random_state=0
        )
        dico.fit(X)

        # function API verbosity
        dict_learning_online(
            X,
            n_components=n_components,
            batch_size=4,
            alpha=1,
            verbose=1,
            random_state=0,
        )
        dict_learning_online(
            X,
            n_components=n_components,
            batch_size=4,
            alpha=1,
            verbose=2,
            random_state=0,
        )
    finally:
        sys.stdout = old_stdout

    assert dico.components_.shape == (n_components, n_features)


def test_dict_learning_online_estimator_shapes():
    n_components = 5
    dico = MiniBatchDictionaryLearning(
        n_components, batch_size=4, max_iter=5, random_state=0
    )
    dico.fit(X)
    assert dico.components_.shape == (n_components, n_features)


def test_dict_learning_online_overcomplete():
    n_components = 12
    dico = MiniBatchDictionaryLearning(
        n_components, batch_size=4, max_iter=5, random_state=0
    ).fit(X)
    assert dico.components_.shape == (n_components, n_features)


def test_dict_learning_online_initialization():
    n_components = 12
    rng = np.random.RandomState(0)
    V = rng.randn(n_components, n_features)
    dico = MiniBatchDictionaryLearning(
        n_components, batch_size=4, max_iter=0, dict_init=V, random_state=0
    ).fit(X)
    assert_array_equal(dico.components_, V)


def test_dict_learning_online_readonly_initialization():
    n_components = 12
    rng = np.random.RandomState(0)
    V = rng.randn(n_components, n_features)
    V.setflags(write=False)
    MiniBatchDictionaryLearning(
        n_components,
        batch_size=4,
        max_iter=1,
        dict_init=V,
        random_state=0,
        shuffle=False,
    ).fit(X)


def test_dict_learning_online_partial_fit():
    n_components = 12
    rng = np.random.RandomState(0)
    V = rng.randn(n_components, n_features)  # random init
    V /= np.sum(V**2, axis=1)[:, np.newaxis]
    dict1 = MiniBatchDictionaryLearning(
        n_components,
        max_iter=10,
        batch_size=1,
        alpha=1,
        shuffle=False,
        dict_init=V,
        max_no_improvement=None,
        tol=0.0,
        random_state=0,
    ).fit(X)
    dict2 = MiniBatchDictionaryLearning(
        n_components, alpha=1, dict_init=V, random_state=0
    )
    for i in range(10):
        for sample in X:
            dict2.partial_fit(sample[np.newaxis, :])

    assert not np.all(sparse_encode(X, dict1.components_, alpha=1) == 0)
    assert_array_almost_equal(dict1.components_, dict2.components_, decimal=2)

    # partial_fit should ignore max_iter (#17433)
    assert dict1.n_steps_ == dict2.n_steps_ == 100


def test_sparse_encode_shapes():
    n_components = 12
    rng = np.random.RandomState(0)
    V = rng.randn(n_components, n_features)  # random init
    V /= np.sum(V**2, axis=1)[:, np.newaxis]
    for algo in ("lasso_lars", "lasso_cd", "lars", "omp", "threshold"):
        code = sparse_encode(X, V, algorithm=algo)
        assert code.shape == (n_samples, n_components)


@pytest.mark.parametrize("algo", ["lasso_lars", "lasso_cd", "threshold"])
@pytest.mark.parametrize("positive", [False, True])
def test_sparse_encode_positivity(algo, positive):
    n_components = 12
    rng = np.random.RandomState(0)
    V = rng.randn(n_components, n_features)  # random init
    V /= np.sum(V**2, axis=1)[:, np.newaxis]
    code = sparse_encode(X, V, algorithm=algo, positive=positive)
    if positive:
        assert (code >= 0).all()
    else:
        assert (code < 0).any()


@pytest.mark.parametrize("algo", ["lars", "omp"])
def test_sparse_encode_unavailable_positivity(algo):
    n_components = 12
    rng = np.random.RandomState(0)
    V = rng.randn(n_components, n_features)  # random init
    V /= np.sum(V**2, axis=1)[:, np.newaxis]
    err_msg = "Positive constraint not supported for '{}' coding method."
    err_msg = err_msg.format(algo)
    with pytest.raises(ValueError, match=err_msg):
        sparse_encode(X, V, algorithm=algo, positive=True)


def test_sparse_encode_input():
    n_components = 100
    rng = np.random.RandomState(0)
    V = rng.randn(n_components, n_features)  # random init
    V /= np.sum(V**2, axis=1)[:, np.newaxis]
    Xf = check_array(X, order="F")
    for algo in ("lasso_lars", "lasso_cd", "lars", "omp", "threshold"):
        a = sparse_encode(X, V, algorithm=algo)
        b = sparse_encode(Xf, V, algorithm=algo)
        assert_array_almost_equal(a, b)


def test_sparse_encode_error():
    n_components = 12
    rng = np.random.RandomState(0)
    V = rng.randn(n_components, n_features)  # random init
    V /= np.sum(V**2, axis=1)[:, np.newaxis]
    code = sparse_encode(X, V, alpha=0.001)
    assert not np.all(code == 0)
    assert np.sqrt(np.sum((np.dot(code, V) - X) ** 2)) < 0.1


def test_sparse_encode_error_default_sparsity():
    rng = np.random.RandomState(0)
    X = rng.randn(100, 64)
    D = rng.randn(2, 64)
    code = ignore_warnings(sparse_encode)(X, D, algorithm="omp", n_nonzero_coefs=None)
    assert code.shape == (100, 2)


def test_sparse_coder_estimator():
    n_components = 12
    rng = np.random.RandomState(0)
    V = rng.randn(n_components, n_features)  # random init
    V /= np.sum(V**2, axis=1)[:, np.newaxis]
    coder = SparseCoder(
        dictionary=V, transform_algorithm="lasso_lars", transform_alpha=0.001
    ).transform(X)
    assert not np.all(coder == 0)
    assert np.sqrt(np.sum((np.dot(coder, V) - X) ** 2)) < 0.1


def test_sparse_coder_estimator_clone():
    n_components = 12
    rng = np.random.RandomState(0)
    V = rng.randn(n_components, n_features)  # random init
    V /= np.sum(V**2, axis=1)[:, np.newaxis]
    coder = SparseCoder(
        dictionary=V, transform_algorithm="lasso_lars", transform_alpha=0.001
    )
    cloned = clone(coder)
    assert id(cloned) != id(coder)
    np.testing.assert_allclose(cloned.dictionary, coder.dictionary)
    assert id(cloned.dictionary) != id(coder.dictionary)
    assert cloned.n_components_ == coder.n_components_
    assert cloned.n_features_in_ == coder.n_features_in_
    data = np.random.rand(n_samples, n_features).astype(np.float32)
    np.testing.assert_allclose(cloned.transform(data), coder.transform(data))


def test_sparse_coder_parallel_mmap():
    # Non-regression test for:
    # https://github.com/scikit-learn/scikit-learn/issues/5956
    # Test that SparseCoder does not error by passing reading only
    # arrays to child processes

    rng = np.random.RandomState(777)
    n_components, n_features = 40, 64
    init_dict = rng.rand(n_components, n_features)
    # Ensure that `data` is >2M. Joblib memory maps arrays
    # if they are larger than 1MB. The 4 accounts for float32
    # data type
    n_samples = int(2e6) // (4 * n_features)
    data = np.random.rand(n_samples, n_features).astype(np.float32)

    sc = SparseCoder(init_dict, transform_algorithm="omp", n_jobs=2)
    sc.fit_transform(data)


def test_sparse_coder_common_transformer():
    rng = np.random.RandomState(777)
    n_components, n_features = 40, 3
    init_dict = rng.rand(n_components, n_features)

    sc = SparseCoder(init_dict)

    check_transformer_data_not_an_array(sc.__class__.__name__, sc)
    check_transformer_general(sc.__class__.__name__, sc)
    check_transformer_general_memmap = partial(
        check_transformer_general, readonly_memmap=True
    )
    check_transformer_general_memmap(sc.__class__.__name__, sc)
    check_transformers_unfitted(sc.__class__.__name__, sc)


def test_sparse_coder_n_features_in():
    d = np.array([[1, 2, 3], [1, 2, 3]])
    sc = SparseCoder(d)
    assert sc.n_features_in_ == d.shape[1]


def test_minibatch_dict_learning_n_iter_deprecated():
    # check the deprecation warning of n_iter
    # TODO(1.4) remove
    depr_msg = (
        "'n_iter' is deprecated in version 1.1 and will be removed in version 1.4"
    )
    est = MiniBatchDictionaryLearning(
        n_components=2, batch_size=4, n_iter=5, random_state=0
    )

    with pytest.warns(FutureWarning, match=depr_msg):
        est.fit(X)


@pytest.mark.parametrize(
    "arg, val",
    [
        ("iter_offset", 0),
        ("inner_stats", None),
        ("return_inner_stats", False),
        ("return_n_iter", False),
        ("n_iter", 5),
    ],
)
def test_dict_learning_online_deprecated_args(arg, val):
    # check the deprecation warning for the deprecated args of
    # dict_learning_online
    # TODO(1.4) remove
    depr_msg = (
        f"'{arg}' is deprecated in version 1.1 and will be removed in version 1.4."
    )

    with pytest.warns(FutureWarning, match=depr_msg):
        dict_learning_online(
            X, n_components=2, batch_size=4, random_state=0, **{arg: val}
        )


def test_update_dict():
    # Check the dict update in batch mode vs online mode
    # Non-regression test for #4866
    rng = np.random.RandomState(0)

    code = np.array([[0.5, -0.5], [0.1, 0.9]])
    dictionary = np.array([[1.0, 0.0], [0.6, 0.8]])

    X = np.dot(code, dictionary) + rng.randn(2, 2)

    # full batch update
    newd_batch = dictionary.copy()
    _update_dict(newd_batch, X, code)

    # online update
    A = np.dot(code.T, code)
    B = np.dot(X.T, code)
    newd_online = dictionary.copy()
    _update_dict(newd_online, X, code, A, B)

    assert_allclose(newd_batch, newd_online)


# TODO(1.4) remove
def test_dict_learning_online_n_iter_deprecated():
    # Check that an error is raised when a deprecated argument is set when max_iter
    # is also set.
    msg = "The following arguments are incompatible with 'max_iter'"
    with pytest.raises(ValueError, match=msg):
        dict_learning_online(X, max_iter=10, return_inner_stats=True)


@pytest.mark.parametrize(
    "algorithm", ("lasso_lars", "lasso_cd", "lars", "threshold", "omp")
)
@pytest.mark.parametrize("data_type", (np.float32, np.float64))
# Note: do not check integer input because `lasso_lars` and `lars` fail with
# `ValueError` in `_lars_path_solver`
def test_sparse_encode_dtype_match(data_type, algorithm):
    n_components = 6
    rng = np.random.RandomState(0)
    dictionary = rng.randn(n_components, n_features)
    code = sparse_encode(
        X.astype(data_type), dictionary.astype(data_type), algorithm=algorithm
    )
    assert code.dtype == data_type


@pytest.mark.parametrize(
    "algorithm", ("lasso_lars", "lasso_cd", "lars", "threshold", "omp")
)
def test_sparse_encode_numerical_consistency(algorithm):
    # verify numerical consistency among np.float32 and np.float64
    rtol = 1e-4
    n_components = 6
    rng = np.random.RandomState(0)
    dictionary = rng.randn(n_components, n_features)
    code_32 = sparse_encode(
        X.astype(np.float32), dictionary.astype(np.float32), algorithm=algorithm
    )
    code_64 = sparse_encode(
        X.astype(np.float64), dictionary.astype(np.float64), algorithm=algorithm
    )
    assert_allclose(code_32, code_64, rtol=rtol)


@pytest.mark.parametrize(
    "transform_algorithm", ("lasso_lars", "lasso_cd", "lars", "threshold", "omp")
)
@pytest.mark.parametrize("data_type", (np.float32, np.float64))
# Note: do not check integer input because `lasso_lars` and `lars` fail with
# `ValueError` in `_lars_path_solver`
def test_sparse_coder_dtype_match(data_type, transform_algorithm):
    # Verify preserving dtype for transform in sparse coder
    n_components = 6
    rng = np.random.RandomState(0)
    dictionary = rng.randn(n_components, n_features)
    coder = SparseCoder(
        dictionary.astype(data_type), transform_algorithm=transform_algorithm
    )
    code = coder.transform(X.astype(data_type))
    assert code.dtype == data_type


@pytest.mark.parametrize("fit_algorithm", ("lars", "cd"))
@pytest.mark.parametrize(
    "transform_algorithm", ("lasso_lars", "lasso_cd", "lars", "threshold", "omp")
)
@pytest.mark.parametrize(
    "data_type, expected_type",
    (
        (np.float32, np.float32),
        (np.float64, np.float64),
        (np.int32, np.float64),
        (np.int64, np.float64),
    ),
)
def test_dictionary_learning_dtype_match(
    data_type,
    expected_type,
    fit_algorithm,
    transform_algorithm,
):
    # Verify preserving dtype for fit and transform in dictionary learning class
    dict_learner = DictionaryLearning(
        n_components=8,
        fit_algorithm=fit_algorithm,
        transform_algorithm=transform_algorithm,
        random_state=0,
    )
    dict_learner.fit(X.astype(data_type))
    assert dict_learner.components_.dtype == expected_type
    assert dict_learner.transform(X.astype(data_type)).dtype == expected_type


@pytest.mark.parametrize("fit_algorithm", ("lars", "cd"))
@pytest.mark.parametrize(
    "transform_algorithm", ("lasso_lars", "lasso_cd", "lars", "threshold", "omp")
)
@pytest.mark.parametrize(
    "data_type, expected_type",
    (
        (np.float32, np.float32),
        (np.float64, np.float64),
        (np.int32, np.float64),
        (np.int64, np.float64),
    ),
)
def test_minibatch_dictionary_learning_dtype_match(
    data_type,
    expected_type,
    fit_algorithm,
    transform_algorithm,
):
    # Verify preserving dtype for fit and transform in minibatch dictionary learning
    dict_learner = MiniBatchDictionaryLearning(
        n_components=8,
        batch_size=10,
        fit_algorithm=fit_algorithm,
        transform_algorithm=transform_algorithm,
        max_iter=100,
        tol=1e-1,
        random_state=0,
    )
    dict_learner.fit(X.astype(data_type))

    assert dict_learner.components_.dtype == expected_type
    assert dict_learner.transform(X.astype(data_type)).dtype == expected_type
    assert dict_learner._A.dtype == expected_type
    assert dict_learner._B.dtype == expected_type


@pytest.mark.parametrize("method", ("lars", "cd"))
@pytest.mark.parametrize(
    "data_type, expected_type",
    (
        (np.float32, np.float32),
        (np.float64, np.float64),
        (np.int32, np.float64),
        (np.int64, np.float64),
    ),
)
def test_dict_learning_dtype_match(data_type, expected_type, method):
    # Verify output matrix dtype
    rng = np.random.RandomState(0)
    n_components = 8
    code, dictionary, _ = dict_learning(
        X.astype(data_type),
        n_components=n_components,
        alpha=1,
        random_state=rng,
        method=method,
    )
    assert code.dtype == expected_type
    assert dictionary.dtype == expected_type


@pytest.mark.parametrize("method", ("lars", "cd"))
def test_dict_learning_numerical_consistency(method):
    # verify numerically consistent among np.float32 and np.float64
    rtol = 1e-6
    n_components = 4
    alpha = 2

    U_64, V_64, _ = dict_learning(
        X.astype(np.float64),
        n_components=n_components,
        alpha=alpha,
        random_state=0,
        method=method,
    )
    U_32, V_32, _ = dict_learning(
        X.astype(np.float32),
        n_components=n_components,
        alpha=alpha,
        random_state=0,
        method=method,
    )

    # Optimal solution (U*, V*) is not unique.
    # If (U*, V*) is optimal solution, (-U*,-V*) is also optimal,
    # and (column permutated U*, row permutated V*) are also optional
    # as long as holding UV.
    # So here UV, ||U||_1,1 and sum(||V_k||_2^2) are verified
    # instead of comparing directly U and V.
    assert_allclose(np.matmul(U_64, V_64), np.matmul(U_32, V_32), rtol=rtol)
    assert_allclose(np.sum(np.abs(U_64)), np.sum(np.abs(U_32)), rtol=rtol)
    assert_allclose(np.sum(V_64**2), np.sum(V_32**2), rtol=rtol)
    # verify an obtained solution is not degenerate
    assert np.mean(U_64 != 0.0) > 0.05
    assert np.count_nonzero(U_64 != 0.0) == np.count_nonzero(U_32 != 0.0)


@pytest.mark.parametrize("method", ("lars", "cd"))
@pytest.mark.parametrize(
    "data_type, expected_type",
    (
        (np.float32, np.float32),
        (np.float64, np.float64),
        (np.int32, np.float64),
        (np.int64, np.float64),
    ),
)
def test_dict_learning_online_dtype_match(data_type, expected_type, method):
    # Verify output matrix dtype
    rng = np.random.RandomState(0)
    n_components = 8
    code, dictionary = dict_learning_online(
        X.astype(data_type),
        n_components=n_components,
        alpha=1,
        batch_size=10,
        random_state=rng,
        method=method,
    )
    assert code.dtype == expected_type
    assert dictionary.dtype == expected_type


@pytest.mark.parametrize("method", ("lars", "cd"))
def test_dict_learning_online_numerical_consistency(method):
    # verify numerically consistent among np.float32 and np.float64
    rtol = 1e-4
    n_components = 4
    alpha = 1

    U_64, V_64 = dict_learning_online(
        X.astype(np.float64),
        n_components=n_components,
        alpha=alpha,
        batch_size=10,
        random_state=0,
        method=method,
    )
    U_32, V_32 = dict_learning_online(
        X.astype(np.float32),
        n_components=n_components,
        alpha=alpha,
        batch_size=10,
        random_state=0,
        method=method,
    )

    # Optimal solution (U*, V*) is not unique.
    # If (U*, V*) is optimal solution, (-U*,-V*) is also optimal,
    # and (column permutated U*, row permutated V*) are also optional
    # as long as holding UV.
    # So here UV, ||U||_1,1 and sum(||V_k||_2) are verified
    # instead of comparing directly U and V.
    assert_allclose(np.matmul(U_64, V_64), np.matmul(U_32, V_32), rtol=rtol)
    assert_allclose(np.sum(np.abs(U_64)), np.sum(np.abs(U_32)), rtol=rtol)
    assert_allclose(np.sum(V_64**2), np.sum(V_32**2), rtol=rtol)
    # verify an obtained solution is not degenerate
    assert np.mean(U_64 != 0.0) > 0.05
    assert np.count_nonzero(U_64 != 0.0) == np.count_nonzero(U_32 != 0.0)


@pytest.mark.parametrize(
    "estimator",
    [
        SparseCoder(X.T),
        DictionaryLearning(),
        MiniBatchDictionaryLearning(batch_size=4, max_iter=10),
    ],
    ids=lambda x: x.__class__.__name__,
)
def test_get_feature_names_out(estimator):
    """Check feature names for dict learning estimators."""
    estimator.fit(X)
    n_components = X.shape[1]

    feature_names_out = estimator.get_feature_names_out()
    estimator_name = estimator.__class__.__name__.lower()
    assert_array_equal(
        feature_names_out,
        [f"{estimator_name}{i}" for i in range(n_components)],
    )


def test_cd_work_on_joblib_memmapped_data(monkeypatch):
    monkeypatch.setattr(
        sklearn.decomposition._dict_learning,
        "Parallel",
        partial(Parallel, max_nbytes=100),
    )

    rng = np.random.RandomState(0)
    X_train = rng.randn(10, 10)

    dict_learner = DictionaryLearning(
        n_components=5,
        random_state=0,
        n_jobs=2,
        fit_algorithm="cd",
        max_iter=50,
        verbose=True,
    )

    # This must run and complete without error.
    dict_learner.fit(X_train)


# TODO(1.4) remove
def test_minibatch_dictionary_learning_warns_and_ignore_n_iter():
    """Check that we always raise a warning when `n_iter` is set even if it is
    ignored if `max_iter` is set.
    """
    warn_msg = "'n_iter' is deprecated in version 1.1"
    with pytest.warns(FutureWarning, match=warn_msg):
        model = MiniBatchDictionaryLearning(batch_size=256, n_iter=2, max_iter=2).fit(X)
    assert model.n_iter_ == 2
