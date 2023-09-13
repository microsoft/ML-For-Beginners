"""
Python implementation of the fast ICA algorithms.

Reference: Tables 8.3 and 8.4 page 196 in the book:
Independent Component Analysis, by  Hyvarinen et al.
"""

# Authors: Pierre Lafaye de Micheaux, Stefan van der Walt, Gael Varoquaux,
#          Bertrand Thirion, Alexandre Gramfort, Denis A. Engemann
# License: BSD 3 clause

import warnings
from numbers import Integral, Real

import numpy as np
from scipy import linalg

from ..base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
    _fit_context,
)
from ..exceptions import ConvergenceWarning
from ..utils import as_float_array, check_array, check_random_state
from ..utils._param_validation import Interval, Options, StrOptions, validate_params
from ..utils.validation import check_is_fitted

__all__ = ["fastica", "FastICA"]


def _gs_decorrelation(w, W, j):
    """
    Orthonormalize w wrt the first j rows of W.

    Parameters
    ----------
    w : ndarray of shape (n,)
        Array to be orthogonalized

    W : ndarray of shape (p, n)
        Null space definition

    j : int < p
        The no of (from the first) rows of Null space W wrt which w is
        orthogonalized.

    Notes
    -----
    Assumes that W is orthogonal
    w changed in place
    """
    w -= np.linalg.multi_dot([w, W[:j].T, W[:j]])
    return w


def _sym_decorrelation(W):
    """Symmetric decorrelation
    i.e. W <- (W * W.T) ^{-1/2} * W
    """
    s, u = linalg.eigh(np.dot(W, W.T))
    # Avoid sqrt of negative values because of rounding errors. Note that
    # np.sqrt(tiny) is larger than tiny and therefore this clipping also
    # prevents division by zero in the next step.
    s = np.clip(s, a_min=np.finfo(W.dtype).tiny, a_max=None)

    # u (resp. s) contains the eigenvectors (resp. square roots of
    # the eigenvalues) of W * W.T
    return np.linalg.multi_dot([u * (1.0 / np.sqrt(s)), u.T, W])


def _ica_def(X, tol, g, fun_args, max_iter, w_init):
    """Deflationary FastICA using fun approx to neg-entropy function

    Used internally by FastICA.
    """

    n_components = w_init.shape[0]
    W = np.zeros((n_components, n_components), dtype=X.dtype)
    n_iter = []

    # j is the index of the extracted component
    for j in range(n_components):
        w = w_init[j, :].copy()
        w /= np.sqrt((w**2).sum())

        for i in range(max_iter):
            gwtx, g_wtx = g(np.dot(w.T, X), fun_args)

            w1 = (X * gwtx).mean(axis=1) - g_wtx.mean() * w

            _gs_decorrelation(w1, W, j)

            w1 /= np.sqrt((w1**2).sum())

            lim = np.abs(np.abs((w1 * w).sum()) - 1)
            w = w1
            if lim < tol:
                break

        n_iter.append(i + 1)
        W[j, :] = w

    return W, max(n_iter)


def _ica_par(X, tol, g, fun_args, max_iter, w_init):
    """Parallel FastICA.

    Used internally by FastICA --main loop

    """
    W = _sym_decorrelation(w_init)
    del w_init
    p_ = float(X.shape[1])
    for ii in range(max_iter):
        gwtx, g_wtx = g(np.dot(W, X), fun_args)
        W1 = _sym_decorrelation(np.dot(gwtx, X.T) / p_ - g_wtx[:, np.newaxis] * W)
        del gwtx, g_wtx
        # builtin max, abs are faster than numpy counter parts.
        # np.einsum allows having the lowest memory footprint.
        # It is faster than np.diag(np.dot(W1, W.T)).
        lim = max(abs(abs(np.einsum("ij,ij->i", W1, W)) - 1))
        W = W1
        if lim < tol:
            break
    else:
        warnings.warn(
            (
                "FastICA did not converge. Consider increasing "
                "tolerance or the maximum number of iterations."
            ),
            ConvergenceWarning,
        )

    return W, ii + 1


# Some standard non-linear functions.
# XXX: these should be optimized, as they can be a bottleneck.
def _logcosh(x, fun_args=None):
    alpha = fun_args.get("alpha", 1.0)  # comment it out?

    x *= alpha
    gx = np.tanh(x, x)  # apply the tanh inplace
    g_x = np.empty(x.shape[0], dtype=x.dtype)
    # XXX compute in chunks to avoid extra allocation
    for i, gx_i in enumerate(gx):  # please don't vectorize.
        g_x[i] = (alpha * (1 - gx_i**2)).mean()
    return gx, g_x


def _exp(x, fun_args):
    exp = np.exp(-(x**2) / 2)
    gx = x * exp
    g_x = (1 - x**2) * exp
    return gx, g_x.mean(axis=-1)


def _cube(x, fun_args):
    return x**3, (3 * x**2).mean(axis=-1)


@validate_params(
    {
        "X": ["array-like"],
        "return_X_mean": ["boolean"],
        "compute_sources": ["boolean"],
        "return_n_iter": ["boolean"],
    },
    prefer_skip_nested_validation=False,
)
def fastica(
    X,
    n_components=None,
    *,
    algorithm="parallel",
    whiten="unit-variance",
    fun="logcosh",
    fun_args=None,
    max_iter=200,
    tol=1e-04,
    w_init=None,
    whiten_solver="svd",
    random_state=None,
    return_X_mean=False,
    compute_sources=True,
    return_n_iter=False,
):
    """Perform Fast Independent Component Analysis.

    The implementation is based on [1]_.

    Read more in the :ref:`User Guide <ICA>`.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training vector, where `n_samples` is the number of samples and
        `n_features` is the number of features.

    n_components : int, default=None
        Number of components to use. If None is passed, all are used.

    algorithm : {'parallel', 'deflation'}, default='parallel'
        Specify which algorithm to use for FastICA.

    whiten : str or bool, default='unit-variance'
        Specify the whitening strategy to use.

        - If 'arbitrary-variance', a whitening with variance
          arbitrary is used.
        - If 'unit-variance', the whitening matrix is rescaled to ensure that
          each recovered source has unit variance.
        - If False, the data is already considered to be whitened, and no
          whitening is performed.

        .. versionchanged:: 1.3
            The default value of `whiten` changed to 'unit-variance' in 1.3.

    fun : {'logcosh', 'exp', 'cube'} or callable, default='logcosh'
        The functional form of the G function used in the
        approximation to neg-entropy. Could be either 'logcosh', 'exp',
        or 'cube'.
        You can also provide your own function. It should return a tuple
        containing the value of the function, and of its derivative, in the
        point. The derivative should be averaged along its last dimension.
        Example::

            def my_g(x):
                return x ** 3, (3 * x ** 2).mean(axis=-1)

    fun_args : dict, default=None
        Arguments to send to the functional form.
        If empty or None and if fun='logcosh', fun_args will take value
        {'alpha' : 1.0}.

    max_iter : int, default=200
        Maximum number of iterations to perform.

    tol : float, default=1e-4
        A positive scalar giving the tolerance at which the
        un-mixing matrix is considered to have converged.

    w_init : ndarray of shape (n_components, n_components), default=None
        Initial un-mixing array. If `w_init=None`, then an array of values
        drawn from a normal distribution is used.

    whiten_solver : {"eigh", "svd"}, default="svd"
        The solver to use for whitening.

        - "svd" is more stable numerically if the problem is degenerate, and
          often faster when `n_samples <= n_features`.

        - "eigh" is generally more memory efficient when
          `n_samples >= n_features`, and can be faster when
          `n_samples >= 50 * n_features`.

        .. versionadded:: 1.2

    random_state : int, RandomState instance or None, default=None
        Used to initialize ``w_init`` when not specified, with a
        normal distribution. Pass an int, for reproducible results
        across multiple function calls.
        See :term:`Glossary <random_state>`.

    return_X_mean : bool, default=False
        If True, X_mean is returned too.

    compute_sources : bool, default=True
        If False, sources are not computed, but only the rotation matrix.
        This can save memory when working with big data. Defaults to True.

    return_n_iter : bool, default=False
        Whether or not to return the number of iterations.

    Returns
    -------
    K : ndarray of shape (n_components, n_features) or None
        If whiten is 'True', K is the pre-whitening matrix that projects data
        onto the first n_components principal components. If whiten is 'False',
        K is 'None'.

    W : ndarray of shape (n_components, n_components)
        The square matrix that unmixes the data after whitening.
        The mixing matrix is the pseudo-inverse of matrix ``W K``
        if K is not None, else it is the inverse of W.

    S : ndarray of shape (n_samples, n_components) or None
        Estimated source matrix.

    X_mean : ndarray of shape (n_features,)
        The mean over features. Returned only if return_X_mean is True.

    n_iter : int
        If the algorithm is "deflation", n_iter is the
        maximum number of iterations run across all components. Else
        they are just the number of iterations taken to converge. This is
        returned only when return_n_iter is set to `True`.

    Notes
    -----
    The data matrix X is considered to be a linear combination of
    non-Gaussian (independent) components i.e. X = AS where columns of S
    contain the independent components and A is a linear mixing
    matrix. In short ICA attempts to `un-mix' the data by estimating an
    un-mixing matrix W where ``S = W K X.``
    While FastICA was proposed to estimate as many sources
    as features, it is possible to estimate less by setting
    n_components < n_features. It this case K is not a square matrix
    and the estimated A is the pseudo-inverse of ``W K``.

    This implementation was originally made for data of shape
    [n_features, n_samples]. Now the input is transposed
    before the algorithm is applied. This makes it slightly
    faster for Fortran-ordered input.

    References
    ----------
    .. [1] A. Hyvarinen and E. Oja, "Fast Independent Component Analysis",
           Algorithms and Applications, Neural Networks, 13(4-5), 2000,
           pp. 411-430.
    """
    est = FastICA(
        n_components=n_components,
        algorithm=algorithm,
        whiten=whiten,
        fun=fun,
        fun_args=fun_args,
        max_iter=max_iter,
        tol=tol,
        w_init=w_init,
        whiten_solver=whiten_solver,
        random_state=random_state,
    )
    est._validate_params()
    S = est._fit_transform(X, compute_sources=compute_sources)

    if est.whiten in ["unit-variance", "arbitrary-variance"]:
        K = est.whitening_
        X_mean = est.mean_
    else:
        K = None
        X_mean = None

    returned_values = [K, est._unmixing, S]
    if return_X_mean:
        returned_values.append(X_mean)
    if return_n_iter:
        returned_values.append(est.n_iter_)

    return returned_values


class FastICA(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    """FastICA: a fast algorithm for Independent Component Analysis.

    The implementation is based on [1]_.

    Read more in the :ref:`User Guide <ICA>`.

    Parameters
    ----------
    n_components : int, default=None
        Number of components to use. If None is passed, all are used.

    algorithm : {'parallel', 'deflation'}, default='parallel'
        Specify which algorithm to use for FastICA.

    whiten : str or bool, default='unit-variance'
        Specify the whitening strategy to use.

        - If 'arbitrary-variance', a whitening with variance
          arbitrary is used.
        - If 'unit-variance', the whitening matrix is rescaled to ensure that
          each recovered source has unit variance.
        - If False, the data is already considered to be whitened, and no
          whitening is performed.

        .. versionchanged:: 1.3
            The default value of `whiten` changed to 'unit-variance' in 1.3.

    fun : {'logcosh', 'exp', 'cube'} or callable, default='logcosh'
        The functional form of the G function used in the
        approximation to neg-entropy. Could be either 'logcosh', 'exp',
        or 'cube'.
        You can also provide your own function. It should return a tuple
        containing the value of the function, and of its derivative, in the
        point. The derivative should be averaged along its last dimension.
        Example::

            def my_g(x):
                return x ** 3, (3 * x ** 2).mean(axis=-1)

    fun_args : dict, default=None
        Arguments to send to the functional form.
        If empty or None and if fun='logcosh', fun_args will take value
        {'alpha' : 1.0}.

    max_iter : int, default=200
        Maximum number of iterations during fit.

    tol : float, default=1e-4
        A positive scalar giving the tolerance at which the
        un-mixing matrix is considered to have converged.

    w_init : array-like of shape (n_components, n_components), default=None
        Initial un-mixing array. If `w_init=None`, then an array of values
        drawn from a normal distribution is used.

    whiten_solver : {"eigh", "svd"}, default="svd"
        The solver to use for whitening.

        - "svd" is more stable numerically if the problem is degenerate, and
          often faster when `n_samples <= n_features`.

        - "eigh" is generally more memory efficient when
          `n_samples >= n_features`, and can be faster when
          `n_samples >= 50 * n_features`.

        .. versionadded:: 1.2

    random_state : int, RandomState instance or None, default=None
        Used to initialize ``w_init`` when not specified, with a
        normal distribution. Pass an int, for reproducible results
        across multiple function calls.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        The linear operator to apply to the data to get the independent
        sources. This is equal to the unmixing matrix when ``whiten`` is
        False, and equal to ``np.dot(unmixing_matrix, self.whitening_)`` when
        ``whiten`` is True.

    mixing_ : ndarray of shape (n_features, n_components)
        The pseudo-inverse of ``components_``. It is the linear operator
        that maps independent sources to the data.

    mean_ : ndarray of shape(n_features,)
        The mean over features. Only set if `self.whiten` is True.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_iter_ : int
        If the algorithm is "deflation", n_iter is the
        maximum number of iterations run across all components. Else
        they are just the number of iterations taken to converge.

    whitening_ : ndarray of shape (n_components, n_features)
        Only set if whiten is 'True'. This is the pre-whitening matrix
        that projects data onto the first `n_components` principal components.

    See Also
    --------
    PCA : Principal component analysis (PCA).
    IncrementalPCA : Incremental principal components analysis (IPCA).
    KernelPCA : Kernel Principal component analysis (KPCA).
    MiniBatchSparsePCA : Mini-batch Sparse Principal Components Analysis.
    SparsePCA : Sparse Principal Components Analysis (SparsePCA).

    References
    ----------
    .. [1] A. Hyvarinen and E. Oja, Independent Component Analysis:
           Algorithms and Applications, Neural Networks, 13(4-5), 2000,
           pp. 411-430.

    Examples
    --------
    >>> from sklearn.datasets import load_digits
    >>> from sklearn.decomposition import FastICA
    >>> X, _ = load_digits(return_X_y=True)
    >>> transformer = FastICA(n_components=7,
    ...         random_state=0,
    ...         whiten='unit-variance')
    >>> X_transformed = transformer.fit_transform(X)
    >>> X_transformed.shape
    (1797, 7)
    """

    _parameter_constraints: dict = {
        "n_components": [Interval(Integral, 1, None, closed="left"), None],
        "algorithm": [StrOptions({"parallel", "deflation"})],
        "whiten": [
            StrOptions({"arbitrary-variance", "unit-variance"}),
            Options(bool, {False}),
        ],
        "fun": [StrOptions({"logcosh", "exp", "cube"}), callable],
        "fun_args": [dict, None],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0.0, None, closed="left")],
        "w_init": ["array-like", None],
        "whiten_solver": [StrOptions({"eigh", "svd"})],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        n_components=None,
        *,
        algorithm="parallel",
        whiten="unit-variance",
        fun="logcosh",
        fun_args=None,
        max_iter=200,
        tol=1e-4,
        w_init=None,
        whiten_solver="svd",
        random_state=None,
    ):
        super().__init__()
        self.n_components = n_components
        self.algorithm = algorithm
        self.whiten = whiten
        self.fun = fun
        self.fun_args = fun_args
        self.max_iter = max_iter
        self.tol = tol
        self.w_init = w_init
        self.whiten_solver = whiten_solver
        self.random_state = random_state

    def _fit_transform(self, X, compute_sources=False):
        """Fit the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        compute_sources : bool, default=False
            If False, sources are not computes but only the rotation matrix.
            This can save memory when working with big data. Defaults to False.

        Returns
        -------
        S : ndarray of shape (n_samples, n_components) or None
            Sources matrix. `None` if `compute_sources` is `False`.
        """
        XT = self._validate_data(
            X, copy=self.whiten, dtype=[np.float64, np.float32], ensure_min_samples=2
        ).T
        fun_args = {} if self.fun_args is None else self.fun_args
        random_state = check_random_state(self.random_state)

        alpha = fun_args.get("alpha", 1.0)
        if not 1 <= alpha <= 2:
            raise ValueError("alpha must be in [1,2]")

        if self.fun == "logcosh":
            g = _logcosh
        elif self.fun == "exp":
            g = _exp
        elif self.fun == "cube":
            g = _cube
        elif callable(self.fun):

            def g(x, fun_args):
                return self.fun(x, **fun_args)

        n_features, n_samples = XT.shape
        n_components = self.n_components
        if not self.whiten and n_components is not None:
            n_components = None
            warnings.warn("Ignoring n_components with whiten=False.")

        if n_components is None:
            n_components = min(n_samples, n_features)
        if n_components > min(n_samples, n_features):
            n_components = min(n_samples, n_features)
            warnings.warn(
                "n_components is too large: it will be set to %s" % n_components
            )

        if self.whiten:
            # Centering the features of X
            X_mean = XT.mean(axis=-1)
            XT -= X_mean[:, np.newaxis]

            # Whitening and preprocessing by PCA
            if self.whiten_solver == "eigh":
                # Faster when num_samples >> n_features
                d, u = linalg.eigh(XT.dot(X))
                sort_indices = np.argsort(d)[::-1]
                eps = np.finfo(d.dtype).eps
                degenerate_idx = d < eps
                if np.any(degenerate_idx):
                    warnings.warn(
                        "There are some small singular values, using "
                        "whiten_solver = 'svd' might lead to more "
                        "accurate results."
                    )
                d[degenerate_idx] = eps  # For numerical issues
                np.sqrt(d, out=d)
                d, u = d[sort_indices], u[:, sort_indices]
            elif self.whiten_solver == "svd":
                u, d = linalg.svd(XT, full_matrices=False, check_finite=False)[:2]

            # Give consistent eigenvectors for both svd solvers
            u *= np.sign(u[0])

            K = (u / d).T[:n_components]  # see (6.33) p.140
            del u, d
            X1 = np.dot(K, XT)
            # see (13.6) p.267 Here X1 is white and data
            # in X has been projected onto a subspace by PCA
            X1 *= np.sqrt(n_samples)
        else:
            # X must be casted to floats to avoid typing issues with numpy
            # 2.0 and the line below
            X1 = as_float_array(XT, copy=False)  # copy has been taken care of

        w_init = self.w_init
        if w_init is None:
            w_init = np.asarray(
                random_state.normal(size=(n_components, n_components)), dtype=X1.dtype
            )

        else:
            w_init = np.asarray(w_init)
            if w_init.shape != (n_components, n_components):
                raise ValueError(
                    "w_init has invalid shape -- should be %(shape)s"
                    % {"shape": (n_components, n_components)}
                )

        kwargs = {
            "tol": self.tol,
            "g": g,
            "fun_args": fun_args,
            "max_iter": self.max_iter,
            "w_init": w_init,
        }

        if self.algorithm == "parallel":
            W, n_iter = _ica_par(X1, **kwargs)
        elif self.algorithm == "deflation":
            W, n_iter = _ica_def(X1, **kwargs)
        del X1

        self.n_iter_ = n_iter

        if compute_sources:
            if self.whiten:
                S = np.linalg.multi_dot([W, K, XT]).T
            else:
                S = np.dot(W, XT).T
        else:
            S = None

        if self.whiten:
            if self.whiten == "unit-variance":
                if not compute_sources:
                    S = np.linalg.multi_dot([W, K, XT]).T
                S_std = np.std(S, axis=0, keepdims=True)
                S /= S_std
                W /= S_std.T

            self.components_ = np.dot(W, K)
            self.mean_ = X_mean
            self.whitening_ = K
        else:
            self.components_ = W

        self.mixing_ = linalg.pinv(self.components_, check_finite=False)
        self._unmixing = W

        return S

    @_fit_context(prefer_skip_nested_validation=True)
    def fit_transform(self, X, y=None):
        """Fit the model and recover the sources from X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Estimated sources obtained by transforming the data with the
            estimated unmixing matrix.
        """
        return self._fit_transform(X, compute_sources=True)

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Fit the model to X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self._fit_transform(X, compute_sources=False)
        return self

    def transform(self, X, copy=True):
        """Recover the sources from X (apply the unmixing matrix).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        copy : bool, default=True
            If False, data passed to fit can be overwritten. Defaults to True.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Estimated sources obtained by transforming the data with the
            estimated unmixing matrix.
        """
        check_is_fitted(self)

        X = self._validate_data(
            X, copy=(copy and self.whiten), dtype=[np.float64, np.float32], reset=False
        )
        if self.whiten:
            X -= self.mean_

        return np.dot(X, self.components_.T)

    def inverse_transform(self, X, copy=True):
        """Transform the sources back to the mixed data (apply mixing matrix).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_components)
            Sources, where `n_samples` is the number of samples
            and `n_components` is the number of components.
        copy : bool, default=True
            If False, data passed to fit are overwritten. Defaults to True.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_features)
            Reconstructed data obtained with the mixing matrix.
        """
        check_is_fitted(self)

        X = check_array(X, copy=(copy and self.whiten), dtype=[np.float64, np.float32])
        X = np.dot(X, self.mixing_.T)
        if self.whiten:
            X += self.mean_

        return X

    @property
    def _n_features_out(self):
        """Number of transformed output features."""
        return self.components_.shape[0]

    def _more_tags(self):
        return {"preserves_dtype": [np.float32, np.float64]}
