"""
Statespace Tools

Author: Chad Fulton
License: Simplified-BSD
"""
import numpy as np
from scipy.linalg import solve_sylvester
import pandas as pd

from statsmodels.compat.pandas import Appender
from statsmodels.tools.data import _is_using_pandas
from scipy.linalg.blas import find_best_blas_type
from . import (_initialization, _representation, _kalman_filter,
               _kalman_smoother, _simulation_smoother,
               _cfa_simulation_smoother, _tools)


compatibility_mode = False
has_trmm = True
prefix_dtype_map = {
    's': np.float32, 'd': np.float64, 'c': np.complex64, 'z': np.complex128
}
prefix_initialization_map = {
    's': _initialization.sInitialization,
    'd': _initialization.dInitialization,
    'c': _initialization.cInitialization,
    'z': _initialization.zInitialization
}
prefix_statespace_map = {
    's': _representation.sStatespace, 'd': _representation.dStatespace,
    'c': _representation.cStatespace, 'z': _representation.zStatespace
}
prefix_kalman_filter_map = {
    's': _kalman_filter.sKalmanFilter,
    'd': _kalman_filter.dKalmanFilter,
    'c': _kalman_filter.cKalmanFilter,
    'z': _kalman_filter.zKalmanFilter
}
prefix_kalman_smoother_map = {
    's': _kalman_smoother.sKalmanSmoother,
    'd': _kalman_smoother.dKalmanSmoother,
    'c': _kalman_smoother.cKalmanSmoother,
    'z': _kalman_smoother.zKalmanSmoother
}
prefix_simulation_smoother_map = {
    's': _simulation_smoother.sSimulationSmoother,
    'd': _simulation_smoother.dSimulationSmoother,
    'c': _simulation_smoother.cSimulationSmoother,
    'z': _simulation_smoother.zSimulationSmoother
}
prefix_cfa_simulation_smoother_map = {
    's': _cfa_simulation_smoother.sCFASimulationSmoother,
    'd': _cfa_simulation_smoother.dCFASimulationSmoother,
    'c': _cfa_simulation_smoother.cCFASimulationSmoother,
    'z': _cfa_simulation_smoother.zCFASimulationSmoother
}
prefix_pacf_map = {
    's': _tools._scompute_coefficients_from_multivariate_pacf,
    'd': _tools._dcompute_coefficients_from_multivariate_pacf,
    'c': _tools._ccompute_coefficients_from_multivariate_pacf,
    'z': _tools._zcompute_coefficients_from_multivariate_pacf
}
prefix_sv_map = {
    's': _tools._sconstrain_sv_less_than_one,
    'd': _tools._dconstrain_sv_less_than_one,
    'c': _tools._cconstrain_sv_less_than_one,
    'z': _tools._zconstrain_sv_less_than_one
}
prefix_reorder_missing_matrix_map = {
    's': _tools.sreorder_missing_matrix,
    'd': _tools.dreorder_missing_matrix,
    'c': _tools.creorder_missing_matrix,
    'z': _tools.zreorder_missing_matrix
}
prefix_reorder_missing_vector_map = {
    's': _tools.sreorder_missing_vector,
    'd': _tools.dreorder_missing_vector,
    'c': _tools.creorder_missing_vector,
    'z': _tools.zreorder_missing_vector
}
prefix_copy_missing_matrix_map = {
    's': _tools.scopy_missing_matrix,
    'd': _tools.dcopy_missing_matrix,
    'c': _tools.ccopy_missing_matrix,
    'z': _tools.zcopy_missing_matrix
}
prefix_copy_missing_vector_map = {
    's': _tools.scopy_missing_vector,
    'd': _tools.dcopy_missing_vector,
    'c': _tools.ccopy_missing_vector,
    'z': _tools.zcopy_missing_vector
}
prefix_copy_index_matrix_map = {
    's': _tools.scopy_index_matrix,
    'd': _tools.dcopy_index_matrix,
    'c': _tools.ccopy_index_matrix,
    'z': _tools.zcopy_index_matrix
}
prefix_copy_index_vector_map = {
    's': _tools.scopy_index_vector,
    'd': _tools.dcopy_index_vector,
    'c': _tools.ccopy_index_vector,
    'z': _tools.zcopy_index_vector
}
prefix_compute_smoothed_state_weights_map = {
    's': _tools._scompute_smoothed_state_weights,
    'd': _tools._dcompute_smoothed_state_weights,
    'c': _tools._ccompute_smoothed_state_weights,
    'z': _tools._zcompute_smoothed_state_weights
}


def set_mode(compatibility=None):
    if compatibility:
        raise NotImplementedError('Compatibility mode is only available in'
                                  ' statsmodels <= 0.9')


def companion_matrix(polynomial):
    r"""
    Create a companion matrix

    Parameters
    ----------
    polynomial : array_like or list
        If an iterable, interpreted as the coefficients of the polynomial from
        which to form the companion matrix. Polynomial coefficients are in
        order of increasing degree, and may be either scalars (as in an AR(p)
        model) or coefficient matrices (as in a VAR(p) model). If an integer,
        it is interpreted as the size of a companion matrix of a scalar
        polynomial, where the polynomial coefficients are initialized to zeros.
        If a matrix polynomial is passed, :math:`C_0` may be set to the scalar
        value 1 to indicate an identity matrix (doing so will improve the speed
        of the companion matrix creation).

    Returns
    -------
    companion_matrix : ndarray

    Notes
    -----
    Given coefficients of a lag polynomial of the form:

    .. math::

        c(L) = c_0 + c_1 L + \dots + c_p L^p

    returns a matrix of the form

    .. math::
        \begin{bmatrix}
            \phi_1 & 1      & 0 & \cdots & 0 \\
            \phi_2 & 0      & 1 &        & 0 \\
            \vdots &        &   & \ddots & 0 \\
                   &        &   &        & 1 \\
            \phi_n & 0      & 0 & \cdots & 0 \\
        \end{bmatrix}

    where some or all of the :math:`\phi_i` may be non-zero (if `polynomial` is
    None, then all are equal to zero).

    If the coefficients provided are scalars :math:`(c_0, c_1, \dots, c_p)`,
    then the companion matrix is an :math:`n \times n` matrix formed with the
    elements in the first column defined as
    :math:`\phi_i = -\frac{c_i}{c_0}, i \in 1, \dots, p`.

    If the coefficients provided are matrices :math:`(C_0, C_1, \dots, C_p)`,
    each of shape :math:`(m, m)`, then the companion matrix is an
    :math:`nm \times nm` matrix formed with the elements in the first column
    defined as :math:`\phi_i = -C_0^{-1} C_i', i \in 1, \dots, p`.

    It is important to understand the expected signs of the coefficients. A
    typical AR(p) model is written as:

    .. math::
        y_t = a_1 y_{t-1} + \dots + a_p y_{t-p} + \varepsilon_t

    This can be rewritten as:

    .. math::
        (1 - a_1 L - \dots - a_p L^p )y_t = \varepsilon_t \\
        (1 + c_1 L + \dots + c_p L^p )y_t = \varepsilon_t \\
        c(L) y_t = \varepsilon_t

    The coefficients from this form are defined to be :math:`c_i = - a_i`, and
    it is the :math:`c_i` coefficients that this function expects to be
    provided.
    """
    identity_matrix = False
    if isinstance(polynomial, (int, np.integer)):
        # GH 5570, allow numpy integer types, but coerce to python int
        n = int(polynomial)
        m = 1
        polynomial = None
    else:
        n = len(polynomial) - 1

        if n < 1:
            raise ValueError("Companion matrix polynomials must include at"
                             " least two terms.")

        if isinstance(polynomial, (list, tuple)):
            try:
                # Note: cannot use polynomial[0] because of the special
                # behavior associated with matrix polynomials and the constant
                # 1, see below.
                m = len(polynomial[1])
            except TypeError:
                m = 1

            # Check if we just have a scalar polynomial
            if m == 1:
                polynomial = np.asanyarray(polynomial)
            # Check if 1 was passed as the first argument (indicating an
            # identity matrix)
            elif polynomial[0] == 1:
                polynomial[0] = np.eye(m)
                identity_matrix = True
        else:
            m = 1
            polynomial = np.asanyarray(polynomial)

    matrix = np.zeros((n * m, n * m), dtype=np.asanyarray(polynomial).dtype)
    idx = np.diag_indices((n - 1) * m)
    idx = (idx[0], idx[1] + m)
    matrix[idx] = 1
    if polynomial is not None and n > 0:
        if m == 1:
            matrix[:, 0] = -polynomial[1:] / polynomial[0]
        elif identity_matrix:
            for i in range(n):
                matrix[i * m:(i + 1) * m, :m] = -polynomial[i+1].T
        else:
            inv = np.linalg.inv(polynomial[0])
            for i in range(n):
                matrix[i * m:(i + 1) * m, :m] = -np.dot(inv, polynomial[i+1]).T
    return matrix


def diff(series, k_diff=1, k_seasonal_diff=None, seasonal_periods=1):
    r"""
    Difference a series simply and/or seasonally along the zero-th axis.

    Given a series (denoted :math:`y_t`), performs the differencing operation

    .. math::

        \Delta^d \Delta_s^D y_t

    where :math:`d =` `diff`, :math:`s =` `seasonal_periods`,
    :math:`D =` `seasonal\_diff`, and :math:`\Delta` is the difference
    operator.

    Parameters
    ----------
    series : array_like
        The series to be differenced.
    k_diff : int, optional
        The number of simple differences to perform. Default is 1.
    k_seasonal_diff : int or None, optional
        The number of seasonal differences to perform. Default is no seasonal
        differencing.
    seasonal_periods : int, optional
        The seasonal lag. Default is 1. Unused if there is no seasonal
        differencing.

    Returns
    -------
    differenced : ndarray
        The differenced array.
    """
    pandas = _is_using_pandas(series, None)
    differenced = np.asanyarray(series) if not pandas else series

    # Seasonal differencing
    if k_seasonal_diff is not None:
        while k_seasonal_diff > 0:
            if not pandas:
                differenced = (differenced[seasonal_periods:] -
                               differenced[:-seasonal_periods])
            else:
                sdiffed = differenced.diff(seasonal_periods)
                differenced = sdiffed[seasonal_periods:]
            k_seasonal_diff -= 1

    # Simple differencing
    if not pandas:
        differenced = np.diff(differenced, k_diff, axis=0)
    else:
        while k_diff > 0:
            differenced = differenced.diff()[1:]
            k_diff -= 1
    return differenced


def concat(series, axis=0, allow_mix=False):
    """
    Concatenate a set of series.

    Parameters
    ----------
    series : iterable
        An iterable of series to be concatenated
    axis : int, optional
        The axis along which to concatenate. Default is 1 (columns).
    allow_mix : bool
        Whether or not to allow a mix of pandas and non-pandas objects. Default
        is False. If true, the returned object is an ndarray, and additional
        pandas metadata (e.g. column names, indices, etc) is lost.

    Returns
    -------
    concatenated : array or pd.DataFrame
        The concatenated array. Will be a DataFrame if series are pandas
        objects.
    """
    is_pandas = np.r_[[_is_using_pandas(s, None) for s in series]]
    ndim = np.r_[[np.ndim(s) for s in series]]
    max_ndim = np.max(ndim)

    if max_ndim > 2:
        raise ValueError('`tools.concat` does not support arrays with 3 or'
                         ' more dimensions.')

    # Make sure the iterable is mutable
    if isinstance(series, tuple):
        series = list(series)

    # Standardize ndim
    for i in range(len(series)):
        if ndim[i] == 0 and max_ndim == 1:
            series[i] = np.atleast_1d(series[i])
        elif ndim[i] == 0 and max_ndim == 2:
            series[i] = np.atleast_2d(series[i])
        elif ndim[i] == 1 and max_ndim == 2 and is_pandas[i]:
            name = series[i].name
            series[i] = series[i].to_frame()
            series[i].columns = [name]
        elif ndim[i] == 1 and max_ndim == 2 and not is_pandas[i]:
            series[i] = np.atleast_2d(series[i]).T

    if np.all(is_pandas):
        if isinstance(series[0], pd.DataFrame):
            base_columns = series[0].columns
        else:
            base_columns = pd.Index([series[0].name])
        for i in range(1, len(series)):
            s = series[i]

            if isinstance(s, pd.DataFrame):
                # Handle case where we were passed a dataframe and a series
                # to concatenate, and the series did not have a name.
                if s.columns.equals(pd.Index([None])):
                    s.columns = base_columns[:1]
                s_columns = s.columns
            else:
                s_columns = pd.Index([s.name])

            if axis == 0 and not base_columns.equals(s_columns):
                raise ValueError('Columns must match to concatenate along'
                                 ' rows.')
            elif axis == 1 and not series[0].index.equals(s.index):
                raise ValueError('Index must match to concatenate along'
                                 ' columns.')
        concatenated = pd.concat(series, axis=axis)
    elif np.all(~is_pandas) or allow_mix:
        concatenated = np.concatenate(series, axis=axis)
    else:
        raise ValueError('Attempted to concatenate Pandas objects with'
                         ' non-Pandas objects with `allow_mix=False`.')

    return concatenated


def is_invertible(polynomial, threshold=1 - 1e-10):
    r"""
    Determine if a polynomial is invertible.

    Requires all roots of the polynomial lie inside the unit circle.

    Parameters
    ----------
    polynomial : array_like or tuple, list
        Coefficients of a polynomial, in order of increasing degree.
        For example, `polynomial=[1, -0.5]` corresponds to the polynomial
        :math:`1 - 0.5x` which has root :math:`2`. If it is a matrix
        polynomial (in which case the coefficients are coefficient matrices),
        a tuple or list of matrices should be passed.
    threshold : number
        Allowed threshold for `is_invertible` to return True. Default is 1.

    See Also
    --------
    companion_matrix

    Notes
    -----

    If the coefficients provided are scalars :math:`(c_0, c_1, \dots, c_n)`,
    then the corresponding polynomial is :math:`c_0 + c_1 L + \dots + c_n L^n`.


    If the coefficients provided are matrices :math:`(C_0, C_1, \dots, C_n)`,
    then the corresponding polynomial is :math:`C_0 + C_1 L + \dots + C_n L^n`.

    There are three equivalent methods of determining if the polynomial
    represented by the coefficients is invertible:

    The first method factorizes the polynomial into:

    .. math::

        C(L) & = c_0 + c_1 L + \dots + c_n L^n \\
             & = constant (1 - \lambda_1 L)
                 (1 - \lambda_2 L) \dots (1 - \lambda_n L)

    In order for :math:`C(L)` to be invertible, it must be that each factor
    :math:`(1 - \lambda_i L)` is invertible; the condition is then that
    :math:`|\lambda_i| < 1`, where :math:`\lambda_i` is a root of the
    polynomial.

    The second method factorizes the polynomial into:

    .. math::

        C(L) & = c_0 + c_1 L + \dots + c_n L^n \\
             & = constant (L - \zeta_1) (L - \zeta_2) \dots (L - \zeta_3)

    The condition is now :math:`|\zeta_i| > 1`, where :math:`\zeta_i` is a root
    of the polynomial with reversed coefficients and
    :math:`\lambda_i = \frac{1}{\zeta_i}`.

    Finally, a companion matrix can be formed using the coefficients of the
    polynomial. Then the eigenvalues of that matrix give the roots of the
    polynomial. This last method is the one actually used.
    """
    # First method:
    # np.all(np.abs(np.roots(np.r_[1, params])) < 1)
    # Second method:
    # np.all(np.abs(np.roots(np.r_[1, params][::-1])) > 1)
    # Final method:
    eigvals = np.linalg.eigvals(companion_matrix(polynomial))
    return np.all(np.abs(eigvals) < threshold)


def solve_discrete_lyapunov(a, q, complex_step=False):
    r"""
    Solves the discrete Lyapunov equation using a bilinear transformation.

    Notes
    -----
    This is a modification of the version in Scipy (see
    https://github.com/scipy/scipy/blob/master/scipy/linalg/_solvers.py)
    which allows passing through the complex numbers in the matrix a
    (usually the transition matrix) in order to allow complex step
    differentiation.
    """
    eye = np.eye(a.shape[0], dtype=a.dtype)
    if not complex_step:
        aH = a.conj().transpose()
        aHI_inv = np.linalg.inv(aH + eye)
        b = np.dot(aH - eye, aHI_inv)
        c = 2*np.dot(np.dot(np.linalg.inv(a + eye), q), aHI_inv)
        return solve_sylvester(b.conj().transpose(), b, -c)
    else:
        aH = a.transpose()
        aHI_inv = np.linalg.inv(aH + eye)
        b = np.dot(aH - eye, aHI_inv)
        c = 2*np.dot(np.dot(np.linalg.inv(a + eye), q), aHI_inv)
        return solve_sylvester(b.transpose(), b, -c)


def constrain_stationary_univariate(unconstrained):
    """
    Transform unconstrained parameters used by the optimizer to constrained
    parameters used in likelihood evaluation

    Parameters
    ----------
    unconstrained : ndarray
        Unconstrained parameters used by the optimizer, to be transformed to
        stationary coefficients of, e.g., an autoregressive or moving average
        component.

    Returns
    -------
    constrained : ndarray
        Constrained parameters of, e.g., an autoregressive or moving average
        component, to be transformed to arbitrary parameters used by the
        optimizer.

    References
    ----------
    .. [*] Monahan, John F. 1984.
       "A Note on Enforcing Stationarity in
       Autoregressive-moving Average Models."
       Biometrika 71 (2) (August 1): 403-404.
    """

    n = unconstrained.shape[0]
    y = np.zeros((n, n), dtype=unconstrained.dtype)
    r = unconstrained/((1 + unconstrained**2)**0.5)
    for k in range(n):
        for i in range(k):
            y[k, i] = y[k - 1, i] + r[k] * y[k - 1, k - i - 1]
        y[k, k] = r[k]
    return -y[n - 1, :]


def unconstrain_stationary_univariate(constrained):
    """
    Transform constrained parameters used in likelihood evaluation
    to unconstrained parameters used by the optimizer

    Parameters
    ----------
    constrained : ndarray
        Constrained parameters of, e.g., an autoregressive or moving average
        component, to be transformed to arbitrary parameters used by the
        optimizer.

    Returns
    -------
    unconstrained : ndarray
        Unconstrained parameters used by the optimizer, to be transformed to
        stationary coefficients of, e.g., an autoregressive or moving average
        component.

    References
    ----------
    .. [*] Monahan, John F. 1984.
       "A Note on Enforcing Stationarity in
       Autoregressive-moving Average Models."
       Biometrika 71 (2) (August 1): 403-404.
    """
    n = constrained.shape[0]
    y = np.zeros((n, n), dtype=constrained.dtype)
    y[n-1:] = -constrained
    for k in range(n-1, 0, -1):
        for i in range(k):
            y[k-1, i] = (y[k, i] - y[k, k]*y[k, k-i-1]) / (1 - y[k, k]**2)
    r = y.diagonal()
    x = r / ((1 - r**2)**0.5)
    return x


def _constrain_sv_less_than_one_python(unconstrained, order=None,
                                       k_endog=None):
    """
    Transform arbitrary matrices to matrices with singular values less than
    one.

    Parameters
    ----------
    unconstrained : list
        Arbitrary matrices. Should be a list of length `order`, where each
        element is an array sized `k_endog` x `k_endog`.
    order : int, optional
        The order of the autoregression.
    k_endog : int, optional
        The dimension of the data vector.

    Returns
    -------
    constrained : list
        Partial autocorrelation matrices. Should be a list of length
        `order`, where each element is an array sized `k_endog` x `k_endog`.

    See Also
    --------
    constrain_stationary_multivariate

    Notes
    -----
    Corresponds to Lemma 2.2 in Ansley and Kohn (1986). See
    `constrain_stationary_multivariate` for more details.
    """

    from scipy import linalg

    constrained = []  # P_s,  s = 1, ..., p
    if order is None:
        order = len(unconstrained)
    if k_endog is None:
        k_endog = unconstrained[0].shape[0]

    eye = np.eye(k_endog)
    for i in range(order):
        A = unconstrained[i]
        B, lower = linalg.cho_factor(eye + np.dot(A, A.T), lower=True)
        constrained.append(linalg.solve_triangular(B, A, lower=lower))
    return constrained


def _compute_coefficients_from_multivariate_pacf_python(
        partial_autocorrelations, error_variance, transform_variance=False,
        order=None, k_endog=None):
    """
    Transform matrices with singular values less than one to matrices
    corresponding to a stationary (or invertible) process.

    Parameters
    ----------
    partial_autocorrelations : list
        Partial autocorrelation matrices. Should be a list of length `order`,
        where each element is an array sized `k_endog` x `k_endog`.
    error_variance : ndarray
        The variance / covariance matrix of the error term. Should be sized
        `k_endog` x `k_endog`. This is used as input in the algorithm even if
        is not transformed by it (when `transform_variance` is False). The
        error term variance is required input when transformation is used
        either to force an autoregressive component to be stationary or to
        force a moving average component to be invertible.
    transform_variance : bool, optional
        Whether or not to transform the error variance term. This option is
        not typically used, and the default is False.
    order : int, optional
        The order of the autoregression.
    k_endog : int, optional
        The dimension of the data vector.

    Returns
    -------
    coefficient_matrices : list
        Transformed coefficient matrices leading to a stationary VAR
        representation.

    See Also
    --------
    constrain_stationary_multivariate

    Notes
    -----
    Corresponds to Lemma 2.1 in Ansley and Kohn (1986). See
    `constrain_stationary_multivariate` for more details.
    """
    from scipy import linalg

    if order is None:
        order = len(partial_autocorrelations)
    if k_endog is None:
        k_endog = partial_autocorrelations[0].shape[0]

    # If we want to keep the provided variance but with the constrained
    # coefficient matrices, we need to make a copy here, and then after the
    # main loop we will transform the coefficients to match the passed variance
    if not transform_variance:
        initial_variance = error_variance
        # Need to make the input variance large enough that the recursions
        # do not lead to zero-matrices due to roundoff error, which would case
        # exceptions from the Cholesky decompositions.
        # Note that this will still not always ensure positive definiteness,
        # and for k_endog, order large enough an exception may still be raised
        error_variance = np.eye(k_endog) * (order + k_endog)**10

    forward_variances = [error_variance]   # \Sigma_s
    backward_variances = [error_variance]  # \Sigma_s^*,  s = 0, ..., p
    autocovariances = [error_variance]     # \Gamma_s
    # \phi_{s,k}, s = 1, ..., p
    #             k = 1, ..., s+1
    forwards = []
    # \phi_{s,k}^*
    backwards = []

    error_variance_factor = linalg.cholesky(error_variance, lower=True)

    forward_factors = [error_variance_factor]
    backward_factors = [error_variance_factor]

    # We fill in the entries as follows:
    # [1,1]
    # [2,2], [2,1]
    # [3,3], [3,1], [3,2]
    # ...
    # [p,p], [p,1], ..., [p,p-1]
    # the last row, correctly ordered, is then used as the coefficients
    for s in range(order):  # s = 0, ..., p-1
        prev_forwards = forwards
        prev_backwards = backwards
        forwards = []
        backwards = []

        # Create the "last" (k = s+1) matrix
        # Note: this is for k = s+1. However, below we then have to fill
        # in for k = 1, ..., s in order.
        # P L*^{-1} = x
        # x L* = P
        # L*' x' = P'
        forwards.append(
            linalg.solve_triangular(
                backward_factors[s], partial_autocorrelations[s].T,
                lower=True, trans='T'))
        forwards[0] = np.dot(forward_factors[s], forwards[0].T)

        # P' L^{-1} = x
        # x L = P'
        # L' x' = P
        backwards.append(
            linalg.solve_triangular(
                forward_factors[s], partial_autocorrelations[s],
                lower=True, trans='T'))
        backwards[0] = np.dot(backward_factors[s], backwards[0].T)

        # Update the variance
        # Note: if s >= 1, this will be further updated in the for loop
        # below
        # Also, this calculation will be re-used in the forward variance
        tmp = np.dot(forwards[0], backward_variances[s])
        autocovariances.append(tmp.copy().T)

        # Create the remaining k = 1, ..., s matrices,
        # only has an effect if s >= 1
        for k in range(s):
            forwards.insert(k, prev_forwards[k] - np.dot(
                forwards[-1], prev_backwards[s-(k+1)]))

            backwards.insert(k, prev_backwards[k] - np.dot(
                backwards[-1], prev_forwards[s-(k+1)]))

            autocovariances[s+1] += np.dot(autocovariances[k+1],
                                           prev_forwards[s-(k+1)].T)

        # Create forward and backwards variances
        forward_variances.append(
            forward_variances[s] - np.dot(tmp, forwards[s].T)
        )
        backward_variances.append(
            backward_variances[s] -
            np.dot(
                np.dot(backwards[s], forward_variances[s]),
                backwards[s].T
            )
        )

        # Cholesky factors
        forward_factors.append(
            linalg.cholesky(forward_variances[s+1], lower=True)
        )
        backward_factors.append(
            linalg.cholesky(backward_variances[s+1], lower=True)
        )

    # If we do not want to use the transformed variance, we need to
    # adjust the constrained matrices, as presented in Lemma 2.3, see above
    variance = forward_variances[-1]
    if not transform_variance:
        # Here, we need to construct T such that:
        # variance = T * initial_variance * T'
        # To do that, consider the Cholesky of variance (L) and
        # input_variance (M) to get:
        # L L' = T M M' T' = (TM) (TM)'
        # => L = T M
        # => L M^{-1} = T
        initial_variance_factor = np.linalg.cholesky(initial_variance)
        transformed_variance_factor = np.linalg.cholesky(variance)
        transform = np.dot(initial_variance_factor,
                           np.linalg.inv(transformed_variance_factor))
        inv_transform = np.linalg.inv(transform)

        for i in range(order):
            forwards[i] = (
                np.dot(np.dot(transform, forwards[i]), inv_transform)
            )

    return forwards, variance


def constrain_stationary_multivariate_python(unconstrained, error_variance,
                                             transform_variance=False,
                                             prefix=None):
    r"""
    Transform unconstrained parameters used by the optimizer to constrained
    parameters used in likelihood evaluation for a vector autoregression.

    Parameters
    ----------
    unconstrained : array or list
        Arbitrary matrices to be transformed to stationary coefficient matrices
        of the VAR. If a list, should be a list of length `order`, where each
        element is an array sized `k_endog` x `k_endog`. If an array, should be
        the matrices horizontally concatenated and sized
        `k_endog` x `k_endog * order`.
    error_variance : ndarray
        The variance / covariance matrix of the error term. Should be sized
        `k_endog` x `k_endog`. This is used as input in the algorithm even if
        is not transformed by it (when `transform_variance` is False). The
        error term variance is required input when transformation is used
        either to force an autoregressive component to be stationary or to
        force a moving average component to be invertible.
    transform_variance : bool, optional
        Whether or not to transform the error variance term. This option is
        not typically used, and the default is False.
    prefix : {'s','d','c','z'}, optional
        The appropriate BLAS prefix to use for the passed datatypes. Only
        use if absolutely sure that the prefix is correct or an error will
        result.

    Returns
    -------
    constrained : array or list
        Transformed coefficient matrices leading to a stationary VAR
        representation. Will match the type of the passed `unconstrained`
        variable (so if a list was passed, a list will be returned).

    Notes
    -----
    In the notation of [1]_, the arguments `(variance, unconstrained)` are
    written as :math:`(\Sigma, A_1, \dots, A_p)`, where :math:`p` is the order
    of the vector autoregression, and is here determined by the length of
    the `unconstrained` argument.

    There are two steps in the constraining algorithm.

    First, :math:`(A_1, \dots, A_p)` are transformed into
    :math:`(P_1, \dots, P_p)` via Lemma 2.2 of [1]_.

    Second, :math:`(\Sigma, P_1, \dots, P_p)` are transformed into
    :math:`(\Sigma, \phi_1, \dots, \phi_p)` via Lemmas 2.1 and 2.3 of [1]_.

    If `transform_variance=True`, then only Lemma 2.1 is applied in the second
    step.

    While this function can be used even in the univariate case, it is much
    slower, so in that case `constrain_stationary_univariate` is preferred.

    References
    ----------
    .. [1] Ansley, Craig F., and Robert Kohn. 1986.
       "A Note on Reparameterizing a Vector Autoregressive Moving Average Model
       to Enforce Stationarity."
       Journal of Statistical Computation and Simulation 24 (2): 99-106.
    .. [*] Ansley, Craig F, and Paul Newbold. 1979.
       "Multivariate Partial Autocorrelations."
       In Proceedings of the Business and Economic Statistics Section, 349-53.
       American Statistical Association
    """

    use_list = type(unconstrained) is list
    if not use_list:
        k_endog, order = unconstrained.shape
        order //= k_endog

        unconstrained = [
            unconstrained[:k_endog, i*k_endog:(i+1)*k_endog]
            for i in range(order)
        ]

    order = len(unconstrained)
    k_endog = unconstrained[0].shape[0]

    # Step 1: convert from arbitrary matrices to those with singular values
    # less than one.
    sv_constrained = _constrain_sv_less_than_one_python(
        unconstrained, order, k_endog)

    # Step 2: convert matrices from our "partial autocorrelation matrix" space
    # (matrices with singular values less than one) to the space of stationary
    # coefficient matrices
    constrained, var = _compute_coefficients_from_multivariate_pacf_python(
        sv_constrained, error_variance, transform_variance, order, k_endog)

    if not use_list:
        constrained = np.concatenate(constrained, axis=1).reshape(
            k_endog, k_endog * order)

    return constrained, var


@Appender(constrain_stationary_multivariate_python.__doc__)
def constrain_stationary_multivariate(unconstrained, variance,
                                      transform_variance=False,
                                      prefix=None):

    use_list = type(unconstrained) is list
    if use_list:
        unconstrained = np.concatenate(unconstrained, axis=1)

    k_endog, order = unconstrained.shape
    order //= k_endog

    if order < 1:
        raise ValueError('Must have order at least 1')
    if k_endog < 1:
        raise ValueError('Must have at least 1 endogenous variable')

    if prefix is None:
        prefix, dtype, _ = find_best_blas_type(
            [unconstrained, variance])
    dtype = prefix_dtype_map[prefix]

    unconstrained = np.asfortranarray(unconstrained, dtype=dtype)
    variance = np.asfortranarray(variance, dtype=dtype)

    # Step 1: convert from arbitrary matrices to those with singular values
    # less than one.
    # sv_constrained = _constrain_sv_less_than_one(unconstrained, order,
    #                                              k_endog, prefix)
    sv_constrained = prefix_sv_map[prefix](unconstrained, order, k_endog)

    # Step 2: convert matrices from our "partial autocorrelation matrix"
    # space (matrices with singular values less than one) to the space of
    # stationary coefficient matrices
    constrained, variance = prefix_pacf_map[prefix](
        sv_constrained, variance, transform_variance, order, k_endog)

    constrained = np.array(constrained, dtype=dtype)
    variance = np.array(variance, dtype=dtype)

    if use_list:
        constrained = [
            constrained[:k_endog, i*k_endog:(i+1)*k_endog]
            for i in range(order)
        ]

    return constrained, variance


def _unconstrain_sv_less_than_one(constrained, order=None, k_endog=None):
    """
    Transform matrices with singular values less than one to arbitrary
    matrices.

    Parameters
    ----------
    constrained : list
        The partial autocorrelation matrices. Should be a list of length
        `order`, where each element is an array sized `k_endog` x `k_endog`.
    order : int, optional
        The order of the autoregression.
    k_endog : int, optional
        The dimension of the data vector.

    Returns
    -------
    unconstrained : list
        Unconstrained matrices. A list of length `order`, where each element is
        an array sized `k_endog` x `k_endog`.

    See Also
    --------
    unconstrain_stationary_multivariate

    Notes
    -----
    Corresponds to the inverse of Lemma 2.2 in Ansley and Kohn (1986). See
    `unconstrain_stationary_multivariate` for more details.
    """
    from scipy import linalg

    unconstrained = []  # A_s,  s = 1, ..., p
    if order is None:
        order = len(constrained)
    if k_endog is None:
        k_endog = constrained[0].shape[0]

    eye = np.eye(k_endog)
    for i in range(order):
        P = constrained[i]
        # B^{-1} B^{-1}' = I - P P'
        B_inv, lower = linalg.cho_factor(eye - np.dot(P, P.T), lower=True)
        # A = BP
        # B^{-1} A = P
        unconstrained.append(linalg.solve_triangular(B_inv, P, lower=lower))
    return unconstrained


def _compute_multivariate_sample_acovf(endog, maxlag):
    r"""
    Computer multivariate sample autocovariances

    Parameters
    ----------
    endog : array_like
        Sample data on which to compute sample autocovariances. Shaped
        `nobs` x `k_endog`.
    maxlag : int
        Maximum lag to use when computing the sample autocovariances.

    Returns
    -------
    sample_autocovariances : list
        A list of the first `maxlag` sample autocovariance matrices. Each
        matrix is shaped `k_endog` x `k_endog`.

    Notes
    -----
    This function computes the forward sample autocovariances:

    .. math::

        \hat \Gamma(s) = \frac{1}{n} \sum_{t=1}^{n-s}
        (Z_t - \bar Z) (Z_{t+s} - \bar Z)'

    See page 353 of Wei (1990). This function is primarily implemented for
    checking the partial autocorrelation functions below, and so is quite slow.

    References
    ----------
    .. [*] Wei, William. 1990.
       Time Series Analysis : Univariate and Multivariate Methods. Boston:
       Pearson.
    """
    # Get the (demeaned) data as an array
    endog = np.array(endog)
    if endog.ndim == 1:
        endog = endog[:, np.newaxis]
    endog -= np.mean(endog, axis=0)

    # Dimensions
    nobs, k_endog = endog.shape

    sample_autocovariances = []
    for s in range(maxlag + 1):
        sample_autocovariances.append(np.zeros((k_endog, k_endog)))
        for t in range(nobs - s):
            sample_autocovariances[s] += np.outer(endog[t], endog[t+s])
        sample_autocovariances[s] /= nobs

    return sample_autocovariances


def _compute_multivariate_acovf_from_coefficients(
        coefficients, error_variance, maxlag=None,
        forward_autocovariances=False):
    r"""
    Compute multivariate autocovariances from vector autoregression coefficient
    matrices

    Parameters
    ----------
    coefficients : array or list
        The coefficients matrices. If a list, should be a list of length
        `order`, where each element is an array sized `k_endog` x `k_endog`. If
        an array, should be the coefficient matrices horizontally concatenated
        and sized `k_endog` x `k_endog * order`.
    error_variance : ndarray
        The variance / covariance matrix of the error term. Should be sized
        `k_endog` x `k_endog`.
    maxlag : int, optional
        The maximum autocovariance to compute. Default is `order`-1. Can be
        zero, in which case it returns the variance.
    forward_autocovariances : bool, optional
        Whether or not to compute forward autocovariances
        :math:`E(y_t y_{t+j}')`. Default is False, so that backward
        autocovariances :math:`E(y_t y_{t-j}')` are returned.

    Returns
    -------
    autocovariances : list
        A list of the first `maxlag` autocovariance matrices. Each matrix is
        shaped `k_endog` x `k_endog`.

    Notes
    -----
    Computes

    .. math::

        \Gamma(j) = E(y_t y_{t-j}')

    for j = 1, ..., `maxlag`, unless `forward_autocovariances` is specified,
    in which case it computes:

    .. math::

        E(y_t y_{t+j}') = \Gamma(j)'

    Coefficients are assumed to be provided from the VAR model:

    .. math::
        y_t = A_1 y_{t-1} + \dots + A_p y_{t-p} + \varepsilon_t

    Autocovariances are calculated by solving the associated discrete Lyapunov
    equation of the state space representation of the VAR process.
    """
    from scipy import linalg

    # Convert coefficients to a list of matrices, for use in
    # `companion_matrix`; get dimensions
    if type(coefficients) is list:
        order = len(coefficients)
        k_endog = coefficients[0].shape[0]
    else:
        k_endog, order = coefficients.shape
        order //= k_endog

        coefficients = [
            coefficients[:k_endog, i*k_endog:(i+1)*k_endog]
            for i in range(order)
        ]

    if maxlag is None:
        maxlag = order-1

    # Start with VAR(p): w_{t+1} = phi_1 w_t + ... + phi_p w_{t-p+1} + u_{t+1}
    # Then stack the VAR(p) into a VAR(1) in companion matrix form:
    # z_{t+1} = F z_t + v_t
    companion = companion_matrix(
        [1] + [-np.squeeze(coefficients[i]) for i in range(order)]
    ).T

    # Compute the error variance matrix for the stacked form: E v_t v_t'
    selected_variance = np.zeros(companion.shape)
    selected_variance[:k_endog, :k_endog] = error_variance

    # Compute the unconditional variance of z_t: E z_t z_t'
    stacked_cov = linalg.solve_discrete_lyapunov(companion, selected_variance)

    # The first (block) row of the variance of z_t gives the first p-1
    # autocovariances of w_t: \Gamma_i = E w_t w_t+i with \Gamma_0 = Var(w_t)
    # Note: these are okay, checked against ArmaProcess
    autocovariances = [
        stacked_cov[:k_endog, i*k_endog:(i+1)*k_endog]
        for i in range(min(order, maxlag+1))
    ]

    for i in range(maxlag - (order-1)):
        stacked_cov = np.dot(companion, stacked_cov)
        autocovariances += [
            stacked_cov[:k_endog, -k_endog:]
        ]

    if forward_autocovariances:
        for i in range(len(autocovariances)):
            autocovariances[i] = autocovariances[i].T

    return autocovariances


def _compute_multivariate_sample_pacf(endog, maxlag):
    """
    Computer multivariate sample partial autocorrelations

    Parameters
    ----------
    endog : array_like
        Sample data on which to compute sample autocovariances. Shaped
        `nobs` x `k_endog`.
    maxlag : int
        Maximum lag for which to calculate sample partial autocorrelations.

    Returns
    -------
    sample_pacf : list
        A list of the first `maxlag` sample partial autocorrelation matrices.
        Each matrix is shaped `k_endog` x `k_endog`.
    """
    sample_autocovariances = _compute_multivariate_sample_acovf(endog, maxlag)

    return _compute_multivariate_pacf_from_autocovariances(
        sample_autocovariances)


def _compute_multivariate_pacf_from_autocovariances(autocovariances,
                                                    order=None, k_endog=None):
    """
    Compute multivariate partial autocorrelations from autocovariances.

    Parameters
    ----------
    autocovariances : list
        Autocorrelations matrices. Should be a list of length `order` + 1,
        where each element is an array sized `k_endog` x `k_endog`.
    order : int, optional
        The order of the autoregression.
    k_endog : int, optional
        The dimension of the data vector.

    Returns
    -------
    pacf : list
        List of first `order` multivariate partial autocorrelations.

    See Also
    --------
    unconstrain_stationary_multivariate

    Notes
    -----
    Note that this computes multivariate partial autocorrelations.

    Corresponds to the inverse of Lemma 2.1 in Ansley and Kohn (1986). See
    `unconstrain_stationary_multivariate` for more details.

    Computes sample partial autocorrelations if sample autocovariances are
    given.
    """
    from scipy import linalg

    if order is None:
        order = len(autocovariances)-1
    if k_endog is None:
        k_endog = autocovariances[0].shape[0]

    # Now apply the Ansley and Kohn (1986) algorithm, except that instead of
    # calculating phi_{s+1, s+1} = L_s P_{s+1} {L_s^*}^{-1} (which requires
    # the partial autocorrelation P_{s+1} which is what we're trying to
    # calculate here), we calculate it as in Ansley and Newbold (1979), using
    # the autocovariances \Gamma_s and the forwards and backwards residual
    # variances \Sigma_s, \Sigma_s^*:
    # phi_{s+1, s+1} = [ \Gamma_{s+1}' - \phi_{s,1} \Gamma_s' - ... -
    #                    \phi_{s,s} \Gamma_1' ] {\Sigma_s^*}^{-1}

    # Forward and backward variances
    forward_variances = []   # \Sigma_s
    backward_variances = []  # \Sigma_s^*,  s = 0, ..., p
    # \phi_{s,k}, s = 1, ..., p
    #             k = 1, ..., s+1
    forwards = []
    # \phi_{s,k}^*
    backwards = []

    forward_factors = []   # L_s
    backward_factors = []  # L_s^*,  s = 0, ..., p

    # Ultimately we want to construct the partial autocorrelation matrices
    # Note that this is "1-indexed" in the sense that it stores P_1, ... P_p
    # rather than starting with P_0.
    partial_autocorrelations = []

    # We fill in the entries of phi_{s,k} as follows:
    # [1,1]
    # [2,2], [2,1]
    # [3,3], [3,1], [3,2]
    # ...
    # [p,p], [p,1], ..., [p,p-1]
    # the last row, correctly ordered, should be the same as the coefficient
    # matrices provided in the argument `constrained`
    for s in range(order):  # s = 0, ..., p-1
        prev_forwards = list(forwards)
        prev_backwards = list(backwards)
        forwards = []
        backwards = []

        # Create forward and backwards variances Sigma_s, Sigma*_s
        forward_variance = autocovariances[0].copy()
        backward_variance = autocovariances[0].T.copy()

        for k in range(s):
            forward_variance -= np.dot(prev_forwards[k],
                                       autocovariances[k+1])
            backward_variance -= np.dot(prev_backwards[k],
                                        autocovariances[k+1].T)

        forward_variances.append(forward_variance)
        backward_variances.append(backward_variance)

        # Cholesky factors
        forward_factors.append(
            linalg.cholesky(forward_variances[s], lower=True)
        )
        backward_factors.append(
            linalg.cholesky(backward_variances[s], lower=True)
        )

        # Create the intermediate sum term
        if s == 0:
            # phi_11 = \Gamma_1' \Gamma_0^{-1}
            # phi_11 \Gamma_0 = \Gamma_1'
            # \Gamma_0 phi_11' = \Gamma_1
            forwards.append(linalg.cho_solve(
                (forward_factors[0], True), autocovariances[1]).T)
            # backwards.append(forwards[-1])
            # phi_11_star = \Gamma_1 \Gamma_0^{-1}
            # phi_11_star \Gamma_0 = \Gamma_1
            # \Gamma_0 phi_11_star' = \Gamma_1'
            backwards.append(linalg.cho_solve(
                (backward_factors[0], True), autocovariances[1].T).T)
        else:
            # G := \Gamma_{s+1}' -
            #      \phi_{s,1} \Gamma_s' - .. - \phi_{s,s} \Gamma_1'
            tmp_sum = autocovariances[s+1].T.copy()

            for k in range(s):
                tmp_sum -= np.dot(prev_forwards[k], autocovariances[s-k].T)

            # Create the "last" (k = s+1) matrix
            # Note: this is for k = s+1. However, below we then have to
            # fill in for k = 1, ..., s in order.
            # phi = G Sigma*^{-1}
            # phi Sigma* = G
            # Sigma*' phi' = G'
            # Sigma* phi' = G'
            # (because Sigma* is symmetric)
            forwards.append(linalg.cho_solve(
                (backward_factors[s], True), tmp_sum.T).T)

            # phi = G' Sigma^{-1}
            # phi Sigma = G'
            # Sigma' phi' = G
            # Sigma phi' = G
            # (because Sigma is symmetric)
            backwards.append(linalg.cho_solve(
                (forward_factors[s], True), tmp_sum).T)

        # Create the remaining k = 1, ..., s matrices,
        # only has an effect if s >= 1
        for k in range(s):
            forwards.insert(k, prev_forwards[k] - np.dot(
                forwards[-1], prev_backwards[s-(k+1)]))
            backwards.insert(k, prev_backwards[k] - np.dot(
                backwards[-1], prev_forwards[s-(k+1)]))

        # Partial autocorrelation matrix: P_{s+1}
        # P = L^{-1} phi L*
        # L P = (phi L*)
        partial_autocorrelations.append(linalg.solve_triangular(
            forward_factors[s], np.dot(forwards[s], backward_factors[s]),
            lower=True))

    return partial_autocorrelations


def _compute_multivariate_pacf_from_coefficients(constrained, error_variance,
                                                 order=None, k_endog=None):
    r"""
    Transform matrices corresponding to a stationary (or invertible) process
    to matrices with singular values less than one.

    Parameters
    ----------
    constrained : array or list
        The coefficients matrices. If a list, should be a list of length
        `order`, where each element is an array sized `k_endog` x `k_endog`. If
        an array, should be the coefficient matrices horizontally concatenated
        and sized `k_endog` x `k_endog * order`.
    error_variance : ndarray
        The variance / covariance matrix of the error term. Should be sized
        `k_endog` x `k_endog`.
    order : int, optional
        The order of the autoregression.
    k_endog : int, optional
        The dimension of the data vector.

    Returns
    -------
    pacf : list
        List of first `order` multivariate partial autocorrelations.

    See Also
    --------
    unconstrain_stationary_multivariate

    Notes
    -----
    Note that this computes multivariate partial autocorrelations.

    Corresponds to the inverse of Lemma 2.1 in Ansley and Kohn (1986). See
    `unconstrain_stationary_multivariate` for more details.

    Notes
    -----
    Coefficients are assumed to be provided from the VAR model:

    .. math::
        y_t = A_1 y_{t-1} + \dots + A_p y_{t-p} + \varepsilon_t
    """

    if type(constrained) is list:
        order = len(constrained)
        k_endog = constrained[0].shape[0]
    else:
        k_endog, order = constrained.shape
        order //= k_endog

    # Get autocovariances for the process; these are defined to be
    # E z_t z_{t-j}'
    # However, we want E z_t z_{t+j}' = (E z_t z_{t-j}')'
    _acovf = _compute_multivariate_acovf_from_coefficients

    autocovariances = [
        autocovariance.T for autocovariance in
        _acovf(constrained, error_variance, maxlag=order)]

    return _compute_multivariate_pacf_from_autocovariances(autocovariances)


def unconstrain_stationary_multivariate(constrained, error_variance):
    """
    Transform constrained parameters used in likelihood evaluation
    to unconstrained parameters used by the optimizer

    Parameters
    ----------
    constrained : array or list
        Constrained parameters of, e.g., an autoregressive or moving average
        component, to be transformed to arbitrary parameters used by the
        optimizer. If a list, should be a list of length `order`, where each
        element is an array sized `k_endog` x `k_endog`. If an array, should be
        the coefficient matrices horizontally concatenated and sized
        `k_endog` x `k_endog * order`.
    error_variance : ndarray
        The variance / covariance matrix of the error term. Should be sized
        `k_endog` x `k_endog`. This is used as input in the algorithm even if
        is not transformed by it (when `transform_variance` is False).

    Returns
    -------
    unconstrained : ndarray
        Unconstrained parameters used by the optimizer, to be transformed to
        stationary coefficients of, e.g., an autoregressive or moving average
        component. Will match the type of the passed `constrained`
        variable (so if a list was passed, a list will be returned).

    Notes
    -----
    Uses the list representation internally, even if an array is passed.

    References
    ----------
    .. [*] Ansley, Craig F., and Robert Kohn. 1986.
       "A Note on Reparameterizing a Vector Autoregressive Moving Average Model
       to Enforce Stationarity."
       Journal of Statistical Computation and Simulation 24 (2): 99-106.
    """
    use_list = type(constrained) is list
    if not use_list:
        k_endog, order = constrained.shape
        order //= k_endog

        constrained = [
            constrained[:k_endog, i*k_endog:(i+1)*k_endog]
            for i in range(order)
        ]
    else:
        order = len(constrained)
        k_endog = constrained[0].shape[0]

    # Step 1: convert matrices from the space of stationary
    # coefficient matrices to our "partial autocorrelation matrix" space
    # (matrices with singular values less than one)
    partial_autocorrelations = _compute_multivariate_pacf_from_coefficients(
        constrained, error_variance, order, k_endog)

    # Step 2: convert from arbitrary matrices to those with singular values
    # less than one.
    unconstrained = _unconstrain_sv_less_than_one(
        partial_autocorrelations, order, k_endog)

    if not use_list:
        unconstrained = np.concatenate(unconstrained, axis=1)

    return unconstrained, error_variance


def validate_matrix_shape(name, shape, nrows, ncols, nobs):
    """
    Validate the shape of a possibly time-varying matrix, or raise an exception

    Parameters
    ----------
    name : str
        The name of the matrix being validated (used in exception messages)
    shape : array_like
        The shape of the matrix to be validated. May be of size 2 or (if
        the matrix is time-varying) 3.
    nrows : int
        The expected number of rows.
    ncols : int
        The expected number of columns.
    nobs : int
        The number of observations (used to validate the last dimension of a
        time-varying matrix)

    Raises
    ------
    ValueError
        If the matrix is not of the desired shape.
    """
    ndim = len(shape)

    # Enforce dimension
    if ndim not in [2, 3]:
        raise ValueError('Invalid value for %s matrix. Requires a'
                         ' 2- or 3-dimensional array, got %d dimensions' %
                         (name, ndim))
    # Enforce the shape of the matrix
    if not shape[0] == nrows:
        raise ValueError('Invalid dimensions for %s matrix: requires %d'
                         ' rows, got %d' % (name, nrows, shape[0]))
    if not shape[1] == ncols:
        raise ValueError('Invalid dimensions for %s matrix: requires %d'
                         ' columns, got %d' % (name, ncols, shape[1]))

    # If we do not yet know `nobs`, do not allow time-varying arrays
    if nobs is None and not (ndim == 2 or shape[-1] == 1):
        raise ValueError('Invalid dimensions for %s matrix: time-varying'
                         ' matrices cannot be given unless `nobs` is specified'
                         ' (implicitly when a dataset is bound or else set'
                         ' explicity)' % name)

    # Enforce time-varying array size
    if ndim == 3 and nobs is not None and not shape[-1] in [1, nobs]:
        raise ValueError('Invalid dimensions for time-varying %s'
                         ' matrix. Requires shape (*,*,%d), got %s' %
                         (name, nobs, str(shape)))


def validate_vector_shape(name, shape, nrows, nobs):
    """
    Validate the shape of a possibly time-varying vector, or raise an exception

    Parameters
    ----------
    name : str
        The name of the vector being validated (used in exception messages)
    shape : array_like
        The shape of the vector to be validated. May be of size 1 or (if
        the vector is time-varying) 2.
    nrows : int
        The expected number of rows (elements of the vector).
    nobs : int
        The number of observations (used to validate the last dimension of a
        time-varying vector)

    Raises
    ------
    ValueError
        If the vector is not of the desired shape.
    """
    ndim = len(shape)
    # Enforce dimension
    if ndim not in [1, 2]:
        raise ValueError('Invalid value for %s vector. Requires a'
                         ' 1- or 2-dimensional array, got %d dimensions' %
                         (name, ndim))
    # Enforce the shape of the vector
    if not shape[0] == nrows:
        raise ValueError('Invalid dimensions for %s vector: requires %d'
                         ' rows, got %d' % (name, nrows, shape[0]))

    # If we do not yet know `nobs`, do not allow time-varying arrays
    if nobs is None and not (ndim == 1 or shape[-1] == 1):
        raise ValueError('Invalid dimensions for %s vector: time-varying'
                         ' vectors cannot be given unless `nobs` is specified'
                         ' (implicitly when a dataset is bound or else set'
                         ' explicity)' % name)

    # Enforce time-varying array size
    if ndim == 2 and not shape[1] in [1, nobs]:
        raise ValueError('Invalid dimensions for time-varying %s'
                         ' vector. Requires shape (*,%d), got %s' %
                         (name, nobs, str(shape)))


def reorder_missing_matrix(matrix, missing, reorder_rows=False,
                           reorder_cols=False, is_diagonal=False,
                           inplace=False, prefix=None):
    """
    Reorder the rows or columns of a time-varying matrix where all non-missing
    values are in the upper left corner of the matrix.

    Parameters
    ----------
    matrix : array_like
        The matrix to be reordered. Must have shape (n, m, nobs).
    missing : array_like of bool
        The vector of missing indices. Must have shape (k, nobs) where `k = n`
        if `reorder_rows is True` and `k = m` if `reorder_cols is True`.
    reorder_rows : bool, optional
        Whether or not the rows of the matrix should be re-ordered. Default
        is False.
    reorder_cols : bool, optional
        Whether or not the columns of the matrix should be re-ordered. Default
        is False.
    is_diagonal : bool, optional
        Whether or not the matrix is diagonal. If this is True, must also have
        `n = m`. Default is False.
    inplace : bool, optional
        Whether or not to reorder the matrix in-place.
    prefix : {'s', 'd', 'c', 'z'}, optional
        The Fortran prefix of the vector. Default is to automatically detect
        the dtype. This parameter should only be used with caution.

    Returns
    -------
    reordered_matrix : array_like
        The reordered matrix.
    """
    if prefix is None:
        prefix = find_best_blas_type((matrix,))[0]
    reorder = prefix_reorder_missing_matrix_map[prefix]

    if not inplace:
        matrix = np.copy(matrix, order='F')

    reorder(matrix, np.asfortranarray(missing), reorder_rows, reorder_cols,
            is_diagonal)

    return matrix


def reorder_missing_vector(vector, missing, inplace=False, prefix=None):
    """
    Reorder the elements of a time-varying vector where all non-missing
    values are in the first elements of the vector.

    Parameters
    ----------
    vector : array_like
        The vector to be reordered. Must have shape (n, nobs).
    missing : array_like of bool
        The vector of missing indices. Must have shape (n, nobs).
    inplace : bool, optional
        Whether or not to reorder the matrix in-place. Default is False.
    prefix : {'s', 'd', 'c', 'z'}, optional
        The Fortran prefix of the vector. Default is to automatically detect
        the dtype. This parameter should only be used with caution.

    Returns
    -------
    reordered_vector : array_like
        The reordered vector.
    """
    if prefix is None:
        prefix = find_best_blas_type((vector,))[0]
    reorder = prefix_reorder_missing_vector_map[prefix]

    if not inplace:
        vector = np.copy(vector, order='F')

    reorder(vector, np.asfortranarray(missing))

    return vector


def copy_missing_matrix(A, B, missing, missing_rows=False, missing_cols=False,
                        is_diagonal=False, inplace=False, prefix=None):
    """
    Copy the rows or columns of a time-varying matrix where all non-missing
    values are in the upper left corner of the matrix.

    Parameters
    ----------
    A : array_like
        The matrix from which to copy. Must have shape (n, m, nobs) or
        (n, m, 1).
    B : array_like
        The matrix to copy to. Must have shape (n, m, nobs).
    missing : array_like of bool
        The vector of missing indices. Must have shape (k, nobs) where `k = n`
        if `reorder_rows is True` and `k = m` if `reorder_cols is True`.
    missing_rows : bool, optional
        Whether or not the rows of the matrix are a missing dimension. Default
        is False.
    missing_cols : bool, optional
        Whether or not the columns of the matrix are a missing dimension.
        Default is False.
    is_diagonal : bool, optional
        Whether or not the matrix is diagonal. If this is True, must also have
        `n = m`. Default is False.
    inplace : bool, optional
        Whether or not to copy to B in-place. Default is False.
    prefix : {'s', 'd', 'c', 'z'}, optional
        The Fortran prefix of the vector. Default is to automatically detect
        the dtype. This parameter should only be used with caution.

    Returns
    -------
    copied_matrix : array_like
        The matrix B with the non-missing submatrix of A copied onto it.
    """
    if prefix is None:
        prefix = find_best_blas_type((A, B))[0]
    copy = prefix_copy_missing_matrix_map[prefix]

    if not inplace:
        B = np.copy(B, order='F')

    # We may have been given an F-contiguous memoryview; in that case, we do
    # not want to alter it or convert it to a numpy array
    try:
        if not A.is_f_contig():
            raise ValueError()
    except (AttributeError, ValueError):
        A = np.asfortranarray(A)

    copy(A, B, np.asfortranarray(missing), missing_rows, missing_cols,
         is_diagonal)

    return B


def copy_missing_vector(a, b, missing, inplace=False, prefix=None):
    """
    Reorder the elements of a time-varying vector where all non-missing
    values are in the first elements of the vector.

    Parameters
    ----------
    a : array_like
        The vector from which to copy. Must have shape (n, nobs) or (n, 1).
    b : array_like
        The vector to copy to. Must have shape (n, nobs).
    missing : array_like of bool
        The vector of missing indices. Must have shape (n, nobs).
    inplace : bool, optional
        Whether or not to copy to b in-place. Default is False.
    prefix : {'s', 'd', 'c', 'z'}, optional
        The Fortran prefix of the vector. Default is to automatically detect
        the dtype. This parameter should only be used with caution.

    Returns
    -------
    copied_vector : array_like
        The vector b with the non-missing subvector of b copied onto it.
    """
    if prefix is None:
        prefix = find_best_blas_type((a, b))[0]
    copy = prefix_copy_missing_vector_map[prefix]

    if not inplace:
        b = np.copy(b, order='F')

    # We may have been given an F-contiguous memoryview; in that case, we do
    # not want to alter it or convert it to a numpy array
    try:
        if not a.is_f_contig():
            raise ValueError()
    except (AttributeError, ValueError):
        a = np.asfortranarray(a)

    copy(a, b, np.asfortranarray(missing))

    return b


def copy_index_matrix(A, B, index, index_rows=False, index_cols=False,
                      is_diagonal=False, inplace=False, prefix=None):
    """
    Copy the rows or columns of a time-varying matrix where all non-index
    values are in the upper left corner of the matrix.

    Parameters
    ----------
    A : array_like
        The matrix from which to copy. Must have shape (n, m, nobs) or
        (n, m, 1).
    B : array_like
        The matrix to copy to. Must have shape (n, m, nobs).
    index : array_like of bool
        The vector of index indices. Must have shape (k, nobs) where `k = n`
        if `reorder_rows is True` and `k = m` if `reorder_cols is True`.
    index_rows : bool, optional
        Whether or not the rows of the matrix are a index dimension. Default
        is False.
    index_cols : bool, optional
        Whether or not the columns of the matrix are a index dimension.
        Default is False.
    is_diagonal : bool, optional
        Whether or not the matrix is diagonal. If this is True, must also have
        `n = m`. Default is False.
    inplace : bool, optional
        Whether or not to copy to B in-place. Default is False.
    prefix : {'s', 'd', 'c', 'z'}, optional
        The Fortran prefix of the vector. Default is to automatically detect
        the dtype. This parameter should only be used with caution.

    Returns
    -------
    copied_matrix : array_like
        The matrix B with the non-index submatrix of A copied onto it.
    """
    if prefix is None:
        prefix = find_best_blas_type((A, B))[0]
    copy = prefix_copy_index_matrix_map[prefix]

    if not inplace:
        B = np.copy(B, order='F')

    # We may have been given an F-contiguous memoryview; in that case, we do
    # not want to alter it or convert it to a numpy array
    try:
        if not A.is_f_contig():
            raise ValueError()
    except (AttributeError, ValueError):
        A = np.asfortranarray(A)

    copy(A, B, np.asfortranarray(index), index_rows, index_cols,
         is_diagonal)

    return B


def copy_index_vector(a, b, index, inplace=False, prefix=None):
    """
    Reorder the elements of a time-varying vector where all non-index
    values are in the first elements of the vector.

    Parameters
    ----------
    a : array_like
        The vector from which to copy. Must have shape (n, nobs) or (n, 1).
    b : array_like
        The vector to copy to. Must have shape (n, nobs).
    index : array_like of bool
        The vector of index indices. Must have shape (n, nobs).
    inplace : bool, optional
        Whether or not to copy to b in-place. Default is False.
    prefix : {'s', 'd', 'c', 'z'}, optional
        The Fortran prefix of the vector. Default is to automatically detect
        the dtype. This parameter should only be used with caution.

    Returns
    -------
    copied_vector : array_like
        The vector b with the non-index subvector of b copied onto it.
    """
    if prefix is None:
        prefix = find_best_blas_type((a, b))[0]
    copy = prefix_copy_index_vector_map[prefix]

    if not inplace:
        b = np.copy(b, order='F')

    # We may have been given an F-contiguous memoryview; in that case, we do
    # not want to alter it or convert it to a numpy array
    try:
        if not a.is_f_contig():
            raise ValueError()
    except (AttributeError, ValueError):
        a = np.asfortranarray(a)

    copy(a, b, np.asfortranarray(index))

    return b


def prepare_exog(exog):
    k_exog = 0
    if exog is not None:
        exog_is_using_pandas = _is_using_pandas(exog, None)
        if not exog_is_using_pandas:
            exog = np.asarray(exog)

        # Make sure we have 2-dimensional array
        if exog.ndim == 1:
            if not exog_is_using_pandas:
                exog = exog[:, None]
            else:
                exog = pd.DataFrame(exog)

        k_exog = exog.shape[1]
    return (k_exog, exog)


def prepare_trend_spec(trend):
    # Trend
    if trend is None or trend == 'n':
        polynomial_trend = np.ones(0)
    elif trend == 'c':
        polynomial_trend = np.r_[1]
    elif trend == 't':
        polynomial_trend = np.r_[0, 1]
    elif trend == 'ct':
        polynomial_trend = np.r_[1, 1]
    elif trend == 'ctt':
        # TODO deprecate ctt?
        polynomial_trend = np.r_[1, 1, 1]
    else:
        trend = np.array(trend)
        if trend.ndim > 0:
            polynomial_trend = (trend > 0).astype(int)
        else:
            raise ValueError(
                "Valid trend inputs are 'c' (constant), 't' (linear trend in "
                "time), 'ct' (both), 'ctt' (both with trend squared) or an "
                "interable defining a polynomial, e.g., [1, 1, 0, 1] is `a + "
                f"b*t + ct**3`. Received {trend}"
            )

    # Note: k_trend is not the degree of the trend polynomial, because e.g.
    # k_trend = 1 corresponds to the degree zero polynomial (with only a
    # constant term).
    k_trend = int(np.sum(polynomial_trend))

    return polynomial_trend, k_trend


def prepare_trend_data(polynomial_trend, k_trend, nobs, offset=1):
    # Cache the arrays for calculating the intercept from the trend
    # components
    time_trend = np.arange(offset, nobs + offset)
    trend_data = np.zeros((nobs, k_trend))
    i = 0
    for k in polynomial_trend.nonzero()[0]:
        if k == 0:
            trend_data[:, i] = np.ones(nobs,)
        else:
            trend_data[:, i] = time_trend**k
        i += 1

    return trend_data


def _safe_cond(a):
    """Compute condition while protecting from LinAlgError"""
    try:
        return np.linalg.cond(a)
    except np.linalg.LinAlgError:
        if np.any(np.isnan(a)):
            return np.nan
        else:
            return np.inf


def _compute_smoothed_state_weights(ssm, compute_t=None, compute_j=None,
                                    compute_prior_weights=None, scale=1.0):
    # Get references to the Cython objects
    _model = ssm._statespace
    _kfilter = ssm._kalman_filter
    _smoother = ssm._kalman_smoother

    # Determine the appropriate function for the dtype
    func = prefix_compute_smoothed_state_weights_map[ssm.prefix]

    # Handle compute_t and compute_j indexes
    if compute_t is None:
        compute_t = np.arange(ssm.nobs)
    if compute_j is None:
        compute_j = np.arange(ssm.nobs)
    compute_t = np.unique(np.atleast_1d(compute_t).astype(np.int32))
    compute_t.sort()
    compute_j = np.unique(np.atleast_1d(compute_j).astype(np.int32))
    compute_j.sort()

    # Default setting for computing the prior weights
    if compute_prior_weights is None:
        compute_prior_weights = compute_j[0] == 0
    # Validate that compute_prior_weights is valid
    if compute_prior_weights and compute_j[0] != 0:
        raise ValueError('If `compute_prior_weights` is set to True, then'
                         ' `compute_j` must include the time period 0.')

    # Compute the weights
    weights, state_intercept_weights, prior_weights, _ = func(
        _smoother, _kfilter, _model, compute_t, compute_j, scale,
        bool(compute_prior_weights))

    # Re-order missing entries correctly and transpose to the appropriate
    # shape
    t0 = min(compute_t[0], compute_j[0])
    missing = np.isnan(ssm.endog[:, t0:])
    if np.any(missing):
        shape = weights.shape
        # Transpose m, p, t, j, -> t, m, p, j so that we can use the
        # `reorder_missing_matrix` function
        weights = np.asfortranarray(weights.transpose(2, 0, 1, 3).reshape(
            shape[2] * shape[0], shape[1], shape[3], order='C'))
        missing = np.asfortranarray(missing.astype(np.int32))
        reorder_missing_matrix(weights, missing, reorder_cols=True,
                               inplace=True)
        # Transpose t, m, p, j -> t, j, m, p,
        weights = (weights.reshape(shape[2], shape[0], shape[1], shape[3])
                          .transpose(0, 3, 1, 2))
    else:
        # Transpose m, p, t, j -> t, j, m, p
        weights = weights.transpose(2, 3, 0, 1)

    # Transpose m, l, t, j -> t, j, m, l
    state_intercept_weights = state_intercept_weights.transpose(2, 3, 0, 1)

    # Transpose m, l, t -> t, m, l
    prior_weights = prior_weights.transpose(2, 0, 1)

    # Subset to the actual computed t, j elements
    ix_tj = np.ix_(compute_t - t0, compute_j - t0)
    weights = weights[ix_tj]
    state_intercept_weights = state_intercept_weights[ix_tj]
    if compute_prior_weights:
        prior_weights = prior_weights[compute_t - t0]

    return weights, state_intercept_weights, prior_weights


def compute_smoothed_state_weights(results, compute_t=None, compute_j=None,
                                   compute_prior_weights=None, resmooth=None):
    r"""
    Construct the weights of observations and the prior on the smoothed state

    Parameters
    ----------
    results : MLEResults object
        Results object from fitting a state space model.
    compute_t : array_like, optional
        An explicit list of periods `t` of the smoothed state vector to compute
        weights for (see the Returns section for more details about the
        dimension `t`). Default is to compute weights for all periods `t`.
        However, if weights for only a few time points are desired, then
        performance can be improved by specifying this argument.
    compute_j : array_like, optional
        An explicit list of periods `j` of observations to compute
        weights for (see the Returns section for more details about the
        dimension `j`). Default is to compute weights for all periods `j`.
        However, if weights for only a few time points are desired, then
        performance can be improved by specifying this argument.
    compute_prior_weights : bool, optional
        Whether or not to compute the weight matrices associated with the prior
        mean (also called the "initial state"). Note that doing so requires
        that period 0 is in the periods defined in `compute_j`. Default is True
        if 0 is in `compute_j` (or if the `compute_j` argument is not passed)
        and False otherwise.
    resmooth : bool, optional
        Whether or not to re-perform filtering and smoothing prior to
        constructing the weights. Default is to resmooth if the smoothed_state
        vector is different between the given results object and the
        underlying smoother. Caution is adviced when changing this setting.
        See the Notes section below for more details.

    Returns
    -------
    weights : array_like
        Weight matrices that can be used to construct the smoothed state from
        the observations. The returned matrix is always shaped
        `(nobs, nobs, k_states, k_endog)`, and entries that are not computed
        are set to NaNs. (Entries will not be computed if they are not
        included in `compute_t` and `compute_j`, or if they correspond to
        missing observations, or if they are for periods in which the exact
        diffuse Kalman filter is operative). The `(t, j, m, p)`-th element of
        this matrix contains the weight of the `p`-th element of the
        observation vector at time `j` in constructing the `m`-th element of
        the smoothed state vector at time `t`.
    prior_weights : array_like
        Weight matrices that describe the impact of the prior (also called the
        initialization) on the smoothed state vector. The returned matrix is
        always shaped `(nobs, k_states, k_states)`. If prior weights are not
        computed, then all entries will be set to NaNs. The `(t, m, l)`-th
        element of this matrix contains the weight of the `l`-th element of the
        prior mean (also called the "initial state") in constructing the
        `m`-th element of the smoothed state vector at time `t`.

    Notes
    -----
    In [1]_, Chapter 4.8, it is shown how the smoothed state vector can be
    written as a weighted vector sum of observations:

    .. math::

        \hat \alpha_t = \sum_{j=1}^n \omega_{jt}^{\hat \alpha} y_j

    One output of this function is the weights
    :math:`\omega_{jt}^{\hat \alpha}`. Note that the description in [1]_
    assumes that the prior mean (or "initial state") is fixed to be zero. More
    generally, the smoothed state vector will also depend partly on the prior.
    The second output of this function are the weights of the prior mean.

    There are two important technical notes about the computations used here:

    1. In the univariate approach to multivariate filtering (see e.g.
       Chapter 6.4 of [1]_), all observations are introduced one at a time,
       including those from the same time period. As a result, the weight of
       each observation can be different than when all observations from the
       same time point are introduced together, as in the typical multivariate
       filtering approach. Here, we always compute weights as in the
       multivariate filtering approach, and we handle singular forecast error
       covariance matrices by using a pseudo-inverse.
    2. Constructing observation weights for periods in which the exact diffuse
       filter (see e.g. Chapter 5 of [1]_) is operative is not done here, and
       so the corresponding entries in the returned weight matrices will always
       be set equal to zeros. While handling these periods may be implemented
       in the future, one option for constructing these weights is to use an
       approximate (instead of exact) diffuse initialization for this purpose.

    Finally, one note about implementation: to compute the weights, we use
    attributes of the underlying filtering and smoothing Cython objects
    directly. However, these objects are not frozen with the result
    computation, and we cannot guarantee that their attributes have not
    changed since `res` was created. As a result, by default we re-run the
    filter and smoother to ensure that the attributes there actually correspond
    to the `res` object. This can be overridden by the user for a small
    performance boost if they are sure that the attributes have not changed;
    see the `resmooth` argument.

    References
    ----------
    .. [1] Durbin, James, and Siem Jan Koopman. 2012.
            Time Series Analysis by State Space Methods: Second Edition.
            Oxford University Press.
    """
    # Get the python model object
    mod = results.model
    # Always update the parameters to be consistent with `res`
    mod.update(results.params)
    # By default, resmooth if it appears the results have changed; check is
    # based on the smoothed state vector
    if resmooth is None:
        resmooth = np.any(results.smoothed_state !=
                          mod.ssm._kalman_smoother.smoothed_state)
    # Resmooth if necessary, otherwise at least update the Cython model
    if resmooth:
        mod.ssm.smooth(conserve_memory=0, update_representation=False,
                       update_filter=False, update_smoother=False)
    else:
        mod.ssm._initialize_representation()

    return _compute_smoothed_state_weights(
        mod.ssm, compute_t=compute_t, compute_j=compute_j,
        compute_prior_weights=compute_prior_weights,
        scale=results.filter_results.scale)


def get_impact_dates(previous_model, updated_model, impact_date=None,
                     start=None, end=None, periods=None):
    """
    Compute start/end periods and an index, often for impacts of data updates

    Parameters
    ----------
    previous_model : MLEModel
        Model used to compute default start/end periods if None are given.
        In the case of computing impacts of data updates, this would be the
        model estimated with the previous dataset. Otherwise, can be the same
        as `updated_model`.
    updated_model : MLEModel
        Model used to compute the index. In the case of computing impacts of
        data updates, this would be the model estimated with the updated
        dataset. Otherwise, can be the same as `previous_model`.
    impact_date : {int, str, datetime}, optional
        Specific individual impact date. Cannot be used in combination with
        `start`, `end`, or `periods`.
    start : {int, str, datetime}, optional
        Starting point of the impact dates. If given, one of `end` or `periods`
        must also be given. If a negative integer, will be computed relative to
        the dates in the `updated_model` index. Cannot be used in combination
        with `impact_date`.
    end : {int, str, datetime}, optional
        Ending point of the impact dates. If given, one of `start` or `periods`
        must also be given. If a negative integer, will be computed relative to
        the dates in the `updated_model` index. Cannot be used in combination
        with `impact_date`.
    periods : int, optional
        Number of impact date periods. If given, one of `start` or `end`
        must also be given. Cannot be used in combination with `impact_date`.

    Returns
    -------
    start : int
        Integer location of the first included impact dates.
    end : int
        Integer location of the last included impact dates (i.e. this integer
        location is included in the returned `index`).
    index : pd.Index
        Index associated with `start` and `end`, as computed from the
        `updated_model`'s index.

    Notes
    -----
    This function is typically used as a helper for standardizing start and
    end periods for a date range where the most sensible default values are
    based on some initial dataset (here contained in the `previous_model`),
    while index-related operations (especially relative start/end dates given
    via negative integers) are most sensibly computed from an updated dataset
    (here contained in the `updated_model`).

    """
    # There doesn't seem to be any universal default that both (a) make
    # sense for all data update combinations, and (b) work with both
    # time-invariant and time-varying models. So we require that the user
    # specify exactly two of start, end, periods.
    if impact_date is not None:
        if not (start is None and end is None and periods is None):
            raise ValueError('Cannot use the `impact_date` argument in'
                             ' combination with `start`, `end`, or'
                             ' `periods`.')
        start = impact_date
        periods = 1
    if start is None and end is None and periods is None:
        start = previous_model.nobs - 1
        end = previous_model.nobs - 1
    if int(start is None) + int(end is None) + int(periods is None) != 1:
        raise ValueError('Of the three parameters: start, end, and'
                         ' periods, exactly two must be specified')
    # If we have the `periods` object, we need to convert `start`/`end` to
    # integers so that we can compute the other one. That's because
    # _get_prediction_index doesn't support a `periods` argument
    elif start is not None and periods is not None:
        start, _, _, _ = updated_model._get_prediction_index(start, start)
        end = start + (periods - 1)
    elif end is not None and periods is not None:
        _, end, _, _ = updated_model._get_prediction_index(end, end)
        start = end - (periods - 1)
    elif start is not None and end is not None:
        pass

    # Get the integer-based start, end and the prediction index
    start, end, out_of_sample, prediction_index = (
        updated_model._get_prediction_index(start, end))
    end = end + out_of_sample

    return start, end, prediction_index


def _atleast_1d(*arys):
    """
    Version of `np.atleast_1d`, copied from
    https://github.com/numpy/numpy/blob/master/numpy/core/shape_base.py,
    with the following modifications:

    1. It allows for `None` arguments, and passes them directly through
    """
    res = []
    for ary in arys:
        if ary is None:
            result = None
        else:
            ary = np.asanyarray(ary)
            if ary.ndim == 0:
                result = ary.reshape(1)
            else:
                result = ary
        res.append(result)
    if len(res) == 1:
        return res[0]
    else:
        return res


def _atleast_2d(*arys):
    """
    Version of `np.atleast_2d`, copied from
    https://github.com/numpy/numpy/blob/master/numpy/core/shape_base.py,
    with the following modifications:

    1. It allows for `None` arguments, and passes them directly through
    2. Instead of creating new axis at the beginning, it creates it at the end
    """
    res = []
    for ary in arys:
        if ary is None:
            result = None
        else:
            ary = np.asanyarray(ary)
            if ary.ndim == 0:
                result = ary.reshape(1, 1)
            elif ary.ndim == 1:
                result = ary[:, np.newaxis]
            else:
                result = ary
        res.append(result)
    if len(res) == 1:
        return res[0]
    else:
        return res
