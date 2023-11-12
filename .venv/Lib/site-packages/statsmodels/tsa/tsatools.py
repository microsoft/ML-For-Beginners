from __future__ import annotations

from statsmodels.compat.python import lrange, Literal

import warnings

import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset

from statsmodels.tools.data import _is_recarray, _is_using_pandas
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tools.typing import NDArray
from statsmodels.tools.validation import (
    array_like,
    bool_like,
    int_like,
    string_like,
)

__all__ = [
    "lagmat",
    "lagmat2ds",
    "add_trend",
    "duplication_matrix",
    "elimination_matrix",
    "commutation_matrix",
    "vec",
    "vech",
    "unvec",
    "unvech",
    "freq_to_period",
]


def add_trend(x, trend="c", prepend=False, has_constant="skip"):
    """
    Add a trend and/or constant to an array.

    Parameters
    ----------
    x : array_like
        Original array of data.
    trend : str {'n', 'c', 't', 'ct', 'ctt'}
        The trend to add.

        * 'n' add no trend.
        * 'c' add constant only.
        * 't' add trend only.
        * 'ct' add constant and linear trend.
        * 'ctt' add constant and linear and quadratic trend.
    prepend : bool
        If True, prepends the new data to the columns of X.
    has_constant : str {'raise', 'add', 'skip'}
        Controls what happens when trend is 'c' and a constant column already
        exists in x. 'raise' will raise an error. 'add' will add a column of
        1s. 'skip' will return the data without change. 'skip' is the default.

    Returns
    -------
    array_like
        The original data with the additional trend columns.  If x is a
        pandas Series or DataFrame, then the trend column names are 'const',
        'trend' and 'trend_squared'.

    See Also
    --------
    statsmodels.tools.tools.add_constant
        Add a constant column to an array.

    Notes
    -----
    Returns columns as ['ctt','ct','c'] whenever applicable. There is currently
    no checking for an existing trend.
    """
    prepend = bool_like(prepend, "prepend")
    trend = string_like(trend, "trend", options=("n", "c", "t", "ct", "ctt"))
    has_constant = string_like(
        has_constant, "has_constant", options=("raise", "add", "skip")
    )

    # TODO: could be generalized for trend of aribitrary order
    columns = ["const", "trend", "trend_squared"]
    if trend == "n":
        return x.copy()
    elif trend == "c":  # handles structured arrays
        columns = columns[:1]
        trendorder = 0
    elif trend == "ct" or trend == "t":
        columns = columns[:2]
        if trend == "t":
            columns = columns[1:2]
        trendorder = 1
    elif trend == "ctt":
        trendorder = 2

    if _is_recarray(x):
        from statsmodels.tools.sm_exceptions import recarray_exception

        raise NotImplementedError(recarray_exception)

    is_pandas = _is_using_pandas(x, None)
    if is_pandas:
        if isinstance(x, pd.Series):
            x = pd.DataFrame(x)
        else:
            x = x.copy()
    else:
        x = np.asanyarray(x)

    nobs = len(x)
    trendarr = np.vander(
        np.arange(1, nobs + 1, dtype=np.float64), trendorder + 1
    )
    # put in order ctt
    trendarr = np.fliplr(trendarr)
    if trend == "t":
        trendarr = trendarr[:, 1]

    if "c" in trend:
        if is_pandas:
            # Mixed type protection
            def safe_is_const(s):
                try:
                    return np.ptp(s) == 0.0 and np.any(s != 0.0)
                except:
                    return False

            col_const = x.apply(safe_is_const, 0)
        else:
            ptp0 = np.ptp(np.asanyarray(x), axis=0)
            col_is_const = ptp0 == 0
            nz_const = col_is_const & (x[0] != 0)
            col_const = nz_const

        if np.any(col_const):
            if has_constant == "raise":
                if x.ndim == 1:
                    base_err = "x is constant."
                else:
                    columns = np.arange(x.shape[1])[col_const]
                    if isinstance(x, pd.DataFrame):
                        columns = x.columns
                    const_cols = ", ".join([str(c) for c in columns])
                    base_err = (
                        "x contains one or more constant columns. Column(s) "
                        f"{const_cols} are constant."
                    )
                msg = f"{base_err} Adding a constant with trend='{trend}' is not allowed."
                raise ValueError(msg)
            elif has_constant == "skip":
                columns = columns[1:]
                trendarr = trendarr[:, 1:]

    order = 1 if prepend else -1
    if is_pandas:
        trendarr = pd.DataFrame(trendarr, index=x.index, columns=columns)
        x = [trendarr, x]
        x = pd.concat(x[::order], axis=1)
    else:
        x = [trendarr, x]
        x = np.column_stack(x[::order])

    return x


def add_lag(x, col=None, lags=1, drop=False, insert=True):
    """
    Returns an array with lags included given an array.

    Parameters
    ----------
    x : array_like
        An array or NumPy ndarray subclass. Can be either a 1d or 2d array with
        observations in columns.
    col : int or None
        `col` can be an int of the zero-based column index. If it's a
        1d array `col` can be None.
    lags : int
        The number of lags desired.
    drop : bool
        Whether to keep the contemporaneous variable for the data.
    insert : bool or int
        If True, inserts the lagged values after `col`. If False, appends
        the data. If int inserts the lags at int.

    Returns
    -------
    array : ndarray
        Array with lags

    Examples
    --------

    >>> import statsmodels.api as sm
    >>> data = sm.datasets.macrodata.load()
    >>> data = data.data[['year','quarter','realgdp','cpi']]
    >>> data = sm.tsa.add_lag(data, 'realgdp', lags=2)

    Notes
    -----
    Trims the array both forward and backward, so that the array returned
    so that the length of the returned array is len(`X`) - lags. The lags are
    returned in increasing order, ie., t-1,t-2,...,t-lags
    """
    lags = int_like(lags, "lags")
    drop = bool_like(drop, "drop")
    x = array_like(x, "x", ndim=2)
    if col is None:
        col = 0

    # handle negative index
    if col < 0:
        col = x.shape[1] + col
    if x.ndim == 1:
        x = x[:, None]
    contemp = x[:, col]

    if insert is True:
        ins_idx = col + 1
    elif insert is False:
        ins_idx = x.shape[1]
    else:
        if insert < 0:  # handle negative index
            insert = x.shape[1] + insert + 1
        if insert > x.shape[1]:
            insert = x.shape[1]

            warnings.warn(
                "insert > number of variables, inserting at the"
                " last position",
                ValueWarning,
            )
        ins_idx = insert

    ndlags = lagmat(contemp, lags, trim="Both")
    first_cols = lrange(ins_idx)
    last_cols = lrange(ins_idx, x.shape[1])
    if drop:
        if col in first_cols:
            first_cols.pop(first_cols.index(col))
        else:
            last_cols.pop(last_cols.index(col))
    return np.column_stack((x[lags:, first_cols], ndlags, x[lags:, last_cols]))


def detrend(x, order=1, axis=0):
    """
    Detrend an array with a trend of given order along axis 0 or 1.

    Parameters
    ----------
    x : array_like, 1d or 2d
        Data, if 2d, then each row or column is independently detrended with
        the same trendorder, but independent trend estimates.
    order : int
        The polynomial order of the trend, zero is constant, one is
        linear trend, two is quadratic trend.
    axis : int
        Axis can be either 0, observations by rows, or 1, observations by
        columns.

    Returns
    -------
    ndarray
        The detrended series is the residual of the linear regression of the
        data on the trend of given order.
    """
    order = int_like(order, "order")
    axis = int_like(axis, "axis")

    if x.ndim == 2 and int(axis) == 1:
        x = x.T
    elif x.ndim > 2:
        raise NotImplementedError(
            "x.ndim > 2 is not implemented until it is needed"
        )

    nobs = x.shape[0]
    if order == 0:
        # Special case demean
        resid = x - x.mean(axis=0)
    else:
        trends = np.vander(np.arange(float(nobs)), N=order + 1)
        beta = np.linalg.pinv(trends).dot(x)
        resid = x - np.dot(trends, beta)

    if x.ndim == 2 and int(axis) == 1:
        resid = resid.T

    return resid


def lagmat(x,
           maxlag: int,
           trim: Literal["forward", "backward", "both", "none"]='forward',
           original: Literal["ex", "sep", "in"]="ex",
           use_pandas: bool=False
           )-> NDArray | DataFrame | tuple[NDArray, NDArray] | tuple[DataFrame, DataFrame]:
    """
    Create 2d array of lags.

    Parameters
    ----------
    x : array_like
        Data; if 2d, observation in rows and variables in columns.
    maxlag : int
        All lags from zero to maxlag are included.
    trim : {'forward', 'backward', 'both', 'none', None}
        The trimming method to use.

        * 'forward' : trim invalid observations in front.
        * 'backward' : trim invalid initial observations.
        * 'both' : trim invalid observations on both sides.
        * 'none', None : no trimming of observations.
    original : {'ex','sep','in'}
        How the original is treated.

        * 'ex' : drops the original array returning only the lagged values.
        * 'in' : returns the original array and the lagged values as a single
          array.
        * 'sep' : returns a tuple (original array, lagged values). The original
                  array is truncated to have the same number of rows as
                  the returned lagmat.
    use_pandas : bool
        If true, returns a DataFrame when the input is a pandas
        Series or DataFrame.  If false, return numpy ndarrays.

    Returns
    -------
    lagmat : ndarray
        The array with lagged observations.
    y : ndarray, optional
        Only returned if original == 'sep'.

    Notes
    -----
    When using a pandas DataFrame or Series with use_pandas=True, trim can only
    be 'forward' or 'both' since it is not possible to consistently extend
    index values.

    Examples
    --------
    >>> from statsmodels.tsa.tsatools import lagmat
    >>> import numpy as np
    >>> X = np.arange(1,7).reshape(-1,2)
    >>> lagmat(X, maxlag=2, trim="forward", original='in')
    array([[ 1.,  2.,  0.,  0.,  0.,  0.],
       [ 3.,  4.,  1.,  2.,  0.,  0.],
       [ 5.,  6.,  3.,  4.,  1.,  2.]])

    >>> lagmat(X, maxlag=2, trim="backward", original='in')
    array([[ 5.,  6.,  3.,  4.,  1.,  2.],
       [ 0.,  0.,  5.,  6.,  3.,  4.],
       [ 0.,  0.,  0.,  0.,  5.,  6.]])

    >>> lagmat(X, maxlag=2, trim="both", original='in')
    array([[ 5.,  6.,  3.,  4.,  1.,  2.]])

    >>> lagmat(X, maxlag=2, trim="none", original='in')
    array([[ 1.,  2.,  0.,  0.,  0.,  0.],
       [ 3.,  4.,  1.,  2.,  0.,  0.],
       [ 5.,  6.,  3.,  4.,  1.,  2.],
       [ 0.,  0.,  5.,  6.,  3.,  4.],
       [ 0.,  0.,  0.,  0.,  5.,  6.]])
    """
    maxlag = int_like(maxlag, "maxlag")
    use_pandas = bool_like(use_pandas, "use_pandas")
    trim = string_like(
        trim,
        "trim",
        optional=True,
        options=("forward", "backward", "both", "none"),
    )
    original = string_like(original, "original", options=("ex", "sep", "in"))

    # TODO:  allow list of lags additional to maxlag
    orig = x
    x = array_like(x, "x", ndim=2, dtype=None)
    is_pandas = _is_using_pandas(orig, None) and use_pandas
    trim = "none" if trim is None else trim
    trim = trim.lower()
    if is_pandas and trim in ("none", "backward"):
        raise ValueError(
            "trim cannot be 'none' or 'backward' when used on "
            "Series or DataFrames"
        )

    dropidx = 0
    nobs, nvar = x.shape
    if original in ["ex", "sep"]:
        dropidx = nvar
    if maxlag >= nobs:
        raise ValueError("maxlag should be < nobs")
    lm = np.zeros((nobs + maxlag, nvar * (maxlag + 1)))
    for k in range(0, int(maxlag + 1)):
        lm[
        maxlag - k: nobs + maxlag - k,
        nvar * (maxlag - k): nvar * (maxlag - k + 1),
        ] = x

    if trim in ("none", "forward"):
        startobs = 0
    elif trim in ("backward", "both"):
        startobs = maxlag
    else:
        raise ValueError("trim option not valid")

    if trim in ("none", "backward"):
        stopobs = len(lm)
    else:
        stopobs = nobs

    if is_pandas:
        x = orig
        if isinstance(x, DataFrame):
            x_columns = [str(c) for c in x.columns]
            if len(set(x_columns)) != x.shape[1]:
                raise ValueError(
                    "Columns names must be distinct after conversion to string "
                    "(if not already strings)."
                )
        else:
            x_columns = [str(x.name)]
        columns = [str(col) for col in x_columns]
        for lag in range(maxlag):
            lag_str = str(lag + 1)
            columns.extend([str(col) + ".L." + lag_str for col in x_columns])
        lm = DataFrame(lm[:stopobs], index=x.index, columns=columns)
        lags = lm.iloc[startobs:]
        if original in ("sep", "ex"):
            leads = lags[x_columns]
            lags = lags.drop(x_columns, axis=1)
    else:
        lags = lm[startobs:stopobs, dropidx:]
        if original == "sep":
            leads = lm[startobs:stopobs, :dropidx]

    if original == "sep":
        return lags, leads
    else:
        return lags


def lagmat2ds(
    x, maxlag0, maxlagex=None, dropex=0, trim="forward", use_pandas=False
):
    """
    Generate lagmatrix for 2d array, columns arranged by variables.

    Parameters
    ----------
    x : array_like
        Data, 2d. Observations in rows and variables in columns.
    maxlag0 : int
        The first variable all lags from zero to maxlag are included.
    maxlagex : {None, int}
        The max lag for all other variables all lags from zero to maxlag are
        included.
    dropex : int
        Exclude first dropex lags from other variables. For all variables,
        except the first, lags from dropex to maxlagex are included.
    trim : str
        The trimming method to use.

        * 'forward' : trim invalid observations in front.
        * 'backward' : trim invalid initial observations.
        * 'both' : trim invalid observations on both sides.
        * 'none' : no trimming of observations.
    use_pandas : bool
        If true, returns a DataFrame when the input is a pandas
        Series or DataFrame.  If false, return numpy ndarrays.

    Returns
    -------
    ndarray
        The array with lagged observations, columns ordered by variable.

    Notes
    -----
    Inefficient implementation for unequal lags, implemented for convenience.
    """
    maxlag0 = int_like(maxlag0, "maxlag0")
    maxlagex = int_like(maxlagex, "maxlagex", optional=True)
    trim = string_like(
        trim,
        "trim",
        optional=True,
        options=("forward", "backward", "both", "none"),
    )
    if maxlagex is None:
        maxlagex = maxlag0
    maxlag = max(maxlag0, maxlagex)
    is_pandas = _is_using_pandas(x, None)

    if x.ndim == 1:
        if is_pandas:
            x = pd.DataFrame(x)
        else:
            x = x[:, None]
    elif x.ndim == 0 or x.ndim > 2:
        raise ValueError("Only supports 1 and 2-dimensional data.")

    nobs, nvar = x.shape

    if is_pandas and use_pandas:
        lags = lagmat(
            x.iloc[:, 0], maxlag, trim=trim, original="in", use_pandas=True
        )
        lagsli = [lags.iloc[:, : maxlag0 + 1]]
        for k in range(1, nvar):
            lags = lagmat(
                x.iloc[:, k], maxlag, trim=trim, original="in", use_pandas=True
            )
            lagsli.append(lags.iloc[:, dropex : maxlagex + 1])
        return pd.concat(lagsli, axis=1)
    elif is_pandas:
        x = np.asanyarray(x)

    lagsli = [
        lagmat(x[:, 0], maxlag, trim=trim, original="in")[:, : maxlag0 + 1]
    ]
    for k in range(1, nvar):
        lagsli.append(
            lagmat(x[:, k], maxlag, trim=trim, original="in")[
                :, dropex : maxlagex + 1
            ]
        )
    return np.column_stack(lagsli)


def vec(mat):
    return mat.ravel("F")


def vech(mat):
    # Gets Fortran-order
    return mat.T.take(_triu_indices(len(mat)))


# tril/triu/diag, suitable for ndarray.take


def _tril_indices(n):
    rows, cols = np.tril_indices(n)
    return rows * n + cols


def _triu_indices(n):
    rows, cols = np.triu_indices(n)
    return rows * n + cols


def _diag_indices(n):
    rows, cols = np.diag_indices(n)
    return rows * n + cols


def unvec(v):
    k = int(np.sqrt(len(v)))
    assert k * k == len(v)
    return v.reshape((k, k), order="F")


def unvech(v):
    # quadratic formula, correct fp error
    rows = 0.5 * (-1 + np.sqrt(1 + 8 * len(v)))
    rows = int(np.round(rows))

    result = np.zeros((rows, rows))
    result[np.triu_indices(rows)] = v
    result = result + result.T

    # divide diagonal elements by 2
    result[np.diag_indices(rows)] /= 2

    return result


def duplication_matrix(n):
    """
    Create duplication matrix D_n which satisfies vec(S) = D_n vech(S) for
    symmetric matrix S

    Returns
    -------
    D_n : ndarray
    """
    n = int_like(n, "n")
    tmp = np.eye(n * (n + 1) // 2)
    return np.array([unvech(x).ravel() for x in tmp]).T


def elimination_matrix(n):
    """
    Create the elimination matrix L_n which satisfies vech(M) = L_n vec(M) for
    any matrix M

    Parameters
    ----------

    Returns
    -------
    """
    n = int_like(n, "n")
    vech_indices = vec(np.tril(np.ones((n, n))))
    return np.eye(n * n)[vech_indices != 0]


def commutation_matrix(p, q):
    """
    Create the commutation matrix K_{p,q} satisfying vec(A') = K_{p,q} vec(A)

    Parameters
    ----------
    p : int
    q : int

    Returns
    -------
    K : ndarray (pq x pq)
    """
    p = int_like(p, "p")
    q = int_like(q, "q")

    K = np.eye(p * q)
    indices = np.arange(p * q).reshape((p, q), order="F")
    return K.take(indices.ravel(), axis=0)


def _ar_transparams(params):
    """
    Transforms params to induce stationarity/invertability.

    Parameters
    ----------
    params : array_like
        The AR coefficients

    Reference
    ---------
    Jones(1980)
    """
    newparams = np.tanh(params / 2)
    tmp = np.tanh(params / 2)
    for j in range(1, len(params)):
        a = newparams[j]
        for kiter in range(j):
            tmp[kiter] -= a * newparams[j - kiter - 1]
        newparams[:j] = tmp[:j]
    return newparams


def _ar_invtransparams(params):
    """
    Inverse of the Jones reparameterization

    Parameters
    ----------
    params : array_like
        The transformed AR coefficients
    """
    params = params.copy()
    tmp = params.copy()
    for j in range(len(params) - 1, 0, -1):
        a = params[j]
        for kiter in range(j):
            tmp[kiter] = (params[kiter] + a * params[j - kiter - 1]) / (
                1 - a ** 2
            )
        params[:j] = tmp[:j]
    invarcoefs = 2 * np.arctanh(params)
    return invarcoefs


def _ma_transparams(params):
    """
    Transforms params to induce stationarity/invertability.

    Parameters
    ----------
    params : ndarray
        The ma coeffecients of an (AR)MA model.

    Reference
    ---------
    Jones(1980)
    """
    newparams = ((1 - np.exp(-params)) / (1 + np.exp(-params))).copy()
    tmp = ((1 - np.exp(-params)) / (1 + np.exp(-params))).copy()

    # levinson-durbin to get macf
    for j in range(1, len(params)):
        b = newparams[j]
        for kiter in range(j):
            tmp[kiter] += b * newparams[j - kiter - 1]
        newparams[:j] = tmp[:j]
    return newparams


def _ma_invtransparams(macoefs):
    """
    Inverse of the Jones reparameterization

    Parameters
    ----------
    params : ndarray
        The transformed MA coefficients
    """
    tmp = macoefs.copy()
    for j in range(len(macoefs) - 1, 0, -1):
        b = macoefs[j]
        for kiter in range(j):
            tmp[kiter] = (macoefs[kiter] - b * macoefs[j - kiter - 1]) / (
                1 - b ** 2
            )
        macoefs[:j] = tmp[:j]
    invmacoefs = -np.log((1 - macoefs) / (1 + macoefs))
    return invmacoefs


def unintegrate_levels(x, d):
    """
    Returns the successive differences needed to unintegrate the series.

    Parameters
    ----------
    x : array_like
        The original series
    d : int
        The number of differences of the differenced series.

    Returns
    -------
    y : array_like
        The increasing differences from 0 to d-1 of the first d elements
        of x.

    See Also
    --------
    unintegrate
    """
    d = int_like(d, "d")
    x = x[:d]
    return np.asarray([np.diff(x, d - i)[0] for i in range(d, 0, -1)])


def unintegrate(x, levels):
    """
    After taking n-differences of a series, return the original series

    Parameters
    ----------
    x : array_like
        The n-th differenced series
    levels : list
        A list of the first-value in each differenced series, for
        [first-difference, second-difference, ..., n-th difference]

    Returns
    -------
    y : array_like
        The original series de-differenced

    Examples
    --------
    >>> x = np.array([1, 3, 9., 19, 8.])
    >>> levels = unintegrate_levels(x, 2)
    >>> levels
    array([ 1.,  2.])
    >>> unintegrate(np.diff(x, 2), levels)
    array([  1.,   3.,   9.,  19.,   8.])
    """
    levels = list(levels)[:]  # copy
    if len(levels) > 1:
        x0 = levels.pop(-1)
        return unintegrate(np.cumsum(np.r_[x0, x]), levels)
    x0 = levels[0]
    return np.cumsum(np.r_[x0, x])


def freq_to_period(freq: str | offsets.DateOffset) -> int:
    """
    Convert a pandas frequency to a periodicity

    Parameters
    ----------
    freq : str or offset
        Frequency to convert

    Returns
    -------
    int
        Periodicity of freq

    Notes
    -----
    Annual maps to 1, quarterly maps to 4, monthly to 12, weekly to 52.
    """
    if not isinstance(freq, offsets.DateOffset):
        freq = to_offset(freq)  # go ahead and standardize
    assert isinstance(freq, offsets.DateOffset)
    freq = freq.rule_code.upper()

    if freq == "A" or freq.startswith(("A-", "AS-")):
        return 1
    elif freq == "Q" or freq.startswith(("Q-", "QS-")):
        return 4
    elif freq == "M" or freq.startswith(("M-", "MS")):
        return 12
    elif freq == "W" or freq.startswith("W-"):
        return 52
    elif freq == "D":
        return 7
    elif freq == "B":
        return 5
    elif freq == "H":
        return 24
    else:  # pragma : no cover
        raise ValueError(
            "freq {} not understood. Please report if you "
            "think this is in error.".format(freq)
        )
