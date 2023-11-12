# -*- coding: utf-8 -*-
"""
Miscellaneous utility code for VAR estimation
"""
from statsmodels.compat.pandas import frequencies
from statsmodels.compat.python import asbytes
from statsmodels.tools.validation import array_like, int_like

import numpy as np
import pandas as pd
from scipy import stats, linalg

import statsmodels.tsa.tsatools as tsa

#-------------------------------------------------------------------------------
# Auxiliary functions for estimation

def get_var_endog(y, lags, trend='c', has_constant='skip'):
    """
    Make predictor matrix for VAR(p) process

    Z := (Z_0, ..., Z_T).T (T x Kp)
    Z_t = [1 y_t y_{t-1} ... y_{t - p + 1}] (Kp x 1)

    Ref: Lütkepohl p.70 (transposed)

    has_constant can be 'raise', 'add', or 'skip'. See add_constant.
    """
    nobs = len(y)
    # Ravel C order, need to put in descending order
    Z = np.array([y[t-lags : t][::-1].ravel() for t in range(lags, nobs)])

    # Add constant, trend, etc.
    if trend != 'n':
        Z = tsa.add_trend(Z, prepend=True, trend=trend,
                          has_constant=has_constant)

    return Z


def get_trendorder(trend='c'):
    # Handle constant, etc.
    if trend == 'c':
        trendorder = 1
    elif trend in ('n', 'nc'):
        trendorder = 0
    elif trend == 'ct':
        trendorder = 2
    elif trend == 'ctt':
        trendorder = 3
    else:
        raise ValueError(f"Unkown trend: {trend}")
    return trendorder


def make_lag_names(names, lag_order, trendorder=1, exog=None):
    """
    Produce list of lag-variable names. Constant / trends go at the beginning

    Examples
    --------
    >>> make_lag_names(['foo', 'bar'], 2, 1)
    ['const', 'L1.foo', 'L1.bar', 'L2.foo', 'L2.bar']
    """
    lag_names = []
    if isinstance(names, str):
        names = [names]

    # take care of lagged endogenous names
    for i in range(1, lag_order + 1):
        for name in names:
            if not isinstance(name, str):
                name = str(name) # will need consistent unicode handling
            lag_names.append('L'+str(i)+'.'+name)

    # handle the constant name
    if trendorder != 0:
        lag_names.insert(0, 'const')
    if trendorder > 1:
        lag_names.insert(1, 'trend')
    if trendorder > 2:
        lag_names.insert(2, 'trend**2')
    if exog is not None:
        if isinstance(exog, pd.Series):
            exog = pd.DataFrame(exog)
        elif not hasattr(exog, 'ndim'):
            exog = np.asarray(exog)
        if exog.ndim == 1:
            exog = exog[:, None]
        for i in range(exog.shape[1]):
            if isinstance(exog, pd.DataFrame):
                exog_name = str(exog.columns[i])
            else:
                exog_name = "exog" + str(i)
            lag_names.insert(trendorder + i, exog_name)
    return lag_names


def comp_matrix(coefs):
    """
    Return compansion matrix for the VAR(1) representation for a VAR(p) process
    (companion form)

    A = [A_1 A_2 ... A_p-1 A_p
         I_K 0       0     0
         0   I_K ... 0     0
         0 ...       I_K   0]
    """
    p, k1, k2 = coefs.shape
    if k1 != k2:
        raise ValueError('coefs must be 3-d with shape (p, k, k).')

    kp = k1 * p

    result = np.zeros((kp, kp))
    result[:k1] = np.concatenate(coefs, axis=1)

    # Set I_K matrices
    if p > 1:
        result[np.arange(k1, kp), np.arange(kp-k1)] = 1

    return result

#-------------------------------------------------------------------------------
# Miscellaneous stuff


def parse_lutkepohl_data(path): # pragma: no cover
    """
    Parse data files from Lütkepohl (2005) book

    Source for data files: www.jmulti.de
    """

    from collections import deque
    from datetime import datetime
    import re

    regex = re.compile(asbytes(r'<(.*) (\w)([\d]+)>.*'))
    with open(path, 'rb') as f:
        lines = deque(f)

    to_skip = 0
    while asbytes('*/') not in lines.popleft():
        #while '*/' not in lines.popleft():
        to_skip += 1

    while True:
        to_skip += 1
        line = lines.popleft()
        m = regex.match(line)
        if m:
            year, freq, start_point = m.groups()
            break

    data = (pd.read_csv(path, delimiter=r"\s+", header=to_skip+1)
            .to_records(index=False))

    n = len(data)

    # generate the corresponding date range (using pandas for now)
    start_point = int(start_point)
    year = int(year)

    offsets = {
        asbytes('Q'): frequencies.BQuarterEnd(),
        asbytes('M'): frequencies.BMonthEnd(),
        asbytes('A'): frequencies.BYearEnd()
    }

    # create an instance
    offset = offsets[freq]

    inc = offset * (start_point - 1)
    start_date = offset.rollforward(datetime(year, 1, 1)) + inc

    offset = offsets[freq]
    date_range = pd.date_range(start=start_date, freq=offset, periods=n)

    return data, date_range


def norm_signif_level(alpha=0.05):
    return stats.norm.ppf(1 - alpha / 2)


def acf_to_acorr(acf):
    diag = np.diag(acf[0])
    # numpy broadcasting sufficient
    return acf / np.sqrt(np.outer(diag, diag))


def varsim(coefs, intercept, sig_u, steps=100, initial_values=None, seed=None, nsimulations=None):
    """
    Simulate VAR(p) process, given coefficients and assuming Gaussian noise

    Parameters
    ----------
    coefs : ndarray
        Coefficients for the VAR lags of endog.
    intercept : None or ndarray 1-D (neqs,) or (steps, neqs)
        This can be either the intercept for each equation or an offset.
        If None, then the VAR process has a zero intercept.
        If intercept is 1-D, then the same (endog specific) intercept is added
        to all observations.
        If intercept is 2-D, then it is treated as an offset and is added as
        an observation specific intercept to the autoregression. In this case,
        the intercept/offset should have same number of rows as steps, and the
        same number of columns as endogenous variables (neqs).
    sig_u : ndarray
        Covariance matrix of the residuals or innovations.
        If sig_u is None, then an identity matrix is used.
    steps : {None, int}
        number of observations to simulate, this includes the initial
        observations to start the autoregressive process.
        If offset is not None, then exog of the model are used if they were
        provided in the model
    initial_values : array_like, optional
        Initial values for use in the simulation. Shape should be
        (nlags, neqs) or (neqs,). Values should be ordered from less to
        most recent. Note that this values will be returned by the
        simulation as the first values of `endog_simulated` and they
        will count for the total number of steps.
    seed : {None, int}
        If seed is not None, then it will be used with for the random
        variables generated by numpy.random.
    nsimulations : {None, int}
        Number of simulations to perform. If `nsimulations` is None it will
        perform one simulation and return value will have shape (steps, neqs).

    Returns
    -------
    endog_simulated : nd_array
        Endog of the simulated VAR process. Shape will be (nsimulations, steps, neqs)
        or (steps, neqs) if `nsimulations` is None.
    """
    rs = np.random.RandomState(seed=seed)
    rmvnorm = rs.multivariate_normal
    p, k, k = coefs.shape
    nsimulations= int_like(nsimulations, "nsimulations", optional=True)
    if isinstance(nsimulations, int) and nsimulations <= 0:
        raise ValueError("nsimulations must be a positive integer if provided")
    if nsimulations is None:
        result_shape = (steps, k)
        nsimulations = 1
    else:
        result_shape = (nsimulations, steps, k)
    if sig_u is None:
        sig_u = np.eye(k)
    ugen = rmvnorm(np.zeros(len(sig_u)), sig_u, steps*nsimulations).reshape(nsimulations, steps, k)
    result = np.zeros((nsimulations, steps, k))
    if intercept is not None:
        # intercept can be 2-D like an offset variable
        if np.ndim(intercept) > 1:
            if not len(intercept) == ugen.shape[1]:
                raise ValueError('2-D intercept needs to have length `steps`')
        # add intercept/offset also to intial values
        result += intercept
        result[:,p:] += ugen[:,p:]
    else:
        result[:,p:] = ugen[:,p:]

    initial_values = array_like(initial_values, "initial_values", optional=True, maxdim=2)
    if initial_values is not None:
        if not (initial_values.shape == (p, k) or initial_values.shape == (k,)):
            raise ValueError("initial_values should have shape (p, k) or (k,) where p is the number of lags and k is the number of equations.")
        result[:,:p] = initial_values

    # add in AR terms
    for t in range(p, steps):
        ygen = result[:,t]
        for j in range(p):
            ygen += np.dot(coefs[j], result[:,t-j-1].T).T

    return result.reshape(result_shape)


def get_index(lst, name):
    try:
        result = lst.index(name)
    except Exception:
        if not isinstance(name, int):
            raise
        result = name
    return result


#method used repeatedly in Sims-Zha error bands
def eigval_decomp(sym_array):
    """
    Returns
    -------
    W: array of eigenvectors
    eigva: list of eigenvalues
    k: largest eigenvector
    """
    #check if symmetric, do not include shock period
    eigva, W = linalg.eig(sym_array, left=True, right=False)
    k = np.argmax(eigva)
    return W, eigva, k


def vech(A):
    """
    Simple vech operator
    Returns
    -------
    vechvec: vector of all elements on and below diagonal
    """

    length=A.shape[1]
    vechvec=[]
    for i in range(length):
        b=i
        while b < length:
            vechvec.append(A[b,i])
            b=b+1
    vechvec=np.asarray(vechvec)
    return vechvec


def seasonal_dummies(n_seasons, len_endog, first_period=0, centered=False):
    """

    Parameters
    ----------
    n_seasons : int >= 0
        Number of seasons (e.g. 12 for monthly data and 4 for quarterly data).
    len_endog : int >= 0
        Total number of observations.
    first_period : int, default: 0
        Season of the first observation. As an example, suppose we have monthly
        data and the first observation is in March (third month of the year).
        In this case we pass 2 as first_period. (0 for the first season,
        1 for the second, ..., n_seasons-1 for the last season).
        An integer greater than n_seasons-1 are treated in the same way as the
        integer modulo n_seasons.
    centered : bool, default: False
        If True, center (demean) the dummy variables. That is useful in order
        to get seasonal dummies that are orthogonal to the vector of constant
        dummy variables (a vector of ones).

    Returns
    -------
    seasonal_dummies : ndarray (len_endog x n_seasons-1)
    """
    if n_seasons == 0:
        return np.empty((len_endog, 0))
    if n_seasons > 0:
        season_exog = np.zeros((len_endog, n_seasons - 1))
        for i in range(n_seasons - 1):
            season_exog[(i-first_period) % n_seasons::n_seasons, i] = 1

        if centered:
            season_exog -= 1 / n_seasons
        return season_exog
