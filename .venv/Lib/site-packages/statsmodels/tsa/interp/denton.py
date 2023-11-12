
import numpy as np
from numpy import (dot, eye, diag_indices, zeros, ones, diag,
        asarray, r_)
from numpy.linalg import solve


# def denton(indicator, benchmark, freq="aq", **kwarg):
#    """
#    Denton's method to convert low-frequency to high frequency data.
#
#    Parameters
#    ----------
#    benchmark : array_like
#        The higher frequency benchmark.  A 1d or 2d data series in columns.
#        If 2d, then M series are assumed.
#    indicator
#        A low-frequency indicator series.  It is assumed that there are no
#        pre-sample indicators.  Ie., the first indicators line up with
#        the first benchmark.
#    freq : str {"aq","qm", "other"}
#        "aq" - Benchmarking an annual series to quarterly.
#        "mq" - Benchmarking a quarterly series to monthly.
#        "other" - Custom stride.  A kwarg, k, must be supplied.
#    kwargs :
#        k : int
#            The number of high-frequency observations that sum to make an
#            aggregate low-frequency observation. `k` is used with
#            `freq` == "other".
#    Returns
#    -------
#    benchmarked series : ndarray
#
#    Notes
#    -----
#    Denton's method minimizes the distance given by the penalty function, in
#    a least squares sense, between the unknown benchmarked series and the
#    indicator series subject to the condition that the sum of the benchmarked
#    series is equal to the benchmark.
#
#
#    References
#    ----------
#    Bloem, A.M, Dippelsman, R.J. and Maehle, N.O.  2001 Quarterly National
#        Accounts Manual--Concepts, Data Sources, and Compilation. IMF.
#        http://www.imf.org/external/pubs/ft/qna/2000/Textbook/index.htm
#    Denton, F.T. 1971. "Adjustment of monthly or quarterly series to annual
#        totals: an approach based on quadratic minimization." Journal of the
#        American Statistical Association. 99-102.
#
#    """
#    # check arrays and make 2d
#    indicator = np.asarray(indicator)
#    if indicator.ndim == 1:
#        indicator = indicator[:,None]
#    benchmark = np.asarray(benchmark)
#    if benchmark.ndim == 1:
#        benchmark = benchmark[:,None]
#
#    # get dimensions
#    N = len(indicator) # total number of high-freq
#    m = len(benchmark) # total number of low-freq
#
#    # number of low-freq observations for aggregate measure
#    # 4 for annual to quarter and 3 for quarter to monthly
#    if freq == "aq":
#        k = 4
#    elif freq == "qm":
#        k = 3
#    elif freq == "other":
#        k = kwargs.get("k")
#        if not k:
#            raise ValueError("k must be supplied with freq=\"other\"")
#    else:
#        raise ValueError("freq %s not understood" % freq)
#
#    n = k*m # number of indicator series with a benchmark for back-series
#    # if k*m != n, then we are going to extrapolate q observations
#
#    B = block_diag(*(np.ones((k,1)),)*m)
#
#    r = benchmark - B.T.dot(indicator)
#TODO: take code in the string at the end and implement Denton's original
# method with a few of the penalty functions.


def dentonm(indicator, benchmark, freq="aq", **kwargs):
    """
    Modified Denton's method to convert low-frequency to high-frequency data.

    Uses proportionate first-differences as the penalty function.  See notes.

    Parameters
    ----------
    indicator : array_like
        A low-frequency indicator series.  It is assumed that there are no
        pre-sample indicators.  Ie., the first indicators line up with
        the first benchmark.
    benchmark : array_like
        The higher frequency benchmark.  A 1d or 2d data series in columns.
        If 2d, then M series are assumed.
    freq : str {"aq","qm", "other"}
        The frequency to use in the conversion.

        * "aq" - Benchmarking an annual series to quarterly.
        * "mq" - Benchmarking a quarterly series to monthly.
        * "other" - Custom stride.  A kwarg, k, must be supplied.
    **kwargs
        Additional keyword argument. For example:

        * k, an int, the number of high-frequency observations that sum to make
          an aggregate low-frequency observation. `k` is used with
          `freq` == "other".

    Returns
    -------
    transformed : ndarray
        The transformed series.

    Examples
    --------
    >>> indicator = [50,100,150,100] * 5
    >>> benchmark = [500,400,300,400,500]
    >>> benchmarked = dentonm(indicator, benchmark, freq="aq")

    Notes
    -----
    Denton's method minimizes the distance given by the penalty function, in
    a least squares sense, between the unknown benchmarked series and the
    indicator series subject to the condition that the sum of the benchmarked
    series is equal to the benchmark. The modification allows that the first
    value not be pre-determined as is the case with Denton's original method.
    If the there is no benchmark provided for the last few indicator
    observations, then extrapolation is performed using the last
    benchmark-indicator ratio of the previous period.

    Minimizes sum((X[t]/I[t] - X[t-1]/I[t-1])**2)

    s.t.

    sum(X) = A, for each period.  Where X is the benchmarked series, I is
    the indicator, and A is the benchmark.

    References
    ----------
    Bloem, A.M, Dippelsman, R.J. and Maehle, N.O.  2001 Quarterly National
        Accounts Manual--Concepts, Data Sources, and Compilation. IMF.
        http://www.imf.org/external/pubs/ft/qna/2000/Textbook/index.htm
    Cholette, P. 1988. "Benchmarking systems of socio-economic time series."
        Statistics Canada, Time Series Research and Analysis Division,
        Working Paper No TSRA-88-017E.
    Denton, F.T. 1971. "Adjustment of monthly or quarterly series to annual
        totals: an approach based on quadratic minimization." Journal of the
        American Statistical Association. 99-102.
    """
#    penalty : str
#        Penalty function.  Can be "D1", "D2", "D3", "D4", "D5".
#        X is the benchmarked series and I is the indicator.
#        D1 - sum((X[t] - X[t-1]) - (I[t] - I[ti-1])**2)
#        D2 - sum((ln(X[t]/X[t-1]) - ln(I[t]/I[t-1]))**2)
#        D3 - sum((X[t]/X[t-1] / I[t]/I[t-1])**2)
#        D4 - sum((X[t]/I[t] - X[t-1]/I[t-1])**2)
#        D5 - sum((X[t]/I[t] / X[t-1]/I[t-1] - 1)**2)
#NOTE: only D4 is the only one implemented, see IMF chapter 6.

    # check arrays and make 2d
    indicator = asarray(indicator)
    if indicator.ndim == 1:
        indicator = indicator[:,None]
    benchmark = asarray(benchmark)
    if benchmark.ndim == 1:
        benchmark = benchmark[:,None]

    # get dimensions
    N = len(indicator) # total number of high-freq
    m = len(benchmark) # total number of low-freq

    # number of low-freq observations for aggregate measure
    # 4 for annual to quarter and 3 for quarter to monthly
    if freq == "aq":
        k = 4
    elif freq == "qm":
        k = 3
    elif freq == "other":
        k = kwargs.get("k")
        if not k:
            raise ValueError("k must be supplied with freq=\"other\"")
    else:
        raise ValueError("freq %s not understood" % freq)

    n = k*m # number of indicator series with a benchmark for back-series
    # if k*m != n, then we are going to extrapolate q observations
    if N > n:
        q = N - n
    else:
        q = 0

    # make the aggregator matrix
    #B = block_diag(*(ones((k,1)),)*m)
    B = np.kron(np.eye(m), ones((k,1)))

    # following the IMF paper, we can do
    Zinv = diag(1./indicator.squeeze()[:n])
    # this is D in Denton's notation (not using initial value correction)
#    D = eye(n)
    # make off-diagonal = -1
#    D[((np.diag_indices(n)[0])[:-1]+1,(np.diag_indices(n)[1])[:-1])] = -1
    # account for starting conditions
#    H = D[1:,:]
#    HTH = dot(H.T,H)
    # just make HTH
    HTH = eye(n)
    diag_idx0, diag_idx1 = diag_indices(n)
    HTH[diag_idx0[1:-1], diag_idx1[1:-1]] += 1
    HTH[diag_idx0[:-1]+1, diag_idx1[:-1]] = -1
    HTH[diag_idx0[:-1], diag_idx1[:-1]+1] = -1

    W = dot(dot(Zinv,HTH),Zinv)

    # make partitioned matrices
    # TODO: break this out so that we can simplify the linalg?
    I = zeros((n+m, n+m))  # noqa:E741
    I[:n,:n] = W
    I[:n,n:] = B
    I[n:,:n] = B.T

    A = zeros((m+n,1)) # zero first-order constraints
    A[-m:] = benchmark # adding up constraints
    X = solve(I,A)
    X = X[:-m]  # drop the lagrange multipliers

    # handle extrapolation
    if q > 0:
        # get last Benchmark-Indicator ratio
        bi = X[n-1]/indicator[n-1]
        extrapolated = bi * indicator[n:]
        X = r_[X,extrapolated]

    return X.squeeze()


if __name__ == "__main__":
    #these will be the tests
    # from IMF paper

    # quarterly data
    indicator = np.array([98.2, 100.8, 102.2, 100.8, 99.0, 101.6,
                          102.7, 101.5, 100.5, 103.0, 103.5, 101.5])
    # two annual observations
    benchmark = np.array([4000.,4161.4])
    x_imf = dentonm(indicator, benchmark, freq="aq")

    imf_stata = np.array([969.8, 998.4, 1018.3, 1013.4, 1007.2, 1042.9,
                                1060.3, 1051.0, 1040.6, 1066.5, 1071.7, 1051.0])
    np.testing.assert_almost_equal(imf_stata, x_imf, 1)

    # Denton example
    zQ = np.array([50,100,150,100] * 5)
    Y = np.array([500,400,300,400,500])
    x_denton = dentonm(zQ, Y, freq="aq")
    x_stata = np.array([64.334796,127.80616,187.82379,120.03526,56.563894,
                    105.97568,147.50144,89.958987,40.547201,74.445963,
                    108.34473,76.66211,42.763347,94.14664,153.41596,
                    109.67405,58.290761,122.62556,190.41409,128.66959])


"""
# Examples from the Denton 1971 paper
k = 4
m = 5
n = m*k

zQ = [50,100,150,100] * m
Y = [500,400,300,400,500]

A = np.eye(n)
B = block_diag(*(np.ones((k,1)),)*m)

r = Y - B.T.dot(zQ)
#Ainv = inv(A)
Ainv = A # shortcut for identity
C = Ainv.dot(B).dot(inv(B.T.dot(Ainv).dot(B)))
x = zQ + C.dot(r)

# minimize first difference d(x-z)
R = linalg.tri(n, dtype=float) # R is tril so actually R.T in paper
Ainv = R.dot(R.T)
C = Ainv.dot(B).dot(inv(B.T.dot(Ainv).dot(B)))
x1 = zQ + C.dot(r)

# minimize the second difference d**2(x-z)
Ainv = R.dot(Ainv).dot(R.T)
C = Ainv.dot(B).dot(inv(B.T.dot(Ainv).dot(B)))
x12 = zQ + C.dot(r)


# # do it proportionately (x-z)/z
Z = np.diag(zQ)
Ainv = np.eye(n)
C = Z.dot(Ainv).dot(Z).dot(B).dot(inv(B.T.dot(Z).dot(Ainv).dot(Z).dot(B)))
x11 = zQ + C.dot(r)

# do it proportionately with differencing d((x-z)/z)
Ainv = R.dot(R.T)
C = Z.dot(Ainv).dot(Z).dot(B).dot(inv(B.T.dot(Z).dot(Ainv).dot(Z).dot(B)))
x111 = zQ + C.dot(r)

x_stata = np.array([64.334796,127.80616,187.82379,120.03526,56.563894,
                    105.97568,147.50144,89.958987,40.547201,74.445963,
                    108.34473,76.66211,42.763347,94.14664,153.41596,
                    109.67405,58.290761,122.62556,190.41409,128.66959])
"""
