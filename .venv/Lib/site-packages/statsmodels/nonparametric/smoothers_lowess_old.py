"""
Univariate lowess function, like in R.

References
----------
Hastie, Tibshirani, Friedman. (2009) The Elements of Statistical Learning: Data Mining, Inference, and Prediction, Second Edition: Chapter 6.

Cleveland, W.S. (1979) "Robust Locally Weighted Regression and Smoothing Scatterplots". Journal of the American Statistical Association 74 (368): 829-836.
"""
import numpy as np
from numpy.linalg import lstsq


def lowess(endog, exog, frac=2./3, it=3):
    """
    LOWESS (Locally Weighted Scatterplot Smoothing)

    A lowess function that outs smoothed estimates of endog
    at the given exog values from points (exog, endog)

    Parameters
    ----------
    endog : 1-D numpy array
        The y-values of the observed points
    exog : 1-D numpy array
        The x-values of the observed points
    frac : float
        Between 0 and 1. The fraction of the data used
        when estimating each y-value.
    it : int
        The number of residual-based reweightings
        to perform.

    Returns
    -------
    out: numpy array
        A numpy array with two columns. The first column
        is the sorted x values and the second column the
        associated estimated y-values.

    Notes
    -----
    This lowess function implements the algorithm given in the
    reference below using local linear estimates.

    Suppose the input data has N points. The algorithm works by
    estimating the true ``y_i`` by taking the frac*N closest points
    to ``(x_i,y_i)`` based on their x values and estimating ``y_i``
    using a weighted linear regression. The weight for ``(x_j,y_j)``
    is `_lowess_tricube` function applied to ``|x_i-x_j|``.

    If ``iter > 0``, then further weighted local linear regressions
    are performed, where the weights are the same as above
    times the `_lowess_bisquare` function of the residuals. Each iteration
    takes approximately the same amount of time as the original fit,
    so these iterations are expensive. They are most useful when
    the noise has extremely heavy tails, such as Cauchy noise.
    Noise with less heavy-tails, such as t-distributions with ``df > 2``,
    are less problematic. The weights downgrade the influence of
    points with large residuals. In the extreme case, points whose
    residuals are larger than 6 times the median absolute residual
    are given weight 0.

    Some experimentation is likely required to find a good
    choice of frac and iter for a particular dataset.

    References
    ----------
    Cleveland, W.S. (1979) "Robust Locally Weighted Regression
    and Smoothing Scatterplots". Journal of the American Statistical
    Association 74 (368): 829-836.

    Examples
    --------
    The below allows a comparison between how different the fits from
    `lowess` for different values of frac can be.

    >>> import numpy as np
    >>> import statsmodels.api as sm
    >>> lowess = sm.nonparametric.lowess
    >>> x = np.random.uniform(low=-2*np.pi, high=2*np.pi, size=500)
    >>> y = np.sin(x) + np.random.normal(size=len(x))
    >>> z = lowess(y, x)
    >>> w = lowess(y, x, frac=1./3)

    This gives a similar comparison for when it is 0 vs not.

    >>> import scipy.stats as stats
    >>> x = np.random.uniform(low=-2*np.pi, high=2*np.pi, size=500)
    >>> y = np.sin(x) + stats.cauchy.rvs(size=len(x))
    >>> z = lowess(y, x, frac= 1./3, it=0)
    >>> w = lowess(y, x, frac=1./3)
    """
    x = exog

    if exog.ndim != 1:
        raise ValueError('exog must be a vector')
    if endog.ndim != 1:
        raise ValueError('endog must be a vector')
    if endog.shape[0] != x.shape[0] :
        raise ValueError('exog and endog must have same length')

    n = exog.shape[0]
    fitted = np.zeros(n)

    k = int(frac * n)

    index_array = np.argsort(exog)
    x_copy = np.array(exog[index_array]) #, dtype ='float32')
    y_copy = endog[index_array]

    fitted, weights = _lowess_initial_fit(x_copy, y_copy, k, n)

    for i in range(it):
        _lowess_robustify_fit(x_copy, y_copy, fitted,
                              weights, k, n)

    out = np.array([x_copy, fitted]).T
    out.shape = (n,2)

    return out


def _lowess_initial_fit(x_copy, y_copy, k, n):
    """
    The initial weighted local linear regression for lowess.

    Parameters
    ----------
    x_copy : 1-d ndarray
        The x-values/exogenous part of the data being smoothed
    y_copy : 1-d ndarray
        The y-values/ endogenous part of the data being smoothed
   k : int
        The number of data points which affect the linear fit for
        each estimated point
    n : int
        The total number of points

    Returns
    -------
    fitted : 1-d ndarray
        The fitted y-values
    weights : 2-d ndarray
        An n by k array. The contribution to the weights in the
        local linear fit coming from the distances between the
        x-values

   """
    weights = np.zeros((n,k), dtype = x_copy.dtype)
    nn_indices = [0,k]

    X = np.ones((k,2))
    fitted = np.zeros(n)

    for i in range(n):
        #note: all _lowess functions are inplace, no return
        left_width = x_copy[i] - x_copy[nn_indices[0]]
        right_width = x_copy[nn_indices[1]-1] - x_copy[i]
        width = max(left_width, right_width)
        _lowess_wt_standardize(weights[i,:],
                                x_copy[nn_indices[0]:nn_indices[1]],
                            x_copy[i], width)
        _lowess_tricube(weights[i,:])
        weights[i,:] = np.sqrt(weights[i,:])

        X[:,1] = x_copy[nn_indices[0]:nn_indices[1]]
        y_i = weights[i,:] * y_copy[nn_indices[0]:nn_indices[1]]

        beta = lstsq(weights[i,:].reshape(k,1) * X, y_i, rcond=-1)[0]
        fitted[i] = beta[0] + beta[1]*x_copy[i]

        _lowess_update_nn(x_copy, nn_indices, i+1)


    return fitted, weights


def _lowess_wt_standardize(weights, new_entries, x_copy_i, width):
    """
    The initial phase of creating the weights.
    Subtract the current x_i and divide by the width.

    Parameters
    ----------
    weights : ndarray
        The memory where (new_entries - x_copy_i)/width will be placed
    new_entries : ndarray
        The x-values of the k closest points to x[i]
    x_copy_i : float
        x[i], the i'th point in the (sorted) x values
    width : float
        The maximum distance between x[i] and any point in new_entries

    Returns
    -------
    Nothing. The modifications are made to weight in place.
    """
    weights[:] = new_entries
    weights -= x_copy_i
    weights /= width


def _lowess_robustify_fit(x_copy, y_copy, fitted, weights, k, n):
    """
    Additional weighted local linear regressions, performed if
    iter>0. They take into account the sizes of the residuals,
    to eliminate the effect of extreme outliers.

    Parameters
    ----------
    x_copy : 1-d ndarray
        The x-values/exogenous part of the data being smoothed
    y_copy : 1-d ndarray
        The y-values/ endogenous part of the data being smoothed
    fitted : 1-d ndarray
        The fitted y-values from the previous iteration
    weights : 2-d ndarray
        An n by k array. The contribution to the weights in the
        local linear fit coming from the distances between the
        x-values
    k : int
        The number of data points which affect the linear fit for
        each estimated point
    n : int
        The total number of points

   Returns
    -------
    Nothing. The fitted values are modified in place.
    """
    nn_indices = [0,k]
    X = np.ones((k,2))

    residual_weights = np.copy(y_copy)
    residual_weights.shape = (n,)
    residual_weights -= fitted
    residual_weights = np.absolute(residual_weights)#, out=residual_weights)
    s = np.median(residual_weights)
    residual_weights /= (6*s)
    too_big = residual_weights>=1
    _lowess_bisquare(residual_weights)
    residual_weights[too_big] = 0


    for i in range(n):
        total_weights = weights[i,:] * np.sqrt(residual_weights[nn_indices[0]:
                                                        nn_indices[1]])

        X[:,1] = x_copy[nn_indices[0]:nn_indices[1]]
        y_i = total_weights * y_copy[nn_indices[0]:nn_indices[1]]
        total_weights.shape = (k,1)

        beta = lstsq(total_weights * X, y_i, rcond=-1)[0]

        fitted[i] = beta[0] + beta[1] * x_copy[i]

        _lowess_update_nn(x_copy, nn_indices, i+1)


def _lowess_update_nn(x, cur_nn,i):
    """
    Update the endpoints of the nearest neighbors to
    the ith point.

    Parameters
    ----------
    x : iterable
        The sorted points of x-values
    cur_nn : list of length 2
        The two current indices between which are the
        k closest points to x[i]. (The actual value of
        k is irrelevant for the algorithm.
    i : int
        The index of the current value in x for which
        the k closest points are desired.

    Returns
    -------
    Nothing. It modifies cur_nn in place.
    """
    while True:
        if cur_nn[1]<x.size:
            left_dist = x[i] - x[cur_nn[0]]
            new_right_dist = x[cur_nn[1]] - x[i]
            if new_right_dist < left_dist:
                cur_nn[0] = cur_nn[0] + 1
                cur_nn[1] = cur_nn[1] + 1
            else:
                break
        else:
            break


def _lowess_tricube(t):
    """
    The _tricube function applied to a numpy array.
    The tricube function is (1-abs(t)**3)**3.

    Parameters
    ----------
    t : ndarray
        Array the tricube function is applied to elementwise and
        in-place.

    Returns
    -------
    Nothing
    """
    #t = (1-np.abs(t)**3)**3
    t[:] = np.absolute(t) #, out=t) #numpy version?
    _lowess_mycube(t)
    t[:] = np.negative(t) #, out = t)
    t += 1
    _lowess_mycube(t)


def _lowess_mycube(t):
    """
    Fast matrix cube

    Parameters
    ----------
    t : ndarray
        Array that is cubed, elementwise and in-place

    Returns
    -------
    Nothing
    """
    #t **= 3
    t2 = t*t
    t *= t2


def _lowess_bisquare(t):
    """
    The bisquare function applied to a numpy array.
    The bisquare function is (1-t**2)**2.

    Parameters
    ----------
    t : ndarray
        array bisquare function is applied to, element-wise and in-place.

    Returns
    -------
    Nothing
    """
    #t = (1-t**2)**2
    t *= t
    t[:] = np.negative(t) #, out=t)
    t += 1
    t *= t
