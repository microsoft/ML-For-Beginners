"""
Empirical CDF Functions
"""
import numpy as np
from scipy.interpolate import interp1d


def _conf_set(F, alpha=.05):
    r"""
    Constructs a Dvoretzky-Kiefer-Wolfowitz confidence band for the eCDF.

    Parameters
    ----------
    F : array_like
        The empirical distributions
    alpha : float
        Set alpha for a (1 - alpha) % confidence band.

    Notes
    -----
    Based on the DKW inequality.

    .. math:: P \left( \sup_x \left| F(x) - \hat(F)_n(X) \right| >
       \epsilon \right) \leq 2e^{-2n\epsilon^2}

    References
    ----------
    Wasserman, L. 2006. `All of Nonparametric Statistics`. Springer.
    """
    nobs = len(F)
    epsilon = np.sqrt(np.log(2./alpha) / (2 * nobs))
    lower = np.clip(F - epsilon, 0, 1)
    upper = np.clip(F + epsilon, 0, 1)
    return lower, upper


class StepFunction:
    """
    A basic step function.

    Values at the ends are handled in the simplest way possible:
    everything to the left of x[0] is set to ival; everything
    to the right of x[-1] is set to y[-1].

    Parameters
    ----------
    x : array_like
    y : array_like
    ival : float
        ival is the value given to the values to the left of x[0]. Default
        is 0.
    sorted : bool
        Default is False.
    side : {'left', 'right'}, optional
        Default is 'left'. Defines the shape of the intervals constituting the
        steps. 'right' correspond to [a, b) intervals and 'left' to (a, b].

    Examples
    --------
    >>> import numpy as np
    >>> from statsmodels.distributions.empirical_distribution import (
    >>>     StepFunction)
    >>>
    >>> x = np.arange(20)
    >>> y = np.arange(20)
    >>> f = StepFunction(x, y)
    >>>
    >>> print(f(3.2))
    3.0
    >>> print(f([[3.2,4.5],[24,-3.1]]))
    [[  3.   4.]
     [ 19.   0.]]
    >>> f2 = StepFunction(x, y, side='right')
    >>>
    >>> print(f(3.0))
    2.0
    >>> print(f2(3.0))
    3.0
    """

    def __init__(self, x, y, ival=0., sorted=False, side='left'):  # noqa

        if side.lower() not in ['right', 'left']:
            msg = "side can take the values 'right' or 'left'"
            raise ValueError(msg)
        self.side = side

        _x = np.asarray(x)
        _y = np.asarray(y)

        if _x.shape != _y.shape:
            msg = "x and y do not have the same shape"
            raise ValueError(msg)
        if len(_x.shape) != 1:
            msg = 'x and y must be 1-dimensional'
            raise ValueError(msg)

        self.x = np.r_[-np.inf, _x]
        self.y = np.r_[ival, _y]

        if not sorted:
            asort = np.argsort(self.x)
            self.x = np.take(self.x, asort, 0)
            self.y = np.take(self.y, asort, 0)
        self.n = self.x.shape[0]

    def __call__(self, time):

        tind = np.searchsorted(self.x, time, self.side) - 1
        return self.y[tind]


class ECDF(StepFunction):
    """
    Return the Empirical CDF of an array as a step function.

    Parameters
    ----------
    x : array_like
        Observations
    side : {'left', 'right'}, optional
        Default is 'right'. Defines the shape of the intervals constituting the
        steps. 'right' correspond to [a, b) intervals and 'left' to (a, b].

    Returns
    -------
    Empirical CDF as a step function.

    Examples
    --------
    >>> import numpy as np
    >>> from statsmodels.distributions.empirical_distribution import ECDF
    >>>
    >>> ecdf = ECDF([3, 3, 1, 4])
    >>>
    >>> ecdf([3, 55, 0.5, 1.5])
    array([ 0.75,  1.  ,  0.  ,  0.25])
    """
    def __init__(self, x, side='right'):
        x = np.array(x, copy=True)
        x.sort()
        nobs = len(x)
        y = np.linspace(1./nobs, 1, nobs)
        super(ECDF, self).__init__(x, y, side=side, sorted=True)
        # TODO: make `step` an arg and have a linear interpolation option?
        # This is the path with `step` is True
        # If `step` is False, a previous version of the code read
        #  `return interp1d(x,y,drop_errors=False,fill_values=ival)`
        # which would have raised a NameError if hit, so would need to be
        # fixed.  See GH#5701.


class ECDFDiscrete(StepFunction):
    """
    Return the Empirical Weighted CDF of an array as a step function.

    Parameters
    ----------
    x : array_like
        Data values. If freq_weights is None, then x is treated as observations
        and the ecdf is computed from the frequency counts of unique values
        using nunpy.unique.
        If freq_weights is not None, then x will be taken as the support of the
        mass point distribution with freq_weights as counts for x values.
        The x values can be arbitrary sortable values and need not be integers.
    freq_weights : array_like
        Weights of the observations.  sum(freq_weights) is interpreted as nobs
        for confint.
        If freq_weights is None, then the frequency counts for unique values
        will be computed from the data x.
    side : {'left', 'right'}, optional
        Default is 'right'. Defines the shape of the intervals constituting the
        steps. 'right' correspond to [a, b) intervals and 'left' to (a, b].

    Returns
    -------
    Weighted ECDF as a step function.

    Examples
    --------
    >>> import numpy as np
    >>> from statsmodels.distributions.empirical_distribution import (
    >>>     ECDFDiscrete)
    >>>
    >>> ewcdf = ECDFDiscrete([3, 3, 1, 4])
    >>> ewcdf([3, 55, 0.5, 1.5])
    array([0.75, 1.  , 0.  , 0.25])
    >>>
    >>> ewcdf = ECDFDiscrete([3, 1, 4], [1.25, 2.5, 5])
    >>>
    >>> ewcdf([3, 55, 0.5, 1.5])
    array([0.42857143, 1., 0. , 0.28571429])
    >>> print('e1 and e2 are equivalent ways of defining the same ECDF')
    e1 and e2 are equivalent ways of defining the same ECDF
    >>> e1 = ECDFDiscrete([3.5, 3.5, 1.5, 1, 4])
    >>> e2 = ECDFDiscrete([3.5, 1.5, 1, 4], freq_weights=[2, 1, 1, 1])
    >>> print(e1.x, e2.x)
    [-inf  1.   1.5  3.5  4. ] [-inf  1.   1.5  3.5  4. ]
    >>> print(e1.y, e2.y)
    [0.  0.2 0.4 0.8 1. ] [0.  0.2 0.4 0.8 1. ]
    """
    def __init__(self, x, freq_weights=None, side='right'):
        if freq_weights is None:
            x, freq_weights = np.unique(x, return_counts=True)
        else:
            x = np.asarray(x)
        assert len(freq_weights) == len(x)
        w = np.asarray(freq_weights)
        sw = np.sum(w)
        assert sw > 0
        ax = x.argsort()
        x = x[ax]
        y = np.cumsum(w[ax])
        y = y / sw
        super(ECDFDiscrete, self).__init__(x, y, side=side, sorted=True)


def monotone_fn_inverter(fn, x, vectorized=True, **keywords):
    """
    Given a monotone function fn (no checking is done to verify monotonicity)
    and a set of x values, return an linearly interpolated approximation
    to its inverse from its values on x.
    """
    x = np.asarray(x)
    if vectorized:
        y = fn(x, **keywords)
    else:
        y = []
        for _x in x:
            y.append(fn(_x, **keywords))
        y = np.array(y)

    a = np.argsort(y)

    return interp1d(y[a], x[a])
