"""
A predict-like function that constructs means and pointwise or
simultaneous confidence bands for the function f(x) = E[Y | X*=x,
X1=x1, ...], where X* is the focus variable and X1, X2, ... are
non-focus variables.  This is especially useful when conducting a
functional regression in which the role of x is modeled with b-splines
or other basis functions.
"""
import pandas as pd
import patsy
import numpy as np
import warnings

from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.compat.pandas import Appender

_predict_functional_doc =\
    """
    Predictions and contrasts of a fitted model as a function of a given covariate.

    The value of the focus variable varies along a sequence of its
    quantiles, calculated from the data used to fit the model.  The
    other variables are held constant either at given values, or at
    values obtained by applying given summary functions to the data
    used to fit the model.  Optionally, a second specification of the
    non-focus variables is provided and the contrast between the two
    specifications is returned.

    Parameters
    ----------
    result : statsmodels result object
        A results object for the fitted model.
    focus_var : str
        The name of the 'focus variable'.
    summaries : dict-like
        A map from names of non-focus variables to summary functions.
        Each summary function is applied to the data used to fit the
        model, to obtain a value at which the variable is held fixed.
    values : dict-like
        Values at which a given non-focus variable is held fixed.
    summaries2 : dict-like
        A second set of summary functions used to define a contrast.
    values2 : dict-like
        A second set of fixed values used to define a contrast.
    alpha : float
        `1 - alpha` is the coverage probability.
    ci_method : str
        The method for constructing the confidence band, one of
        'pointwise', 'scheffe', and 'simultaneous'.
    num_points : int
        The number of equally-spaced quantile points where the
        prediction is made.
    exog : array_like
        Explicitly provide points to cover with the confidence band.
    exog2 : array_like
        Explicitly provide points to contrast to `exog` in a functional
        confidence band.
    kwargs :
        Arguments passed to the `predict` method.

    Returns
    -------
    pred : array_like
        The predicted mean values.
    cb : array_like
        An array with two columns, containing respectively the lower
        and upper limits of a confidence band.
    fvals : array_like
        The values of the focus variable at which the prediction is
        made.

    Notes
    -----
    All variables in the model except for the focus variable should be
    included as a key in either `summaries` or `values` (unless `exog`
    is provided).

    If `summaries2` and `values2` are not provided, the returned value
    contains predicted conditional means for the outcome as the focus
    variable varies, with the other variables fixed as specified.

    If `summaries2` and/or `values2` is provided, two sets of
    predicted conditional means are calculated, and the returned value
    is the contrast between them.

    If `exog` is provided, then the rows should contain a sequence of
    values approximating a continuous path through the domain of the
    covariates.  For example, if Z(s) is the covariate expressed as a
    function of s, then the rows of exog may approximate Z(g(s)) for
    some continuous function g.  If `exog` is provided then neither of
    the summaries or values arguments should be provided.  If `exog2`
    is also provided, then the returned value is a contrast between
    the functionas defined by `exog` and `exog2`.

    Examples
    --------
    Fit a model using a formula in which the predictors are age
    (modeled with splines), ethnicity (which is categorical), gender,
    and income.  Then we obtain the fitted mean values as a function
    of age for females with mean income and the most common
    ethnicity.

    >>> model = sm.OLS.from_formula('y ~ bs(age, df=4) + C(ethnicity) + gender + income', data)
    >>> result = model.fit()
    >>> mode = lambda x : x.value_counts().argmax()
    >>> summaries = {'income': np.mean, ethnicity=mode}
    >>> values = {'gender': 'female'}
    >>> pr, cb, x = predict_functional(result, 'age', summaries, values)

    Fit a model using arrays.  Plot the means as a function of x3,
    holding x1 fixed at its mean value in the data used to fit the
    model, and holding x2 fixed at 1.

    >>> model = sm.OLS(y ,x)
    >>> result = model.fit()
    >>> summaries = {'x1': np.mean}
    >>> values = {'x2': 1}
    >>> pr, cb, x = predict_functional(result, 'x3', summaries, values)

    Fit a model usng a formula and construct a contrast comparing the
    female and male predicted mean functions.

    >>> model = sm.OLS.from_formula('y ~ bs(age, df=4) + gender', data)
    >>> result = model.fit()
    >>> values = {'gender': 'female'}
    >>> values2 = {'gender': 'male'}
    >>> pr, cb, x = predict_functional(result, 'age', values=values, values2=values2)
    """


def _make_exog_from_formula(result, focus_var, summaries, values, num_points):
    """
    Create dataframes for exploring a fitted model as a function of one variable.

    This works for models fit with a formula.

    Returns
    -------
    dexog : data frame
        A data frame in which the focus variable varies and the other variables
        are fixed at specified or computed values.
    fexog : data frame
        The data frame `dexog` processed through the model formula.
    """

    model = result.model
    exog = model.data.frame

    if summaries is None:
        summaries = {}
    if values is None:
        values = {}

    if exog[focus_var].dtype is np.dtype('O'):
        raise ValueError('focus variable may not have object type')

    colnames = list(summaries.keys()) + list(values.keys()) + [focus_var]
    dtypes = [exog[x].dtype for x in colnames]

    # Check for variables whose values are not set either through
    # `values` or `summaries`.  Since the model data frame can contain
    # extra variables not referenced in the formula RHS, this may not
    # be a problem, so just warn.  There is no obvious way to extract
    # from a formula all the variable names that it references.
    varl = set(exog.columns.tolist()) - set([model.endog_names])
    unmatched = varl - set(colnames)
    unmatched = list(unmatched)
    if len(unmatched) > 0:
        warnings.warn("%s in data frame but not in summaries or values."
                      % ", ".join(["'%s'" % x for x in unmatched]),
                      ValueWarning)

    # Initialize at zero so each column can be converted to any dtype.
    ix = range(num_points)
    fexog = pd.DataFrame(index=ix, columns=colnames)
    for d, x in zip(dtypes, colnames):
        fexog[x] = pd.Series(index=ix, dtype=d)

    # The values of the 'focus variable' are a sequence of percentiles
    pctls = np.linspace(0, 100, num_points).tolist()
    fvals = np.percentile(exog[focus_var], pctls)
    fvals = np.asarray(fvals)
    fexog.loc[:, focus_var] = fvals

    # The values of the other variables may be given by summary functions...
    for ky in summaries.keys():
        fexog.loc[:, ky] = summaries[ky](exog.loc[:, ky])

    # or they may be provided as given values.
    for ky in values.keys():
        fexog[ky] = values[ky]

    dexog = patsy.dmatrix(model.data.design_info, fexog,
                          return_type='dataframe')
    return dexog, fexog, fvals


def _make_exog_from_arrays(result, focus_var, summaries, values, num_points):
    """
    Create dataframes for exploring a fitted model as a function of one variable.

    This works for models fit without a formula.

    Returns
    -------
    exog : data frame
        A data frame in which the focus variable varies and the other variables
        are fixed at specified or computed values.
    """

    model = result.model
    model_exog = model.exog
    exog_names = model.exog_names

    if summaries is None:
        summaries = {}
    if values is None:
        values = {}

    exog = np.zeros((num_points, model_exog.shape[1]))

    # Check for variables whose values are not set either through
    # `values` or `summaries`.
    colnames = list(values.keys()) + list(summaries.keys()) + [focus_var]
    unmatched = set(exog_names) - set(colnames)
    unmatched = list(unmatched)
    if len(unmatched) > 0:
        warnings.warn("%s in model but not in `summaries` or `values`."
                      % ", ".join(["'%s'" % x for x in unmatched]),
                      ValueWarning)

    # The values of the 'focus variable' are a sequence of percentiles
    pctls = np.linspace(0, 100, num_points).tolist()
    ix = exog_names.index(focus_var)
    fvals = np.percentile(model_exog[:, ix], pctls)
    exog[:, ix] = fvals

    # The values of the other variables may be given by summary functions...
    for ky in summaries.keys():
        ix = exog_names.index(ky)
        exog[:, ix] = summaries[ky](model_exog[:, ix])

    # or they may be provided as given values.
    for ky in values.keys():
        ix = exog_names.index(ky)
        exog[:, ix] = values[ky]

    return exog, fvals


def _make_exog(result, focus_var, summaries, values, num_points):

    # Branch depending on whether the model was fit with a formula.
    if hasattr(result.model.data, "frame"):
        dexog, fexog, fvals = _make_exog_from_formula(result, focus_var,
                                       summaries, values, num_points)
    else:
        exog, fvals = _make_exog_from_arrays(result, focus_var, summaries,
                                 values, num_points)
        dexog, fexog = exog, exog

    return dexog, fexog, fvals


def _check_args(values, summaries, values2, summaries2):

    if values is None:
        values = {}
    if values2 is None:
        values2 = {}
    if summaries is None:
        summaries = {}
    if summaries2 is None:
        summaries2 = {}

    for (s,v) in (summaries, values), (summaries2, values2):
        ky = set(v.keys()) & set(s.keys())
        ky = list(ky)
        if len(ky) > 0:
            raise ValueError("One or more variable names are contained in both `summaries` and `values`:" +
                             ", ".join(ky))

    return values, summaries, values2, summaries2


@Appender(_predict_functional_doc)
def predict_functional(result, focus_var, summaries=None, values=None,
                       summaries2=None, values2=None, alpha=0.05,
                       ci_method="pointwise", linear=True, num_points=10,
                       exog=None, exog2=None, **kwargs):

    if ci_method not in ("pointwise", "scheffe", "simultaneous"):
        raise ValueError('confidence band method must be one of '
                         '`pointwise`, `scheffe`, and `simultaneous`.')

    contrast = (values2 is not None) or (summaries2 is not None)

    if contrast and not linear:
        raise ValueError("`linear` must be True for computing contrasts")

    model = result.model
    if exog is not None:

        if any(x is not None for x in [summaries, summaries2, values, values2]):
            raise ValueError("if `exog` is provided then do not "
                             "provide `summaries` or `values`")

        fexog = exog
        dexog = patsy.dmatrix(model.data.design_info,
                              fexog, return_type='dataframe')
        fvals = exog[focus_var]

        if exog2 is not None:
            fexog2 = exog
            dexog2 = patsy.dmatrix(model.data.design_info,
                                   fexog2, return_type='dataframe')
            fvals2 = fvals

    else:

        values, summaries, values2, summaries2 = _check_args(values,
                                                             summaries,
                                                             values2,
                                                             summaries2)

        dexog, fexog, fvals = _make_exog(result, focus_var, summaries,
                                         values, num_points)

        if len(summaries2) + len(values2) > 0:
            dexog2, fexog2, fvals2 = _make_exog(result, focus_var, summaries2,
                                                values2, num_points)

    from statsmodels.genmod.generalized_linear_model import GLM
    from statsmodels.genmod.generalized_estimating_equations import GEE
    if isinstance(result.model, (GLM, GEE)):
        kwargs_pred = kwargs.copy()
        kwargs_pred.update({"which": "linear"})
    else:
        kwargs_pred = kwargs

    pred = result.predict(exog=fexog, **kwargs_pred)
    if contrast:
        pred2 = result.predict(exog=fexog2, **kwargs_pred)
        pred = pred - pred2
        dexog = dexog - dexog2

    if ci_method == 'pointwise':

        t_test = result.t_test(dexog)
        cb = t_test.conf_int(alpha=alpha)

    elif ci_method == 'scheffe':

        t_test = result.t_test(dexog)
        sd = t_test.sd
        cb = np.zeros((num_points, 2))

        # Scheffe's method
        from scipy.stats.distributions import f as fdist
        df1 = result.model.exog.shape[1]
        df2 = result.model.exog.shape[0] - df1
        qf = fdist.cdf(1 - alpha, df1, df2)
        fx = sd * np.sqrt(df1 * qf)
        cb[:, 0] = pred - fx
        cb[:, 1] = pred + fx

    elif ci_method == 'simultaneous':

        sigma, c = _glm_basic_scr(result, dexog, alpha)
        cb = np.zeros((dexog.shape[0], 2))
        cb[:, 0] = pred - c*sigma
        cb[:, 1] = pred + c*sigma

    if not linear:
        # May need to support other models with link-like functions.
        link = result.family.link
        pred = link.inverse(pred)
        cb = link.inverse(cb)

    return pred, cb, fvals


def _glm_basic_scr(result, exog, alpha):
    """
    The basic SCR from (Sun et al. Annals of Statistics 2000).

    Computes simultaneous confidence regions (SCR).

    Parameters
    ----------
    result : results instance
        The fitted GLM results instance
    exog : array_like
        The exog values spanning the interval
    alpha : float
        `1 - alpha` is the coverage probability.

    Returns
    -------
    An array with two columns, containing the lower and upper
    confidence bounds, respectively.

    Notes
    -----
    The rows of `exog` should be a sequence of covariate values
    obtained by taking one 'free variable' x and varying it over an
    interval.  The matrix `exog` is thus the basis functions and any
    other covariates evaluated as x varies.
    """

    model = result.model
    n = model.exog.shape[0]

    # Get the Hessian without recomputing.
    cov = result.cov_params()
    hess = np.linalg.inv(cov)

    # Proposition 3.1 of Sun et al.
    A = hess / n
    B = np.linalg.cholesky(A).T # Upper Cholesky triangle

    # The variance and SD of the linear predictor at each row of exog.
    sigma2 = (np.dot(exog, cov) * exog).sum(1)
    sigma = np.asarray(np.sqrt(sigma2))

    # Calculate kappa_0 (formula 42 from Sun et al)
    bz = np.linalg.solve(B.T, exog.T).T
    bz /= np.sqrt(n)
    bz /= sigma[:, None]
    bzd = np.diff(bz, 1, axis=0)
    bzdn = (bzd**2).sum(1)
    kappa_0 = np.sqrt(bzdn).sum()

    from scipy.stats.distributions import norm

    # The root of this function is the multiplier for the confidence
    # band, see Sun et al. equation 35.
    def func(c):
        return kappa_0 * np.exp(-c**2/2) / np.pi + 2*(1 - norm.cdf(c)) - alpha

    from scipy.optimize import brentq

    c, rslt = brentq(func, 1, 10, full_output=True)
    if not rslt.converged:
        raise ValueError("Root finding error in basic SCR")

    return sigma, c
