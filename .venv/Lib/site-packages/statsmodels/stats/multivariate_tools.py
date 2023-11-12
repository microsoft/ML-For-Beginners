'''Tools for multivariate analysis


Author : Josef Perktold
License : BSD-3



TODO:

- names of functions, currently just "working titles"

'''



import numpy as np

from statsmodels.tools.tools import Bunch

def partial_project(endog, exog):
    '''helper function to get linear projection or partialling out of variables

    endog variables are projected on exog variables

    Parameters
    ----------
    endog : ndarray
        array of variables where the effect of exog is partialled out.
    exog : ndarray
        array of variables on which the endog variables are projected.

    Returns
    -------
    res : instance of Bunch with

        - params : OLS parameter estimates from projection of endog on exog
        - fittedvalues : predicted values of endog given exog
        - resid : residual of the regression, values of endog with effect of
          exog partialled out

    Notes
    -----
    This is no-frills mainly for internal calculations, no error checking or
    array conversion is performed, at least for now.

    '''
    x1, x2 = endog, exog
    params = np.linalg.pinv(x2).dot(x1)
    predicted = x2.dot(params)
    residual = x1 - predicted
    res = Bunch(params=params,
                fittedvalues=predicted,
                resid=residual)

    return res



def cancorr(x1, x2, demean=True, standardize=False):
    '''canonical correlation coefficient beween 2 arrays

    Parameters
    ----------
    x1, x2 : ndarrays, 2_D
        two 2-dimensional data arrays, observations in rows, variables in columns
    demean : bool
         If demean is true, then the mean is subtracted from each variable
    standardize : bool
         If standardize is true, then each variable is demeaned and divided by
         its standard deviation. Rescaling does not change the canonical
         correlation coefficients.

    Returns
    -------
    ccorr : ndarray, 1d
        canonical correlation coefficients, sorted from largest to smallest.
        Note, that these are the square root of the eigenvalues.

    Notes
    -----
    This is a helper function for other statistical functions. It only
    calculates the canonical correlation coefficients and does not do a full
    canoncial correlation analysis

    The canonical correlation coefficient is calculated with the generalized
    matrix inverse and does not raise an exception if one of the data arrays
    have less than full column rank.

    See Also
    --------
    cc_ranktest
    cc_stats
    CCA not yet

    '''
    #x, y = x1, x2
    if demean or standardize:
        x1 = (x1 - x1.mean(0))
        x2 = (x2 - x2.mean(0))

    if standardize:
        #std does not make a difference to canonical correlation coefficients
        x1 /= x1.std(0)
        x2 /= x2.std(0)

    t1 = np.linalg.pinv(x1).dot(x2)
    t2 = np.linalg.pinv(x2).dot(x1)
    m = t1.dot(t2)
    cc = np.sqrt(np.linalg.eigvals(m))
    cc.sort()
    return cc[::-1]


def cc_ranktest(x1, x2, demean=True, fullrank=False):
    '''rank tests based on smallest canonical correlation coefficients

    Anderson canonical correlations test (LM test) and
    Cragg-Donald test (Wald test)
    Assumes homoskedasticity and independent observations, overrejects if
    there is heteroscedasticity or autocorrelation.

    The Null Hypothesis is that the rank is k - 1, the alternative hypothesis
    is that the rank is at least k.


    Parameters
    ----------
    x1, x2 : ndarrays, 2_D
        two 2-dimensional data arrays, observations in rows, variables in columns
    demean : bool
         If demean is true, then the mean is subtracted from each variable.
    fullrank : bool
         If true, then only the test that the matrix has full rank is returned.
         If false, the test for all possible ranks are returned. However, no
         the p-values are not corrected for the multiplicity of tests.

    Returns
    -------
    value : float
        value of the test statistic
    p-value : float
        p-value for the test Null Hypothesis tha the smallest canonical
        correlation coefficient is zero. based on chi-square distribution
    df : int
        degrees of freedom for thechi-square distribution in the hypothesis test
    ccorr : ndarray, 1d
        All canonical correlation coefficients sorted from largest to smallest.

    Notes
    -----
    Degrees of freedom for the distribution of the test statistic are based on
    number of columns of x1 and x2 and not on their matrix rank.
    (I'm not sure yet what the interpretation of the test is if x1 or x2 are of
    reduced rank.)

    See Also
    --------
    cancorr
    cc_stats

    '''

    from scipy import stats

    nobs1, k1 = x1.shape
    nobs2, k2 = x2.shape

    cc = cancorr(x1, x2, demean=demean)
    cc2 = cc * cc
    if fullrank:
        df = np.abs(k1 - k2) + 1
        value = nobs1 * cc2[-1]
        w_value = nobs1 * (cc2[-1] / (1. - cc2[-1]))
        return value, stats.chi2.sf(value, df), df, cc, w_value, stats.chi2.sf(w_value, df)
    else:
        r = np.arange(min(k1, k2))[::-1]
        df = (k1 - r) * (k2 - r)
        values = nobs1 * cc2[::-1].cumsum()
        w_values = nobs1 * (cc2 / (1. - cc2))[::-1].cumsum()
        return values, stats.chi2.sf(values, df), df, cc, w_values, stats.chi2.sf(w_values, df)


def cc_stats(x1, x2, demean=True):
    '''MANOVA statistics based on canonical correlation coefficient

    Calculates Pillai's Trace, Wilk's Lambda, Hotelling's Trace and
    Roy's Largest Root.

    Parameters
    ----------
    x1, x2 : ndarrays, 2_D
        two 2-dimensional data arrays, observations in rows, variables in columns
    demean : bool
         If demean is true, then the mean is subtracted from each variable.

    Returns
    -------
    res : dict
        Dictionary containing the test statistics.

    Notes
    -----

    same as `canon` in Stata

    missing: F-statistics and p-values

    TODO: should return a results class instead
    produces nans sometimes, singular, perfect correlation of x1, x2 ?

    '''

    nobs1, k1 = x1.shape  # endogenous ?
    nobs2, k2 = x2.shape
    cc = cancorr(x1, x2, demean=demean)
    cc2 = cc**2
    lam = (cc2 / (1 - cc2))  # what if max cc2 is 1 ?
    # Problem: ccr might not care if x1 or x2 are reduced rank,
    #          but df will depend on rank
    df_model = k1 * k2  # df_hypothesis (we do not include mean in x1, x2)
    df_resid = k1 * (nobs1 - k2 - demean)
    s = min(df_model, k1)
    m = 0.5 * (df_model - k1)
    n = 0.5 * (df_resid - k1 - 1)

    df1 = k1 * df_model
    df2 = k2


    pt_value = cc2.sum()    # Pillai's trace
    wl_value = np.product(1 / (1 + lam))   # Wilk's Lambda
    ht_value = lam.sum()    # Hotelling's Trace
    rm_value = lam.max()    # Roy's largest root
    #from scipy import stats
    # what's the distribution, the test statistic ?
    res = {}
    res['canonical correlation coefficient'] = cc
    res['eigenvalues'] = lam
    res["Pillai's Trace"] = pt_value
    res["Wilk's Lambda"] = wl_value
    res["Hotelling's Trace"] = ht_value
    res["Roy's Largest Root"] = rm_value
    res['df_resid'] = df_resid
    res['df_m'] = m
    return res
