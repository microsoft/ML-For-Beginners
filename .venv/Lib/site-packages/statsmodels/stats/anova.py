import numpy as np
from scipy import stats
import pandas as pd
from pandas import DataFrame, Index
import patsy

from statsmodels.regression.linear_model import OLS
from statsmodels.compat.python import lrange
from statsmodels.formula.formulatools import (_remove_intercept_patsy,
                                    _has_intercept, _intercept_idx)
from statsmodels.iolib import summary2


def _get_covariance(model, robust):
    if robust is None:
        return model.cov_params()
    elif robust == "hc0":
        return model.cov_HC0
    elif robust == "hc1":
        return model.cov_HC1
    elif robust == "hc2":
        return model.cov_HC2
    elif robust == "hc3":
        return model.cov_HC3
    else:  # pragma: no cover
        raise ValueError("robust options %s not understood" % robust)


# NOTE: these need to take into account weights !

def anova_single(model, **kwargs):
    """
    Anova table for one fitted linear model.

    Parameters
    ----------
    model : fitted linear model results instance
        A fitted linear model
    typ : int or str {1,2,3} or {"I","II","III"}
        Type of sum of squares to use.

    **kwargs**

    scale : float
        Estimate of variance, If None, will be estimated from the largest
    model. Default is None.
        test : str {"F", "Chisq", "Cp"} or None
        Test statistics to provide. Default is "F".

    Notes
    -----
    Use of this function is discouraged. Use anova_lm instead.
    """
    test = kwargs.get("test", "F")
    scale = kwargs.get("scale", None)
    typ = kwargs.get("typ", 1)
    robust = kwargs.get("robust", None)
    if robust:
        robust = robust.lower()

    endog = model.model.endog
    exog = model.model.exog
    nobs = exog.shape[0]

    response_name = model.model.endog_names
    design_info = model.model.data.design_info
    exog_names = model.model.exog_names
    # +1 for resids
    n_rows = (len(design_info.terms) - _has_intercept(design_info) + 1)

    pr_test = "PR(>%s)" % test
    names = ['df', 'sum_sq', 'mean_sq', test, pr_test]

    table = DataFrame(np.zeros((n_rows, 5)), columns=names)

    if typ in [1, "I"]:
        return anova1_lm_single(model, endog, exog, nobs, design_info, table,
                                n_rows, test, pr_test, robust)
    elif typ in [2, "II"]:
        return anova2_lm_single(model, design_info, n_rows, test, pr_test,
                                robust)
    elif typ in [3, "III"]:
        return anova3_lm_single(model, design_info, n_rows, test, pr_test,
                                robust)
    elif typ in [4, "IV"]:
        raise NotImplementedError("Type IV not yet implemented")
    else:  # pragma: no cover
        raise ValueError("Type %s not understood" % str(typ))


def anova1_lm_single(model, endog, exog, nobs, design_info, table, n_rows, test,
                     pr_test, robust):
    """
    Anova table for one fitted linear model.

    Parameters
    ----------
    model : fitted linear model results instance
        A fitted linear model

    **kwargs**

    scale : float
        Estimate of variance, If None, will be estimated from the largest
    model. Default is None.
        test : str {"F", "Chisq", "Cp"} or None
        Test statistics to provide. Default is "F".

    Notes
    -----
    Use of this function is discouraged. Use anova_lm instead.
    """
    #maybe we should rethink using pinv > qr in OLS/linear models?
    effects = getattr(model, 'effects', None)
    if effects is None:
        q,r = np.linalg.qr(exog)
        effects = np.dot(q.T, endog)

    arr = np.zeros((len(design_info.terms), len(design_info.column_names)))
    slices = [design_info.slice(name) for name in design_info.term_names]
    for i,slice_ in enumerate(slices):
        arr[i, slice_] = 1

    sum_sq = np.dot(arr, effects**2)
    #NOTE: assumes intercept is first column
    idx = _intercept_idx(design_info)
    sum_sq = sum_sq[~idx]
    term_names = np.array(design_info.term_names) # want boolean indexing
    term_names = term_names[~idx]

    index = term_names.tolist()
    table.index = Index(index + ['Residual'])
    table.loc[index, ['df', 'sum_sq']] = np.c_[arr[~idx].sum(1), sum_sq]
    # fill in residual
    table.loc['Residual', ['sum_sq','df']] = model.ssr, model.df_resid
    if test == 'F':
        table[test] = ((table['sum_sq'] / table['df']) /
                       (model.ssr / model.df_resid))
        table[pr_test] = stats.f.sf(table["F"], table["df"],
                                    model.df_resid)
        table.loc['Residual', [test, pr_test]] = np.nan, np.nan
    table['mean_sq'] = table['sum_sq'] / table['df']
    return table

#NOTE: the below is not agnostic about formula...
def anova2_lm_single(model, design_info, n_rows, test, pr_test, robust):
    """
    Anova type II table for one fitted linear model.

    Parameters
    ----------
    model : fitted linear model results instance
        A fitted linear model

    **kwargs**

    scale : float
        Estimate of variance, If None, will be estimated from the largest
    model. Default is None.
        test : str {"F", "Chisq", "Cp"} or None
        Test statistics to provide. Default is "F".

    Notes
    -----
    Use of this function is discouraged. Use anova_lm instead.

    Type II
    Sum of Squares compares marginal contribution of terms. Thus, it is
    not particularly useful for models with significant interaction terms.
    """
    terms_info = design_info.terms[:] # copy
    terms_info = _remove_intercept_patsy(terms_info)

    names = ['sum_sq', 'df', test, pr_test]

    table = DataFrame(np.zeros((n_rows, 4)), columns = names)
    cov = _get_covariance(model, None)
    robust_cov = _get_covariance(model, robust)
    col_order = []
    index = []
    for i, term in enumerate(terms_info):
        # grab all varaibles except interaction effects that contain term
        # need two hypotheses matrices L1 is most restrictive, ie., term==0
        # L2 is everything except term==0
        cols = design_info.slice(term)
        L1 = lrange(cols.start, cols.stop)
        L2 = []
        term_set = set(term.factors)
        for t in terms_info: # for the term you have
            other_set = set(t.factors)
            if term_set.issubset(other_set) and not term_set == other_set:
                col = design_info.slice(t)
                # on a higher order term containing current `term`
                L1.extend(lrange(col.start, col.stop))
                L2.extend(lrange(col.start, col.stop))

        L1 = np.eye(model.model.exog.shape[1])[L1]
        L2 = np.eye(model.model.exog.shape[1])[L2]

        if L2.size:
            LVL = np.dot(np.dot(L1,robust_cov),L2.T)
            from scipy import linalg
            orth_compl,_ = linalg.qr(LVL)
            r = L1.shape[0] - L2.shape[0]
            # L1|2
            # use the non-unique orthogonal completion since L12 is rank r
            L12 = np.dot(orth_compl[:,-r:].T, L1)
        else:
            L12 = L1
            r = L1.shape[0]
        #from IPython.core.debugger import Pdb; Pdb().set_trace()
        if test == 'F':
            f = model.f_test(L12, cov_p=robust_cov)
            table.loc[table.index[i], test] = test_value = f.fvalue
            table.loc[table.index[i], pr_test] = f.pvalue

        # need to back out SSR from f_test
        table.loc[table.index[i], 'df'] = r
        col_order.append(cols.start)
        index.append(term.name())

    table.index = Index(index + ['Residual'])
    table = table.iloc[np.argsort(col_order + [model.model.exog.shape[1]+1])]
    # back out sum of squares from f_test
    ssr = table[test] * table['df'] * model.ssr/model.df_resid
    table['sum_sq'] = ssr
    # fill in residual
    table.loc['Residual', ['sum_sq','df', test, pr_test]] = (model.ssr,
                                                            model.df_resid,
                                                            np.nan, np.nan)

    return table

def anova3_lm_single(model, design_info, n_rows, test, pr_test, robust):
    n_rows += _has_intercept(design_info)
    terms_info = design_info.terms

    names = ['sum_sq', 'df', test, pr_test]

    table = DataFrame(np.zeros((n_rows, 4)), columns = names)
    cov = _get_covariance(model, robust)
    col_order = []
    index = []
    for i, term in enumerate(terms_info):
        # grab term, hypothesis is that term == 0
        cols = design_info.slice(term)
        L1 = np.eye(model.model.exog.shape[1])[cols]
        L12 = L1
        r = L1.shape[0]

        if test == 'F':
            f = model.f_test(L12, cov_p=cov)
            table.loc[table.index[i], test] = test_value = f.fvalue
            table.loc[table.index[i], pr_test] = f.pvalue

        # need to back out SSR from f_test
        table.loc[table.index[i], 'df'] = r
        #col_order.append(cols.start)
        index.append(term.name())

    table.index = Index(index + ['Residual'])
    #NOTE: Do not need to sort because terms are an ordered dict now
    #table = table.iloc[np.argsort(col_order + [model.model.exog.shape[1]+1])]
    # back out sum of squares from f_test
    ssr = table[test] * table['df'] * model.ssr/model.df_resid
    table['sum_sq'] = ssr
    # fill in residual
    table.loc['Residual', ['sum_sq','df', test, pr_test]] = (model.ssr,
                                                            model.df_resid,
                                                            np.nan, np.nan)
    return table

def anova_lm(*args, **kwargs):
    """
    Anova table for one or more fitted linear models.

    Parameters
    ----------
    args : fitted linear model results instance
        One or more fitted linear models
    scale : float
        Estimate of variance, If None, will be estimated from the largest
        model. Default is None.
    test : str {"F", "Chisq", "Cp"} or None
        Test statistics to provide. Default is "F".
    typ : str or int {"I","II","III"} or {1,2,3}
        The type of Anova test to perform. See notes.
    robust : {None, "hc0", "hc1", "hc2", "hc3"}
        Use heteroscedasticity-corrected coefficient covariance matrix.
        If robust covariance is desired, it is recommended to use `hc3`.

    Returns
    -------
    anova : DataFrame
        When args is a single model, return is DataFrame with columns:

        sum_sq : float64
            Sum of squares for model terms.
        df : float64
            Degrees of freedom for model terms.
        F : float64
            F statistic value for significance of adding model terms.
        PR(>F) : float64
            P-value for significance of adding model terms.

        When args is multiple models, return is DataFrame with columns:

        df_resid : float64
            Degrees of freedom of residuals in models.
        ssr : float64
            Sum of squares of residuals in models.
        df_diff : float64
            Degrees of freedom difference from previous model in args
        ss_dff : float64
            Difference in ssr from previous model in args
        F : float64
            F statistic comparing to previous model in args
        PR(>F): float64
            P-value for significance comparing to previous model in args

    Notes
    -----
    Model statistics are given in the order of args. Models must have been fit
    using the formula api.

    See Also
    --------
    model_results.compare_f_test, model_results.compare_lm_test

    Examples
    --------
    >>> import statsmodels.api as sm
    >>> from statsmodels.formula.api import ols
    >>> moore = sm.datasets.get_rdataset("Moore", "carData", cache=True) # load
    >>> data = moore.data
    >>> data = data.rename(columns={"partner.status" :
    ...                             "partner_status"}) # make name pythonic
    >>> moore_lm = ols('conformity ~ C(fcategory, Sum)*C(partner_status, Sum)',
    ...                 data=data).fit()
    >>> table = sm.stats.anova_lm(moore_lm, typ=2) # Type 2 Anova DataFrame
    >>> print(table)
    """
    typ = kwargs.get('typ', 1)

    ### Farm Out Single model Anova Type I, II, III, and IV ###

    if len(args) == 1:
        model = args[0]
        return anova_single(model, **kwargs)

    if typ not in [1, "I"]:
        raise ValueError("Multiple models only supported for type I. "
                         "Got type %s" % str(typ))

    test = kwargs.get("test", "F")
    scale = kwargs.get("scale", None)
    n_models = len(args)
    pr_test = "Pr(>%s)" % test
    names = ['df_resid', 'ssr', 'df_diff', 'ss_diff', test, pr_test]
    table = DataFrame(np.zeros((n_models, 6)), columns=names)

    if not scale: # assume biggest model is last
        scale = args[-1].scale

    table["ssr"] = [mdl.ssr for mdl in args]
    table["df_resid"] = [mdl.df_resid for mdl in args]
    table.loc[table.index[1:], "df_diff"] = -np.diff(table["df_resid"].values)
    table["ss_diff"] = -table["ssr"].diff()
    if test == "F":
        table["F"] = table["ss_diff"] / table["df_diff"] / scale
        table[pr_test] = stats.f.sf(table["F"], table["df_diff"],
                                    table["df_resid"])
        # for earlier scipy - stats.f.sf(np.nan, 10, 2) -> 0 not nan
        table.loc[table['F'].isnull(), pr_test] = np.nan

    return table


def _not_slice(slices, slices_to_exclude, n):
    ind = np.array([True]*n)
    for term in slices_to_exclude:
        s = slices[term]
        ind[s] = False
    return ind


def _ssr_reduced_model(y, x, term_slices, params, keys):
    """
    Residual sum of squares of OLS model excluding factors in `keys`
    Assumes x matrix is orthogonal

    Parameters
    ----------
    y : array_like
        dependent variable
    x : array_like
        independent variables
    term_slices : a dict of slices
        term_slices[key] is a boolean array specifies the parameters
        associated with the factor `key`
    params : ndarray
        OLS solution of y = x * params
    keys : keys for term_slices
        factors to be excluded

    Returns
    -------
    rss : float
        residual sum of squares
    df : int
        degrees of freedom
    """
    ind = _not_slice(term_slices, keys, x.shape[1])
    params1 = params[ind]
    ssr = np.subtract(y, x[:, ind].dot(params1))
    ssr = ssr.T.dot(ssr)
    df_resid = len(y) - len(params1)
    return ssr, df_resid


class AnovaRM:
    """
    Repeated measures Anova using least squares regression

    The full model regression residual sum of squares is
    used to compare with the reduced model for calculating the
    within-subject effect sum of squares [1].

    Currently, only fully balanced within-subject designs are supported.
    Calculation of between-subject effects and corrections for violation of
    sphericity are not yet implemented.

    Parameters
    ----------
    data : DataFrame
    depvar : str
        The dependent variable in `data`
    subject : str
        Specify the subject id
    within : list[str]
        The within-subject factors
    between : list[str]
        The between-subject factors, this is not yet implemented
    aggregate_func : {None, 'mean', callable}
        If the data set contains more than a single observation per subject
        and cell of the specified model, this function will be used to
        aggregate the data before running the Anova. `None` (the default) will
        not perform any aggregation; 'mean' is s shortcut to `numpy.mean`.
        An exception will be raised if aggregation is required, but no
        aggregation function was specified.

    Returns
    -------
    results : AnovaResults instance

    Raises
    ------
    ValueError
        If the data need to be aggregated, but `aggregate_func` was not
        specified.

    Notes
    -----
    This implementation currently only supports fully balanced designs. If the
    data contain more than one observation per subject and cell of the design,
    these observations need to be aggregated into a single observation
    before the Anova is calculated, either manually or by passing an aggregation
    function via the `aggregate_func` keyword argument.
    Note that if the input data set was not balanced before performing the
    aggregation, the implied heteroscedasticity of the data is ignored.

    References
    ----------
    .. [*] Rutherford, Andrew. Anova and ANCOVA: a GLM approach. John Wiley & Sons, 2011.
    """

    def __init__(self, data, depvar, subject, within=None, between=None,
                 aggregate_func=None):
        self.data = data
        self.depvar = depvar
        self.within = within
        if 'C' in within:
            raise ValueError("Factor name cannot be 'C'! This is in conflict "
                             "with patsy's contrast function name.")
        self.between = between
        if between is not None:
            raise NotImplementedError('Between subject effect not '
                                      'yet supported!')
        self.subject = subject

        if aggregate_func == 'mean':
            self.aggregate_func = np.mean
        else:
            self.aggregate_func = aggregate_func

        if not data.equals(data.drop_duplicates(subset=[subject] + within)):
            if self.aggregate_func is not None:
                self._aggregate()
            else:
                msg = ('The data set contains more than one observation per '
                       'subject and cell. Either aggregate the data manually, '
                       'or pass the `aggregate_func` parameter.')
                raise ValueError(msg)

        self._check_data_balanced()

    def _aggregate(self):
        self.data = (self.data
                     .groupby([self.subject] + self.within,
                              as_index=False)[self.depvar]
                     .agg(self.aggregate_func))

    def _check_data_balanced(self):
        """raise if data is not balanced

        This raises a ValueError if the data is not balanced, and
        returns None if it is balance

        Return might change
        """
        factor_levels = 1
        for wi in self.within:
            factor_levels *= len(self.data[wi].unique())

        cell_count = {}
        for index in range(self.data.shape[0]):
            key = []
            for col in self.within:
                key.append(self.data[col].iloc[index])
            key = tuple(key)
            if key in cell_count:
                cell_count[key] = cell_count[key] + 1
            else:
                cell_count[key] = 1
        error_message = "Data is unbalanced."
        if len(cell_count) != factor_levels:
            raise ValueError(error_message)
        count = cell_count[key]
        for key in cell_count:
            if count != cell_count[key]:
                raise ValueError(error_message)
        if self.data.shape[0] > count * factor_levels:
            raise ValueError('There are more than 1 element in a cell! Missing'
                             ' factors?')

    def fit(self):
        """estimate the model and compute the Anova table

        Returns
        -------
        AnovaResults instance
        """
        y = self.data[self.depvar].values

        # Construct OLS endog and exog from string using patsy
        within = ['C(%s, Sum)' % i for i in self.within]
        subject = 'C(%s, Sum)' % self.subject
        factors = within + [subject]
        x = patsy.dmatrix('*'.join(factors), data=self.data)
        term_slices = x.design_info.term_name_slices
        for key in term_slices:
            ind = np.array([False]*x.shape[1])
            ind[term_slices[key]] = True
            term_slices[key] = np.array(ind)
        term_exclude = [':'.join(factors)]
        ind = _not_slice(term_slices, term_exclude, x.shape[1])
        x = x[:, ind]

        # Fit OLS
        model = OLS(y, x)
        results = model.fit()
        if model.rank < x.shape[1]:
            raise ValueError('Independent variables are collinear.')
        for i in term_exclude:
            term_slices.pop(i)
        for key in term_slices:
            term_slices[key] = term_slices[key][ind]
        params = results.params
        df_resid = results.df_resid
        ssr = results.ssr

        columns = ['F Value', 'Num DF', 'Den DF', 'Pr > F']
        anova_table = pd.DataFrame(np.zeros((0, 4)), columns=columns)

        for key in term_slices:
            if self.subject not in key and key != 'Intercept':
                #  Independen variables are orthogonal
                ssr1, df_resid1 = _ssr_reduced_model(
                    y, x, term_slices, params, [key])
                df1 = df_resid1 - df_resid
                msm = (ssr1 - ssr) / df1
                if (key == ':'.join(factors[:-1]) or
                        (key + ':' + subject not in term_slices)):
                    mse = ssr / df_resid
                    df2 = df_resid
                else:
                    ssr1, df_resid1 = _ssr_reduced_model(
                        y, x, term_slices, params,
                        [key + ':' + subject])
                    df2 = df_resid1 - df_resid
                    mse = (ssr1 - ssr) / df2
                F = msm / mse
                p = stats.f.sf(F, df1, df2)
                term = key.replace('C(', '').replace(', Sum)', '')
                anova_table.loc[term, 'F Value'] = F
                anova_table.loc[term, 'Num DF'] = df1
                anova_table.loc[term, 'Den DF'] = df2
                anova_table.loc[term, 'Pr > F'] = p

        return AnovaResults(anova_table)


class AnovaResults:
    """
    Anova results class

    Attributes
    ----------
    anova_table : DataFrame
    """
    def __init__(self, anova_table):
        self.anova_table = anova_table

    def __str__(self):
        return self.summary().__str__()

    def summary(self):
        """create summary results

        Returns
        -------
        summary : summary2.Summary instance
        """
        summ = summary2.Summary()
        summ.add_title('Anova')
        summ.add_df(self.anova_table)

        return summ


if __name__ == "__main__":
    import pandas
    from statsmodels.formula.api import ols
    # in R
    #library(car)
    #write.csv(Moore, "moore.csv", row.names=FALSE)
    moore = pandas.read_csv('moore.csv', skiprows=1,
                            names=['partner_status','conformity',
                                   'fcategory','fscore'])
    moore_lm = ols('conformity ~ C(fcategory, Sum)*C(partner_status, Sum)',
                    data=moore).fit()

    mooreB = ols('conformity ~ C(partner_status, Sum)', data=moore).fit()

    # for each term you just want to test vs the model without its
    # higher-order terms

    # using Monette-Fox slides and Marden class notes for linear algebra /
    # orthogonal complement
    # https://netfiles.uiuc.edu/jimarden/www/Classes/STAT324/

    table = anova_lm(moore_lm, typ=2)
