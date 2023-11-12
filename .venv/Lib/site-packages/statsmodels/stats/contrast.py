import numpy as np
from scipy.stats import f as fdist
from scipy.stats import t as student_t
from scipy import stats
from statsmodels.tools.tools import clean0, fullrank
from statsmodels.stats.multitest import multipletests


#TODO: should this be public if it's just a container?
class ContrastResults:
    """
    Class for results of tests of linear restrictions on coefficients in a model.

    This class functions mainly as a container for `t_test`, `f_test` and
    `wald_test` for the parameters of a model.

    The attributes depend on the statistical test and are either based on the
    normal, the t, the F or the chisquare distribution.
    """

    def __init__(self, t=None, F=None, sd=None, effect=None, df_denom=None,
                 df_num=None, alpha=0.05, **kwds):

        self.effect = effect  # Let it be None for F
        if F is not None:
            self.distribution = 'F'
            self.fvalue = F
            self.statistic = self.fvalue
            self.df_denom = df_denom
            self.df_num = df_num
            self.dist = fdist
            self.dist_args = (df_num, df_denom)
            self.pvalue = fdist.sf(F, df_num, df_denom)
        elif t is not None:
            self.distribution = 't'
            self.tvalue = t
            self.statistic = t  # generic alias
            self.sd = sd
            self.df_denom = df_denom
            self.dist = student_t
            self.dist_args = (df_denom,)
            self.pvalue = self.dist.sf(np.abs(t), df_denom) * 2
        elif 'statistic' in kwds:
            # TODO: currently targeted to normal distribution, and chi2
            self.distribution = kwds['distribution']
            self.statistic = kwds['statistic']
            self.tvalue = value = kwds['statistic']  # keep alias
            # TODO: for results instance we decided to use tvalues also for normal
            self.sd = sd
            self.dist = getattr(stats, self.distribution)
            self.dist_args = kwds.get('dist_args', ())
            if self.distribution == 'chi2':
                self.pvalue = self.dist.sf(self.statistic, df_denom)
                self.df_denom = df_denom
            else:
                "normal"
                self.pvalue = np.full_like(value, np.nan)
                not_nan = ~np.isnan(value)
                self.pvalue[not_nan] = self.dist.sf(np.abs(value[not_nan])) * 2
        else:
            self.pvalue = np.nan

        # cleanup
        # should we return python scalar?
        self.pvalue = np.squeeze(self.pvalue)

        if self.effect is not None:
            self.c_names = ['c%d' % ii for ii in range(len(self.effect))]
        else:
            self.c_names = None

    def conf_int(self, alpha=0.05):
        """
        Returns the confidence interval of the value, `effect` of the constraint.

        This is currently only available for t and z tests.

        Parameters
        ----------
        alpha : float, optional
            The significance level for the confidence interval.
            ie., The default `alpha` = .05 returns a 95% confidence interval.

        Returns
        -------
        ci : ndarray, (k_constraints, 2)
            The array has the lower and the upper limit of the confidence
            interval in the columns.
        """
        if self.effect is not None:
            # confidence intervals
            q = self.dist.ppf(1 - alpha / 2., *self.dist_args)
            lower = self.effect - q * self.sd
            upper = self.effect + q * self.sd
            return np.column_stack((lower, upper))
        else:
            raise NotImplementedError('Confidence Interval not available')

    def __str__(self):
        return self.summary().__str__()

    def __repr__(self):
        return str(self.__class__) + '\n' + self.__str__()

    def summary(self, xname=None, alpha=0.05, title=None):
        """Summarize the Results of the hypothesis test

        Parameters
        ----------
        xname : list[str], optional
            Default is `c_##` for ## in the number of regressors
        alpha : float
            significance level for the confidence intervals. Default is
            alpha = 0.05 which implies a confidence level of 95%.
        title : str, optional
            Title for the params table. If not None, then this replaces the
            default title

        Returns
        -------
        smry : str or Summary instance
            This contains a parameter results table in the case of t or z test
            in the same form as the parameter results table in the model
            results summary.
            For F or Wald test, the return is a string.
        """
        if self.effect is not None:
            # TODO: should also add some extra information, e.g. robust cov ?
            # TODO: can we infer names for constraints, xname in __init__ ?
            if title is None:
                title = 'Test for Constraints'
            elif title == '':
                # do not add any title,
                # I think SimpleTable skips on None - check
                title = None
            # we have everything for a params table
            use_t = (self.distribution == 't')
            yname='constraints' # Not used in params_frame
            if xname is None:
                xname = self.c_names
            from statsmodels.iolib.summary import summary_params
            pvalues = np.atleast_1d(self.pvalue)
            summ = summary_params((self, self.effect, self.sd, self.statistic,
                                   pvalues, self.conf_int(alpha)),
                                  yname=yname, xname=xname, use_t=use_t,
                                  title=title, alpha=alpha)
            return summ
        elif hasattr(self, 'fvalue'):
            # TODO: create something nicer for these casee
            return ('<F test: F=%s, p=%s, df_denom=%.3g, df_num=%.3g>' %
                   (repr(self.fvalue), self.pvalue, self.df_denom,
                    self.df_num))
        elif self.distribution == 'chi2':
            return ('<Wald test (%s): statistic=%s, p-value=%s, df_denom=%.3g>' %
                   (self.distribution, self.statistic, self.pvalue,
                    self.df_denom))
        else:
            # generic
            return ('<Wald test: statistic=%s, p-value=%s>' %
                   (self.statistic, self.pvalue))


    def summary_frame(self, xname=None, alpha=0.05):
        """Return the parameter table as a pandas DataFrame

        This is only available for t and normal tests
        """
        if self.effect is not None:
            # we have everything for a params table
            use_t = (self.distribution == 't')
            yname='constraints'  # Not used in params_frame
            if xname is None:
                xname = self.c_names
            from statsmodels.iolib.summary import summary_params_frame
            summ = summary_params_frame((self, self.effect, self.sd,
                                         self.statistic,self.pvalue,
                                         self.conf_int(alpha)), yname=yname,
                                         xname=xname, use_t=use_t,
                                         alpha=alpha)
            return summ
        else:
            # TODO: create something nicer
            raise NotImplementedError('only available for t and z')



class Contrast:
    """
    This class is used to construct contrast matrices in regression models.

    They are specified by a (term, design) pair.  The term, T, is a linear
    combination of columns of the design matrix. The matrix attribute of
    Contrast is a contrast matrix C so that

    colspan(dot(D, C)) = colspan(dot(D, dot(pinv(D), T)))

    where pinv(D) is the generalized inverse of D. Further, the matrix

    Tnew = dot(C, D)

    is full rank. The rank attribute is the rank of

    dot(D, dot(pinv(D), T))

    In a regression model, the contrast tests that E(dot(Tnew, Y)) = 0
    for each column of Tnew.

    Parameters
    ----------
    term : array_like
    design : array_like

    Attributes
    ----------
    contrast_matrix

    Examples
    --------
    >>> import statsmodels.api as sm
    >>> from statsmodels.stats.contrast import Contrast
    >>> import numpy as np
    >>> np.random.seed(54321)
    >>> X = np.random.standard_normal((40,10))
    # Get a contrast
    >>> new_term = np.column_stack((X[:,0], X[:,2]))
    >>> c = Contrast(new_term, X)
    >>> test = [[1] + [0]*9, [0]*2 + [1] + [0]*7]
    >>> np.allclose(c.contrast_matrix, test)
    True

    Get another contrast

    >>> P = np.dot(X, np.linalg.pinv(X))
    >>> resid = np.identity(40) - P
    >>> noise = np.dot(resid,np.random.standard_normal((40,5)))
    >>> new_term2 = np.column_stack((noise,X[:,2]))
    >>> c2 = Contrast(new_term2, X)
    >>> print(c2.contrast_matrix)
    [ -1.26424750e-16   8.59467391e-17   1.56384718e-01  -2.60875560e-17
    -7.77260726e-17  -8.41929574e-18  -7.36359622e-17  -1.39760860e-16
    1.82976904e-16  -3.75277947e-18]

    Get another contrast

    >>> zero = np.zeros((40,))
    >>> new_term3 = np.column_stack((zero,X[:,2]))
    >>> c3 = Contrast(new_term3, X)
    >>> test2 = [0]*2 + [1] + [0]*7
    >>> np.allclose(c3.contrast_matrix, test2)
    True
    """
    def _get_matrix(self):
        """
        Gets the contrast_matrix property
        """
        if not hasattr(self, "_contrast_matrix"):
            self.compute_matrix()
        return self._contrast_matrix

    contrast_matrix = property(_get_matrix)

    def __init__(self, term, design):
        self.term = np.asarray(term)
        self.design = np.asarray(design)

    def compute_matrix(self):
        """
        Construct a contrast matrix C so that

        colspan(dot(D, C)) = colspan(dot(D, dot(pinv(D), T)))

        where pinv(D) is the generalized inverse of D=design.
        """

        T = self.term
        if T.ndim == 1:
            T = T[:,None]

        self.T = clean0(T)
        self.D = self.design
        self._contrast_matrix = contrastfromcols(self.T, self.D)
        try:
            self.rank = self.matrix.shape[1]
        except:
            self.rank = 1

#TODO: fix docstring after usage is settled
def contrastfromcols(L, D, pseudo=None):
    """
    From an n x p design matrix D and a matrix L, tries
    to determine a p x q contrast matrix C which
    determines a contrast of full rank, i.e. the
    n x q matrix

    dot(transpose(C), pinv(D))

    is full rank.

    L must satisfy either L.shape[0] == n or L.shape[1] == p.

    If L.shape[0] == n, then L is thought of as representing
    columns in the column space of D.

    If L.shape[1] == p, then L is thought of as what is known
    as a contrast matrix. In this case, this function returns an estimable
    contrast corresponding to the dot(D, L.T)

    Note that this always produces a meaningful contrast, not always
    with the intended properties because q is always non-zero unless
    L is identically 0. That is, it produces a contrast that spans
    the column space of L (after projection onto the column space of D).

    Parameters
    ----------
    L : array_like
    D : array_like
    """
    L = np.asarray(L)
    D = np.asarray(D)

    n, p = D.shape

    if L.shape[0] != n and L.shape[1] != p:
        raise ValueError("shape of L and D mismatched")

    if pseudo is None:
        pseudo = np.linalg.pinv(D)    # D^+ \approx= ((dot(D.T,D))^(-1),D.T)

    if L.shape[0] == n:
        C = np.dot(pseudo, L).T
    else:
        C = L
        C = np.dot(pseudo, np.dot(D, C.T)).T

    Lp = np.dot(D, C.T)

    if len(Lp.shape) == 1:
        Lp.shape = (n, 1)

    if np.linalg.matrix_rank(Lp) != Lp.shape[1]:
        Lp = fullrank(Lp)
        C = np.dot(pseudo, Lp).T

    return np.squeeze(C)


# TODO: this is currently a minimal version, stub
class WaldTestResults:
    # for F and chi2 tests of joint hypothesis, mainly for vectorized

    def __init__(self, statistic, distribution, dist_args, table=None,
                 pvalues=None):
        self.table = table

        self.distribution = distribution
        self.statistic = statistic
        #self.sd = sd
        self.dist_args = dist_args

        # The following is because I do not know which we want
        if table is not None:
            self.statistic = table['statistic'].values
            self.pvalues = table['pvalue'].values
            self.df_constraints = table['df_constraint'].values
            if self.distribution == 'F':
                self.df_denom = table['df_denom'].values

        else:
            if self.distribution == 'chi2':
                self.dist = stats.chi2
                self.df_constraints = self.dist_args[0]  # assumes tuple
                # using dist_args[0] is a bit dangerous,
            elif self.distribution == 'F':
                self.dist = stats.f
                self.df_constraints, self.df_denom = self.dist_args

            else:
                raise ValueError('only F and chi2 are possible distribution')

            if pvalues is None:
                self.pvalues = self.dist.sf(np.abs(statistic), *dist_args)
            else:
                self.pvalues = pvalues

    @property
    def col_names(self):
        """column names for summary table
        """

        pr_test = "P>%s" % self.distribution
        col_names = [self.distribution, pr_test, 'df constraint']
        if self.distribution == 'F':
            col_names.append('df denom')
        return col_names

    def summary_frame(self):
        # needs to be a method for consistency
        if hasattr(self, '_dframe'):
            return self._dframe
        # rename the column nambes, but do not copy data
        renaming = dict(zip(self.table.columns, self.col_names))
        self.dframe = self.table.rename(columns=renaming)
        return self.dframe


    def __str__(self):
        return self.summary_frame().to_string()


    def __repr__(self):
        return str(self.__class__) + '\n' + self.__str__()


# t_test for pairwise comparison and automatic contrast/restrictions


def _get_pairs_labels(k_level, level_names):
    """helper function for labels for pairwise comparisons
    """
    idx_pairs_all = np.triu_indices(k_level, 1)
    labels = ['%s-%s' % (level_names[name[1]], level_names[name[0]])
              for name in zip(*idx_pairs_all)]
    return labels

def _contrast_pairs(k_params, k_level, idx_start):
    """create pairwise contrast for reference coding

    currently not used,
    using encoding contrast matrix is more general, but requires requires
    factor information from patsy design_info.


    Parameters
    ----------
    k_params : int
        number of parameters
    k_level : int
        number of levels or categories (including reference case)
    idx_start : int
        Index of the first parameter of this factor. The restrictions on the
        factor are inserted as a block in the full restriction matrix starting
        at column with index `idx_start`.

    Returns
    -------
    contrasts : ndarray
        restriction matrix with k_params columns and number of rows equal to
        the number of restrictions.
    """
    k_level_m1 = k_level - 1
    idx_pairs = np.triu_indices(k_level_m1, 1)

    k = len(idx_pairs[0])
    c_pairs = np.zeros((k, k_level_m1))
    c_pairs[np.arange(k), idx_pairs[0]] = -1
    c_pairs[np.arange(k), idx_pairs[1]] = 1
    c_reference = np.eye(k_level_m1)
    c = np.concatenate((c_reference, c_pairs), axis=0)
    k_all = c.shape[0]

    contrasts = np.zeros((k_all, k_params))
    contrasts[:, idx_start : idx_start + k_level_m1] = c

    return contrasts


def t_test_multi(result, contrasts, method='hs', alpha=0.05, ci_method=None,
                 contrast_names=None):
    """perform t_test and add multiplicity correction to results dataframe

    Parameters
    ----------
    result results instance
        results of an estimated model
    contrasts : ndarray
        restriction matrix for t_test
    method : str or list of strings
        method for multiple testing p-value correction, default is'hs'.
    alpha : float
        significance level for multiple testing reject decision.
    ci_method : None
        not used yet, will be for multiplicity corrected confidence intervals
    contrast_names : {list[str], None}
        If contrast_names are provided, then they are used in the index of the
        returned dataframe, otherwise some generic default names are created.

    Returns
    -------
    res_df : pandas DataFrame
        The dataframe contains the results of the t_test and additional columns
        for multiplicity corrected p-values and boolean indicator for whether
        the Null hypothesis is rejected.
    """
    tt = result.t_test(contrasts)
    res_df = tt.summary_frame(xname=contrast_names)

    if type(method) is not list:
        method = [method]
    for meth in method:
        mt = multipletests(tt.pvalue, method=meth, alpha=alpha)
        res_df['pvalue-%s' % meth] = mt[1]
        res_df['reject-%s' % meth] = mt[0]
    return res_df


class MultiCompResult:
    """class to hold return of t_test_pairwise

    currently just a minimal class to hold attributes.
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _embed_constraints(contrasts, k_params, idx_start, index=None):
    """helper function to expand constraints to a full restriction matrix

    Parameters
    ----------
    contrasts : ndarray
        restriction matrix for t_test
    k_params : int
        number of parameters
    idx_start : int
        Index of the first parameter of this factor. The restrictions on the
        factor are inserted as a block in the full restriction matrix starting
        at column with index `idx_start`.
    index : slice or ndarray
        Column index if constraints do not form a block in the full restriction
        matrix, i.e. if parameters that are subject to restrictions are not
        consecutive in the list of parameters.
        If index is not None, then idx_start is ignored.

    Returns
    -------
    contrasts : ndarray
        restriction matrix with k_params columns and number of rows equal to
        the number of restrictions.
    """

    k_c, k_p = contrasts.shape
    c = np.zeros((k_c, k_params))
    if index is None:
        c[:, idx_start : idx_start + k_p] = contrasts
    else:
        c[:, index] = contrasts
    return c


def _constraints_factor(encoding_matrix, comparison='pairwise', k_params=None,
                        idx_start=None):
    """helper function to create constraints based on encoding matrix

    Parameters
    ----------
    encoding_matrix : ndarray
        contrast matrix for the encoding of a factor as defined by patsy.
        The number of rows should be equal to the number of levels or categories
        of the factor, the number of columns should be equal to the number
        of parameters for this factor.
    comparison : str
        Currently only 'pairwise' is implemented. The restriction matrix
        can be used for testing the hypothesis that all pairwise differences
        are zero.
    k_params : int
        number of parameters
    idx_start : int
        Index of the first parameter of this factor. The restrictions on the
        factor are inserted as a block in the full restriction matrix starting
        at column with index `idx_start`.

    Returns
    -------
    contrast : ndarray
        Contrast or restriction matrix that can be used in hypothesis test
        of model results. The number of columns is k_params.
    """

    cm = encoding_matrix
    k_level, k_p = cm.shape

    import statsmodels.sandbox.stats.multicomp as mc
    if comparison in ['pairwise', 'pw', 'pairs']:
        c_all = -mc.contrast_allpairs(k_level)
    else:
        raise NotImplementedError('currentlyonly pairwise comparison')

    contrasts = c_all.dot(cm)
    if k_params is not None:
        if idx_start is None:
            raise ValueError("if k_params is not None, then idx_start is "
                             "required")
        contrasts = _embed_constraints(contrasts, k_params, idx_start)
    return contrasts


def t_test_pairwise(result, term_name, method='hs', alpha=0.05,
                    factor_labels=None, ignore=False):
    """
    Perform pairwise t_test with multiple testing corrected p-values.

    This uses the formula design_info encoding contrast matrix and should
    work for all encodings of a main effect.

    Parameters
    ----------
    result : result instance
        The results of an estimated model with a categorical main effect.
    term_name : str
        name of the term for which pairwise comparisons are computed.
        Term names for categorical effects are created by patsy and
        correspond to the main part of the exog names.
    method : {str, list[str]}
        multiple testing p-value correction, default is 'hs',
        see stats.multipletesting
    alpha : float
        significance level for multiple testing reject decision.
    factor_labels : {list[str], None}
        Labels for the factor levels used for pairwise labels. If not
        provided, then the labels from the formula design_info are used.
    ignore : bool
        Turn off some of the exceptions raised by input checks.

    Returns
    -------
    MultiCompResult
        The results are stored as attributes, the main attributes are the
        following two. Other attributes are added for debugging purposes
        or as background information.

        - result_frame : pandas DataFrame with t_test results and multiple
          testing corrected p-values.
        - contrasts : matrix of constraints of the null hypothesis in the
          t_test.

    Notes
    -----

    Status: experimental. Currently only checked for treatment coding with
    and without specified reference level.

    Currently there are no multiple testing corrected confidence intervals
    available.
    """

    desinfo = result.model.data.design_info
    term_idx = desinfo.term_names.index(term_name)
    term = desinfo.terms[term_idx]
    idx_start = desinfo.term_slices[term].start
    if not ignore and len(term.factors) > 1:
        raise ValueError('interaction effects not yet supported')
    factor = term.factors[0]
    cat = desinfo.factor_infos[factor].categories
    if factor_labels is not None:
        if len(factor_labels) == len(cat):
            cat = factor_labels
        else:
            raise ValueError("factor_labels has the wrong length, should be %d" % len(cat))


    k_level = len(cat)
    cm = desinfo.term_codings[term][0].contrast_matrices[factor].matrix

    k_params = len(result.params)
    labels = _get_pairs_labels(k_level, cat)

    import statsmodels.sandbox.stats.multicomp as mc
    c_all_pairs = -mc.contrast_allpairs(k_level)
    contrasts_sub = c_all_pairs.dot(cm)
    contrasts = _embed_constraints(contrasts_sub, k_params, idx_start)
    res_df = t_test_multi(result, contrasts, method=method, ci_method=None,
                          alpha=alpha, contrast_names=labels)
    res = MultiCompResult(result_frame=res_df,
                          contrasts=contrasts,
                          term=term,
                          contrast_labels=labels,
                          term_encoding_matrix=cm)
    return res


def _offset_constraint(r_matrix, params_est, params_alt):
    """offset to the value of a linear constraint for new params

    usage:
    (cm, v) is original constraint

    vo = offset_constraint(cm, res2.params, params_alt)
    fs = res2.wald_test((cm, v + vo))
    nc = fs.statistic * fs.df_num

    """
    diff_est = r_matrix @ params_est
    diff_alt = r_matrix @ params_alt
    return diff_est - diff_alt


def wald_test_noncent(params, r_matrix, value, results, diff=None, joint=True):
    """Moncentrality parameter for a wald test in model results

    The null hypothesis is ``diff = r_matrix @ params - value = 0``

    Parameters
    ----------
    params : ndarray
        parameters of the model at which to evaluate noncentrality. This can
        be estimated parameters or parameters under an alternative.
    r_matrix : ndarray
        Restriction matrix or contrasts for the Null hypothesis
    value : None or ndarray
        Value of the linear combination of parameters under the null
        hypothesis. If value is None, then it will be replaced by zero.
    results : Results instance of a model
        The results instance is used to compute the covariance matrix of the
        linear constraints using `cov_params.
    diff : None or ndarray
        If diff is not None, then it will be used instead of
        ``diff = r_matrix @ params - value``
    joint : bool
        If joint is True, then the noncentrality parameter for the joint
        hypothesis will be returned.
        If joint is True, then an array of noncentrality parameters will be
        returned, where elements correspond to rows of the restriction matrix.
        This correspond to the `t_test` in models and is not a quadratic form.

    Returns
    -------
    nc : float or ndarray
        Noncentrality parameter for Wald tests, correspondig to `wald_test`
        or `t_test` depending on whether `joint` is true or not.
        It needs to be divided by nobs to obtain effect size.


    Notes
    -----
    Status : experimental, API will likely change

    """
    if diff is None:
        diff = r_matrix @ params - value  # at parameter under alternative

    cov_c = results.cov_params(r_matrix=r_matrix)
    if joint:
        nc = diff @ np.linalg.solve(cov_c, diff)
    else:
        nc = diff / np.sqrt(np.diag(cov_c))
    return nc


def wald_test_noncent_generic(params, r_matrix, value, cov_params, diff=None,
                              joint=True):
    """noncentrality parameter for a wald test

    The null hypothesis is ``diff = r_matrix @ params - value = 0``

    Parameters
    ----------
    params : ndarray
        parameters of the model at which to evaluate noncentrality. This can
        be estimated parameters or parameters under an alternative.
    r_matrix : ndarray
        Restriction matrix or contrasts for the Null hypothesis

    value : None or ndarray
        Value of the linear combination of parameters under the null
        hypothesis. If value is None, then it will be replace by zero.
    cov_params : ndarray
        covariance matrix of the parameter estimates
    diff : None or ndarray
        If diff is not None, then it will be used instead of
        ``diff = r_matrix @ params - value``
    joint : bool
        If joint is True, then the noncentrality parameter for the joint
        hypothesis will be returned.
        If joint is True, then an array of noncentrality parameters will be
        returned, where elements correspond to rows of the restriction matrix.
        This correspond to the `t_test` in models and is not a quadratic form.

    Returns
    -------
    nc : float or ndarray
        Noncentrality parameter for Wald tests, correspondig to `wald_test`
        or `t_test` depending on whether `joint` is true or not.
        It needs to be divided by nobs to obtain effect size.


    Notes
    -----
    Status : experimental, API will likely change
    """
    if value is None:
        value = 0
    if diff is None:
        # at parameter under alternative
        diff = r_matrix @ params - value

    c = r_matrix
    cov_c = c.dot(cov_params).dot(c.T)
    if joint:
        nc = diff @ np.linalg.solve(cov_c, diff)
    else:
        nc = diff / np.sqrt(np.diag(cov_c))
    return nc
