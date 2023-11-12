# -*- coding: utf-8 -*-
"""
F test for null hypothesis that coefficients in several regressions are the same

* implemented by creating groupdummies*exog and testing appropriate contrast
  matrices
* similar to test for structural change in all variables at predefined break points
* allows only one group variable
* currently tests for change in all exog variables
* allows for heterogscedasticity, error variance varies across groups
* does not work if there is a group with only a single observation

TODO
----

* generalize anova structure,
  - structural break in only some variables
  - compare structural breaks in several exog versus constant only
  - fast way to construct comparisons
* print anova style results
* add all pairwise comparison tests (DONE) with and without Bonferroni correction
* add additional test, likelihood-ratio, lagrange-multiplier, wald ?
* test for heteroscedasticity, equality of variances
  - how?
  - like lagrange-multiplier in stattools heteroscedasticity tests
* permutation or bootstrap test statistic or pvalues


References
----------

Greene: section 7.4 Modeling and Testing for a Structural Break
   is not the same because I use a different normalization, which looks easier
   for more than 2 groups/subperiods

after looking at Greene:
* my version assumes that all groups are large enough to estimate the coefficients
* in sections 7.4.2 and 7.5.3, predictive tests can also be used when there are
  insufficient (nobs<nvars) observations in one group/subperiods
  question: can this be used to test structural change for last period?
        cusum test but only for current period,
        in general cusum is better done with recursive ols
        check other references again for this, there was one for non-recursive
        calculation of cusum (if I remember correctly)
* Greene 7.4.4: with unequal variances Greene mentions Wald test, but where
  size of test might not be very good
  no mention of F-test based on GLS, is there a reference for what I did?
  alternative: use Wald test with bootstrap pvalues?


Created on Sat Mar 27 01:48:01 2010
Author: josef-pktd
"""
import numpy as np
from scipy import stats
from statsmodels.regression.linear_model import OLS, WLS


class OneWayLS:
    '''Class to test equality of regression coefficients across groups

    This class performs tests whether the linear regression coefficients are
    the same across pre-specified groups. This can be used to test for
    structural breaks at given change points, or for ANOVA style analysis of
    differences in the effect of explanatory variables across groups.

    Notes
    -----
    The test is implemented by regression on the original pooled exogenous
    variables and on group dummies times the exogenous regressors.

    y_i = X_i beta_i + u_i  for all groups i

    The test is for the null hypothesis: beta_i = beta for all i
    against the alternative that at least one beta_i is different.

    By default it is assumed that all u_i have the same variance. If the
    keyword option het is True, then it is assumed that the variance is
    group specific. This uses WLS with weights given by the standard errors
    from separate regressions for each group.
    Note: het=True is not sufficiently tested

    The F-test assumes that the errors are normally distributed.



    original question from mailing list for equality of coefficients
    across regressions, and example in Stata FAQ

    *testing*:

    * if constant is the only regressor then the result for the F-test is
      the same as scipy.stats.f_oneway
      (which in turn is verified against NIST for not badly scaled problems)
    * f-test for simple structural break is the same as in original script
    * power and size of test look ok in examples
    * not checked/verified for heteroskedastic case
      - for constant only: ftest result is the same with WLS as with OLS - check?

    check: I might be mixing up group names (unique)
           and group id (integers in arange(ngroups)
           not tested for groups that are not arange(ngroups)
           make sure groupnames are always consistently sorted/ordered
           Fixed for getting the results, but groups are not printed yet, still
           inconsistent use for summaries of results.
    '''
    def __init__(self, y, x, groups=None, het=False, data=None, meta=None):
        if groups is None:
            raise ValueError('use OLS if there are no groups')
            #maybe replace by dispatch to OLS
        if data:
            y = data[y]
            x = [data[v] for v in x]
            try:
                groups = data[groups]
            except [KeyError, ValueError]:
                pass
        self.endog = np.asarray(y)
        self.exog = np.asarray(x)
        if self.exog.ndim == 1:
            self.exog = self.exog[:,None]
        self.groups = np.asarray(groups)
        self.het = het

        self.groupsint = None
        if np.issubdtype(self.groups.dtype, int):
            self.unique = np.unique(self.groups)
            if (self.unique == np.arange(len(self.unique))).all():
                self.groupsint = self.groups

        if self.groupsint is None: # groups are not consecutive integers
            self.unique, self.groupsint = np.unique(self.groups, return_inverse=True)
        self.uniqueint = np.arange(len(self.unique)) #as shortcut

    def fitbygroups(self):
        '''Fit OLS regression for each group separately.

        Returns
        -------
        results are attached

        olsbygroup : dictionary of result instance
            the returned regression results for each group
        sigmabygroup : array (ngroups,) (this should be called sigma2group ??? check)
            mse_resid for each group
        weights : array (nobs,)
            standard deviation of group extended to the original observations. This can
            be used as weights in WLS for group-wise heteroscedasticity.



        '''
        olsbygroup = {}
        sigmabygroup = []
        for gi, group in enumerate(self.unique): #np.arange(len(self.unique))):
            groupmask = self.groupsint == gi   #group index
            res = OLS(self.endog[groupmask], self.exog[groupmask]).fit()
            olsbygroup[group] = res
            sigmabygroup.append(res.mse_resid)
        self.olsbygroup = olsbygroup
        self.sigmabygroup = np.array(sigmabygroup)
        self.weights = np.sqrt(self.sigmabygroup[self.groupsint]) #TODO:chk sqrt

    def fitjoint(self):
        '''fit a joint fixed effects model to all observations

        The regression results are attached as `lsjoint`.

        The contrasts for overall and pairwise tests for equality of coefficients are
        attached as a dictionary `contrasts`. This also includes the contrasts for the test
        that the coefficients of a level are zero. ::

        >>> res.contrasts.keys()
        [(0, 1), 1, 'all', 3, (1, 2), 2, (1, 3), (2, 3), (0, 3), (0, 2)]

        The keys are based on the original names or labels of the groups.

        TODO: keys can be numpy scalars and then the keys cannot be sorted



        '''
        if not hasattr(self, 'weights'):
            self.fitbygroups()
        groupdummy = (self.groupsint[:,None] == self.uniqueint).astype(int)
        #order of dummy variables by variable - not used
        #dummyexog = self.exog[:,:,None]*groupdummy[:,None,1:]
        #order of dummy variables by grous - used
        dummyexog = self.exog[:,None,:]*groupdummy[:,1:,None]
        exog = np.c_[self.exog, dummyexog.reshape(self.exog.shape[0],-1)] #self.nobs ??
        #Notes: I changed to drop first group from dummy
        #instead I want one full set dummies
        if self.het:
            weights = self.weights
            res = WLS(self.endog, exog, weights=weights).fit()
        else:
            res = OLS(self.endog, exog).fit()
        self.lsjoint = res
        contrasts = {}
        nvars = self.exog.shape[1]
        nparams = exog.shape[1]
        ndummies = nparams - nvars
        contrasts['all'] = np.c_[np.zeros((ndummies, nvars)), np.eye(ndummies)]
        for groupind, group in enumerate(self.unique[1:]):  #need enumerate if groups != groupsint
            groupind = groupind + 1
            contr = np.zeros((nvars, nparams))
            contr[:,nvars*groupind:nvars*(groupind+1)] = np.eye(nvars)
            contrasts[group] = contr
            #save also for pairs, see next
            contrasts[(self.unique[0], group)] = contr

        #Note: I'm keeping some duplication for testing
        pairs = np.triu_indices(len(self.unique),1)
        for ind1,ind2 in zip(*pairs):  #replace with group1, group2 in sorted(keys)
            if ind1 == 0:
                continue # need comparison with benchmark/normalization group separate
            g1 = self.unique[ind1]
            g2 = self.unique[ind2]
            group = (g1, g2)
            contr = np.zeros((nvars, nparams))
            contr[:,nvars*ind1:nvars*(ind1+1)] = np.eye(nvars)
            contr[:,nvars*ind2:nvars*(ind2+1)] = -np.eye(nvars)
            contrasts[group] = contr


        self.contrasts = contrasts

    def fitpooled(self):
        '''fit the pooled model, which assumes there are no differences across groups
        '''
        if self.het:
            if not hasattr(self, 'weights'):
                self.fitbygroups()
            weights = self.weights
            res = WLS(self.endog, self.exog, weights=weights).fit()
        else:
            res = OLS(self.endog, self.exog).fit()
        self.lspooled = res

    def ftest_summary(self):
        '''run all ftests on the joint model

        Returns
        -------
        fres : str
           a string that lists the results of all individual f-tests
        summarytable : list of tuples
           contains (pair, (fvalue, pvalue,df_denom, df_num)) for each f-test

        Note
        ----
        This are the raw results and not formatted for nice printing.

        '''
        if not hasattr(self, 'lsjoint'):
            self.fitjoint()
        txt = []
        summarytable = []

        txt.append('F-test for equality of coefficients across groups')
        fres = self.lsjoint.f_test(self.contrasts['all'])
        txt.append(fres.__str__())
        summarytable.append(('all',(fres.fvalue, fres.pvalue, fres.df_denom, fres.df_num)))

#        for group in self.unique[1:]:  #replace with group1, group2 in sorted(keys)
#            txt.append('F-test for equality of coefficients between group'
#                       ' %s and group %s' % (group, '0'))
#            fres = self.lsjoint.f_test(self.contrasts[group])
#            txt.append(fres.__str__())
#            summarytable.append((group,(fres.fvalue, fres.pvalue, fres.df_denom, fres.df_num)))
        pairs = np.triu_indices(len(self.unique),1)
        for ind1,ind2 in zip(*pairs):  #replace with group1, group2 in sorted(keys)
            g1 = self.unique[ind1]
            g2 = self.unique[ind2]
            txt.append('F-test for equality of coefficients between group'
                       ' %s and group %s' % (g1, g2))
            group = (g1, g2)
            fres = self.lsjoint.f_test(self.contrasts[group])
            txt.append(fres.__str__())
            summarytable.append((group,(fres.fvalue, fres.pvalue, fres.df_denom, fres.df_num)))

        self.summarytable = summarytable
        return '\n'.join(txt), summarytable


    def print_summary(self, res):
        '''printable string of summary

        '''
        groupind = res.groups
        #res.fitjoint()  #not really necessary, because called by ftest_summary
        if hasattr(res, 'self.summarytable'):
            summtable = self.summarytable
        else:
            _, summtable = res.ftest_summary()
        txt = ''
        #print ft[0]  #skip because table is nicer
        templ = \
'''Table of F-tests for overall or pairwise equality of coefficients'
%(tab)s


Notes: p-values are not corrected for many tests
       (no Bonferroni correction)
       * : reject at 5%% uncorrected confidence level
Null hypothesis: all or pairwise coefficient are the same'
Alternative hypothesis: all coefficients are different'


Comparison with stats.f_oneway
%(statsfow)s


Likelihood Ratio Test
%(lrtest)s
Null model: pooled all coefficients are the same across groups,'
Alternative model: all coefficients are allowed to be different'
not verified but looks close to f-test result'


OLS parameters by group from individual, separate ols regressions'
%(olsbg)s
for group in sorted(res.olsbygroup):
    r = res.olsbygroup[group]
    print group, r.params


Check for heteroscedasticity, '
variance and standard deviation for individual regressions'
%(grh)s
variance    ', res.sigmabygroup
standard dev', np.sqrt(res.sigmabygroup)
'''

        from statsmodels.iolib import SimpleTable
        resvals = {}
        resvals['tab'] = str(SimpleTable([(['%r' % (row[0],)]
                            + list(row[1])
                            + ['*']*(row[1][1]>0.5).item() ) for row in summtable],
                          headers=['pair', 'F-statistic','p-value','df_denom',
                                   'df_num']))
        resvals['statsfow'] = str(stats.f_oneway(*[res.endog[groupind==gr] for gr in
                                                   res.unique]))
        #resvals['lrtest'] = str(res.lr_test())
        resvals['lrtest'] = str(SimpleTable([res.lr_test()],
                                    headers=['likelihood ratio', 'p-value', 'df'] ))

        resvals['olsbg'] = str(SimpleTable([[group]
                                            + res.olsbygroup[group].params.tolist()
                                            for group in sorted(res.olsbygroup)]))
        resvals['grh'] = str(SimpleTable(np.vstack([res.sigmabygroup,
                                               np.sqrt(res.sigmabygroup)]),
                                     headers=res.unique.tolist()))

        return templ % resvals

    # a variation of this has been added to RegressionResults as compare_lr
    def lr_test(self):
        r'''
        generic likelihood ratio test between nested models

            \begin{align}
            D & = -2(\ln(\text{likelihood for null model}) - \ln(\text{likelihood for alternative model})) \\
            & = -2\ln\left( \frac{\text{likelihood for null model}}{\text{likelihood for alternative model}} \right).
            \end{align}

        is distributed as chisquare with df equal to difference in number of parameters or equivalently
        difference in residual degrees of freedom  (sign?)

        TODO: put into separate function
        '''
        if not hasattr(self, 'lsjoint'):
            self.fitjoint()
        if not hasattr(self, 'lspooled'):
            self.fitpooled()
        loglikejoint = self.lsjoint.llf
        loglikepooled = self.lspooled.llf
        lrstat = -2*(loglikepooled - loglikejoint)   #??? check sign
        lrdf = self.lspooled.df_resid - self.lsjoint.df_resid
        lrpval = stats.chi2.sf(lrstat, lrdf)

        return lrstat, lrpval, lrdf
