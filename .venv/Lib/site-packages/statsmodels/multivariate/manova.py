# -*- coding: utf-8 -*-

"""Multivariate analysis of variance

author: Yichuan Liu
"""
import numpy as np

from statsmodels.compat.pandas import Substitution
from statsmodels.base.model import Model
from .multivariate_ols import MultivariateTestResults
from .multivariate_ols import _multivariate_ols_fit
from .multivariate_ols import _multivariate_ols_test, _hypotheses_doc

__docformat__ = 'restructuredtext en'


class MANOVA(Model):
    """
    Multivariate Analysis of Variance

    The implementation of MANOVA is based on multivariate regression and does
    not assume that the explanatory variables are categorical. Any type of
    variables as in regression is allowed.

    Parameters
    ----------
    endog : array_like
        Dependent variables. A nobs x k_endog array where nobs is
        the number of observations and k_endog is the number of dependent
        variables.
    exog : array_like
        Independent variables. A nobs x k_exog array where nobs is the
        number of observations and k_exog is the number of independent
        variables. An intercept is not included by default and should be added
        by the user. Models specified using a formula include an intercept by
        default.

    Attributes
    ----------
    endog : ndarray
        See Parameters.
    exog : ndarray
        See Parameters.

    Notes
    -----
    MANOVA is used though the `mv_test` function, and `fit` is not used.

    The ``from_formula`` interface is the recommended method to specify
    a model and simplifies testing without needing to manually configure
    the contrast matrices.

    References
    ----------
    .. [*] ftp://public.dhe.ibm.com/software/analytics/spss/documentation/
       statistics/20.0/en/client/Manuals/IBM_SPSS_Statistics_Algorithms.pdf
    """
    _formula_max_endog = None

    def __init__(self, endog, exog, missing='none', hasconst=None, **kwargs):
        if len(endog.shape) == 1 or endog.shape[1] == 1:
            raise ValueError('There must be more than one dependent variable'
                             ' to fit MANOVA!')
        super(MANOVA, self).__init__(endog, exog, missing=missing,
                                     hasconst=hasconst, **kwargs)
        self._fittedmod = _multivariate_ols_fit(self.endog, self.exog)

    def fit(self):
        raise NotImplementedError('fit is not needed to use MANOVA. Call'
                                  'mv_test directly on a MANOVA instance.')

    @Substitution(hypotheses_doc=_hypotheses_doc)
    def mv_test(self, hypotheses=None, skip_intercept_test=False):
        """
        Linear hypotheses testing

        Parameters
        ----------
        %(hypotheses_doc)s
        skip_intercept_test : bool
            If true, then testing the intercept is skipped, the model is not
            changed.
            Note: If a term has a numerically insignificant effect, then
            an exception because of emtpy arrays may be raised. This can
            happen for the intercept if the data has been demeaned.

        Returns
        -------
        results: MultivariateTestResults

        Notes
        -----
        Testing the linear hypotheses

            L * params * M = 0

        where `params` is the regression coefficient matrix for the
        linear model y = x * params

        If the model is not specified using the formula interfact, then the
        hypotheses test each included exogenous variable, one at a time. In
        most applications with categorical variables, the ``from_formula``
        interface should be preferred when specifying a model since it
        provides knowledge about the model when specifying the hypotheses.
        """
        if hypotheses is None:
            if (hasattr(self, 'data') and self.data is not None and
                        hasattr(self.data, 'design_info')):
                terms = self.data.design_info.term_name_slices
                hypotheses = []
                for key in terms:
                    if skip_intercept_test and key == 'Intercept':
                        continue
                    L_contrast = np.eye(self.exog.shape[1])[terms[key], :]
                    hypotheses.append([key, L_contrast, None])
            else:
                hypotheses = []
                for i in range(self.exog.shape[1]):
                    name = 'x%d' % (i)
                    L = np.zeros([1, self.exog.shape[1]])
                    L[0, i] = 1
                    hypotheses.append([name, L, None])

        results = _multivariate_ols_test(hypotheses, self._fittedmod,
                                         self.exog_names, self.endog_names)

        return MultivariateTestResults(results, self.endog_names,
                                       self.exog_names)
