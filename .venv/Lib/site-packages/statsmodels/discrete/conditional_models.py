"""
Conditional logistic, Poisson, and multinomial logit regression
"""

import numpy as np
import statsmodels.base.model as base
import statsmodels.regression.linear_model as lm
import statsmodels.base.wrapper as wrap
from statsmodels.discrete.discrete_model import (MultinomialResults,
      MultinomialResultsWrapper)
import collections
import warnings
import itertools


class _ConditionalModel(base.LikelihoodModel):

    def __init__(self, endog, exog, missing='none', **kwargs):

        if "groups" not in kwargs:
            raise ValueError("'groups' is a required argument")
        groups = kwargs["groups"]

        if groups.size != endog.size:
            msg = "'endog' and 'groups' should have the same dimensions"
            raise ValueError(msg)

        if exog.shape[0] != endog.size:
            msg = "The leading dimension of 'exog' should equal the length of 'endog'"
            raise ValueError(msg)

        super(_ConditionalModel, self).__init__(
            endog, exog, missing=missing, **kwargs)

        if self.data.const_idx is not None:
            msg = ("Conditional models should not have an intercept in the " +
                  "design matrix")
            raise ValueError(msg)

        exog = self.exog
        self.k_params = exog.shape[1]

        # Get the row indices for each group
        row_ix = {}
        for i, g in enumerate(groups):
            if g not in row_ix:
                row_ix[g] = []
            row_ix[g].append(i)

        # Split the data into groups and remove groups with no variation
        endog, exog = np.asarray(endog), np.asarray(exog)
        offset = kwargs.get("offset")
        self._endog_grp = []
        self._exog_grp = []
        self._groupsize = []
        if offset is not None:
            offset = np.asarray(offset)
            self._offset_grp = []
        self._offset = []
        self._sumy = []
        self.nobs = 0
        drops = [0, 0]
        for g, ix in row_ix.items():
            y = endog[ix].flat
            if np.std(y) == 0:
                drops[0] += 1
                drops[1] += len(y)
                continue
            self.nobs += len(y)
            self._endog_grp.append(y)
            if offset is not None:
                self._offset_grp.append(offset[ix])
            self._groupsize.append(len(y))
            self._exog_grp.append(exog[ix, :])
            self._sumy.append(np.sum(y))

        if drops[0] > 0:
            msg = ("Dropped %d groups and %d observations for having " +
                   "no within-group variance") % tuple(drops)
            warnings.warn(msg)

        # This can be pre-computed
        if offset is not None:
            self._endofs = []
            for k, ofs in enumerate(self._offset_grp):
                self._endofs.append(np.dot(self._endog_grp[k], ofs))

        # Number of groups
        self._n_groups = len(self._endog_grp)

        # These are the sufficient statistics
        self._xy = []
        self._n1 = []
        for g in range(self._n_groups):
            self._xy.append(np.dot(self._endog_grp[g], self._exog_grp[g]))
            self._n1.append(np.sum(self._endog_grp[g]))

    def hessian(self, params):

        from statsmodels.tools.numdiff import approx_fprime
        hess = approx_fprime(params, self.score)
        hess = np.atleast_2d(hess)
        return hess

    def fit(self,
            start_params=None,
            method='BFGS',
            maxiter=100,
            full_output=True,
            disp=False,
            fargs=(),
            callback=None,
            retall=False,
            skip_hessian=False,
            **kwargs):

        rslt = super(_ConditionalModel, self).fit(
            start_params=start_params,
            method=method,
            maxiter=maxiter,
            full_output=full_output,
            disp=disp,
            skip_hessian=skip_hessian)

        crslt = ConditionalResults(self, rslt.params, rslt.cov_params(), 1)
        crslt.method = method
        crslt.nobs = self.nobs
        crslt.n_groups = self._n_groups
        crslt._group_stats = [
            "%d" % min(self._groupsize),
            "%d" % max(self._groupsize),
            "%.1f" % np.mean(self._groupsize)
        ]
        rslt = ConditionalResultsWrapper(crslt)
        return rslt

    def fit_regularized(self,
                        method="elastic_net",
                        alpha=0.,
                        start_params=None,
                        refit=False,
                        **kwargs):
        """
        Return a regularized fit to a linear regression model.

        Parameters
        ----------
        method : {'elastic_net'}
            Only the `elastic_net` approach is currently implemented.
        alpha : scalar or array_like
            The penalty weight.  If a scalar, the same penalty weight
            applies to all variables in the model.  If a vector, it
            must have the same length as `params`, and contains a
            penalty weight for each coefficient.
        start_params : array_like
            Starting values for `params`.
        refit : bool
            If True, the model is refit using only the variables that
            have non-zero coefficients in the regularized fit.  The
            refitted model is not regularized.
        **kwargs
            Additional keyword argument that are used when fitting the model.

        Returns
        -------
        Results
            A results instance.
        """

        from statsmodels.base.elastic_net import fit_elasticnet

        if method != "elastic_net":
            raise ValueError("method for fit_regularized must be elastic_net")

        defaults = {"maxiter": 50, "L1_wt": 1, "cnvrg_tol": 1e-10,
                    "zero_tol": 1e-10}
        defaults.update(kwargs)

        return fit_elasticnet(self, method=method,
                              alpha=alpha,
                              start_params=start_params,
                              refit=refit,
                              **defaults)

    # Override to allow groups to be passed as a variable name.
    @classmethod
    def from_formula(cls,
                     formula,
                     data,
                     subset=None,
                     drop_cols=None,
                     *args,
                     **kwargs):

        try:
            groups = kwargs["groups"]
            del kwargs["groups"]
        except KeyError:
            raise ValueError("'groups' is a required argument")

        if isinstance(groups, str):
            groups = data[groups]

        if "0+" not in formula.replace(" ", ""):
            warnings.warn("Conditional models should not include an intercept")

        model = super(_ConditionalModel, cls).from_formula(
            formula, data=data, groups=groups, *args, **kwargs)

        return model


class ConditionalLogit(_ConditionalModel):
    """
    Fit a conditional logistic regression model to grouped data.

    Every group is implicitly given an intercept, but the model is fit using
    a conditional likelihood in which the intercepts are not present.  Thus,
    intercept estimates are not given, but the other parameter estimates can
    be interpreted as being adjusted for any group-level confounders.

    Parameters
    ----------
    endog : array_like
        The response variable, must contain only 0 and 1.
    exog : array_like
        The array of covariates.  Do not include an intercept
        in this array.
    groups : array_like
        Codes defining the groups. This is a required keyword parameter.
    """

    def __init__(self, endog, exog, missing='none', **kwargs):

        super(ConditionalLogit, self).__init__(
            endog, exog, missing=missing, **kwargs)

        if np.any(np.unique(self.endog) != np.r_[0, 1]):
            msg = "endog must be coded as 0, 1"
            raise ValueError(msg)

        self.K = self.exog.shape[1]
        # i.e. self.k_params, for compatibility with MNLogit

    def loglike(self, params):

        ll = 0
        for g in range(len(self._endog_grp)):
            ll += self.loglike_grp(g, params)

        return ll

    def score(self, params):

        score = 0
        for g in range(self._n_groups):
            score += self.score_grp(g, params)

        return score

    def _denom(self, grp, params, ofs=None):

        if ofs is None:
            ofs = 0

        exb = np.exp(np.dot(self._exog_grp[grp], params) + ofs)

        # In the recursions, f may be called multiple times with the
        # same arguments, so we memoize the results.
        memo = {}

        def f(t, k):
            if t < k:
                return 0
            if k == 0:
                return 1

            try:
                return memo[(t, k)]
            except KeyError:
                pass

            v = f(t - 1, k) + f(t - 1, k - 1) * exb[t - 1]
            memo[(t, k)] = v

            return v

        return f(self._groupsize[grp], self._n1[grp])

    def _denom_grad(self, grp, params, ofs=None):

        if ofs is None:
            ofs = 0

        ex = self._exog_grp[grp]
        exb = np.exp(np.dot(ex, params) + ofs)

        # s may be called multiple times in the recursions with the
        # same arguments, so memoize the results.
        memo = {}

        def s(t, k):

            if t < k:
                return 0, np.zeros(self.k_params)
            if k == 0:
                return 1, 0

            try:
                return memo[(t, k)]
            except KeyError:
                pass

            h = exb[t - 1]
            a, b = s(t - 1, k)
            c, e = s(t - 1, k - 1)
            d = c * h * ex[t - 1, :]

            u, v = a + c * h, b + d + e * h
            memo[(t, k)] = (u, v)

            return u, v

        return s(self._groupsize[grp], self._n1[grp])

    def loglike_grp(self, grp, params):

        ofs = None
        if hasattr(self, 'offset'):
            ofs = self._offset_grp[grp]

        llg = np.dot(self._xy[grp], params)

        if ofs is not None:
            llg += self._endofs[grp]

        llg -= np.log(self._denom(grp, params, ofs))

        return llg

    def score_grp(self, grp, params):

        ofs = 0
        if hasattr(self, 'offset'):
            ofs = self._offset_grp[grp]

        d, h = self._denom_grad(grp, params, ofs)
        return self._xy[grp] - h / d


class ConditionalPoisson(_ConditionalModel):
    """
    Fit a conditional Poisson regression model to grouped data.

    Every group is implicitly given an intercept, but the model is fit using
    a conditional likelihood in which the intercepts are not present.  Thus,
    intercept estimates are not given, but the other parameter estimates can
    be interpreted as being adjusted for any group-level confounders.

    Parameters
    ----------
    endog : array_like
        The response variable
    exog : array_like
        The covariates
    groups : array_like
        Codes defining the groups. This is a required keyword parameter.
    """

    def loglike(self, params):

        ofs = None
        if hasattr(self, 'offset'):
            ofs = self._offset_grp

        ll = 0.0

        for i in range(len(self._endog_grp)):

            xb = np.dot(self._exog_grp[i], params)
            if ofs is not None:
                xb += ofs[i]
            exb = np.exp(xb)
            y = self._endog_grp[i]
            ll += np.dot(y, xb)
            s = exb.sum()
            ll -= self._sumy[i] * np.log(s)

        return ll

    def score(self, params):

        ofs = None
        if hasattr(self, 'offset'):
            ofs = self._offset_grp

        score = 0.0

        for i in range(len(self._endog_grp)):

            x = self._exog_grp[i]
            xb = np.dot(x, params)
            if ofs is not None:
                xb += ofs[i]
            exb = np.exp(xb)
            s = exb.sum()
            y = self._endog_grp[i]
            score += np.dot(y, x)
            score -= self._sumy[i] * np.dot(exb, x) / s

        return score


class ConditionalResults(base.LikelihoodModelResults):
    def __init__(self, model, params, normalized_cov_params, scale):

        super(ConditionalResults, self).__init__(
            model,
            params,
            normalized_cov_params=normalized_cov_params,
            scale=scale)

    def summary(self, yname=None, xname=None, title=None, alpha=.05):
        """
        Summarize the fitted model.

        Parameters
        ----------
        yname : str, optional
            Default is `y`
        xname : list[str], optional
            Names for the exogenous variables, default is "var_xx".
            Must match the number of parameters in the model
        title : str, optional
            Title for the top table. If not None, then this replaces the
            default title
        alpha : float
            Significance level for the confidence intervals

        Returns
        -------
        smry : Summary instance
            This holds the summary tables and text, which can be printed or
            converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary.Summary : class to hold summary
            results
        """

        top_left = [
            ('Dep. Variable:', None),
            ('Model:', None),
            ('Log-Likelihood:', None),
            ('Method:', [self.method]),
            ('Date:', None),
            ('Time:', None),
        ]

        top_right = [
            ('No. Observations:', None),
            ('No. groups:', [self.n_groups]),
            ('Min group size:', [self._group_stats[0]]),
            ('Max group size:', [self._group_stats[1]]),
            ('Mean group size:', [self._group_stats[2]]),
        ]

        if title is None:
            title = "Conditional Logit Model Regression Results"

        # create summary tables
        from statsmodels.iolib.summary import Summary
        smry = Summary()
        smry.add_table_2cols(
            self,
            gleft=top_left,
            gright=top_right,  # [],
            yname=yname,
            xname=xname,
            title=title)
        smry.add_table_params(
            self, yname=yname, xname=xname, alpha=alpha, use_t=self.use_t)

        return smry

class ConditionalMNLogit(_ConditionalModel):
    """
    Fit a conditional multinomial logit model to grouped data.

    Parameters
    ----------
    endog : array_like
        The dependent variable, must be integer-valued, coded
        0, 1, ..., c-1, where c is the number of response
        categories.
    exog : array_like
        The independent variables.
    groups : array_like
        Codes defining the groups. This is a required keyword parameter.

    Notes
    -----
    Equivalent to femlogit in Stata.

    References
    ----------
    Gary Chamberlain (1980).  Analysis of covariance with qualitative
    data. The Review of Economic Studies.  Vol. 47, No. 1, pp. 225-238.
    """

    def __init__(self, endog, exog, missing='none', **kwargs):

        super(ConditionalMNLogit, self).__init__(
            endog, exog, missing=missing, **kwargs)

        # endog must be integers
        self.endog = self.endog.astype(int)

        self.k_cat = self.endog.max() + 1
        self.df_model = (self.k_cat - 1) * self.exog.shape[1]
        self.df_resid = self.nobs - self.df_model
        self._ynames_map = {j: str(j) for j in range(self.k_cat)}
        self.J = self.k_cat  # Unfortunate name, needed for results
        self.K = self.exog.shape[1]  # for compatibility with MNLogit

        if self.endog.min() < 0:
            msg = "endog may not contain negative values"
            raise ValueError(msg)

        grx = collections.defaultdict(list)
        for k, v in enumerate(self.groups):
            grx[v].append(k)
        self._group_labels = list(grx.keys())
        self._group_labels.sort()
        self._grp_ix = [grx[k] for k in self._group_labels]

    def fit(self,
            start_params=None,
            method='BFGS',
            maxiter=100,
            full_output=True,
            disp=False,
            fargs=(),
            callback=None,
            retall=False,
            skip_hessian=False,
            **kwargs):

        if start_params is None:
            q = self.exog.shape[1]
            c = self.k_cat - 1
            start_params = np.random.normal(size=q * c)

        # Do not call super(...).fit because it cannot handle the 2d-params.
        rslt = base.LikelihoodModel.fit(
            self,
            start_params=start_params,
            method=method,
            maxiter=maxiter,
            full_output=full_output,
            disp=disp,
            skip_hessian=skip_hessian)

        rslt.params = rslt.params.reshape((self.exog.shape[1], -1))
        rslt = MultinomialResults(self, rslt)

        # Not clear what the null likelihood should be, there is no intercept
        # so the null model is not clearly defined.  This is needed for summary
        # to work.
        rslt.set_null_options(llnull=np.nan)

        return MultinomialResultsWrapper(rslt)

    def loglike(self, params):

        q = self.exog.shape[1]
        c = self.k_cat - 1

        pmat = params.reshape((q, c))
        pmat = np.concatenate((np.zeros((q, 1)), pmat), axis=1)
        lpr = np.dot(self.exog, pmat)

        ll = 0.0
        for ii in self._grp_ix:
            x = lpr[ii, :]
            jj = np.arange(x.shape[0], dtype=int)
            y = self.endog[ii]
            denom = 0.0
            for p in itertools.permutations(y):
                denom += np.exp(x[(jj, p)].sum())
            ll += x[(jj, y)].sum() - np.log(denom)

        return ll


    def score(self, params):

        q = self.exog.shape[1]
        c = self.k_cat - 1

        pmat = params.reshape((q, c))
        pmat = np.concatenate((np.zeros((q, 1)), pmat), axis=1)
        lpr = np.dot(self.exog, pmat)

        grad = np.zeros((q, c))
        for ii in self._grp_ix:
            x = lpr[ii, :]
            jj = np.arange(x.shape[0], dtype=int)
            y = self.endog[ii]
            denom = 0.0
            denomg = np.zeros((q, c))
            for p in itertools.permutations(y):
                v = np.exp(x[(jj, p)].sum())
                denom += v
                for i, r in enumerate(p):
                    if r != 0:
                        denomg[:, r - 1] += v * self.exog[ii[i], :]

            for i, r in enumerate(y):
                if r != 0:
                    grad[:, r - 1] += self.exog[ii[i], :]

            grad -= denomg / denom

        return grad.flatten()



class ConditionalResultsWrapper(lm.RegressionResultsWrapper):
    pass


wrap.populate_wrapper(ConditionalResultsWrapper, ConditionalResults)
