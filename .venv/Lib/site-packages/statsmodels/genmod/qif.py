import numpy as np
from collections import defaultdict
import statsmodels.base.model as base
from statsmodels.genmod import families
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import links
from statsmodels.genmod.families import varfuncs
import statsmodels.regression.linear_model as lm
import statsmodels.base.wrapper as wrap
from statsmodels.tools.decorators import cache_readonly


class QIFCovariance:
    """
    A covariance model for quadratic inference function regression.

    The mat method returns a basis matrix B such that the inverse
    of the working covariance lies in the linear span of the
    basis matrices.

    Subclasses should set the number of basis matrices `num_terms`,
    so that `mat(d, j)` for j=0, ..., num_terms-1 gives the basis
    of dimension d.`
    """

    def mat(self, dim, term):
        """
        Returns the term'th basis matrix, which is a dim x dim
        matrix.
        """
        raise NotImplementedError


class QIFIndependence(QIFCovariance):
    """
    Independent working covariance for QIF regression.  This covariance
    model gives identical results to GEE with the independence working
    covariance.  When using QIFIndependence as the working covariance,
    the QIF value will be zero, and cannot be used for chi^2 testing, or
    for model selection using AIC, BIC, etc.
    """

    def __init__(self):
        self.num_terms = 1

    def mat(self, dim, term):
        if term == 0:
            return np.eye(dim)
        else:
            return None


class QIFExchangeable(QIFCovariance):
    """
    Exchangeable working covariance for QIF regression.
    """

    def __init__(self):
        self.num_terms = 2

    def mat(self, dim, term):
        if term == 0:
            return np.eye(dim)
        elif term == 1:
            return np.ones((dim, dim))
        else:
            return None


class QIFAutoregressive(QIFCovariance):
    """
    Autoregressive working covariance for QIF regression.
    """

    def __init__(self):
        self.num_terms = 3

    def mat(self, dim, term):

        if dim < 3:
            msg = ("Groups must have size at least 3 for " +
                   "autoregressive covariance.")
            raise ValueError(msg)

        if term == 0:
            return np.eye(dim)
        elif term == 1:
            mat = np.zeros((dim, dim))
            mat.flat[1::(dim+1)] = 1
            mat += mat.T
            return mat
        elif term == 2:
            mat = np.zeros((dim, dim))
            mat[0, 0] = 1
            mat[dim-1, dim-1] = 1
            return mat
        else:
            return None


class QIF(base.Model):
    """
    Fit a regression model using quadratic inference functions (QIF).

    QIF is an alternative to GEE that can be more efficient, and that
    offers different approaches for model selection and inference.

    Parameters
    ----------
    endog : array_like
        The dependent variables of the regression.
    exog : array_like
        The independent variables of the regression.
    groups : array_like
        Labels indicating which group each observation belongs to.
        Observations in different groups should be independent.
    family : genmod family
        An instance of a GLM family.
    cov_struct : QIFCovariance instance
        An instance of a QIFCovariance.

    References
    ----------
    A. Qu, B. Lindsay, B. Li (2000).  Improving Generalized Estimating
    Equations using Quadratic Inference Functions, Biometrika 87:4.
    www.jstor.org/stable/2673612
    """

    def __init__(self, endog, exog, groups, family=None,
                 cov_struct=None, missing='none', **kwargs):

        # Handle the family argument
        if family is None:
            family = families.Gaussian()
        else:
            if not issubclass(family.__class__, families.Family):
                raise ValueError("QIF: `family` must be a genmod "
                                 "family instance")
        self.family = family

        self._fit_history = defaultdict(list)

        # Handle the cov_struct argument
        if cov_struct is None:
            cov_struct = QIFIndependence()
        else:
            if not isinstance(cov_struct, QIFCovariance):
                raise ValueError(
                    "QIF: `cov_struct` must be a QIFCovariance instance")
        self.cov_struct = cov_struct

        groups = np.asarray(groups)

        super(QIF, self).__init__(endog, exog, groups=groups,
                                  missing=missing, **kwargs)

        self.group_names = list(set(groups))
        self.nobs = len(self.endog)

        groups_ix = defaultdict(list)
        for i, g in enumerate(groups):
            groups_ix[g].append(i)
        self.groups_ix = [groups_ix[na] for na in self.group_names]

        self._check_args(groups)

    def _check_args(self, groups):

        if len(groups) != len(self.endog):
            msg = "QIF: groups and endog should have the same length"
            raise ValueError(msg)

        if len(self.endog) != self.exog.shape[0]:
            msg = ("QIF: the length of endog should be equal to the "
                   "number of rows of exog.")
            raise ValueError(msg)

    def objective(self, params):
        """
        Calculate the gradient of the QIF objective function.

        Parameters
        ----------
        params : array_like
            The model parameters at which the gradient is evaluated.

        Returns
        -------
        grad : array_like
            The gradient vector of the QIF objective function.
        gn_deriv : array_like
            The gradients of each estimating equation with
            respect to the parameter.
        """

        endog = self.endog
        exog = self.exog
        lpr = np.dot(exog, params)
        mean = self.family.link.inverse(lpr)
        va = self.family.variance(mean)

        # Mean derivative
        idl = self.family.link.inverse_deriv(lpr)
        idl2 = self.family.link.inverse_deriv2(lpr)
        vd = self.family.variance.deriv(mean)

        m = self.cov_struct.num_terms
        p = exog.shape[1]

        d = p * m
        gn = np.zeros(d)
        gi = np.zeros(d)
        gi_deriv = np.zeros((d, p))
        gn_deriv = np.zeros((d, p))
        cn_deriv = [0] * p
        cmat = np.zeros((d, d))

        fastvar = self.family.variance is varfuncs.constant
        fastlink = isinstance(
            self.family.link,
            # TODO: Remove links.identity after deprecation final
            (links.Identity, links.identity)
        )

        for ix in self.groups_ix:
            sd = np.sqrt(va[ix])
            resid = endog[ix] - mean[ix]
            sresid = resid / sd
            deriv = exog[ix, :] * idl[ix, None]

            jj = 0
            for j in range(m):
                # The derivative of each term in (5) of Qu et al.
                # There are four terms involving beta in a product.
                # Iterated application of the product rule gives
                # the gradient as a sum of four terms.
                c = self.cov_struct.mat(len(ix), j)
                crs1 = np.dot(c, sresid) / sd
                gi[jj:jj+p] = np.dot(deriv.T, crs1)
                crs2 = np.dot(c, -deriv / sd[:, None]) / sd[:, None]
                gi_deriv[jj:jj+p, :] = np.dot(deriv.T, crs2)
                if not (fastlink and fastvar):
                    for k in range(p):
                        m1 = np.dot(exog[ix, :].T,
                                    idl2[ix] * exog[ix, k] * crs1)
                        if not fastvar:
                            vx = -0.5 * vd[ix] * deriv[:, k] / va[ix]**1.5
                            m2 = np.dot(deriv.T, vx * np.dot(c, sresid))
                            m3 = np.dot(deriv.T, np.dot(c, vx * resid) / sd)
                        else:
                            m2, m3 = 0, 0
                        gi_deriv[jj:jj+p, k] += m1 + m2 + m3
                jj += p

            for j in range(p):
                u = np.outer(gi, gi_deriv[:, j])
                cn_deriv[j] += u + u.T

            gn += gi
            gn_deriv += gi_deriv

            cmat += np.outer(gi, gi)

        ngrp = len(self.groups_ix)
        gn /= ngrp
        gn_deriv /= ngrp
        cmat /= ngrp**2

        qif = np.dot(gn, np.linalg.solve(cmat, gn))

        gcg = np.zeros(p)
        for j in range(p):
            cn_deriv[j] /= len(self.groups_ix)**2
            u = np.linalg.solve(cmat, cn_deriv[j]).T
            u = np.linalg.solve(cmat, u)
            gcg[j] = np.dot(gn, np.dot(u, gn))

        grad = 2 * np.dot(gn_deriv.T, np.linalg.solve(cmat, gn)) - gcg

        return qif, grad, cmat, gn, gn_deriv

    def estimate_scale(self, params):
        """
        Estimate the dispersion/scale.

        The scale parameter for binomial and Poisson families is
        fixed at 1, otherwise it is estimated from the data.
        """

        if isinstance(self.family, (families.Binomial, families.Poisson)):
            return 1.

        if hasattr(self, "ddof_scale"):
            ddof_scale = self.ddof_scale
        else:
            ddof_scale = self.exog[1]

        lpr = np.dot(self.exog, params)
        mean = self.family.link.inverse(lpr)
        resid = self.endog - mean
        scale = np.sum(resid**2) / (self.nobs - ddof_scale)

        return scale

    @classmethod
    def from_formula(cls, formula, groups, data, subset=None,
                     *args, **kwargs):
        """
        Create a QIF model instance from a formula and dataframe.

        Parameters
        ----------
        formula : str or generic Formula object
            The formula specifying the model
        groups : array_like or string
            Array of grouping labels.  If a string, this is the name
            of a variable in `data` that contains the grouping labels.
        data : array_like
            The data for the model.
        subset : array_like
            An array_like object of booleans, integers, or index
            values that indicate the subset of the data to used when
            fitting the model.

        Returns
        -------
        model : QIF model instance
        """

        if isinstance(groups, str):
            groups = data[groups]

        model = super(QIF, cls).from_formula(
                   formula, data=data, subset=subset,
                   groups=groups, *args, **kwargs)

        return model

    def fit(self, maxiter=100, start_params=None, tol=1e-6, gtol=1e-4,
            ddof_scale=None):
        """
        Fit a GLM to correlated data using QIF.

        Parameters
        ----------
        maxiter : int
            Maximum number of iterations.
        start_params : array_like, optional
            Starting values
        tol : float
            Convergence threshold for difference of successive
            estimates.
        gtol : float
            Convergence threshold for gradient.
        ddof_scale : int, optional
            Degrees of freedom for the scale parameter

        Returns
        -------
        QIFResults object
        """

        if ddof_scale is None:
            self.ddof_scale = self.exog.shape[1]
        else:
            self.ddof_scale = ddof_scale

        if start_params is None:
            model = GLM(self.endog, self.exog, family=self.family)
            result = model.fit()
            params = result.params
        else:
            params = start_params

        for _ in range(maxiter):

            qif, grad, cmat, _, gn_deriv = self.objective(params)

            gnorm = np.sqrt(np.sum(grad * grad))
            self._fit_history["qif"].append(qif)
            self._fit_history["gradnorm"].append(gnorm)

            if gnorm < gtol:
                break

            cjac = 2 * np.dot(gn_deriv.T, np.linalg.solve(cmat, gn_deriv))
            step = np.linalg.solve(cjac, grad)

            snorm = np.sqrt(np.sum(step * step))
            self._fit_history["stepnorm"].append(snorm)
            if snorm < tol:
                break
            params -= step

        vcov = np.dot(gn_deriv.T, np.linalg.solve(cmat, gn_deriv))
        vcov = np.linalg.inv(vcov)
        scale = self.estimate_scale(params)

        rslt = QIFResults(self, params, vcov / scale, scale)
        rslt.fit_history = self._fit_history
        self._fit_history = defaultdict(list)

        return QIFResultsWrapper(rslt)


class QIFResults(base.LikelihoodModelResults):
    """Results class for QIF Regression"""
    def __init__(self, model, params, cov_params, scale,
                 use_t=False, **kwds):

        super(QIFResults, self).__init__(
            model, params, normalized_cov_params=cov_params,
            scale=scale)

        self.qif, _, _, _, _ = self.model.objective(params)

    @cache_readonly
    def aic(self):
        """
        An AIC-like statistic for models fit using QIF.
        """
        if isinstance(self.model.cov_struct, QIFIndependence):
            msg = "AIC not available with QIFIndependence covariance"
            raise ValueError(msg)
        df = self.model.exog.shape[1]
        return self.qif + 2*df

    @cache_readonly
    def bic(self):
        """
        A BIC-like statistic for models fit using QIF.
        """
        if isinstance(self.model.cov_struct, QIFIndependence):
            msg = "BIC not available with QIFIndependence covariance"
            raise ValueError(msg)
        df = self.model.exog.shape[1]
        return self.qif + np.log(self.model.nobs)*df

    @cache_readonly
    def fittedvalues(self):
        """
        Returns the fitted values from the model.
        """
        return self.model.family.link.inverse(
                np.dot(self.model.exog, self.params))

    def summary(self, yname=None, xname=None, title=None, alpha=.05):
        """
        Summarize the QIF regression results

        Parameters
        ----------
        yname : str, optional
            Default is `y`
        xname : list[str], optional
            Names for the exogenous variables, default is `var_#` for ## in
            the number of regressors. Must match the number of parameters in
            the model
        title : str, optional
            Title for the top table. If not None, then this replaces
            the default title
        alpha : float
            significance level for the confidence intervals

        Returns
        -------
        smry : Summary instance
            this holds the summary tables and text, which can be
            printed or converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary.Summary : class to hold summary results
        """

        top_left = [('Dep. Variable:', None),
                    ('Method:', ['QIF']),
                    ('Family:', [self.model.family.__class__.__name__]),
                    ('Covariance structure:',
                     [self.model.cov_struct.__class__.__name__]),
                    ('Date:', None),
                    ('Time:', None),
                    ]

        NY = [len(y) for y in self.model.groups_ix]

        top_right = [('No. Observations:', [sum(NY)]),
                     ('No. clusters:', [len(NY)]),
                     ('Min. cluster size:', [min(NY)]),
                     ('Max. cluster size:', [max(NY)]),
                     ('Mean cluster size:', ["%.1f" % np.mean(NY)]),
                     ('Scale:', ["%.3f" % self.scale]),
                     ]

        if title is None:
            title = self.model.__class__.__name__ + ' ' +\
                "Regression Results"

        # Override the exog variable names if xname is provided as an
        # argument.
        if xname is None:
            xname = self.model.exog_names

        if yname is None:
            yname = self.model.endog_names

        # Create summary table instance
        from statsmodels.iolib.summary import Summary
        smry = Summary()
        smry.add_table_2cols(self, gleft=top_left, gright=top_right,
                             yname=yname, xname=xname,
                             title=title)
        smry.add_table_params(self, yname=yname, xname=xname,
                              alpha=alpha, use_t=False)

        return smry


class QIFResultsWrapper(lm.RegressionResultsWrapper):
    pass


wrap.populate_wrapper(QIFResultsWrapper, QIFResults)
