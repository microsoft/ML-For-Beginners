import warnings

import numpy as np
import pandas as pd

from statsmodels.base import model
import statsmodels.base.wrapper as wrap
from statsmodels.tools.sm_exceptions import ConvergenceWarning


class _DimReductionRegression(model.Model):
    """
    A base class for dimension reduction regression methods.
    """

    def __init__(self, endog, exog, **kwargs):
        super(_DimReductionRegression, self).__init__(endog, exog, **kwargs)

    def _prep(self, n_slice):

        # Sort the data by endog
        ii = np.argsort(self.endog)
        x = self.exog[ii, :]

        # Whiten the data
        x -= x.mean(0)
        covx = np.dot(x.T, x) / x.shape[0]
        covxr = np.linalg.cholesky(covx)
        x = np.linalg.solve(covxr, x.T).T
        self.wexog = x
        self._covxr = covxr

        # Split the data into slices
        self._split_wexog = np.array_split(x, n_slice)


class SlicedInverseReg(_DimReductionRegression):
    """
    Sliced Inverse Regression (SIR)

    Parameters
    ----------
    endog : array_like (1d)
        The dependent variable
    exog : array_like (2d)
        The covariates

    References
    ----------
    KC Li (1991).  Sliced inverse regression for dimension reduction.
    JASA 86, 316-342.
    """

    def fit(self, slice_n=20, **kwargs):
        """
        Estimate the EDR space using Sliced Inverse Regression.

        Parameters
        ----------
        slice_n : int, optional
            Target number of observations per slice
        """

        # Sample size per slice
        if len(kwargs) > 0:
            msg = "SIR.fit does not take any extra keyword arguments"
            warnings.warn(msg)

        # Number of slices
        n_slice = self.exog.shape[0] // slice_n

        self._prep(n_slice)

        mn = [z.mean(0) for z in self._split_wexog]
        n = [z.shape[0] for z in self._split_wexog]
        mn = np.asarray(mn)
        n = np.asarray(n)

        # Estimate Cov E[X | Y=y]
        mnc = np.dot(mn.T, n[:, None] * mn) / n.sum()

        a, b = np.linalg.eigh(mnc)
        jj = np.argsort(-a)
        a = a[jj]
        b = b[:, jj]
        params = np.linalg.solve(self._covxr.T, b)

        results = DimReductionResults(self, params, eigs=a)
        return DimReductionResultsWrapper(results)

    def _regularized_objective(self, A):
        # The objective function for regularized SIR

        p = self.k_vars
        covx = self._covx
        mn = self._slice_means
        ph = self._slice_props
        v = 0
        A = np.reshape(A, (p, self.ndim))

        # The penalty
        for k in range(self.ndim):
            u = np.dot(self.pen_mat, A[:, k])
            v += np.sum(u * u)

        # The SIR objective function
        covxa = np.dot(covx, A)
        q, _ = np.linalg.qr(covxa)
        qd = np.dot(q, np.dot(q.T, mn.T))
        qu = mn.T - qd
        v += np.dot(ph, (qu * qu).sum(0))

        return v

    def _regularized_grad(self, A):
        # The gradient of the objective function for regularized SIR

        p = self.k_vars
        ndim = self.ndim
        covx = self._covx
        n_slice = self.n_slice
        mn = self._slice_means
        ph = self._slice_props
        A = A.reshape((p, ndim))

        # Penalty gradient
        gr = 2 * np.dot(self.pen_mat.T, np.dot(self.pen_mat, A))

        A = A.reshape((p, ndim))
        covxa = np.dot(covx, A)
        covx2a = np.dot(covx, covxa)
        Q = np.dot(covxa.T, covxa)
        Qi = np.linalg.inv(Q)
        jm = np.zeros((p, ndim))
        qcv = np.linalg.solve(Q, covxa.T)

        ft = [None] * (p * ndim)
        for q in range(p):
            for r in range(ndim):
                jm *= 0
                jm[q, r] = 1
                umat = np.dot(covx2a.T, jm)
                umat += umat.T
                umat = -np.dot(Qi, np.dot(umat, Qi))
                fmat = np.dot(np.dot(covx, jm), qcv)
                fmat += np.dot(covxa, np.dot(umat, covxa.T))
                fmat += np.dot(covxa, np.linalg.solve(Q, np.dot(jm.T, covx)))
                ft[q*ndim + r] = fmat

        ch = np.linalg.solve(Q, np.dot(covxa.T, mn.T))
        cu = mn - np.dot(covxa, ch).T
        for i in range(n_slice):
            u = cu[i, :]
            v = mn[i, :]
            for q in range(p):
                for r in range(ndim):
                    f = np.dot(u, np.dot(ft[q*ndim + r], v))
                    gr[q, r] -= 2 * ph[i] * f

        return gr.ravel()

    def fit_regularized(self, ndim=1, pen_mat=None, slice_n=20, maxiter=100,
                        gtol=1e-3, **kwargs):
        """
        Estimate the EDR space using regularized SIR.

        Parameters
        ----------
        ndim : int
            The number of EDR directions to estimate
        pen_mat : array_like
            A 2d array such that the squared Frobenius norm of
            `dot(pen_mat, dirs)`` is added to the objective function,
            where `dirs` is an orthogonal array whose columns span
            the estimated EDR space.
        slice_n : int, optional
            Target number of observations per slice
        maxiter :int
            The maximum number of iterations for estimating the EDR
            space.
        gtol : float
            If the norm of the gradient of the objective function
            falls below this value, the algorithm has converged.

        Returns
        -------
        A results class instance.

        Notes
        -----
        If each row of `exog` can be viewed as containing the values of a
        function evaluated at equally-spaced locations, then setting the
        rows of `pen_mat` to [[1, -2, 1, ...], [0, 1, -2, 1, ..], ...]
        will give smooth EDR coefficients.  This is a form of "functional
        SIR" using the squared second derivative as a penalty.

        References
        ----------
        L. Ferre, A.F. Yao (2003).  Functional sliced inverse regression
        analysis.  Statistics: a journal of theoretical and applied
        statistics 37(6) 475-488.
        """

        if len(kwargs) > 0:
            msg = "SIR.fit_regularized does not take keyword arguments"
            warnings.warn(msg)

        if pen_mat is None:
            raise ValueError("pen_mat is a required argument")

        start_params = kwargs.get("start_params", None)

        # Sample size per slice
        slice_n = kwargs.get("slice_n", 20)

        # Number of slices
        n_slice = self.exog.shape[0] // slice_n

        # Sort the data by endog
        ii = np.argsort(self.endog)
        x = self.exog[ii, :]
        x -= x.mean(0)

        covx = np.cov(x.T)

        # Split the data into slices
        split_exog = np.array_split(x, n_slice)

        mn = [z.mean(0) for z in split_exog]
        n = [z.shape[0] for z in split_exog]
        mn = np.asarray(mn)
        n = np.asarray(n)
        self._slice_props = n / n.sum()
        self.ndim = ndim
        self.k_vars = covx.shape[0]
        self.pen_mat = pen_mat
        self._covx = covx
        self.n_slice = n_slice
        self._slice_means = mn

        if start_params is None:
            params = np.zeros((self.k_vars, ndim))
            params[0:ndim, 0:ndim] = np.eye(ndim)
            params = params
        else:
            if start_params.shape[1] != ndim:
                msg = "Shape of start_params is not compatible with ndim"
                raise ValueError(msg)
            params = start_params

        params, _, cnvrg = _grass_opt(params, self._regularized_objective,
                                      self._regularized_grad, maxiter, gtol)

        if not cnvrg:
            g = self._regularized_grad(params.ravel())
            gn = np.sqrt(np.dot(g, g))
            msg = "SIR.fit_regularized did not converge, |g|=%f" % gn
            warnings.warn(msg)

        results = DimReductionResults(self, params, eigs=None)
        return DimReductionResultsWrapper(results)


class PrincipalHessianDirections(_DimReductionRegression):
    """
    Principal Hessian Directions (PHD)

    Parameters
    ----------
    endog : array_like (1d)
        The dependent variable
    exog : array_like (2d)
        The covariates

    Returns
    -------
    A model instance.  Call `fit` to obtain a results instance,
    from which the estimated parameters can be obtained.

    References
    ----------
    KC Li (1992).  On Principal Hessian Directions for Data
    Visualization and Dimension Reduction: Another application
    of Stein's lemma. JASA 87:420.
    """

    def fit(self, **kwargs):
        """
        Estimate the EDR space using PHD.

        Parameters
        ----------
        resid : bool, optional
            If True, use least squares regression to remove the
            linear relationship between each covariate and the
            response, before conducting PHD.

        Returns
        -------
        A results instance which can be used to access the estimated
        parameters.
        """

        resid = kwargs.get("resid", False)

        y = self.endog - self.endog.mean()
        x = self.exog - self.exog.mean(0)

        if resid:
            from statsmodels.regression.linear_model import OLS
            r = OLS(y, x).fit()
            y = r.resid

        cm = np.einsum('i,ij,ik->jk', y, x, x)
        cm /= len(y)

        cx = np.cov(x.T)
        cb = np.linalg.solve(cx, cm)

        a, b = np.linalg.eig(cb)
        jj = np.argsort(-np.abs(a))
        a = a[jj]
        params = b[:, jj]

        results = DimReductionResults(self, params, eigs=a)
        return DimReductionResultsWrapper(results)


class SlicedAverageVarianceEstimation(_DimReductionRegression):
    """
    Sliced Average Variance Estimation (SAVE)

    Parameters
    ----------
    endog : array_like (1d)
        The dependent variable
    exog : array_like (2d)
        The covariates
    bc : bool, optional
        If True, use the bias-corrected CSAVE method of Li and Zhu.

    References
    ----------
    RD Cook.  SAVE: A method for dimension reduction and graphics
    in regression.
    http://www.stat.umn.edu/RegGraph/RecentDev/save.pdf

    Y Li, L-X Zhu (2007). Asymptotics for sliced average
    variance estimation.  The Annals of Statistics.
    https://arxiv.org/pdf/0708.0462.pdf
    """

    def __init__(self, endog, exog, **kwargs):
        super(SAVE, self).__init__(endog, exog, **kwargs)

        self.bc = False
        if "bc" in kwargs and kwargs["bc"] is True:
            self.bc = True

    def fit(self, **kwargs):
        """
        Estimate the EDR space.

        Parameters
        ----------
        slice_n : int
            Number of observations per slice
        """

        # Sample size per slice
        slice_n = kwargs.get("slice_n", 50)

        # Number of slices
        n_slice = self.exog.shape[0] // slice_n

        self._prep(n_slice)

        cv = [np.cov(z.T) for z in self._split_wexog]
        ns = [z.shape[0] for z in self._split_wexog]

        p = self.wexog.shape[1]

        if not self.bc:
            # Cook's original approach
            vm = 0
            for w, cvx in zip(ns, cv):
                icv = np.eye(p) - cvx
                vm += w * np.dot(icv, icv)
            vm /= len(cv)
        else:
            # The bias-corrected approach of Li and Zhu

            # \Lambda_n in Li, Zhu
            av = 0
            for c in cv:
                av += np.dot(c, c)
            av /= len(cv)

            # V_n in Li, Zhu
            vn = 0
            for x in self._split_wexog:
                r = x - x.mean(0)
                for i in range(r.shape[0]):
                    u = r[i, :]
                    m = np.outer(u, u)
                    vn += np.dot(m, m)
            vn /= self.exog.shape[0]

            c = np.mean(ns)
            k1 = c * (c - 1) / ((c - 1)**2 + 1)
            k2 = (c - 1) / ((c - 1)**2 + 1)
            av2 = k1 * av - k2 * vn

            vm = np.eye(p) - 2 * sum(cv) / len(cv) + av2

        a, b = np.linalg.eigh(vm)
        jj = np.argsort(-a)
        a = a[jj]
        b = b[:, jj]
        params = np.linalg.solve(self._covxr.T, b)

        results = DimReductionResults(self, params, eigs=a)
        return DimReductionResultsWrapper(results)


class DimReductionResults(model.Results):
    """
    Results class for a dimension reduction regression.

    Notes
    -----
    The `params` attribute is a matrix whose columns span
    the effective dimension reduction (EDR) space.  Some
    methods produce a corresponding set of eigenvalues
    (`eigs`) that indicate how much information is contained
    in each basis direction.
    """

    def __init__(self, model, params, eigs):
        super(DimReductionResults, self).__init__(
              model, params)
        self.eigs = eigs


class DimReductionResultsWrapper(wrap.ResultsWrapper):
    _attrs = {
        'params': 'columns',
    }
    _wrap_attrs = _attrs

wrap.populate_wrapper(DimReductionResultsWrapper,  # noqa:E305
                      DimReductionResults)


def _grass_opt(params, fun, grad, maxiter, gtol):
    """
    Minimize a function on a Grassmann manifold.

    Parameters
    ----------
    params : array_like
        Starting value for the optimization.
    fun : function
        The function to be minimized.
    grad : function
        The gradient of fun.
    maxiter : int
        The maximum number of iterations.
    gtol : float
        Convergence occurs when the gradient norm falls below this value.

    Returns
    -------
    params : array_like
        The minimizing value for the objective function.
    fval : float
        The smallest achieved value of the objective function.
    cnvrg : bool
        True if the algorithm converged to a limit point.

    Notes
    -----
    `params` is 2-d, but `fun` and `grad` should take 1-d arrays
    `params.ravel()` as arguments.

    Reference
    ---------
    A Edelman, TA Arias, ST Smith (1998).  The geometry of algorithms with
    orthogonality constraints. SIAM J Matrix Anal Appl.
    http://math.mit.edu/~edelman/publications/geometry_of_algorithms.pdf
    """

    p, d = params.shape
    params = params.ravel()

    f0 = fun(params)
    cnvrg = False

    for _ in range(maxiter):

        # Project the gradient to the tangent space
        g = grad(params)
        g -= np.dot(g, params) * params / np.dot(params, params)

        if np.sqrt(np.sum(g * g)) < gtol:
            cnvrg = True
            break

        gm = g.reshape((p, d))
        u, s, vt = np.linalg.svd(gm, 0)

        paramsm = params.reshape((p, d))
        pa0 = np.dot(paramsm, vt.T)

        def geo(t):
            # Parameterize the geodesic path in the direction
            # of the gradient as a function of a real value t.
            pa = pa0 * np.cos(s * t) + u * np.sin(s * t)
            return np.dot(pa, vt).ravel()

        # Try to find a downhill step along the geodesic path.
        step = 2.
        while step > 1e-10:
            pa = geo(-step)
            f1 = fun(pa)
            if f1 < f0:
                params = pa
                f0 = f1
                break
            step /= 2

    params = params.reshape((p, d))
    return params, f0, cnvrg


class CovarianceReduction(_DimReductionRegression):
    """
    Dimension reduction for covariance matrices (CORE).

    Parameters
    ----------
    endog : array_like
        The dependent variable, treated as group labels
    exog : array_like
        The independent variables.
    dim : int
        The dimension of the subspace onto which the covariance
        matrices are projected.

    Returns
    -------
    A model instance.  Call `fit` on the model instance to obtain
    a results instance, which contains the fitted model parameters.

    Notes
    -----
    This is a likelihood-based dimension reduction procedure based
    on Wishart models for sample covariance matrices.  The goal
    is to find a projection matrix P so that C_i | P'C_iP and
    C_j | P'C_jP are equal in distribution for all i, j, where
    the C_i are the within-group covariance matrices.

    The model and methodology are as described in Cook and Forzani.
    The optimization method follows Edelman et. al.

    References
    ----------
    DR Cook, L Forzani (2008).  Covariance reducing models: an alternative
    to spectral modeling of covariance matrices.  Biometrika 95:4.

    A Edelman, TA Arias, ST Smith (1998).  The geometry of algorithms with
    orthogonality constraints. SIAM J Matrix Anal Appl.
    http://math.mit.edu/~edelman/publications/geometry_of_algorithms.pdf
    """

    def __init__(self, endog, exog, dim):

        super(CovarianceReduction, self).__init__(endog, exog)

        covs, ns = [], []
        df = pd.DataFrame(self.exog, index=self.endog)
        for _, v in df.groupby(df.index):
            covs.append(v.cov().values)
            ns.append(v.shape[0])

        self.nobs = len(endog)

        # The marginal covariance
        covm = 0
        for i, _ in enumerate(covs):
            covm += covs[i] * ns[i]
        covm /= self.nobs
        self.covm = covm

        self.covs = covs
        self.ns = ns
        self.dim = dim

    def loglike(self, params):
        """
        Evaluate the log-likelihood

        Parameters
        ----------
        params : array_like
            The projection matrix used to reduce the covariances, flattened
            to 1d.

        Returns the log-likelihood.
        """

        p = self.covm.shape[0]
        proj = params.reshape((p, self.dim))

        c = np.dot(proj.T, np.dot(self.covm, proj))
        _, ldet = np.linalg.slogdet(c)
        f = self.nobs * ldet / 2

        for j, c in enumerate(self.covs):
            c = np.dot(proj.T, np.dot(c, proj))
            _, ldet = np.linalg.slogdet(c)
            f -= self.ns[j] * ldet / 2

        return f

    def score(self, params):
        """
        Evaluate the score function.

        Parameters
        ----------
        params : array_like
            The projection matrix used to reduce the covariances,
            flattened to 1d.

        Returns the score function evaluated at 'params'.
        """

        p = self.covm.shape[0]
        proj = params.reshape((p, self.dim))

        c0 = np.dot(proj.T, np.dot(self.covm, proj))
        cP = np.dot(self.covm, proj)
        g = self.nobs * np.linalg.solve(c0, cP.T).T

        for j, c in enumerate(self.covs):
            c0 = np.dot(proj.T, np.dot(c, proj))
            cP = np.dot(c, proj)
            g -= self.ns[j] * np.linalg.solve(c0, cP.T).T

        return g.ravel()

    def fit(self, start_params=None, maxiter=200, gtol=1e-4):
        """
        Fit the covariance reduction model.

        Parameters
        ----------
        start_params : array_like
            Starting value for the projection matrix. May be
            rectangular, or flattened.
        maxiter : int
            The maximum number of gradient steps to take.
        gtol : float
            Convergence criterion for the gradient norm.

        Returns
        -------
        A results instance that can be used to access the
        fitted parameters.
        """

        p = self.covm.shape[0]
        d = self.dim

        # Starting value for params
        if start_params is None:
            params = np.zeros((p, d))
            params[0:d, 0:d] = np.eye(d)
            params = params
        else:
            params = start_params

        # _grass_opt is designed for minimization, we are doing maximization
        # here so everything needs to be flipped.
        params, llf, cnvrg = _grass_opt(params, lambda x: -self.loglike(x),
                                        lambda x: -self.score(x), maxiter,
                                        gtol)
        llf *= -1
        if not cnvrg:
            g = self.score(params.ravel())
            gn = np.sqrt(np.sum(g * g))
            msg = "CovReduce optimization did not converge, |g|=%f" % gn
            warnings.warn(msg, ConvergenceWarning)

        results = DimReductionResults(self, params, eigs=None)
        results.llf = llf
        return DimReductionResultsWrapper(results)


# aliases for expert users
SIR = SlicedInverseReg
PHD = PrincipalHessianDirections
SAVE = SlicedAverageVarianceEstimation
CORE = CovarianceReduction
