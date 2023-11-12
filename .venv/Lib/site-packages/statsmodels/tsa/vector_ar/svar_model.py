# -*- coding: utf-8 -*-
"""
Vector Autoregression (VAR) processes

References
----------
Lütkepohl (2005) New Introduction to Multiple Time Series Analysis
"""
import numpy as np
import numpy.linalg as npl
from numpy.linalg import slogdet

from statsmodels.tools.decorators import deprecated_alias
from statsmodels.tools.numdiff import approx_fprime, approx_hess
import statsmodels.tsa.base.tsa_model as tsbase
from statsmodels.tsa.vector_ar.irf import IRAnalysis
import statsmodels.tsa.vector_ar.util as util
from statsmodels.tsa.vector_ar.var_model import VARProcess, VARResults


def svar_ckerr(svar_type, A, B):
    if A is None and (svar_type == 'A' or svar_type == 'AB'):
        raise ValueError('SVAR of type A or AB but A array not given.')
    if B is None and (svar_type == 'B' or svar_type == 'AB'):

        raise ValueError('SVAR of type B or AB but B array not given.')


class SVAR(tsbase.TimeSeriesModel):
    r"""
    Fit VAR and then estimate structural components of A and B, defined:

    .. math:: Ay_t = A_1 y_{t-1} + \ldots + A_p y_{t-p} + B\var(\epsilon_t)

    Parameters
    ----------
    endog : array_like
        1-d endogenous response variable. The independent variable.
    dates : array_like
        must match number of rows of endog
    svar_type : str
        "A" - estimate structural parameters of A matrix, B assumed = I
        "B" - estimate structural parameters of B matrix, A assumed = I
        "AB" - estimate structural parameters indicated in both A and B matrix
    A : array_like
        neqs x neqs with unknown parameters marked with 'E' for estimate
    B : array_like
        neqs x neqs with unknown parameters marked with 'E' for estimate

    References
    ----------
    Hamilton (1994) Time Series Analysis
    """

    y = deprecated_alias("y", "endog", remove_version="0.11.0")

    def __init__(self, endog, svar_type, dates=None,
                 freq=None, A=None, B=None, missing='none'):
        super().__init__(endog, None, dates, freq, missing=missing)
        #(self.endog, self.names,
        # self.dates) = data_util.interpret_data(endog, names, dates)

        self.neqs = self.endog.shape[1]

        types = ['A', 'B', 'AB']
        if svar_type not in types:
            raise ValueError('SVAR type not recognized, must be in '
                             + str(types))
        self.svar_type = svar_type

        svar_ckerr(svar_type, A, B)

        self.A_original = A
        self.B_original = B

        # initialize A, B as I if not given
        # Initialize SVAR masks
        if A is None:
            A = np.identity(self.neqs)
            self.A_mask = A_mask = np.zeros(A.shape, dtype=bool)
        else:
            A_mask = np.logical_or(A == 'E', A == 'e')
            self.A_mask = A_mask
        if B is None:
            B = np.identity(self.neqs)
            self.B_mask = B_mask = np.zeros(B.shape, dtype=bool)
        else:
            B_mask = np.logical_or(B == 'E', B == 'e')
            self.B_mask = B_mask

        # convert A and B to numeric
        #TODO: change this when masked support is better or with formula
        #integration
        Anum = np.zeros(A.shape, dtype=float)
        Anum[~A_mask] = A[~A_mask]
        Anum[A_mask] = np.nan
        self.A = Anum

        Bnum = np.zeros(B.shape, dtype=float)
        Bnum[~B_mask] = B[~B_mask]
        Bnum[B_mask] = np.nan
        self.B = Bnum

        #LikelihoodModel.__init__(self, endog)

        #super().__init__(endog)

    def fit(self, A_guess=None, B_guess=None, maxlags=None, method='ols',
            ic=None, trend='c', verbose=False, s_method='mle',
            solver="bfgs", override=False, maxiter=500, maxfun=500):
        """
        Fit the SVAR model and solve for structural parameters

        Parameters
        ----------
        A_guess : array_like, optional
            A vector of starting values for all parameters to be estimated
            in A.
        B_guess : array_like, optional
            A vector of starting values for all parameters to be estimated
            in B.
        maxlags : int
            Maximum number of lags to check for order selection, defaults to
            12 * (nobs/100.)**(1./4), see select_order function
        method : {'ols'}
            Estimation method to use
        ic : {'aic', 'fpe', 'hqic', 'bic', None}
            Information criterion to use for VAR order selection.
            aic : Akaike
            fpe : Final prediction error
            hqic : Hannan-Quinn
            bic : Bayesian a.k.a. Schwarz
        verbose : bool, default False
            Print order selection output to the screen
        trend, str {"c", "ct", "ctt", "n"}
            "c" - add constant
            "ct" - constant and trend
            "ctt" - constant, linear and quadratic trend
            "n" - co constant, no trend
            Note that these are prepended to the columns of the dataset.
        s_method : {'mle'}
            Estimation method for structural parameters
        solver : {'nm', 'newton', 'bfgs', 'cg', 'ncg', 'powell'}
            Solution method
            See statsmodels.base for details
        override : bool, default False
            If True, returns estimates of A and B without checking
            order or rank condition
        maxiter : int, default 500
            Number of iterations to perform in solution method
        maxfun : int
            Number of function evaluations to perform

        Notes
        -----
        Lütkepohl pp. 146-153
        Hamilton pp. 324-336

        Returns
        -------
        est : SVARResults
        """
        lags = maxlags
        if ic is not None:
            selections = self.select_order(maxlags=maxlags, verbose=verbose)
            if ic not in selections:
                raise ValueError("%s not recognized, must be among %s"
                                 % (ic, sorted(selections)))
            lags = selections[ic]
            if verbose:
                print('Using %d based on %s criterion' %  (lags, ic))
        else:
            if lags is None:
                lags = 1

        self.nobs = len(self.endog) - lags

        # initialize starting parameters
        start_params = self._get_init_params(A_guess, B_guess)

        return self._estimate_svar(start_params, lags, trend=trend,
                                   solver=solver, override=override,
                                   maxiter=maxiter, maxfun=maxfun)

    def _get_init_params(self, A_guess, B_guess):
        """
        Returns either the given starting or .1 if none are given.
        """

        var_type = self.svar_type.lower()

        n_masked_a = self.A_mask.sum()
        if var_type in ['ab', 'a']:
            if A_guess is None:
                A_guess = np.array([.1]*n_masked_a)
            else:
                if len(A_guess) != n_masked_a:
                    msg = 'len(A_guess) = %s, there are %s parameters in A'
                    raise ValueError(msg % (len(A_guess), n_masked_a))
        else:
            A_guess = []

        n_masked_b = self.B_mask.sum()
        if var_type in ['ab', 'b']:
            if B_guess is None:
                B_guess = np.array([.1]*n_masked_b)
            else:
                if len(B_guess) != n_masked_b:
                    msg = 'len(B_guess) = %s, there are %s parameters in B'
                    raise ValueError(msg % (len(B_guess), n_masked_b))
        else:
            B_guess = []

        return np.r_[A_guess, B_guess]

    def _estimate_svar(self, start_params, lags, maxiter, maxfun,
                       trend='c', solver="nm", override=False):
        """
        lags : int
        trend : {str, None}
            As per above
        """
        k_trend = util.get_trendorder(trend)
        y = self.endog
        z = util.get_var_endog(y, lags, trend=trend, has_constant='raise')
        y_sample = y[lags:]

        # Lutkepohl p75, about 5x faster than stated formula
        var_params = np.linalg.lstsq(z, y_sample, rcond=-1)[0]
        resid = y_sample - np.dot(z, var_params)

        # Unbiased estimate of covariance matrix $\Sigma_u$ of the white noise
        # process $u$
        # equivalent definition
        # .. math:: \frac{1}{T - Kp - 1} Y^\prime (I_T - Z (Z^\prime Z)^{-1}
        # Z^\prime) Y
        # Ref: Lutkepohl p.75
        # df_resid right now is T - Kp - 1, which is a suggested correction

        avobs = len(y_sample)

        df_resid = avobs - (self.neqs * lags + k_trend)

        sse = np.dot(resid.T, resid)
        #TODO: should give users the option to use a dof correction or not
        omega = sse / df_resid
        self.sigma_u = omega

        A, B = self._solve_AB(start_params, override=override,
                              solver=solver,
                              maxiter=maxiter)
        A_mask = self.A_mask
        B_mask = self.B_mask

        return SVARResults(y, z, var_params, omega, lags,
                           names=self.endog_names, trend=trend,
                           dates=self.data.dates, model=self,
                           A=A, B=B, A_mask=A_mask, B_mask=B_mask)

    def loglike(self, params):
        """
        Loglikelihood for SVAR model

        Notes
        -----
        This method assumes that the autoregressive parameters are
        first estimated, then likelihood with structural parameters
        is estimated
        """

        #TODO: this does not look robust if A or B is None
        A = self.A
        B = self.B
        A_mask = self.A_mask
        B_mask = self.B_mask
        A_len = len(A[A_mask])
        B_len = len(B[B_mask])

        if A is not None:
            A[A_mask] = params[:A_len]
        if B is not None:
            B[B_mask] = params[A_len:A_len+B_len]

        nobs = self.nobs
        neqs = self.neqs
        sigma_u = self.sigma_u

        W = np.dot(npl.inv(B),A)
        trc_in = np.dot(np.dot(W.T,W),sigma_u)
        sign, b_logdet = slogdet(B**2) #numpy 1.4 compat
        b_slogdet = sign * b_logdet

        likl = -nobs/2. * (neqs * np.log(2 * np.pi) -
                           np.log(npl.det(A)**2) + b_slogdet +
                           np.trace(trc_in))

        return likl

    def score(self, AB_mask):
        """
        Return the gradient of the loglike at AB_mask.

        Parameters
        ----------
        AB_mask : unknown values of A and B matrix concatenated

        Notes
        -----
        Return numerical gradient
        """
        loglike = self.loglike
        return approx_fprime(AB_mask, loglike, epsilon=1e-8)

    def hessian(self, AB_mask):
        """
        Returns numerical hessian.
        """
        loglike = self.loglike
        return approx_hess(AB_mask, loglike)

    def _solve_AB(self, start_params, maxiter, override=False, solver='bfgs'):
        """
        Solves for MLE estimate of structural parameters

        Parameters
        ----------

        override : bool, default False
            If True, returns estimates of A and B without checking
            order or rank condition
        solver : str or None, optional
            Solver to be used. The default is 'nm' (Nelder-Mead). Other
            choices are 'bfgs', 'newton' (Newton-Raphson), 'cg'
            conjugate, 'ncg' (non-conjugate gradient), and 'powell'.
        maxiter : int, optional
            The maximum number of iterations. Default is 500.

        Returns
        -------
        A_solve, B_solve: ML solutions for A, B matrices
        """
        #TODO: this could stand a refactor
        A_mask = self.A_mask
        B_mask = self.B_mask
        A = self.A
        B = self.B
        A_len = len(A[A_mask])

        A[A_mask] = start_params[:A_len]
        B[B_mask] = start_params[A_len:]

        if not override:
            J = self._compute_J(A, B)
            self.check_order(J)
            self.check_rank(J)
        else: #TODO: change to a warning?
            print("Order/rank conditions have not been checked")

        retvals = super().fit(start_params=start_params,
                              method=solver, maxiter=maxiter,
                              gtol=1e-20, disp=False).params

        A[A_mask] = retvals[:A_len]
        B[B_mask] = retvals[A_len:]

        return A, B

    def _compute_J(self, A_solve, B_solve):

        #first compute appropriate duplication matrix
        # taken from Magnus and Neudecker (1980),
        #"The Elimination Matrix: Some Lemmas and Applications
        # the creation of the D_n matrix follows MN (1980) directly,
        #while the rest follows Hamilton (1994)

        neqs = self.neqs
        sigma_u = self.sigma_u
        A_mask = self.A_mask
        B_mask = self.B_mask

        #first generate duplication matrix, see MN (1980) for notation

        D_nT = np.zeros([int((1.0 / 2) * (neqs) * (neqs + 1)), neqs**2])

        for j in range(neqs):
            i=j
            while j <= i < neqs:
                u=np.zeros([int((1.0/2)*neqs*(neqs+1)), 1])
                u[int(j * neqs + (i + 1) - (1.0 / 2) * (j + 1) * j - 1)] = 1
                Tij=np.zeros([neqs,neqs])
                Tij[i,j]=1
                Tij[j,i]=1
                D_nT=D_nT+np.dot(u,(Tij.ravel('F')[:,None]).T)
                i=i+1

        D_n=D_nT.T
        D_pl=npl.pinv(D_n)

        #generate S_B
        S_B = np.zeros((neqs**2, len(A_solve[A_mask])))
        S_D = np.zeros((neqs**2, len(B_solve[B_mask])))

        j = 0
        j_d = 0
        if len(A_solve[A_mask]) != 0:
            A_vec = np.ravel(A_mask, order='F')
            for k in range(neqs**2):
                if A_vec[k]:
                    S_B[k,j] = -1
                    j += 1
        if len(B_solve[B_mask]) != 0:
            B_vec = np.ravel(B_mask, order='F')
            for k in range(neqs**2):
                if B_vec[k]:
                    S_D[k,j_d] = 1
                    j_d +=1

        #now compute J
        invA = npl.inv(A_solve)
        J_p1i = np.dot(np.dot(D_pl, np.kron(sigma_u, invA)), S_B)
        J_p1 = -2.0 * J_p1i
        J_p2 = np.dot(np.dot(D_pl, np.kron(invA, invA)), S_D)

        J = np.append(J_p1, J_p2, axis=1)

        return J

    def check_order(self, J):
        if np.size(J, axis=0) < np.size(J, axis=1):
            raise ValueError("Order condition not met: "
                             "solution may not be unique")

    def check_rank(self, J):
        rank = np.linalg.matrix_rank(J)
        if rank < np.size(J, axis=1):
            raise ValueError("Rank condition not met: "
                             "solution may not be unique.")


class SVARProcess(VARProcess):
    """
    Class represents a known SVAR(p) process

    Parameters
    ----------
    coefs : ndarray (p x k x k)
    intercept : ndarray (length k)
    sigma_u : ndarray (k x k)
    names : sequence (length k)
    A : neqs x neqs np.ndarray with unknown parameters marked with 'E'
    A_mask : neqs x neqs mask array with known parameters masked
    B : neqs x neqs np.ndarry with unknown parameters marked with 'E'
    B_mask : neqs x neqs mask array with known parameters masked
    """
    def __init__(self, coefs, intercept, sigma_u, A_solve, B_solve,
                 names=None):
        self.k_ar = len(coefs)
        self.neqs = coefs.shape[1]
        self.coefs = coefs
        self.intercept = intercept
        self.sigma_u = sigma_u
        self.A_solve = A_solve
        self.B_solve = B_solve
        self.names = names

    def orth_ma_rep(self, maxn=10, P=None):
        """

        Unavailable for SVAR
        """
        raise NotImplementedError

    def svar_ma_rep(self, maxn=10, P=None):
        """

        Compute Structural MA coefficient matrices using MLE
        of A, B
        """
        if P is None:
            A_solve = self.A_solve
            B_solve = self.B_solve
            P = np.dot(npl.inv(A_solve), B_solve)

        ma_mats = self.ma_rep(maxn=maxn)
        return np.array([np.dot(coefs, P) for coefs in ma_mats])


class SVARResults(SVARProcess, VARResults):
    """
    Estimate VAR(p) process with fixed number of lags

    Parameters
    ----------
    endog : ndarray
    endog_lagged : ndarray
    params : ndarray
    sigma_u : ndarray
    lag_order : int
    model : VAR model instance
    trend : str {'n', 'c', 'ct'}
    names : array_like
        List of names of the endogenous variables in order of appearance in `endog`.
    dates

    Attributes
    ----------
    aic
    bic
    bse
    coefs : ndarray (p x K x K)
        Estimated A_i matrices, A_i = coefs[i-1]
    cov_params
    dates
    detomega
    df_model : int
    df_resid : int
    endog
    endog_lagged
    fittedvalues
    fpe
    intercept
    info_criteria
    k_ar : int
    k_trend : int
    llf
    model
    names
    neqs : int
        Number of variables (equations)
    nobs : int
    n_totobs : int
    params
    k_ar : int
        Order of VAR process
    params : ndarray (Kp + 1) x K
        A_i matrices and intercept in stacked form [int A_1 ... A_p]
    pvalue
    names : list
        variables names
    resid
    sigma_u : ndarray (K x K)
        Estimate of white noise process variance Var[u_t]
    sigma_u_mle
    stderr
    trenorder
    tvalues
    """

    _model_type = 'SVAR'

    def __init__(self, endog, endog_lagged, params, sigma_u, lag_order,
                 A=None, B=None, A_mask=None, B_mask=None, model=None,
                 trend='c', names=None, dates=None):

        self.model = model
        self.endog = endog
        self.endog_lagged = endog_lagged
        self.dates = dates

        self.n_totobs, self.neqs = self.endog.shape
        self.nobs = self.n_totobs - lag_order
        k_trend = util.get_trendorder(trend)
        if k_trend > 0: # make this the polynomial trend order
            trendorder = k_trend - 1
        else:
            trendorder = None
        self.k_trend = k_trend
        self.k_exog = k_trend  # now (0.9) required by VARProcess
        self.trendorder = trendorder

        self.exog_names = util.make_lag_names(names, lag_order, k_trend)
        self.params = params
        self.sigma_u = sigma_u

        # Each matrix needs to be transposed
        reshaped = self.params[self.k_trend:]
        reshaped = reshaped.reshape((lag_order, self.neqs, self.neqs))

        # Need to transpose each coefficient matrix
        intercept = self.params[0]
        coefs = reshaped.swapaxes(1, 2).copy()

        #SVAR components
        #TODO: if you define these here, you do not also have to define
        #them in SVAR process, but I left them for now -ss
        self.A = A
        self.B = B
        self.A_mask = A_mask
        self.B_mask = B_mask

        super().__init__(coefs, intercept, sigma_u, A, B,
                                          names=names)

    def irf(self, periods=10, var_order=None):
        """
        Analyze structural impulse responses to shocks in system

        Parameters
        ----------
        periods : int

        Returns
        -------
        irf : IRAnalysis
        """
        A = self.A
        B= self.B
        P = np.dot(npl.inv(A), B)

        return IRAnalysis(self, P=P, periods=periods, svar=True)

    def sirf_errband_mc(self, orth=False, repl=1000, steps=10,
                        signif=0.05, seed=None, burn=100, cum=False):
        """
        Compute Monte Carlo integrated error bands assuming normally
        distributed for impulse response functions

        Parameters
        ----------
        orth : bool, default False
            Compute orthogonalized impulse response error bands
        repl : int
            number of Monte Carlo replications to perform
        steps : int, default 10
            number of impulse response periods
        signif : float (0 < signif <1)
            Significance level for error bars, defaults to 95% CI
        seed : int
            np.random.seed for replications
        burn : int
            number of initial observations to discard for simulation
        cum : bool, default False
            produce cumulative irf error bands

        Notes
        -----
        Lütkepohl (2005) Appendix D

        Returns
        -------
        Tuple of lower and upper arrays of ma_rep monte carlo standard errors
        """
        neqs = self.neqs
        mean = self.mean()
        k_ar = self.k_ar
        coefs = self.coefs
        sigma_u = self.sigma_u
        intercept = self.intercept
        df_model = self.df_model
        nobs = self.nobs

        ma_coll = np.zeros((repl, steps + 1, neqs, neqs))
        A = self.A
        B = self.B
        A_mask = self.A_mask
        B_mask = self.B_mask
        A_pass = self.model.A_original
        B_pass = self.model.B_original
        s_type = self.model.svar_type

        g_list = []

        def agg(impulses):
            if cum:
                return impulses.cumsum(axis=0)
            return impulses

        opt_A = A[A_mask]
        opt_B = B[B_mask]
        for i in range(repl):
            # discard first hundred to correct for starting bias
            sim = util.varsim(coefs, intercept, sigma_u, seed=seed,
                              steps=nobs + burn)
            sim = sim[burn:]

            smod = SVAR(sim, svar_type=s_type, A=A_pass, B=B_pass)
            if i == 10:
                # Use first 10 to update starting val for remainder of fits
                mean_AB = np.mean(g_list, axis=0)
                split = len(A[A_mask])
                opt_A = mean_AB[:split]
                opt_B = mean_AB[split:]

            sres = smod.fit(maxlags=k_ar, A_guess=opt_A, B_guess=opt_B)

            if i < 10:
                # save estimates for starting val if in first 10
                g_list.append(np.append(sres.A[A_mask].tolist(),
                                        sres.B[B_mask].tolist()))
            ma_coll[i] = agg(sres.svar_ma_rep(maxn=steps))

        ma_sort = np.sort(ma_coll, axis=0)  # sort to get quantiles
        index = (int(round(signif / 2 * repl) - 1),
                 int(round((1 - signif / 2) * repl) - 1))
        lower = ma_sort[index[0], :, :, :]
        upper = ma_sort[index[1], :, :, :]
        return lower, upper
