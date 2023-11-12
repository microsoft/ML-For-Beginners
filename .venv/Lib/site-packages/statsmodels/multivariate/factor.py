# -*- coding: utf-8 -*-

import warnings

import numpy as np
from numpy.linalg import eigh, inv, norm, matrix_rank
import pandas as pd
from scipy.optimize import minimize

from statsmodels.tools.decorators import cache_readonly
from statsmodels.base.model import Model
from statsmodels.iolib import summary2
from statsmodels.graphics.utils import _import_mpl

from .factor_rotation import rotate_factors, promax


_opt_defaults = {'gtol': 1e-7}


def _check_args_1(endog, n_factor, corr, nobs):

    msg = "Either endog or corr must be provided."
    if endog is not None and corr is not None:
        raise ValueError(msg)
    if endog is None and corr is None:
        warnings.warn('Both endog and corr are provided, ' +
                      'corr will be used for factor analysis.')

    if n_factor <= 0:
        raise ValueError('n_factor must be larger than 0! %d < 0' %
                         (n_factor))

    if nobs is not None and endog is not None:
        warnings.warn("nobs is ignored when endog is provided")


def _check_args_2(endog, n_factor, corr, nobs, k_endog):

    if n_factor > k_endog:
        raise ValueError('n_factor cannot be greater than the number'
                         ' of variables! %d > %d' %
                         (n_factor, k_endog))

    if np.max(np.abs(np.diag(corr) - 1)) > 1e-10:
        raise ValueError("corr must be a correlation matrix")

    if corr.shape[0] != corr.shape[1]:
        raise ValueError('Correlation matrix corr must be a square '
                         '(rows %d != cols %d)' % corr.shape)


class Factor(Model):
    """
    Factor analysis

    Parameters
    ----------
    endog : array_like
        Variables in columns, observations in rows.  May be `None` if
        `corr` is not `None`.
    n_factor : int
        The number of factors to extract
    corr : array_like
        Directly specify the correlation matrix instead of estimating
        it from `endog`.  If provided, `endog` is not used for the
        factor analysis, it may be used in post-estimation.
    method : str
        The method to extract factors, currently must be either 'pa'
        for principal axis factor analysis or 'ml' for maximum
        likelihood estimation.
    smc : True or False
        Whether or not to apply squared multiple correlations (method='pa')
    endog_names : str
        Names of endogenous variables.  If specified, it will be used
        instead of the column names in endog
    nobs : int
        The number of observations, not used if endog is present. Needs to
        be provided for inference if endog is None.
    missing : 'none', 'drop', or 'raise'
        Missing value handling for endog, default is row-wise deletion 'drop'
        If 'none', no nan checking is done. If 'drop', any observations with
        nans are dropped. If 'raise', an error is raised.


    Notes
    -----
    **Experimental**

    Supported rotations: 'varimax', 'quartimax', 'biquartimax',
    'equamax', 'oblimin', 'parsimax', 'parsimony', 'biquartimin',
    'promax'

    If method='ml', the factors are rotated to satisfy condition IC3
    of Bai and Li (2012).  This means that the scores have covariance
    I, so the model for the covariance matrix is L * L' + diag(U),
    where L are the loadings and U are the uniquenesses.  In addition,
    L' * diag(U)^{-1} L must be diagonal.

    References
    ----------
    .. [*] Hofacker, C. (2004). Exploratory Factor Analysis, Mathematical
       Marketing. http://www.openaccesstexts.org/pdf/Quant_Chapter_11_efa.pdf
    .. [*] J Bai, K Li (2012).  Statistical analysis of factor models of high
       dimension.  Annals of Statistics. https://arxiv.org/pdf/1205.6617.pdf
    """
    def __init__(self, endog=None, n_factor=1, corr=None, method='pa',
                 smc=True, endog_names=None, nobs=None, missing='drop'):

        _check_args_1(endog, n_factor, corr, nobs)

        if endog is not None:
            super(Factor, self).__init__(endog, exog=None, missing=missing)
            endog = self.endog   # after preprocessing like missing, asarray
            k_endog = endog.shape[1]
            nobs = endog.shape[0]
            corr = self.corr = np.corrcoef(endog, rowvar=0)
        elif corr is not None:
            corr = self.corr = np.asarray(corr)
            k_endog = self.corr.shape[0]
            self.endog = None
        else:
            msg = "Either endog or corr must be provided."
            raise ValueError(msg)

        _check_args_2(endog, n_factor, corr, nobs, k_endog)

        self.n_factor = n_factor
        self.loadings = None
        self.communality = None
        self.method = method
        self.smc = smc
        self.nobs = nobs
        self.method = method
        self.corr = corr
        self.k_endog = k_endog

        if endog_names is None:
            if hasattr(corr, 'index'):
                endog_names = corr.index
            if hasattr(corr, 'columns'):
                endog_names = corr.columns
        self.endog_names = endog_names

    @property
    def endog_names(self):
        """Names of endogenous variables"""
        if self._endog_names is not None:
            return self._endog_names
        else:
            if self.endog is not None:
                return self.data.ynames
            else:
                d = 0
                n = self.corr.shape[0] - 1
                while n > 0:
                    d += 1
                    n //= 10
                return [('var%0' + str(d) + 'd') % i
                        for i in range(self.corr.shape[0])]

    @endog_names.setter
    def endog_names(self, value):
        # Check validity of endog_names:
        if value is not None:
            if len(value) != self.corr.shape[0]:
                raise ValueError('The length of `endog_names` must '
                                 'equal the number of variables.')
            self._endog_names = np.asarray(value)
        else:
            self._endog_names = None

    def fit(self, maxiter=50, tol=1e-8, start=None, opt_method='BFGS',
            opt=None, em_iter=3):
        """
        Estimate factor model parameters.

        Parameters
        ----------
        maxiter : int
            Maximum number of iterations for iterative estimation algorithms
        tol : float
            Stopping criteria (error tolerance) for iterative estimation
            algorithms
        start : array_like
            Starting values, currently only used for ML estimation
        opt_method : str
            Optimization method for ML estimation
        opt : dict-like
            Keyword arguments passed to optimizer, only used for ML estimation
        em_iter : int
            The number of EM iterations before starting gradient optimization,
            only used for ML estimation.

        Returns
        -------
        FactorResults
            Results class instance.
        """
        method = self.method.lower()
        if method == 'pa':
            return self._fit_pa(maxiter=maxiter, tol=tol)
        elif method == 'ml':
            return self._fit_ml(start, em_iter, opt_method, opt)
        else:
            msg = "Unknown factor extraction approach '%s'" % self.method
            raise ValueError(msg)

    def _fit_pa(self, maxiter=50, tol=1e-8):
        """
        Extract factors using the iterative principal axis method

        Parameters
        ----------
        maxiter : int
            Maximum number of iterations for communality estimation
        tol : float
            If `norm(communality - last_communality)  < tolerance`,
            estimation stops

        Returns
        -------
        results : FactorResults instance
        """

        R = self.corr.copy()  # inplace modification below

        # Parameter validation
        self.n_comp = matrix_rank(R)
        if self.n_factor > self.n_comp:
            raise ValueError('n_factor must be smaller or equal to the rank'
                             ' of endog! %d > %d' %
                             (self.n_factor, self.n_comp))
        if maxiter <= 0:
            raise ValueError('n_max_iter must be larger than 0! %d < 0' %
                             (maxiter))
        if tol <= 0 or tol > 0.01:
            raise ValueError('tolerance must be larger than 0 and smaller than'
                             ' 0.01! Got %f instead' % (tol))

        #  Initial communality estimation
        if self.smc:
            c = 1 - 1 / np.diag(inv(R))
        else:
            c = np.ones(len(R))

        # Iterative communality estimation
        eigenvals = None
        for i in range(maxiter):
            # Get eigenvalues/eigenvectors of R with diag replaced by
            # communality
            for j in range(len(R)):
                R[j, j] = c[j]
            L, V = eigh(R, UPLO='U')
            c_last = np.array(c)
            ind = np.argsort(L)
            ind = ind[::-1]
            L = L[ind]
            n_pos = (L > 0).sum()
            V = V[:, ind]
            eigenvals = np.array(L)

            # Select eigenvectors with positive eigenvalues
            n = np.min([n_pos, self.n_factor])
            sL = np.diag(np.sqrt(L[:n]))
            V = V[:, :n]

            # Calculate new loadings and communality
            A = V.dot(sL)
            c = np.power(A, 2).sum(axis=1)
            if norm(c_last - c) < tol:
                break

        self.eigenvals = eigenvals
        self.communality = c
        self.uniqueness = 1 - c
        self.loadings = A
        return FactorResults(self)

    # Unpacks the model parameters from a flat vector, used for ML
    # estimation.  The first k_endog elements of par are the square
    # roots of the uniquenesses.  The remaining elements are the
    # factor loadings, packed one factor at a time.
    def _unpack(self, par):
        return (par[0:self.k_endog]**2,
                np.reshape(par[self.k_endog:], (-1, self.k_endog)).T)

    # Packs the model parameters into a flat parameter, used for ML
    # estimation.
    def _pack(self, load, uniq):
        return np.concatenate((np.sqrt(uniq), load.T.flat))

    def loglike(self, par):
        """
        Evaluate the log-likelihood function.

        Parameters
        ----------
        par : ndarray or tuple of 2 ndarray's
            The model parameters, either a packed representation of
            the model parameters or a 2-tuple containing a `k_endog x
            n_factor` matrix of factor loadings and a `k_endog` vector
            of uniquenesses.

        Returns
        -------
        float
            The value of the log-likelihood evaluated at par.
        """

        if type(par) is np.ndarray:
            uniq, load = self._unpack(par)
        else:
            load, uniq = par[0], par[1]

        loadu = load / uniq[:, None]
        lul = np.dot(load.T, loadu)

        # log|GG' + S|
        # Using matrix determinant lemma:
        # |GG' + S| = |I + G'S^{-1}G|*|S|
        lul.flat[::lul.shape[0]+1] += 1
        _, ld = np.linalg.slogdet(lul)
        v = np.sum(np.log(uniq)) + ld

        # tr((GG' + S)^{-1}C)
        # Using Sherman-Morrison-Woodbury
        w = np.sum(1 / uniq)
        b = np.dot(load.T, self.corr / uniq[:, None])
        b = np.linalg.solve(lul, b)
        b = np.dot(loadu, b)
        w -= np.trace(b)

        # Scaled log-likelihood
        return -(v + w) / (2*self.k_endog)

    def score(self, par):
        """
        Evaluate the score function (first derivative of loglike).

        Parameters
        ----------
        par : ndarray or tuple of 2 ndarray's
            The model parameters, either a packed representation of
            the model parameters or a 2-tuple containing a `k_endog x
            n_factor` matrix of factor loadings and a `k_endog` vector
            of uniquenesses.

        Returns
        -------
        ndarray
            The score function evaluated at par.
        """

        if type(par) is np.ndarray:
            uniq, load = self._unpack(par)
        else:
            load, uniq = par[0], par[1]

        # Center term of SMW
        loadu = load / uniq[:, None]
        c = np.dot(load.T, loadu)
        c.flat[::c.shape[0]+1] += 1
        d = np.linalg.solve(c, load.T)

        # Precompute these terms
        lud = np.dot(loadu, d)
        cu = (self.corr / uniq) / uniq[:, None]
        r = np.dot(cu, load)
        lul = np.dot(lud.T, load)
        luz = np.dot(cu, lul)

        # First term
        du = 2*np.sqrt(uniq) * (1/uniq - (d * load.T).sum(0) / uniq**2)
        dl = 2*(loadu - np.dot(lud, loadu))

        # Second term
        h = np.dot(lud, cu)
        f = np.dot(h, lud.T)
        du -= 2*np.sqrt(uniq) * (np.diag(cu) - 2*np.diag(h) + np.diag(f))
        dl -= 2*r
        dl += 2*np.dot(lud, r)
        dl += 2*luz
        dl -= 2*np.dot(lud, luz)

        # Cannot use _pack because we are working with the square root
        # uniquenesses directly.
        return -np.concatenate((du, dl.T.flat)) / (2*self.k_endog)

    # Maximum likelihood factor analysis.
    def _fit_ml(self, start, em_iter, opt_method, opt):
        """estimate Factor model using Maximum Likelihood
        """

        # Starting values
        if start is None:
            load, uniq = self._fit_ml_em(em_iter)
            start = self._pack(load, uniq)
        elif len(start) == 2:
            if len(start[1]) != start[0].shape[0]:
                msg = "Starting values have incompatible dimensions"
                raise ValueError(msg)
            start = self._pack(start[0], start[1])
        else:
            raise ValueError("Invalid starting values")

        def nloglike(par):
            return -self.loglike(par)

        def nscore(par):
            return -self.score(par)

        # Do the optimization
        if opt is None:
            opt = _opt_defaults
        r = minimize(nloglike, start, jac=nscore, method=opt_method,
                     options=opt)
        if not r.success:
            warnings.warn("Fitting did not converge")
        par = r.x
        uniq, load = self._unpack(par)

        if uniq.min() < 1e-10:
            warnings.warn("Some uniquenesses are nearly zero")

        # Rotate solution to satisfy IC3 of Bai and Li
        load = self._rotate(load, uniq)

        self.uniqueness = uniq
        self.communality = 1 - uniq
        self.loadings = load
        self.mle_retvals = r

        return FactorResults(self)

    def _fit_ml_em(self, iter, random_state=None):
        """estimate Factor model using EM algorithm
        """
        # Starting values
        if random_state is None:
            random_state = np.random.RandomState(3427)
        load = 0.1 * random_state.standard_normal(size=(self.k_endog, self.n_factor))
        uniq = 0.5 * np.ones(self.k_endog)

        for k in range(iter):

            loadu = load / uniq[:, None]

            f = np.dot(load.T, loadu)
            f.flat[::f.shape[0]+1] += 1

            r = np.linalg.solve(f, loadu.T)
            q = np.dot(loadu.T, load)
            h = np.dot(r, load)

            c = load - np.dot(load, h)
            c /= uniq[:, None]

            g = np.dot(q, r)
            e = np.dot(g, self.corr)
            d = np.dot(loadu.T, self.corr) - e

            a = np.dot(d, c)
            a -= np.dot(load.T, c)
            a.flat[::a.shape[0]+1] += 1

            b = np.dot(self.corr, c)

            load = np.linalg.solve(a, b.T).T
            uniq = np.diag(self.corr) - (load * d.T).sum(1)

        return load, uniq

    def _rotate(self, load, uniq):
        """rotate loadings for MLE
        """
        # Rotations used in ML estimation.
        load, s, _ = np.linalg.svd(load, 0)
        load *= s

        if self.nobs is None:
            nobs = 1
        else:
            nobs = self.nobs

        cm = np.dot(load.T, load / uniq[:, None]) / nobs
        _, f = np.linalg.eig(cm)
        load = np.dot(load, f)
        return load


class FactorResults:
    """
    Factor results class

    For result summary, scree/loading plots and factor rotations

    Parameters
    ----------
    factor : Factor
        Fitted Factor class

    Attributes
    ----------
    uniqueness : ndarray
        The uniqueness (variance of uncorrelated errors unique to
        each variable)
    communality : ndarray
        1 - uniqueness
    loadings : ndarray
        Each column is the loading vector for one factor
    loadings_no_rot : ndarray
        Unrotated loadings, not available under maximum likelihood
        analysis.
    eigenvals : ndarray
        The eigenvalues for a factor analysis obtained using
        principal components; not available under ML estimation.
    n_comp : int
        Number of components (factors)
    nbs : int
        Number of observations
    fa_method : str
        The method used to obtain the decomposition, either 'pa' for
        'principal axes' or 'ml' for maximum likelihood.
    df : int
        Degrees of freedom of the factor model.

    Notes
    -----
    Under ML estimation, the default rotation (used for `loadings`) is
    condition IC3 of Bai and Li (2012).  Under this rotation, the
    factor scores are iid and standardized.  If `G` is the canonical
    loadings and `U` is the vector of uniquenesses, then the
    covariance matrix implied by the factor analysis is `GG' +
    diag(U)`.

    Status: experimental, Some refactoring will be necessary when new
        features are added.
    """
    def __init__(self, factor):
        self.model = factor
        self.endog_names = factor.endog_names
        self.loadings_no_rot = factor.loadings
        if hasattr(factor, "eigenvals"):
            self.eigenvals = factor.eigenvals

        self.communality = factor.communality
        self.uniqueness = factor.uniqueness
        self.rotation_method = None
        self.fa_method = factor.method
        self.n_comp = factor.loadings.shape[1]
        self.nobs = factor.nobs
        self._factor = factor
        if hasattr(factor, "mle_retvals"):
            self.mle_retvals = factor.mle_retvals

        p, k = self.loadings_no_rot.shape
        self.df = ((p - k)**2 - (p + k)) // 2

        # no rotation, overwritten in `rotate`
        self.loadings = factor.loadings
        self.rotation_matrix = np.eye(self.n_comp)


    def __str__(self):
        return self.summary().__str__()

    def rotate(self, method):
        """
        Apply rotation, inplace modification of this Results instance

        Parameters
        ----------
        method : str
            Rotation to be applied.  Allowed methods are varimax,
            quartimax, biquartimax, equamax, oblimin, parsimax,
            parsimony, biquartimin, promax.

        Returns
        -------
        None : nothing returned, modifications are inplace


        Notes
        -----
        Warning: 'varimax', 'quartimax' and 'oblimin' are verified against R or
        Stata. Some rotation methods such as promax do not produce the same
        results as the R or Stata default functions.

        See Also
        --------
        factor_rotation : subpackage that implements rotation methods
        """
        self.rotation_method = method
        if method not in ['varimax', 'quartimax', 'biquartimax',
                          'equamax', 'oblimin', 'parsimax', 'parsimony',
                          'biquartimin', 'promax']:
            raise ValueError('Unknown rotation method %s' % (method))

        if method in ['varimax', 'quartimax', 'biquartimax', 'equamax',
                      'parsimax', 'parsimony', 'biquartimin']:
            self.loadings, T = rotate_factors(self.loadings_no_rot, method)
        elif method == 'oblimin':
            self.loadings, T = rotate_factors(self.loadings_no_rot,
                                              'quartimin')
        elif method == 'promax':
            self.loadings, T = promax(self.loadings_no_rot)
        else:
            raise ValueError('rotation method not recognized')

        self.rotation_matrix = T

    def _corr_factors(self):
        """correlation of factors implied by rotation

        If the rotation is oblique, then the factors are correlated.

        currently not cached

        Returns
        -------
        corr_f : ndarray
            correlation matrix of rotated factors, assuming initial factors are
            orthogonal
        """
        T = self.rotation_matrix
        corr_f = T.T.dot(T)
        return corr_f

    def factor_score_params(self, method='bartlett'):
        """
        Compute factor scoring coefficient matrix

        The coefficient matrix is not cached.

        Parameters
        ----------
        method : 'bartlett' or 'regression'
            Method to use for factor scoring.
            'regression' can be abbreviated to `reg`

        Returns
        -------
        coeff_matrix : ndarray
            matrix s to compute factors f from a standardized endog ys.
            ``f = ys dot s``

        Notes
        -----
        The `regression` method follows the Stata definition.
        Method bartlett and regression are verified against Stats.
        Two unofficial methods, 'ols' and 'gls', produce similar factor scores
        but are not verified.

        See Also
        --------
        statsmodels.multivariate.factor.FactorResults.factor_scoring
        """
        L = self.loadings
        T = self.rotation_matrix.T
        #TODO: check row versus column convention for T
        uni = 1 - self.communality #self.uniqueness

        if method == 'bartlett':
            s_mat = np.linalg.inv(L.T.dot(L/(uni[:,None]))).dot((L.T / uni)).T
        elif method.startswith('reg'):
            corr = self.model.corr
            corr_f = self._corr_factors()
            # if orthogonal then corr_f is just eye
            s_mat = corr_f.dot(L.T.dot(np.linalg.inv(corr))).T
        elif method == 'ols':
            # not verified
            corr = self.model.corr
            corr_f = self._corr_factors()
            s_mat = corr_f.dot(np.linalg.pinv(L)).T
        elif method == 'gls':
            # not verified
            #s_mat = np.linalg.inv(1*np.eye(L.shape[1]) + L.T.dot(L/(uni[:,None])))
            corr = self.model.corr
            corr_f = self._corr_factors()
            s_mat = np.linalg.inv(np.linalg.inv(corr_f) + L.T.dot(L/(uni[:,None])))
            s_mat = s_mat.dot(L.T / uni).T
        else:
            raise ValueError('method not available, use "bartlett ' +
                             'or "regression"')
        return s_mat

    def factor_scoring(self, endog=None, method='bartlett', transform=True):
        """
        factor scoring: compute factors for endog

        If endog was not provided when creating the factor class, then
        a standarized endog needs to be provided here.

        Parameters
        ----------
        method : 'bartlett' or 'regression'
            Method to use for factor scoring.
            'regression' can be abbreviated to `reg`
        transform : bool
            If transform is true and endog is provided, then it will be
            standardized using mean and scale of original data, which has to
            be available in this case.
            If transform is False, then a provided endog will be used unchanged.
            The original endog in the Factor class will
            always be standardized if endog is None, independently of `transform`.

        Returns
        -------
        factor_score : ndarray
            estimated factors using scoring matrix s and standarized endog ys
            ``f = ys dot s``

        Notes
        -----
        Status: transform option is experimental and might change.

        See Also
        --------
        statsmodels.multivariate.factor.FactorResults.factor_score_params
        """

        if transform is False and endog is not None:
            # no transformation in this case
            endog = np.asarray(endog)
        else:
            # we need to standardize with the original mean and scale
            if self.model.endog is not None:
                m = self.model.endog.mean(0)
                s = self.model.endog.std(ddof=1, axis=0)
                if endog is None:
                    endog = self.model.endog
                else:
                    endog = np.asarray(endog)
            else:
                raise ValueError('If transform is True, then `endog` needs ' +
                                 'to be available in the Factor instance.')

            endog = (endog - m) / s

        s_mat = self.factor_score_params(method=method)
        factors = endog.dot(s_mat)
        return factors

    def summary(self):
        """Summary"""
        summ = summary2.Summary()
        summ.add_title('Factor analysis results')
        loadings_no_rot = pd.DataFrame(
            self.loadings_no_rot,
            columns=["factor %d" % (i)
                     for i in range(self.loadings_no_rot.shape[1])],
            index=self.endog_names
        )
        if hasattr(self, "eigenvals"):
            # eigenvals not available for ML method
            eigenvals = pd.DataFrame(
                [self.eigenvals], columns=self.endog_names, index=[''])
            summ.add_dict({'': 'Eigenvalues'})
            summ.add_df(eigenvals)
        communality = pd.DataFrame([self.communality],
                                   columns=self.endog_names, index=[''])
        summ.add_dict({'': ''})
        summ.add_dict({'': 'Communality'})
        summ.add_df(communality)
        summ.add_dict({'': ''})
        summ.add_dict({'': 'Pre-rotated loadings'})
        summ.add_df(loadings_no_rot)
        summ.add_dict({'': ''})
        if self.rotation_method is not None:
            loadings = pd.DataFrame(
                self.loadings,
                columns=["factor %d" % (i)
                         for i in range(self.loadings.shape[1])],
                index=self.endog_names
            )
            summ.add_dict({'': '%s rotated loadings' % (self.rotation_method)})
            summ.add_df(loadings)
        return summ

    def get_loadings_frame(self, style='display', sort_=True, threshold=0.3,
                           highlight_max=True, color_max='yellow',
                           decimals=None):
        """get loadings matrix as DataFrame or pandas Styler

        Parameters
        ----------
        style : 'display' (default), 'raw' or 'strings'
            Style to use for display

            * 'raw' returns just a DataFrame of the loadings matrix, no options are
               applied
            * 'display' add sorting and styling as defined by other keywords
            * 'strings' returns a DataFrame with string elements with optional sorting
               and suppressing small loading coefficients.

        sort_ : bool
            If True, then the rows of the DataFrame is sorted by contribution of each
            factor. applies if style is either 'display' or 'strings'
        threshold : float
            If the threshold is larger than zero, then loading coefficients are
            either colored white (if style is 'display') or replace by empty
            string (if style is 'strings').
        highlight_max : bool
            This add a background color to the largest coefficient in each row.
        color_max : html color
            default is 'yellow'. color for background of row maximum
        decimals : None or int
            If None, then pandas default precision applies. Otherwise values are
            rounded to the specified decimals. If style is 'display', then the
            underlying dataframe is not changed. If style is 'strings', then
            values are rounded before conversion to strings.

        Returns
        -------
        loadings : DataFrame or pandas Styler instance
            The return is a pandas Styler instance, if style is 'display' and
            at least one of highlight_max, threshold or decimals is applied.
            Otherwise, the returned loadings is a DataFrame.

        Examples
        --------
        >>> mod = Factor(df, 3, smc=True)
        >>> res = mod.fit()
        >>> res.get_loadings_frame(style='display', decimals=3, threshold=0.2)

        To get a sorted DataFrame, all styling options need to be turned off:

        >>> df_sorted = res.get_loadings_frame(style='display',
        ...             highlight_max=False, decimals=None, threshold=0)

        Options except for highlighting are available for plain test or Latex
        usage:

        >>> lds = res_u.get_loadings_frame(style='strings', decimals=3,
        ...                                threshold=0.3)
        >>> print(lds.to_latex())
        """

        loadings_df = pd.DataFrame(
                self.loadings,
                columns=["factor %d" % (i)
                         for i in range(self.loadings.shape[1])],
                index=self.endog_names
                )

        if style not in ['raw', 'display', 'strings']:
            msg = "style has to be one of 'raw', 'display', 'strings'"
            raise ValueError(msg)

        if style == 'raw':
            return loadings_df

        # add sorting and some formatting
        if sort_ is True:
            loadings_df2 = loadings_df.copy()
            n_f = len(loadings_df2)
            high = np.abs(loadings_df2.values).argmax(1)
            loadings_df2['high'] = high
            loadings_df2['largest'] = np.abs(loadings_df.values[np.arange(n_f), high])
            loadings_df2.sort_values(by=['high', 'largest'], ascending=[True, False], inplace=True)
            loadings_df = loadings_df2.drop(['high', 'largest'], axis=1)

        if style == 'display':
            sty = None
            if threshold > 0:
                def color_white_small(val):
                    """
                    Takes a scalar and returns a string with
                    the css property `'color: white'` for small values, black otherwise.

                    takes threshold from outer scope
                    """
                    color = 'white' if np.abs(val) < threshold else 'black'
                    return 'color: %s' % color
                try:
                    sty = loadings_df.style.map(color_white_small)
                except AttributeError:
                    # Deprecated in pandas 2.1
                    sty = loadings_df.style.applymap(color_white_small)

            if highlight_max is True:
                def highlight_max(s):
                    '''
                    highlight the maximum in a Series yellow.
                    '''
                    s = np.abs(s)
                    is_max = s == s.max()
                    return ['background-color: '+ color_max if v else '' for v in is_max]

                if sty is None:
                    sty = loadings_df.style

                sty = sty.apply(highlight_max, axis=1)

            if decimals is not None:
                if sty is None:
                    sty = loadings_df.style

                sty.format("{:.%sf}" % decimals)

            if sty is None:
                return loadings_df
            else:
                return sty

        if style == 'strings':
            ld = loadings_df
            if decimals is not None:
                ld = ld.round(decimals)
            ld = ld.astype(str)
            if threshold > 0:
                ld[loadings_df.abs() < threshold] = ''
            return ld

    def plot_scree(self, ncomp=None):
        """
        Plot of the ordered eigenvalues and variance explained for the loadings

        Parameters
        ----------
        ncomp : int, optional
            Number of loadings to include in the plot.  If None, will
            included the same as the number of maximum possible loadings

        Returns
        -------
        Figure
            Handle to the figure.
        """
        _import_mpl()
        from .plots import plot_scree
        return plot_scree(self.eigenvals, self.n_comp, ncomp)

    def plot_loadings(self, loading_pairs=None, plot_prerotated=False):
        """
        Plot factor loadings in 2-d plots

        Parameters
        ----------
        loading_pairs : None or a list of tuples
            Specify plots. Each tuple (i, j) represent one figure, i and j is
            the loading number for x-axis and y-axis, respectively. If `None`,
            all combinations of the loadings will be plotted.
        plot_prerotated : True or False
            If True, the loadings before rotation applied will be plotted. If
            False, rotated loadings will be plotted.

        Returns
        -------
        figs : a list of figure handles
        """
        _import_mpl()
        from .plots import plot_loadings

        if self.rotation_method is None:
            plot_prerotated = True
        loadings = self.loadings_no_rot if plot_prerotated else self.loadings
        if plot_prerotated:
            title = 'Prerotated Factor Pattern'
        else:
            title = '%s Rotated Factor Pattern' % (self.rotation_method)
        var_explained = self.eigenvals / self.n_comp * 100

        return plot_loadings(loadings, loading_pairs=loading_pairs,
                             title=title, row_names=self.endog_names,
                             percent_variance=var_explained)

    @cache_readonly
    def fitted_cov(self):
        """
        Returns the fitted covariance matrix.
        """

        c = np.dot(self.loadings, self.loadings.T)
        c.flat[::c.shape[0]+1] += self.uniqueness
        return c

    @cache_readonly
    def uniq_stderr(self, kurt=0):
        """
        The standard errors of the uniquenesses.

        Parameters
        ----------
        kurt : float
            Excess kurtosis

        Notes
        -----
        If excess kurtosis is known, provide as `kurt`.  Standard
        errors are only available if the model was fit using maximum
        likelihood.  If `endog` is not provided, `nobs` must be
        provided to obtain standard errors.

        These are asymptotic standard errors.  See Bai and Li (2012)
        for conditions under which the standard errors are valid.

        The standard errors are only applicable to the original,
        unrotated maximum likelihood solution.
        """

        if self.fa_method.lower() != "ml":
            msg = "Standard errors only available under ML estimation"
            raise ValueError(msg)

        if self.nobs is None:
            msg = "nobs is required to obtain standard errors."
            raise ValueError(msg)

        v = self.uniqueness**2 * (2 + kurt)
        return np.sqrt(v / self.nobs)

    @cache_readonly
    def load_stderr(self):
        """
        The standard errors of the loadings.

        Standard errors are only available if the model was fit using
        maximum likelihood.  If `endog` is not provided, `nobs` must be
        provided to obtain standard errors.

        These are asymptotic standard errors.  See Bai and Li (2012)
        for conditions under which the standard errors are valid.

        The standard errors are only applicable to the original,
        unrotated maximum likelihood solution.
        """

        if self.fa_method.lower() != "ml":
            msg = "Standard errors only available under ML estimation"
            raise ValueError(msg)

        if self.nobs is None:
            msg = "nobs is required to obtain standard errors."
            raise ValueError(msg)

        v = np.outer(self.uniqueness, np.ones(self.loadings.shape[1]))
        return np.sqrt(v / self.nobs)
