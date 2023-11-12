"""Principal Component Analysis

Author: josef-pktd
Modified by Kevin Sheppard
"""

import numpy as np
import pandas as pd

from statsmodels.tools.sm_exceptions import (ValueWarning,
                                             EstimationWarning)
from statsmodels.tools.validation import (string_like,
                                          array_like,
                                          bool_like,
                                          float_like,
                                          int_like,
                                          )


def _norm(x):
    return np.sqrt(np.sum(x * x))


class PCA:
    """
    Principal Component Analysis

    Parameters
    ----------
    data : array_like
        Variables in columns, observations in rows.
    ncomp : int, optional
        Number of components to return.  If None, returns the as many as the
        smaller of the number of rows or columns in data.
    standardize : bool, optional
        Flag indicating to use standardized data with mean 0 and unit
        variance.  standardized being True implies demean.  Using standardized
        data is equivalent to computing principal components from the
        correlation matrix of data.
    demean : bool, optional
        Flag indicating whether to demean data before computing principal
        components.  demean is ignored if standardize is True. Demeaning data
        but not standardizing is equivalent to computing principal components
        from the covariance matrix of data.
    normalize : bool , optional
        Indicates whether to normalize the factors to have unit inner product.
        If False, the loadings will have unit inner product.
    gls : bool, optional
        Flag indicating to implement a two-step GLS estimator where
        in the first step principal components are used to estimate residuals,
        and then the inverse residual variance is used as a set of weights to
        estimate the final principal components.  Setting gls to True requires
        ncomp to be less then the min of the number of rows or columns.
    weights : ndarray, optional
        Series weights to use after transforming data according to standardize
        or demean when computing the principal components.
    method : str, optional
        Sets the linear algebra routine used to compute eigenvectors:

        * 'svd' uses a singular value decomposition (default).
        * 'eig' uses an eigenvalue decomposition of a quadratic form
        * 'nipals' uses the NIPALS algorithm and can be faster than SVD when
          ncomp is small and nvars is large. See notes about additional changes
          when using NIPALS.
    missing : {str, None}
        Method for missing data.  Choices are:

        * 'drop-row' - drop rows with missing values.
        * 'drop-col' - drop columns with missing values.
        * 'drop-min' - drop either rows or columns, choosing by data retention.
        * 'fill-em' - use EM algorithm to fill missing value.  ncomp should be
          set to the number of factors required.
        * `None` raises if data contains NaN values.
    tol : float, optional
        Tolerance to use when checking for convergence when using NIPALS.
    max_iter : int, optional
        Maximum iterations when using NIPALS.
    tol_em : float
        Tolerance to use when checking for convergence of the EM algorithm.
    max_em_iter : int
        Maximum iterations for the EM algorithm.
    svd_full_matrices : bool, optional
        If the 'svd' method is selected, this flag is used to set the parameter
        'full_matrices' in the singular value decomposition method. Is set to
        False by default.

    Attributes
    ----------
    factors : array or DataFrame
        nobs by ncomp array of principal components (scores)
    scores :  array or DataFrame
        nobs by ncomp array of principal components - identical to factors
    loadings : array or DataFrame
        ncomp by nvar array of principal component loadings for constructing
        the factors
    coeff : array or DataFrame
        nvar by ncomp array of principal component loadings for constructing
        the projections
    projection : array or DataFrame
        nobs by var array containing the projection of the data onto the ncomp
        estimated factors
    rsquare : array or Series
        ncomp array where the element in the ith position is the R-square
        of including the fist i principal components.  Note: values are
        calculated on the transformed data, not the original data
    ic : array or DataFrame
        ncomp by 3 array containing the Bai and Ng (2003) Information
        criteria.  Each column is a different criteria, and each row
        represents the number of included factors.
    eigenvals : array or Series
        nvar array of eigenvalues
    eigenvecs : array or DataFrame
        nvar by nvar array of eigenvectors
    weights : ndarray
        nvar array of weights used to compute the principal components,
        normalized to unit length
    transformed_data : ndarray
        Standardized, demeaned and weighted data used to compute
        principal components and related quantities
    cols : ndarray
        Array of indices indicating columns used in the PCA
    rows : ndarray
        Array of indices indicating rows used in the PCA

    Notes
    -----
    The default options perform principal component analysis on the
    demeaned, unit variance version of data.  Setting standardize to False
    will instead only demean, and setting both standardized and
    demean to False will not alter the data.

    Once the data have been transformed, the following relationships hold when
    the number of components (ncomp) is the same as tne minimum of the number
    of observation or the number of variables.

    .. math:

        X' X = V \\Lambda V'

    .. math:

        F = X V

    .. math:

        X = F V'

    where X is the `data`, F is the array of principal components (`factors`
    or `scores`), and V is the array of eigenvectors (`loadings`) and V' is
    the array of factor coefficients (`coeff`).

    When weights are provided, the principal components are computed from the
    modified data

    .. math:

        \\Omega^{-\\frac{1}{2}} X

    where :math:`\\Omega` is a diagonal matrix composed of the weights. For
    example, when using the GLS version of PCA, the elements of :math:`\\Omega`
    will be the inverse of the variances of the residuals from

    .. math:

        X - F V'

    where the number of factors is less than the rank of X

    References
    ----------
    .. [*] J. Bai and S. Ng, "Determining the number of factors in approximate
       factor models," Econometrica, vol. 70, number 1, pp. 191-221, 2002

    Examples
    --------
    Basic PCA using the correlation matrix of the data

    >>> import numpy as np
    >>> from statsmodels.multivariate.pca import PCA
    >>> x = np.random.randn(100)[:, None]
    >>> x = x + np.random.randn(100, 100)
    >>> pc = PCA(x)

    Note that the principal components are computed using a SVD and so the
    correlation matrix is never constructed, unless method='eig'.

    PCA using the covariance matrix of the data

    >>> pc = PCA(x, standardize=False)

    Limiting the number of factors returned to 1 computed using NIPALS

    >>> pc = PCA(x, ncomp=1, method='nipals')
    >>> pc.factors.shape
    (100, 1)
    """

    def __init__(self, data, ncomp=None, standardize=True, demean=True,
                 normalize=True, gls=False, weights=None, method='svd',
                 missing=None, tol=5e-8, max_iter=1000, tol_em=5e-8,
                 max_em_iter=100, svd_full_matrices=False):
        self._index = None
        self._columns = []
        if isinstance(data, pd.DataFrame):
            self._index = data.index
            self._columns = data.columns

        self.data = array_like(data, "data", ndim=2)
        # Store inputs
        self._gls = bool_like(gls, "gls")
        self._normalize = bool_like(normalize, "normalize")
        self._svd_full_matrices = bool_like(svd_full_matrices, "svd_fm")
        self._tol = float_like(tol, "tol")
        if not 0 < self._tol < 1:
            raise ValueError('tol must be strictly between 0 and 1')
        self._max_iter = int_like(max_iter, "int_like")
        self._max_em_iter = int_like(max_em_iter, "max_em_iter")
        self._tol_em = float_like(tol_em, "tol_em")

        # Prepare data
        self._standardize = bool_like(standardize, "standardize")
        self._demean = bool_like(demean, "demean")

        self._nobs, self._nvar = self.data.shape
        weights = array_like(weights, "weights", maxdim=1, optional=True)
        if weights is None:
            weights = np.ones(self._nvar)
        else:
            weights = np.array(weights).flatten()
            if weights.shape[0] != self._nvar:
                raise ValueError('weights should have nvar elements')
            weights = weights / np.sqrt((weights ** 2.0).mean())
        self.weights = weights

        # Check ncomp against maximum
        min_dim = min(self._nobs, self._nvar)
        self._ncomp = min_dim if ncomp is None else ncomp
        if self._ncomp > min_dim:
            import warnings

            warn = 'The requested number of components is more than can be ' \
                   'computed from data. The maximum number of components is ' \
                   'the minimum of the number of observations or variables'
            warnings.warn(warn, ValueWarning)
            self._ncomp = min_dim

        self._method = method
        # Workaround to avoid instance methods in __dict__
        if self._method not in ('eig', 'svd', 'nipals'):
            raise ValueError('method {0} is not known.'.format(method))
        if self._method == 'svd':
            self._svd_full_matrices = True

        self.rows = np.arange(self._nobs)
        self.cols = np.arange(self._nvar)
        # Handle missing
        self._missing = string_like(missing, "missing", optional=True)
        self._adjusted_data = self.data
        self._adjust_missing()

        # Update size
        self._nobs, self._nvar = self._adjusted_data.shape
        if self._ncomp == np.min(self.data.shape):
            self._ncomp = np.min(self._adjusted_data.shape)
        elif self._ncomp > np.min(self._adjusted_data.shape):
            raise ValueError('When adjusting for missing values, user '
                             'provided ncomp must be no larger than the '
                             'smallest dimension of the '
                             'missing-value-adjusted data size.')

        # Attributes and internal values
        self._tss = 0.0
        self._ess = None
        self.transformed_data = None
        self._mu = None
        self._sigma = None
        self._ess_indiv = None
        self._tss_indiv = None
        self.scores = self.factors = None
        self.loadings = None
        self.coeff = None
        self.eigenvals = None
        self.eigenvecs = None
        self.projection = None
        self.rsquare = None
        self.ic = None

        # Prepare data
        self.transformed_data = self._prepare_data()
        # Perform the PCA
        self._pca()
        if gls:
            self._compute_gls_weights()
            self.transformed_data = self._prepare_data()
            self._pca()

        # Final calculations
        self._compute_rsquare_and_ic()
        if self._index is not None:
            self._to_pandas()

    def _adjust_missing(self):
        """
        Implements alternatives for handling missing values
        """

        def keep_col(x):
            index = np.logical_not(np.any(np.isnan(x), 0))
            return x[:, index], index

        def keep_row(x):
            index = np.logical_not(np.any(np.isnan(x), 1))
            return x[index, :], index

        if self._missing == 'drop-col':
            self._adjusted_data, index = keep_col(self.data)
            self.cols = np.where(index)[0]
            self.weights = self.weights[index]
        elif self._missing == 'drop-row':
            self._adjusted_data, index = keep_row(self.data)
            self.rows = np.where(index)[0]
        elif self._missing == 'drop-min':
            drop_col, drop_col_index = keep_col(self.data)
            drop_col_size = drop_col.size

            drop_row, drop_row_index = keep_row(self.data)
            drop_row_size = drop_row.size

            if drop_row_size > drop_col_size:
                self._adjusted_data = drop_row
                self.rows = np.where(drop_row_index)[0]
            else:
                self._adjusted_data = drop_col
                self.weights = self.weights[drop_col_index]
                self.cols = np.where(drop_col_index)[0]
        elif self._missing == 'fill-em':
            self._adjusted_data = self._fill_missing_em()
        elif self._missing is None:
            if not np.isfinite(self._adjusted_data).all():
                raise ValueError("""\
data contains non-finite values (inf, NaN). You should drop these values or
use one of the methods for adjusting data for missing-values.""")
        else:
            raise ValueError('missing method is not known.')

        if self._index is not None:
            self._columns = self._columns[self.cols]
            self._index = self._index[self.rows]

        # Check adjusted data size
        if self._adjusted_data.size == 0:
            raise ValueError('Removal of missing values has eliminated '
                             'all data.')

    def _compute_gls_weights(self):
        """
        Computes GLS weights based on percentage of data fit
        """
        projection = np.asarray(self.project(transform=False))
        errors = self.transformed_data - projection
        if self._ncomp == self._nvar:
            raise ValueError('gls can only be used when ncomp < nvar '
                             'so that residuals have non-zero variance')
        var = (errors ** 2.0).mean(0)
        weights = 1.0 / var
        weights = weights / np.sqrt((weights ** 2.0).mean())
        nvar = self._nvar
        eff_series_perc = (1.0 / sum((weights / weights.sum()) ** 2.0)) / nvar
        if eff_series_perc < 0.1:
            eff_series = int(np.round(eff_series_perc * nvar))
            import warnings

            warn = f"""\
Many series are being down weighted by GLS. Of the {nvar} series, the GLS
estimates are based on only {eff_series} (effective) series."""
            warnings.warn(warn, EstimationWarning)

        self.weights = weights

    def _pca(self):
        """
        Main PCA routine
        """
        self._compute_eig()
        self._compute_pca_from_eig()
        self.projection = self.project()

    def __repr__(self):
        string = self.__str__()
        string = string[:-1]
        string += ', id: ' + hex(id(self)) + ')'
        return string

    def __str__(self):
        string = 'Principal Component Analysis('
        string += 'nobs: ' + str(self._nobs) + ', '
        string += 'nvar: ' + str(self._nvar) + ', '
        if self._standardize:
            kind = 'Standardize (Correlation)'
        elif self._demean:
            kind = 'Demean (Covariance)'
        else:
            kind = 'None'
        string += 'transformation: ' + kind + ', '
        if self._gls:
            string += 'GLS, '
        string += 'normalization: ' + str(self._normalize) + ', '
        string += 'number of components: ' + str(self._ncomp) + ', '
        string += 'method: ' + 'Eigenvalue' if self._method == 'eig' else 'SVD'
        string += ')'
        return string

    def _prepare_data(self):
        """
        Standardize or demean data.
        """
        adj_data = self._adjusted_data
        if np.all(np.isnan(adj_data)):
            return np.empty(adj_data.shape[1]).fill(np.nan)

        self._mu = np.nanmean(adj_data, axis=0)
        self._sigma = np.sqrt(np.nanmean((adj_data - self._mu) ** 2.0, axis=0))
        if self._standardize:
            data = (adj_data - self._mu) / self._sigma
        elif self._demean:
            data = (adj_data - self._mu)
        else:
            data = adj_data
        return data / np.sqrt(self.weights)

    def _compute_eig(self):
        """
        Wrapper for actual eigenvalue method

        This is a workaround to avoid instance methods in __dict__
        """
        if self._method == 'eig':
            return self._compute_using_eig()
        elif self._method == 'svd':
            return self._compute_using_svd()
        else:  # self._method == 'nipals'
            return self._compute_using_nipals()

    def _compute_using_svd(self):
        """SVD method to compute eigenvalues and eigenvecs"""
        x = self.transformed_data
        u, s, v = np.linalg.svd(x, full_matrices=self._svd_full_matrices)
        self.eigenvals = s ** 2.0
        self.eigenvecs = v.T

    def _compute_using_eig(self):
        """
        Eigenvalue decomposition method to compute eigenvalues and eigenvectors
        """
        x = self.transformed_data
        self.eigenvals, self.eigenvecs = np.linalg.eigh(x.T.dot(x))

    def _compute_using_nipals(self):
        """
        NIPALS implementation to compute small number of eigenvalues
        and eigenvectors
        """
        x = self.transformed_data
        if self._ncomp > 1:
            x = x + 0.0  # Copy

        tol, max_iter, ncomp = self._tol, self._max_iter, self._ncomp
        vals = np.zeros(self._ncomp)
        vecs = np.zeros((self._nvar, self._ncomp))
        for i in range(ncomp):
            max_var_ind = np.argmax(x.var(0))
            factor = x[:, [max_var_ind]]
            _iter = 0
            diff = 1.0
            while diff > tol and _iter < max_iter:
                vec = x.T.dot(factor) / (factor.T.dot(factor))
                vec = vec / np.sqrt(vec.T.dot(vec))
                factor_last = factor
                factor = x.dot(vec) / (vec.T.dot(vec))
                diff = _norm(factor - factor_last) / _norm(factor)
                _iter += 1
            vals[i] = (factor ** 2).sum()
            vecs[:, [i]] = vec
            if ncomp > 1:
                x -= factor.dot(vec.T)

        self.eigenvals = vals
        self.eigenvecs = vecs

    def _fill_missing_em(self):
        """
        EM algorithm to fill missing values
        """
        non_missing = np.logical_not(np.isnan(self.data))

        # If nothing missing, return without altering the data
        if np.all(non_missing):
            return self.data

        # 1. Standardized data as needed
        data = self.transformed_data = np.asarray(self._prepare_data())

        ncomp = self._ncomp

        # 2. Check for all nans
        col_non_missing = np.sum(non_missing, 1)
        row_non_missing = np.sum(non_missing, 0)
        if np.any(col_non_missing < ncomp) or np.any(row_non_missing < ncomp):
            raise ValueError('Implementation requires that all columns and '
                             'all rows have at least ncomp non-missing values')
        # 3. Get mask
        mask = np.isnan(data)

        # 4. Compute mean
        mu = np.nanmean(data, 0)

        # 5. Replace missing with mean
        projection = np.ones((self._nobs, 1)) * mu
        projection_masked = projection[mask]
        data[mask] = projection_masked

        # 6. Compute eigenvalues and fit
        diff = 1.0
        _iter = 0
        while diff > self._tol_em and _iter < self._max_em_iter:
            last_projection_masked = projection_masked
            # Set transformed data to compute eigenvalues
            self.transformed_data = data
            # Call correct eig function here
            self._compute_eig()
            # Call function to compute factors and projection
            self._compute_pca_from_eig()
            projection = np.asarray(self.project(transform=False,
                                                 unweight=False))
            projection_masked = projection[mask]
            data[mask] = projection_masked
            delta = last_projection_masked - projection_masked
            diff = _norm(delta) / _norm(projection_masked)
            _iter += 1
        # Must copy to avoid overwriting original data since replacing values
        data = self._adjusted_data + 0.0
        projection = np.asarray(self.project())
        data[mask] = projection[mask]

        return data

    def _compute_pca_from_eig(self):
        """
        Compute relevant statistics after eigenvalues have been computed
        """
        # Ensure sorted largest to smallest
        vals, vecs = self.eigenvals, self.eigenvecs
        indices = np.argsort(vals)
        indices = indices[::-1]
        vals = vals[indices]
        vecs = vecs[:, indices]
        if (vals <= 0).any():
            # Discard and warn
            num_good = vals.shape[0] - (vals <= 0).sum()
            if num_good < self._ncomp:
                import warnings

                warnings.warn('Only {num:d} eigenvalues are positive.  '
                              'This is the maximum number of components '
                              'that can be extracted.'.format(num=num_good),
                              EstimationWarning)

                self._ncomp = num_good
                vals[num_good:] = np.finfo(np.float64).tiny
        # Use ncomp for the remaining calculations
        vals = vals[:self._ncomp]
        vecs = vecs[:, :self._ncomp]
        self.eigenvals, self.eigenvecs = vals, vecs
        # Select correct number of components to return
        self.scores = self.factors = self.transformed_data.dot(vecs)
        self.loadings = vecs
        self.coeff = vecs.T
        if self._normalize:
            self.coeff = (self.coeff.T * np.sqrt(vals)).T
            self.factors /= np.sqrt(vals)
            self.scores = self.factors

    def _compute_rsquare_and_ic(self):
        """
        Final statistics to compute
        """
        # TSS and related calculations
        # TODO: This needs careful testing, with and without weights,
        #   gls, standardized and demean
        weights = self.weights
        ss_data = self.transformed_data * np.sqrt(weights)
        self._tss_indiv = np.sum(ss_data ** 2, 0)
        self._tss = np.sum(self._tss_indiv)
        self._ess = np.zeros(self._ncomp + 1)
        self._ess_indiv = np.zeros((self._ncomp + 1, self._nvar))
        for i in range(self._ncomp + 1):
            # Projection in the same space as transformed_data
            projection = self.project(ncomp=i, transform=False, unweight=False)
            indiv_rss = (projection ** 2).sum(axis=0)
            rss = indiv_rss.sum()
            self._ess[i] = self._tss - rss
            self._ess_indiv[i, :] = self._tss_indiv - indiv_rss
        self.rsquare = 1.0 - self._ess / self._tss
        # Information Criteria
        ess = self._ess
        invalid = ess <= 0  # Prevent log issues of 0
        if invalid.any():
            last_obs = (np.where(invalid)[0]).min()
            ess = ess[:last_obs]

        log_ess = np.log(ess)
        r = np.arange(ess.shape[0])

        nobs, nvar = self._nobs, self._nvar
        sum_to_prod = (nobs + nvar) / (nobs * nvar)
        min_dim = min(nobs, nvar)
        penalties = np.array([sum_to_prod * np.log(1.0 / sum_to_prod),
                              sum_to_prod * np.log(min_dim),
                              np.log(min_dim) / min_dim])
        penalties = penalties[:, None]
        ic = log_ess + r * penalties
        self.ic = ic.T

    def project(self, ncomp=None, transform=True, unweight=True):
        """
        Project series onto a specific number of factors.

        Parameters
        ----------
        ncomp : int, optional
            Number of components to use.  If omitted, all components
            initially computed are used.
        transform : bool, optional
            Flag indicating whether to return the projection in the original
            space of the data (True, default) or in the space of the
            standardized/demeaned data.
        unweight : bool, optional
            Flag indicating whether to undo the effects of the estimation
            weights.

        Returns
        -------
        array_like
            The nobs by nvar array of the projection onto ncomp factors.

        Notes
        -----
        """
        # Projection needs to be scaled/shifted based on inputs
        ncomp = self._ncomp if ncomp is None else ncomp
        if ncomp > self._ncomp:
            raise ValueError('ncomp must be smaller than the number of '
                             'components computed.')
        factors = np.asarray(self.factors)
        coeff = np.asarray(self.coeff)

        projection = factors[:, :ncomp].dot(coeff[:ncomp, :])
        if transform or unweight:
            projection *= np.sqrt(self.weights)
        if transform:
            # Remove the weights, which do not depend on transformation
            if self._standardize:
                projection *= self._sigma
            if self._standardize or self._demean:
                projection += self._mu
        if self._index is not None:
            projection = pd.DataFrame(projection,
                                      columns=self._columns,
                                      index=self._index)
        return projection

    def _to_pandas(self):
        """
        Returns pandas DataFrames for all values
        """
        index = self._index
        # Principal Components
        num_zeros = np.ceil(np.log10(self._ncomp))
        comp_str = 'comp_{0:0' + str(int(num_zeros)) + 'd}'
        cols = [comp_str.format(i) for i in range(self._ncomp)]
        df = pd.DataFrame(self.factors, columns=cols, index=index)
        self.scores = self.factors = df
        # Projections
        df = pd.DataFrame(self.projection,
                          columns=self._columns,
                          index=index)
        self.projection = df
        # Weights
        df = pd.DataFrame(self.coeff, index=cols,
                          columns=self._columns)
        self.coeff = df
        # Loadings
        df = pd.DataFrame(self.loadings,
                          index=self._columns, columns=cols)
        self.loadings = df
        # eigenvals
        self.eigenvals = pd.Series(self.eigenvals)
        self.eigenvals.name = 'eigenvals'
        # eigenvecs
        vec_str = comp_str.replace('comp', 'eigenvec')
        cols = [vec_str.format(i) for i in range(self.eigenvecs.shape[1])]
        self.eigenvecs = pd.DataFrame(self.eigenvecs, columns=cols)
        # R2
        self.rsquare = pd.Series(self.rsquare)
        self.rsquare.index.name = 'ncomp'
        self.rsquare.name = 'rsquare'
        # IC
        self.ic = pd.DataFrame(self.ic, columns=['IC_p1', 'IC_p2', 'IC_p3'])
        self.ic.index.name = 'ncomp'

    def plot_scree(self, ncomp=None, log_scale=True,
                   cumulative=False, ax=None):
        """
        Plot of the ordered eigenvalues

        Parameters
        ----------
        ncomp : int, optional
            Number of components ot include in the plot.  If None, will
            included the same as the number of components computed
        log_scale : boot, optional
            Flag indicating whether ot use a log scale for the y-axis
        cumulative : bool, optional
            Flag indicating whether to plot the eigenvalues or cumulative
            eigenvalues
        ax : AxesSubplot, optional
            An axes on which to draw the graph.  If omitted, new a figure
            is created

        Returns
        -------
        matplotlib.figure.Figure
            The handle to the figure.
        """
        import statsmodels.graphics.utils as gutils

        fig, ax = gutils.create_mpl_ax(ax)

        ncomp = self._ncomp if ncomp is None else ncomp
        vals = np.asarray(self.eigenvals)
        vals = vals[:self._ncomp]
        if cumulative:
            vals = np.cumsum(vals)

        if log_scale:
            ax.set_yscale('log')
        ax.plot(np.arange(ncomp), vals[: ncomp], 'bo')
        ax.autoscale(tight=True)
        xlim = np.array(ax.get_xlim())
        sp = xlim[1] - xlim[0]
        xlim += 0.02 * np.array([-sp, sp])
        ax.set_xlim(xlim)

        ylim = np.array(ax.get_ylim())
        scale = 0.02
        if log_scale:
            sp = np.log(ylim[1] / ylim[0])
            ylim = np.exp(np.array([np.log(ylim[0]) - scale * sp,
                                    np.log(ylim[1]) + scale * sp]))
        else:
            sp = ylim[1] - ylim[0]
            ylim += scale * np.array([-sp, sp])
        ax.set_ylim(ylim)
        ax.set_title('Scree Plot')
        ax.set_ylabel('Eigenvalue')
        ax.set_xlabel('Component Number')
        fig.tight_layout()

        return fig

    def plot_rsquare(self, ncomp=None, ax=None):
        """
        Box plots of the individual series R-square against the number of PCs.

        Parameters
        ----------
        ncomp : int, optional
            Number of components ot include in the plot.  If None, will
            plot the minimum of 10 or the number of computed components.
        ax : AxesSubplot, optional
            An axes on which to draw the graph.  If omitted, new a figure
            is created.

        Returns
        -------
        matplotlib.figure.Figure
            The handle to the figure.
        """
        import statsmodels.graphics.utils as gutils

        fig, ax = gutils.create_mpl_ax(ax)

        ncomp = 10 if ncomp is None else ncomp
        ncomp = min(ncomp, self._ncomp)
        # R2s in rows, series in columns
        r2s = 1.0 - self._ess_indiv / self._tss_indiv
        r2s = r2s[1:]
        r2s = r2s[:ncomp]
        ax.boxplot(r2s.T)
        ax.set_title('Individual Input $R^2$')
        ax.set_ylabel('$R^2$')
        ax.set_xlabel('Number of Included Principal Components')

        return fig


def pca(data, ncomp=None, standardize=True, demean=True, normalize=True,
        gls=False, weights=None, method='svd'):
    """
    Perform Principal Component Analysis (PCA).

    Parameters
    ----------
    data : ndarray
        Variables in columns, observations in rows.
    ncomp : int, optional
        Number of components to return.  If None, returns the as many as the
        smaller to the number of rows or columns of data.
    standardize : bool, optional
        Flag indicating to use standardized data with mean 0 and unit
        variance.  standardized being True implies demean.
    demean : bool, optional
        Flag indicating whether to demean data before computing principal
        components.  demean is ignored if standardize is True.
    normalize : bool , optional
        Indicates whether th normalize the factors to have unit inner
        product.  If False, the loadings will have unit inner product.
    gls : bool, optional
        Flag indicating to implement a two-step GLS estimator where
        in the first step principal components are used to estimate residuals,
        and then the inverse residual variance is used as a set of weights to
        estimate the final principal components
    weights : ndarray, optional
        Series weights to use after transforming data according to standardize
        or demean when computing the principal components.
    method : str, optional
        Determines the linear algebra routine uses.  'eig', the default,
        uses an eigenvalue decomposition. 'svd' uses a singular value
        decomposition.

    Returns
    -------
    factors : {ndarray, DataFrame}
        Array (nobs, ncomp) of principal components (also known as scores).
    loadings : {ndarray, DataFrame}
        Array (ncomp, nvar) of principal component loadings for constructing
        the factors.
    projection : {ndarray, DataFrame}
        Array (nobs, nvar) containing the projection of the data onto the ncomp
        estimated factors.
    rsquare : {ndarray, Series}
        Array (ncomp,) where the element in the ith position is the R-square
        of including the fist i principal components.  The values are
        calculated on the transformed data, not the original data.
    ic : {ndarray, DataFrame}
        Array (ncomp, 3) containing the Bai and Ng (2003) Information
        criteria.  Each column is a different criteria, and each row
        represents the number of included factors.
    eigenvals : {ndarray, Series}
        Array of eigenvalues (nvar,).
    eigenvecs : {ndarray, DataFrame}
        Array of eigenvectors. (nvar, nvar).

    Notes
    -----
    This is a simple function wrapper around the PCA class. See PCA for
    more information and additional methods.
    """
    pc = PCA(data, ncomp=ncomp, standardize=standardize, demean=demean,
             normalize=normalize, gls=gls, weights=weights, method=method)

    return (pc.factors, pc.loadings, pc.projection, pc.rsquare, pc.ic,
            pc.eigenvals, pc.eigenvecs)
