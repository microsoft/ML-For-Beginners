"""
Module containing the base object for multivariate kernel density and
regression, plus some utilities.
"""
import copy

import numpy as np
from scipy import optimize
from scipy.stats.mstats import mquantiles

try:
    import joblib
    has_joblib = True
except ImportError:
    has_joblib = False

from . import kernels


kernel_func = dict(wangryzin=kernels.wang_ryzin,
                   aitchisonaitken=kernels.aitchison_aitken,
                   gaussian=kernels.gaussian,
                   aitchison_aitken_reg = kernels.aitchison_aitken_reg,
                   wangryzin_reg = kernels.wang_ryzin_reg,
                   gauss_convolution=kernels.gaussian_convolution,
                   wangryzin_convolution=kernels.wang_ryzin_convolution,
                   aitchisonaitken_convolution=kernels.aitchison_aitken_convolution,
                   gaussian_cdf=kernels.gaussian_cdf,
                   aitchisonaitken_cdf=kernels.aitchison_aitken_cdf,
                   wangryzin_cdf=kernels.wang_ryzin_cdf,
                   d_gaussian=kernels.d_gaussian,
                   tricube=kernels.tricube)


def _compute_min_std_IQR(data):
    """Compute minimum of std and IQR for each variable."""
    s1 = np.std(data, axis=0)
    q75 = mquantiles(data, 0.75, axis=0).data[0]
    q25 = mquantiles(data, 0.25, axis=0).data[0]
    s2 = (q75 - q25) / 1.349  # IQR
    dispersion = np.minimum(s1, s2)
    return dispersion


def _compute_subset(class_type, data, bw, co, do, n_cvars, ix_ord,
                    ix_unord, n_sub, class_vars, randomize, bound):
    """"Compute bw on subset of data.

    Called from ``GenericKDE._compute_efficient_*``.

    Notes
    -----
    Needs to be outside the class in order for joblib to be able to pickle it.
    """
    if randomize:
        np.random.shuffle(data)
        sub_data = data[:n_sub, :]
    else:
        sub_data = data[bound[0]:bound[1], :]

    if class_type == 'KDEMultivariate':
        from .kernel_density import KDEMultivariate
        var_type = class_vars[0]
        sub_model = KDEMultivariate(sub_data, var_type, bw=bw,
                        defaults=EstimatorSettings(efficient=False))
    elif class_type == 'KDEMultivariateConditional':
        from .kernel_density import KDEMultivariateConditional
        k_dep, dep_type, indep_type = class_vars
        endog = sub_data[:, :k_dep]
        exog = sub_data[:, k_dep:]
        sub_model = KDEMultivariateConditional(endog, exog, dep_type,
            indep_type, bw=bw, defaults=EstimatorSettings(efficient=False))
    elif class_type == 'KernelReg':
        from .kernel_regression import KernelReg
        var_type, k_vars, reg_type = class_vars
        endog = _adjust_shape(sub_data[:, 0], 1)
        exog = _adjust_shape(sub_data[:, 1:], k_vars)
        sub_model = KernelReg(endog=endog, exog=exog, reg_type=reg_type,
                              var_type=var_type, bw=bw,
                              defaults=EstimatorSettings(efficient=False))
    else:
        raise ValueError("class_type not recognized, should be one of " \
                 "{KDEMultivariate, KDEMultivariateConditional, KernelReg}")

    # Compute dispersion in next 4 lines
    if class_type == 'KernelReg':
        sub_data = sub_data[:, 1:]

    dispersion = _compute_min_std_IQR(sub_data)

    fct = dispersion * n_sub**(-1. / (n_cvars + co))
    fct[ix_unord] = n_sub**(-2. / (n_cvars + do))
    fct[ix_ord] = n_sub**(-2. / (n_cvars + do))
    sample_scale_sub = sub_model.bw / fct  #TODO: check if correct
    bw_sub = sub_model.bw
    return sample_scale_sub, bw_sub


class GenericKDE (object):
    """
    Base class for density estimation and regression KDE classes.
    """
    def _compute_bw(self, bw):
        """
        Computes the bandwidth of the data.

        Parameters
        ----------
        bw : {array_like, str}
            If array_like: user-specified bandwidth.
            If a string, should be one of:

                - cv_ml: cross validation maximum likelihood
                - normal_reference: normal reference rule of thumb
                - cv_ls: cross validation least squares

        Notes
        -----
        The default values for bw is 'normal_reference'.
        """
        if bw is None:
            bw = 'normal_reference'

        if not isinstance(bw, str):
            self._bw_method = "user-specified"
            res = np.asarray(bw)
        else:
            # The user specified a bandwidth selection method
            self._bw_method = bw
            # Workaround to avoid instance methods in __dict__
            if bw == 'normal_reference':
                bwfunc = self._normal_reference
            elif bw == 'cv_ml':
                bwfunc = self._cv_ml
            else:  # bw == 'cv_ls'
                bwfunc = self._cv_ls
            res = bwfunc()

        return res

    def _compute_dispersion(self, data):
        """
        Computes the measure of dispersion.

        The minimum of the standard deviation and interquartile range / 1.349

        Notes
        -----
        Reimplemented in `KernelReg`, because the first column of `data` has to
        be removed.

        References
        ----------
        See the user guide for the np package in R.
        In the notes on bwscaling option in npreg, npudens, npcdens there is
        a discussion on the measure of dispersion
        """
        return _compute_min_std_IQR(data)

    def _get_class_vars_type(self):
        """Helper method to be able to pass needed vars to _compute_subset.

        Needs to be implemented by subclasses."""
        pass

    def _compute_efficient(self, bw):
        """
        Computes the bandwidth by estimating the scaling factor (c)
        in n_res resamples of size ``n_sub`` (in `randomize` case), or by
        dividing ``nobs`` into as many ``n_sub`` blocks as needed (if
        `randomize` is False).

        References
        ----------
        See p.9 in socserv.mcmaster.ca/racine/np_faq.pdf
        """

        if bw is None:
            self._bw_method = 'normal_reference'
        if isinstance(bw, str):
            self._bw_method = bw
        else:
            self._bw_method = "user-specified"
            return bw

        nobs = self.nobs
        n_sub = self.n_sub
        data = copy.deepcopy(self.data)
        n_cvars = self.data_type.count('c')
        co = 4  # 2*order of continuous kernel
        do = 4  # 2*order of discrete kernel
        _, ix_ord, ix_unord = _get_type_pos(self.data_type)

        # Define bounds for slicing the data
        if self.randomize:
            # randomize chooses blocks of size n_sub, independent of nobs
            bounds = [None] * self.n_res
        else:
            bounds = [(i * n_sub, (i+1) * n_sub) for i in range(nobs // n_sub)]
            if nobs % n_sub > 0:
                bounds.append((nobs - nobs % n_sub, nobs))

        n_blocks = self.n_res if self.randomize else len(bounds)
        sample_scale = np.empty((n_blocks, self.k_vars))
        only_bw = np.empty((n_blocks, self.k_vars))

        class_type, class_vars = self._get_class_vars_type()
        if has_joblib:
            # `res` is a list of tuples (sample_scale_sub, bw_sub)
            res = joblib.Parallel(n_jobs=self.n_jobs)(
                joblib.delayed(_compute_subset)(
                    class_type, data, bw, co, do, n_cvars, ix_ord, ix_unord, \
                    n_sub, class_vars, self.randomize, bounds[i]) \
                for i in range(n_blocks))
        else:
            res = []
            for i in range(n_blocks):
                res.append(_compute_subset(class_type, data, bw, co, do,
                                           n_cvars, ix_ord, ix_unord, n_sub,
                                           class_vars, self.randomize,
                                           bounds[i]))

        for i in range(n_blocks):
            sample_scale[i, :] = res[i][0]
            only_bw[i, :] = res[i][1]

        s = self._compute_dispersion(data)
        order_func = np.median if self.return_median else np.mean
        m_scale = order_func(sample_scale, axis=0)
        # TODO: Check if 1/5 is correct in line below!
        bw = m_scale * s * nobs**(-1. / (n_cvars + co))
        bw[ix_ord] = m_scale[ix_ord] * nobs**(-2./ (n_cvars + do))
        bw[ix_unord] = m_scale[ix_unord] * nobs**(-2./ (n_cvars + do))

        if self.return_only_bw:
            bw = np.median(only_bw, axis=0)

        return bw

    def _set_defaults(self, defaults):
        """Sets the default values for the efficient estimation"""
        self.n_res = defaults.n_res
        self.n_sub = defaults.n_sub
        self.randomize = defaults.randomize
        self.return_median = defaults.return_median
        self.efficient = defaults.efficient
        self.return_only_bw = defaults.return_only_bw
        self.n_jobs = defaults.n_jobs

    def _normal_reference(self):
        """
        Returns Scott's normal reference rule of thumb bandwidth parameter.

        Notes
        -----
        See p.13 in [2] for an example and discussion.  The formula for the
        bandwidth is

        .. math:: h = 1.06n^{-1/(4+q)}

        where ``n`` is the number of observations and ``q`` is the number of
        variables.
        """
        X = np.std(self.data, axis=0)
        return 1.06 * X * self.nobs ** (- 1. / (4 + self.data.shape[1]))

    def _set_bw_bounds(self, bw):
        """
        Sets bandwidth lower bound to effectively zero )1e-10), and for
        discrete values upper bound to 1.
        """
        bw[bw < 0] = 1e-10
        _, ix_ord, ix_unord = _get_type_pos(self.data_type)
        bw[ix_ord] = np.minimum(bw[ix_ord], 1.)
        bw[ix_unord] = np.minimum(bw[ix_unord], 1.)

        return bw

    def _cv_ml(self):
        r"""
        Returns the cross validation maximum likelihood bandwidth parameter.

        Notes
        -----
        For more details see p.16, 18, 27 in Ref. [1] (see module docstring).

        Returns the bandwidth estimate that maximizes the leave-out-out
        likelihood.  The leave-one-out log likelihood function is:

        .. math:: \ln L=\sum_{i=1}^{n}\ln f_{-i}(X_{i})

        The leave-one-out kernel estimator of :math:`f_{-i}` is:

        .. math:: f_{-i}(X_{i})=\frac{1}{(n-1)h}
                        \sum_{j=1,j\neq i}K_{h}(X_{i},X_{j})

        where :math:`K_{h}` represents the Generalized product kernel
        estimator:

        .. math:: K_{h}(X_{i},X_{j})=\prod_{s=1}^
                        {q}h_{s}^{-1}k\left(\frac{X_{is}-X_{js}}{h_{s}}\right)
        """
        # the initial value for the optimization is the normal_reference
        h0 = self._normal_reference()
        bw = optimize.fmin(self.loo_likelihood, x0=h0, args=(np.log, ),
                           maxiter=1e3, maxfun=1e3, disp=0, xtol=1e-3)
        bw = self._set_bw_bounds(bw)  # bound bw if necessary
        return bw

    def _cv_ls(self):
        r"""
        Returns the cross-validation least squares bandwidth parameter(s).

        Notes
        -----
        For more details see pp. 16, 27 in Ref. [1] (see module docstring).

        Returns the value of the bandwidth that maximizes the integrated mean
        square error between the estimated and actual distribution.  The
        integrated mean square error (IMSE) is given by:

        .. math:: \int\left[\hat{f}(x)-f(x)\right]^{2}dx

        This is the general formula for the IMSE.  The IMSE differs for
        conditional (``KDEMultivariateConditional``) and unconditional
        (``KDEMultivariate``) kernel density estimation.
        """
        h0 = self._normal_reference()
        bw = optimize.fmin(self.imse, x0=h0, maxiter=1e3, maxfun=1e3, disp=0,
                           xtol=1e-3)
        bw = self._set_bw_bounds(bw)  # bound bw if necessary
        return bw

    def loo_likelihood(self):
        raise NotImplementedError


class EstimatorSettings:
    """
    Object to specify settings for density estimation or regression.

    `EstimatorSettings` has several properties related to how bandwidth
    estimation for the `KDEMultivariate`, `KDEMultivariateConditional`,
    `KernelReg` and `CensoredKernelReg` classes behaves.

    Parameters
    ----------
    efficient : bool, optional
        If True, the bandwidth estimation is to be performed
        efficiently -- by taking smaller sub-samples and estimating
        the scaling factor of each subsample.  This is useful for large
        samples (nobs >> 300) and/or multiple variables (k_vars > 3).
        If False (default), all data is used at the same time.
    randomize : bool, optional
        If True, the bandwidth estimation is to be performed by
        taking `n_res` random resamples (with replacement) of size `n_sub` from
        the full sample.  If set to False (default), the estimation is
        performed by slicing the full sample in sub-samples of size `n_sub` so
        that all samples are used once.
    n_sub : int, optional
        Size of the sub-samples.  Default is 50.
    n_res : int, optional
        The number of random re-samples used to estimate the bandwidth.
        Only has an effect if ``randomize == True``.  Default value is 25.
    return_median : bool, optional
        If True (default), the estimator uses the median of all scaling factors
        for each sub-sample to estimate the bandwidth of the full sample.
        If False, the estimator uses the mean.
    return_only_bw : bool, optional
        If True, the estimator is to use the bandwidth and not the
        scaling factor.  This is *not* theoretically justified.
        Should be used only for experimenting.
    n_jobs : int, optional
        The number of jobs to use for parallel estimation with
        ``joblib.Parallel``.  Default is -1, meaning ``n_cores - 1``, with
        ``n_cores`` the number of available CPU cores.
        See the `joblib documentation
        <https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html>`_ for more details.

    Examples
    --------
    >>> settings = EstimatorSettings(randomize=True, n_jobs=3)
    >>> k_dens = KDEMultivariate(data, var_type, defaults=settings)
    """
    def __init__(self, efficient=False, randomize=False, n_res=25, n_sub=50,
                 return_median=True, return_only_bw=False, n_jobs=-1):
        self.efficient = efficient
        self.randomize = randomize
        self.n_res = n_res
        self.n_sub = n_sub
        self.return_median = return_median
        self.return_only_bw = return_only_bw  # TODO: remove this?
        self.n_jobs = n_jobs


class LeaveOneOut:
    """
    Generator to give leave-one-out views on X.

    Parameters
    ----------
    X : array_like
        2-D array.

    Examples
    --------
    >>> X = np.random.normal(0, 1, [10,2])
    >>> loo = LeaveOneOut(X)
    >>> for x in loo:
    ...    print x

    Notes
    -----
    A little lighter weight than sklearn LOO. We do not need test index.
    Also passes views on X, not the index.
    """
    def __init__(self, X):
        self.X = np.asarray(X)

    def __iter__(self):
        X = self.X
        nobs, k_vars = np.shape(X)

        for i in range(nobs):
            index = np.ones(nobs, dtype=bool)
            index[i] = False
            yield X[index, :]


def _get_type_pos(var_type):
    ix_cont = np.array([c == 'c' for c in var_type])
    ix_ord = np.array([c == 'o' for c in var_type])
    ix_unord = np.array([c == 'u' for c in var_type])
    return ix_cont, ix_ord, ix_unord


def _adjust_shape(dat, k_vars):
    """ Returns an array of shape (nobs, k_vars) for use with `gpke`."""
    dat = np.asarray(dat)
    if dat.ndim > 2:
        dat = np.squeeze(dat)
    if dat.ndim == 1 and k_vars > 1:  # one obs many vars
        nobs = 1
    elif dat.ndim == 1 and k_vars == 1:  # one obs one var
        nobs = len(dat)
    else:
        if np.shape(dat)[0] == k_vars and np.shape(dat)[1] != k_vars:
            dat = dat.T

        nobs = np.shape(dat)[0]  # ndim >1 so many obs many vars

    dat = np.reshape(dat, (nobs, k_vars))
    return dat


def gpke(bw, data, data_predict, var_type, ckertype='gaussian',
         okertype='wangryzin', ukertype='aitchisonaitken', tosum=True):
    r"""
    Returns the non-normalized Generalized Product Kernel Estimator

    Parameters
    ----------
    bw : 1-D ndarray
        The user-specified bandwidth parameters.
    data : 1D or 2-D ndarray
        The training data.
    data_predict : 1-D ndarray
        The evaluation points at which the kernel estimation is performed.
    var_type : str, optional
        The variable type (continuous, ordered, unordered).
    ckertype : str, optional
        The kernel used for the continuous variables.
    okertype : str, optional
        The kernel used for the ordered discrete variables.
    ukertype : str, optional
        The kernel used for the unordered discrete variables.
    tosum : bool, optional
        Whether or not to sum the calculated array of densities.  Default is
        True.

    Returns
    -------
    dens : array_like
        The generalized product kernel density estimator.

    Notes
    -----
    The formula for the multivariate kernel estimator for the pdf is:

    .. math:: f(x)=\frac{1}{nh_{1}...h_{q}}\sum_{i=1}^
                        {n}K\left(\frac{X_{i}-x}{h}\right)

    where

    .. math:: K\left(\frac{X_{i}-x}{h}\right) =
                k\left( \frac{X_{i1}-x_{1}}{h_{1}}\right)\times
                k\left( \frac{X_{i2}-x_{2}}{h_{2}}\right)\times...\times
                k\left(\frac{X_{iq}-x_{q}}{h_{q}}\right)
    """
    kertypes = dict(c=ckertype, o=okertype, u=ukertype)
    #Kval = []
    #for ii, vtype in enumerate(var_type):
    #    func = kernel_func[kertypes[vtype]]
    #    Kval.append(func(bw[ii], data[:, ii], data_predict[ii]))

    #Kval = np.column_stack(Kval)

    Kval = np.empty(data.shape)
    for ii, vtype in enumerate(var_type):
        func = kernel_func[kertypes[vtype]]
        Kval[:, ii] = func(bw[ii], data[:, ii], data_predict[ii])

    iscontinuous = np.array([c == 'c' for c in var_type])
    dens = Kval.prod(axis=1) / np.prod(bw[iscontinuous])
    if tosum:
        return dens.sum(axis=0)
    else:
        return dens
