"""
Linear mixed effects models are regression models for dependent data.
They can be used to estimate regression relationships involving both
means and variances.

These models are also known as multilevel linear models, and
hierarchical linear models.

The MixedLM class fits linear mixed effects models to data, and
provides support for some common post-estimation tasks.  This is a
group-based implementation that is most efficient for models in which
the data can be partitioned into independent groups.  Some models with
crossed effects can be handled by specifying a model with a single
group.

The data are partitioned into disjoint groups.  The probability model
for group i is:

Y = X*beta + Z*gamma + epsilon

where

* n_i is the number of observations in group i

* Y is a n_i dimensional response vector (called endog in MixedLM)

* X is a n_i x k_fe dimensional design matrix for the fixed effects
  (called exog in MixedLM)

* beta is a k_fe-dimensional vector of fixed effects parameters
  (called fe_params in MixedLM)

* Z is a design matrix for the random effects with n_i rows (called
  exog_re in MixedLM).  The number of columns in Z can vary by group
  as discussed below.

* gamma is a random vector with mean 0.  The covariance matrix for the
  first `k_re` elements of `gamma` (called cov_re in MixedLM) is
  common to all groups.  The remaining elements of `gamma` are
  variance components as discussed in more detail below. Each group
  receives its own independent realization of gamma.

* epsilon is a n_i dimensional vector of iid normal
  errors with mean 0 and variance sigma^2; the epsilon
  values are independent both within and between groups

Y, X and Z must be entirely observed.  beta, Psi, and sigma^2 are
estimated using ML or REML estimation, and gamma and epsilon are
random so define the probability model.

The marginal mean structure is E[Y | X, Z] = X*beta.  If only the mean
structure is of interest, GEE is an alternative to using linear mixed
models.

Two types of random effects are supported.  Standard random effects
are correlated with each other in arbitrary ways.  Every group has the
same number (`k_re`) of standard random effects, with the same joint
distribution (but with independent realizations across the groups).

Variance components are uncorrelated with each other, and with the
standard random effects.  Each variance component has mean zero, and
all realizations of a given variance component have the same variance
parameter.  The number of realized variance components per variance
parameter can differ across the groups.

The primary reference for the implementation details is:

MJ Lindstrom, DM Bates (1988).  "Newton Raphson and EM algorithms for
linear mixed effects models for repeated measures data".  Journal of
the American Statistical Association. Volume 83, Issue 404, pages
1014-1022.

See also this more recent document:

http://econ.ucsb.edu/~doug/245a/Papers/Mixed%20Effects%20Implement.pdf

All the likelihood, gradient, and Hessian calculations closely follow
Lindstrom and Bates 1988, adapted to support variance components.

The following two documents are written more from the perspective of
users:

http://lme4.r-forge.r-project.org/lMMwR/lrgprt.pdf

http://lme4.r-forge.r-project.org/slides/2009-07-07-Rennes/3Longitudinal-4.pdf

Notation:

* `cov_re` is the random effects covariance matrix (referred to above
  as Psi) and `scale` is the (scalar) error variance.  For a single
  group, the marginal covariance matrix of endog given exog is scale*I
  + Z * cov_re * Z', where Z is the design matrix for the random
  effects in one group.

* `vcomp` is a vector of variance parameters.  The length of `vcomp`
  is determined by the number of keys in either the `exog_vc` argument
  to ``MixedLM``, or the `vc_formula` argument when using formulas to
  fit a model.

Notes:

1. Three different parameterizations are used in different places.
The regression slopes (usually called `fe_params`) are identical in
all three parameterizations, but the variance parameters differ.  The
parameterizations are:

* The "user parameterization" in which cov(endog) = scale*I + Z *
  cov_re * Z', as described above.  This is the main parameterization
  visible to the user.

* The "profile parameterization" in which cov(endog) = I +
  Z * cov_re1 * Z'.  This is the parameterization of the profile
  likelihood that is maximized to produce parameter estimates.
  (see Lindstrom and Bates for details).  The "user" cov_re is
  equal to the "profile" cov_re1 times the scale.

* The "square root parameterization" in which we work with the Cholesky
  factor of cov_re1 instead of cov_re directly.  This is hidden from the
  user.

All three parameterizations can be packed into a vector by
(optionally) concatenating `fe_params` together with the lower
triangle or Cholesky square root of the dependence structure, followed
by the variance parameters for the variance components.  The are
stored as square roots if (and only if) the random effects covariance
matrix is stored as its Cholesky factor.  Note that when unpacking, it
is important to either square or reflect the dependence structure
depending on which parameterization is being used.

Two score methods are implemented.  One takes the score with respect
to the elements of the random effects covariance matrix (used for
inference once the MLE is reached), and the other takes the score with
respect to the parameters of the Cholesky square root of the random
effects covariance matrix (used for optimization).

The numerical optimization uses GLS to avoid explicitly optimizing
over the fixed effects parameters.  The likelihood that is optimized
is profiled over both the scale parameter (a scalar) and the fixed
effects parameters (if any).  As a result of this profiling, it is
difficult and unnecessary to calculate the Hessian of the profiled log
likelihood function, so that calculation is not implemented here.
Therefore, optimization methods requiring the Hessian matrix such as
the Newton-Raphson algorithm cannot be used for model fitting.
"""
import warnings

import numpy as np
import pandas as pd
import patsy
from scipy import sparse
from scipy.stats.distributions import norm

from statsmodels.base._penalties import Penalty
import statsmodels.base.model as base
from statsmodels.tools import data as data_tools
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.sm_exceptions import ConvergenceWarning

_warn_cov_sing = "The random effects covariance matrix is singular."


def _dot(x, y):
    """
    Returns the dot product of the arrays, works for sparse and dense.
    """

    if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        return np.dot(x, y)
    elif sparse.issparse(x):
        return x.dot(y)
    elif sparse.issparse(y):
        return y.T.dot(x.T).T


# From numpy, adapted to work with sparse and dense arrays.
def _multi_dot_three(A, B, C):
    """
    Find best ordering for three arrays and do the multiplication.

    Doing in manually instead of using dynamic programing is
    approximately 15 times faster.
    """
    # cost1 = cost((AB)C)
    cost1 = (A.shape[0] * A.shape[1] * B.shape[1] +  # (AB)
             A.shape[0] * B.shape[1] * C.shape[1])   # (--)C
    # cost2 = cost((AB)C)
    cost2 = (B.shape[0] * B.shape[1] * C.shape[1] +  # (BC)
             A.shape[0] * A.shape[1] * C.shape[1])   # A(--)

    if cost1 < cost2:
        return _dot(_dot(A, B), C)
    else:
        return _dot(A, _dot(B, C))


def _dotsum(x, y):
    """
    Returns sum(x * y), where '*' is the pointwise product, computed
    efficiently for dense and sparse matrices.
    """

    if sparse.issparse(x):
        return x.multiply(y).sum()
    else:
        # This way usually avoids allocating a temporary.
        return np.dot(x.ravel(), y.ravel())


class VCSpec:
    """
    Define the variance component structure of a multilevel model.

    An instance of the class contains three attributes:

    - names : names[k] is the name of variance component k.

    - mats : mats[k][i] is the design matrix for group index
      i in variance component k.

    - colnames : colnames[k][i] is the list of column names for
      mats[k][i].

    The groups in colnames and mats must be in sorted order.
    """

    def __init__(self, names, colnames, mats):
        self.names = names
        self.colnames = colnames
        self.mats = mats


def _get_exog_re_names(self, exog_re):
    """
    Passes through if given a list of names. Otherwise, gets pandas names
    or creates some generic variable names as needed.
    """
    if self.k_re == 0:
        return []
    if isinstance(exog_re, pd.DataFrame):
        return exog_re.columns.tolist()
    elif isinstance(exog_re, pd.Series) and exog_re.name is not None:
        return [exog_re.name]
    elif isinstance(exog_re, list):
        return exog_re

    # Default names
    defnames = ["x_re{0:1d}".format(k + 1) for k in range(exog_re.shape[1])]
    return defnames


class MixedLMParams:
    """
    This class represents a parameter state for a mixed linear model.

    Parameters
    ----------
    k_fe : int
        The number of covariates with fixed effects.
    k_re : int
        The number of covariates with random coefficients (excluding
        variance components).
    k_vc : int
        The number of variance components parameters.

    Notes
    -----
    This object represents the parameter state for the model in which
    the scale parameter has been profiled out.
    """

    def __init__(self, k_fe, k_re, k_vc):

        self.k_fe = k_fe
        self.k_re = k_re
        self.k_re2 = k_re * (k_re + 1) // 2
        self.k_vc = k_vc
        self.k_tot = self.k_fe + self.k_re2 + self.k_vc
        self._ix = np.tril_indices(self.k_re)

    def from_packed(params, k_fe, k_re, use_sqrt, has_fe):
        """
        Create a MixedLMParams object from packed parameter vector.

        Parameters
        ----------
        params : array_like
            The mode parameters packed into a single vector.
        k_fe : int
            The number of covariates with fixed effects
        k_re : int
            The number of covariates with random effects (excluding
            variance components).
        use_sqrt : bool
            If True, the random effects covariance matrix is provided
            as its Cholesky factor, otherwise the lower triangle of
            the covariance matrix is stored.
        has_fe : bool
            If True, `params` contains fixed effects parameters.
            Otherwise, the fixed effects parameters are set to zero.

        Returns
        -------
        A MixedLMParams object.
        """
        k_re2 = int(k_re * (k_re + 1) / 2)

        # The number of covariance parameters.
        if has_fe:
            k_vc = len(params) - k_fe - k_re2
        else:
            k_vc = len(params) - k_re2

        pa = MixedLMParams(k_fe, k_re, k_vc)

        cov_re = np.zeros((k_re, k_re))
        ix = pa._ix
        if has_fe:
            pa.fe_params = params[0:k_fe]
            cov_re[ix] = params[k_fe:k_fe+k_re2]
        else:
            pa.fe_params = np.zeros(k_fe)
            cov_re[ix] = params[0:k_re2]

        if use_sqrt:
            cov_re = np.dot(cov_re, cov_re.T)
        else:
            cov_re = (cov_re + cov_re.T) - np.diag(np.diag(cov_re))

        pa.cov_re = cov_re
        if k_vc > 0:
            if use_sqrt:
                pa.vcomp = params[-k_vc:]**2
            else:
                pa.vcomp = params[-k_vc:]
        else:
            pa.vcomp = np.array([])

        return pa

    from_packed = staticmethod(from_packed)

    def from_components(fe_params=None, cov_re=None, cov_re_sqrt=None,
                        vcomp=None):
        """
        Create a MixedLMParams object from each parameter component.

        Parameters
        ----------
        fe_params : array_like
            The fixed effects parameter (a 1-dimensional array).  If
            None, there are no fixed effects.
        cov_re : array_like
            The random effects covariance matrix (a square, symmetric
            2-dimensional array).
        cov_re_sqrt : array_like
            The Cholesky (lower triangular) square root of the random
            effects covariance matrix.
        vcomp : array_like
            The variance component parameters.  If None, there are no
            variance components.

        Returns
        -------
        A MixedLMParams object.
        """

        if vcomp is None:
            vcomp = np.empty(0)
        if fe_params is None:
            fe_params = np.empty(0)
        if cov_re is None and cov_re_sqrt is None:
            cov_re = np.empty((0, 0))

        k_fe = len(fe_params)
        k_vc = len(vcomp)
        k_re = cov_re.shape[0] if cov_re is not None else cov_re_sqrt.shape[0]

        pa = MixedLMParams(k_fe, k_re, k_vc)
        pa.fe_params = fe_params
        if cov_re_sqrt is not None:
            pa.cov_re = np.dot(cov_re_sqrt, cov_re_sqrt.T)
        elif cov_re is not None:
            pa.cov_re = cov_re

        pa.vcomp = vcomp

        return pa

    from_components = staticmethod(from_components)

    def copy(self):
        """
        Returns a copy of the object.
        """
        obj = MixedLMParams(self.k_fe, self.k_re, self.k_vc)
        obj.fe_params = self.fe_params.copy()
        obj.cov_re = self.cov_re.copy()
        obj.vcomp = self.vcomp.copy()
        return obj

    def get_packed(self, use_sqrt, has_fe=False):
        """
        Return the model parameters packed into a single vector.

        Parameters
        ----------
        use_sqrt : bool
            If True, the Cholesky square root of `cov_re` is
            included in the packed result.  Otherwise the
            lower triangle of `cov_re` is included.
        has_fe : bool
            If True, the fixed effects parameters are included
            in the packed result, otherwise they are omitted.
        """

        if self.k_re > 0:
            if use_sqrt:
                try:
                    L = np.linalg.cholesky(self.cov_re)
                except np.linalg.LinAlgError:
                    L = np.diag(np.sqrt(np.diag(self.cov_re)))
                cpa = L[self._ix]
            else:
                cpa = self.cov_re[self._ix]
        else:
            cpa = np.zeros(0)

        if use_sqrt:
            vcomp = np.sqrt(self.vcomp)
        else:
            vcomp = self.vcomp

        if has_fe:
            pa = np.concatenate((self.fe_params, cpa, vcomp))
        else:
            pa = np.concatenate((cpa, vcomp))

        return pa


def _smw_solver(s, A, AtA, Qi, di):
    r"""
    Returns a solver for the linear system:

    .. math::

        (sI + ABA^\prime) y = x

    The returned function f satisfies f(x) = y as defined above.

    B and its inverse matrix are block diagonal.  The upper left block
    of :math:`B^{-1}` is Qi and its lower right block is diag(di).

    Parameters
    ----------
    s : scalar
        See above for usage
    A : ndarray
        p x q matrix, in general q << p, may be sparse.
    AtA : square ndarray
        :math:`A^\prime  A`, a q x q matrix.
    Qi : square symmetric ndarray
        The matrix `B` is q x q, where q = r + d.  `B` consists of a r
        x r diagonal block whose inverse is `Qi`, and a d x d diagonal
        block, whose inverse is diag(di).
    di : 1d array_like
        See documentation for Qi.

    Returns
    -------
    A function for solving a linear system, as documented above.

    Notes
    -----
    Uses Sherman-Morrison-Woodbury identity:
        https://en.wikipedia.org/wiki/Woodbury_matrix_identity
    """

    # Use SMW identity
    qmat = AtA / s
    m = Qi.shape[0]
    qmat[0:m, 0:m] += Qi

    if sparse.issparse(A):
        qmat[m:, m:] += sparse.diags(di)

        def solver(rhs):
            ql = A.T.dot(rhs)
            # Based on profiling, the next line can be the
            # majority of the entire run time of fitting the model.
            ql = sparse.linalg.spsolve(qmat, ql)
            if ql.ndim < rhs.ndim:
                # spsolve squeezes nx1 rhs
                ql = ql[:, None]
            ql = A.dot(ql)
            return rhs / s - ql / s**2

    else:
        d = qmat.shape[0]
        qmat.flat[m*(d+1)::d+1] += di
        qmati = np.linalg.solve(qmat, A.T)

        def solver(rhs):
            # A is tall and qmati is wide, so we want
            # A * (qmati * rhs) not (A * qmati) * rhs
            ql = np.dot(qmati, rhs)
            ql = np.dot(A, ql)
            return rhs / s - ql / s**2

    return solver


def _smw_logdet(s, A, AtA, Qi, di, B_logdet):
    r"""
    Returns the log determinant of

    .. math::

        sI + ABA^\prime

    Uses the matrix determinant lemma to accelerate the calculation.
    B is assumed to be positive definite, and s > 0, therefore the
    determinant is positive.

    Parameters
    ----------
    s : positive scalar
        See above for usage
    A : ndarray
        p x q matrix, in general q << p.
    AtA : square ndarray
        :math:`A^\prime  A`, a q x q matrix.
    Qi : square symmetric ndarray
        The matrix `B` is q x q, where q = r + d.  `B` consists of a r
        x r diagonal block whose inverse is `Qi`, and a d x d diagonal
        block, whose inverse is diag(di).
    di : 1d array_like
        See documentation for Qi.
    B_logdet : real
        The log determinant of B

    Returns
    -------
    The log determinant of s*I + A*B*A'.

    Notes
    -----
    Uses the matrix determinant lemma:
        https://en.wikipedia.org/wiki/Matrix_determinant_lemma
    """

    p = A.shape[0]
    ld = p * np.log(s)
    qmat = AtA / s
    m = Qi.shape[0]
    qmat[0:m, 0:m] += Qi

    if sparse.issparse(qmat):
        qmat[m:, m:] += sparse.diags(di)

        # There are faster but much more difficult ways to do this
        # https://stackoverflow.com/questions/19107617
        lu = sparse.linalg.splu(qmat)
        dl = lu.L.diagonal().astype(np.complex128)
        du = lu.U.diagonal().astype(np.complex128)
        ld1 = np.log(dl).sum() + np.log(du).sum()
        ld1 = ld1.real
    else:
        d = qmat.shape[0]
        qmat.flat[m*(d+1)::d+1] += di
        _, ld1 = np.linalg.slogdet(qmat)

    return B_logdet + ld + ld1


def _convert_vc(exog_vc):

    vc_names = []
    vc_colnames = []
    vc_mats = []

    # Get the groups in sorted order
    groups = set()
    for k, v in exog_vc.items():
        groups |= set(v.keys())
    groups = list(groups)
    groups.sort()

    for k, v in exog_vc.items():
        vc_names.append(k)
        colnames, mats = [], []
        for g in groups:
            try:
                colnames.append(v[g].columns)
            except AttributeError:
                colnames.append([str(j) for j in range(v[g].shape[1])])
            mats.append(v[g])
        vc_colnames.append(colnames)
        vc_mats.append(mats)

    ii = np.argsort(vc_names)
    vc_names = [vc_names[i] for i in ii]
    vc_colnames = [vc_colnames[i] for i in ii]
    vc_mats = [vc_mats[i] for i in ii]

    return VCSpec(vc_names, vc_colnames, vc_mats)


class MixedLM(base.LikelihoodModel):
    """
    Linear Mixed Effects Model

    Parameters
    ----------
    endog : 1d array_like
        The dependent variable
    exog : 2d array_like
        A matrix of covariates used to determine the
        mean structure (the "fixed effects" covariates).
    groups : 1d array_like
        A vector of labels determining the groups -- data from
        different groups are independent
    exog_re : 2d array_like
        A matrix of covariates used to determine the variance and
        covariance structure (the "random effects" covariates).  If
        None, defaults to a random intercept for each group.
    exog_vc : VCSpec instance or dict-like (deprecated)
        A VCSPec instance defines the structure of the variance
        components in the model.  Alternatively, see notes below
        for a dictionary-based format.  The dictionary format is
        deprecated and may be removed at some point in the future.
    use_sqrt : bool
        If True, optimization is carried out using the lower
        triangle of the square root of the random effects
        covariance matrix, otherwise it is carried out using the
        lower triangle of the random effects covariance matrix.
    missing : str
        The approach to missing data handling

    Notes
    -----
    If `exog_vc` is not a `VCSpec` instance, then it must be a
    dictionary of dictionaries.  Specifically, `exog_vc[a][g]` is a
    matrix whose columns are linearly combined using independent
    random coefficients.  This random term then contributes to the
    variance structure of the data for group `g`.  The random
    coefficients all have mean zero, and have the same variance.  The
    matrix must be `m x k`, where `m` is the number of observations in
    group `g`.  The number of columns may differ among the top-level
    groups.

    The covariates in `exog`, `exog_re` and `exog_vc` may (but need
    not) partially or wholly overlap.

    `use_sqrt` should almost always be set to True.  The main use case
    for use_sqrt=False is when complicated patterns of fixed values in
    the covariance structure are set (using the `free` argument to
    `fit`) that cannot be expressed in terms of the Cholesky factor L.

    Examples
    --------
    A basic mixed model with fixed effects for the columns of
    ``exog`` and a random intercept for each distinct value of
    ``group``:

    >>> model = sm.MixedLM(endog, exog, groups)
    >>> result = model.fit()

    A mixed model with fixed effects for the columns of ``exog`` and
    correlated random coefficients for the columns of ``exog_re``:

    >>> model = sm.MixedLM(endog, exog, groups, exog_re=exog_re)
    >>> result = model.fit()

    A mixed model with fixed effects for the columns of ``exog`` and
    independent random coefficients for the columns of ``exog_re``:

    >>> free = MixedLMParams.from_components(
                     fe_params=np.ones(exog.shape[1]),
                     cov_re=np.eye(exog_re.shape[1]))
    >>> model = sm.MixedLM(endog, exog, groups, exog_re=exog_re)
    >>> result = model.fit(free=free)

    A different way to specify independent random coefficients for the
    columns of ``exog_re``.  In this example ``groups`` must be a
    Pandas Series with compatible indexing with ``exog_re``, and
    ``exog_re`` has two columns.

    >>> g = pd.groupby(groups, by=groups).groups
    >>> vc = {}
    >>> vc['1'] = {k : exog_re.loc[g[k], 0] for k in g}
    >>> vc['2'] = {k : exog_re.loc[g[k], 1] for k in g}
    >>> model = sm.MixedLM(endog, exog, groups, vcomp=vc)
    >>> result = model.fit()
    """

    def __init__(self, endog, exog, groups, exog_re=None,
                 exog_vc=None, use_sqrt=True, missing='none',
                 **kwargs):

        _allowed_kwargs = ["missing_idx", "design_info", "formula"]
        for x in kwargs.keys():
            if x not in _allowed_kwargs:
                raise ValueError(
                    "argument %s not permitted for MixedLM initialization" % x)

        self.use_sqrt = use_sqrt

        # Some defaults
        self.reml = True
        self.fe_pen = None
        self.re_pen = None

        if isinstance(exog_vc, dict):
            warnings.warn("Using deprecated variance components format")
            # Convert from old to new representation
            exog_vc = _convert_vc(exog_vc)

        if exog_vc is not None:
            self.k_vc = len(exog_vc.names)
            self.exog_vc = exog_vc
        else:
            self.k_vc = 0
            self.exog_vc = VCSpec([], [], [])

        # If there is one covariate, it may be passed in as a column
        # vector, convert these to 2d arrays.
        # TODO: Can this be moved up in the class hierarchy?
        #       yes, it should be done up the hierarchy
        if (exog is not None and
                data_tools._is_using_ndarray_type(exog, None) and
                exog.ndim == 1):
            exog = exog[:, None]
        if (exog_re is not None and
                data_tools._is_using_ndarray_type(exog_re, None) and
                exog_re.ndim == 1):
            exog_re = exog_re[:, None]

        # Calling super creates self.endog, etc. as ndarrays and the
        # original exog, endog, etc. are self.data.endog, etc.
        super(MixedLM, self).__init__(endog, exog, groups=groups,
                                      exog_re=exog_re, missing=missing,
                                      **kwargs)

        self._init_keys.extend(["use_sqrt", "exog_vc"])

        # Number of fixed effects parameters
        self.k_fe = exog.shape[1]

        if exog_re is None and len(self.exog_vc.names) == 0:
            # Default random effects structure (random intercepts).
            self.k_re = 1
            self.k_re2 = 1
            self.exog_re = np.ones((len(endog), 1), dtype=np.float64)
            self.data.exog_re = self.exog_re
            names = ['Group Var']
            self.data.param_names = self.exog_names + names
            self.data.exog_re_names = names
            self.data.exog_re_names_full = names

        elif exog_re is not None:
            # Process exog_re the same way that exog is handled
            # upstream
            # TODO: this is wrong and should be handled upstream wholly
            self.data.exog_re = exog_re
            self.exog_re = np.asarray(exog_re)
            if self.exog_re.ndim == 1:
                self.exog_re = self.exog_re[:, None]
            # Model dimensions
            # Number of random effect covariates
            self.k_re = self.exog_re.shape[1]
            # Number of covariance parameters
            self.k_re2 = self.k_re * (self.k_re + 1) // 2

        else:
            # All random effects are variance components
            self.k_re = 0
            self.k_re2 = 0

        if not self.data._param_names:
            # HACK: could have been set in from_formula already
            # needs refactor
            (param_names, exog_re_names,
             exog_re_names_full) = self._make_param_names(exog_re)
            self.data.param_names = param_names
            self.data.exog_re_names = exog_re_names
            self.data.exog_re_names_full = exog_re_names_full

        self.k_params = self.k_fe + self.k_re2

        # Convert the data to the internal representation, which is a
        # list of arrays, corresponding to the groups.
        group_labels = list(set(groups))
        group_labels.sort()
        row_indices = dict((s, []) for s in group_labels)
        for i, g in enumerate(groups):
            row_indices[g].append(i)
        self.row_indices = row_indices
        self.group_labels = group_labels
        self.n_groups = len(self.group_labels)

        # Split the data by groups
        self.endog_li = self.group_list(self.endog)
        self.exog_li = self.group_list(self.exog)
        self.exog_re_li = self.group_list(self.exog_re)

        # Precompute this.
        if self.exog_re is None:
            self.exog_re2_li = None
        else:
            self.exog_re2_li = [np.dot(x.T, x) for x in self.exog_re_li]

        # The total number of observations, summed over all groups
        self.nobs = len(self.endog)
        self.n_totobs = self.nobs

        # Set the fixed effects parameter names
        if self.exog_names is None:
            self.exog_names = ["FE%d" % (k + 1) for k in
                               range(self.exog.shape[1])]

        # Precompute this
        self._aex_r = []
        self._aex_r2 = []
        for i in range(self.n_groups):
            a = self._augment_exog(i)
            self._aex_r.append(a)

            ma = _dot(a.T, a)
            self._aex_r2.append(ma)

        # Precompute this
        self._lin, self._quad = self._reparam()

    def _make_param_names(self, exog_re):
        """
        Returns the full parameter names list, just the exogenous random
        effects variables, and the exogenous random effects variables with
        the interaction terms.
        """
        exog_names = list(self.exog_names)
        exog_re_names = _get_exog_re_names(self, exog_re)
        param_names = []

        jj = self.k_fe
        for i in range(len(exog_re_names)):
            for j in range(i + 1):
                if i == j:
                    param_names.append(exog_re_names[i] + " Var")
                else:
                    param_names.append(exog_re_names[j] + " x " +
                                       exog_re_names[i] + " Cov")
                jj += 1

        vc_names = [x + " Var" for x in self.exog_vc.names]

        return exog_names + param_names + vc_names, exog_re_names, param_names

    @classmethod
    def from_formula(cls, formula, data, re_formula=None, vc_formula=None,
                     subset=None, use_sparse=False, missing='none', *args,
                     **kwargs):
        """
        Create a Model from a formula and dataframe.

        Parameters
        ----------
        formula : str or generic Formula object
            The formula specifying the model
        data : array_like
            The data for the model. See Notes.
        re_formula : str
            A one-sided formula defining the variance structure of the
            model.  The default gives a random intercept for each
            group.
        vc_formula : dict-like
            Formulas describing variance components.  `vc_formula[vc]` is
            the formula for the component with variance parameter named
            `vc`.  The formula is processed into a matrix, and the columns
            of this matrix are linearly combined with independent random
            coefficients having mean zero and a common variance.
        subset : array_like
            An array-like object of booleans, integers, or index
            values that indicate the subset of df to use in the
            model. Assumes df is a `pandas.DataFrame`
        missing : str
            Either 'none' or 'drop'
        args : extra arguments
            These are passed to the model
        kwargs : extra keyword arguments
            These are passed to the model with one exception. The
            ``eval_env`` keyword is passed to patsy. It can be either a
            :class:`patsy:patsy.EvalEnvironment` object or an integer
            indicating the depth of the namespace to use. For example, the
            default ``eval_env=0`` uses the calling namespace. If you wish
            to use a "clean" environment set ``eval_env=-1``.

        Returns
        -------
        model : Model instance

        Notes
        -----
        `data` must define __getitem__ with the keys in the formula
        terms args and kwargs are passed on to the model
        instantiation. E.g., a numpy structured or rec array, a
        dictionary, or a pandas DataFrame.

        If the variance component is intended to produce random
        intercepts for disjoint subsets of a group, specified by
        string labels or a categorical data value, always use '0 +' in
        the formula so that no overall intercept is included.

        If the variance components specify random slopes and you do
        not also want a random group-level intercept in the model,
        then use '0 +' in the formula to exclude the intercept.

        The variance components formulas are processed separately for
        each group.  If a variable is categorical the results will not
        be affected by whether the group labels are distinct or
        re-used over the top-level groups.

        Examples
        --------
        Suppose we have data from an educational study with students
        nested in classrooms nested in schools.  The students take a
        test, and we want to relate the test scores to the students'
        ages, while accounting for the effects of classrooms and
        schools.  The school will be the top-level group, and the
        classroom is a nested group that is specified as a variance
        component.  Note that the schools may have different number of
        classrooms, and the classroom labels may (but need not be)
        different across the schools.

        >>> vc = {'classroom': '0 + C(classroom)'}
        >>> MixedLM.from_formula('test_score ~ age', vc_formula=vc, \
                                  re_formula='1', groups='school', data=data)

        Now suppose we also have a previous test score called
        'pretest'.  If we want the relationship between pretest
        scores and the current test to vary by classroom, we can
        specify a random slope for the pretest score

        >>> vc = {'classroom': '0 + C(classroom)', 'pretest': '0 + pretest'}
        >>> MixedLM.from_formula('test_score ~ age + pretest', vc_formula=vc, \
                                  re_formula='1', groups='school', data=data)

        The following model is almost equivalent to the previous one,
        but here the classroom random intercept and pretest slope may
        be correlated.

        >>> vc = {'classroom': '0 + C(classroom)'}
        >>> MixedLM.from_formula('test_score ~ age + pretest', vc_formula=vc, \
                                  re_formula='1 + pretest', groups='school', \
                                  data=data)
        """

        if "groups" not in kwargs.keys():
            raise AttributeError("'groups' is a required keyword argument " +
                                 "in MixedLM.from_formula")
        groups = kwargs["groups"]

        # If `groups` is a variable name, retrieve the data for the
        # groups variable.
        group_name = "Group"
        if isinstance(groups, str):
            group_name = groups
            groups = np.asarray(data[groups])
        else:
            groups = np.asarray(groups)
        del kwargs["groups"]

        # Bypass all upstream missing data handling to properly handle
        # variance components
        if missing == 'drop':
            data, groups = _handle_missing(data, groups, formula, re_formula,
                                           vc_formula)
            missing = 'none'

        if re_formula is not None:
            if re_formula.strip() == "1":
                # Work around Patsy bug, fixed by 0.3.
                exog_re = np.ones((data.shape[0], 1))
                exog_re_names = [group_name]
            else:
                eval_env = kwargs.get('eval_env', None)
                if eval_env is None:
                    eval_env = 1
                elif eval_env == -1:
                    from patsy import EvalEnvironment
                    eval_env = EvalEnvironment({})
                exog_re = patsy.dmatrix(re_formula, data, eval_env=eval_env)
                exog_re_names = exog_re.design_info.column_names
                exog_re_names = [x.replace("Intercept", group_name)
                                 for x in exog_re_names]
                exog_re = np.asarray(exog_re)
            if exog_re.ndim == 1:
                exog_re = exog_re[:, None]
        else:
            exog_re = None
            if vc_formula is None:
                exog_re_names = [group_name]
            else:
                exog_re_names = []

        if vc_formula is not None:
            eval_env = kwargs.get('eval_env', None)
            if eval_env is None:
                eval_env = 1
            elif eval_env == -1:
                from patsy import EvalEnvironment
                eval_env = EvalEnvironment({})

            vc_mats = []
            vc_colnames = []
            vc_names = []
            gb = data.groupby(groups)
            kylist = sorted(gb.groups.keys())
            vcf = sorted(vc_formula.keys())
            for vc_name in vcf:
                md = patsy.ModelDesc.from_formula(vc_formula[vc_name])
                vc_names.append(vc_name)
                evc_mats, evc_colnames = [], []
                for group_ix, group in enumerate(kylist):
                    ii = gb.groups[group]
                    mat = patsy.dmatrix(
                             md,
                             data.loc[ii, :],
                             eval_env=eval_env,
                             return_type='dataframe')
                    evc_colnames.append(mat.columns.tolist())
                    if use_sparse:
                        evc_mats.append(sparse.csr_matrix(mat))
                    else:
                        evc_mats.append(np.asarray(mat))
                vc_mats.append(evc_mats)
                vc_colnames.append(evc_colnames)
            exog_vc = VCSpec(vc_names, vc_colnames, vc_mats)
        else:
            exog_vc = VCSpec([], [], [])

        kwargs["subset"] = None
        kwargs["exog_re"] = exog_re
        kwargs["exog_vc"] = exog_vc
        kwargs["groups"] = groups
        mod = super(MixedLM, cls).from_formula(
            formula, data, *args, **kwargs)

        # expand re names to account for pairs of RE
        (param_names,
         exog_re_names,
         exog_re_names_full) = mod._make_param_names(exog_re_names)

        mod.data.param_names = param_names
        mod.data.exog_re_names = exog_re_names
        mod.data.exog_re_names_full = exog_re_names_full

        if vc_formula is not None:
            mod.data.vcomp_names = mod.exog_vc.names

        return mod

    def predict(self, params, exog=None):
        """
        Return predicted values from a design matrix.

        Parameters
        ----------
        params : array_like
            Parameters of a mixed linear model.  Can be either a
            MixedLMParams instance, or a vector containing the packed
            model parameters in which the fixed effects parameters are
            at the beginning of the vector, or a vector containing
            only the fixed effects parameters.
        exog : array_like, optional
            Design / exogenous data for the fixed effects. Model exog
            is used if None.

        Returns
        -------
        An array of fitted values.  Note that these predicted values
        only reflect the fixed effects mean structure of the model.
        """
        if exog is None:
            exog = self.exog

        if isinstance(params, MixedLMParams):
            params = params.fe_params
        else:
            params = params[0:self.k_fe]

        return np.dot(exog, params)

    def group_list(self, array):
        """
        Returns `array` split into subarrays corresponding to the
        grouping structure.
        """

        if array is None:
            return None

        if array.ndim == 1:
            return [np.array(array[self.row_indices[k]])
                    for k in self.group_labels]
        else:
            return [np.array(array[self.row_indices[k], :])
                    for k in self.group_labels]

    def fit_regularized(self, start_params=None, method='l1', alpha=0,
                        ceps=1e-4, ptol=1e-6, maxit=200, **fit_kwargs):
        """
        Fit a model in which the fixed effects parameters are
        penalized.  The dependence parameters are held fixed at their
        estimated values in the unpenalized model.

        Parameters
        ----------
        method : str of Penalty object
            Method for regularization.  If a string, must be 'l1'.
        alpha : array_like
            Scalar or vector of penalty weights.  If a scalar, the
            same weight is applied to all coefficients; if a vector,
            it contains a weight for each coefficient.  If method is a
            Penalty object, the weights are scaled by alpha.  For L1
            regularization, the weights are used directly.
        ceps : positive real scalar
            Fixed effects parameters smaller than this value
            in magnitude are treated as being zero.
        ptol : positive real scalar
            Convergence occurs when the sup norm difference
            between successive values of `fe_params` is less than
            `ptol`.
        maxit : int
            The maximum number of iterations.
        **fit_kwargs
            Additional keyword arguments passed to fit.

        Returns
        -------
        A MixedLMResults instance containing the results.

        Notes
        -----
        The covariance structure is not updated as the fixed effects
        parameters are varied.

        The algorithm used here for L1 regularization is a"shooting"
        or cyclic coordinate descent algorithm.

        If method is 'l1', then `fe_pen` and `cov_pen` are used to
        obtain the covariance structure, but are ignored during the
        L1-penalized fitting.

        References
        ----------
        Friedman, J. H., Hastie, T. and Tibshirani, R. Regularized
        Paths for Generalized Linear Models via Coordinate
        Descent. Journal of Statistical Software, 33(1) (2008)
        http://www.jstatsoft.org/v33/i01/paper

        http://statweb.stanford.edu/~tibs/stat315a/Supplements/fuse.pdf
        """

        if isinstance(method, str) and (method.lower() != 'l1'):
            raise ValueError("Invalid regularization method")

        # If method is a smooth penalty just optimize directly.
        if isinstance(method, Penalty):
            # Scale the penalty weights by alpha
            method.alpha = alpha
            fit_kwargs.update({"fe_pen": method})
            return self.fit(**fit_kwargs)

        if np.isscalar(alpha):
            alpha = alpha * np.ones(self.k_fe, dtype=np.float64)

        # Fit the unpenalized model to get the dependence structure.
        mdf = self.fit(**fit_kwargs)
        fe_params = mdf.fe_params
        cov_re = mdf.cov_re
        vcomp = mdf.vcomp
        scale = mdf.scale
        try:
            cov_re_inv = np.linalg.inv(cov_re)
        except np.linalg.LinAlgError:
            cov_re_inv = None

        for itr in range(maxit):

            fe_params_s = fe_params.copy()
            for j in range(self.k_fe):

                if abs(fe_params[j]) < ceps:
                    continue

                # The residuals
                fe_params[j] = 0.
                expval = np.dot(self.exog, fe_params)
                resid_all = self.endog - expval

                # The loss function has the form
                # a*x^2 + b*x + pwt*|x|
                a, b = 0., 0.
                for group_ix, group in enumerate(self.group_labels):

                    vc_var = self._expand_vcomp(vcomp, group_ix)

                    exog = self.exog_li[group_ix]
                    ex_r, ex2_r = self._aex_r[group_ix], self._aex_r2[group_ix]

                    resid = resid_all[self.row_indices[group]]
                    solver = _smw_solver(scale, ex_r, ex2_r, cov_re_inv,
                                         1 / vc_var)

                    x = exog[:, j]
                    u = solver(x)
                    a += np.dot(u, x)
                    b -= 2 * np.dot(u, resid)

                pwt1 = alpha[j]
                if b > pwt1:
                    fe_params[j] = -(b - pwt1) / (2 * a)
                elif b < -pwt1:
                    fe_params[j] = -(b + pwt1) / (2 * a)

            if np.abs(fe_params_s - fe_params).max() < ptol:
                break

        # Replace the fixed effects estimates with their penalized
        # values, leave the dependence parameters in their unpenalized
        # state.
        params_prof = mdf.params.copy()
        params_prof[0:self.k_fe] = fe_params

        scale = self.get_scale(fe_params, mdf.cov_re_unscaled, mdf.vcomp)

        # Get the Hessian including only the nonzero fixed effects,
        # then blow back up to the full size after inverting.
        hess, sing = self.hessian(params_prof)
        if sing:
            warnings.warn(_warn_cov_sing)

        pcov = np.nan * np.ones_like(hess)
        ii = np.abs(params_prof) > ceps
        ii[self.k_fe:] = True
        ii = np.flatnonzero(ii)
        hess1 = hess[ii, :][:, ii]
        pcov[np.ix_(ii, ii)] = np.linalg.inv(-hess1)

        params_object = MixedLMParams.from_components(fe_params, cov_re=cov_re)

        results = MixedLMResults(self, params_prof, pcov / scale)
        results.params_object = params_object
        results.fe_params = fe_params
        results.cov_re = cov_re
        results.vcomp = vcomp
        results.scale = scale
        results.cov_re_unscaled = mdf.cov_re_unscaled
        results.method = mdf.method
        results.converged = True
        results.cov_pen = self.cov_pen
        results.k_fe = self.k_fe
        results.k_re = self.k_re
        results.k_re2 = self.k_re2
        results.k_vc = self.k_vc

        return MixedLMResultsWrapper(results)

    def get_fe_params(self, cov_re, vcomp, tol=1e-10):
        """
        Use GLS to update the fixed effects parameter estimates.

        Parameters
        ----------
        cov_re : array_like (2d)
            The covariance matrix of the random effects.
        vcomp : array_like (1d)
            The variance components.
        tol : float
            A tolerance parameter to determine when covariances
            are singular.

        Returns
        -------
        params : ndarray
            The GLS estimates of the fixed effects parameters.
        singular : bool
            True if the covariance is singular
        """

        if self.k_fe == 0:
            return np.array([]), False

        sing = False

        if self.k_re == 0:
            cov_re_inv = np.empty((0, 0))
        else:
            w, v = np.linalg.eigh(cov_re)
            if w.min() < tol:
                # Singular, use pseudo-inverse
                sing = True
                ii = np.flatnonzero(w >= tol)
                if len(ii) == 0:
                    cov_re_inv = np.zeros_like(cov_re)
                else:
                    vi = v[:, ii]
                    wi = w[ii]
                    cov_re_inv = np.dot(vi / wi, vi.T)
            else:
                cov_re_inv = np.linalg.inv(cov_re)

        # Cache these quantities that do not change.
        if not hasattr(self, "_endex_li"):
            self._endex_li = []
            for group_ix, _ in enumerate(self.group_labels):
                mat = np.concatenate(
                    (self.exog_li[group_ix],
                     self.endog_li[group_ix][:, None]), axis=1)
                self._endex_li.append(mat)

        xtxy = 0.
        for group_ix, group in enumerate(self.group_labels):
            vc_var = self._expand_vcomp(vcomp, group_ix)
            if vc_var.size > 0:
                if vc_var.min() < tol:
                    # Pseudo-inverse
                    sing = True
                    ii = np.flatnonzero(vc_var >= tol)
                    vc_vari = np.zeros_like(vc_var)
                    vc_vari[ii] = 1 / vc_var[ii]
                else:
                    vc_vari = 1 / vc_var
            else:
                vc_vari = np.empty(0)
            exog = self.exog_li[group_ix]
            ex_r, ex2_r = self._aex_r[group_ix], self._aex_r2[group_ix]
            solver = _smw_solver(1., ex_r, ex2_r, cov_re_inv, vc_vari)
            u = solver(self._endex_li[group_ix])
            xtxy += np.dot(exog.T, u)

        if sing:
            fe_params = np.dot(np.linalg.pinv(xtxy[:, 0:-1]), xtxy[:, -1])
        else:
            fe_params = np.linalg.solve(xtxy[:, 0:-1], xtxy[:, -1])

        return fe_params, sing

    def _reparam(self):
        """
        Returns parameters of the map converting parameters from the
        form used in optimization to the form returned to the user.

        Returns
        -------
        lin : list-like
            Linear terms of the map
        quad : list-like
            Quadratic terms of the map

        Notes
        -----
        If P are the standard form parameters and R are the
        transformed parameters (i.e. with the Cholesky square root
        covariance and square root transformed variance components),
        then P[i] = lin[i] * R + R' * quad[i] * R
        """

        k_fe, k_re, k_re2, k_vc = self.k_fe, self.k_re, self.k_re2, self.k_vc
        k_tot = k_fe + k_re2 + k_vc
        ix = np.tril_indices(self.k_re)

        lin = []
        for k in range(k_fe):
            e = np.zeros(k_tot)
            e[k] = 1
            lin.append(e)
        for k in range(k_re2):
            lin.append(np.zeros(k_tot))
        for k in range(k_vc):
            lin.append(np.zeros(k_tot))

        quad = []
        # Quadratic terms for fixed effects.
        for k in range(k_tot):
            quad.append(np.zeros((k_tot, k_tot)))

        # Quadratic terms for random effects covariance.
        ii = np.tril_indices(k_re)
        ix = [(a, b) for a, b in zip(ii[0], ii[1])]
        for i1 in range(k_re2):
            for i2 in range(k_re2):
                ix1 = ix[i1]
                ix2 = ix[i2]
                if (ix1[1] == ix2[1]) and (ix1[0] <= ix2[0]):
                    ii = (ix2[0], ix1[0])
                    k = ix.index(ii)
                    quad[k_fe+k][k_fe+i2, k_fe+i1] += 1
        for k in range(k_tot):
            quad[k] = 0.5*(quad[k] + quad[k].T)

        # Quadratic terms for variance components.
        km = k_fe + k_re2
        for k in range(km, km+k_vc):
            quad[k][k, k] = 1

        return lin, quad

    def _expand_vcomp(self, vcomp, group_ix):
        """
        Replicate variance parameters to match a group's design.

        Parameters
        ----------
        vcomp : array_like
            The variance parameters for the variance components.
        group_ix : int
            The group index

        Returns an expanded version of vcomp, in which each variance
        parameter is copied as many times as there are independent
        realizations of the variance component in the given group.
        """
        if len(vcomp) == 0:
            return np.empty(0)
        vc_var = []
        for j in range(len(self.exog_vc.names)):
            d = self.exog_vc.mats[j][group_ix].shape[1]
            vc_var.append(vcomp[j] * np.ones(d))
        if len(vc_var) > 0:
            return np.concatenate(vc_var)
        else:
            # Cannot reach here?
            return np.empty(0)

    def _augment_exog(self, group_ix):
        """
        Concatenate the columns for variance components to the columns
        for other random effects to obtain a single random effects
        exog matrix for a given group.
        """
        ex_r = self.exog_re_li[group_ix] if self.k_re > 0 else None
        if self.k_vc == 0:
            return ex_r

        ex = [ex_r] if self.k_re > 0 else []
        any_sparse = False
        for j, _ in enumerate(self.exog_vc.names):
            ex.append(self.exog_vc.mats[j][group_ix])
            any_sparse |= sparse.issparse(ex[-1])
        if any_sparse:
            for j, x in enumerate(ex):
                if not sparse.issparse(x):
                    ex[j] = sparse.csr_matrix(x)
            ex = sparse.hstack(ex)
            ex = sparse.csr_matrix(ex)
        else:
            ex = np.concatenate(ex, axis=1)

        return ex

    def loglike(self, params, profile_fe=True):
        """
        Evaluate the (profile) log-likelihood of the linear mixed
        effects model.

        Parameters
        ----------
        params : MixedLMParams, or array_like.
            The parameter value.  If array-like, must be a packed
            parameter vector containing only the covariance
            parameters.
        profile_fe : bool
            If True, replace the provided value of `fe_params` with
            the GLS estimates.

        Returns
        -------
        The log-likelihood value at `params`.

        Notes
        -----
        The scale parameter `scale` is always profiled out of the
        log-likelihood.  In addition, if `profile_fe` is true the
        fixed effects parameters are also profiled out.
        """

        if type(params) is not MixedLMParams:
            params = MixedLMParams.from_packed(params, self.k_fe,
                                               self.k_re, self.use_sqrt,
                                               has_fe=False)

        cov_re = params.cov_re
        vcomp = params.vcomp

        # Move to the profile set
        if profile_fe:
            fe_params, sing = self.get_fe_params(cov_re, vcomp)
            if sing:
                self._cov_sing += 1
        else:
            fe_params = params.fe_params

        if self.k_re > 0:
            try:
                cov_re_inv = np.linalg.inv(cov_re)
            except np.linalg.LinAlgError:
                cov_re_inv = np.linalg.pinv(cov_re)
                self._cov_sing += 1
            _, cov_re_logdet = np.linalg.slogdet(cov_re)
        else:
            cov_re_inv = np.zeros((0, 0))
            cov_re_logdet = 0

        # The residuals
        expval = np.dot(self.exog, fe_params)
        resid_all = self.endog - expval

        likeval = 0.

        # Handle the covariance penalty
        if (self.cov_pen is not None) and (self.k_re > 0):
            likeval -= self.cov_pen.func(cov_re, cov_re_inv)

        # Handle the fixed effects penalty
        if (self.fe_pen is not None):
            likeval -= self.fe_pen.func(fe_params)

        xvx, qf = 0., 0.
        for group_ix, group in enumerate(self.group_labels):

            vc_var = self._expand_vcomp(vcomp, group_ix)
            cov_aug_logdet = cov_re_logdet + np.sum(np.log(vc_var))

            exog = self.exog_li[group_ix]
            ex_r, ex2_r = self._aex_r[group_ix], self._aex_r2[group_ix]
            solver = _smw_solver(1., ex_r, ex2_r, cov_re_inv, 1 / vc_var)

            resid = resid_all[self.row_indices[group]]

            # Part 1 of the log likelihood (for both ML and REML)
            ld = _smw_logdet(1., ex_r, ex2_r, cov_re_inv, 1 / vc_var,
                             cov_aug_logdet)
            likeval -= ld / 2.

            # Part 2 of the log likelihood (for both ML and REML)
            u = solver(resid)
            qf += np.dot(resid, u)

            # Adjustment for REML
            if self.reml:
                mat = solver(exog)
                xvx += np.dot(exog.T, mat)

        if self.reml:
            likeval -= (self.n_totobs - self.k_fe) * np.log(qf) / 2.
            _, ld = np.linalg.slogdet(xvx)
            likeval -= ld / 2.
            likeval -= (self.n_totobs - self.k_fe) * np.log(2 * np.pi) / 2.
            likeval += ((self.n_totobs - self.k_fe) *
                        np.log(self.n_totobs - self.k_fe) / 2.)
            likeval -= (self.n_totobs - self.k_fe) / 2.
        else:
            likeval -= self.n_totobs * np.log(qf) / 2.
            likeval -= self.n_totobs * np.log(2 * np.pi) / 2.
            likeval += self.n_totobs * np.log(self.n_totobs) / 2.
            likeval -= self.n_totobs / 2.

        return likeval

    def _gen_dV_dPar(self, ex_r, solver, group_ix, max_ix=None):
        """
        A generator that yields the element-wise derivative of the
        marginal covariance matrix with respect to the random effects
        variance and covariance parameters.

        ex_r : array_like
            The random effects design matrix
        solver : function
            A function that given x returns V^{-1}x, where V
            is the group's marginal covariance matrix.
        group_ix : int
            The group index
        max_ix : {int, None}
            If not None, the generator ends when this index
            is reached.
        """

        axr = solver(ex_r)

        # Regular random effects
        jj = 0
        for j1 in range(self.k_re):
            for j2 in range(j1 + 1):
                if max_ix is not None and jj > max_ix:
                    return
                # Need 2d
                mat_l, mat_r = ex_r[:, j1:j1+1], ex_r[:, j2:j2+1]
                vsl, vsr = axr[:, j1:j1+1], axr[:, j2:j2+1]
                yield jj, mat_l, mat_r, vsl, vsr, j1 == j2
                jj += 1

        # Variance components
        for j, _ in enumerate(self.exog_vc.names):
            if max_ix is not None and jj > max_ix:
                return
            mat = self.exog_vc.mats[j][group_ix]
            axmat = solver(mat)
            yield jj, mat, mat, axmat, axmat, True
            jj += 1

    def score(self, params, profile_fe=True):
        """
        Returns the score vector of the profile log-likelihood.

        Notes
        -----
        The score vector that is returned is computed with respect to
        the parameterization defined by this model instance's
        `use_sqrt` attribute.
        """

        if type(params) is not MixedLMParams:
            params = MixedLMParams.from_packed(
                params, self.k_fe, self.k_re, self.use_sqrt,
                has_fe=False)

        if profile_fe:
            params.fe_params, sing = \
                self.get_fe_params(params.cov_re, params.vcomp)

            if sing:
                msg = "Random effects covariance is singular"
                warnings.warn(msg)

        if self.use_sqrt:
            score_fe, score_re, score_vc = self.score_sqrt(
                params, calc_fe=not profile_fe)
        else:
            score_fe, score_re, score_vc = self.score_full(
                params, calc_fe=not profile_fe)

        if self._freepat is not None:
            score_fe *= self._freepat.fe_params
            score_re *= self._freepat.cov_re[self._freepat._ix]
            score_vc *= self._freepat.vcomp

        if profile_fe:
            return np.concatenate((score_re, score_vc))
        else:
            return np.concatenate((score_fe, score_re, score_vc))

    def score_full(self, params, calc_fe):
        """
        Returns the score with respect to untransformed parameters.

        Calculates the score vector for the profiled log-likelihood of
        the mixed effects model with respect to the parameterization
        in which the random effects covariance matrix is represented
        in its full form (not using the Cholesky factor).

        Parameters
        ----------
        params : MixedLMParams or array_like
            The parameter at which the score function is evaluated.
            If array-like, must contain the packed random effects
            parameters (cov_re and vcomp) without fe_params.
        calc_fe : bool
            If True, calculate the score vector for the fixed effects
            parameters.  If False, this vector is not calculated, and
            a vector of zeros is returned in its place.

        Returns
        -------
        score_fe : array_like
            The score vector with respect to the fixed effects
            parameters.
        score_re : array_like
            The score vector with respect to the random effects
            parameters (excluding variance components parameters).
        score_vc : array_like
            The score vector with respect to variance components
            parameters.

        Notes
        -----
        `score_re` is taken with respect to the parameterization in
        which `cov_re` is represented through its lower triangle
        (without taking the Cholesky square root).
        """

        fe_params = params.fe_params
        cov_re = params.cov_re
        vcomp = params.vcomp

        try:
            cov_re_inv = np.linalg.inv(cov_re)
        except np.linalg.LinAlgError:
            cov_re_inv = np.linalg.pinv(cov_re)
            self._cov_sing += 1

        score_fe = np.zeros(self.k_fe)
        score_re = np.zeros(self.k_re2)
        score_vc = np.zeros(self.k_vc)

        # Handle the covariance penalty.
        if self.cov_pen is not None:
            score_re -= self.cov_pen.deriv(cov_re, cov_re_inv)

        # Handle the fixed effects penalty.
        if calc_fe and (self.fe_pen is not None):
            score_fe -= self.fe_pen.deriv(fe_params)

        # resid' V^{-1} resid, summed over the groups (a scalar)
        rvir = 0.

        # exog' V^{-1} resid, summed over the groups (a k_fe
        # dimensional vector)
        xtvir = 0.

        # exog' V^{_1} exog, summed over the groups (a k_fe x k_fe
        # matrix)
        xtvix = 0.

        # V^{-1} exog' dV/dQ_jj exog V^{-1}, where Q_jj is the jj^th
        # covariance parameter.
        xtax = [0., ] * (self.k_re2 + self.k_vc)

        # Temporary related to the gradient of log |V|
        dlv = np.zeros(self.k_re2 + self.k_vc)

        # resid' V^{-1} dV/dQ_jj V^{-1} resid (a scalar)
        rvavr = np.zeros(self.k_re2 + self.k_vc)

        for group_ix, group in enumerate(self.group_labels):

            vc_var = self._expand_vcomp(vcomp, group_ix)

            exog = self.exog_li[group_ix]
            ex_r, ex2_r = self._aex_r[group_ix], self._aex_r2[group_ix]
            solver = _smw_solver(1., ex_r, ex2_r, cov_re_inv, 1 / vc_var)

            # The residuals
            resid = self.endog_li[group_ix]
            if self.k_fe > 0:
                expval = np.dot(exog, fe_params)
                resid = resid - expval

            if self.reml:
                viexog = solver(exog)
                xtvix += np.dot(exog.T, viexog)

            # Contributions to the covariance parameter gradient
            vir = solver(resid)
            for (jj, matl, matr, vsl, vsr, sym) in\
                    self._gen_dV_dPar(ex_r, solver, group_ix):
                dlv[jj] = _dotsum(matr, vsl)
                if not sym:
                    dlv[jj] += _dotsum(matl, vsr)

                ul = _dot(vir, matl)
                ur = ul.T if sym else _dot(matr.T, vir)
                ulr = np.dot(ul, ur)
                rvavr[jj] += ulr
                if not sym:
                    rvavr[jj] += ulr.T

                if self.reml:
                    ul = _dot(viexog.T, matl)
                    ur = ul.T if sym else _dot(matr.T, viexog)
                    ulr = np.dot(ul, ur)
                    xtax[jj] += ulr
                    if not sym:
                        xtax[jj] += ulr.T

            # Contribution of log|V| to the covariance parameter
            # gradient.
            if self.k_re > 0:
                score_re -= 0.5 * dlv[0:self.k_re2]
            if self.k_vc > 0:
                score_vc -= 0.5 * dlv[self.k_re2:]

            rvir += np.dot(resid, vir)

            if calc_fe:
                xtvir += np.dot(exog.T, vir)

        fac = self.n_totobs
        if self.reml:
            fac -= self.k_fe

        if calc_fe and self.k_fe > 0:
            score_fe += fac * xtvir / rvir

        if self.k_re > 0:
            score_re += 0.5 * fac * rvavr[0:self.k_re2] / rvir
        if self.k_vc > 0:
            score_vc += 0.5 * fac * rvavr[self.k_re2:] / rvir

        if self.reml:
            xtvixi = np.linalg.inv(xtvix)
            for j in range(self.k_re2):
                score_re[j] += 0.5 * _dotsum(xtvixi.T, xtax[j])
            for j in range(self.k_vc):
                score_vc[j] += 0.5 * _dotsum(xtvixi.T, xtax[self.k_re2 + j])

        return score_fe, score_re, score_vc

    def score_sqrt(self, params, calc_fe=True):
        """
        Returns the score with respect to transformed parameters.

        Calculates the score vector with respect to the
        parameterization in which the random effects covariance matrix
        is represented through its Cholesky square root.

        Parameters
        ----------
        params : MixedLMParams or array_like
            The model parameters.  If array-like must contain packed
            parameters that are compatible with this model instance.
        calc_fe : bool
            If True, calculate the score vector for the fixed effects
            parameters.  If False, this vector is not calculated, and
            a vector of zeros is returned in its place.

        Returns
        -------
        score_fe : array_like
            The score vector with respect to the fixed effects
            parameters.
        score_re : array_like
            The score vector with respect to the random effects
            parameters (excluding variance components parameters).
        score_vc : array_like
            The score vector with respect to variance components
            parameters.
        """

        score_fe, score_re, score_vc = self.score_full(params, calc_fe=calc_fe)
        params_vec = params.get_packed(use_sqrt=True, has_fe=True)

        score_full = np.concatenate((score_fe, score_re, score_vc))
        scr = 0.
        for i in range(len(params_vec)):
            v = self._lin[i] + 2 * np.dot(self._quad[i], params_vec)
            scr += score_full[i] * v
        score_fe = scr[0:self.k_fe]
        score_re = scr[self.k_fe:self.k_fe + self.k_re2]
        score_vc = scr[self.k_fe + self.k_re2:]

        return score_fe, score_re, score_vc

    def hessian(self, params):
        """
        Returns the model's Hessian matrix.

        Calculates the Hessian matrix for the linear mixed effects
        model with respect to the parameterization in which the
        covariance matrix is represented directly (without square-root
        transformation).

        Parameters
        ----------
        params : MixedLMParams or array_like
            The model parameters at which the Hessian is calculated.
            If array-like, must contain the packed parameters in a
            form that is compatible with this model instance.

        Returns
        -------
        hess : 2d ndarray
            The Hessian matrix, evaluated at `params`.
        sing : boolean
            If True, the covariance matrix is singular and a
            pseudo-inverse is returned.
        """

        if type(params) is not MixedLMParams:
            params = MixedLMParams.from_packed(params, self.k_fe, self.k_re,
                                               use_sqrt=self.use_sqrt,
                                               has_fe=True)

        fe_params = params.fe_params
        vcomp = params.vcomp
        cov_re = params.cov_re
        sing = False

        if self.k_re > 0:
            try:
                cov_re_inv = np.linalg.inv(cov_re)
            except np.linalg.LinAlgError:
                cov_re_inv = np.linalg.pinv(cov_re)
                sing = True
        else:
            cov_re_inv = np.empty((0, 0))

        # Blocks for the fixed and random effects parameters.
        hess_fe = 0.
        hess_re = np.zeros((self.k_re2 + self.k_vc, self.k_re2 + self.k_vc))
        hess_fere = np.zeros((self.k_re2 + self.k_vc, self.k_fe))

        fac = self.n_totobs
        if self.reml:
            fac -= self.exog.shape[1]

        rvir = 0.
        xtvix = 0.
        xtax = [0., ] * (self.k_re2 + self.k_vc)
        m = self.k_re2 + self.k_vc
        B = np.zeros(m)
        D = np.zeros((m, m))
        F = [[0.] * m for k in range(m)]
        for group_ix, group in enumerate(self.group_labels):

            vc_var = self._expand_vcomp(vcomp, group_ix)
            vc_vari = np.zeros_like(vc_var)
            ii = np.flatnonzero(vc_var >= 1e-10)
            if len(ii) > 0:
                vc_vari[ii] = 1 / vc_var[ii]
            if len(ii) < len(vc_var):
                sing = True

            exog = self.exog_li[group_ix]
            ex_r, ex2_r = self._aex_r[group_ix], self._aex_r2[group_ix]
            solver = _smw_solver(1., ex_r, ex2_r, cov_re_inv, vc_vari)

            # The residuals
            resid = self.endog_li[group_ix]
            if self.k_fe > 0:
                expval = np.dot(exog, fe_params)
                resid = resid - expval

            viexog = solver(exog)
            xtvix += np.dot(exog.T, viexog)
            vir = solver(resid)
            rvir += np.dot(resid, vir)

            for (jj1, matl1, matr1, vsl1, vsr1, sym1) in\
                    self._gen_dV_dPar(ex_r, solver, group_ix):

                ul = _dot(viexog.T, matl1)
                ur = _dot(matr1.T, vir)
                hess_fere[jj1, :] += np.dot(ul, ur)
                if not sym1:
                    ul = _dot(viexog.T, matr1)
                    ur = _dot(matl1.T, vir)
                    hess_fere[jj1, :] += np.dot(ul, ur)

                if self.reml:
                    ul = _dot(viexog.T, matl1)
                    ur = ul if sym1 else np.dot(viexog.T, matr1)
                    ulr = _dot(ul, ur.T)
                    xtax[jj1] += ulr
                    if not sym1:
                        xtax[jj1] += ulr.T

                ul = _dot(vir, matl1)
                ur = ul if sym1 else _dot(vir, matr1)
                B[jj1] += np.dot(ul, ur) * (1 if sym1 else 2)

                # V^{-1} * dV/d_theta
                E = [(vsl1, matr1)]
                if not sym1:
                    E.append((vsr1, matl1))

                for (jj2, matl2, matr2, vsl2, vsr2, sym2) in\
                        self._gen_dV_dPar(ex_r, solver, group_ix, jj1):

                    re = sum([_multi_dot_three(matr2.T, x[0], x[1].T)
                              for x in E])
                    vt = 2 * _dot(_multi_dot_three(vir[None, :], matl2, re),
                                  vir[:, None])

                    if not sym2:
                        le = sum([_multi_dot_three(matl2.T, x[0], x[1].T)
                                  for x in E])
                        vt += 2 * _dot(_multi_dot_three(
                            vir[None, :], matr2, le), vir[:, None])

                    D[jj1, jj2] += np.squeeze(vt)
                    if jj1 != jj2:
                        D[jj2, jj1] += np.squeeze(vt)

                    rt = _dotsum(vsl2, re.T) / 2
                    if not sym2:
                        rt += _dotsum(vsr2, le.T) / 2

                    hess_re[jj1, jj2] += rt
                    if jj1 != jj2:
                        hess_re[jj2, jj1] += rt

                    if self.reml:
                        ev = sum([_dot(x[0], _dot(x[1].T, viexog)) for x in E])
                        u1 = _dot(viexog.T, matl2)
                        u2 = _dot(matr2.T, ev)
                        um = np.dot(u1, u2)
                        F[jj1][jj2] += um + um.T
                        if not sym2:
                            u1 = np.dot(viexog.T, matr2)
                            u2 = np.dot(matl2.T, ev)
                            um = np.dot(u1, u2)
                            F[jj1][jj2] += um + um.T

        hess_fe -= fac * xtvix / rvir
        hess_re = hess_re - 0.5 * fac * (D/rvir - np.outer(B, B) / rvir**2)
        hess_fere = -fac * hess_fere / rvir

        if self.reml:
            QL = [np.linalg.solve(xtvix, x) for x in xtax]
            for j1 in range(self.k_re2 + self.k_vc):
                for j2 in range(j1 + 1):
                    a = _dotsum(QL[j1].T, QL[j2])
                    a -= np.trace(np.linalg.solve(xtvix, F[j1][j2]))
                    a *= 0.5
                    hess_re[j1, j2] += a
                    if j1 > j2:
                        hess_re[j2, j1] += a

        # Put the blocks together to get the Hessian.
        m = self.k_fe + self.k_re2 + self.k_vc
        hess = np.zeros((m, m))
        hess[0:self.k_fe, 0:self.k_fe] = hess_fe
        hess[0:self.k_fe, self.k_fe:] = hess_fere.T
        hess[self.k_fe:, 0:self.k_fe] = hess_fere
        hess[self.k_fe:, self.k_fe:] = hess_re

        return hess, sing

    def get_scale(self, fe_params, cov_re, vcomp):
        """
        Returns the estimated error variance based on given estimates
        of the slopes and random effects covariance matrix.

        Parameters
        ----------
        fe_params : array_like
            The regression slope estimates
        cov_re : 2d array_like
            Estimate of the random effects covariance matrix
        vcomp : array_like
            Estimate of the variance components

        Returns
        -------
        scale : float
            The estimated error variance.
        """

        try:
            cov_re_inv = np.linalg.inv(cov_re)
        except np.linalg.LinAlgError:
            cov_re_inv = np.linalg.pinv(cov_re)
            warnings.warn(_warn_cov_sing)

        qf = 0.
        for group_ix, group in enumerate(self.group_labels):

            vc_var = self._expand_vcomp(vcomp, group_ix)

            exog = self.exog_li[group_ix]
            ex_r, ex2_r = self._aex_r[group_ix], self._aex_r2[group_ix]

            solver = _smw_solver(1., ex_r, ex2_r, cov_re_inv, 1 / vc_var)

            # The residuals
            resid = self.endog_li[group_ix]
            if self.k_fe > 0:
                expval = np.dot(exog, fe_params)
                resid = resid - expval

            mat = solver(resid)
            qf += np.dot(resid, mat)

        if self.reml:
            qf /= (self.n_totobs - self.k_fe)
        else:
            qf /= self.n_totobs

        return qf

    def fit(self, start_params=None, reml=True, niter_sa=0,
            do_cg=True, fe_pen=None, cov_pen=None, free=None,
            full_output=False, method=None, **fit_kwargs):
        """
        Fit a linear mixed model to the data.

        Parameters
        ----------
        start_params : array_like or MixedLMParams
            Starting values for the profile log-likelihood.  If not a
            `MixedLMParams` instance, this should be an array
            containing the packed parameters for the profile
            log-likelihood, including the fixed effects
            parameters.
        reml : bool
            If true, fit according to the REML likelihood, else
            fit the standard likelihood using ML.
        niter_sa : int
            Currently this argument is ignored and has no effect
            on the results.
        cov_pen : CovariancePenalty object
            A penalty for the random effects covariance matrix
        do_cg : bool, defaults to True
            If False, the optimization is skipped and a results
            object at the given (or default) starting values is
            returned.
        fe_pen : Penalty object
            A penalty on the fixed effects
        free : MixedLMParams object
            If not `None`, this is a mask that allows parameters to be
            held fixed at specified values.  A 1 indicates that the
            corresponding parameter is estimated, a 0 indicates that
            it is fixed at its starting value.  Setting the `cov_re`
            component to the identity matrix fits a model with
            independent random effects.  Note that some optimization
            methods do not respect this constraint (bfgs and lbfgs both
            work).
        full_output : bool
            If true, attach iteration history to results
        method : str
            Optimization method.  Can be a scipy.optimize method name,
            or a list of such names to be tried in sequence.
        **fit_kwargs
            Additional keyword arguments passed to fit.

        Returns
        -------
        A MixedLMResults instance.
        """

        _allowed_kwargs = ['gtol', 'maxiter', 'eps', 'maxcor', 'ftol',
                           'tol', 'disp', 'maxls']
        for x in fit_kwargs.keys():
            if x not in _allowed_kwargs:
                warnings.warn("Argument %s not used by MixedLM.fit" % x)

        if method is None:
            method = ['bfgs', 'lbfgs', 'cg']
        elif isinstance(method, str):
            method = [method]

        for meth in method:
            if meth.lower() in ["newton", "ncg"]:
                raise ValueError(
                    "method %s not available for MixedLM" % meth)

        self.reml = reml
        self.cov_pen = cov_pen
        self.fe_pen = fe_pen
        self._cov_sing = 0
        self._freepat = free

        if full_output:
            hist = []
        else:
            hist = None

        if start_params is None:
            params = MixedLMParams(self.k_fe, self.k_re, self.k_vc)
            params.fe_params = np.zeros(self.k_fe)
            params.cov_re = np.eye(self.k_re)
            params.vcomp = np.ones(self.k_vc)
        else:
            if isinstance(start_params, MixedLMParams):
                params = start_params
            else:
                # It's a packed array
                if len(start_params) == self.k_fe + self.k_re2 + self.k_vc:
                    params = MixedLMParams.from_packed(
                        start_params, self.k_fe, self.k_re, self.use_sqrt,
                        has_fe=True)
                elif len(start_params) == self.k_re2 + self.k_vc:
                    params = MixedLMParams.from_packed(
                        start_params, self.k_fe, self.k_re, self.use_sqrt,
                        has_fe=False)
                else:
                    raise ValueError("invalid start_params")

        if do_cg:
            fit_kwargs["retall"] = hist is not None
            if "disp" not in fit_kwargs:
                fit_kwargs["disp"] = False
            packed = params.get_packed(use_sqrt=self.use_sqrt, has_fe=False)

            if niter_sa > 0:
                warnings.warn("niter_sa is currently ignored")

            # Try optimizing one or more times
            for j in range(len(method)):
                rslt = super(MixedLM, self).fit(start_params=packed,
                                                skip_hessian=True,
                                                method=method[j],
                                                **fit_kwargs)
                if rslt.mle_retvals['converged']:
                    break
                packed = rslt.params
                if j + 1 < len(method):
                    next_method = method[j + 1]
                    warnings.warn(
                        "Retrying MixedLM optimization with %s" % next_method,
                        ConvergenceWarning)
                else:
                    msg = ("MixedLM optimization failed, " +
                           "trying a different optimizer may help.")
                    warnings.warn(msg, ConvergenceWarning)

            # The optimization succeeded
            params = np.atleast_1d(rslt.params)
            if hist is not None:
                hist.append(rslt.mle_retvals)

        converged = rslt.mle_retvals['converged']
        if not converged:
            gn = self.score(rslt.params)
            gn = np.sqrt(np.sum(gn**2))
            msg = "Gradient optimization failed, |grad| = %f" % gn
            warnings.warn(msg, ConvergenceWarning)

        # Convert to the final parameterization (i.e. undo the square
        # root transform of the covariance matrix, and the profiling
        # over the error variance).
        params = MixedLMParams.from_packed(
            params, self.k_fe, self.k_re, use_sqrt=self.use_sqrt, has_fe=False)
        cov_re_unscaled = params.cov_re
        vcomp_unscaled = params.vcomp
        fe_params, sing = self.get_fe_params(cov_re_unscaled, vcomp_unscaled)
        params.fe_params = fe_params
        scale = self.get_scale(fe_params, cov_re_unscaled, vcomp_unscaled)
        cov_re = scale * cov_re_unscaled
        vcomp = scale * vcomp_unscaled

        f1 = (self.k_re > 0) and (np.min(np.abs(np.diag(cov_re))) < 0.01)
        f2 = (self.k_vc > 0) and (np.min(np.abs(vcomp)) < 0.01)
        if f1 or f2:
            msg = "The MLE may be on the boundary of the parameter space."
            warnings.warn(msg, ConvergenceWarning)

        # Compute the Hessian at the MLE.  Note that this is the
        # Hessian with respect to the random effects covariance matrix
        # (not its square root).  It is used for obtaining standard
        # errors, not for optimization.
        hess, sing = self.hessian(params)
        if sing:
            warnings.warn(_warn_cov_sing)

        hess_diag = np.diag(hess)
        if free is not None:
            pcov = np.zeros_like(hess)
            pat = self._freepat.get_packed(use_sqrt=False, has_fe=True)
            ii = np.flatnonzero(pat)
            hess_diag = hess_diag[ii]
            if len(ii) > 0:
                hess1 = hess[np.ix_(ii, ii)]
                pcov[np.ix_(ii, ii)] = np.linalg.inv(-hess1)
        else:
            pcov = np.linalg.inv(-hess)
        if np.any(hess_diag >= 0):
            msg = ("The Hessian matrix at the estimated parameter values " +
                   "is not positive definite.")
            warnings.warn(msg, ConvergenceWarning)

        # Prepare a results class instance
        params_packed = params.get_packed(use_sqrt=False, has_fe=True)
        results = MixedLMResults(self, params_packed, pcov / scale)
        results.params_object = params
        results.fe_params = fe_params
        results.cov_re = cov_re
        results.vcomp = vcomp
        results.scale = scale
        results.cov_re_unscaled = cov_re_unscaled
        results.method = "REML" if self.reml else "ML"
        results.converged = converged
        results.hist = hist
        results.reml = self.reml
        results.cov_pen = self.cov_pen
        results.k_fe = self.k_fe
        results.k_re = self.k_re
        results.k_re2 = self.k_re2
        results.k_vc = self.k_vc
        results.use_sqrt = self.use_sqrt
        results.freepat = self._freepat

        return MixedLMResultsWrapper(results)

    def get_distribution(self, params, scale, exog):
        return _mixedlm_distribution(self, params, scale, exog)


class _mixedlm_distribution:
    """
    A private class for simulating data from a given mixed linear model.

    Parameters
    ----------
    model : MixedLM instance
        A mixed linear model
    params : array_like
        A parameter vector defining a mixed linear model.  See
        notes for more information.
    scale : scalar
        The unexplained variance
    exog : array_like
        An array of fixed effect covariates.  If None, model.exog
        is used.

    Notes
    -----
    The params array is a vector containing fixed effects parameters,
    random effects parameters, and variance component parameters, in
    that order.  The lower triangle of the random effects covariance
    matrix is stored.  The random effects and variance components
    parameters are divided by the scale parameter.

    This class is used in Mediation, and possibly elsewhere.
    """

    def __init__(self, model, params, scale, exog):

        self.model = model
        self.exog = exog if exog is not None else model.exog

        po = MixedLMParams.from_packed(
                params, model.k_fe, model.k_re, False, True)

        self.fe_params = po.fe_params
        self.cov_re = scale * po.cov_re
        self.vcomp = scale * po.vcomp
        self.scale = scale

        group_idx = np.zeros(model.nobs, dtype=int)
        for k, g in enumerate(model.group_labels):
            group_idx[model.row_indices[g]] = k
        self.group_idx = group_idx

    def rvs(self, n):
        """
        Return a vector of simulated values from a mixed linear
        model.

        The parameter n is ignored, but required by the interface
        """

        model = self.model

        # Fixed effects
        y = np.dot(self.exog, self.fe_params)

        # Random effects
        u = np.random.normal(size=(model.n_groups, model.k_re))
        u = np.dot(u, np.linalg.cholesky(self.cov_re).T)
        y += (u[self.group_idx, :] * model.exog_re).sum(1)

        # Variance components
        for j, _ in enumerate(model.exog_vc.names):
            ex = model.exog_vc.mats[j]
            v = self.vcomp[j]
            for i, g in enumerate(model.group_labels):
                exg = ex[i]
                ii = model.row_indices[g]
                u = np.random.normal(size=exg.shape[1])
                y[ii] += np.sqrt(v) * np.dot(exg, u)

        # Residual variance
        y += np.sqrt(self.scale) * np.random.normal(size=len(y))

        return y


class MixedLMResults(base.LikelihoodModelResults, base.ResultMixin):
    '''
    Class to contain results of fitting a linear mixed effects model.

    MixedLMResults inherits from statsmodels.LikelihoodModelResults

    Parameters
    ----------
    See statsmodels.LikelihoodModelResults

    Attributes
    ----------
    model : class instance
        Pointer to MixedLM model instance that called fit.
    normalized_cov_params : ndarray
        The sampling covariance matrix of the estimates
    params : ndarray
        A packed parameter vector for the profile parameterization.
        The first `k_fe` elements are the estimated fixed effects
        coefficients.  The remaining elements are the estimated
        variance parameters.  The variance parameters are all divided
        by `scale` and are not the variance parameters shown
        in the summary.
    fe_params : ndarray
        The fitted fixed-effects coefficients
    cov_re : ndarray
        The fitted random-effects covariance matrix
    bse_fe : ndarray
        The standard errors of the fitted fixed effects coefficients
    bse_re : ndarray
        The standard errors of the fitted random effects covariance
        matrix and variance components.  The first `k_re * (k_re + 1)`
        parameters are the standard errors for the lower triangle of
        `cov_re`, the remaining elements are the standard errors for
        the variance components.

    See Also
    --------
    statsmodels.LikelihoodModelResults
    '''

    def __init__(self, model, params, cov_params):

        super(MixedLMResults, self).__init__(model, params,
                                             normalized_cov_params=cov_params)
        self.nobs = self.model.nobs
        self.df_resid = self.nobs - np.linalg.matrix_rank(self.model.exog)

    @cache_readonly
    def fittedvalues(self):
        """
        Returns the fitted values for the model.

        The fitted values reflect the mean structure specified by the
        fixed effects and the predicted random effects.
        """
        fit = np.dot(self.model.exog, self.fe_params)
        re = self.random_effects
        for group_ix, group in enumerate(self.model.group_labels):
            ix = self.model.row_indices[group]

            mat = []
            if self.model.exog_re_li is not None:
                mat.append(self.model.exog_re_li[group_ix])
            for j in range(self.k_vc):
                mat.append(self.model.exog_vc.mats[j][group_ix])
            mat = np.concatenate(mat, axis=1)

            fit[ix] += np.dot(mat, re[group])

        return fit

    @cache_readonly
    def resid(self):
        """
        Returns the residuals for the model.

        The residuals reflect the mean structure specified by the
        fixed effects and the predicted random effects.
        """
        return self.model.endog - self.fittedvalues

    @cache_readonly
    def bse_fe(self):
        """
        Returns the standard errors of the fixed effect regression
        coefficients.
        """
        p = self.model.exog.shape[1]
        return np.sqrt(np.diag(self.cov_params())[0:p])

    @cache_readonly
    def bse_re(self):
        """
        Returns the standard errors of the variance parameters.

        The first `k_re x (k_re + 1)` elements of the returned array
        are the standard errors of the lower triangle of `cov_re`.
        The remaining elements are the standard errors of the variance
        components.

        Note that the sampling distribution of variance parameters is
        strongly skewed unless the sample size is large, so these
        standard errors may not give meaningful confidence intervals
        or p-values if used in the usual way.
        """
        p = self.model.exog.shape[1]
        return np.sqrt(self.scale * np.diag(self.cov_params())[p:])

    def _expand_re_names(self, group_ix):
        names = list(self.model.data.exog_re_names)

        for j, v in enumerate(self.model.exog_vc.names):
            vg = self.model.exog_vc.colnames[j][group_ix]
            na = ["%s[%s]" % (v, s) for s in vg]
            names.extend(na)

        return names

    @cache_readonly
    def random_effects(self):
        """
        The conditional means of random effects given the data.

        Returns
        -------
        random_effects : dict
            A dictionary mapping the distinct `group` values to the
            conditional means of the random effects for the group
            given the data.
        """
        try:
            cov_re_inv = np.linalg.inv(self.cov_re)
        except np.linalg.LinAlgError:
            raise ValueError("Cannot predict random effects from " +
                             "singular covariance structure.")

        vcomp = self.vcomp
        k_re = self.k_re

        ranef_dict = {}
        for group_ix, group in enumerate(self.model.group_labels):

            endog = self.model.endog_li[group_ix]
            exog = self.model.exog_li[group_ix]
            ex_r = self.model._aex_r[group_ix]
            ex2_r = self.model._aex_r2[group_ix]
            vc_var = self.model._expand_vcomp(vcomp, group_ix)

            # Get the residuals relative to fixed effects
            resid = endog
            if self.k_fe > 0:
                expval = np.dot(exog, self.fe_params)
                resid = resid - expval

            solver = _smw_solver(self.scale, ex_r, ex2_r, cov_re_inv,
                                 1 / vc_var)
            vir = solver(resid)

            xtvir = _dot(ex_r.T, vir)

            xtvir[0:k_re] = np.dot(self.cov_re, xtvir[0:k_re])
            xtvir[k_re:] *= vc_var
            ranef_dict[group] = pd.Series(
                xtvir, index=self._expand_re_names(group_ix))

        return ranef_dict

    @cache_readonly
    def random_effects_cov(self):
        """
        Returns the conditional covariance matrix of the random
        effects for each group given the data.

        Returns
        -------
        random_effects_cov : dict
            A dictionary mapping the distinct values of the `group`
            variable to the conditional covariance matrix of the
            random effects given the data.
        """

        try:
            cov_re_inv = np.linalg.inv(self.cov_re)
        except np.linalg.LinAlgError:
            cov_re_inv = None

        vcomp = self.vcomp

        ranef_dict = {}
        for group_ix in range(self.model.n_groups):

            ex_r = self.model._aex_r[group_ix]
            ex2_r = self.model._aex_r2[group_ix]
            label = self.model.group_labels[group_ix]
            vc_var = self.model._expand_vcomp(vcomp, group_ix)

            solver = _smw_solver(self.scale, ex_r, ex2_r, cov_re_inv,
                                 1 / vc_var)

            n = ex_r.shape[0]
            m = self.cov_re.shape[0]
            mat1 = np.empty((n, m + len(vc_var)))
            mat1[:, 0:m] = np.dot(ex_r[:, 0:m], self.cov_re)
            mat1[:, m:] = np.dot(ex_r[:, m:], np.diag(vc_var))
            mat2 = solver(mat1)
            mat2 = np.dot(mat1.T, mat2)

            v = -mat2
            v[0:m, 0:m] += self.cov_re
            ix = np.arange(m, v.shape[0])
            v[ix, ix] += vc_var
            na = self._expand_re_names(group_ix)
            v = pd.DataFrame(v, index=na, columns=na)
            ranef_dict[label] = v

        return ranef_dict

    # Need to override since t-tests are only used for fixed effects
    # parameters.
    def t_test(self, r_matrix, use_t=None):
        """
        Compute a t-test for a each linear hypothesis of the form Rb = q

        Parameters
        ----------
        r_matrix : array_like
            If an array is given, a p x k 2d array or length k 1d
            array specifying the linear restrictions. It is assumed
            that the linear combination is equal to zero.
        scale : float, optional
            An optional `scale` to use.  Default is the scale specified
            by the model fit.
        use_t : bool, optional
            If use_t is None, then the default of the model is used.
            If use_t is True, then the p-values are based on the t
            distribution.
            If use_t is False, then the p-values are based on the normal
            distribution.

        Returns
        -------
        res : ContrastResults instance
            The results for the test are attributes of this results instance.
            The available results have the same elements as the parameter table
            in `summary()`.
        """
        if r_matrix.shape[1] != self.k_fe:
            raise ValueError("r_matrix for t-test should have %d columns"
                             % self.k_fe)

        d = self.k_re2 + self.k_vc
        z0 = np.zeros((r_matrix.shape[0], d))
        r_matrix = np.concatenate((r_matrix, z0), axis=1)
        tst_rslt = super(MixedLMResults, self).t_test(r_matrix, use_t=use_t)
        return tst_rslt

    def summary(self, yname=None, xname_fe=None, xname_re=None,
                title=None, alpha=.05):
        """
        Summarize the mixed model regression results.

        Parameters
        ----------
        yname : str, optional
            Default is `y`
        xname_fe : list[str], optional
            Fixed effects covariate names
        xname_re : list[str], optional
            Random effects covariate names
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
        statsmodels.iolib.summary2.Summary : class to hold summary results
        """

        from statsmodels.iolib import summary2
        smry = summary2.Summary()

        info = {}
        info["Model:"] = "MixedLM"
        if yname is None:
            yname = self.model.endog_names

        param_names = self.model.data.param_names[:]
        k_fe_params = len(self.fe_params)
        k_re_params = len(param_names) - len(self.fe_params)

        if xname_fe is not None:
            if len(xname_fe) != k_fe_params:
                msg = "xname_fe should be a list of length %d" % k_fe_params
                raise ValueError(msg)
            param_names[:k_fe_params] = xname_fe

        if xname_re is not None:
            if len(xname_re) != k_re_params:
                msg = "xname_re should be a list of length %d" % k_re_params
                raise ValueError(msg)
            param_names[k_fe_params:] = xname_re

        info["No. Observations:"] = str(self.model.n_totobs)
        info["No. Groups:"] = str(self.model.n_groups)

        gs = np.array([len(x) for x in self.model.endog_li])
        info["Min. group size:"] = "%.0f" % min(gs)
        info["Max. group size:"] = "%.0f" % max(gs)
        info["Mean group size:"] = "%.1f" % np.mean(gs)

        info["Dependent Variable:"] = yname
        info["Method:"] = self.method
        info["Scale:"] = self.scale
        info["Log-Likelihood:"] = self.llf
        info["Converged:"] = "Yes" if self.converged else "No"
        smry.add_dict(info)
        smry.add_title("Mixed Linear Model Regression Results")

        float_fmt = "%.3f"

        sdf = np.nan * np.ones((self.k_fe + self.k_re2 + self.k_vc, 6))

        # Coefficient estimates
        sdf[0:self.k_fe, 0] = self.fe_params

        # Standard errors
        sdf[0:self.k_fe, 1] = np.sqrt(np.diag(self.cov_params()[0:self.k_fe]))

        # Z-scores
        sdf[0:self.k_fe, 2] = sdf[0:self.k_fe, 0] / sdf[0:self.k_fe, 1]

        # p-values
        sdf[0:self.k_fe, 3] = 2 * norm.cdf(-np.abs(sdf[0:self.k_fe, 2]))

        # Confidence intervals
        qm = -norm.ppf(alpha / 2)
        sdf[0:self.k_fe, 4] = sdf[0:self.k_fe, 0] - qm * sdf[0:self.k_fe, 1]
        sdf[0:self.k_fe, 5] = sdf[0:self.k_fe, 0] + qm * sdf[0:self.k_fe, 1]

        # All random effects variances and covariances
        jj = self.k_fe
        for i in range(self.k_re):
            for j in range(i + 1):
                sdf[jj, 0] = self.cov_re[i, j]
                sdf[jj, 1] = np.sqrt(self.scale) * self.bse[jj]
                jj += 1

        # Variance components
        for i in range(self.k_vc):
            sdf[jj, 0] = self.vcomp[i]
            sdf[jj, 1] = np.sqrt(self.scale) * self.bse[jj]
            jj += 1

        sdf = pd.DataFrame(index=param_names, data=sdf)
        sdf.columns = ['Coef.', 'Std.Err.', 'z', 'P>|z|',
                       '[' + str(alpha/2), str(1-alpha/2) + ']']
        for col in sdf.columns:
            sdf[col] = [float_fmt % x if np.isfinite(x) else ""
                        for x in sdf[col]]

        smry.add_df(sdf, align='r')

        return smry

    @cache_readonly
    def llf(self):
        return self.model.loglike(self.params_object, profile_fe=False)

    @cache_readonly
    def aic(self):
        """Akaike information criterion"""
        if self.reml:
            return np.nan
        if self.freepat is not None:
            df = self.freepat.get_packed(use_sqrt=False, has_fe=True).sum() + 1
        else:
            df = self.params.size + 1
        return -2 * (self.llf - df)

    @cache_readonly
    def bic(self):
        """Bayesian information criterion"""
        if self.reml:
            return np.nan
        if self.freepat is not None:
            df = self.freepat.get_packed(use_sqrt=False, has_fe=True).sum() + 1
        else:
            df = self.params.size + 1
        return -2 * self.llf + np.log(self.nobs) * df

    def profile_re(self, re_ix, vtype, num_low=5, dist_low=1., num_high=5,
                   dist_high=1., **fit_kwargs):
        """
        Profile-likelihood inference for variance parameters.

        Parameters
        ----------
        re_ix : int
            If vtype is `re`, this value is the index of the variance
            parameter for which to construct a profile likelihood.  If
            `vtype` is 'vc' then `re_ix` is the name of the variance
            parameter to be profiled.
        vtype : str
            Either 're' or 'vc', depending on whether the profile
            analysis is for a random effect or a variance component.
        num_low : int
            The number of points at which to calculate the likelihood
            below the MLE of the parameter of interest.
        dist_low : float
            The distance below the MLE of the parameter of interest to
            begin calculating points on the profile likelihood.
        num_high : int
            The number of points at which to calculate the likelihood
            above the MLE of the parameter of interest.
        dist_high : float
            The distance above the MLE of the parameter of interest to
            begin calculating points on the profile likelihood.
        **fit_kwargs
            Additional keyword arguments passed to fit.

        Returns
        -------
        An array with two columns.  The first column contains the
        values to which the parameter of interest is constrained.  The
        second column contains the corresponding likelihood values.

        Notes
        -----
        Only variance parameters can be profiled.
        """

        pmodel = self.model
        k_fe = pmodel.k_fe
        k_re = pmodel.k_re
        k_vc = pmodel.k_vc
        endog, exog = pmodel.endog, pmodel.exog

        # Need to permute the columns of the random effects design
        # matrix so that the profiled variable is in the first column.
        if vtype == 're':
            ix = np.arange(k_re)
            ix[0] = re_ix
            ix[re_ix] = 0
            exog_re = pmodel.exog_re.copy()[:, ix]

            # Permute the covariance structure to match the permuted
            # design matrix.
            params = self.params_object.copy()
            cov_re_unscaled = params.cov_re
            cov_re_unscaled = cov_re_unscaled[np.ix_(ix, ix)]
            params.cov_re = cov_re_unscaled
            ru0 = cov_re_unscaled[0, 0]

            # Convert dist_low and dist_high to the profile
            # parameterization
            cov_re = self.scale * cov_re_unscaled
            low = (cov_re[0, 0] - dist_low) / self.scale
            high = (cov_re[0, 0] + dist_high) / self.scale

        elif vtype == 'vc':
            re_ix = self.model.exog_vc.names.index(re_ix)
            params = self.params_object.copy()
            vcomp = self.vcomp
            low = (vcomp[re_ix] - dist_low) / self.scale
            high = (vcomp[re_ix] + dist_high) / self.scale
            ru0 = vcomp[re_ix] / self.scale

        # Define the sequence of values to which the parameter of
        # interest will be constrained.
        if low <= 0:
            raise ValueError("dist_low is too large and would result in a "
                             "negative variance. Try a smaller value.")
        left = np.linspace(low, ru0, num_low + 1)
        right = np.linspace(ru0, high, num_high+1)[1:]
        rvalues = np.concatenate((left, right))

        # Indicators of which parameters are free and fixed.
        free = MixedLMParams(k_fe, k_re, k_vc)
        if self.freepat is None:
            free.fe_params = np.ones(k_fe)
            vcomp = np.ones(k_vc)
            mat = np.ones((k_re, k_re))
        else:
            # If a freepat already has been specified, we add the
            # constraint to it.
            free.fe_params = self.freepat.fe_params
            vcomp = self.freepat.vcomp
            mat = self.freepat.cov_re
            if vtype == 're':
                mat = mat[np.ix_(ix, ix)]
        if vtype == 're':
            mat[0, 0] = 0
        else:
            vcomp[re_ix] = 0
        free.cov_re = mat
        free.vcomp = vcomp

        klass = self.model.__class__
        init_kwargs = pmodel._get_init_kwds()
        if vtype == 're':
            init_kwargs['exog_re'] = exog_re

        likev = []
        for x in rvalues:

            model = klass(endog, exog, **init_kwargs)

            if vtype == 're':
                cov_re = params.cov_re.copy()
                cov_re[0, 0] = x
                params.cov_re = cov_re
            else:
                params.vcomp[re_ix] = x

            # TODO should use fit_kwargs
            rslt = model.fit(start_params=params, free=free,
                             reml=self.reml, cov_pen=self.cov_pen,
                             **fit_kwargs)._results
            likev.append([x * rslt.scale, rslt.llf])

        likev = np.asarray(likev)

        return likev


class MixedLMResultsWrapper(base.LikelihoodResultsWrapper):
    _attrs = {'bse_re': ('generic_columns', 'exog_re_names_full'),
              'fe_params': ('generic_columns', 'xnames'),
              'bse_fe': ('generic_columns', 'xnames'),
              'cov_re': ('generic_columns_2d', 'exog_re_names'),
              'cov_re_unscaled': ('generic_columns_2d', 'exog_re_names'),
              }
    _upstream_attrs = base.LikelihoodResultsWrapper._wrap_attrs
    _wrap_attrs = base.wrap.union_dicts(_attrs, _upstream_attrs)

    _methods = {}
    _upstream_methods = base.LikelihoodResultsWrapper._wrap_methods
    _wrap_methods = base.wrap.union_dicts(_methods, _upstream_methods)


def _handle_missing(data, groups, formula, re_formula, vc_formula):

    tokens = set()

    forms = [formula]
    if re_formula is not None:
        forms.append(re_formula)
    if vc_formula is not None:
        forms.extend(vc_formula.values())

    from statsmodels.compat.python import asunicode

    from io import StringIO
    import tokenize
    skiptoks = {"(", ")", "*", ":", "+", "-", "**", "/"}

    for fml in forms:
        # Unicode conversion is for Py2 compatability
        rl = StringIO(fml)

        def rlu():
            line = rl.readline()
            return asunicode(line, 'ascii')
        g = tokenize.generate_tokens(rlu)
        for tok in g:
            if tok not in skiptoks:
                tokens.add(tok.string)
    tokens = sorted(tokens & set(data.columns))

    data = data[tokens]
    ii = pd.notnull(data).all(1)
    if type(groups) != "str":
        ii &= pd.notnull(groups)

    return data.loc[ii, :], groups[np.asarray(ii)]
