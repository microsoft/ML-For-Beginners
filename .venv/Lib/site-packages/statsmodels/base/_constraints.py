# -*- coding: utf-8 -*-
"""
Created on Thu May 15 16:36:05 2014

Author: Josef Perktold
License: BSD-3

"""

import numpy as np


class LinearConstraints:
    """Class to hold linear constraints information

    Affine constraints are defined as ``R b = q` where `R` is the constraints
    matrix and `q` are the constraints values and `b` are the parameters.

    This is in analogy to patsy's LinearConstraints class but can be pickled.

    Parameters
    ----------
    constraint_matrix : ndarray
        R matrix, 2-dim with number of columns equal to the number of
        parameters. Each row defines one constraint.
    constraint_values : ndarray
        1-dim array of constant values
    variable_names : list of strings
        parameter names, used only for display
    kwds : keyword arguments
        keywords are attached to the instance.

    """

    def __init__(self, constraint_matrix, constraint_values,
                 variable_names, **kwds):

        self.constraint_matrix = constraint_matrix
        self.constraint_values = constraint_values
        self.variable_names = variable_names

        # alias for patsy compatibility
        self.coefs = constraint_matrix
        self.constants = constraint_values

        self.__dict__.update(kwds)
        self.tuple = (self.constraint_matrix, self.constraint_values)

    def __iter__(self):
        yield from self.tuple

    def __getitem__(self, idx):
        return self.tuple[idx]

    def __str__(self):
        def prod_string(v, name):
            v = np.abs(v)
            if v != 1:
                ss = str(v) + " * " + name
            else:
                ss = name
            return ss

        constraints_strings = []
        for r, q in zip(*self):
            ss = []
            for v, name in zip(r, self.variable_names):
                if v != 0 and ss == []:
                    ss += prod_string(v, name)
                elif v > 0:
                    ss += " + " + prod_string(v, name)
                elif v < 0:
                    ss += " - " + prod_string(np.abs(v), name)
            ss += " = " + str(q.item())
            constraints_strings.append(''.join(ss))

        return '\n'.join(constraints_strings)

    @classmethod
    def from_patsy(cls, lc):
        """class method to create instance from patsy instance

        Parameters
        ----------
        lc : instance
            instance of patsy LinearConstraint, or other instances that have
            attributes ``lc.coefs, lc.constants, lc.variable_names``

        Returns
        -------
        instance of this class

        """
        return cls(lc.coefs, lc.constants, lc.variable_names)


class TransformRestriction:
    """Transformation for linear constraints `R params = q`

    Note, the transformation from the reduced to the full parameters is an
    affine and not a linear transformation if q is not zero.


    Parameters
    ----------
    R : array_like
        Linear restriction matrix
    q : arraylike or None
        values of the linear restrictions


    Notes
    -----
    The reduced parameters are not sorted with respect to constraints.

    TODO: error checking, eg. inconsistent constraints, how?

    Inconsistent constraints will raise an exception in the calculation of
    the constant or offset. However, homogeneous constraints, where q=0, will
    can have a solution where the relevant parameters are constraint to be
    zero, as in the following example::

        b1 + b2 = 0 and b1 + 2*b2 = 0, implies that b2 = 0.

    The transformation applied from full to reduced parameter space does not
    raise and exception if the constraint does not hold.
    TODO: maybe change this, what's the behavior in this case?


    The `reduce` transform is applied to the array of explanatory variables,
    `exog`, when transforming a linear model to impose the constraints.
    """

    def __init__(self, R, q=None):

        # The calculations are based on Stata manual for makecns
        R = self.R = np.atleast_2d(R)
        if q is not None:
            q = self.q = np.asarray(q)

        k_constr, k_vars = R.shape
        self.k_constr, self.k_vars = k_constr, k_vars
        self.k_unconstr = k_vars - k_constr

        m = np.eye(k_vars) - R.T.dot(np.linalg.pinv(R).T)
        evals, evecs = np.linalg.eigh(m)

        # This normalizes the transformation so the larges element is 1.
        # It makes it easier to interpret simple restrictions, e.g. b1 + b2 = 0
        # TODO: make this work, there is something wrong, does not round-trip
        #       need to adjust constant
        #evecs_maxabs = np.max(np.abs(evecs), 0)
        #evecs = evecs / evecs_maxabs

        self.evals = evals
        self.evecs = evecs # temporarily attach as attribute
        L = self.L = evecs[:, :k_constr]
        self.transf_mat = evecs[:, k_constr:]

        if q is not None:
            # use solve instead of inv
            #self.constant = q.T.dot(np.linalg.inv(L.T.dot(R.T)).dot(L.T))
            try:
                self.constant = q.T.dot(np.linalg.solve(L.T.dot(R.T), L.T))
            except np.linalg.linalg.LinAlgError as e:
                raise ValueError('possibly inconsistent constraints. error '
                                 'generated by\n%r' % (e, ))
        else:
            self.constant = 0

    def expand(self, params_reduced):
        """transform from the reduced to the full parameter space

        Parameters
        ----------
        params_reduced : array_like
            parameters in the transformed space

        Returns
        -------
        params : array_like
            parameters in the original space

        Notes
        -----
        If the restriction is not homogeneous, i.e. q is not equal to zero,
        then this is an affine transform.
        """
        params_reduced = np.asarray(params_reduced)
        return self.transf_mat.dot(params_reduced.T).T + self.constant

    def reduce(self, params):
        """transform from the full to the reduced parameter space

        Parameters
        ----------
        params : array_like
            parameters or data in the original space

        Returns
        -------
        params_reduced : array_like
            parameters in the transformed space

        This transform can be applied to the original parameters as well
        as to the data. If params is 2-d, then each row is transformed.
        """
        params = np.asarray(params)
        return params.dot(self.transf_mat)


def transform_params_constraint(params, Sinv, R, q):
    """find the parameters that statisfy linear constraint from unconstrained

    The linear constraint R params = q is imposed.

    Parameters
    ----------
    params : array_like
        unconstrained parameters
    Sinv : ndarray, 2d, symmetric
        covariance matrix of the parameter estimate
    R : ndarray, 2d
        constraint matrix
    q : ndarray, 1d
        values of the constraint

    Returns
    -------
    params_constraint : ndarray
        parameters of the same length as params satisfying the constraint

    Notes
    -----
    This is the exact formula for OLS and other linear models. It will be
    a local approximation for nonlinear models.

    TODO: Is Sinv always the covariance matrix?
    In the linear case it can be (X'X)^{-1} or sigmahat^2 (X'X)^{-1}.

    My guess is that this is the point in the subspace that satisfies
    the constraint that has minimum Mahalanobis distance. Proof ?
    """

    rsr = R.dot(Sinv).dot(R.T)

    reduction = Sinv.dot(R.T).dot(np.linalg.solve(rsr, R.dot(params) - q))
    return params - reduction


def fit_constrained(model, constraint_matrix, constraint_values,
                    start_params=None, fit_kwds=None):
    # note: self is model instance
    """fit model subject to linear equality constraints

    The constraints are of the form   `R params = q`
    where R is the constraint_matrix and q is the vector of constraint_values.

    The estimation creates a new model with transformed design matrix,
    exog, and converts the results back to the original parameterization.


    Parameters
    ----------
    model: model instance
        An instance of a model, see limitations in Notes section
    constraint_matrix : array_like, 2D
        This is R in the linear equality constraint `R params = q`.
        The number of columns needs to be the same as the number of columns
        in exog.
    constraint_values :
        This is `q` in the linear equality constraint `R params = q`
        If it is a tuple, then the constraint needs to be given by two
        arrays (constraint_matrix, constraint_value), i.e. (R, q).
        Otherwise, the constraints can be given as strings or list of
        strings.
        see t_test for details
    start_params : None or array_like
        starting values for the optimization. `start_params` needs to be
        given in the original parameter space and are internally
        transformed.
    **fit_kwds : keyword arguments
        fit_kwds are used in the optimization of the transformed model.

    Returns
    -------
    params : ndarray ?
        estimated parameters (in the original parameterization
    cov_params : ndarray
        covariance matrix of the parameter estimates. This is a reverse
        transformation of the covariance matrix of the transformed model given
        by `cov_params()`
        Note: `fit_kwds` can affect the choice of covariance, e.g. by
        specifying `cov_type`, which will be reflected in the returned
        covariance.
    res_constr : results instance
        This is the results instance for the created transformed model.


    Notes
    -----
    Limitations:

    Models where the number of parameters is different from the number of
    columns of exog are not yet supported.

    Requires a model that implement an offset option.
    """
    self = model   # internal alias, used for methods
    if fit_kwds is None:
        fit_kwds = {}

    R, q = constraint_matrix, constraint_values
    endog, exog = self.endog, self.exog

    transf = TransformRestriction(R, q)

    exogp_st = transf.reduce(exog)

    offset = exog.dot(transf.constant.squeeze())
    if hasattr(self, 'offset'):
        offset += self.offset

    if start_params is not None:
        start_params =  transf.reduce(start_params)

    #need copy, because we do not want to change it, we do not need deepcopy
    import copy
    init_kwds = copy.copy(self._get_init_kwds())

    # TODO: refactor to combine with above or offset_all
    if 'offset' in init_kwds:
        del init_kwds['offset']

    # using offset as keywords is not supported in all modules
    mod_constr = self.__class__(endog, exogp_st, offset=offset, **init_kwds)
    res_constr = mod_constr.fit(start_params=start_params, **fit_kwds)
    params_orig = transf.expand(res_constr.params).squeeze()
    cov_params = transf.transf_mat.dot(res_constr.cov_params()).dot(transf.transf_mat.T)

    return params_orig, cov_params, res_constr


def fit_constrained_wrap(model, constraints, start_params=None, **fit_kwds):
    """fit_constraint that returns a results instance

    This is a development version for fit_constrained methods or
    fit_constrained as standalone function.

    It will not work correctly for all models because creating a new
    results instance is not standardized for use outside the `fit` methods,
    and might need adjustements for this.

    This is the prototype for the fit_constrained method that has been added
    to Poisson and GLM.
    """

    self = model  # alias for use as method

    #constraints = (R, q)
    # TODO: temporary trailing underscore to not overwrite the monkey
    #       patched version
    # TODO: decide whether to move the imports
    from patsy import DesignInfo
    # we need this import if we copy it to a different module
    #from statsmodels.base._constraints import fit_constrained

    # same pattern as in base.LikelihoodModel.t_test
    lc = DesignInfo(self.exog_names).linear_constraint(constraints)
    R, q = lc.coefs, lc.constants

    # TODO: add start_params option, need access to tranformation
    #       fit_constrained needs to do the transformation
    params, cov, res_constr = fit_constrained(self, R, q,
                                              start_params=start_params,
                                              fit_kwds=fit_kwds)
    #create dummy results Instance, TODO: wire up properly
    res = self.fit(start_params=params, maxiter=0,
                   warn_convergence=False)  # we get a wrapper back
    res._results.params = params
    res._results.cov_params_default = cov
    cov_type = fit_kwds.get('cov_type', 'nonrobust')
    if cov_type == 'nonrobust':
        res._results.normalized_cov_params = cov / res_constr.scale
    else:
        res._results.normalized_cov_params = None

    k_constr = len(q)
    res._results.df_resid += k_constr
    res._results.df_model -= k_constr
    res._results.constraints = LinearConstraints.from_patsy(lc)
    res._results.k_constr = k_constr
    res._results.results_constrained = res_constr
    return res
