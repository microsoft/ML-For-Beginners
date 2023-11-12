"""
Holds files for l1 regularization of LikelihoodModel, using
scipy.optimize.slsqp
"""
import numpy as np
from scipy.optimize import fmin_slsqp
import statsmodels.base.l1_solvers_common as l1_solvers_common


def fit_l1_slsqp(
        f, score, start_params, args, kwargs, disp=False, maxiter=1000,
        callback=None, retall=False, full_output=False, hess=None):
    """
    Solve the l1 regularized problem using scipy.optimize.fmin_slsqp().

    Specifically:  We convert the convex but non-smooth problem

    .. math:: \\min_\\beta f(\\beta) + \\sum_k\\alpha_k |\\beta_k|

    via the transformation to the smooth, convex, constrained problem in twice
    as many variables (adding the "added variables" :math:`u_k`)

    .. math:: \\min_{\\beta,u} f(\\beta) + \\sum_k\\alpha_k u_k,

    subject to

    .. math:: -u_k \\leq \\beta_k \\leq u_k.

    Parameters
    ----------
    All the usual parameters from LikelhoodModel.fit
    alpha : non-negative scalar or numpy array (same size as parameters)
        The weight multiplying the l1 penalty term
    trim_mode : 'auto, 'size', or 'off'
        If not 'off', trim (set to zero) parameters that would have been zero
            if the solver reached the theoretical minimum.
        If 'auto', trim params using the Theory above.
        If 'size', trim params if they have very small absolute value
    size_trim_tol : float or 'auto' (default = 'auto')
        For use when trim_mode === 'size'
    auto_trim_tol : float
        For sue when trim_mode == 'auto'.  Use
    qc_tol : float
        Print warning and do not allow auto trim when (ii) in "Theory" (above)
        is violated by this much.
    qc_verbose : bool
        If true, print out a full QC report upon failure
    acc : float (default 1e-6)
        Requested accuracy as used by slsqp
    """
    start_params = np.array(start_params).ravel('F')

    ### Extract values
    # k_params is total number of covariates,
    # possibly including a leading constant.
    k_params = len(start_params)
    # The start point
    x0 = np.append(start_params, np.fabs(start_params))
    # alpha is the regularization parameter
    alpha = np.array(kwargs['alpha_rescaled']).ravel('F')
    # Make sure it's a vector
    alpha = alpha * np.ones(k_params)
    assert alpha.min() >= 0
    # Convert display parameters to scipy.optimize form
    disp_slsqp = _get_disp_slsqp(disp, retall)
    # Set/retrieve the desired accuracy
    acc = kwargs.setdefault('acc', 1e-10)

    ### Wrap up for use in fmin_slsqp
    func = lambda x_full: _objective_func(f, x_full, k_params, alpha, *args)
    f_ieqcons_wrap = lambda x_full: _f_ieqcons(x_full, k_params)
    fprime_wrap = lambda x_full: _fprime(score, x_full, k_params, alpha)
    fprime_ieqcons_wrap = lambda x_full: _fprime_ieqcons(x_full, k_params)

    ### Call the solver
    results = fmin_slsqp(
        func, x0, f_ieqcons=f_ieqcons_wrap, fprime=fprime_wrap, acc=acc,
        iter=maxiter, disp=disp_slsqp, full_output=full_output,
        fprime_ieqcons=fprime_ieqcons_wrap)
    params = np.asarray(results[0][:k_params])

    ### Post-process
    # QC
    qc_tol = kwargs['qc_tol']
    qc_verbose = kwargs['qc_verbose']
    passed = l1_solvers_common.qc_results(
        params, alpha, score, qc_tol, qc_verbose)
    # Possibly trim
    trim_mode = kwargs['trim_mode']
    size_trim_tol = kwargs['size_trim_tol']
    auto_trim_tol = kwargs['auto_trim_tol']
    params, trimmed = l1_solvers_common.do_trim_params(
        params, k_params, alpha, score, passed, trim_mode, size_trim_tol,
        auto_trim_tol)

    ### Pack up return values for statsmodels optimizers
    # TODO These retvals are returned as mle_retvals...but the fit was not ML.
    # This could be confusing someday.
    if full_output:
        x_full, fx, its, imode, smode = results
        fopt = func(np.asarray(x_full))
        converged = (imode == 0)
        warnflag = str(imode) + ' ' + smode
        iterations = its
        gopt = float('nan')     # Objective is non-differentiable
        hopt = float('nan')
        retvals = {
            'fopt': fopt, 'converged': converged, 'iterations': iterations,
            'gopt': gopt, 'hopt': hopt, 'trimmed': trimmed,
            'warnflag': warnflag}

    ### Return
    if full_output:
        return params, retvals
    else:
        return params


def _get_disp_slsqp(disp, retall):
    if disp or retall:
        if disp:
            disp_slsqp = 1
        if retall:
            disp_slsqp = 2
    else:
        disp_slsqp = 0
    return disp_slsqp


def _objective_func(f, x_full, k_params, alpha, *args):
    """
    The regularized objective function
    """
    x_params = x_full[:k_params]
    x_added = x_full[k_params:]
    ## Return
    return f(x_params, *args) + (alpha * x_added).sum()


def _fprime(score, x_full, k_params, alpha):
    """
    The regularized derivative
    """
    x_params = x_full[:k_params]
    # The derivative just appends a vector of constants
    return np.append(score(x_params), alpha)


def _f_ieqcons(x_full, k_params):
    """
    The inequality constraints.
    """
    x_params = x_full[:k_params]
    x_added = x_full[k_params:]
    # All entries in this vector must be \geq 0 in a feasible solution
    return np.append(x_params + x_added, x_added - x_params)


def _fprime_ieqcons(x_full, k_params):
    """
    Derivative of the inequality constraints
    """
    I = np.eye(k_params)  # noqa:E741
    A = np.concatenate((I, I), axis=1)
    B = np.concatenate((-I, I), axis=1)
    C = np.concatenate((A, B), axis=0)
    ## Return
    return C
