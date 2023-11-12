"""
Holds files for l1 regularization of LikelihoodModel, using cvxopt.
"""
import numpy as np
import statsmodels.base.l1_solvers_common as l1_solvers_common


def fit_l1_cvxopt_cp(
        f, score, start_params, args, kwargs, disp=False, maxiter=100,
        callback=None, retall=False, full_output=False, hess=None):
    """
    Solve the l1 regularized problem using cvxopt.solvers.cp

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
    abstol : float
        absolute accuracy (default: 1e-7).
    reltol : float
        relative accuracy (default: 1e-6).
    feastol : float
        tolerance for feasibility conditions (default: 1e-7).
    refinement : int
        number of iterative refinement steps when solving KKT equations
        (default: 1).
    """
    from cvxopt import solvers, matrix

    start_params = np.array(start_params).ravel('F')

    ## Extract arguments
    # k_params is total number of covariates, possibly including a leading constant.
    k_params = len(start_params)
    # The start point
    x0 = np.append(start_params, np.fabs(start_params))
    x0 = matrix(x0, (2 * k_params, 1))
    # The regularization parameter
    alpha = np.array(kwargs['alpha_rescaled']).ravel('F')
    # Make sure it's a vector
    alpha = alpha * np.ones(k_params)
    assert alpha.min() >= 0

    ## Wrap up functions for cvxopt
    f_0 = lambda x: _objective_func(f, x, k_params, alpha, *args)
    Df = lambda x: _fprime(score, x, k_params, alpha)
    G = _get_G(k_params)  # Inequality constraint matrix, Gx \leq h
    h = matrix(0.0, (2 * k_params, 1))  # RHS in inequality constraint
    H = lambda x, z: _hessian_wrapper(hess, x, z, k_params)

    ## Define the optimization function
    def F(x=None, z=None):
        if x is None:
            return 0, x0
        elif z is None:
            return f_0(x), Df(x)
        else:
            return f_0(x), Df(x), H(x, z)

    ## Convert optimization settings to cvxopt form
    solvers.options['show_progress'] = disp
    solvers.options['maxiters'] = maxiter
    if 'abstol' in kwargs:
        solvers.options['abstol'] = kwargs['abstol']
    if 'reltol' in kwargs:
        solvers.options['reltol'] = kwargs['reltol']
    if 'feastol' in kwargs:
        solvers.options['feastol'] = kwargs['feastol']
    if 'refinement' in kwargs:
        solvers.options['refinement'] = kwargs['refinement']

    ### Call the optimizer
    results = solvers.cp(F, G, h)
    x = np.asarray(results['x']).ravel()
    params = x[:k_params]

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

    ### Pack up return values for statsmodels
    # TODO These retvals are returned as mle_retvals...but the fit was not ML
    if full_output:
        fopt = f_0(x)
        gopt = float('nan')  # Objective is non-differentiable
        hopt = float('nan')
        iterations = float('nan')
        converged = (results['status'] == 'optimal')
        warnflag = results['status']
        retvals = {
            'fopt': fopt, 'converged': converged, 'iterations': iterations,
            'gopt': gopt, 'hopt': hopt, 'trimmed': trimmed,
            'warnflag': warnflag}
    else:
        x = np.array(results['x']).ravel()
        params = x[:k_params]

    ### Return results
    if full_output:
        return params, retvals
    else:
        return params


def _objective_func(f, x, k_params, alpha, *args):
    """
    The regularized objective function.
    """
    from cvxopt import matrix

    x_arr = np.asarray(x)
    params = x_arr[:k_params].ravel()
    u = x_arr[k_params:]
    # Call the numpy version
    objective_func_arr = f(params, *args) + (alpha * u).sum()
    # Return
    return matrix(objective_func_arr)


def _fprime(score, x, k_params, alpha):
    """
    The regularized derivative.
    """
    from cvxopt import matrix

    x_arr = np.asarray(x)
    params = x_arr[:k_params].ravel()
    # Call the numpy version
    # The derivative just appends a vector of constants
    fprime_arr = np.append(score(params), alpha)
    # Return
    return matrix(fprime_arr, (1, 2 * k_params))


def _get_G(k_params):
    """
    The linear inequality constraint matrix.
    """
    from cvxopt import matrix

    I = np.eye(k_params)  # noqa:E741
    A = np.concatenate((-I, -I), axis=1)
    B = np.concatenate((I, -I), axis=1)
    C = np.concatenate((A, B), axis=0)
    # Return
    return matrix(C)


def _hessian_wrapper(hess, x, z, k_params):
    """
    Wraps the hessian up in the form for cvxopt.

    cvxopt wants the hessian of the objective function and the constraints.
        Since our constraints are linear, this part is all zeros.
    """
    from cvxopt import matrix

    x_arr = np.asarray(x)
    params = x_arr[:k_params].ravel()
    zh_x = np.asarray(z[0]) * hess(params)
    zero_mat = np.zeros(zh_x.shape)
    A = np.concatenate((zh_x, zero_mat), axis=1)
    B = np.concatenate((zero_mat, zero_mat), axis=1)
    zh_x_ext = np.concatenate((A, B), axis=0)
    return matrix(zh_x_ext, (2 * k_params, 2 * k_params))
