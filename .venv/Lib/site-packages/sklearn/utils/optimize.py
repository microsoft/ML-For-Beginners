"""
Our own implementation of the Newton algorithm

Unlike the scipy.optimize version, this version of the Newton conjugate
gradient solver uses only one function call to retrieve the
func value, the gradient value and a callable for the Hessian matvec
product. If the function call is very expensive (e.g. for logistic
regression with large design matrix), this approach gives very
significant speedups.
"""
# This is a modified file from scipy.optimize
# Original authors: Travis Oliphant, Eric Jones
# Modifications by Gael Varoquaux, Mathieu Blondel and Tom Dupre la Tour
# License: BSD

import warnings

import numpy as np
import scipy

from ..exceptions import ConvergenceWarning
from .fixes import line_search_wolfe1, line_search_wolfe2


class _LineSearchError(RuntimeError):
    pass


def _line_search_wolfe12(f, fprime, xk, pk, gfk, old_fval, old_old_fval, **kwargs):
    """
    Same as line_search_wolfe1, but fall back to line_search_wolfe2 if
    suitable step length is not found, and raise an exception if a
    suitable step length is not found.

    Raises
    ------
    _LineSearchError
        If no suitable step size is found.

    """
    ret = line_search_wolfe1(f, fprime, xk, pk, gfk, old_fval, old_old_fval, **kwargs)

    if ret[0] is None:
        # Have a look at the line_search method of our NewtonSolver class. We borrow
        # the logic from there
        # Deal with relative loss differences around machine precision.
        args = kwargs.get("args", tuple())
        fval = f(xk + pk, *args)
        eps = 16 * np.finfo(np.asarray(old_fval).dtype).eps
        tiny_loss = np.abs(old_fval * eps)
        loss_improvement = fval - old_fval
        check = np.abs(loss_improvement) <= tiny_loss
        if check:
            # 2.1 Check sum of absolute gradients as alternative condition.
            sum_abs_grad_old = scipy.linalg.norm(gfk, ord=1)
            grad = fprime(xk + pk, *args)
            sum_abs_grad = scipy.linalg.norm(grad, ord=1)
            check = sum_abs_grad < sum_abs_grad_old
            if check:
                ret = (
                    1.0,  # step size
                    ret[1] + 1,  # number of function evaluations
                    ret[2] + 1,  # number of gradient evaluations
                    fval,
                    old_fval,
                    grad,
                )

    if ret[0] is None:
        # line search failed: try different one.
        # TODO: It seems that the new check for the sum of absolute gradients above
        # catches all cases that, earlier, ended up here. In fact, our tests never
        # trigger this "if branch" here and we can consider to remove it.
        ret = line_search_wolfe2(
            f, fprime, xk, pk, gfk, old_fval, old_old_fval, **kwargs
        )

    if ret[0] is None:
        raise _LineSearchError()

    return ret


def _cg(fhess_p, fgrad, maxiter, tol):
    """
    Solve iteratively the linear system 'fhess_p . xsupi = fgrad'
    with a conjugate gradient descent.

    Parameters
    ----------
    fhess_p : callable
        Function that takes the gradient as a parameter and returns the
        matrix product of the Hessian and gradient.

    fgrad : ndarray of shape (n_features,) or (n_features + 1,)
        Gradient vector.

    maxiter : int
        Number of CG iterations.

    tol : float
        Stopping criterion.

    Returns
    -------
    xsupi : ndarray of shape (n_features,) or (n_features + 1,)
        Estimated solution.
    """
    xsupi = np.zeros(len(fgrad), dtype=fgrad.dtype)
    ri = np.copy(fgrad)
    psupi = -ri
    i = 0
    dri0 = np.dot(ri, ri)
    # We also track of |p_i|^2.
    psupi_norm2 = dri0

    while i <= maxiter:
        if np.sum(np.abs(ri)) <= tol:
            break

        Ap = fhess_p(psupi)
        # check curvature
        curv = np.dot(psupi, Ap)
        if 0 <= curv <= 16 * np.finfo(np.float64).eps * psupi_norm2:
            # See https://arxiv.org/abs/1803.02924, Algo 1 Capped Conjugate Gradient.
            break
        elif curv < 0:
            if i > 0:
                break
            else:
                # fall back to steepest descent direction
                xsupi += dri0 / curv * psupi
                break
        alphai = dri0 / curv
        xsupi += alphai * psupi
        ri += alphai * Ap
        dri1 = np.dot(ri, ri)
        betai = dri1 / dri0
        psupi = -ri + betai * psupi
        # We use  |p_i|^2 = |r_i|^2 + beta_i^2 |p_{i-1}|^2
        psupi_norm2 = dri1 + betai**2 * psupi_norm2
        i = i + 1
        dri0 = dri1  # update np.dot(ri,ri) for next time.

    return xsupi


def _newton_cg(
    grad_hess,
    func,
    grad,
    x0,
    args=(),
    tol=1e-4,
    maxiter=100,
    maxinner=200,
    line_search=True,
    warn=True,
):
    """
    Minimization of scalar function of one or more variables using the
    Newton-CG algorithm.

    Parameters
    ----------
    grad_hess : callable
        Should return the gradient and a callable returning the matvec product
        of the Hessian.

    func : callable
        Should return the value of the function.

    grad : callable
        Should return the function value and the gradient. This is used
        by the linesearch functions.

    x0 : array of float
        Initial guess.

    args : tuple, default=()
        Arguments passed to func_grad_hess, func and grad.

    tol : float, default=1e-4
        Stopping criterion. The iteration will stop when
        ``max{|g_i | i = 1, ..., n} <= tol``
        where ``g_i`` is the i-th component of the gradient.

    maxiter : int, default=100
        Number of Newton iterations.

    maxinner : int, default=200
        Number of CG iterations.

    line_search : bool, default=True
        Whether to use a line search or not.

    warn : bool, default=True
        Whether to warn when didn't converge.

    Returns
    -------
    xk : ndarray of float
        Estimated minimum.
    """
    x0 = np.asarray(x0).flatten()
    xk = np.copy(x0)
    k = 0

    if line_search:
        old_fval = func(x0, *args)
        old_old_fval = None

    # Outer loop: our Newton iteration
    while k < maxiter:
        # Compute a search direction pk by applying the CG method to
        #  del2 f(xk) p = - fgrad f(xk) starting from 0.
        fgrad, fhess_p = grad_hess(xk, *args)

        absgrad = np.abs(fgrad)
        if np.max(absgrad) <= tol:
            break

        maggrad = np.sum(absgrad)
        eta = min([0.5, np.sqrt(maggrad)])
        termcond = eta * maggrad

        # Inner loop: solve the Newton update by conjugate gradient, to
        # avoid inverting the Hessian
        xsupi = _cg(fhess_p, fgrad, maxiter=maxinner, tol=termcond)

        alphak = 1.0

        if line_search:
            try:
                alphak, fc, gc, old_fval, old_old_fval, gfkp1 = _line_search_wolfe12(
                    func, grad, xk, xsupi, fgrad, old_fval, old_old_fval, args=args
                )
            except _LineSearchError:
                warnings.warn("Line Search failed")
                break

        xk += alphak * xsupi  # upcast if necessary
        k += 1

    if warn and k >= maxiter:
        warnings.warn(
            "newton-cg failed to converge. Increase the number of iterations.",
            ConvergenceWarning,
        )
    return xk, k


def _check_optimize_result(solver, result, max_iter=None, extra_warning_msg=None):
    """Check the OptimizeResult for successful convergence

    Parameters
    ----------
    solver : str
       Solver name. Currently only `lbfgs` is supported.

    result : OptimizeResult
       Result of the scipy.optimize.minimize function.

    max_iter : int, default=None
       Expected maximum number of iterations.

    extra_warning_msg : str, default=None
        Extra warning message.

    Returns
    -------
    n_iter : int
       Number of iterations.
    """
    # handle both scipy and scikit-learn solver names
    if solver == "lbfgs":
        if result.status != 0:
            try:
                # The message is already decoded in scipy>=1.6.0
                result_message = result.message.decode("latin1")
            except AttributeError:
                result_message = result.message
            warning_msg = (
                "{} failed to converge (status={}):\n{}.\n\n"
                "Increase the number of iterations (max_iter) "
                "or scale the data as shown in:\n"
                "    https://scikit-learn.org/stable/modules/"
                "preprocessing.html"
            ).format(solver, result.status, result_message)
            if extra_warning_msg is not None:
                warning_msg += "\n" + extra_warning_msg
            warnings.warn(warning_msg, ConvergenceWarning, stacklevel=2)
        if max_iter is not None:
            # In scipy <= 1.0.0, nit may exceed maxiter for lbfgs.
            # See https://github.com/scipy/scipy/issues/7854
            n_iter_i = min(result.nit, max_iter)
        else:
            n_iter_i = result.nit
    else:
        raise NotImplementedError

    return n_iter_i
