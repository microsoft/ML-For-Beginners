"""
Holds common functions for l1 solvers.
"""

import numpy as np

from statsmodels.tools.sm_exceptions import ConvergenceWarning


def qc_results(params, alpha, score, qc_tol, qc_verbose=False):
    """
    Theory dictates that one of two conditions holds:
        i) abs(score[i]) == alpha[i]  and  params[i] != 0
        ii) abs(score[i]) <= alpha[i]  and  params[i] == 0
    qc_results checks to see that (ii) holds, within qc_tol

    qc_results also checks for nan or results of the wrong shape.

    Parameters
    ----------
    params : ndarray
        model parameters.  Not including the added variables x_added.
    alpha : ndarray
        regularization coefficients
    score : function
        Gradient of unregularized objective function
    qc_tol : float
        Tolerance to hold conditions (i) and (ii) to for QC check.
    qc_verbose : bool
        If true, print out a full QC report upon failure

    Returns
    -------
    passed : bool
        True if QC check passed
    qc_dict : Dictionary
        Keys are fprime, alpha, params, passed_array

    Prints
    ------
    Warning message if QC check fails.
    """
    ## Check for fatal errors
    assert not np.isnan(params).max()
    assert (params == params.ravel('F')).min(), \
        "params should have already been 1-d"

    ## Start the theory compliance check
    fprime = score(params)
    k_params = len(params)

    passed_array = np.array([True] * k_params)
    for i in range(k_params):
        if alpha[i] > 0:
            # If |fprime| is too big, then something went wrong
            if (abs(fprime[i]) - alpha[i]) / alpha[i] > qc_tol:
                passed_array[i] = False
    qc_dict = dict(
        fprime=fprime, alpha=alpha, params=params, passed_array=passed_array)
    passed = passed_array.min()
    if not passed:
        num_failed = (~passed_array).sum()
        message = 'QC check did not pass for %d out of %d parameters' % (
            num_failed, k_params)
        message += '\nTry increasing solver accuracy or number of iterations'\
            ', decreasing alpha, or switch solvers'
        if qc_verbose:
            message += _get_verbose_addon(qc_dict)

        import warnings
        warnings.warn(message, ConvergenceWarning)

    return passed


def _get_verbose_addon(qc_dict):
    alpha = qc_dict['alpha']
    params = qc_dict['params']
    fprime = qc_dict['fprime']
    passed_array = qc_dict['passed_array']

    addon = '\n------ verbose QC printout -----------------'
    addon = '\n------ Recall the problem was rescaled by 1 / nobs ---'
    addon += '\n|%-10s|%-10s|%-10s|%-10s|' % (
        'passed', 'alpha', 'fprime', 'param')
    addon += '\n--------------------------------------------'
    for i in range(len(alpha)):
        addon += '\n|%-10s|%-10.3e|%-10.3e|%-10.3e|' % (
                passed_array[i], alpha[i], fprime[i], params[i])
    return addon


def do_trim_params(params, k_params, alpha, score, passed, trim_mode,
        size_trim_tol, auto_trim_tol):
    """
    Trims (set to zero) params that are zero at the theoretical minimum.
    Uses heuristics to account for the solver not actually finding the minimum.

    In all cases, if alpha[i] == 0, then do not trim the ith param.
    In all cases, do nothing with the added variables.

    Parameters
    ----------
    params : ndarray
        model parameters.  Not including added variables.
    k_params : Int
        Number of parameters
    alpha : ndarray
        regularization coefficients
    score : Function.
        score(params) should return a 1-d vector of derivatives of the
        unpenalized objective function.
    passed : bool
        True if the QC check passed
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

    Returns
    -------
    params : ndarray
        Trimmed model parameters
    trimmed : ndarray of booleans
        trimmed[i] == True if the ith parameter was trimmed.
    """
    ## Trim the small params
    trimmed = [False] * k_params

    if trim_mode == 'off':
        trimmed = np.array([False] * k_params)
    elif trim_mode == 'auto' and not passed:
        import warnings
        msg = "Could not trim params automatically due to failed QC check. " \
              "Trimming using trim_mode == 'size' will still work."
        warnings.warn(msg, ConvergenceWarning)
        trimmed = np.array([False] * k_params)
    elif trim_mode == 'auto' and passed:
        fprime = score(params)
        for i in range(k_params):
            if alpha[i] != 0:
                if (alpha[i] - abs(fprime[i])) / alpha[i] > auto_trim_tol:
                    params[i] = 0.0
                    trimmed[i] = True
    elif trim_mode == 'size':
        for i in range(k_params):
            if alpha[i] != 0:
                if abs(params[i]) < size_trim_tol:
                    params[i] = 0.0
                    trimmed[i] = True
    else:
        raise ValueError(
            "trim_mode == %s, which is not recognized" % (trim_mode))

    return params, np.asarray(trimmed)
