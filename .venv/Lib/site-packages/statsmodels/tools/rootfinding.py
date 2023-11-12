# -*- coding: utf-8 -*-
"""

Created on Mon Mar 18 15:48:23 2013
Author: Josef Perktold

TODO:
  - test behavior if nans or infs are encountered during the evaluation.
    now partially robust to nans, if increasing can be determined or is given.
  - rewrite core loop to use for...except instead of while.

"""
import numpy as np
from scipy import optimize

from statsmodels.tools.testing import Holder

DEBUG = False


# based on scipy.stats.distributions._ppf_single_call
def brentq_expanding(func, low=None, upp=None, args=(), xtol=1e-5,
                     start_low=None, start_upp=None, increasing=None,
                     max_it=100, maxiter_bq=100, factor=10,
                     full_output=False):
    '''find the root of a function in one variable by expanding and brentq

    Assumes function ``func`` is monotonic.

    Parameters
    ----------
    func : callable
        function for which we find the root ``x`` such that ``func(x) = 0``
    low : float or None
        lower bound for brentq
    upp : float or None
        upper bound for brentq
    args : tuple
        optional additional arguments for ``func``
    xtol : float
        parameter x tolerance given to brentq
    start_low : float (positive) or None
        starting bound for expansion with increasing ``x``. It needs to be
        positive. If None, then it is set to 1.
    start_upp : float (negative) or None
        starting bound for expansion with decreasing ``x``. It needs to be
        negative. If None, then it is set to -1.
    increasing : bool or None
        If None, then the function is evaluated at the initial bounds to
        determine wether the function is increasing or not. If increasing is
        True (False), then it is assumed that the function is monotonically
        increasing (decreasing).
    max_it : int
        maximum number of expansion steps.
    maxiter_bq : int
        maximum number of iterations of brentq.
    factor : float
        expansion factor for step of shifting the bounds interval, default is
        10.
    full_output : bool, optional
        If full_output is False, the root is returned. If full_output is True,
        the return value is (x, r), where x is the root, and r is a
        RootResults object.


    Returns
    -------
    x : float
        root of the function, value at which ``func(x) = 0``.
    info : RootResult (optional)
        returned if ``full_output`` is True.
        attributes:

         - start_bounds : starting bounds for expansion stage
         - brentq_bounds : bounds used with ``brentq``
         - iterations_expand : number of iterations in expansion stage
         - converged : True if brentq converged.
         - flag : return status, 'converged' if brentq converged
         - function_calls : number of function calls by ``brentq``
         - iterations : number of iterations in ``brentq``


    Notes
    -----
    If increasing is None, then whether the function is monotonically
    increasing or decreasing is inferred from evaluating the function at the
    initial bounds. This can fail if there is numerically no variation in the
    data in this range. In this case, using different starting bounds or
    directly specifying ``increasing`` can make it possible to move the
    expansion in the right direction.

    If

    '''
    # TODO: rtol is missing, what does it do?
    left, right = low, upp  # alias

    # start_upp first because of possible sl = -1 > upp
    if upp is not None:
        su = upp
    elif start_upp is not None:
        if start_upp < 0:
            raise ValueError('start_upp needs to be positive')
        su = start_upp
    else:
        su = 1.

    if low is not None:
        sl = low
    elif start_low is not None:
        if start_low > 0:
            raise ValueError('start_low needs to be negative')
        sl = start_low
    else:
        sl = min(-1., su - 1.)

    # need sl < su
    if upp is None:
        su = max(su, sl + 1.)

    # increasing or not ?
    if ((low is None) or (upp is None)) and increasing is None:
        assert sl < su  # check during development
        f_low = func(sl, *args)
        f_upp = func(su, *args)

        # special case for F-distribution (symmetric around zero for effect
        # size)
        # chisquare also takes an indefinite time (did not wait see if it
        # returns)
        if np.max(np.abs(f_upp - f_low)) < 1e-15 and sl == -1 and su == 1:
            sl = 1e-8
            f_low = func(sl, *args)
            increasing = (f_low < f_upp)

        # possibly func returns nan
        delta = su - sl
        if np.isnan(f_low):
            # try just 3 points to find ``increasing``
            # do not change sl because brentq can handle one nan bound
            for fraction in [0.25, 0.5, 0.75]:
                sl_ = sl + fraction * delta
                f_low = func(sl_, *args)
                if not np.isnan(f_low):
                    break
            else:
                raise ValueError('could not determine whether function is ' +
                                 'increasing based on starting interval.' +
                                 '\nspecify increasing or change starting ' +
                                 'bounds')
        if np.isnan(f_upp):
            for fraction in [0.25, 0.5, 0.75]:
                su_ = su + fraction * delta
                f_upp = func(su_, *args)
                if not np.isnan(f_upp):
                    break
            else:
                raise ValueError('could not determine whether function is' +
                                 'increasing based on starting interval.' +
                                 '\nspecify increasing or change starting ' +
                                 'bounds')

        increasing = (f_low < f_upp)

    if not increasing:
        sl, su = su, sl
        left, right = right, left

    n_it = 0
    if left is None and sl != 0:
        left = sl
        while func(left, *args) > 0:
            # condition is also false if func returns nan
            right = left
            left *= factor
            if n_it >= max_it:
                break
            n_it += 1
        # left is now such that func(left) < q
    if right is None and su != 0:
        right = su
        while func(right, *args) < 0:
            left = right
            right *= factor
            if n_it >= max_it:
                break
            n_it += 1
        # right is now such that func(right) > q

    if n_it >= max_it:
        # print('Warning: max_it reached')
        # TODO: use Warnings, Note: brentq might still work even with max_it
        f_low = func(sl, *args)
        f_upp = func(su, *args)
        if np.isnan(f_low) and np.isnan(f_upp):
            # can we still get here?
            raise ValueError('max_it reached' +
                             '\nthe function values at boths bounds are NaN' +
                             '\nchange the starting bounds, set bounds' +
                             'or increase max_it')

    res = optimize.brentq(func, left, right, args=args,
                          xtol=xtol, maxiter=maxiter_bq,
                          full_output=full_output)
    if full_output:
        val = res[0]
        info = Holder(
            # from brentq
            root=res[1].root,
            iterations=res[1].iterations,
            function_calls=res[1].function_calls,
            converged=res[1].converged,
            flag=res[1].flag,
            # ours:
            iterations_expand=n_it,
            start_bounds=(sl, su),
            brentq_bounds=(left, right),
            increasing=increasing,
            )
        return val, info
    else:
        return res
