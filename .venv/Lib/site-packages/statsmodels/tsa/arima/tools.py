"""
SARIMAX tools.

Author: Chad Fulton
License: BSD-3
"""
import numpy as np


def standardize_lag_order(order, title=None):
    """
    Standardize lag order input.

    Parameters
    ----------
    order : int or array_like
        Maximum lag order (if integer) or iterable of specific lag orders.
    title : str, optional
        Description of the order (e.g. "autoregressive") to use in error
        messages.

    Returns
    -------
    order : int or list of int
        Maximum lag order if consecutive lag orders were specified, otherwise
        a list of integer lag orders.

    Notes
    -----
    It is ambiguous if order=[1] is meant to be a boolean list or
    a list of lag orders to include, but this is irrelevant because either
    interpretation gives the same result.

    Order=[0] would be ambiguous, except that 0 is not a valid lag
    order to include, so there is no harm in interpreting as a boolean
    list, in which case it is the same as order=0, which seems like
    reasonable behavior.

    Examples
    --------
    >>> standardize_lag_order(3)
    3
    >>> standardize_lag_order(np.arange(1, 4))
    3
    >>> standardize_lag_order([1, 3])
    [1, 3]
    """
    order = np.array(order)
    title = 'order' if title is None else '%s order' % title

    # Only integer orders are valid
    if not np.all(order == order.astype(int)):
        raise ValueError('Invalid %s. Non-integer order (%s) given.'
                         % (title, order))
    order = order.astype(int)

    # Only positive integers are valid
    if np.any(order < 0):
        raise ValueError('Terms in the %s cannot be negative.' % title)

    # Try to squeeze out an irrelevant trailing dimension
    if order.ndim == 2 and order.shape[1] == 1:
        order = order[:, 0]
    elif order.ndim > 1:
        raise ValueError('Invalid %s. Must be an integer or'
                         ' 1-dimensional array-like object (e.g. list,'
                         ' ndarray, etc.). Got %s.' % (title, order))

    # Option 1: the typical integer response (implies including all
    # lags up through and including the value)
    if order.ndim == 0:
        order = order.item()
    elif len(order) == 0:
        order = 0
    else:
        # Option 2: boolean list
        has_zeros = (0 in order)
        has_multiple_ones = np.sum(order == 1) > 1
        has_gt_one = np.any(order > 1)
        if has_zeros or has_multiple_ones:
            if has_gt_one:
                raise ValueError('Invalid %s. Appears to be a boolean list'
                                 ' (since it contains a 0 element and/or'
                                 ' multiple elements) but also contains'
                                 ' elements greater than 1 like a list of'
                                 ' lag orders.' % title)
            order = (np.where(order == 1)[0] + 1)

        # (Default) Option 3: list of lag orders to include
        else:
            order = np.sort(order)

        # If we have an empty list, set order to zero
        if len(order) == 0:
            order = 0
        # If we actually were given consecutive lag orders, just use integer
        elif np.all(order == np.arange(1, len(order) + 1)):
            order = order[-1]
        # Otherwise, convert to list
        else:
            order = order.tolist()

    # Check for duplicates
    has_duplicate = isinstance(order, list) and np.any(np.diff(order) == 0)
    if has_duplicate:
        raise ValueError('Invalid %s. Cannot have duplicate elements.' % title)

    return order


def validate_basic(params, length, allow_infnan=False, title=None):
    """
    Validate parameter vector for basic correctness.

    Parameters
    ----------
    params : array_like
        Array of parameters to validate.
    length : int
        Expected length of the parameter vector.
    allow_infnan : bool, optional
            Whether or not to allow `params` to contain -np.Inf, np.Inf, and
            np.nan. Default is False.
    title : str, optional
        Description of the parameters (e.g. "autoregressive") to use in error
        messages.

    Returns
    -------
    params : ndarray
        Array of validated parameters.

    Notes
    -----
    Basic check that the parameters are numeric and that they are the right
    shape. Optionally checks for NaN / infinite values.
    """
    title = '' if title is None else ' for %s' % title

    # Check for invalid type and coerce to non-integer
    try:
        params = np.array(params, dtype=object)
        is_complex = [isinstance(p, complex) for p in params.ravel()]
        dtype = complex if any(is_complex) else float
        params = np.array(params, dtype=dtype)
    except TypeError:
        raise ValueError('Parameters vector%s includes invalid values.'
                         % title)

    # Check for NaN, inf
    if not allow_infnan and (np.any(np.isnan(params)) or
                             np.any(np.isinf(params))):
        raise ValueError('Parameters vector%s includes NaN or Inf values.'
                         % title)

    params = np.atleast_1d(np.squeeze(params))

    # Check for right number of parameters
    if params.shape != (length,):
        plural = '' if length == 1 else 's'
        raise ValueError('Specification%s implies %d parameter%s, but'
                         ' values with shape %s were provided.'
                         % (title, length, plural, params.shape))

    return params
