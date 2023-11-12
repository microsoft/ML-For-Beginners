from typing import Any, Optional
from collections.abc import Mapping

import numpy as np
import pandas as pd


def _right_squeeze(arr, stop_dim=0):
    """
    Remove trailing singleton dimensions

    Parameters
    ----------
    arr : ndarray
        Input array
    stop_dim : int
        Dimension where checking should stop so that shape[i] is not checked
        for i < stop_dim

    Returns
    -------
    squeezed : ndarray
        Array with all trailing singleton dimensions (0 or 1) removed.
        Singleton dimensions for dimension < stop_dim are retained.
    """
    last = arr.ndim
    for s in reversed(arr.shape):
        if s > 1:
            break
        last -= 1
    last = max(last, stop_dim)

    return arr.reshape(arr.shape[:last])


def array_like(
    obj,
    name,
    dtype=np.double,
    ndim=1,
    maxdim=None,
    shape=None,
    order=None,
    contiguous=False,
    optional=False,
    writeable=True,
):
    """
    Convert array-like to a ndarray and check conditions

    Parameters
    ----------
    obj : array_like
         An array, any object exposing the array interface, an object whose
        __array__ method returns an array, or any (nested) sequence.
    name : str
        Name of the variable to use in exceptions
    dtype : {None, numpy.dtype, str}
        Required dtype. Default is double. If None, does not change the dtype
        of obj (if present) or uses NumPy to automatically detect the dtype
    ndim : {int, None}
        Required number of dimensions of obj. If None, no check is performed.
        If the number of dimensions of obj is less than ndim, additional axes
        are inserted on the right. See examples.
    maxdim : {int, None}
        Maximum allowed dimension.  Use ``maxdim`` instead of ``ndim`` when
        inputs are allowed to have ndim 1, 2, ..., or maxdim.
    shape : {tuple[int], None}
        Required shape obj.  If None, no check is performed. Partially
        restricted shapes can be checked using None. See examples.
    order : {'C', 'F', None}
        Order of the array
    contiguous : bool
        Ensure that the array's data is contiguous with order ``order``
    optional : bool
        Flag indicating whether None is allowed
    writeable : bool
        Whether to ensure the returned array is writeable

    Returns
    -------
    ndarray
        The converted input.

    Examples
    --------
    Convert a list or pandas series to an array
    >>> import pandas as pd
    >>> x = [0, 1, 2, 3]
    >>> a = array_like(x, 'x', ndim=1)
    >>> a.shape
    (4,)

    >>> a = array_like(pd.Series(x), 'x', ndim=1)
    >>> a.shape
    (4,)

    >>> type(a.orig)
    pandas.core.series.Series

    Squeezes singleton dimensions when required
    >>> x = np.array(x).reshape((4, 1))
    >>> a = array_like(x, 'x', ndim=1)
    >>> a.shape
    (4,)

    Right-appends when required size is larger than actual
    >>> x = [0, 1, 2, 3]
    >>> a = array_like(x, 'x', ndim=2)
    >>> a.shape
    (4, 1)

    Check only the first and last dimension of the input
    >>> x = np.arange(4*10*4).reshape((4, 10, 4))
    >>> y = array_like(x, 'x', ndim=3, shape=(4, None, 4))

    Check only the first two dimensions
    >>> z = array_like(x, 'x', ndim=3, shape=(4, 10))

    Raises ValueError if constraints are not satisfied
    >>> z = array_like(x, 'x', ndim=2)
    Traceback (most recent call last):
     ...
    ValueError: x is required to have ndim 2 but has ndim 3

    >>> z = array_like(x, 'x', shape=(10, 4, 4))
    Traceback (most recent call last):
     ...
    ValueError: x is required to have shape (10, 4, 4) but has shape (4, 10, 4)

    >>> z = array_like(x, 'x', shape=(None, 4, 4))
    Traceback (most recent call last):
     ...
    ValueError: x is required to have shape (*, 4, 4) but has shape (4, 10, 4)
    """
    if optional and obj is None:
        return None
    reqs = ["W"] if writeable else []
    if order == "C" or contiguous:
        reqs += ["C"]
    elif order == "F":
        reqs += ["F"]
    arr = np.require(obj, dtype=dtype, requirements=reqs)
    if maxdim is not None:
        if arr.ndim > maxdim:
            msg = "{0} must have ndim <= {1}".format(name, maxdim)
            raise ValueError(msg)
    elif ndim is not None:
        if arr.ndim > ndim:
            arr = _right_squeeze(arr, stop_dim=ndim)
        elif arr.ndim < ndim:
            arr = np.reshape(arr, arr.shape + (1,) * (ndim - arr.ndim))
        if arr.ndim != ndim:
            msg = "{0} is required to have ndim {1} but has ndim {2}"
            raise ValueError(msg.format(name, ndim, arr.ndim))
    if shape is not None:
        for actual, req in zip(arr.shape, shape):
            if req is not None and actual != req:
                req_shape = str(shape).replace("None, ", "*, ")
                msg = "{0} is required to have shape {1} but has shape {2}"
                raise ValueError(msg.format(name, req_shape, arr.shape))
    return arr


class PandasWrapper:
    """
    Wrap array_like using the index from the original input, if pandas

    Parameters
    ----------
    pandas_obj : {Series, DataFrame}
        Object to extract the index from for wrapping

    Notes
    -----
    Raises if ``orig`` is a pandas type but obj and and ``orig`` have
    different numbers of elements in axis 0. Also raises if the ndim of obj
    is larger than 2.
    """

    def __init__(self, pandas_obj):
        self._pandas_obj = pandas_obj
        self._is_pandas = isinstance(pandas_obj, (pd.Series, pd.DataFrame))

    def wrap(self, obj, columns=None, append=None, trim_start=0, trim_end=0):
        """
        Parameters
        ----------
        obj : {array_like}
            The value to wrap like to a pandas Series or DataFrame.
        columns : {str, list[str]}
            Column names or series name, if obj is 1d.
        append : str
            String to append to the columns to create a new column name.
        trim_start : int
            The number of observations to drop from the start of the index, so
            that the index applied is index[trim_start:].
        trim_end : int
            The number of observations to drop from the end of the index , so
            that the index applied is index[:nobs - trim_end].

        Returns
        -------
        array_like
            A pandas Series or DataFrame, depending on the shape of obj.
        """
        obj = np.asarray(obj)
        if not self._is_pandas:
            return obj

        if obj.shape[0] + trim_start + trim_end != self._pandas_obj.shape[0]:
            raise ValueError(
                "obj must have the same number of elements in "
                "axis 0 as orig"
            )
        index = self._pandas_obj.index
        index = index[trim_start: index.shape[0] - trim_end]
        if obj.ndim == 1:
            if columns is None:
                name = getattr(self._pandas_obj, "name", None)
            elif isinstance(columns, str):
                name = columns
            else:
                name = columns[0]
            if append is not None:
                name = append if name is None else f"{name}_{append}"

            return pd.Series(obj, name=name, index=index)
        elif obj.ndim == 2:
            if columns is None:
                columns = getattr(self._pandas_obj, "columns", None)
            if append is not None:
                new = []
                for c in columns:
                    new.append(append if c is None else f"{c}_{append}")
                columns = new
            return pd.DataFrame(obj, columns=columns, index=index)
        else:
            raise ValueError("Can only wrap 1 or 2-d array_like")


def bool_like(value, name, optional=False, strict=False):
    """
    Convert to bool or raise if not bool_like

    Parameters
    ----------
    value : object
        Value to verify
    name : str
        Variable name for exceptions
    optional : bool
        Flag indicating whether None is allowed
    strict : bool
        If True, then only allow bool. If False, allow types that support
        casting to bool.

    Returns
    -------
    converted : bool
        value converted to a bool
    """
    if optional and value is None:
        return value
    extra_text = " or None" if optional else ""
    if strict:
        if isinstance(value, bool):
            return value
        else:
            raise TypeError("{0} must be a bool{1}".format(name, extra_text))

    if hasattr(value, "squeeze") and callable(value.squeeze):
        value = value.squeeze()
    try:
        return bool(value)
    except Exception:
        raise TypeError(
            "{0} must be a bool (or bool-compatible)"
            "{1}".format(name, extra_text)
        )


def int_like(
    value: Any, name: str, optional: bool = False, strict: bool = False
) -> Optional[int]:
    """
    Convert to int or raise if not int_like

    Parameters
    ----------
    value : object
        Value to verify
    name : str
        Variable name for exceptions
    optional : bool
        Flag indicating whether None is allowed
    strict : bool
        If True, then only allow int or np.integer that are not bool. If False,
        allow types that support integer division by 1 and conversion to int.

    Returns
    -------
    converted : int
        value converted to a int
    """
    if optional and value is None:
        return None
    is_bool_timedelta = isinstance(value, (bool, np.timedelta64))

    if hasattr(value, "squeeze") and callable(value.squeeze):
        value = value.squeeze()

    if isinstance(value, (int, np.integer)) and not is_bool_timedelta:
        return int(value)
    elif not strict and not is_bool_timedelta:
        try:
            if value == (value // 1):
                return int(value)
        except Exception:
            pass
    extra_text = " or None" if optional else ""
    raise TypeError(
        "{0} must be integer_like (int or np.integer, but not bool"
        " or timedelta64){1}".format(name, extra_text)
    )


def required_int_like(value: Any, name: str, strict: bool = False) -> int:
    """
    Convert to int or raise if not int_like

    Parameters
    ----------
    value : object
        Value to verify
    name : str
        Variable name for exceptions
    optional : bool
        Flag indicating whether None is allowed
    strict : bool
        If True, then only allow int or np.integer that are not bool. If False,
        allow types that support integer division by 1 and conversion to int.

    Returns
    -------
    converted : int
        value converted to a int
    """
    _int = int_like(value, name, optional=False, strict=strict)
    assert _int is not None
    return _int


def float_like(value, name, optional=False, strict=False):
    """
    Convert to float or raise if not float_like

    Parameters
    ----------
    value : object
        Value to verify
    name : str
        Variable name for exceptions
    optional : bool
        Flag indicating whether None is allowed
    strict : bool
        If True, then only allow int, np.integer, float or np.inexact that are
        not bool or complex. If False, allow complex types with 0 imag part or
        any other type that is float like in the sense that it support
        multiplication by 1.0 and conversion to float.

    Returns
    -------
    converted : float
        value converted to a float
    """
    if optional and value is None:
        return None
    is_bool = isinstance(value, bool)
    is_complex = isinstance(value, (complex, np.complexfloating))
    if hasattr(value, "squeeze") and callable(value.squeeze):
        value = value.squeeze()

    if isinstance(value, (int, np.integer, float, np.inexact)) and not (
        is_bool or is_complex
    ):
        return float(value)
    elif not strict and is_complex:
        imag = np.imag(value)
        if imag == 0:
            return float(np.real(value))
    elif not strict and not is_bool:
        try:
            return float(value / 1.0)
        except Exception:
            pass
    extra_text = " or None" if optional else ""
    raise TypeError(
        "{0} must be float_like (float or np.inexact)"
        "{1}".format(name, extra_text)
    )


def string_like(value, name, optional=False, options=None, lower=True):
    """
    Check if object is string-like and raise if not

    Parameters
    ----------
    value : object
        Value to verify.
    name : str
        Variable name for exceptions.
    optional : bool
        Flag indicating whether None is allowed.
    options : tuple[str]
        Allowed values for input parameter `value`.
    lower : bool
        Convert all case-based characters in `value` into lowercase.

    Returns
    -------
    str
        The validated input

    Raises
    ------
    TypeError
        If the value is not a string or None when optional is True.
    ValueError
        If the input is not in ``options`` when ``options`` is set.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        extra_text = " or None" if optional else ""
        raise TypeError("{0} must be a string{1}".format(name, extra_text))
    if lower:
        value = value.lower()
    if options is not None and value not in options:
        extra_text = "If not None, " if optional else ""
        options_text = "'" + "', '".join(options) + "'"
        msg = "{0}{1} must be one of: {2}".format(
            extra_text, name, options_text
        )
        raise ValueError(msg)
    return value


def dict_like(value, name, optional=False, strict=True):
    """
    Check if dict_like (dict, Mapping) or raise if not

    Parameters
    ----------
    value : object
        Value to verify
    name : str
        Variable name for exceptions
    optional : bool
        Flag indicating whether None is allowed
    strict : bool
        If True, then only allow dict. If False, allow any Mapping-like object.

    Returns
    -------
    converted : dict_like
        value
    """
    if optional and value is None:
        return None
    if not isinstance(value, Mapping) or (
        strict and not (isinstance(value, dict))
    ):
        extra_text = "If not None, " if optional else ""
        strict_text = " or dict_like (i.e., a Mapping)" if strict else ""
        msg = "{0}{1} must be a dict{2}".format(extra_text, name, strict_text)
        raise TypeError(msg)
    return value
