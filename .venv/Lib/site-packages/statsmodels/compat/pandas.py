from typing import Optional

import numpy as np
from packaging.version import Version, parse
import pandas as pd
from pandas.util._decorators import (
    Appender,
    Substitution,
    cache_readonly,
    deprecate_kwarg,
)

__all__ = [
    "assert_frame_equal",
    "assert_index_equal",
    "assert_series_equal",
    "data_klasses",
    "frequencies",
    "is_numeric_dtype",
    "testing",
    "cache_readonly",
    "deprecate_kwarg",
    "Appender",
    "Substitution",
    "is_int_index",
    "is_float_index",
    "make_dataframe",
    "to_numpy",
    "PD_LT_1_0_0",
    "get_cached_func",
    "get_cached_doc",
    "call_cached_func",
    "PD_LT_1_4",
    "PD_LT_2",
]

version = parse(pd.__version__)

PD_LT_1_0_0 = version < Version("1.0.0")
PD_LT_1_4 = version < Version("1.3.99")
PD_LT_2 = version < Version("1.9.99")

try:
    from pandas.api.types import is_numeric_dtype
except ImportError:
    from pandas.core.common import is_numeric_dtype

try:
    from pandas.tseries import offsets as frequencies
except ImportError:
    from pandas.tseries import frequencies

data_klasses = (pd.Series, pd.DataFrame)

try:
    import pandas.testing as testing
except ImportError:
    import pandas.util.testing as testing

assert_frame_equal = testing.assert_frame_equal
assert_index_equal = testing.assert_index_equal
assert_series_equal = testing.assert_series_equal


def is_int_index(index: pd.Index) -> bool:
    """
    Check if an index is integral

    Parameters
    ----------
    index : pd.Index
        Any numeric index

    Returns
    -------
    bool
        True if is an index with a standard integral type
    """
    return (
        isinstance(index, pd.Index)
        and isinstance(index.dtype, np.dtype)
        and np.issubdtype(index.dtype, np.integer)
    )


def is_float_index(index: pd.Index) -> bool:
    """
    Check if an index is floating

    Parameters
    ----------
    index : pd.Index
        Any numeric index

    Returns
    -------
    bool
        True if an index with a standard numpy floating dtype
    """
    return (
        isinstance(index, pd.Index)
        and isinstance(index.dtype, np.dtype)
        and np.issubdtype(index.dtype, np.floating)
    )


try:
    from pandas._testing import makeDataFrame as make_dataframe
except ImportError:
    import string

    def rands_array(nchars, size, dtype="O"):
        """
        Generate an array of byte strings.
        """
        rands_chars = np.array(
            list(string.ascii_letters + string.digits), dtype=(np.str_, 1)
        )
        retval = (
            np.random.choice(rands_chars, size=nchars * np.prod(size))
            .view((np.str_, nchars))
            .reshape(size)
        )
        if dtype is None:
            return retval
        else:
            return retval.astype(dtype)

    def make_dataframe():
        """
        Simple verion of pandas._testing.makeDataFrame
        """
        n = 30
        k = 4
        index = pd.Index(rands_array(nchars=10, size=n), name=None)
        data = {
            c: pd.Series(np.random.randn(n), index=index)
            for c in string.ascii_uppercase[:k]
        }

        return pd.DataFrame(data)


def to_numpy(po: pd.DataFrame) -> np.ndarray:
    """
    Workaround legacy pandas lacking to_numpy

    Parameters
    ----------
    po : Pandas obkect

    Returns
    -------
    ndarray
        A numpy array
    """
    try:
        return po.to_numpy()
    except AttributeError:
        return po.values


def get_cached_func(cached_prop):
    try:
        return cached_prop.fget
    except AttributeError:
        return cached_prop.func


def call_cached_func(cached_prop, *args, **kwargs):
    f = get_cached_func(cached_prop)
    return f(*args, **kwargs)


def get_cached_doc(cached_prop) -> Optional[str]:
    return get_cached_func(cached_prop).__doc__
