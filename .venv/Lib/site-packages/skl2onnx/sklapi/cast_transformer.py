# SPDX-License-Identifier: Apache-2.0

import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator

try:
    from sklearn.utils.validation import _deprecate_positional_args
except ImportError:

    def _deprecate_positional_args(x):
        return x  # noqa


class CastTransformer(TransformerMixin, BaseEstimator):

    """
    Cast features into a specific types.
    This should be used to minimize the conversion
    of a pipeline using float32 instead of double.

    Parameters
    ----------
    dtype : numpy type,
        output are cast into that type
    """  # noqa

    @_deprecate_positional_args
    def __init__(self, *, dtype=np.float32):
        self.dtype = dtype

    def _cast(self, a, name):
        if not isinstance(a, np.ndarray):
            if hasattr(a, "values") and hasattr(a, "iloc"):
                # dataframe
                a = a.values
            elif not hasattr(a, "astype"):
                raise TypeError("{} must be a numpy array or a dataframe.".format(name))
        try:
            a2 = a.astype(self.dtype)
        except ValueError:
            raise ValueError(
                "Unable to cast {} from {} into {}.".format(name, a.dtype, self.dtype)
            )
        return a2

    def fit(self, X, y=None, sample_weight=None):
        """
        Does nothing except checking *dtype* may be applied.
        """
        self._cast(X, "X")
        return self

    def transform(self, X, y=None):
        """
        Casts array X.
        """
        return self._cast(X, "X")
