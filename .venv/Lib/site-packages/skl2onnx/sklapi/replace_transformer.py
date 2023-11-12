# SPDX-License-Identifier: Apache-2.0

import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator

try:
    from sklearn.utils.validation import _deprecate_positional_args
except ImportError:

    def _deprecate_positional_args(x):
        return x  # noqa


class ReplaceTransformer(TransformerMixin, BaseEstimator):

    """
    Replaces a value by another one.
    It can be used to replace 0 by nan.

    Parameters
    ----------
    from_value : value to replace
    to_value : new value
    dtype: dtype of replaced values
    """  # noqa

    @_deprecate_positional_args
    def __init__(self, *, from_value=0, to_value=np.nan, dtype=np.float32):
        BaseEstimator.__init__(self)
        self.dtype = dtype
        self.from_value = from_value
        self.to_value = to_value

    def _replace(self, a):
        if hasattr(a, "todense"):
            if np.isnan(self.to_value) and self.from_value == 0:
                # implicit
                return a
            raise RuntimeError(
                "Unable to replace 0 by nan one value by another " "in sparse matrix."
            )
        return np.where(a == self.from_value, self.to_value, a)

    def fit(self, X, y=None, sample_weight=None):
        """
        Does nothing except checking *dtype* may be applied.
        """
        self._replace(X)
        return self

    def transform(self, X, y=None):
        """
        Casts array X.
        """
        return self._replace(X)
