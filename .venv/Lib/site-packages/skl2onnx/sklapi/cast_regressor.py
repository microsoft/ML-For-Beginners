# SPDX-License-Identifier: Apache-2.0

import numpy as np
from sklearn.base import RegressorMixin, BaseEstimator

try:
    from sklearn.utils.validation import _deprecate_positional_args
except ImportError:

    def _deprecate_positional_args(x):
        return x  # noqa


class CastRegressor(RegressorMixin, BaseEstimator):  # noqa

    """
    Cast predictions into a specific types.
    This should be used to minimize the conversion
    of a pipeline using float32 instead of double
    when onnx do not support double.

    Parameters
    ----------
    estimator : regressor
        wrapped regressor
    dtype : numpy type,
        output are cast into that type
    """  # noqa

    @_deprecate_positional_args
    def __init__(self, estimator, *, dtype=np.float32):
        self.dtype = dtype
        self.estimator = estimator

    def _cast(self, a, name):
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
        self.estimator.fit(X, y=y, sample_weight=sample_weight)
        return self

    def predict(self, X, y=None):
        """
        Predicts and casts the prediction.
        """
        return self._cast(self.estimator.predict(X), "predict(X)")

    def decision_function(self, X, y=None):
        """
        Calls *decision_function* and casts the outputs.
        """
        if not hasattr(self.estimator, "decision_function"):
            raise AttributeError(
                "%r object has no attribute 'decision_function'."
                % self.estimator.__class__.__name__
            )
        return self._cast(self.estimator.decision_function(X), "decision_function(X)")
