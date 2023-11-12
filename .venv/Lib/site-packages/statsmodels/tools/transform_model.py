# -*- coding: utf-8 -*-
"""
Created on Tue May 27 13:23:24 2014

Author: Josef Perktold
License: BSD-3

"""

import numpy as np


class StandardizeTransform:
    """class to reparameterize a model for standardized exog

    Parameters
    ----------
    data : array_like
        data that is standardized along axis=0
    ddof : None or int
        degrees of freedom for calculation of standard deviation.
        default is 1, in contrast to numpy.std
    const_idx : None or int
        If None, then the presence of a constant is detected if the standard
        deviation of a column is **equal** to zero. A constant column is
        not transformed. If this is an integer, then the corresponding column
        will not be transformed.
    demean : bool, default is True
        If demean is true, then the data will be demeaned, otherwise it will
        only be rescaled.

    Notes
    -----
    Warning: Not all options are tested and it is written for one use case.
    API changes are expected.

    This can be used to transform only the design matrix, exog, in a model,
    which is required in some discrete models when the endog cannot be rescaled
    or demeaned.
    The transformation is full rank and does not drop the constant.
    """

    def __init__(self, data, ddof=1, const_idx=None, demean=True):
        data = np.asarray(data)
        self.mean = data.mean(0)
        self.scale = data.std(0, ddof=1)

        # do not transform a constant
        if const_idx is None:
            const_idx = np.nonzero(self.scale == 0)[0]
            if len(const_idx) == 0:
                const_idx = 'n'
            else:
                const_idx = int(np.squeeze(const_idx))

        if const_idx != 'n':
            self.mean[const_idx] = 0
            self.scale[const_idx] = 1

        if demean is False:
            self.mean = None

        self.const_idx = const_idx

    def transform(self, data):
        """standardize the data using the stored transformation
        """
        # could use scipy.stats.zscore instead
        if self.mean is None:
            return np.asarray(data) / self.scale
        else:
            return (np.asarray(data) - self.mean) / self.scale

    def transform_params(self, params):
        """Transform parameters of the standardized model to the original model

        Parameters
        ----------
        params : ndarray
            parameters estimated with the standardized model

        Returns
        -------
        params_new : ndarray
            parameters transformed to the parameterization of the original
            model
        """

        params_new = params / self.scale
        if self.const_idx != 'n':
            params_new[self.const_idx] -= (params_new * self.mean).sum()

        return params_new

    __call__ = transform
