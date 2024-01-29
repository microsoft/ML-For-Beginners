"""Principal Component Analysis Base Classes"""

# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Mathieu Blondel <mathieu@mblondel.org>
#         Denis A. Engemann <denis-alexander.engemann@inria.fr>
#         Kyle Kastner <kastnerkyle@gmail.com>
#
# License: BSD 3 clause

from abc import ABCMeta, abstractmethod

import numpy as np
from scipy import linalg
from scipy.sparse import issparse

from ..base import BaseEstimator, ClassNamePrefixFeaturesOutMixin, TransformerMixin
from ..utils._array_api import _add_to_diagonal, device, get_namespace
from ..utils.sparsefuncs import _implicit_column_offset
from ..utils.validation import check_is_fitted


class _BasePCA(
    ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator, metaclass=ABCMeta
):
    """Base class for PCA methods.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """

    def get_covariance(self):
        """Compute data covariance with the generative model.

        ``cov = components_.T * S**2 * components_ + sigma2 * eye(n_features)``
        where S**2 contains the explained variances, and sigma2 contains the
        noise variances.

        Returns
        -------
        cov : array of shape=(n_features, n_features)
            Estimated covariance of data.
        """
        xp, _ = get_namespace(self.components_)

        components_ = self.components_
        exp_var = self.explained_variance_
        if self.whiten:
            components_ = components_ * xp.sqrt(exp_var[:, np.newaxis])
        exp_var_diff = exp_var - self.noise_variance_
        exp_var_diff = xp.where(
            exp_var > self.noise_variance_,
            exp_var_diff,
            xp.asarray(0.0, device=device(exp_var)),
        )
        cov = (components_.T * exp_var_diff) @ components_
        _add_to_diagonal(cov, self.noise_variance_, xp)
        return cov

    def get_precision(self):
        """Compute data precision matrix with the generative model.

        Equals the inverse of the covariance but computed with
        the matrix inversion lemma for efficiency.

        Returns
        -------
        precision : array, shape=(n_features, n_features)
            Estimated precision of data.
        """
        xp, is_array_api_compliant = get_namespace(self.components_)

        n_features = self.components_.shape[1]

        # handle corner cases first
        if self.n_components_ == 0:
            return xp.eye(n_features) / self.noise_variance_

        if is_array_api_compliant:
            linalg_inv = xp.linalg.inv
        else:
            linalg_inv = linalg.inv

        if self.noise_variance_ == 0.0:
            return linalg_inv(self.get_covariance())

        # Get precision using matrix inversion lemma
        components_ = self.components_
        exp_var = self.explained_variance_
        if self.whiten:
            components_ = components_ * xp.sqrt(exp_var[:, np.newaxis])
        exp_var_diff = exp_var - self.noise_variance_
        exp_var_diff = xp.where(
            exp_var > self.noise_variance_,
            exp_var_diff,
            xp.asarray(0.0, device=device(exp_var)),
        )
        precision = components_ @ components_.T / self.noise_variance_
        _add_to_diagonal(precision, 1.0 / exp_var_diff, xp)
        precision = components_.T @ linalg_inv(precision) @ components_
        precision /= -(self.noise_variance_**2)
        _add_to_diagonal(precision, 1.0 / self.noise_variance_, xp)
        return precision

    @abstractmethod
    def fit(self, X, y=None):
        """Placeholder for fit. Subclasses should implement this method!

        Fit the model with X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        self : object
            Returns the instance itself.
        """

    def transform(self, X):
        """Apply dimensionality reduction to X.

        X is projected on the first principal components previously extracted
        from a training set.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        Returns
        -------
        X_new : array-like of shape (n_samples, n_components)
            Projection of X in the first principal components, where `n_samples`
            is the number of samples and `n_components` is the number of the components.
        """
        xp, _ = get_namespace(X)

        check_is_fitted(self)

        X = self._validate_data(
            X, accept_sparse=("csr", "csc"), dtype=[xp.float64, xp.float32], reset=False
        )
        if self.mean_ is not None:
            if issparse(X):
                X = _implicit_column_offset(X, self.mean_)
            else:
                X = X - self.mean_
        X_transformed = X @ self.components_.T
        if self.whiten:
            X_transformed /= xp.sqrt(self.explained_variance_)
        return X_transformed

    def inverse_transform(self, X):
        """Transform data back to its original space.

        In other words, return an input `X_original` whose transform would be X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_components)
            New data, where `n_samples` is the number of samples
            and `n_components` is the number of components.

        Returns
        -------
        X_original array-like of shape (n_samples, n_features)
            Original data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        Notes
        -----
        If whitening is enabled, inverse_transform will compute the
        exact inverse operation, which includes reversing whitening.
        """
        xp, _ = get_namespace(X)

        if self.whiten:
            scaled_components = (
                xp.sqrt(self.explained_variance_[:, np.newaxis]) * self.components_
            )
            return X @ scaled_components + self.mean_
        else:
            return X @ self.components_ + self.mean_

    @property
    def _n_features_out(self):
        """Number of transformed output features."""
        return self.components_.shape[0]
