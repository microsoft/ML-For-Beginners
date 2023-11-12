# -*- coding: utf-8 -*-
"""
Cross-validation classes for GAM

Author: Luca Puggini

"""

from abc import ABCMeta, abstractmethod
from statsmodels.compat.python import with_metaclass
import itertools
import numpy as np
from statsmodels.gam.smooth_basis import (GenericSmoothers,
                                          UnivariateGenericSmoother)


class BaseCV(with_metaclass(ABCMeta)):
    """
    BaseCV class. It computes the cross validation error of a given model.
    All the cross validation classes can be derived by this one
    (e.g. GamCV, LassoCV,...)
    """

    def __init__(self, cv_iterator, endog, exog):
        self.cv_iterator = cv_iterator
        self.exog = exog
        self.endog = endog
        # TODO: cv_iterator.split only needs nobs from endog or exog
        self.train_test_cv_indices = self.cv_iterator.split(self.exog,
                                                            self.endog,
                                                            label=None)

    def fit(self, **kwargs):
        # kwargs are the input values for the fit method of the
        # cross-validated object

        cv_err = []

        for train_index, test_index in self.train_test_cv_indices:
            cv_err.append(self._error(train_index, test_index, **kwargs))

        return np.array(cv_err)

    @abstractmethod
    def _error(self, train_index, test_index, **kwargs):
        # train the model on the train set
        #   and returns the error on the test set
        pass


def _split_train_test_smoothers(x, smoother, train_index, test_index):
    """split smoothers in test and train sets and create GenericSmoothers

    Note: this does not take exog_linear into account
    """
    train_smoothers = []
    test_smoothers = []
    for smoother in smoother.smoothers:
        train_basis = smoother.basis[train_index]
        train_der_basis = smoother.der_basis[train_index]
        train_der2_basis = smoother.der2_basis[train_index]
        train_cov_der2 = smoother.cov_der2
        # TODO: Double check this part. cov_der2 is calculated with all data
        train_x = smoother.x[train_index]

        train_smoothers.append(
            UnivariateGenericSmoother(
                train_x, train_basis, train_der_basis, train_der2_basis,
                train_cov_der2, smoother.variable_name + ' train'))

        test_basis = smoother.basis[test_index]
        test_der_basis = smoother.der_basis[test_index]
        test_cov_der2 = smoother.cov_der2
        # TODO: Double check this part. cov_der2 is calculated with all data
        test_x = smoother.x[test_index]

        test_smoothers.append(
            UnivariateGenericSmoother(
                test_x, test_basis, test_der_basis, train_der2_basis,
                test_cov_der2, smoother.variable_name + ' test'))

    train_multivariate_smoothers = GenericSmoothers(x[train_index],
                                                    train_smoothers)
    test_multivariate_smoothers = GenericSmoothers(x[test_index],
                                                   test_smoothers)

    return train_multivariate_smoothers, test_multivariate_smoothers


class MultivariateGAMCV(BaseCV):
    def __init__(self, smoother, alphas, gam, cost, endog, exog, cv_iterator):
        self.cost = cost
        self.gam = gam
        self.smoother = smoother
        self.exog_linear = exog
        self.alphas = alphas
        self.cv_iterator = cv_iterator
        # TODO: super does not do anything with endog, exog, except get nobs
        # refactor to clean up what where `exog` and `exog_linear` is attached
        super(MultivariateGAMCV, self).__init__(cv_iterator,
                                                endog,
                                                # exog,  # not used in super
                                                self.smoother.basis)

    def _error(self, train_index, test_index, **kwargs):
        train_smoother, test_smoother = _split_train_test_smoothers(
            self.smoother.x, self.smoother, train_index, test_index)

        endog_train = self.endog[train_index]
        endog_test = self.endog[test_index]
        if self.exog_linear is not None:
            exog_linear_train = self.exog_linear[train_index]
            exog_linear_test = self.exog_linear[test_index]
        else:
            exog_linear_train = None
            exog_linear_test = None

        gam = self.gam(endog_train, exog=exog_linear_train,
                       smoother=train_smoother, alpha=self.alphas)
        gam_res = gam.fit(**kwargs)
        # exog_linear_test and test_smoother.basis will be column_stacked
        #     but not transformed in predict
        endog_est = gam_res.predict(exog_linear_test, test_smoother.basis,
                                    transform=False)

        return self.cost(endog_test, endog_est)


class BasePenaltiesPathCV(with_metaclass(ABCMeta)):
    """
    Base class for cross validation over a grid of parameters.

    The best parameter is saved in alpha_cv

    This class is currently not used
    """

    def __init__(self, alphas):
        self.alphas = alphas
        self.alpha_cv = None
        self.cv_error = None
        self.cv_std = None

    def plot_path(self):
        from statsmodels.graphics.utils import _import_mpl
        plt = _import_mpl()
        plt.plot(self.alphas, self.cv_error, c='black')
        plt.plot(self.alphas, self.cv_error + 1.96 * self.cv_std,
                 c='blue')
        plt.plot(self.alphas, self.cv_error - 1.96 * self.cv_std,
                 c='blue')

        plt.plot(self.alphas, self.cv_error, 'o', c='black')
        plt.plot(self.alphas, self.cv_error + 1.96 * self.cv_std, 'o',
                 c='blue')
        plt.plot(self.alphas, self.cv_error - 1.96 * self.cv_std, 'o',
                 c='blue')

        return
        # TODO add return


class MultivariateGAMCVPath:
    """k-fold cross-validation for GAM

    Warning: The API of this class is preliminary and will change.

    Parameters
    ----------
    smoother : additive smoother instance
    alphas : list of iteratables
        list of alpha for smooths. The product space will be used as alpha
        grid for cross-validation
    gam : model class
        model class for creating a model with k-fole training data
    cost : function
        cost function for the prediction error
    endog : ndarray
        dependent (response) variable of the model
    cv_iterator : instance of cross-validation iterator
    """

    def __init__(self, smoother, alphas, gam, cost, endog, exog, cv_iterator):
        self.cost = cost
        self.smoother = smoother
        self.gam = gam
        self.alphas = alphas
        self.alphas_grid = list(itertools.product(*self.alphas))
        self.endog = endog
        self.exog = exog
        self.cv_iterator = cv_iterator
        self.cv_error = np.zeros(shape=(len(self.alphas_grid, )))
        self.cv_std = np.zeros(shape=(len(self.alphas_grid, )))
        self.alpha_cv = None

    def fit(self, **kwargs):
        for i, alphas_i in enumerate(self.alphas_grid):
            gam_cv = MultivariateGAMCV(smoother=self.smoother,
                                       alphas=alphas_i,
                                       gam=self.gam,
                                       cost=self.cost,
                                       endog=self.endog,
                                       exog=self.exog,
                                       cv_iterator=self.cv_iterator)
            cv_err = gam_cv.fit(**kwargs)
            self.cv_error[i] = cv_err.mean()
            self.cv_std[i] = cv_err.std()

        self.alpha_cv = self.alphas_grid[np.argmin(self.cv_error)]
        return self
