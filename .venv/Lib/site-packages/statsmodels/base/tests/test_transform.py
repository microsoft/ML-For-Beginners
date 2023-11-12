import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal, assert_raises)
from statsmodels.base.transform import (BoxCox)
from statsmodels.datasets import macrodata


class TestTransform:

    @classmethod
    def setup_class(cls):
        data = macrodata.load_pandas()
        cls.x = data.data['realgdp'].values
        cls.bc = BoxCox()

    def test_nonpositive(self):
        # Testing negative values
        y = [1, -1, 1]
        assert_raises(ValueError, self.bc.transform_boxcox, y)

        # Testing nonzero
        y = [1, 0, 1]
        assert_raises(ValueError, self.bc.transform_boxcox, y)

    def test_invalid_bounds(self):
        # more than two bounds
        assert_raises(ValueError, self.bc._est_lambda, self.x, (-3, 2, 3))

        # upper bound <= lower bound
        assert_raises(ValueError, self.bc._est_lambda, self.x, (2, -1))

    def test_unclear_methods(self):
        # Both _est_lambda and untransform have a method argument that should
        # be tested.
        assert_raises(ValueError, self.bc._est_lambda,
                      self.x, (-1, 2), 'test')
        assert_raises(ValueError, self.bc.untransform_boxcox,
                      self.x, 1, 'test')

    def test_unclear_scale_parameter(self):
        # bc.guerrero allows for 'mad' and 'sd', for the MAD and Standard
        # Deviation, respectively
        assert_raises(ValueError, self.bc._est_lambda,
                      self.x, scale='test')

        # Next, check if mad/sd work:
        self.bc._est_lambda(self.x, scale='mad')
        self.bc._est_lambda(self.x, scale='MAD')

        self.bc._est_lambda(self.x, scale='sd')
        self.bc._est_lambda(self.x, scale='SD')

    def test_valid_guerrero(self):
        # `l <- BoxCox.lambda(x, method="guerrero")` on a ts object
        # with frequency 4 (BoxCox.lambda defaults to 2, but we use
        # Guerrero and Perera (2004) as a guideline)
        lmbda = self.bc._est_lambda(self.x, method='guerrero', window_length=4)
        assert_almost_equal(lmbda, 0.507624, 4)

        # `l <- BoxCox.lambda(x, method="guerrero")` with the default grouping
        # parameter (namely, window_length=2).
        lmbda = self.bc._est_lambda(self.x, method='guerrero', window_length=2)
        assert_almost_equal(lmbda, 0.513893, 4)

    def test_guerrero_robust_scale(self):
        # The lambda is derived from a manual check of the values for the MAD.
        # Compare also the result for the standard deviation on R=4: 0.5076,
        # i.e. almost the same value.
        lmbda = self.bc._est_lambda(self.x, scale='mad')
        assert_almost_equal(lmbda, 0.488621, 4)

    def test_loglik_lambda_estimation(self):
        # 0.2 is the value returned by `BoxCox.lambda(x, method="loglik")`
        lmbda = self.bc._est_lambda(self.x, method='loglik')
        assert_almost_equal(lmbda, 0.2, 1)

    def test_boxcox_transformation_methods(self):
        # testing estimated lambda vs. provided. Should result in almost
        # the same transformed data. Value taken from R.
        y_transformed_no_lambda = self.bc.transform_boxcox(self.x)
        y_transformed_lambda = self.bc.transform_boxcox(self.x, 0.507624)

        assert_almost_equal(y_transformed_no_lambda[0],
                            y_transformed_lambda[0], 3)

        # a perfectly increasing set has a constant variance over the entire
        # series, hence stabilising should result in the same scale: lmbda = 1.
        y, lmbda = self.bc.transform_boxcox(np.arange(1, 100))
        assert_almost_equal(lmbda, 1., 5)

    def test_zero_lambda(self):
        # zero lambda should be a log transform.
        y_transform_zero_lambda, lmbda = self.bc.transform_boxcox(self.x, 0.)

        assert_equal(lmbda, 0.)
        assert_almost_equal(y_transform_zero_lambda, np.log(self.x), 5)

    def test_naive_back_transformation(self):
        # test both transformations functions -> 0. and .5
        y_zero_lambda = self.bc.transform_boxcox(self.x, 0.)
        y_half_lambda = self.bc.transform_boxcox(self.x, .5)

        y_zero_lambda_un = self.bc.untransform_boxcox(*y_zero_lambda,
                                                      method='naive')
        y_half_lambda_un = self.bc.untransform_boxcox(*y_half_lambda,
                                                      method='naive')

        assert_almost_equal(self.x, y_zero_lambda_un, 5)
        assert_almost_equal(self.x, y_half_lambda_un, 5)
