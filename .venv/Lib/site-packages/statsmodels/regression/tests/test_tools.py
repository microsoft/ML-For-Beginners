import numpy as np
from numpy.testing import assert_allclose
import pytest

from statsmodels.regression.linear_model import WLS
from statsmodels.regression._tools import _MinimalWLS


class TestMinimalWLS:
    @classmethod
    def setup_class(cls):
        rs = np.random.RandomState(1234)
        cls.exog1 = rs.randn(200, 5)
        cls.endog1 = cls.exog1.sum(1) + rs.randn(200)
        cls.weights1 = 1.0 + np.sin(np.arange(200.0) / 100.0 * np.pi)
        cls.exog2 = rs.randn(50, 1)
        cls.endog2 = 0.3 * cls.exog2.ravel() + rs.randn(50)
        cls.weights2 = 1.0 + np.log(np.arange(1.0, 51.0))

    @pytest.mark.parametrize('check', [True, False])
    def test_equivalence_with_wls(self, check):
        res = WLS(self.endog1, self.exog1).fit()
        minres = _MinimalWLS(self.endog1, self.exog1,
                             check_endog=check, check_weights=check).fit()
        assert_allclose(res.params, minres.params)
        assert_allclose(res.resid, minres.resid)

        res = WLS(self.endog2, self.exog2).fit()
        minres = _MinimalWLS(self.endog2, self.exog2,
                             check_endog=check, check_weights=check).fit()
        assert_allclose(res.params, minres.params)
        assert_allclose(res.resid, minres.resid)

        res = WLS(self.endog1, self.exog1, weights=self.weights1).fit()
        minres = _MinimalWLS(self.endog1, self.exog1, weights=self.weights1,
                             check_endog=check, check_weights=check).fit()
        assert_allclose(res.params, minres.params)
        assert_allclose(res.resid, minres.resid)

        res = WLS(self.endog2, self.exog2, weights=self.weights2).fit()
        minres = _MinimalWLS(self.endog2, self.exog2, weights=self.weights2,
                             check_endog=check, check_weights=check).fit()
        assert_allclose(res.params, minres.params)
        assert_allclose(res.resid, minres.resid)

    @pytest.mark.parametrize('bad_value', [np.nan, np.inf])
    def test_inf_nan(self, bad_value):
        with pytest.raises(
                ValueError,
                match=r'detected in endog, estimation infeasible'):
            endog = self.endog1.copy()
            endog[0] = bad_value
            _MinimalWLS(endog, self.exog1, self.weights1,
                        check_endog=True, check_weights=True).fit()

        with pytest.raises(
                ValueError,
                match=r'detected in weights, estimation infeasible'):
            weights = self.weights1.copy()
            weights[-1] = bad_value
            _MinimalWLS(self.endog1, self.exog1, weights,
                        check_endog=True, check_weights=True).fit()
