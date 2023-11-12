import numpy as np
from numpy.testing import assert_almost_equal
import pytest

from statsmodels.datasets import heart
from statsmodels.emplike.aft_el import emplikeAFT
from statsmodels.tools import add_constant

from .results.el_results import AFTRes


class GenRes:
    @classmethod
    def setup_class(cls):
        data = heart.load()
        data.endog = np.asarray(data.endog)
        data.exog = np.asarray(data.exog)
        endog = np.log10(data.endog)
        exog = add_constant(data.exog)
        cls.mod1 = emplikeAFT(endog, exog, data.censors)
        cls.res1 = cls.mod1.fit()
        cls.res2 = AFTRes()


class Test_AFTModel(GenRes):

    def test_params(self):
        assert_almost_equal(self.res1.params(), self.res2.test_params,
                            decimal=4)

    def test_beta0(self):
        assert_almost_equal(self.res1.test_beta([4], [0]),
                            self.res2.test_beta0, decimal=4)

    def test_beta1(self):
        assert_almost_equal(self.res1.test_beta([-.04], [1]),
                            self.res2.test_beta1, decimal=4)

    def test_beta_vect(self):
        assert_almost_equal(self.res1.test_beta([3.5, -.035], [0, 1]),
                            self.res2.test_joint, decimal=4)

    @pytest.mark.slow
    def test_betaci(self):
        ci = self.res1.ci_beta(1, -.06, 0)
        ll = ci[0]
        ul = ci[1]
        ll_pval = self.res1.test_beta([ll], [1])[1]
        ul_pval = self.res1.test_beta([ul], [1])[1]
        assert_almost_equal(ul_pval, .050000, decimal=4)
        assert_almost_equal(ll_pval, .05000, decimal=4)
