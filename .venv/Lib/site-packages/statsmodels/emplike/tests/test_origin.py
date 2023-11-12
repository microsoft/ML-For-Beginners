import numpy as np
from numpy.testing import assert_almost_equal

from statsmodels.datasets import cancer
from statsmodels.emplike.originregress import ELOriginRegress

from .results.el_results import OriginResults


class GenRes:
    """
    Loads data and creates class instance ot be tested.
    """
    @classmethod
    def setup_class(cls):
        data = cancer.load()
        cls.res1 = ELOriginRegress(data.endog, data.exog).fit()
        cls.res2 = OriginResults()


class TestOrigin(GenRes):
    """
    See OriginResults for details on how tests were computed
    """
    def test_params(self):
        assert_almost_equal(self.res1.params, self.res2.test_params, 4)

    def test_llf(self):
        assert_almost_equal(self.res1.llf_el, self.res2.test_llf_hat, 4)

    def test_hypothesis_beta1(self):
        assert_almost_equal(self.res1.el_test([.0034], [1])[0],
                            self.res2.test_llf_hypoth, 4)

    def test_ci_beta(self):
        ci = self.res1.conf_int_el(1)
        ll = ci[0]
        ul = ci[1]
        llf_low = np.sum(np.log(self.res1.el_test([ll], [1],
                                                  return_weights=1)[2]))
        llf_high = np.sum(np.log(self.res1.el_test([ul], [1],
                                                   return_weights=1)[2]))
        assert_almost_equal(llf_low, self.res2.test_llf_conf, 4)
        assert_almost_equal(llf_high, self.res2.test_llf_conf, 4)
