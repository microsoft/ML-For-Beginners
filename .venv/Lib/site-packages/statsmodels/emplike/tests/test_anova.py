import numpy as np
from numpy.testing import assert_almost_equal

from statsmodels.datasets import star98
from statsmodels.emplike.elanova import ANOVA

from .results.el_results import ANOVAResults


class TestANOVA:
    """
    Tests ANOVA difference in means
    """

    @classmethod
    def setup_class(cls):
        cls.data = np.asarray(star98.load().exog)[:30, 1:3]
        cls.res1 = ANOVA([cls.data[:, 0], cls.data[:, 1]])
        cls.res2 = ANOVAResults()

    def test_anova(self):
        assert_almost_equal(self.res1.compute_ANOVA()[:2],
                            self.res2.compute_ANOVA[:2], 4)
        assert_almost_equal(self.res1.compute_ANOVA()[2],
                            self.res2.compute_ANOVA[2], 4)
        assert_almost_equal(self.res1.compute_ANOVA(return_weights=1)[3],
                            self.res2.compute_ANOVA[3], 4)
