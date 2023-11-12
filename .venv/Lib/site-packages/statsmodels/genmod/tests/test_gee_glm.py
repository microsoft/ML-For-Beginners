import numpy as np
import pandas as pd

from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
from statsmodels.genmod.families import links
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.cov_struct import Independence

from numpy.testing import assert_allclose


class CheckGEEGLM:

    def test_basic(self):
        res1 = self.result1
        res2 = self.result2
        assert_allclose(res1.params.values, res2.params.values,
                        rtol=1e-6, atol=1e-10)


    def test_resid(self):
        res1 = self.result1
        res2 = self.result2
        assert_allclose(res1.resid_response, res2.resid_response,
                        rtol=1e-6, atol=1e-10)
        assert_allclose(res1.resid_pearson, res2.resid_pearson,
                        rtol=1e-6, atol=1e-10)
        assert_allclose(res1.resid_deviance, res2.resid_deviance,
                        rtol=1e-6, atol=1e-10)
        assert_allclose(res1.resid_anscombe, res2.resid_anscombe,
                            rtol=1e-6, atol=1e-10)
        assert_allclose(res1.resid_working, res2.resid_working,
                        rtol=1e-6, atol=1e-10)


#def test_compare_logit(self):
class TestCompareLogit(CheckGEEGLM):

    @classmethod
    def setup_class(cls):
        vs = Independence()
        family = families.Binomial()
        np.random.seed(987126)
        Y = 1 * (np.random.normal(size=100) < 0)
        X1 = np.random.normal(size=100)
        X2 = np.random.normal(size=100)
        X3 = np.random.normal(size=100)
        groups = np.random.randint(0, 4, size=100)

        D = pd.DataFrame({"Y": Y, "X1": X1, "X2": X2, "X3": X3})

        mod1 = GEE.from_formula("Y ~ X1 + X2 + X3", groups, D,
                            family=family, cov_struct=vs)
        cls.result1 = mod1.fit()

        mod2 = GLM.from_formula("Y ~ X1 + X2 + X3", data=D, family=family)
        cls.result2 = mod2.fit(disp=False)


class TestComparePoisson(CheckGEEGLM):

    @classmethod
    def setup_class(cls):
        vs = Independence()
        family = families.Poisson()
        np.random.seed(987126)
        Y = np.exp(1 + np.random.normal(size=100))
        X1 = np.random.normal(size=100)
        X2 = np.random.normal(size=100)
        X3 = np.random.normal(size=100)
        groups = np.random.randint(0, 4, size=100)

        D = pd.DataFrame({"Y": Y, "X1": X1, "X2": X2, "X3": X3})

        mod1 = GEE.from_formula("Y ~ X1 + X2 + X3", groups, D,
                            family=family, cov_struct=vs)
        cls.result1 = mod1.fit()

        mod2 = GLM.from_formula("Y ~ X1 + X2 + X3", data=D, family=family)
        cls.result2 = mod2.fit(disp=False)


class TestCompareGaussian(CheckGEEGLM):

    @classmethod
    def setup_class(cls):

        vs = Independence()
        family = families.Gaussian()
        np.random.seed(987126)
        Y = np.random.normal(size=100)
        X1 = np.random.normal(size=100)
        X2 = np.random.normal(size=100)
        X3 = np.random.normal(size=100)
        groups = np.kron(np.arange(20), np.ones(5))

        D = pd.DataFrame({"Y": Y, "X1": X1, "X2": X2, "X3": X3})

        md = GEE.from_formula("Y ~ X1 + X2 + X3", groups, D,
                              family=family, cov_struct=vs)
        cls.result1 = md.fit()

        cls.result2 = GLM.from_formula("Y ~ X1 + X2 + X3", data=D).fit()


class TestCompareGamma(CheckGEEGLM):

    @classmethod
    def setup_class(cls):
        # adjusted for Gamma, not in test_gee.py
        vs = Independence()
        family = families.Gamma(link=links.Log())
        np.random.seed(987126)
        #Y = np.random.normal(size=100)**2
        Y = np.exp(0.1 + np.random.normal(size=100))   # log-normal
        X1 = np.random.normal(size=100)
        X2 = np.random.normal(size=100)
        X3 = np.random.normal(size=100)
        groups = np.random.randint(0, 4, size=100)

        D = pd.DataFrame({"Y": Y, "X1": X1, "X2": X2, "X3": X3})

        mod1 = GEE.from_formula("Y ~ X1 + X2 + X3", groups, D,
                                family=family, cov_struct=vs)
        cls.result1 = mod1.fit()

        mod2 = GLM.from_formula("Y ~ X1 + X2 + X3", data=D, family=family)
        cls.result2 = mod2.fit(disp=False)
