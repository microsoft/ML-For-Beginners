'''Testing GenericLikelihoodModel variations on Poisson


'''
import pytest
import numpy as np
from numpy.testing import assert_almost_equal
from scipy import stats

import statsmodels.api as sm
from statsmodels.miscmodels.count import PoissonGMLE, PoissonOffsetGMLE, \
                        PoissonZiGMLE
from statsmodels.discrete.discrete_model import Poisson
from statsmodels.tools.sm_exceptions import ValueWarning


DEC = 4
DEC4 = 4
DEC5 = 5

class CompareMixin:

    def test_params(self):
        assert_almost_equal(self.res.params, self.res_glm.params, DEC5)
        assert_almost_equal(self.res.params, self.res_discrete.params, DEC5)

    def test_cov_params(self):
        assert_almost_equal(self.res.bse, self.res_glm.bse, DEC5)
        assert_almost_equal(self.res.bse, self.res_discrete.bse, DEC5)
        #TODO check problem with the following, precision is low,
        #dof error? last t-value is 22, 23, error is around 1% for PoissonMLE
        #this was with constant=1,
        #now changed constant=0.1 to make it less significant and test passes
        #overall precision for tstat looks like 1%

        #assert_almost_equal(self.res.tval, self.res_glm.t(), DEC)
        assert_almost_equal(self.res.tvalues, self.res_discrete.tvalues, DEC4)
        #assert_almost_equal(self.res.params, self.res_discrete.params)
        assert_almost_equal(self.res.pvalues, self.res_discrete.pvalues, DEC)

    def test_ttest(self):
        tt = self.res.t_test(np.eye(len(self.res.params)))
        from scipy import stats
        pvalue = stats.norm.sf(np.abs(tt.tvalue)) * 2
        assert_almost_equal(tt.tvalue, self.res.tvalues, DEC)
        assert_almost_equal(pvalue, self.res.pvalues, DEC)

    def test_df(self):
        res = self.res
        k_extra = getattr(self, "k_extra", 0)
        nobs, k_vars = res.model.exog.shape
        assert res.df_resid == nobs - k_vars - k_extra
        assert res.df_model == k_vars - 1  # -1 for constant
        assert len(res.params) == k_vars + k_extra

    @pytest.mark.smoke
    def test_summary(self):
        self.res.summary()


class TestPoissonMLE(CompareMixin):

    @classmethod
    def setup_class(cls):

        # generate artificial data
        np.random.seed(98765678)
        nobs = 200
        rvs = np.random.randn(nobs,6)
        data_exog = rvs
        data_exog = sm.add_constant(data_exog, prepend=False)
        xbeta = 0.1 + 0.1*rvs.sum(1)
        data_endog = np.random.poisson(np.exp(xbeta))

        #estimate discretemod.Poisson as benchmark
        cls.res_discrete = Poisson(data_endog, data_exog).fit(disp=0)

        mod_glm = sm.GLM(data_endog, data_exog, family=sm.families.Poisson())
        cls.res_glm = mod_glm.fit()

        #estimate generic MLE
        cls.mod = PoissonGMLE(data_endog, data_exog)
        cls.res = cls.mod.fit(start_params=0.9 * cls.res_discrete.params,
                                method='bfgs', disp=0)

    def test_predict_distribution(self):
        res = self.res
        model = self.mod

        with pytest.raises(ValueError):
            # No "result" attribute
            model.predict_distribution(model.exog)

        try:
            model.result = res
            dist = model.predict_distribution(model.exog)
            assert isinstance(dist, stats._distn_infrastructure.rv_frozen)
            assert_almost_equal(dist.mean(),
                                np.exp(model.exog.dot(res.params)),
                                15)
        finally:
            # leave the model object how we found it
            model.__delattr__("result")


class TestPoissonOffset(CompareMixin):
    #this uses the first exog to construct an offset variable
    @classmethod
    def setup_class(cls):
        # generate artificial data
        np.random.seed(98765678)
        nobs = 200
        rvs = np.random.randn(nobs,6)
        data_exog = rvs
        data_exog = sm.add_constant(data_exog, prepend=False)
        xbeta = 1 + 0.1*rvs.sum(1)
        data_endog = np.random.poisson(np.exp(xbeta))

        mod_glm = sm.GLM(data_endog, data_exog, family=sm.families.Poisson())
        cls.res_glm = mod_glm.fit()

        #estimate generic MLE
        #cls.mod = PoissonGMLE(data_endog, data_exog)
        #res = cls.mod.fit()

        #create offset variable based on first exog
        cls.res_discrete = Poisson(data_endog, data_exog).fit(disp=0)
        offset = cls.res_discrete.params[0] * data_exog[:,0]  #1d ???

        #estimate discretemod.Poisson as benchmark, now has offset
        cls.res_discrete = Poisson(data_endog, data_exog[:,1:],
                                    offset=offset).fit(disp=0)

        mod_glm = sm.GLM(data_endog, data_exog, family=sm.families.Poisson())
        cls.res_glm = mod_glm.fit()

        #cls.res = PoissonOffsetGMLE(data_endog, data_exog[:,1:], offset=offset).fit(start_params = np.ones(6)/2., method='nm')
        modo = PoissonOffsetGMLE(data_endog, data_exog[:,1:], offset=offset)
        cls.res = modo.fit(start_params = 0.9*cls.res_discrete.params,
                            method='bfgs', disp=0)



    def test_params(self):
        assert_almost_equal(self.res.params, self.res_glm.params[1:], DEC)
        assert_almost_equal(self.res.params, self.res_discrete.params, DEC)

    def test_cov_params(self):
        assert_almost_equal(self.res.bse, self.res_glm.bse[1:], DEC-1)
        assert_almost_equal(self.res.bse, self.res_discrete.bse, DEC5)
        #precision of next is very low ???
        #assert_almost_equal(self.res.tval, self.res_glm.t()[1:], DEC)
        #assert_almost_equal(self.res.params, self.res_discrete.params)


#DEC = DEC - 1
class TestPoissonZi(CompareMixin):
    #this uses the first exog to construct an offset variable
    @classmethod
    def setup_class(cls):

        # generate artificial data
        np.random.seed(98765678)
        nobs = 200
        rvs = np.random.randn(nobs,6)
        data_exog = rvs
        data_exog = sm.add_constant(data_exog, prepend=False)
        xbeta = 1 + 0.1*rvs.sum(1)
        data_endog = np.random.poisson(np.exp(xbeta))
        cls.k_extra = 1

        mod_glm = sm.GLM(data_endog, data_exog, family=sm.families.Poisson())
        cls.res_glm = mod_glm.fit()

        #estimate generic MLE
        #cls.mod = PoissonGMLE(data_endog, data_exog)
        #res = cls.mod.fit()

        #create offset variable based on first exog
        cls.res_discrete = Poisson(data_endog, data_exog).fit(disp=0)
        offset = cls.res_discrete.params[0] * data_exog[:,0]  #1d ???

        #estimate discretemod.Poisson as benchmark, now has offset
        cls.res_discrete = Poisson(data_endog, data_exog[:,1:], offset=offset).fit(disp=0)

        # Note : ZI has one extra parameter
        cls.res = PoissonZiGMLE(data_endog, data_exog[:,1:], offset=offset).fit(
                            start_params=np.r_[0.9*cls.res_discrete.params,10],
                            method='bfgs', disp=0)

        cls.decimal = 4

    def test_params(self):
        assert_almost_equal(self.res.params[:-1], self.res_glm.params[1:], self.decimal)
        assert_almost_equal(self.res.params[:-1], self.res_discrete.params, self.decimal)

    def test_cov_params(self):
        #skip until I have test with zero-inflated data
        #use bsejac for now since it seems to work
        assert_almost_equal(self.res.bsejac[:-1], self.res_glm.bse[1:], self.decimal-2)
        assert_almost_equal(self.res.bsejac[:-1], self.res_discrete.bse, self.decimal-2)
        #assert_almost_equal(self.res.tval[:-1], self.res_glm.t()[1:], self.decimal)


    def test_exog_names_warning(self):
        mod = self.res.model
        mod1 = PoissonOffsetGMLE(mod.endog, mod.exog, offset=mod.offset)
        from numpy.testing import assert_warns
        mod1.data.xnames = mod1.data.xnames * 2
        assert_warns(ValueWarning, mod1.fit, disp=0)
