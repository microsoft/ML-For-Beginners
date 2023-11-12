
import pytest
import numpy as np
import numpy.testing as npt

import statsmodels.api as sm
nparam = sm.nonparametric


class KernelRegressionTestBase:
    @classmethod
    def setup_class(cls):
        nobs = 60
        np.random.seed(123456)
        cls.o = np.random.binomial(2, 0.7, size=(nobs, 1))
        cls.o2 = np.random.binomial(3, 0.7, size=(nobs, 1))
        cls.c1 = np.random.normal(size=(nobs, 1))
        cls.c2 = np.random.normal(10, 1, size=(nobs, 1))
        cls.c3 = np.random.normal(10, 2, size=(nobs, 1))
        cls.noise = np.random.normal(size=(nobs, 1))
        b0 = 0.3
        b1 = 1.2
        b2 = 3.7  # regression coefficients
        cls.y = b0 + b1 * cls.c1 + b2 * cls.c2 + cls.noise
        cls.y2 = b0 + b1 * cls.c1 + b2 * cls.c2 + cls.o + cls.noise
        # Italy data from R's np package (the first 50 obs) R>> data (Italy)

        cls.Italy_gdp = \
            [8.556, 12.262, 9.587, 8.119, 5.537, 6.796, 8.638,
             6.483, 6.212, 5.111, 6.001, 7.027, 4.616, 3.922,
             4.688, 3.957, 3.159, 3.763, 3.829, 5.242, 6.275,
             8.518, 11.542, 9.348, 8.02, 5.527, 6.865, 8.666,
             6.672, 6.289, 5.286, 6.271, 7.94, 4.72, 4.357,
             4.672, 3.883, 3.065, 3.489, 3.635, 5.443, 6.302,
             9.054, 12.485, 9.896, 8.33, 6.161, 7.055, 8.717,
             6.95]

        cls.Italy_year = \
            [1951, 1951, 1951, 1951, 1951, 1951, 1951, 1951, 1951, 1951, 1951,
             1951, 1951, 1951, 1951, 1951, 1951, 1951, 1951, 1951, 1951, 1952,
             1952, 1952, 1952, 1952, 1952, 1952, 1952, 1952, 1952, 1952, 1952,
             1952, 1952, 1952, 1952, 1952, 1952, 1952, 1952, 1952, 1953, 1953,
             1953, 1953, 1953, 1953, 1953, 1953]

        # OECD panel data from NP  R>> data(oecdpanel)
        cls.growth = \
            [-0.0017584, 0.00740688, 0.03424461, 0.03848719, 0.02932506,
             0.03769199, 0.0466038, 0.00199456, 0.03679607, 0.01917304,
             -0.00221, 0.00787269, 0.03441118, -0.0109228, 0.02043064,
             -0.0307962, 0.02008947, 0.00580313, 0.00344502, 0.04706358,
             0.03585851, 0.01464953, 0.04525762, 0.04109222, -0.0087903,
             0.04087915, 0.04551403, 0.036916, 0.00369293, 0.0718669,
             0.02577732, -0.0130759, -0.01656641, 0.00676429, 0.08833017,
             0.05092105, 0.02005877, 0.00183858, 0.03903173, 0.05832116,
             0.0494571, 0.02078484, 0.09213897, 0.0070534, 0.08677202,
             0.06830603, -0.00041, 0.0002856, 0.03421225, -0.0036825]

        cls.oecd = \
            [0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
             0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0,
             0, 0, 0, 0]

    def write2file(self, file_name, data):  # pragma: no cover
        """Write some data to a csv file.  Only use for debugging!"""
        import csv

        data_file = csv.writer(open(file_name, "w", encoding="utf-8"))
        data = np.column_stack(data)
        nobs = max(np.shape(data))
        K = min(np.shape(data))
        data = np.reshape(data, (nobs,K))
        for i in range(nobs):
            data_file.writerow(list(data[i, :]))


class TestKernelReg(KernelRegressionTestBase):
    def test_ordered_lc_cvls(self):
        model = nparam.KernelReg(endog=[self.Italy_gdp],
                                 exog=[self.Italy_year], reg_type='lc',
                                 var_type='o', bw='cv_ls')
        sm_bw = model.bw
        R_bw = 0.1390096

        sm_mean, sm_mfx = model.fit()
        sm_mean = sm_mean[0:5]
        sm_mfx = sm_mfx[0:5]
        R_mean = 6.190486

        sm_R2 = model.r_squared()
        R_R2 = 0.1435323

        ## CODE TO REPRODUCE IN R
        ## library(np)
        ## data(Italy)
        ## attach(Italy)
        ## bw <- npregbw(formula=gdp[1:50]~ordered(year[1:50]))
        npt.assert_allclose(sm_bw, R_bw, atol=1e-2)
        npt.assert_allclose(sm_mean, R_mean, atol=1e-2)
        npt.assert_allclose(sm_R2, R_R2, atol=1e-2)

    def test_continuousdata_lc_cvls(self):
        model = nparam.KernelReg(endog=[self.y], exog=[self.c1, self.c2],
                                 reg_type='lc', var_type='cc', bw='cv_ls')
        # Bandwidth
        sm_bw = model.bw
        R_bw = [0.6163835, 0.1649656]
        # Conditional Mean
        sm_mean, sm_mfx = model.fit()
        sm_mean = sm_mean[0:5]
        sm_mfx = sm_mfx[0:5]
        R_mean = [31.49157, 37.29536, 43.72332, 40.58997, 36.80711]
        # R-Squared
        sm_R2 = model.r_squared()
        R_R2 = 0.956381720885

        npt.assert_allclose(sm_bw, R_bw, atol=1e-2)
        npt.assert_allclose(sm_mean, R_mean, atol=1e-2)
        npt.assert_allclose(sm_R2, R_R2, atol=1e-2)

    def test_continuousdata_ll_cvls(self):
        model = nparam.KernelReg(endog=[self.y], exog=[self.c1, self.c2],
                                 reg_type='ll', var_type='cc', bw='cv_ls')

        sm_bw = model.bw
        R_bw = [1.717891, 2.449415]
        sm_mean, sm_mfx = model.fit()
        sm_mean = sm_mean[0:5]
        sm_mfx = sm_mfx[0:5]
        R_mean = [31.16003, 37.30323, 44.49870, 40.73704, 36.19083]

        sm_R2 = model.r_squared()
        R_R2 = 0.9336019

        npt.assert_allclose(sm_bw, R_bw, atol=1e-2)
        npt.assert_allclose(sm_mean, R_mean, atol=1e-2)
        npt.assert_allclose(sm_R2, R_R2, atol=1e-2)

    def test_continuous_mfx_ll_cvls(self, file_name='RegData.csv'):
        nobs = 200
        np.random.seed(1234)
        C1 = np.random.normal(size=(nobs, ))
        C2 = np.random.normal(2, 1, size=(nobs, ))
        C3 = np.random.beta(0.5,0.2, size=(nobs,))
        noise = np.random.normal(size=(nobs, ))
        b0 = 3
        b1 = 1.2
        b2 = 3.7  # regression coefficients
        b3 = 2.3
        Y = b0+ b1 * C1 + b2*C2+ b3 * C3 + noise
        bw_cv_ls = np.array([0.96075, 0.5682, 0.29835])
        model = nparam.KernelReg(endog=[Y], exog=[C1, C2, C3],
                                 reg_type='ll', var_type='ccc', bw=bw_cv_ls)
        sm_mean, sm_mfx = model.fit()
        sm_mean = sm_mean[0:5]
        npt.assert_allclose(sm_mfx[0,:], [b1,b2,b3], rtol=2e-1)

    def test_mixed_mfx_ll_cvls(self, file_name='RegData.csv'):
        nobs = 200
        np.random.seed(1234)
        ovals = np.random.binomial(2, 0.5, size=(nobs, ))
        C1 = np.random.normal(size=(nobs, ))
        C2 = np.random.normal(2, 1, size=(nobs, ))
        noise = np.random.normal(size=(nobs, ))
        b0 = 3
        b1 = 1.2
        b2 = 3.7  # regression coefficients
        b3 = 2.3
        Y = b0+ b1 * C1 + b2*C2+ b3 * ovals + noise
        bw_cv_ls = np.array([1.04726, 1.67485, 0.39852])
        model = nparam.KernelReg(endog=[Y], exog=[C1, C2, ovals],
                                 reg_type='ll', var_type='cco', bw=bw_cv_ls)
        sm_mean, sm_mfx = model.fit()
        # TODO: add expected result
        sm_R2 = model.r_squared()  # noqa: F841
        npt.assert_allclose(sm_mfx[0, :], [b1, b2, b3], rtol=2e-1)

    @pytest.mark.slow
    @pytest.mark.xfail(reason="Test does not make much sense - always passes "
                              "with very small bw.")
    def test_mfx_nonlinear_ll_cvls(self, file_name='RegData.csv'):
        nobs = 200
        np.random.seed(1234)
        C1 = np.random.normal(size=(nobs,))
        C2 = np.random.normal(2, 1, size=(nobs,))
        C3 = np.random.beta(0.5,0.2, size=(nobs,))
        noise = np.random.normal(size=(nobs,))
        b0 = 3
        b1 = 1.2
        b3 = 2.3
        Y = b0+ b1 * C1 * C2 + b3 * C3 + noise
        model = nparam.KernelReg(endog=[Y], exog=[C1, C2, C3],
                                 reg_type='ll', var_type='ccc', bw='cv_ls')
        sm_bw = model.bw
        sm_mean, sm_mfx = model.fit()
        sm_R2 = model.r_squared()
        # Theoretical marginal effects
        mfx1 = b1 * C2
        mfx2 = b1 * C1
        npt.assert_allclose(sm_mean, Y, rtol = 2e-1)

        npt.assert_allclose(sm_mfx[:, 0], mfx1, rtol=2e-1)
        npt.assert_allclose(sm_mfx[0:10, 1], mfx2[0:10], rtol=2e-1)

    @pytest.mark.slow
    def test_continuous_cvls_efficient(self):
        nobs = 500
        np.random.seed(12345)
        C1 = np.random.normal(size=(nobs, ))
        C2 = np.random.normal(2, 1, size=(nobs, ))
        b0 = 3
        b1 = 1.2
        b2 = 3.7  # regression coefficients
        Y = b0+ b1 * C1 + b2*C2

        model_efficient = nparam.KernelReg(endog=[Y], exog=[C1], reg_type='lc',
                              var_type='c', bw='cv_ls',
                              defaults=nparam.EstimatorSettings(efficient=True,
                                                                n_sub=100))

        model = nparam.KernelReg(endog=[Y], exog=[C1], reg_type='ll',
                                 var_type='c', bw='cv_ls')
        npt.assert_allclose(model.bw, model_efficient.bw, atol=5e-2, rtol=1e-1)

    @pytest.mark.slow
    def test_censored_ll_cvls(self):
        nobs = 200
        np.random.seed(1234)
        C1 = np.random.normal(size=(nobs, ))
        C2 = np.random.normal(2, 1, size=(nobs, ))
        noise = np.random.normal(size=(nobs, ))
        Y = 0.3 +1.2 * C1 - 0.9 * C2 + noise
        Y[Y>0] = 0  # censor the data
        model = nparam.KernelCensoredReg(endog=[Y], exog=[C1, C2],
                                         reg_type='ll', var_type='cc',
                                         bw='cv_ls', censor_val=0)
        sm_mean, sm_mfx = model.fit()
        npt.assert_allclose(sm_mfx[0,:], [1.2, -0.9], rtol = 2e-1)

    @pytest.mark.slow
    def test_continuous_lc_aic(self):
        nobs = 200
        np.random.seed(1234)
        C1 = np.random.normal(size=(nobs, ))
        C2 = np.random.normal(2, 1, size=(nobs, ))
        noise = np.random.normal(size=(nobs, ))
        Y = 0.3 +1.2 * C1 - 0.9 * C2 + noise
        #self.write2file('RegData.csv', (Y, C1, C2))

        #CODE TO PRODUCE BANDWIDTH ESTIMATION IN R
        #library(np)
        #data <- read.csv('RegData.csv', header=FALSE)
        #bw <- npregbw(formula=data$V1 ~ data$V2 + data$V3,
        #                bwmethod='cv.aic', regtype='lc')
        model = nparam.KernelReg(endog=[Y], exog=[C1, C2],
                                 reg_type='lc', var_type='cc', bw='aic')
        #R_bw = [0.4017893, 0.4943397]  # Bandwidth obtained in R
        bw_expected = [0.3987821, 0.50933458]
        npt.assert_allclose(model.bw, bw_expected, rtol=1e-3)

    @pytest.mark.slow
    def test_significance_continuous(self):
        nobs = 250
        np.random.seed(12345)
        C1 = np.random.normal(size=(nobs, ))
        C2 = np.random.normal(2, 1, size=(nobs, ))
        C3 = np.random.beta(0.5,0.2, size=(nobs,))
        noise = np.random.normal(size=(nobs, ))
        b1 = 1.2
        b2 = 3.7  # regression coefficients
        Y = b1 * C1 + b2 * C2 + noise

        # This is the cv_ls bandwidth estimated earlier
        bw=[11108137.1087194, 1333821.85150218]
        model = nparam.KernelReg(endog=[Y], exog=[C1, C3],
                                 reg_type='ll', var_type='cc', bw=bw)
        nboot = 45  # Number of bootstrap samples
        sig_var12 = model.sig_test([0,1], nboot=nboot)  # H0: b1 = 0 and b2 = 0
        npt.assert_equal(sig_var12 == 'Not Significant', False)
        sig_var1 = model.sig_test([0], nboot=nboot)  # H0: b1 = 0
        npt.assert_equal(sig_var1 == 'Not Significant', False)
        sig_var2 = model.sig_test([1], nboot=nboot)  # H0: b2 = 0
        npt.assert_equal(sig_var2 == 'Not Significant', True)

    @pytest.mark.slow
    def test_significance_discrete(self):
        nobs = 200
        np.random.seed(12345)
        ovals = np.random.binomial(2, 0.5, size=(nobs, ))
        C2 = np.random.normal(2, 1, size=(nobs, ))
        C3 = np.random.beta(0.5,0.2, size=(nobs,))
        noise = np.random.normal(size=(nobs, ))
        b1 = 1.2
        b2 = 3.7  # regression coefficients
        Y = b1 * ovals + b2 * C2 + noise

        bw= [3.63473198e+00, 1.21404803e+06]
        # This is the cv_ls bandwidth estimated earlier
        # The cv_ls bandwidth was estimated earlier to save time
        model = nparam.KernelReg(endog=[Y], exog=[ovals, C3],
                                 reg_type='ll', var_type='oc', bw=bw)
        # This was also tested with local constant estimator
        nboot = 45  # Number of bootstrap samples
        sig_var1 = model.sig_test([0], nboot=nboot)  # H0: b1 = 0
        npt.assert_equal(sig_var1 == 'Not Significant', False)
        sig_var2 = model.sig_test([1], nboot=nboot)  # H0: b2 = 0
        npt.assert_equal(sig_var2 == 'Not Significant', True)

    def test_user_specified_kernel(self):
        model = nparam.KernelReg(endog=[self.y], exog=[self.c1, self.c2],
                                 reg_type='ll', var_type='cc', bw='cv_ls',
                                 ckertype='tricube')
        # Bandwidth
        sm_bw = model.bw
        R_bw = [0.581663, 0.5652]
        # Conditional Mean
        sm_mean, sm_mfx = model.fit()
        sm_mean = sm_mean[0:5]
        sm_mfx = sm_mfx[0:5]
        R_mean = [30.926714, 36.994604, 44.438358, 40.680598, 35.961593]
        # R-Squared
        sm_R2 = model.r_squared()
        R_R2 = 0.934825

        npt.assert_allclose(sm_bw, R_bw, atol=1e-2)
        npt.assert_allclose(sm_mean, R_mean, atol=1e-2)
        npt.assert_allclose(sm_R2, R_R2, atol=1e-2)

    def test_censored_user_specified_kernel(self):
        model = nparam.KernelCensoredReg(endog=[self.y], exog=[self.c1, self.c2],
                                 reg_type='ll', var_type='cc', bw='cv_ls',
                                 censor_val=0, ckertype='tricube')
        # Bandwidth
        sm_bw = model.bw
        R_bw = [0.581663, 0.5652]
        # Conditional Mean
        sm_mean, sm_mfx = model.fit()
        sm_mean = sm_mean[0:5]
        sm_mfx = sm_mfx[0:5]
        R_mean = [29.205526, 29.538008, 31.667581, 31.978866, 30.926714]
        # R-Squared
        sm_R2 = model.r_squared()
        R_R2 = 0.934825

        npt.assert_allclose(sm_bw, R_bw, atol=1e-2)
        npt.assert_allclose(sm_mean, R_mean, atol=1e-2)
        npt.assert_allclose(sm_R2, R_R2, atol=1e-2)

    def test_efficient_user_specificed_bw(self):

        bw_user=[0.23, 434697.22]
        model = nparam.KernelReg(endog=[self.y], exog=[self.c1, self.c2],
                                 reg_type='lc', var_type='cc', bw=bw_user,
                                 defaults=nparam.EstimatorSettings(efficient=True))
        # Bandwidth
        npt.assert_equal(model.bw, bw_user)

    def test_censored_efficient_user_specificed_bw(self):
        nobs = 200
        np.random.seed(1234)
        C1 = np.random.normal(size=(nobs, ))
        C2 = np.random.normal(2, 1, size=(nobs, ))
        noise = np.random.normal(size=(nobs, ))
        Y = 0.3 +1.2 * C1 - 0.9 * C2 + noise
        Y[Y>0] = 0  # censor the data

        bw_user=[0.23, 434697.22]
        model = nparam.KernelCensoredReg(endog=[Y], exog=[C1, C2],
                                         reg_type='ll', var_type='cc',
                                         bw=bw_user, censor_val=0,
                                 defaults=nparam.EstimatorSettings(efficient=True))
        # Bandwidth
        npt.assert_equal(model.bw, bw_user)


def test_invalid_bw():
    # GH4873
    x = np.arange(400)
    y = x ** 2
    with pytest.raises(ValueError):
        nparam.KernelReg(x, y, 'c', bw=[12.5, 1.])


def test_invalid_kernel():
    x = np.arange(400)
    y = x ** 2
    # silverman kernel is not currently in statsmodels kernel library
    with pytest.raises(ValueError):
        nparam.KernelReg(x, y, reg_type='ll', var_type='cc', bw='cv_ls',
                         ckertype='silverman')

    with pytest.raises(ValueError):
        nparam.KernelCensoredReg(x, y, reg_type='ll', var_type='cc', bw='cv_ls',
                                 censor_val=0, ckertype='silverman')
