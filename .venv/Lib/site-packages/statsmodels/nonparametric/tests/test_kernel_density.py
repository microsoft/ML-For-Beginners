import numpy as np
import numpy.testing as npt
import pytest
from numpy.testing import assert_allclose, assert_equal

import statsmodels.api as sm

nparam = sm.nonparametric


class KDETestBase:
    def setup_method(self):
        nobs = 60
        np.random.seed(123456)
        self.o = np.random.binomial(2, 0.7, size=(nobs, 1))
        self.o2 = np.random.binomial(3, 0.7, size=(nobs, 1))
        self.c1 = np.random.normal(size=(nobs, 1))
        self.c2 = np.random.normal(10, 1, size=(nobs, 1))
        self.c3 = np.random.normal(10, 2, size=(nobs, 1))
        self.noise = np.random.normal(size=(nobs, 1))
        b0 = 0.3
        b1 = 1.2
        b2 = 3.7  # regression coefficients
        self.y = b0 + b1 * self.c1 + b2 * self.c2 + self.noise
        self.y2 = b0 + b1 * self.c1 + b2 * self.c2 + self.o + self.noise
        # Italy data from R's np package (the first 50 obs) R>> data (Italy)

        self.Italy_gdp = \
        [8.556, 12.262, 9.587, 8.119, 5.537, 6.796, 8.638,
         6.483, 6.212, 5.111, 6.001, 7.027, 4.616, 3.922,
         4.688, 3.957, 3.159, 3.763, 3.829, 5.242, 6.275,
         8.518, 11.542, 9.348, 8.02, 5.527, 6.865, 8.666,
         6.672, 6.289, 5.286, 6.271, 7.94, 4.72, 4.357,
         4.672, 3.883, 3.065, 3.489, 3.635, 5.443, 6.302,
         9.054, 12.485, 9.896, 8.33, 6.161, 7.055, 8.717,
         6.95]

        self.Italy_year = \
        [1951, 1951, 1951, 1951, 1951, 1951, 1951, 1951, 1951, 1951, 1951,
       1951, 1951, 1951, 1951, 1951, 1951, 1951, 1951, 1951, 1951, 1952,
       1952, 1952, 1952, 1952, 1952, 1952, 1952, 1952, 1952, 1952, 1952,
       1952, 1952, 1952, 1952, 1952, 1952, 1952, 1952, 1952, 1953, 1953,
       1953, 1953, 1953, 1953, 1953, 1953]

        # OECD panel data from NP  R>> data(oecdpanel)
        self.growth = \
        [-0.0017584, 0.00740688, 0.03424461, 0.03848719, 0.02932506,
        0.03769199, 0.0466038,  0.00199456, 0.03679607, 0.01917304,
       -0.00221, 0.00787269, 0.03441118, -0.0109228, 0.02043064,
       -0.0307962, 0.02008947, 0.00580313, 0.00344502, 0.04706358,
        0.03585851, 0.01464953, 0.04525762, 0.04109222, -0.0087903,
        0.04087915, 0.04551403, 0.036916, 0.00369293, 0.0718669,
        0.02577732, -0.0130759, -0.01656641, 0.00676429, 0.08833017,
        0.05092105, 0.02005877,  0.00183858, 0.03903173, 0.05832116,
        0.0494571, 0.02078484,  0.09213897, 0.0070534, 0.08677202,
        0.06830603, -0.00041, 0.0002856, 0.03421225, -0.0036825]

        self.oecd = \
        [0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
       0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0,
       0, 0, 0, 0]

        self.weights = np.random.random(nobs)


class TestKDEUnivariate(KDETestBase):

    def test_pdf_non_fft(self):

        kde = nparam.KDEUnivariate(self.noise)
        kde.fit(fft=False, bw='scott')


        grid = kde.support
        testx = [grid[10*i] for i in range(6)]

        # Test against values from R 'ks' package
        kde_expected = [0.00016808277984236013,
                        0.030759614592368954,
                        0.14123404934759243,
                        0.28807147408162409,
                        0.25594519303876273,
                        0.056593973915651047]

        kde_vals0 = kde.density[10 * np.arange(6)]
        kde_vals = kde.evaluate(testx)

        npt.assert_allclose(kde_vals, kde_expected,
                            atol=1e-6)
        npt.assert_allclose(kde_vals0, kde_expected,
                            atol=1e-6)


    def test_weighted_pdf_non_fft(self):

        kde = nparam.KDEUnivariate(self.noise)
        kde.fit(weights=self.weights, fft=False, bw='scott')

        grid = kde.support
        testx = [grid[10*i] for i in range(6)]

        # Test against values from R 'ks' package
        kde_expected = [9.1998858033950757e-05,
                        0.018761981151370496,
                        0.14425925509365087,
                        0.30307631742267443,
                        0.2405445849994125,
                        0.06433170684797665]

        kde_vals0 = kde.density[10 * np.arange(6)]
        kde_vals = kde.evaluate(testx)

        npt.assert_allclose(kde_vals, kde_expected,
                            atol=1e-6)
        npt.assert_allclose(kde_vals0, kde_expected,
                            atol=1e-6)

    def test_all_samples_same_location_bw(self):
        x = np.ones(100)
        kde = nparam.KDEUnivariate(x)
        with pytest.raises(RuntimeError, match="Selected KDE bandwidth is 0"):
            kde.fit()

    def test_int(self, reset_randomstate):
        x = np.random.randint(0, 100, size=1000)
        kde = nparam.KDEUnivariate(x)
        kde.fit()

        kde_double = nparam.KDEUnivariate(x.astype("double"))
        kde_double.fit()

        assert_allclose(kde.bw, kde_double.bw)


class TestKDEMultivariate(KDETestBase):
    @pytest.mark.slow
    def test_pdf_mixeddata_CV_LS(self):
        dens_u = nparam.KDEMultivariate(data=[self.c1, self.o, self.o2],
                                        var_type='coo', bw='cv_ls')
        npt.assert_allclose(dens_u.bw, [0.70949447, 0.08736727, 0.09220476],
                            atol=1e-6)

        # Matches R to 3 decimals; results seem more stable than with R.
        # Can be checked with following code:
        # import rpy2.robjects as robjects
        # from rpy2.robjects.packages import importr
        # NP = importr('np')
        # r = robjects.r
        # D = {"S1": robjects.FloatVector(c1), "S2":robjects.FloatVector(c2),
        #      "S3":robjects.FloatVector(c3), "S4":robjects.FactorVector(o),
        #      "S5":robjects.FactorVector(o2)}
        # df = robjects.DataFrame(D)
        # formula = r('~S1+ordered(S4)+ordered(S5)')
        # r_bw = NP.npudensbw(formula, data=df, bwmethod='cv.ls')

    @pytest.mark.slow
    def test_pdf_mixeddata_LS_vs_ML(self):
        dens_ls = nparam.KDEMultivariate(data=[self.c1, self.o, self.o2],
                                         var_type='coo', bw='cv_ls')
        dens_ml = nparam.KDEMultivariate(data=[self.c1, self.o, self.o2],
                                         var_type='coo', bw='cv_ml')
        npt.assert_allclose(dens_ls.bw, dens_ml.bw, atol=0, rtol=0.5)

    def test_pdf_mixeddata_CV_ML(self):
        # Test ML cross-validation
        dens_ml = nparam.KDEMultivariate(data=[self.c1, self.o, self.c2],
                                         var_type='coc', bw='cv_ml')
        R_bw = [1.021563, 2.806409e-14, 0.5142077]
        npt.assert_allclose(dens_ml.bw, R_bw, atol=0.1, rtol=0.1)

    @pytest.mark.slow
    def test_pdf_continuous(self):
        # Test for only continuous data
        dens = nparam.KDEMultivariate(data=[self.growth, self.Italy_gdp],
                                      var_type='cc', bw='cv_ls')
        # take the first data points from the training set
        sm_result = np.squeeze(dens.pdf()[0:5])
        R_result = [1.6202284, 0.7914245, 1.6084174, 2.4987204, 1.3705258]

        ## CODE TO REPRODUCE THE RESULTS IN R
        ## library(np)
        ## data(oecdpanel)
        ## data (Italy)
        ## bw <-npudensbw(formula = ~oecdpanel$growth[1:50] + Italy$gdp[1:50],
        ## bwmethod ='cv.ls')
        ## fhat <- fitted(npudens(bws=bw))
        ## fhat[1:5]
        npt.assert_allclose(sm_result, R_result, atol=1e-3)

    def test_pdf_ordered(self):
        # Test for only ordered data
        dens = nparam.KDEMultivariate(data=[self.oecd], var_type='o', bw='cv_ls')
        sm_result = np.squeeze(dens.pdf()[0:5])
        R_result = [0.7236395, 0.7236395, 0.2763605, 0.2763605, 0.7236395]
        # lower tol here. only 2nd decimal
        npt.assert_allclose(sm_result, R_result, atol=1e-1)

    @pytest.mark.slow
    def test_unordered_CV_LS(self):
        dens = nparam.KDEMultivariate(data=[self.growth, self.oecd],
                                      var_type='cu', bw='cv_ls')
        R_result = [0.0052051, 0.05835941]
        npt.assert_allclose(dens.bw, R_result, atol=1e-2)

    def test_continuous_cdf(self, data_predict=None):
        dens = nparam.KDEMultivariate(data=[self.Italy_gdp, self.growth],
                                      var_type='cc', bw='cv_ml')
        sm_result = dens.cdf()[0:5]
        R_result = [0.192180770, 0.299505196, 0.557303666,
                    0.513387712, 0.210985350]
        npt.assert_allclose(sm_result, R_result, atol=1e-3)

    def test_mixeddata_cdf(self, data_predict=None):
        dens = nparam.KDEMultivariate(data=[self.Italy_gdp, self.oecd],
                                      var_type='cu', bw='cv_ml')
        sm_result = dens.cdf()[0:5]
        R_result = [0.54700010, 0.65907039, 0.89676865, 0.74132941, 0.25291361]
        npt.assert_allclose(sm_result, R_result, atol=1e-3)

    @pytest.mark.slow
    def test_continuous_cvls_efficient(self):
        nobs = 400
        np.random.seed(12345)
        C1 = np.random.normal(size=(nobs, ))
        C2 = np.random.normal(2, 1, size=(nobs, ))
        Y = 0.3 +1.2 * C1 - 0.9 * C2
        dens_efficient = nparam.KDEMultivariate(data=[Y, C1], var_type='cc',
            bw='cv_ls',
            defaults=nparam.EstimatorSettings(efficient=True, n_sub=100))
        #dens = nparam.KDEMultivariate(data=[Y, C1], var_type='cc', bw='cv_ls',
        #                  defaults=nparam.EstimatorSettings(efficient=False))
        #bw = dens.bw
        bw = np.array([0.3404, 0.1666])
        npt.assert_allclose(bw, dens_efficient.bw, atol=0.1, rtol=0.2)

    @pytest.mark.slow
    def test_continuous_cvml_efficient(self):
        nobs = 400
        np.random.seed(12345)
        C1 = np.random.normal(size=(nobs, ))
        C2 = np.random.normal(2, 1, size=(nobs, ))
        Y = 0.3 +1.2 * C1 - 0.9 * C2

        dens_efficient = nparam.KDEMultivariate(data=[Y, C1], var_type='cc',
            bw='cv_ml', defaults=nparam.EstimatorSettings(efficient=True,
                                                          n_sub=100))
        #dens = nparam.KDEMultivariate(data=[Y, C1], var_type='cc', bw='cv_ml',
        #                  defaults=nparam.EstimatorSettings(efficient=False))
        #bw = dens.bw
        bw = np.array([0.4471, 0.2861])
        npt.assert_allclose(bw, dens_efficient.bw, atol=0.1, rtol = 0.2)

    @pytest.mark.slow
    def test_efficient_notrandom(self):
        nobs = 400
        np.random.seed(12345)
        C1 = np.random.normal(size=(nobs, ))
        C2 = np.random.normal(2, 1, size=(nobs, ))
        Y = 0.3 +1.2 * C1 - 0.9 * C2

        dens_efficient = nparam.KDEMultivariate(data=[Y, C1], var_type='cc',
            bw='cv_ml', defaults=nparam.EstimatorSettings(efficient=True,
                                                          randomize=False,
                                                          n_sub=100))
        dens = nparam.KDEMultivariate(data=[Y, C1], var_type='cc', bw='cv_ml')
        npt.assert_allclose(dens.bw, dens_efficient.bw, atol=0.1, rtol = 0.2)

    def test_efficient_user_specified_bw(self):
        nobs = 400
        np.random.seed(12345)
        C1 = np.random.normal(size=(nobs, ))
        C2 = np.random.normal(2, 1, size=(nobs, ))
        bw_user=[0.23, 434697.22]

        dens = nparam.KDEMultivariate(data=[C1, C2], var_type='cc',
            bw=bw_user, defaults=nparam.EstimatorSettings(efficient=True,
                                                          randomize=False,
                                                          n_sub=100))
        npt.assert_equal(dens.bw, bw_user)


class TestKDEMultivariateConditional(KDETestBase):
    @pytest.mark.slow
    def test_mixeddata_CV_LS(self):
        dens_ls = nparam.KDEMultivariateConditional(endog=[self.Italy_gdp],
                                                    exog=[self.Italy_year],
                                                    dep_type='c',
                                                    indep_type='o', bw='cv_ls')
        # R result: [1.6448, 0.2317373]
        npt.assert_allclose(dens_ls.bw, [1.01203728, 0.31905144], atol=1e-5)

    def test_continuous_CV_ML(self):
        dens_ml = nparam.KDEMultivariateConditional(endog=[self.Italy_gdp],
                                                    exog=[self.growth],
                                                    dep_type='c',
                                                    indep_type='c', bw='cv_ml')
        # Results from R
        npt.assert_allclose(dens_ml.bw, [0.5341164, 0.04510836], atol=1e-3)

    @pytest.mark.slow
    def test_unordered_CV_LS(self):
        dens_ls = nparam.KDEMultivariateConditional(endog=[self.oecd],
                                                    exog=[self.growth],
                                                    dep_type='u',
                                                    indep_type='c', bw='cv_ls')
        # TODO: assert missing

    def test_pdf_continuous(self):
        # Hardcode here the bw that will be calculated is we had used
        # ``bw='cv_ml'``.  That calculation is slow, and tested in other tests.
        bw_cv_ml = np.array([0.010043, 12095254.7]) # TODO: odd numbers (?!)
        dens = nparam.KDEMultivariateConditional(endog=[self.growth],
                                                 exog=[self.Italy_gdp],
                                                 dep_type='c', indep_type='c',
                                                 bw=bw_cv_ml)
        sm_result = np.squeeze(dens.pdf()[0:5])
        R_result = [11.97964, 12.73290, 13.23037, 13.46438, 12.22779]
        npt.assert_allclose(sm_result, R_result, atol=1e-3)

    @pytest.mark.slow
    def test_pdf_mixeddata(self):
        dens = nparam.KDEMultivariateConditional(endog=[self.Italy_gdp],
                                                 exog=[self.Italy_year],
                                                 dep_type='c', indep_type='o',
                                                 bw='cv_ls')
        sm_result = np.squeeze(dens.pdf()[0:5])
        #R_result = [0.08469226, 0.01737731, 0.05679909, 0.09744726, 0.15086674]
        expected = [0.08592089, 0.0193275, 0.05310327, 0.09642667, 0.171954]

        ## CODE TO REPRODUCE IN R
        ## library(np)
        ## data (Italy)
        ## bw <- npcdensbw(formula =
        ## Italy$gdp[1:50]~ordered(Italy$year[1:50]),bwmethod='cv.ls')
        ## fhat <- fitted(npcdens(bws=bw))
        ## fhat[1:5]
        npt.assert_allclose(sm_result, expected, atol=0, rtol=1e-5)

    def test_continuous_normal_ref(self):
        # test for normal reference rule of thumb with continuous data
        dens_nm = nparam.KDEMultivariateConditional(endog=[self.Italy_gdp],
                                                    exog=[self.growth],
                                                    dep_type='c',
                                                    indep_type='c',
                                                    bw='normal_reference')
        sm_result = dens_nm.bw
        R_result = [1.283532, 0.01535401]
        # TODO: here we need a smaller tolerance.check!
        npt.assert_allclose(sm_result, R_result, atol=1e-1)

        # test default bandwidth method, should be normal_reference
        dens_nm2 = nparam.KDEMultivariateConditional(endog=[self.Italy_gdp],
                                                    exog=[self.growth],
                                                    dep_type='c',
                                                    indep_type='c',
                                                    bw=None)
        assert_allclose(dens_nm2.bw, dens_nm.bw, rtol=1e-10)
        assert_equal(dens_nm2._bw_method, 'normal_reference')
        # check repr works #3125
        repr(dens_nm2)

    def test_continuous_cdf(self):
        dens_nm = nparam.KDEMultivariateConditional(endog=[self.Italy_gdp],
                                                    exog=[self.growth],
                                                    dep_type='c',
                                                    indep_type='c',
                                                    bw='normal_reference')
        sm_result = dens_nm.cdf()[0:5]
        R_result = [0.81304920, 0.95046942, 0.86878727, 0.71961748, 0.38685423]
        npt.assert_allclose(sm_result, R_result, atol=1e-3)

    @pytest.mark.slow
    def test_mixeddata_cdf(self):
        dens = nparam.KDEMultivariateConditional(endog=[self.Italy_gdp],
                                                 exog=[self.Italy_year],
                                                 dep_type='c',
                                                 indep_type='o',
                                                 bw='cv_ls')
        sm_result = dens.cdf()[0:5]
        #R_result = [0.8118257, 0.9724863, 0.8843773, 0.7720359, 0.4361867]
        expected = [0.83378885, 0.97684477, 0.90655143, 0.79393161, 0.43629083]
        npt.assert_allclose(sm_result, expected, atol=0, rtol=1e-5)

    @pytest.mark.slow
    def test_continuous_cvml_efficient(self):
        nobs = 500
        np.random.seed(12345)
        ovals = np.random.binomial(2, 0.5, size=(nobs, ))
        C1 = np.random.normal(size=(nobs, ))
        noise = np.random.normal(size=(nobs, ))
        b0 = 3
        b1 = 1.2
        b2 = 3.7  # regression coefficients
        Y = b0+ b1 * C1 + b2*ovals  + noise

        dens_efficient = nparam.KDEMultivariateConditional(endog=[Y],
            exog=[C1], dep_type='c', indep_type='c', bw='cv_ml',
            defaults=nparam.EstimatorSettings(efficient=True, n_sub=50))

        #dens = nparam.KDEMultivariateConditional(endog=[Y], exog=[C1],
        #                   dep_type='c', indep_type='c', bw='cv_ml')
        #bw = dens.bw
        bw_expected = np.array([0.73387, 0.43715])
        npt.assert_allclose(dens_efficient.bw, bw_expected, atol=0, rtol=1e-3)

    def test_efficient_user_specified_bw(self):
        nobs = 400
        np.random.seed(12345)
        C1 = np.random.normal(size=(nobs, ))
        C2 = np.random.normal(2, 1, size=(nobs, ))
        bw_user=[0.23, 434697.22]

        dens = nparam.KDEMultivariate(data=[C1, C2], var_type='cc',
            bw=bw_user, defaults=nparam.EstimatorSettings(efficient=True,
                                                          randomize=False,
                                                          n_sub=100))
        npt.assert_equal(dens.bw, bw_user)


@pytest.mark.parametrize("kernel", ["biw", "cos", "epa", "gau",
                                    "tri", "triw", "uni"])
def test_all_kernels(kernel, reset_randomstate):
    data = np.random.normal(size=200)
    x_grid = np.linspace(min(data), max(data), 200)
    density = sm.nonparametric.KDEUnivariate(data)
    density.fit(kernel="gau", fft=False)
    assert isinstance(density.evaluate(x_grid), np.ndarray)
