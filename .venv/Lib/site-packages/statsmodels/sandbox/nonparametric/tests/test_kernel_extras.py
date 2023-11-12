import numpy as np
import numpy.testing as npt

from statsmodels.sandbox.nonparametric.kernel_extras import SemiLinear


class KernelExtrasTestBase:
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


class TestSemiLinear(KernelExtrasTestBase):

    def test_basic(self):
        nobs = 300
        np.random.seed(1234)
        C1 = np.random.normal(0,2, size=(nobs, ))
        C2 = np.random.normal(2, 1, size=(nobs, ))
        e = np.random.normal(size=(nobs, ))
        b1 = 1.3
        b2 = -0.7
        Y = b1 * C1 + np.exp(b2 * C2) + e
        model = SemiLinear(endog=[Y], exog=[C1], exog_nonparametric=[C2],
                           var_type='c', k_linear=1)
        b_hat = np.squeeze(model.b)
        # Only tests for the linear part of the regression
        # Currently does not work well with the nonparametric part
        # Needs some more work
        npt.assert_allclose(b1, b_hat, rtol=0.1)
