# -*- coding: utf-8 -*-
"""Test Johansen's Cointegration test against jplv, Spatial Econometrics Toolbox

Created on Thu Aug 30 21:51:08 2012
Author: Josef Perktold

"""
import os
import warnings

import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
import pandas as pd
import pytest

from statsmodels.tools.sm_exceptions import HypothesisTestWarning
from statsmodels.tsa.vector_ar.vecm import coint_johansen

current_path = os.path.dirname(os.path.abspath(__file__))
dta_path = os.path.join(current_path, "Matlab_results", "test_coint.csv")
with open(dta_path, "rb") as fd:
    dta = np.genfromtxt(fd)


class CheckCointJoh:

    def test_basic(self):
        assert_equal(self.res.ind, np.arange(len(self.res.ind), dtype=int))
        assert_equal(self.res.r0t.shape, (self.nobs_r, 8))

    def test_table_trace(self):
        table1 = np.column_stack((self.res.lr1, self.res.cvt))
        assert_almost_equal(table1,
                            self.res1_m.reshape(table1.shape, order='F'))

    def test_table_maxeval(self):
        table2 = np.column_stack((self.res.lr2, self.res.cvm))
        assert_almost_equal(table2,
                            self.res2_m.reshape(table2.shape, order='F'))

    def test_normalization(self):
        # GH 5517
        evec = self.res.evec
        non_zero = evec.flat != 0
        assert evec.flat[non_zero][0] > 0


class TestCointJoh12(CheckCointJoh):

    @classmethod
    def setup_class(cls):
        cls.res = coint_johansen(dta, 1, 2)
        cls.nobs_r = 173 - 1 - 2

        cls.res1_m = np.array([241.985452556075,  166.4781461662553,  110.3298006342814,  70.79801574443575,  44.90887371527634,  27.22385073668511,  11.74205493173769,  3.295435325623445,           169.0618,           133.7852,           102.4674,            75.1027,            51.6492,            32.0645,            16.1619,             2.7055,           175.1584,            139.278,           107.3429,  79.34220000000001,            55.2459,            35.0116,            18.3985,             3.8415,           187.1891,           150.0778,           116.9829,            87.7748,            62.5202,            41.0815,            23.1485,             6.6349])
        cls.res2_m = np.array([75.50730638981975,  56.14834553197396,   39.5317848898456,   25.8891420291594,  17.68502297859124,  15.48179580494741,  8.446619606114249,  3.295435325623445,            52.5858,            46.5583,            40.5244,            34.4202,            28.2398,            21.8731,            15.0006,             2.7055,            55.7302,            49.5875,            43.4183,            37.1646,            30.8151,            24.2522,            17.1481,             3.8415,            62.1741,            55.8171,            49.4095,            42.8612,             36.193,            29.2631,            21.7465,             6.6349,])

        evec = np.array([
            0.01102517075074406, -0.2185481584930077, 0.04565819524210763, -0.06556394587400775,
            0.04711496306104131, -0.1500111976629196, 0.03775327003706507, 0.03479475877437702,

            0.007517888890275335, -0.2014629352546497, 0.01526001455616041, 0.0707900418057458,
            -0.002388919695513273, 0.04486516694838273, -0.02936314422571188, 0.009900554050392113,

            0.02846074144367176, 0.02021385478834498, -0.04276914888645468, 0.1738024290422287,
            0.07821155002012749, -0.1066523077111768, -0.3011042488399306, 0.04965189679477353,

            0.07141291326159237, -0.01406702689857725, -0.07842109866080313, -0.04773566072362181,
            -0.04768640728128824, -0.04428737926285261, 0.4143225656833862, 0.04512787132114879,

            -0.06817130121837202, 0.2246249779872569, -0.009356548567565763, 0.006685350535849125,
            -0.02040894506833539, 0.008131690308487425, -0.2503209797396666, 0.01560186979508953,

            0.03327070126502506, -0.263036624535624, -0.04669882107497259, 0.0146457545413255,
            0.01408691619062709, 0.1004753600191269, -0.02239205763487946, -0.02169291468272568,

            0.08782313160608619, -0.07696508791577318, 0.008925177304198475, -0.06230900392092828,
            -0.01548907461158638, 0.04574831652028973, -0.2972228156126774, 0.003469819004961912,

            -0.001868995544352928, 0.05993345996347871, 0.01213394328069316, 0.02096614212178651,
            -0.08624395993789938, 0.02108183181049973, -0.08470307289295617, -5.135072530480897e-005])
        cls.evec_m = evec.reshape(cls.res.evec.shape, order='F')

        cls.eig_m = np.array([0.3586376068088151, 0.2812806889719111, 0.2074818815675726,  0.141259991767926, 0.09880133062878599, 0.08704563854307619,  0.048471840356709, 0.01919823444066367])

    def test_evec(self):
        for col in range(self.evec_m.shape[1]):
            try:
                assert_almost_equal(self.res.evec[:, col],
                                    self.evec_m[:, col])
            except AssertionError:
                assert_almost_equal(self.res.evec[:, col],
                                    -self.evec_m[:, col])

    def test_evals(self):
        assert_almost_equal(self.res.eig, self.eig_m)


class TestCointJoh09(CheckCointJoh):

    @classmethod
    def setup_class(cls):
        cls.res = coint_johansen(dta, 0, 9)
        cls.nobs_r = 173 - 1 - 9
        #fprintf(1, '%18.16g, ', r1)
        cls.res1_m = np.array([307.6888935095814,  205.3839229398245,  129.1330243009336,   83.3101865760208,  52.51955460357912,  30.20027050520502,  13.84158157562689, 0.4117390188204866,           153.6341,           120.3673,             91.109,            65.8202,            44.4929,            27.0669,            13.4294,             2.7055,            159.529,           125.6185,            95.7542,            69.8189,            47.8545,            29.7961,            15.4943,             3.8415,           171.0905,           135.9825,           104.9637,            77.8202,            54.6815,            35.4628,            19.9349,             6.6349])
        #r2 = [res.lr2 res.cvm]
        cls.res2_m = np.array([102.3049705697569,  76.25089863889085,  45.82283772491284,   30.7906319724417,  22.31928409837409,  16.35868892957814,   13.4298425568064, 0.4117390188204866,            49.2855,            43.2947,            37.2786,            31.2379,            25.1236,            18.8928,            12.2971,             2.7055,            52.3622,            46.2299,            40.0763,            33.8777,            27.5858,            21.1314,            14.2639,             3.8415,            58.6634,            52.3069,            45.8662,            39.3693,            32.7172,             25.865,              18.52,             6.6349])


class TestCointJohMin18(CheckCointJoh):

    @classmethod
    def setup_class(cls):
        cls.res = coint_johansen(dta, -1, 8)
        cls.nobs_r = 173 - 1 - 8

        cls.res1_m = np.array([260.6786029744658,  162.7966072512681,  105.8253545950566,  71.16133060790817,  47.68490211260372,  28.11843682526138,  13.03968537077271,   2.25398078597622,           137.9954,           106.7351,            79.5329,            56.2839,            37.0339,            21.7781,            10.4741,             2.9762,           143.6691,           111.7797,            83.9383,            60.0627,            40.1749,            24.2761,            12.3212,             4.1296,           154.7977,           121.7375,            92.7136,  67.63670000000001,            46.5716,            29.5147,             16.364,             6.9406])
        cls.res2_m = np.array([97.88199572319769,  56.97125265621156,  34.66402398714837,  23.47642849530445,  19.56646528734234,  15.07875145448866,   10.7857045847965,   2.25398078597622,             45.893,            39.9085,            33.9271,             27.916,             21.837,            15.7175,             9.4748,             2.9762,            48.8795,            42.7679,            36.6301,            30.4428,            24.1592,            17.7961,            11.2246,             4.1296,            55.0335,            48.6606,            42.2333,            35.7359,            29.0609,            22.2519,            15.0923,             6.9406])


class TestCointJoh25(CheckCointJoh):

    @classmethod
    def setup_class(cls):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=HypothesisTestWarning)
            cls.res = coint_johansen(dta, 2, 5)
        cls.nobs_r = 173 - 1 - 5

        # Note: critical values not available if trend>1
        cls.res1_m = np.array([270.1887263915158, 171.6870096307863,
                               107.8613367358704, 70.82424032233558,
                               44.62551818267534, 25.74352073857572,
                               14.17882426926978, 4.288656185006764,
                               0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0])
        cls.res1_m[cls.res1_m == 0] = np.nan
        cls.res2_m = np.array([98.50171676072955, 63.82567289491584,
                               37.03709641353485, 26.19872213966024,
                               18.88199744409963, 11.56469646930594,
                               9.890168084263012, 4.288656185006764,
                               0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0])
        cls.res2_m[cls.res2_m == 0] = np.nan


@pytest.mark.smoke
def test_coint_johansen_0lag(reset_randomstate):
    # GH 5731
    x_diff = np.random.normal(0, 1, 1000)
    x = pd.Series(np.cumsum(x_diff))
    e1 = np.random.normal(0, 1, 1000)
    y = x + 5 + e1
    data = pd.concat([x, y], axis=1)
    result = coint_johansen(data, det_order=-1, k_ar_diff=0)
    assert result.eig.shape == (2,)
