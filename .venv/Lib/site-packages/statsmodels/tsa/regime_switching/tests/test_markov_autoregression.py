"""
Tests for Markov Autoregression models

Author: Chad Fulton
License: BSD-3
"""

import warnings
import os

import numpy as np
from numpy.testing import assert_equal, assert_allclose
import pandas as pd
import pytest

from statsmodels.tools import add_constant
from statsmodels.tsa.regime_switching import markov_autoregression

current_path = os.path.dirname(os.path.abspath(__file__))


rgnp = [2.59316421, 2.20217133, 0.45827562, 0.9687438,
        -0.24130757, 0.89647478, 2.05393219, 1.73353648,
        0.93871289, -0.46477833, -0.80983406, -1.39763689,
        -0.39886093, 1.1918416, 1.45620048, 2.11808228,
        1.08957863, 1.32390273, 0.87296367, -0.19773273,
        0.45420215, 0.07221876, 1.1030364, 0.82097489,
        -0.05795795, 0.58447772, -1.56192672, -2.05041027,
        0.53637183, 2.33676839, 2.34014559, 1.2339263,
        1.8869648, -0.45920792, 0.84940469, 1.70139849,
        -0.28756312, 0.09594627, -0.86080289, 1.03447127,
        1.23685944, 1.42004502, 2.22410631, 1.30210173,
        1.03517699, 0.9253425, -0.16559951, 1.3444382,
        1.37500131, 1.73222184, 0.71605635, 2.21032143,
        0.85333031, 1.00238776, 0.42725441, 2.14368343,
        1.43789184, 1.57959926, 2.27469826, 1.95962656,
        0.25992399, 1.01946914, 0.49016398, 0.5636338,
        0.5959546, 1.43082857, 0.56230122, 1.15388393,
        1.68722844, 0.77438205, -0.09647045, 1.39600146,
        0.13646798, 0.55223715, -0.39944872, -0.61671102,
        -0.08722561, 1.2101835, -0.90729755, 2.64916158,
        -0.0080694, 0.51111895, -0.00401437, 2.16821432,
        1.92586732, 1.03504717, 1.85897219, 2.32004929,
        0.25570789, -0.09855274, 0.89073682, -0.55896485,
        0.28350255, -1.31155407, -0.88278776, -1.97454941,
        1.01275265, 1.68264723, 1.38271284, 1.86073637,
        0.4447377, 0.41449001, 0.99202275, 1.36283576,
        1.59970522, 1.98845816, -0.25684232, 0.87786949,
        3.1095655, 0.85324478, 1.23337317, 0.00314302,
        -0.09433369, 0.89883322, -0.19036628, 0.99772376,
        -2.39120054, 0.06649673, 1.26136017, 1.91637838,
        -0.3348029, 0.44207108, -1.40664911, -1.52129889,
        0.29919869, -0.80197448, 0.15204792, 0.98585027,
        2.13034606, 1.34397924, 1.61550522, 2.70930099,
        1.24461412, 0.50835466, 0.14802167]

rec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
       1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]


def test_predict():
    # AR(1) without mean, k_regimes=2
    endog = np.ones(10)
    markov_autoregression.MarkovAutoregression(
        endog,
        k_regimes=2,
        order=1,
        trend='n'
    )
    mod = markov_autoregression.MarkovAutoregression(
        endog, k_regimes=2, order=1, trend='n')
    assert_equal(mod.nobs, 9)
    assert_equal(mod.endog, np.ones(9))

    params = np.r_[0.5, 0.5, 1., 0.1, 0.5]
    mod_resid = mod._resid(params)
    resids = np.zeros((2, 2, mod.nobs))
    # Resids when: S_{t} = 0
    resids[0, :, :] = np.ones(9) - 0.1 * np.ones(9)
    assert_allclose(mod_resid[0, :, :], resids[0, :, :])
    # Resids when: S_{t} = 1
    resids[1, :, :] = np.ones(9) - 0.5 * np.ones(9)
    assert_allclose(mod_resid[1, :, :], resids[1, :, :])

    # AR(1) with mean, k_regimes=2
    endog = np.arange(10)
    mod = markov_autoregression.MarkovAutoregression(
        endog, k_regimes=2, order=1)
    assert_equal(mod.nobs, 9)
    assert_equal(mod.endog, np.arange(1, 10))

    params = np.r_[0.5, 0.5, 2., 3., 1., 0.1, 0.5]
    mod_resid = mod._resid(params)
    resids = np.zeros((2, 2, mod.nobs))
    # Resids when: S_t = 0, S_{t-1} = 0
    resids[0, 0, :] = (np.arange(1, 10) - 2.) - 0.1 * (np.arange(9) - 2.)
    assert_allclose(mod_resid[0, 0, :], resids[0, 0, :])
    # Resids when: S_t = 0, S_{t-1} = 1
    resids[0, 1, :] = (np.arange(1, 10) - 2.) - 0.1 * (np.arange(9) - 3.)
    assert_allclose(mod_resid[0, 1, :], resids[0, 1, :])
    # Resids when: S_t = 1, S_{t-1} = 0
    resids[1, 0, :] = (np.arange(1, 10) - 3.) - 0.5 * (np.arange(9) - 2.)
    assert_allclose(mod_resid[1, 0, :], resids[1, 0, :])
    # Resids when: S_t = 1, S_{t-1} = 1
    resids[1, 1, :] = (np.arange(1, 10) - 3.) - 0.5 * (np.arange(9) - 3.)
    assert_allclose(mod_resid[1, 1, :], resids[1, 1, :])

    # AR(2) with mean, k_regimes=3
    endog = np.arange(10)
    mod = markov_autoregression.MarkovAutoregression(
        endog, k_regimes=3, order=2)
    assert_equal(mod.nobs, 8)
    assert_equal(mod.endog, np.arange(2, 10))

    params = np.r_[[0.3] * 6, 2., 3., 4, 1., 0.1, 0.5, 0.8, -0.05, -0.25, -0.4]
    mod_resid = mod._resid(params)
    resids = np.zeros((3, 3, 3, mod.nobs))
    # Resids when: S_t = 0, S_{t-1} = 0, S_{t-2} = 0
    resids[0, 0, 0, :] = (
        (np.arange(2, 10) - 2.) -
        0.1 * (np.arange(1, 9) - 2.) -
        (-0.05) * (np.arange(8) - 2.))
    assert_allclose(mod_resid[0, 0, 0, :], resids[0, 0, 0, :])

    # Resids when: S_t = 1, S_{t-1} = 0, S_{t-2} = 0
    resids[1, 0, 0, :] = (
        (np.arange(2, 10) - 3.) -
        0.5 * (np.arange(1, 9) - 2.) -
        (-0.25) * (np.arange(8) - 2.))
    assert_allclose(mod_resid[1, 0, 0, :], resids[1, 0, 0, :])

    # Resids when: S_t = 0, S_{t-1} = 2, S_{t-2} = 1
    resids[0, 2, 1, :] = (
        (np.arange(2, 10) - 2.) -
        0.1 * (np.arange(1, 9) - 4.) -
        (-0.05) * (np.arange(8) - 3.))
    assert_allclose(mod_resid[0, 2, 1, :], resids[0, 2, 1, :])

    # AR(1) with mean + non-switching exog
    endog = np.arange(10)
    exog = np.r_[0.4, 5, 0.2, 1.2, -0.3, 2.5, 0.2, -0.7, 2., -1.1]
    mod = markov_autoregression.MarkovAutoregression(
        endog, k_regimes=2, order=1, exog=exog)
    assert_equal(mod.nobs, 9)
    assert_equal(mod.endog, np.arange(1, 10))

    params = np.r_[0.5, 0.5, 2., 3., 1.5, 1., 0.1, 0.5]
    mod_resid = mod._resid(params)
    resids = np.zeros((2, 2, mod.nobs))
    # Resids when: S_t = 0, S_{t-1} = 0
    resids[0, 0, :] = (
        (np.arange(1, 10) - 2. - 1.5 * exog[1:]) -
        0.1 * (np.arange(9) - 2. - 1.5 * exog[:-1]))
    assert_allclose(mod_resid[0, 0, :], resids[0, 0, :])
    # Resids when: S_t = 0, S_{t-1} = 1
    resids[0, 1, :] = (
        (np.arange(1, 10) - 2. - 1.5 * exog[1:]) -
        0.1 * (np.arange(9) - 3. - 1.5 * exog[:-1]))
    assert_allclose(mod_resid[0, 1, :], resids[0, 1, :])
    # Resids when: S_t = 1, S_{t-1} = 0
    resids[1, 0, :] = (
        (np.arange(1, 10) - 3. - 1.5 * exog[1:]) -
        0.5 * (np.arange(9) - 2. - 1.5 * exog[:-1]))
    assert_allclose(mod_resid[1, 0, :], resids[1, 0, :])
    # Resids when: S_t = 1, S_{t-1} = 1
    resids[1, 1, :] = (
        (np.arange(1, 10) - 3. - 1.5 * exog[1:]) -
        0.5 * (np.arange(9) - 3. - 1.5 * exog[:-1]))
    assert_allclose(mod_resid[1, 1, :], resids[1, 1, :])


def test_conditional_loglikelihoods():
    # AR(1) without mean, k_regimes=2, non-switching variance
    endog = np.ones(10)
    mod = markov_autoregression.MarkovAutoregression(
        endog, k_regimes=2, order=1)
    assert_equal(mod.nobs, 9)
    assert_equal(mod.endog, np.ones(9))

    params = np.r_[0.5, 0.5, 2., 3., 2., 0.1, 0.5]
    resid = mod._resid(params)
    conditional_likelihoods = (
        np.exp(-0.5 * resid**2 / 2) / np.sqrt(2 * np.pi * 2))
    assert_allclose(mod._conditional_loglikelihoods(params),
                    np.log(conditional_likelihoods))

    # AR(1) without mean, k_regimes=3, switching variance
    endog = np.ones(10)
    mod = markov_autoregression.MarkovAutoregression(
        endog, k_regimes=3, order=1, switching_variance=True)
    assert_equal(mod.nobs, 9)
    assert_equal(mod.endog, np.ones(9))

    params = np.r_[[0.3]*6, 2., 3., 4., 1.5, 3., 4.5, 0.1, 0.5, 0.8]
    mod_conditional_loglikelihoods = mod._conditional_loglikelihoods(params)
    conditional_likelihoods = mod._resid(params)

    # S_t = 0
    conditional_likelihoods[0, :, :] = (
        np.exp(-0.5 * conditional_likelihoods[0, :, :]**2 / 1.5) /
        np.sqrt(2 * np.pi * 1.5))
    assert_allclose(mod_conditional_loglikelihoods[0, :, :],
                    np.log(conditional_likelihoods[0, :, :]))
    # S_t = 1
    conditional_likelihoods[1, :, :] = (
        np.exp(-0.5 * conditional_likelihoods[1, :, :]**2 / 3.) /
        np.sqrt(2 * np.pi * 3.))
    assert_allclose(mod_conditional_loglikelihoods[1, :, :],
                    np.log(conditional_likelihoods[1, :, :]))
    # S_t = 2
    conditional_likelihoods[2, :, :] = (
        np.exp(-0.5 * conditional_likelihoods[2, :, :]**2 / 4.5) /
        np.sqrt(2 * np.pi * 4.5))
    assert_allclose(mod_conditional_loglikelihoods[2, :, :],
                    np.log(conditional_likelihoods[2, :, :]))


class MarkovAutoregression:
    @classmethod
    def setup_class(cls, true, endog, atol=1e-5, rtol=1e-7, **kwargs):
        cls.model = markov_autoregression.MarkovAutoregression(endog, **kwargs)
        cls.true = true
        cls.result = cls.model.smooth(cls.true['params'])
        cls.atol = atol
        cls.rtol = rtol

    def test_llf(self):
        assert_allclose(self.result.llf, self.true['llf'], atol=self.atol,
                        rtol=self.rtol)

    def test_fit(self, **kwargs):
        # Test fitting against Stata
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = self.model.fit(disp=False, **kwargs)
        assert_allclose(res.llf, self.true['llf_fit'], atol=self.atol,
                        rtol=self.rtol)

    @pytest.mark.smoke
    def test_fit_em(self, **kwargs):
        # Test EM fitting (smoke test)
        res_em = self.model._fit_em(**kwargs)
        assert_allclose(res_em.llf, self.true['llf_fit_em'], atol=self.atol,
                        rtol=self.rtol)


hamilton_ar2_short_filtered_joint_probabilities = np.array([
         [[[4.99506987e-02,   6.44048275e-04,   6.22227140e-05,
            4.45756755e-06,   5.26645567e-07,   7.99846146e-07,
            1.19425705e-05,   6.87762063e-03],
           [1.95930395e-02,   3.25884335e-04,   1.12955091e-04,
            3.38537103e-04,   9.81927968e-06,   2.71696750e-05,
            5.83828290e-03,   7.64261509e-02]],

          [[1.97113193e-03,   9.50372207e-05,   1.98390978e-04,
            1.88188953e-06,   4.83449400e-07,   1.14872860e-05,
            4.02918239e-06,   4.35015431e-04],
           [2.24870443e-02,   1.27331172e-03,   9.62155856e-03,
            4.04178695e-03,   2.75516282e-04,   1.18179572e-02,
            5.99778157e-02,   1.48149567e-01]]],


         [[[6.70912859e-02,   1.84223872e-02,   2.55621792e-04,
            4.48500688e-05,   7.80481515e-05,   2.73734559e-06,
            7.59835896e-06,   1.42930726e-03],
           [2.10053328e-02,   7.44036383e-03,   3.70388879e-04,
            2.71878370e-03,   1.16152088e-03,   7.42182691e-05,
            2.96490192e-03,   1.26774695e-02]],

          [[8.09335679e-02,   8.31016518e-02,   2.49149080e-02,
            5.78825626e-04,   2.19019941e-03,   1.20179130e-03,
            7.83659430e-05,   2.76363377e-03],
           [7.36967899e-01,   8.88697316e-01,   9.64463954e-01,
            9.92270877e-01,   9.96283886e-01,   9.86863839e-01,
            9.31117063e-01,   7.51241236e-01]]]])


hamilton_ar2_short_predicted_joint_probabilities = np.array([[
          [[[1.20809334e-01,   3.76964436e-02,   4.86045844e-04,
             4.69578023e-05,   3.36400588e-06,   3.97445190e-07,
             6.03622290e-07,   9.01273552e-06],
            [3.92723623e-02,   1.47863379e-02,   2.45936108e-04,
             8.52441571e-05,   2.55484811e-04,   7.41034525e-06,
             2.05042201e-05,   4.40599447e-03]],

           [[4.99131230e-03,   1.48756005e-03,   7.17220245e-05,
             1.49720314e-04,   1.42021122e-06,   3.64846209e-07,
             8.66914462e-06,   3.04071516e-06],
            [4.70476003e-02,   1.69703652e-02,   9.60933974e-04,
             7.26113047e-03,   3.05022748e-03,   2.07924699e-04,
             8.91869322e-03,   4.52636381e-02]]],


          [[[4.99131230e-03,   6.43506069e-03,   1.76698327e-03,
             2.45179642e-05,   4.30179435e-06,   7.48598845e-06,
             2.62552503e-07,   7.28796600e-07],
            [1.62256192e-03,   2.01472650e-03,   7.13642497e-04,
             3.55258493e-05,   2.60772139e-04,   1.11407276e-04,
             7.11864528e-06,   2.84378568e-04]],

           [[5.97950448e-03,   7.76274317e-03,   7.97069493e-03,
             2.38971340e-03,   5.55180599e-05,   2.10072977e-04,
             1.15269812e-04,   7.51646942e-06],
            [5.63621989e-02,   7.06862760e-02,   8.52394030e-02,
             9.25065601e-02,   9.51736612e-02,   9.55585689e-02,
             9.46550451e-02,   8.93080931e-02]]]],



         [[[[3.92723623e-02,   1.22542551e-02,   1.58002431e-04,
             1.52649118e-05,   1.09356167e-06,   1.29200377e-07,
             1.96223855e-07,   2.92983500e-06],
            [1.27665503e-02,   4.80670161e-03,   7.99482261e-05,
             2.77109335e-05,   8.30522919e-05,   2.40893443e-06,
             6.66545485e-06,   1.43228843e-03]],

           [[1.62256192e-03,   4.83571884e-04,   2.33151963e-05,
             4.86706634e-05,   4.61678312e-07,   1.18603191e-07,
             2.81814142e-06,   9.88467229e-07],
            [1.52941031e-02,   5.51667911e-03,   3.12377744e-04,
             2.36042810e-03,   9.91559466e-04,   6.75915830e-05,
             2.89926399e-03,   1.47141776e-02]]],


          [[[4.70476003e-02,   6.06562252e-02,   1.66554040e-02,
             2.31103828e-04,   4.05482745e-05,   7.05621631e-05,
             2.47479309e-06,   6.86956236e-06],
            [1.52941031e-02,   1.89906063e-02,   6.72672133e-03,
             3.34863029e-04,   2.45801156e-03,   1.05011361e-03,
             6.70996238e-05,   2.68052335e-03]],

           [[5.63621989e-02,   7.31708248e-02,   7.51309569e-02,
             2.25251946e-02,   5.23307566e-04,   1.98012644e-03,
             1.08652148e-03,   7.08494735e-05],
            [5.31264334e-01,   6.66281623e-01,   8.03457913e-01,
             8.71957394e-01,   8.97097216e-01,   9.00725317e-01,
             8.92208794e-01,   8.41808970e-01]]]]])


hamilton_ar2_short_smoothed_joint_probabilities = np.array([
         [[[1.29898189e-02,   1.66298475e-04,   1.29822987e-05,
            9.95268382e-07,   1.84473346e-07,   7.18761267e-07,
            1.69576494e-05,   6.87762063e-03],
           [5.09522472e-03,   8.41459714e-05,   2.35672254e-05,
            7.55872505e-05,   3.43949612e-06,   2.44153330e-05,
            8.28997024e-03,   7.64261509e-02]],

          [[5.90021731e-04,   2.55342733e-05,   4.50698224e-05,
            5.30734135e-07,   1.80741761e-07,   1.11483792e-05,
            5.98539007e-06,   4.35015431e-04],
           [6.73107901e-03,   3.42109009e-04,   2.18579464e-03,
            1.13987259e-03,   1.03004157e-04,   1.14692946e-02,
            8.90976350e-02,   1.48149567e-01]]],


         [[[6.34648123e-02,   1.79187451e-02,   2.37462147e-04,
            3.55542558e-05,   7.63980455e-05,   2.90520820e-06,
            8.17644492e-06,   1.42930726e-03],
           [1.98699352e-02,   7.23695477e-03,   3.44076057e-04,
            2.15527721e-03,   1.13696383e-03,   7.87695658e-05,
            3.19047276e-03,   1.26774695e-02]],

          [[8.81925054e-02,   8.33092133e-02,   2.51106301e-02,
            5.81007470e-04,   2.19065072e-03,   1.20221350e-03,
            7.56893839e-05,   2.76363377e-03],
           [8.03066603e-01,   8.90916999e-01,   9.72040418e-01,
            9.96011175e-01,   9.96489179e-01,   9.87210535e-01,
            8.99315113e-01,   7.51241236e-01]]]])


class TestHamiltonAR2Short(MarkovAutoregression):
    # This is just a set of regression tests
    @classmethod
    def setup_class(cls):
        true = {
            'params': np.r_[0.754673, 0.095915, -0.358811, 1.163516,
                            np.exp(-0.262658)**2, 0.013486, -0.057521],
            'llf': -10.14066,
            'llf_fit': -4.0523073,
            'llf_fit_em': -8.885836
        }
        super(TestHamiltonAR2Short, cls).setup_class(
            true, rgnp[-10:], k_regimes=2, order=2, switching_ar=False)

    def test_fit_em(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            super(TestHamiltonAR2Short, self).test_fit_em()

    def test_filter_output(self, **kwargs):
        res = self.result

        # Filtered
        assert_allclose(res.filtered_joint_probabilities,
                        hamilton_ar2_short_filtered_joint_probabilities)

        # Predicted
        desired = hamilton_ar2_short_predicted_joint_probabilities
        if desired.ndim > res.predicted_joint_probabilities.ndim:
            desired = desired.sum(axis=-2)
        assert_allclose(res.predicted_joint_probabilities, desired)

    def test_smoother_output(self, **kwargs):
        res = self.result

        # Filtered
        assert_allclose(res.filtered_joint_probabilities,
                        hamilton_ar2_short_filtered_joint_probabilities)

        # Predicted
        desired = hamilton_ar2_short_predicted_joint_probabilities
        if desired.ndim > res.predicted_joint_probabilities.ndim:
            desired = desired.sum(axis=-2)
        assert_allclose(res.predicted_joint_probabilities, desired)

        # Smoothed, entry-by-entry
        assert_allclose(
            res.smoothed_joint_probabilities[..., -1],
            hamilton_ar2_short_smoothed_joint_probabilities[..., -1])
        assert_allclose(
            res.smoothed_joint_probabilities[..., -2],
            hamilton_ar2_short_smoothed_joint_probabilities[..., -2])
        assert_allclose(
            res.smoothed_joint_probabilities[..., -3],
            hamilton_ar2_short_smoothed_joint_probabilities[..., -3])
        assert_allclose(
            res.smoothed_joint_probabilities[..., :-3],
            hamilton_ar2_short_smoothed_joint_probabilities[..., :-3])


hamilton_ar4_filtered = [
    0.776712, 0.949192, 0.996320, 0.990258, 0.940111, 0.537442,
    0.140001, 0.008942, 0.048480, 0.614097, 0.910889, 0.995463,
    0.979465, 0.992324, 0.984561, 0.751038, 0.776268, 0.522048,
    0.814956, 0.821786, 0.472729, 0.673567, 0.029031, 0.001556,
    0.433276, 0.985463, 0.995025, 0.966067, 0.998445, 0.801467,
    0.960997, 0.996431, 0.461365, 0.199357, 0.027398, 0.703626,
    0.946388, 0.985321, 0.998244, 0.989567, 0.984510, 0.986811,
    0.793788, 0.973675, 0.984848, 0.990418, 0.918427, 0.998769,
    0.977647, 0.978742, 0.927635, 0.998691, 0.988934, 0.991654,
    0.999288, 0.999073, 0.918636, 0.987710, 0.966876, 0.910015,
    0.826150, 0.969451, 0.844049, 0.941525, 0.993363, 0.949978,
    0.615206, 0.970915, 0.787585, 0.707818, 0.200476, 0.050835,
    0.140723, 0.809850, 0.086422, 0.990344, 0.785963, 0.817425,
    0.659152, 0.996578, 0.992860, 0.948501, 0.996883, 0.999712,
    0.906694, 0.725013, 0.963690, 0.386960, 0.241302, 0.009078,
    0.015789, 0.000896, 0.541530, 0.928686, 0.953704, 0.992741,
    0.935877, 0.918958, 0.977316, 0.987941, 0.987300, 0.996769,
    0.645469, 0.921285, 0.999917, 0.949335, 0.968914, 0.886025,
    0.777141, 0.904381, 0.368277, 0.607429, 0.002491, 0.227610,
    0.871284, 0.987717, 0.288705, 0.512124, 0.030329, 0.005177,
    0.256183, 0.020955, 0.051620, 0.549009, 0.991715, 0.987892,
    0.995377, 0.999833, 0.993756, 0.956164, 0.927714]

hamilton_ar4_smoothed = [
    0.968096, 0.991071, 0.998559, 0.958534, 0.540652, 0.072784,
    0.010999, 0.006228, 0.172144, 0.898574, 0.989054, 0.998293,
    0.986434, 0.993248, 0.976868, 0.858521, 0.847452, 0.675670,
    0.596294, 0.165407, 0.035270, 0.127967, 0.007414, 0.004944,
    0.815829, 0.998128, 0.998091, 0.993227, 0.999283, 0.921100,
    0.977171, 0.971757, 0.124680, 0.063710, 0.114570, 0.954701,
    0.994852, 0.997302, 0.999345, 0.995817, 0.996218, 0.994580,
    0.933990, 0.996054, 0.998151, 0.996976, 0.971489, 0.999786,
    0.997362, 0.996755, 0.993053, 0.999947, 0.998469, 0.997987,
    0.999830, 0.999360, 0.953176, 0.992673, 0.975235, 0.938121,
    0.946784, 0.986897, 0.905792, 0.969755, 0.995379, 0.914480,
    0.772814, 0.931385, 0.541742, 0.394596, 0.063428, 0.027829,
    0.124527, 0.286105, 0.069362, 0.995950, 0.961153, 0.962449,
    0.945022, 0.999855, 0.998943, 0.980041, 0.999028, 0.999838,
    0.863305, 0.607421, 0.575983, 0.013300, 0.007562, 0.000635,
    0.001806, 0.002196, 0.803550, 0.972056, 0.984503, 0.998059,
    0.985211, 0.988486, 0.994452, 0.994498, 0.998873, 0.999192,
    0.870482, 0.976282, 0.999961, 0.984283, 0.973045, 0.786176,
    0.403673, 0.275418, 0.115199, 0.257560, 0.004735, 0.493936,
    0.907360, 0.873199, 0.052959, 0.076008, 0.001653, 0.000847,
    0.062027, 0.021257, 0.219547, 0.955654, 0.999851, 0.997685,
    0.998324, 0.999939, 0.996858, 0.969209, 0.927714]


class TestHamiltonAR4(MarkovAutoregression):
    @classmethod
    def setup_class(cls):
        # Results from E-views:
        # Dependent variable followed by a list of switching regressors:
        #     rgnp c
        # List of non-switching regressors:
        #     ar(1) ar(2) ar(3) ar(4)
        # Do not check "Regime specific error variances"
        # Switching type: Markov
        # Number of Regimes: 2
        # Probability regressors:
        #     c
        # Method SWITCHREG
        # Sample 1951q1 1984q4
        true = {
            'params': np.r_[0.754673, 0.095915, -0.358811, 1.163516,
                            np.exp(-0.262658)**2, 0.013486, -0.057521,
                            -0.246983, -0.212923],
            'llf': -181.26339,
            'llf_fit': -181.26339,
            'llf_fit_em': -183.85444,
            'bse_oim': np.r_[.0965189, .0377362, .2645396, .0745187, np.nan,
                             .1199942, .137663, .1069103, .1105311, ]
        }
        super(TestHamiltonAR4, cls).setup_class(
            true, rgnp, k_regimes=2, order=4, switching_ar=False)

    def test_filtered_regimes(self):
        res = self.result
        assert_equal(len(res.filtered_marginal_probabilities[:, 1]),
                     self.model.nobs)
        assert_allclose(res.filtered_marginal_probabilities[:, 1],
                        hamilton_ar4_filtered, atol=1e-5)

    def test_smoothed_regimes(self):
        res = self.result
        assert_equal(len(res.smoothed_marginal_probabilities[:, 1]),
                     self.model.nobs)
        assert_allclose(res.smoothed_marginal_probabilities[:, 1],
                        hamilton_ar4_smoothed, atol=1e-5)

    def test_bse(self):
        # Cannot compare middle element of bse because we estimate sigma^2
        # rather than sigma
        bse = self.result.cov_params_approx.diagonal()**0.5
        assert_allclose(bse[:4], self.true['bse_oim'][:4], atol=1e-6)
        assert_allclose(bse[6:], self.true['bse_oim'][6:], atol=1e-6)


class TestHamiltonAR2Switch(MarkovAutoregression):
    # Results from Stata, see http://www.stata.com/manuals14/tsmswitch.pdf
    @classmethod
    def setup_class(cls):
        path = os.path.join(current_path, 'results',
                            'results_predict_rgnp.csv')
        results = pd.read_csv(path)

        true = {
            'params': np.r_[.3812383, .3564492, -.0055216, 1.195482,
                            .6677098**2, .3710719, .4621503, .7002937,
                            -.3206652],
            'llf': -179.32354,
            'llf_fit': -179.38684,
            'llf_fit_em': -184.99606,
            'bse_oim': np.r_[.1424841, .0994742, .2057086, .1225987, np.nan,
                             .1754383, .1652473, .187409, .1295937],
            'smoothed0': results.iloc[3:]['switchar2_sm1'],
            'smoothed1': results.iloc[3:]['switchar2_sm2'],
            'predict0': results.iloc[3:]['switchar2_yhat1'],
            'predict1': results.iloc[3:]['switchar2_yhat2'],
            'predict_predicted': results.iloc[3:]['switchar2_pyhat'],
            'predict_filtered': results.iloc[3:]['switchar2_fyhat'],
            'predict_smoothed': results.iloc[3:]['switchar2_syhat'],
        }
        super(TestHamiltonAR2Switch, cls).setup_class(
            true, rgnp, k_regimes=2, order=2)

    def test_smoothed_marginal_probabilities(self):
        assert_allclose(self.result.smoothed_marginal_probabilities[:, 0],
                        self.true['smoothed0'], atol=1e-6)
        assert_allclose(self.result.smoothed_marginal_probabilities[:, 1],
                        self.true['smoothed1'], atol=1e-6)

    def test_predict(self):
        # Smoothed
        actual = self.model.predict(
            self.true['params'], probabilities='smoothed')
        assert_allclose(actual, self.true['predict_smoothed'], atol=1e-6)
        actual = self.model.predict(
            self.true['params'], probabilities=None)
        assert_allclose(actual, self.true['predict_smoothed'], atol=1e-6)

        actual = self.result.predict(probabilities='smoothed')
        assert_allclose(actual, self.true['predict_smoothed'], atol=1e-6)
        actual = self.result.predict(probabilities=None)
        assert_allclose(actual, self.true['predict_smoothed'], atol=1e-6)

    def test_bse(self):
        # Cannot compare middle element of bse because we estimate sigma^2
        # rather than sigma
        bse = self.result.cov_params_approx.diagonal()**0.5
        assert_allclose(bse[:4], self.true['bse_oim'][:4], atol=1e-7)
        assert_allclose(bse[6:], self.true['bse_oim'][6:], atol=1e-7)


hamilton_ar1_switch_filtered = [
    0.840288, 0.730337, 0.900234, 0.596492, 0.921618, 0.983828,
    0.959039, 0.898366, 0.477335, 0.251089, 0.049367, 0.386782,
    0.942868, 0.965632, 0.982857, 0.897603, 0.946986, 0.916413,
    0.640912, 0.849296, 0.778371, 0.954420, 0.929906, 0.723930,
    0.891196, 0.061163, 0.004806, 0.977369, 0.997871, 0.977950,
    0.896580, 0.963246, 0.430539, 0.906586, 0.974589, 0.514506,
    0.683457, 0.276571, 0.956475, 0.966993, 0.971618, 0.987019,
    0.916670, 0.921652, 0.930265, 0.655554, 0.965858, 0.964981,
    0.976790, 0.868267, 0.983240, 0.852052, 0.919150, 0.854467,
    0.987868, 0.935840, 0.958138, 0.979535, 0.956541, 0.716322,
    0.919035, 0.866437, 0.899609, 0.914667, 0.976448, 0.867252,
    0.953075, 0.977850, 0.884242, 0.688299, 0.968461, 0.737517,
    0.870674, 0.559413, 0.380339, 0.582813, 0.941311, 0.240020,
    0.999349, 0.619258, 0.828343, 0.729726, 0.991009, 0.966291,
    0.899148, 0.970798, 0.977684, 0.695877, 0.637555, 0.915824,
    0.434600, 0.771277, 0.113756, 0.144002, 0.008466, 0.994860,
    0.993173, 0.961722, 0.978555, 0.789225, 0.836283, 0.940383,
    0.968368, 0.974473, 0.980248, 0.518125, 0.904086, 0.993023,
    0.802936, 0.920906, 0.685445, 0.666524, 0.923285, 0.643861,
    0.938184, 0.008862, 0.945406, 0.990061, 0.991500, 0.486669,
    0.805039, 0.089036, 0.025067, 0.863309, 0.352784, 0.733295,
    0.928710, 0.984257, 0.926597, 0.959887, 0.984051, 0.872682,
    0.824375, 0.780157]

hamilton_ar1_switch_smoothed = [
    0.900074, 0.758232, 0.914068, 0.637248, 0.901951, 0.979905,
    0.958935, 0.888641, 0.261602, 0.148761, 0.056919, 0.424396,
    0.932184, 0.954962, 0.983958, 0.895595, 0.949519, 0.923473,
    0.678898, 0.848793, 0.807294, 0.958868, 0.942936, 0.809137,
    0.960892, 0.032947, 0.007127, 0.967967, 0.996551, 0.979278,
    0.896181, 0.987462, 0.498965, 0.908803, 0.986893, 0.488720,
    0.640492, 0.325552, 0.951996, 0.959703, 0.960914, 0.986989,
    0.916779, 0.924570, 0.935348, 0.677118, 0.960749, 0.958966,
    0.976974, 0.838045, 0.986562, 0.847774, 0.908866, 0.821110,
    0.984965, 0.915302, 0.938196, 0.976518, 0.973780, 0.744159,
    0.922006, 0.873292, 0.904035, 0.917547, 0.978559, 0.870915,
    0.948420, 0.979747, 0.884791, 0.711085, 0.973235, 0.726311,
    0.828305, 0.446642, 0.411135, 0.639357, 0.973151, 0.141707,
    0.999805, 0.618207, 0.783239, 0.672193, 0.987618, 0.964655,
    0.877390, 0.962437, 0.989002, 0.692689, 0.699370, 0.937934,
    0.522535, 0.824567, 0.058746, 0.146549, 0.009864, 0.994072,
    0.992084, 0.956945, 0.984297, 0.795926, 0.845698, 0.935364,
    0.963285, 0.972767, 0.992168, 0.528278, 0.826349, 0.996574,
    0.811431, 0.930873, 0.680756, 0.721072, 0.937977, 0.731879,
    0.996745, 0.016121, 0.951187, 0.989820, 0.996968, 0.592477,
    0.889144, 0.036015, 0.040084, 0.858128, 0.418984, 0.746265,
    0.907990, 0.980984, 0.900449, 0.934741, 0.986807, 0.872818,
    0.812080, 0.780157]


class TestHamiltonAR1Switch(MarkovAutoregression):
    @classmethod
    def setup_class(cls):
        # Results from E-views:
        # Dependent variable followed by a list of switching regressors:
        #     rgnp c ar(1)
        # List of non-switching regressors: <blank>
        # Do not check "Regime specific error variances"
        # Switching type: Markov
        # Number of Regimes: 2
        # Probability regressors:
        #     c
        # Method SWITCHREG
        # Sample 1951q1 1984q4
        true = {
            'params': np.r_[0.85472458, 0.53662099, 1.041419, -0.479157,
                            np.exp(-0.231404)**2, 0.243128, 0.713029],
            'llf': -186.7575,
            'llf_fit': -186.7575,
            'llf_fit_em': -189.25446
        }
        super(TestHamiltonAR1Switch, cls).setup_class(
            true, rgnp, k_regimes=2, order=1)

    def test_filtered_regimes(self):
        assert_allclose(self.result.filtered_marginal_probabilities[:, 0],
                        hamilton_ar1_switch_filtered, atol=1e-5)

    def test_smoothed_regimes(self):
        assert_allclose(self.result.smoothed_marginal_probabilities[:, 0],
                        hamilton_ar1_switch_smoothed, atol=1e-5)

    def test_expected_durations(self):
        expected_durations = [6.883477, 1.863513]
        assert_allclose(self.result.expected_durations, expected_durations,
                        atol=1e-5)


hamilton_ar1_switch_tvtp_filtered = [
    0.999996, 0.999211, 0.999849, 0.996007, 0.999825, 0.999991,
    0.999981, 0.999819, 0.041745, 0.001116, 1.74e-05, 0.000155,
    0.999976, 0.999958, 0.999993, 0.999878, 0.999940, 0.999791,
    0.996553, 0.999486, 0.998485, 0.999894, 0.999765, 0.997657,
    0.999619, 0.002853, 1.09e-05, 0.999884, 0.999996, 0.999997,
    0.999919, 0.999987, 0.989762, 0.999807, 0.999978, 0.050734,
    0.010660, 0.000217, 0.006174, 0.999977, 0.999954, 0.999995,
    0.999934, 0.999867, 0.999824, 0.996783, 0.999941, 0.999948,
    0.999981, 0.999658, 0.999994, 0.999753, 0.999859, 0.999330,
    0.999993, 0.999956, 0.999970, 0.999996, 0.999991, 0.998674,
    0.999869, 0.999432, 0.999570, 0.999600, 0.999954, 0.999499,
    0.999906, 0.999978, 0.999712, 0.997441, 0.999948, 0.998379,
    0.999578, 0.994745, 0.045936, 0.006816, 0.027384, 0.000278,
    1.000000, 0.996382, 0.999541, 0.998130, 0.999992, 0.999990,
    0.999860, 0.999986, 0.999997, 0.998520, 0.997777, 0.999821,
    0.033353, 0.011629, 6.95e-05, 4.52e-05, 2.04e-06, 0.999963,
    0.999977, 0.999949, 0.999986, 0.999240, 0.999373, 0.999858,
    0.999946, 0.999972, 0.999991, 0.994039, 0.999817, 0.999999,
    0.999715, 0.999924, 0.997763, 0.997944, 0.999825, 0.996592,
    0.695147, 0.000161, 0.999665, 0.999928, 0.999988, 0.992742,
    0.374214, 0.001569, 2.16e-05, 0.000941, 4.32e-05, 0.000556,
    0.999955, 0.999993, 0.999942, 0.999973, 0.999999, 0.999919,
    0.999438, 0.998738]

hamilton_ar1_switch_tvtp_smoothed = [
    0.999997, 0.999246, 0.999918, 0.996118, 0.999740, 0.999990,
    0.999984, 0.999783, 0.035454, 0.000958, 1.53e-05, 0.000139,
    0.999973, 0.999939, 0.999994, 0.999870, 0.999948, 0.999884,
    0.997243, 0.999668, 0.998424, 0.999909, 0.999860, 0.998037,
    0.999559, 0.002533, 1.16e-05, 0.999801, 0.999993, 0.999997,
    0.999891, 0.999994, 0.990096, 0.999753, 0.999974, 0.048495,
    0.009289, 0.000542, 0.005991, 0.999974, 0.999929, 0.999995,
    0.999939, 0.999880, 0.999901, 0.996221, 0.999937, 0.999935,
    0.999985, 0.999450, 0.999995, 0.999768, 0.999897, 0.998930,
    0.999992, 0.999949, 0.999954, 0.999995, 0.999994, 0.998687,
    0.999902, 0.999547, 0.999653, 0.999538, 0.999966, 0.999485,
    0.999883, 0.999982, 0.999831, 0.996940, 0.999968, 0.998678,
    0.999780, 0.993895, 0.055372, 0.020421, 0.022913, 0.000127,
    1.000000, 0.997072, 0.999715, 0.996893, 0.999990, 0.999991,
    0.999811, 0.999978, 0.999998, 0.999100, 0.997866, 0.999787,
    0.034912, 0.009932, 5.91e-05, 3.99e-05, 1.77e-06, 0.999954,
    0.999976, 0.999932, 0.999991, 0.999429, 0.999393, 0.999845,
    0.999936, 0.999961, 0.999995, 0.994246, 0.999570, 1.000000,
    0.999702, 0.999955, 0.998611, 0.998019, 0.999902, 0.998486,
    0.673991, 0.000205, 0.999627, 0.999902, 0.999994, 0.993707,
    0.338707, 0.001359, 2.36e-05, 0.000792, 4.47e-05, 0.000565,
    0.999932, 0.999993, 0.999931, 0.999950, 0.999999, 0.999940,
    0.999626, 0.998738]

expected_durations = [
    [710.7573, 1.000391], [710.7573, 1.000391], [710.7573, 1.000391],
    [710.7573, 1.000391], [710.7573, 1.000391], [710.7573, 1.000391],
    [710.7573, 1.000391], [710.7573, 1.000391], [1.223309, 1864.084],
    [1.223309, 1864.084], [1.223309, 1864.084], [1.223309, 1864.084],
    [710.7573, 1.000391], [710.7573, 1.000391], [710.7573, 1.000391],
    [710.7573, 1.000391], [710.7573, 1.000391], [710.7573, 1.000391],
    [710.7573, 1.000391], [710.7573, 1.000391], [710.7573, 1.000391],
    [710.7573, 1.000391], [710.7573, 1.000391], [710.7573, 1.000391],
    [710.7573, 1.000391], [1.223309, 1864.084], [1.223309, 1864.084],
    [710.7573, 1.000391], [710.7573, 1.000391], [710.7573, 1.000391],
    [710.7573, 1.000391], [710.7573, 1.000391], [710.7573, 1.000391],
    [710.7573, 1.000391], [710.7573, 1.000391], [1.223309, 1864.084],
    [1.223309, 1864.084], [1.223309, 1864.084], [1.223309, 1864.084],
    [710.7573, 1.000391], [710.7573, 1.000391], [710.7573, 1.000391],
    [710.7573, 1.000391], [710.7573, 1.000391], [710.7573, 1.000391],
    [710.7573, 1.000391], [710.7573, 1.000391], [710.7573, 1.000391],
    [710.7573, 1.000391], [710.7573, 1.000391], [710.7573, 1.000391],
    [710.7573, 1.000391], [710.7573, 1.000391], [710.7573, 1.000391],
    [710.7573, 1.000391], [710.7573, 1.000391], [710.7573, 1.000391],
    [710.7573, 1.000391], [710.7573, 1.000391], [710.7573, 1.000391],
    [710.7573, 1.000391], [710.7573, 1.000391], [710.7573, 1.000391],
    [710.7573, 1.000391], [710.7573, 1.000391], [710.7573, 1.000391],
    [710.7573, 1.000391], [710.7573, 1.000391], [710.7573, 1.000391],
    [710.7573, 1.000391], [710.7573, 1.000391], [710.7573, 1.000391],
    [710.7573, 1.000391], [710.7573, 1.000391], [1.223309, 1864.084],
    [1.223309, 1864.084], [1.223309, 1864.084], [1.223309, 1864.084],
    [710.7573, 1.000391], [710.7573, 1.000391], [710.7573, 1.000391],
    [710.7573, 1.000391], [710.7573, 1.000391], [710.7573, 1.000391],
    [710.7573, 1.000391], [710.7573, 1.000391], [710.7573, 1.000391],
    [710.7573, 1.000391], [710.7573, 1.000391], [710.7573, 1.000391],
    [1.223309, 1864.084], [1.223309, 1864.084], [1.223309, 1864.084],
    [1.223309, 1864.084], [1.223309, 1864.084], [710.7573, 1.000391],
    [710.7573, 1.000391], [710.7573, 1.000391], [710.7573, 1.000391],
    [710.7573, 1.000391], [710.7573, 1.000391], [710.7573, 1.000391],
    [710.7573, 1.000391], [710.7573, 1.000391], [710.7573, 1.000391],
    [710.7573, 1.000391], [710.7573, 1.000391], [710.7573, 1.000391],
    [710.7573, 1.000391], [710.7573, 1.000391], [710.7573, 1.000391],
    [710.7573, 1.000391], [710.7573, 1.000391], [710.7573, 1.000391],
    [1.223309, 1864.084], [1.223309, 1864.084], [710.7573, 1.000391],
    [710.7573, 1.000391], [710.7573, 1.000391], [710.7573, 1.000391],
    [1.223309, 1864.084], [1.223309, 1864.084], [1.223309, 1864.084],
    [1.223309, 1864.084], [1.223309, 1864.084], [1.223309, 1864.084],
    [710.7573, 1.000391], [710.7573, 1.000391], [710.7573, 1.000391],
    [710.7573, 1.000391], [710.7573, 1.000391], [710.7573, 1.000391],
    [710.7573, 1.000391], [710.7573, 1.000391]]


class TestHamiltonAR1SwitchTVTP(MarkovAutoregression):
    @classmethod
    def setup_class(cls):
        # Results from E-views:
        # Dependent variable followed by a list of switching regressors:
        #     rgnp c ar(1)
        # List of non-switching regressors: <blank>
        # Do not check "Regime specific error variances"
        # Switching type: Markov
        # Number of Regimes: 2
        # Probability regressors:
        #     c recession
        # Method SWITCHREG
        # Sample 1951q1 1984q4
        true = {
            'params': np.r_[6.564923, 7.846371, -8.064123, -15.37636,
                            1.027190, -0.719760,
                            np.exp(-0.217003)**2, 0.161489, 0.022536],
            'llf': -163.914049,
            'llf_fit': -161.786477,
            'llf_fit_em': -163.914049
        }
        exog_tvtp = np.c_[np.ones(len(rgnp)), rec]
        super(TestHamiltonAR1SwitchTVTP, cls).setup_class(
            true, rgnp, k_regimes=2, order=1, exog_tvtp=exog_tvtp)

    @pytest.mark.skip  # TODO(ChadFulton): give reason for skip
    def test_fit_em(self):
        pass

    def test_filtered_regimes(self):
        assert_allclose(self.result.filtered_marginal_probabilities[:, 0],
                        hamilton_ar1_switch_tvtp_filtered, atol=1e-5)

    def test_smoothed_regimes(self):
        assert_allclose(self.result.smoothed_marginal_probabilities[:, 0],
                        hamilton_ar1_switch_tvtp_smoothed, atol=1e-5)

    def test_expected_durations(self):
        assert_allclose(self.result.expected_durations, expected_durations,
                        rtol=1e-5, atol=1e-7)


class TestFilardo(MarkovAutoregression):
    @classmethod
    def setup_class(cls):
        path = os.path.join(current_path, 'results', 'mar_filardo.csv')
        cls.mar_filardo = pd.read_csv(path)
        true = {
            'params': np.r_[4.35941747, -1.6493936, 1.7702123, 0.9945672,
                            0.517298, -0.865888,
                            np.exp(-0.362469)**2,
                            0.189474, 0.079344, 0.110944, 0.122251],
            'llf': -586.5718,
            'llf_fit': -586.5718,
            'llf_fit_em': -586.5718
        }
        endog = cls.mar_filardo['dlip'].iloc[1:].values
        exog_tvtp = add_constant(
            cls.mar_filardo['dmdlleading'].iloc[:-1].values)
        super(TestFilardo, cls).setup_class(
            true, endog, k_regimes=2, order=4, switching_ar=False,
            exog_tvtp=exog_tvtp)

    @pytest.mark.skip  # TODO(ChadFulton): give reason for skip
    def test_fit(self, **kwargs):
        pass

    @pytest.mark.skip  # TODO(ChadFulton): give reason for skip
    def test_fit_em(self):
        pass

    def test_filtered_regimes(self):
        assert_allclose(self.result.filtered_marginal_probabilities[:, 0],
                        self.mar_filardo['filtered_0'].iloc[5:], atol=1e-5)

    def test_smoothed_regimes(self):
        assert_allclose(self.result.smoothed_marginal_probabilities[:, 0],
                        self.mar_filardo['smoothed_0'].iloc[5:], atol=1e-5)

    def test_expected_durations(self):
        assert_allclose(self.result.expected_durations,
                        self.mar_filardo[['duration0', 'duration1']].iloc[5:],
                        rtol=1e-5, atol=1e-7)


class TestFilardoPandas(MarkovAutoregression):
    @classmethod
    def setup_class(cls):
        path = os.path.join(current_path, 'results', 'mar_filardo.csv')
        cls.mar_filardo = pd.read_csv(path)
        cls.mar_filardo.index = pd.date_range('1948-02-01', '1991-04-01',
                                              freq='MS')
        true = {
            'params': np.r_[4.35941747, -1.6493936, 1.7702123, 0.9945672,
                            0.517298, -0.865888,
                            np.exp(-0.362469)**2,
                            0.189474, 0.079344, 0.110944, 0.122251],
            'llf': -586.5718,
            'llf_fit': -586.5718,
            'llf_fit_em': -586.5718
        }
        endog = cls.mar_filardo['dlip'].iloc[1:]
        exog_tvtp = add_constant(
            cls.mar_filardo['dmdlleading'].iloc[:-1])
        super(TestFilardoPandas, cls).setup_class(
            true, endog, k_regimes=2, order=4, switching_ar=False,
            exog_tvtp=exog_tvtp)

    @pytest.mark.skip  # TODO(ChadFulton): give reason for skip
    def test_fit(self, **kwargs):
        pass

    @pytest.mark.skip  # TODO(ChadFulton): give reason for skip
    def test_fit_em(self):
        pass

    def test_filtered_regimes(self):
        assert_allclose(self.result.filtered_marginal_probabilities[0],
                        self.mar_filardo['filtered_0'].iloc[5:], atol=1e-5)

    def test_smoothed_regimes(self):
        assert_allclose(self.result.smoothed_marginal_probabilities[0],
                        self.mar_filardo['smoothed_0'].iloc[5:], atol=1e-5)

    def test_expected_durations(self):
        assert_allclose(self.result.expected_durations,
                        self.mar_filardo[['duration0', 'duration1']].iloc[5:],
                        rtol=1e-5, atol=1e-7)
