# Copyright (c) 2013 Ana Martinez Pardo <anamartinezpardo@gmail.com>
# License: BSD-3 [see LICENSE.txt]

import numpy as np
import numpy.testing as npt
from statsmodels.distributions.mixture_rvs import (mv_mixture_rvs,
                                                   MixtureDistribution)
import statsmodels.sandbox.distributions.mv_normal as mvd
from scipy import stats

class TestMixtureDistributions:

    def test_mixture_rvs_random(self):
        # Test only medium small sample at 1 decimal
        np.random.seed(0)
        mix = MixtureDistribution()
        res = mix.rvs([.75,.25], 1000, dist=[stats.norm, stats.norm], kwargs =
                (dict(loc=-1,scale=.5),dict(loc=1,scale=.5)))
        npt.assert_almost_equal(
                np.array([res.std(),res.mean(),res.var()]),
                np.array([1,-0.5,1]),
                decimal=1)

    def test_mv_mixture_rvs_random(self):
        cov3 = np.array([[ 1.  ,  0.5 ,  0.75],
                       [ 0.5 ,  1.5 ,  0.6 ],
                       [ 0.75,  0.6 ,  2.  ]])
        mu = np.array([-1, 0.0, 2.0])
        mu2 = np.array([4, 2.0, 2.0])
        mvn3 = mvd.MVNormal(mu, cov3)
        mvn32 = mvd.MVNormal(mu2, cov3/2.)
        np.random.seed(0)
        res = mv_mixture_rvs([0.4, 0.6], 5000, [mvn3, mvn32], 3)
        npt.assert_almost_equal(
                np.array([res.std(),res.mean(),res.var()]),
                np.array([1.874,1.733,3.512]),
                decimal=1)

    def test_mixture_pdf(self):
        mix = MixtureDistribution()
        grid = np.linspace(-4,4, 10)
        res = mix.pdf(grid, [1/3.,2/3.], dist=[stats.norm, stats.norm], kwargs=
                (dict(loc=-1,scale=.25),dict(loc=1,scale=.75)))
        npt.assert_almost_equal(
                res,
                np.array([  7.92080017e-11,   1.05977272e-07,   3.82368500e-05,
                            2.21485447e-01,   1.00534607e-01,   2.69531536e-01,
                            3.21265627e-01,   9.39899015e-02,   6.74932493e-03,
                            1.18960201e-04]))

    def test_mixture_cdf(self):
        mix = MixtureDistribution()
        grid = np.linspace(-4,4, 10)
        res = mix.cdf(grid, [1/3.,2/3.], dist=[stats.norm, stats.norm], kwargs=
                   (dict(loc=-1,scale=.25),dict(loc=1,scale=.75)))
        npt.assert_almost_equal(
                res,
                np.array([  8.72261646e-12,   1.40592960e-08,   5.95819161e-06,
                         3.10250226e-02,   3.46993159e-01,   4.86283549e-01,
                         7.81092904e-01,   9.65606734e-01,   9.98373155e-01,
                         9.99978886e-01]))

    def test_mixture_rvs_fixed(self):
        mix = MixtureDistribution()
        np.random.seed(1234)
        res = mix.rvs([.15,.85], 50, dist=[stats.norm, stats.norm], kwargs =
                (dict(loc=1,scale=.5),dict(loc=-1,scale=.5)))
        npt.assert_almost_equal(
                res,
                np.array([-0.5794956 , -1.72290504, -1.70098664, -1.0504591 ,
                            -1.27412122,-1.07230975, -0.82298983, -1.01775651,
                            -0.71713085,-0.2271706 ,-1.48711817, -1.03517244,
                            -0.84601557, -1.10424938, -0.48309963,-2.20022682,
                            0.01530181,  1.1238961 , -1.57131564, -0.89405831,
                            -0.64763969, -1.39271761,  0.55142161, -0.76897013,
                            -0.64788589,-0.73824602, -1.46312716,  0.00392148,
                            -0.88651873, -1.57632955,-0.68401028, -0.98024366,
                            -0.76780384,  0.93160258,-2.78175833,-0.33944719,
                            -0.92368472, -0.91773523, -1.21504785, -0.61631563,
                            1.0091446 , -0.50754008,  1.37770699, -0.86458208,
                            -0.3040069 ,-0.96007884,  1.10763429, -1.19998229,
                            -1.51392528, -1.29235911]))

    def test_mv_mixture_rvs_fixed(self):
        np.random.seed(1234)
        cov3 = np.array([[ 1.  ,  0.5 ,  0.75],
                       [ 0.5 ,  1.5 ,  0.6 ],
                       [ 0.75,  0.6 ,  2.  ]])
        mu = np.array([-1, 0.0, 2.0])
        mu2 = np.array([4, 2.0, 2.0])
        mvn3 = mvd.MVNormal(mu, cov3)
        mvn32 = mvd.MVNormal(mu2, cov3/2)
        res = mv_mixture_rvs([0.2, 0.8], 10, [mvn3, mvn32], 3)
        npt.assert_almost_equal(
                res,
                np.array([[-0.23955497,  1.73426482,  0.36100243],
                       [ 2.52063189,  1.0832677 ,  1.89947131],
                       [ 4.36755379,  2.14480498,  2.22003966],
                       [ 3.1141545 ,  1.21250505,  2.58511199],
                       [ 4.1980202 ,  2.50017561,  1.87324933],
                       [ 3.48717503,  0.91847424,  2.14004598],
                       [ 3.55904133,  2.74367622,  0.68619582],
                       [ 3.60521933,  1.57316531,  0.82784584],
                       [ 3.86102275,  0.6211812 ,  1.33016426],
                       [ 3.91074761,  2.037155  ,  2.22247051]]))
