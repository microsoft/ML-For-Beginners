# Copyright (c) 2011 BSD, Roger Lew [see LICENSE.txt]
# This software is funded in part by NIH Grant P20 RR016454.

"""The 'handful' tests are intended to aid refactoring. The tests with the
@pytest.mark..slow are empirical (test within error limits) and intended to more
extensively ensure the stability and accuracy of the functions"""

from statsmodels.compat.python import lzip, lmap

from numpy.testing import (
    assert_equal,
    assert_almost_equal, assert_array_almost_equal,
    assert_raises)

import numpy as np
import pytest

from statsmodels.stats.libqsturng import qsturng, psturng


def read_ch(fname):
    with open(fname, encoding="utf-8") as f:
        lines = f.readlines()
    ps,rs,vs,qs = lzip(*[L.split(',') for L in lines])
    return lmap(float, ps), lmap(float, rs),lmap(float, vs), lmap(float, qs)


class TestQsturng:
    def test_scalar(self):
        # scalar input -> scalar output
        assert_almost_equal(4.43645545899562, qsturng(.9,5,6), 5)

    def test_vector(self):
        # vector input -> vector output
        assert_array_almost_equal(np.array([3.98832389,
                                            4.56835318,
                                            6.26400894]),
                                  qsturng([.8932, .9345,.9827],
                                          [4, 4, 4],
                                          [6, 6, 6]),
                                  5)

    def test_invalid_parameters(self):
        # p < .1
        assert_raises(ValueError, qsturng, -.1,5,6)
        # p > .999
        assert_raises(ValueError, qsturng, .9991,5,6)
        # p < .9, v = 1
        assert_raises(ValueError, qsturng, .89,5,1)
        # p >= .9, v = 0
        assert_raises(ValueError, qsturng, .9,5,0)
        # r < 2
        assert_raises((ValueError, OverflowError), qsturng, .9,1,2)

    def test_handful_to_tbl(self):
        cases = [(0.75, 30.0, 12.0, 5.01973488482),
                 (0.975, 15.0, 18.0, 6.00428263999),
                 (0.1, 8.0, 11.0, 1.76248712658),
                 (0.995, 6.0, 17.0, 6.13684839819),
                 (0.85, 15.0, 18.0, 4.65007986215),
                 (0.75, 17.0, 18.0, 4.33179650607),
                 (0.75, 60.0, 16.0, 5.50520795792),
                 (0.99, 100.0, 2.0, 50.3860723433),
                 (0.9, 2.0, 40.0, 2.38132493732),
                 (0.8, 12.0, 20.0, 4.15361239056),
                 (0.675, 8.0, 14.0, 3.35011529943),
                 (0.75, 30.0, 24.0, 4.77976803574),
                 (0.75, 2.0, 18.0, 1.68109190167),
                 (0.99, 7.0, 120.0, 5.00525918406),
                 (0.8, 19.0, 15.0, 4.70694373713),
                 (0.8, 15.0, 8.0, 4.80392205906),
                 (0.5, 12.0, 11.0, 3.31672775449),
                 (0.85, 30.0, 2.0, 10.2308503607),
                 (0.675, 20.0, 18.0, 4.23706426096),
                 (0.1, 60.0, 60.0, 3.69215469278)]

        for p,r,v,q in cases:
            assert_almost_equal(q, qsturng(p,r,v), 5)

    # TODO: do something with this?
    #remove from testsuite, used only for table generation and fails on
    #Debian S390, no idea why
    @pytest.mark.skip
    def test_all_to_tbl(self):
        from statsmodels.stats.libqsturng.make_tbls import T,R
        ps, rs, vs, qs = [], [], [], []
        for p in T:
            for v in T[p]:
                for r in R.keys():
                    ps.append(p)
                    vs.append(v)
                    rs.append(r)
                    qs.append(T[p][v][R[r]])

        qs = np.array(qs)
        errors = np.abs(qs-qsturng(ps,rs,vs))/qs
        assert_equal(np.array([]), np.where(errors > .03)[0])

    def test_handful_to_ch(self):
        cases = [(0.8699908, 10.0, 465.4956, 3.997799075635331),
                 (0.8559087, 43.0, 211.7474, 5.1348419692951675),
                 (0.6019187, 11.0, 386.5556, 3.3383101487698821),
                 (0.658888, 51.0, 74.652, 4.8108880483153733),
                 (0.6183604, 77.0, 479.8493, 4.9864059321732874),
                 (0.9238978, 77.0, 787.5278, 5.7871053003022936),
                 (0.8408322, 7.0, 227.3483, 3.5555798311413578),
                 (0.5930279, 60.0, 325.3461, 4.7658023123882396),
                 (0.6236158, 61.0, 657.5285, 4.8207812755987867),
                 (0.9344575, 72.0, 846.4138, 5.8014341329259107),
                 (0.8761198, 56.0, 677.8171, 5.362460718311719),
                 (0.7901517, 41.0, 131.525, 4.9222831341950544),
                 (0.6396423, 44.0, 624.3828, 4.6015127250083152),
                 (0.8085966, 14.0, 251.4224, 4.0793058424719746),
                 (0.716179, 45.0, 136.7055, 4.8055498089340087),
                 (0.8204, 6.0, 290.9876, 3.3158771384085597),
                 (0.8705345, 83.0, 759.6216, 5.5969334564485376),
                 (0.8249085, 18.0, 661.9321, 4.3283725986180395),
                 (0.9503, 2.0, 4.434, 3.7871158594867262),
                 (0.7276132, 95.0, 91.43983, 5.4100384868499889)]

        for p,r,v,q in cases:
            assert_almost_equal(q, qsturng(p,r,v), 5)

    @pytest.mark.slow
    def test_10000_to_ch(self):
        import os
        curdir = os.path.dirname(os.path.abspath(__file__))
        #ps, rs, vs, qs = read_ch(curdir + '/bootleg.dat') # <- generated by qtukey in R
        # work around problem getting libqsturng.tests.bootleg.dat installed
        ps, rs, vs, qs = read_ch(os.path.split(os.path.split(curdir)[0])[0]
                                 + '/tests/results/bootleg.csv')
        qs = np.array(qs)
        errors = np.abs(qs-qsturng(ps,rs,vs))/qs
        assert_equal(np.array([]), np.where(errors > .03)[0])

class TestPsturng:
    def test_scalar(self):
        "scalar input -> scalar output"
        assert_almost_equal(.1, psturng(4.43645545899562,5,6), 5)

    def test_vector(self):
        "vector input -> vector output"
        assert_array_almost_equal(np.array([0.10679889,
                                             0.06550009,
                                             0.01730145]),
                                  psturng([3.98832389,
                                           4.56835318,
                                           6.26400894],
                                          [4, 4, 4],
                                          [6, 6, 6]),
                                  5)

    def test_v_equal_one(self):
        assert_almost_equal(.1, psturng(.2,5,1), 5)

    def test_invalid_parameters(self):
        # q < .1
        assert_raises(ValueError, psturng, -.1,5,6)
        # r < 2
        assert_raises((ValueError, OverflowError), psturng, .9,1,2)

    def test_handful_to_known_values(self):
        cases = [(0.71499578726111435, 67, 956.70742488392386, 5.0517658443070692),
                 (0.42974234855067672, 16, 723.50261736502318, 3.3303582093701354),
                 (0.94936429359548424, 2, 916.1867328010926, 2.7677975546417244),
                 (0.85357381770725038, 66, 65.67055060832368, 5.5647438108270109),
                 (0.87372108021900929, 74, 626.42369474993632, 5.5355540570701107),
                 (0.53891960564713726, 49, 862.63799438485785, 4.5108645923377146),
                 (0.98818659555664567, 18, 36.269686711464274, 6.0906643750886156),
                 (0.53031994896037626, 50, 265.29558652727917, 4.5179640079726795),
                 (0.7318857887397332, 59, 701.41497552251201, 4.9980139875409915),
                 (0.65332019368982697, 61, 591.01183664195912, 4.8706581766706893),
                 (0.55403221657248558, 77, 907.34156725405194, 4.8786135917984632),
                 (0.30783916857266003, 83, 82.446923487980882, 4.4396401242858294),
                 (0.29321720242415661, 16, 709.64382575553009, 3.0304277540702729),
                 (0.27146478168880306, 31, 590.00594683574172, 3.5870031664477215),
                 (0.67348796958433776, 81, 608.02706111127657, 5.1096199974432936),
                 (0.32774393945968938, 18, 17.706224399250839, 3.2119038163765432),
                 (0.7081637474795982, 72, 443.10678914889695, 5.0990030889410649),
                 (0.33354939276757861, 47, 544.0772192199048, 4.0613352964193279),
                 (0.60412143947363051, 36, 895.83526933271548, 4.381717596850172),
                 (0.88739052300665977, 77, 426.03665511558262, 5.6333929480341309)]

        for p,r,v,q in cases:
            assert_almost_equal(1.-p, psturng(q,r,v), 5)

    @pytest.mark.slow
    def test_100_random_values(self, reset_randomstate):
        n = 100
        random_state = np.random.RandomState(12345)
        ps = random_state.random_sample(n)*(.999 - .1) + .1
        rs = random_state.randint(2, 101, n)
        vs = random_state.random_sample(n)*998. + 2.
        qs = qsturng(ps, rs, vs)
        estimates = psturng(qs, rs, vs)
        actuals = 1. - ps
        errors = estimates - actuals

        assert_equal(np.array([]), np.where(errors > 1e-5)[0])
