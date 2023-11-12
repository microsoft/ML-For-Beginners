# -*- coding: utf-8 -*-

import pandas as pd
from ..cancorr import CanCorr
from numpy.testing import assert_almost_equal

data_fit = pd.DataFrame([[191, 36, 50,  5, 162,  60],
                         [189, 37, 52,  2, 110,  60],
                         [193, 38, 58, 12, 101, 101],
                         [162, 35, 62, 12, 105,  37],
                         [189, 35, 46, 13, 155,  58],
                         [182, 36, 56,  4, 101,  42],
                         [211, 38, 56,  8, 101,  38],
                         [167, 34, 60,  6, 125,  40],
                         [176, 31, 74, 15, 200,  40],
                         [154, 33, 56, 17, 251, 250],
                         [169, 34, 50, 17, 120,  38],
                         [166, 33, 52, 13, 210, 115],
                         [154, 34, 64, 14, 215, 105],
                         [247, 46, 50,  1,  50,  50],
                         [193, 36, 46,  6,  70,  31],
                         [202, 37, 62, 12, 210, 120],
                         [176, 37, 54,  4,  60,  25],
                         [157, 32, 52, 11, 230,  80],
                         [156, 33, 54, 15, 225,  73],
                         [138, 33, 68,  2, 110,  43]])


def test_cancorr():
    # Compare results to SAS example:
    # https://support.sas.com/documentation/cdl/en/statug/63347/HTML/default/
    # viewer.htm#statug_cancorr_sect020.htm
    X1 = data_fit.iloc[:, :3]
    Y1 = data_fit.iloc[:, 3:]
    mod = CanCorr(Y1, X1)
    r = mod.corr_test()
    assert_almost_equal(r.stats_mv.loc["Wilks' lambda", 'Value'],
                        0.35039053, decimal=8)
    assert_almost_equal(r.stats_mv.loc["Pillai's trace", 'Value'],
                        0.67848151, decimal=8)
    assert_almost_equal(r.stats_mv.loc["Hotelling-Lawley trace", 'Value'],
                        1.77194146, decimal=8)
    assert_almost_equal(r.stats_mv.loc["Roy's greatest root", 'Value'],
                        1.72473874, decimal=8)
    assert_almost_equal(r.stats_mv.loc["Wilks' lambda", 'F Value'],
                        2.05, decimal=2)
    assert_almost_equal(r.stats_mv.loc["Pillai's trace", 'F Value'],
                        1.56, decimal=2)
    assert_almost_equal(r.stats_mv.loc["Hotelling-Lawley trace",
                                            'F Value'],
                        2.64, decimal=2)
    assert_almost_equal(r.stats_mv.loc["Roy's greatest root", 'F Value'],
                        9.20, decimal=2)
    assert_almost_equal(r.stats_mv.loc["Wilks' lambda", 'Num DF'],
                        9, decimal=3)
    assert_almost_equal(r.stats_mv.loc["Pillai's trace", 'Num DF'],
                        9, decimal=3)
    assert_almost_equal(r.stats_mv.loc["Hotelling-Lawley trace",
                                            'Num DF'],
                        9, decimal=3)
    assert_almost_equal(r.stats_mv.loc["Roy's greatest root", 'Num DF'],
                        3, decimal=3)
    assert_almost_equal(r.stats_mv.loc["Wilks' lambda", 'Den DF'],
                        34.223, decimal=3)
    assert_almost_equal(r.stats_mv.loc["Pillai's trace", 'Den DF'],
                        48, decimal=3)
    assert_almost_equal(r.stats_mv.loc["Hotelling-Lawley trace",
                                            'Den DF'],
                        19.053, decimal=3)
    assert_almost_equal(r.stats_mv.loc["Roy's greatest root", 'Den DF'],
                        16, decimal=3)
    assert_almost_equal(r.stats_mv.loc["Wilks' lambda", 'Pr > F'],
                        0.0635, decimal=4)
    assert_almost_equal(r.stats_mv.loc["Pillai's trace", 'Pr > F'],
                        0.1551, decimal=4)
    assert_almost_equal(r.stats_mv.loc["Hotelling-Lawley trace",
                                            'Pr > F'],
                        0.0357, decimal=4)
    assert_almost_equal(r.stats_mv.loc["Roy's greatest root", 'Pr > F'],
                        0.0009, decimal=4)
    assert_almost_equal(r.stats.loc[0, "Wilks' lambda"],
                        0.35039053, decimal=8)
    assert_almost_equal(r.stats.loc[1, "Wilks' lambda"],
                        0.95472266, decimal=8)
    assert_almost_equal(r.stats.loc[2, "Wilks' lambda"],
                        0.99473355, decimal=8)
    assert_almost_equal(r.stats.loc[0, 'F Value'],
                        2.05, decimal=2)
    assert_almost_equal(r.stats.loc[1, 'F Value'],
                        0.18, decimal=2)
    assert_almost_equal(r.stats.loc[2, 'F Value'],
                        0.08, decimal=2)
    assert_almost_equal(r.stats.loc[0, 'Num DF'],
                        9, decimal=2)
    assert_almost_equal(r.stats.loc[1, 'Num DF'],
                        4, decimal=2)
    assert_almost_equal(r.stats.loc[2, 'Num DF'],
                        1, decimal=2)
    assert_almost_equal(r.stats.loc[0, 'Den DF'],
                        34.223, decimal=3)
    assert_almost_equal(r.stats.loc[1, 'Den DF'],
                        30, decimal=2)
    assert_almost_equal(r.stats.loc[2, 'Den DF'],
                        16, decimal=2)
    assert_almost_equal(r.stats.loc[0, 'Pr > F'],
                        0.0635, decimal=4)
    assert_almost_equal(r.stats.loc[1, 'Pr > F'],
                        0.9491, decimal=4)
    assert_almost_equal(r.stats.loc[2, 'Pr > F'],
                        0.7748, decimal=4)
