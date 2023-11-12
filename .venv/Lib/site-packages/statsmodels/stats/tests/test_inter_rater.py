# -*- coding: utf-8 -*-
"""

Created on Mon Dec 10 09:18:14 2012

Author: Josef Perktold
"""

import numpy as np
from numpy.testing import assert_almost_equal, assert_equal, assert_allclose

from statsmodels.stats.inter_rater import (fleiss_kappa, cohens_kappa,
                                           to_table, aggregate_raters)
from statsmodels.tools.testing import Holder


table0 = np.asarray('''\
1 	0 	0 	0 	0 	14 	1.000
2 	0 	2 	6 	4 	2 	0.253
3 	0 	0 	3 	5 	6 	0.308
4 	0 	3 	9 	2 	0 	0.440
5 	2 	2 	8 	1 	1 	0.330
6 	7 	7 	0 	0 	0 	0.462
7 	3 	2 	6 	3 	0 	0.242
8 	2 	5 	3 	2 	2 	0.176
9 	6 	5 	2 	1 	0 	0.286
10 	0 	2 	2 	3 	7 	0.286'''.split(), float).reshape(10,-1)

table1 = table0[:, 1:-1]

table10 = [[0, 4, 1],
           [0, 8, 0],
           [0, 1, 5]]

#Fleiss 1971, Fleiss has only the transformed table
diagnoses = np.array( [[4, 4, 4, 4, 4, 4],
                       [2, 2, 2, 5, 5, 5],
                       [2, 3, 3, 3, 3, 5],
                       [5, 5, 5, 5, 5, 5],
                       [2, 2, 2, 4, 4, 4],
                       [1, 1, 3, 3, 3, 3],
                       [3, 3, 3, 3, 5, 5],
                       [1, 1, 3, 3, 3, 4],
                       [1, 1, 4, 4, 4, 4],
                       [5, 5, 5, 5, 5, 5],
                       [1, 4, 4, 4, 4, 4],
                       [1, 2, 4, 4, 4, 4],
                       [2, 2, 2, 3, 3, 3],
                       [1, 4, 4, 4, 4, 4],
                       [2, 2, 4, 4, 4, 5],
                       [3, 3, 3, 3, 3, 5],
                       [1, 1, 1, 4, 5, 5],
                       [1, 1, 1, 1, 1, 2],
                       [2, 2, 4, 4, 4, 4],
                       [1, 3, 3, 5, 5, 5],
                       [5, 5, 5, 5, 5, 5],
                       [2, 4, 4, 4, 4, 4],
                       [2, 2, 4, 5, 5, 5],
                       [1, 1, 4, 4, 4, 4],
                       [1, 4, 4, 4, 4, 5],
                       [2, 2, 2, 2, 2, 4],
                       [1, 1, 1, 1, 5, 5],
                       [2, 2, 4, 4, 4, 4],
                       [1, 3, 3, 3, 3, 3],
                       [5, 5, 5, 5, 5, 5]])
diagnoses_rownames = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', ]
diagnoses_colnames = ['rater1', 'rater2', 'rater3', 'rater4', 'rater5', 'rater6', ]



def test_fleiss_kappa():
    #currently only example from Wikipedia page
    kappa_wp = 0.210
    assert_almost_equal(fleiss_kappa(table1), kappa_wp, decimal=3)


def test_fleis_randolph():
    # reference numbers from online calculator
    # http://justusrandolph.net/kappa/#dInfo
    table = [[7, 0], [7, 0]]
    assert_equal(fleiss_kappa(table, method='unif'), 1)

    table = [[6.99, 0.01], [6.99, 0.01]]
    # % Overall Agreement 0.996671
    # Fixed Marginal Kappa: -0.166667
    # Free Marginal Kappa: 0.993343
    assert_allclose(fleiss_kappa(table), -0.166667, atol=6e-6)
    assert_allclose(fleiss_kappa(table, method='unif'), 0.993343, atol=6e-6)

    table = [[7, 1], [3, 5]]
    # % Overall Agreement 0.607143
    # Fixed Marginal Kappa: 0.161905
    # Free Marginal Kappa: 0.214286
    assert_allclose(fleiss_kappa(table, method='fleiss'), 0.161905, atol=6e-6)
    assert_allclose(fleiss_kappa(table, method='randolph'), 0.214286, atol=6e-6)

    table = [[7, 0], [0, 7]]
    # % Overall Agreement 1.000000
    # Fixed Marginal Kappa: 1.000000
    # Free Marginal Kappa: 1.000000
    assert_allclose(fleiss_kappa(table), 1)
    assert_allclose(fleiss_kappa(table, method='uniform'), 1)

    table = [[6, 1, 0], [0, 7, 0]]
    # % Overall Agreement 0.857143
    # Fixed Marginal Kappa: 0.708333
    # Free Marginal Kappa: 0.785714
    assert_allclose(fleiss_kappa(table), 0.708333, atol=6e-6)
    assert_allclose(fleiss_kappa(table, method='rand'), 0.785714, atol=6e-6)


class CheckCohens:

    def test_results(self):
        res = self.res
        res2 = self.res2

        res_ = [res.kappa, res.std_kappa, res.kappa_low, res.kappa_upp, res.std_kappa0,
                res.z_value, res.pvalue_one_sided, res.pvalue_two_sided]

        assert_almost_equal(res_, res2, decimal=4)
        assert_equal(str(res), self.res_string)


class TestUnweightedCohens(CheckCohens):
    # comparison to printout of a SAS example
    @classmethod
    def setup_class(cls):
        #temporary: res instance is at last position
        cls.res = cohens_kappa(table10)
        res10_sas = [0.4842, 0.1380, 0.2137, 0.7547]
        res10_sash0 = [0.1484, 3.2626, 0.0006, 0.0011]  #for test H0:kappa=0
        cls.res2 = res10_sas + res10_sash0 #concatenate

        cls.res_string = '''\
                  Simple Kappa Coefficient
              --------------------------------
              Kappa                     0.4842
              ASE                       0.1380
              95% Lower Conf Limit      0.2137
              95% Upper Conf Limit      0.7547

                 Test of H0: Simple Kappa = 0

              ASE under H0              0.1484
              Z                         3.2626
              One-sided Pr >  Z         0.0006
              Two-sided Pr > |Z|        0.0011''' + '\n'

    def test_option(self):
        kappa = cohens_kappa(table10, return_results=False)
        assert_almost_equal(kappa, self.res2[0], decimal=4)


class TestWeightedCohens(CheckCohens):
    #comparison to printout of a SAS example
    @classmethod
    def setup_class(cls):
        #temporary: res instance is at last position
        cls.res = cohens_kappa(table10, weights=[0, 1, 2])
        res10w_sas = [0.4701, 0.1457, 0.1845, 0.7558]
        res10w_sash0 = [0.1426, 3.2971, 0.0005, 0.0010]  #for test H0:kappa=0
        cls.res2 = res10w_sas + res10w_sash0 #concatenate

        cls.res_string = '''\
                  Weighted Kappa Coefficient
              --------------------------------
              Kappa                     0.4701
              ASE                       0.1457
              95% Lower Conf Limit      0.1845
              95% Upper Conf Limit      0.7558

                 Test of H0: Weighted Kappa = 0

              ASE under H0              0.1426
              Z                         3.2971
              One-sided Pr >  Z         0.0005
              Two-sided Pr > |Z|        0.0010''' + '\n'

    def test_option(self):
        kappa = cohens_kappa(table10, weights=[0, 1, 2], return_results=False)
        assert_almost_equal(kappa, self.res2[0], decimal=4)


def test_cohenskappa_weights():
    #some tests for equivalent results with different options
    np.random.seed(9743678)
    table = np.random.randint(0, 10, size=(5,5)) + 5*np.eye(5)

    #example aggregation, 2 groups of levels
    mat = np.array([[1,1,1, 0,0],[0,0,0,1,1]])
    table_agg = np.dot(np.dot(mat, table), mat.T)
    res1 = cohens_kappa(table, weights=np.arange(5) > 2, wt='linear')
    res2 = cohens_kappa(table_agg, weights=np.arange(2), wt='linear')
    assert_almost_equal(res1.kappa, res2.kappa, decimal=14)
    assert_almost_equal(res1.var_kappa, res2.var_kappa, decimal=14)

    #equivalence toeplitz with linear for special cases
    res1 = cohens_kappa(table, weights=2*np.arange(5), wt='linear')
    res2 = cohens_kappa(table, weights=2*np.arange(5), wt='toeplitz')
    res3 = cohens_kappa(table, weights=res1.weights[0], wt='toeplitz')
    #2-Dim weights
    res4 = cohens_kappa(table, weights=res1.weights)

    assert_almost_equal(res1.kappa, res2.kappa, decimal=14)
    assert_almost_equal(res1.var_kappa, res2.var_kappa, decimal=14)

    assert_almost_equal(res1.kappa, res3.kappa, decimal=14)
    assert_almost_equal(res1.var_kappa, res3.var_kappa, decimal=14)

    assert_almost_equal(res1.kappa, res4.kappa, decimal=14)
    assert_almost_equal(res1.var_kappa, res4.var_kappa, decimal=14)

    #equivalence toeplitz with quadratic for special cases
    res1 = cohens_kappa(table, weights=5*np.arange(5)**2, wt='toeplitz')
    res2 = cohens_kappa(table, weights=5*np.arange(5), wt='quadratic')
    assert_almost_equal(res1.kappa, res2.kappa, decimal=14)
    assert_almost_equal(res1.var_kappa, res2.var_kappa, decimal=14)

anxiety = np.array([
     3, 3, 3, 4, 5, 5, 2, 3, 5, 2, 2, 6, 1, 5, 2, 2, 1, 2, 4, 3, 3, 6, 4,
     6, 2, 4, 2, 4, 3, 3, 2, 3, 3, 3, 2, 2, 1, 3, 3, 4, 2, 1, 4, 4, 3, 2,
     1, 6, 1, 1, 1, 2, 3, 3, 1, 1, 3, 3, 2, 2
    ]).reshape(20,3, order='F')
anxiety_rownames = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', ]
anxiety_colnames = ['rater1', 'rater2', 'rater3', ]


def test_cohens_kappa_irr():

    ck_w3 = Holder()
    ck_w4 = Holder()

    #>r = kappa2(anxiety[,1:2], c(0,0,0,1,1,1))
    #> cat_items(r, pref="ck_w3.")
    ck_w3.method = "Cohen's Kappa for 2 Raters (Weights: 0,0,0,1,1,1)"
    ck_w3.irr_name = 'Kappa'
    ck_w3.value = 0.1891892
    ck_w3.stat_name = 'z'
    ck_w3.statistic = 0.5079002
    ck_w3.p_value = 0.6115233

    #> r = kappa2(anxiety[,1:2], c(0,0,1,1,2,2))
    #> cat_items(r, pref="ck_w4.")
    ck_w4.method = "Cohen's Kappa for 2 Raters (Weights: 0,0,1,1,2,2)"
    ck_w4.irr_name = 'Kappa'
    ck_w4.value = 0.2820513
    ck_w4.stat_name = 'z'
    ck_w4.statistic = 1.257410
    ck_w4.p_value = 0.2086053

    ck_w1 = Holder()
    ck_w2 = Holder()
    ck_w3 = Holder()
    ck_w4 = Holder()
    #> r = kappa2(anxiety[,2:3])
    #> cat_items(r, pref="ck_w1.")
    ck_w1.method = "Cohen's Kappa for 2 Raters (Weights: unweighted)"
    ck_w1.irr_name = 'Kappa'
    ck_w1.value = -0.006289308
    ck_w1.stat_name = 'z'
    ck_w1.statistic = -0.0604067
    ck_w1.p_value = 0.9518317

    #> r = kappa2(anxiety[,2:3], "equal")
    #> cat_items(r, pref="ck_w2.")
    ck_w2.method = "Cohen's Kappa for 2 Raters (Weights: equal)"
    ck_w2.irr_name = 'Kappa'
    ck_w2.value = 0.1459075
    ck_w2.stat_name = 'z'
    ck_w2.statistic = 1.282472
    ck_w2.p_value = 0.1996772

    #> r = kappa2(anxiety[,2:3], "squared")
    #> cat_items(r, pref="ck_w3.")
    ck_w3.method = "Cohen's Kappa for 2 Raters (Weights: squared)"
    ck_w3.irr_name = 'Kappa'
    ck_w3.value = 0.2520325
    ck_w3.stat_name = 'z'
    ck_w3.statistic = 1.437451
    ck_w3.p_value = 0.1505898

    #> r = kappa2(anxiety[,2:3], c(0,0,1,1,2))
    #> cat_items(r, pref="ck_w4.")
    ck_w4.method = "Cohen's Kappa for 2 Raters (Weights: 0,0,1,1,2)"
    ck_w4.irr_name = 'Kappa'
    ck_w4.value = 0.2391304
    ck_w4.stat_name = 'z'
    ck_w4.statistic = 1.223734
    ck_w4.p_value = 0.2210526

    all_cases = [(ck_w1, None, None),
                 (ck_w2, None, 'linear'),
                 (ck_w2, np.arange(5), None),
                 (ck_w2, np.arange(5), 'toeplitz'),
                 (ck_w3, None, 'quadratic'),
                 (ck_w3, np.arange(5)**2, 'toeplitz'),
                 (ck_w3, 4*np.arange(5)**2, 'toeplitz'),
                 (ck_w4, [0,0,1,1,2], 'toeplitz')]

    #Note R:irr drops the missing category level 4 and uses the reduced matrix
    r = np.histogramdd(anxiety[:,1:], ([1, 2, 3, 4, 6, 7], [1, 2, 3, 4, 6, 7]))

    for res2, w, wt in all_cases:
        msg = repr(w) + repr(wt)
        res1 = cohens_kappa(r[0], weights=w, wt=wt)
        assert_almost_equal(res1.kappa, res2.value, decimal=6, err_msg=msg)
        assert_almost_equal(res1.z_value, res2.statistic, decimal=5, err_msg=msg)
        assert_almost_equal(res1.pvalue_two_sided, res2.p_value, decimal=6, err_msg=msg)


def test_fleiss_kappa_irr():
    fleiss = Holder()
    #> r = kappam.fleiss(diagnoses)
    #> cat_items(r, pref="fleiss.")
    fleiss.method = "Fleiss' Kappa for m Raters"
    fleiss.irr_name = 'Kappa'
    fleiss.value = 0.4302445
    fleiss.stat_name = 'z'
    fleiss.statistic = 17.65183
    fleiss.p_value = 0
    data_, _ = aggregate_raters(diagnoses)
    res1_kappa = fleiss_kappa(data_)
    assert_almost_equal(res1_kappa, fleiss.value, decimal=7)

def test_to_table():
    data = diagnoses
    res1 = to_table(data[:,:2]-1, 5)
    res0 = np.asarray([[(data[:,:2]-1 == [i,j]).all(1).sum()
                            for j in range(5)]
                                for i in range(5)] )
    assert_equal(res1[0], res0)

    res2 = to_table(data[:,:2])
    assert_equal(res2[0], res0)

    bins = [0.5,  1.5,  2.5,  3.5,  4.5, 5.5]
    res3 = to_table(data[:,:2], bins)
    assert_equal(res3[0], res0)

    #more than 2 columns
    res4 = to_table(data[:,:3]-1, bins=[-0.5,  0.5,  1.5,  2.5,  3.5,  4.5])
    res5 = to_table(data[:,:3]-1, bins=5)
    assert_equal(res4[0].sum(-1), res0)
    assert_equal(res5[0].sum(-1), res0)


def test_aggregate_raters():
    data = diagnoses
    data_, categories = aggregate_raters(data)
    colsum = np.array([26, 26, 30, 55, 43])
    assert_equal(data_.sum(0), colsum)
    assert_equal(np.unique(diagnoses), categories)
