"""
Tests for contingency table analyses.
"""

import os
import warnings

import numpy as np
import statsmodels.stats.contingency_tables as ctab
import pandas as pd
from numpy.testing import assert_allclose, assert_equal
import statsmodels.api as sm

cur_dir = os.path.dirname(os.path.abspath(__file__))
fname = "contingency_table_r_results.csv"
fpath = os.path.join(cur_dir, 'results', fname)
r_results = pd.read_csv(fpath)


tables = [None, None, None]

tables[0] = np.asarray([[23, 15], [19, 31]])

tables[1] = np.asarray([[144, 33, 84, 126],
                        [2, 4, 14, 29],
                        [0, 2, 6, 25],
                        [0, 0, 1, 5]])

tables[2] = np.asarray([[20, 10, 5],
                        [3, 30, 15],
                        [0, 5, 40]])


def test_homogeneity():

    for k,table in enumerate(tables):
        st = sm.stats.SquareTable(table, shift_zeros=False)
        hm = st.homogeneity()
        assert_allclose(hm.statistic, r_results.loc[k, "homog_stat"])
        assert_allclose(hm.df, r_results.loc[k, "homog_df"])

        # Test Bhapkar via its relationship to Stuart_Maxwell.
        hmb = st.homogeneity(method="bhapkar")
        assert_allclose(hmb.statistic, hm.statistic / (1 - hm.statistic / table.sum()))


def test_SquareTable_from_data():

    np.random.seed(434)
    df = pd.DataFrame(index=range(100), columns=["v1", "v2"])
    df["v1"] = np.random.randint(0, 5, 100)
    df["v2"] = np.random.randint(0, 5, 100)
    table = pd.crosstab(df["v1"], df["v2"])

    rslt1 = ctab.SquareTable(table)
    rslt2 = ctab.SquareTable.from_data(df)
    rslt3 = ctab.SquareTable(np.asarray(table))

    assert_equal(rslt1.summary().as_text(),
                 rslt2.summary().as_text())

    assert_equal(rslt2.summary().as_text(),
                 rslt3.summary().as_text())

    s = str(rslt1)
    assert_equal(s.startswith('A 5x5 contingency table with counts:'), True)
    assert_equal(rslt1.table[0, 0], 8.)


def test_SquareTable_nonsquare():

    tab = [[1, 0, 3], [2, 1, 4], [3, 0, 5]]
    df = pd.DataFrame(tab, index=[0, 1, 3], columns=[0, 2, 3])

    df2 = ctab.SquareTable(df, shift_zeros=False)

    e = np.asarray([[1, 0, 0, 3], [2, 0, 1, 4], [0, 0, 0, 0], [3, 0, 0, 5]],
                   dtype=np.float64)

    assert_equal(e, df2.table)


def test_cumulative_odds():

    table = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    table = np.asarray(table)
    tbl_obj = ctab.Table(table)

    cum_odds = tbl_obj.cumulative_oddsratios
    assert_allclose(cum_odds[0, 0], 28 / float(5 * 11))
    assert_allclose(cum_odds[0, 1], (3 * 15) / float(3 * 24), atol=1e-5,
                    rtol=1e-5)
    assert_allclose(np.log(cum_odds), tbl_obj.cumulative_log_oddsratios,
                    atol=1e-5, rtol=1e-5)


def test_local_odds():

    table = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    table = np.asarray(table)
    tbl_obj = ctab.Table(table)

    loc_odds = tbl_obj.local_oddsratios
    assert_allclose(loc_odds[0, 0], 5 / 8.)
    assert_allclose(loc_odds[0, 1], 12 / float(15), atol=1e-5,
                    rtol=1e-5)
    assert_allclose(np.log(loc_odds), tbl_obj.local_log_oddsratios,
                    atol=1e-5, rtol=1e-5)


def test_shifting():

    t = np.zeros((3, 4), dtype=np.float64)
    result = np.full((3, 4), 0.5)
    assert_equal(ctab.Table(t, shift_zeros=False).table, t)
    assert_equal(ctab.Table(t, shift_zeros=True).table, result)

    t = np.asarray([[0, 1, 2],
                    [3, 0, 4],
                    [5, 6, 0]], dtype=np.float64)
    r = np.asarray([[0.5, 1, 2],
                    [3, 0.5, 4],
                    [5, 6, 0.5]], dtype=np.float64)
    assert_equal(ctab.Table(t).table, r)
    assert_equal(ctab.Table(t, shift_zeros=True).table, r)


def test_stratified_table_cube():
    # Test that we can pass a rank 3 ndarray or a list of rank 2
    # ndarrays to StratifiedTable and get the same results.

    tab1 = [[[8, 9], [6, 7]], [[4, 9], [5, 5]], [[8, 8], [9, 11]]]
    tab2 = np.asarray(tab1).T

    ct1 = ctab.StratifiedTable(tab1)
    ct2 = ctab.StratifiedTable(tab2)

    assert_allclose(ct1.oddsratio_pooled, ct2.oddsratio_pooled)
    assert_allclose(ct1.logodds_pooled, ct2.logodds_pooled)


def test_resids():

    # CHD x serum data
    table = [[12, 8, 31, 41], [307, 246, 439, 245]]

    # These results come from SAS
    fit = [[22.083, 17.583, 32.536, 19.798],
           [296.92, 236.42, 437.46, 266.2]]
    c2 = [[4.6037, 5.223, 0.0725, 22.704],
          [0.3424, 0.3885, 0.0054, 1.6886]]

    # These are regression tests
    pr = np.array([[-2.14562121, -2.28538719, -0.26923882,  4.7649169 ],
                   [ 0.58514314,  0.62325942,  0.07342547, -1.29946443]])
    sr = np.array([[-2.55112945, -2.6338782 , -0.34712127,  5.5751083 ],
                   [ 2.55112945,  2.6338782 ,  0.34712127, -5.5751083 ]])

    tab = ctab.Table(table)
    assert_allclose(tab.fittedvalues, fit, atol=1e-4, rtol=1e-4)
    assert_allclose(tab.chi2_contribs, c2, atol=1e-4, rtol=1e-4)
    assert_allclose(tab.resid_pearson, pr, atol=1e-4, rtol=1e-4)
    assert_allclose(tab.standardized_resids, sr, atol=1e-4, rtol=1e-4)


def test_ordinal_association():

    for k,table in enumerate(tables):

        row_scores = 1 + np.arange(table.shape[0])
        col_scores = 1 + np.arange(table.shape[1])

        # First set of scores
        rslt = ctab.Table(table, shift_zeros=False).test_ordinal_association(row_scores, col_scores)
        assert_allclose(rslt.statistic, r_results.loc[k, "lbl_stat"])
        assert_allclose(rslt.null_mean, r_results.loc[k, "lbl_expval"])
        assert_allclose(rslt.null_sd**2, r_results.loc[k, "lbl_var"])
        assert_allclose(rslt.zscore**2, r_results.loc[k, "lbl_chi2"], rtol=1e-5, atol=1e-5)
        assert_allclose(rslt.pvalue, r_results.loc[k, "lbl_pvalue"], rtol=1e-5, atol=1e-5)

        # Second set of scores
        rslt = ctab.Table(table, shift_zeros=False).test_ordinal_association(row_scores, col_scores**2)
        assert_allclose(rslt.statistic, r_results.loc[k, "lbl2_stat"])
        assert_allclose(rslt.null_mean, r_results.loc[k, "lbl2_expval"])
        assert_allclose(rslt.null_sd**2, r_results.loc[k, "lbl2_var"])
        assert_allclose(rslt.zscore**2, r_results.loc[k, "lbl2_chi2"])
        assert_allclose(rslt.pvalue, r_results.loc[k, "lbl2_pvalue"], rtol=1e-5, atol=1e-5)


def test_chi2_association():

    np.random.seed(8743)

    table = np.random.randint(10, 30, size=(4, 4))

    from scipy.stats import chi2_contingency
    rslt_scipy = chi2_contingency(table)

    b = ctab.Table(table).test_nominal_association()

    assert_allclose(b.statistic, rslt_scipy[0])
    assert_allclose(b.pvalue, rslt_scipy[1])


def test_symmetry():

    for k,table in enumerate(tables):
        st = sm.stats.SquareTable(table, shift_zeros=False)
        b = st.symmetry()
        assert_allclose(b.statistic, r_results.loc[k, "bowker_stat"])
        assert_equal(b.df, r_results.loc[k, "bowker_df"])
        assert_allclose(b.pvalue, r_results.loc[k, "bowker_pvalue"])


def test_mcnemar():

    # Use chi^2 without continuity correction
    b1 = ctab.mcnemar(tables[0], exact=False, correction=False)

    st = sm.stats.SquareTable(tables[0])
    b2 = st.homogeneity()
    assert_allclose(b1.statistic, b2.statistic)
    assert_equal(b2.df, 1)

    # Use chi^2 with continuity correction
    b3 = ctab.mcnemar(tables[0], exact=False, correction=True)
    assert_allclose(b3.pvalue, r_results.loc[0, "homog_cont_p"])

    # Use binomial reference distribution
    b4 = ctab.mcnemar(tables[0], exact=True)
    assert_allclose(b4.pvalue, r_results.loc[0, "homog_binom_p"])

def test_from_data_stratified():

    df = pd.DataFrame([[1, 1, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0, 0],
                       [0, 0, 0, 0, 1, 1, 1, 1]]).T
    e = np.asarray([[[0, 1], [1, 1]], [[2, 2], [1, 0]]])

    # Test pandas
    tab1 = ctab.StratifiedTable.from_data(0, 1, 2, df)
    assert_equal(tab1.table, e)

    # Test ndarray
    tab1 = ctab.StratifiedTable.from_data(0, 1, 2, np.asarray(df))
    assert_equal(tab1.table, e)

def test_from_data_2x2():

    df = pd.DataFrame([[1, 1, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0, 0]]).T
    e = np.asarray([[1, 2], [4, 1]])

    # Test pandas
    tab1 = ctab.Table2x2.from_data(df, shift_zeros=False)
    assert_equal(tab1.table, e)

    # Test ndarray
    tab1 = ctab.Table2x2.from_data(np.asarray(df), shift_zeros=False)
    assert_equal(tab1.table, e)


def test_cochranq():
    # library(CVST)
    # table1 = matrix(c(1, 0, 1, 1,
    #                   0, 1, 1, 1,
    #                   1, 1, 1, 0,
    #                   0, 1, 0, 0,
    #                   0, 1, 0, 0,
    #                   1, 0, 1, 0,
    #                   0, 1, 0, 0,
    #                   1, 1, 1, 1,
    #                   0, 1, 0, 0), ncol=4, byrow=TRUE)
    # rslt1 = cochranq.test(table1)
    # table2 = matrix(c(0, 0, 1, 1, 0,
    #                   0, 1, 0, 1, 0,
    #                   0, 1, 1, 0, 1,
    #                   1, 0, 0, 0, 1,
    #                   1, 1, 0, 0, 0,
    #                   1, 0, 1, 0, 0,
    #                   0, 1, 0, 0, 0,
    #                   0, 0, 1, 1, 0,
    #                   0, 0, 0, 0, 0), ncol=5, byrow=TRUE)
    # rslt2 = cochranq.test(table2)

    table = [[1, 0, 1, 1],
             [0, 1, 1, 1],
             [1, 1, 1, 0],
             [0, 1, 0, 0],
             [0, 1, 0, 0],
             [1, 0, 1, 0],
             [0, 1, 0, 0],
             [1, 1, 1, 1],
             [0, 1, 0, 0]]
    table = np.asarray(table)

    stat, pvalue, df = ctab.cochrans_q(table, return_object=False)
    assert_allclose(stat, 4.2)
    assert_allclose(df, 3)

    table = [[0, 0, 1, 1, 0],
             [0, 1, 0, 1, 0],
             [0, 1, 1, 0, 1],
             [1, 0, 0, 0, 1],
             [1, 1, 0, 0, 0],
             [1, 0, 1, 0, 0],
             [0, 1, 0, 0, 0],
             [0, 0, 1, 1, 0],
             [0, 0, 0, 0, 0]]
    table = np.asarray(table)

    stat, pvalue, df = ctab.cochrans_q(table, return_object=False)
    assert_allclose(stat, 1.2174, rtol=1e-4)
    assert_allclose(df, 4)

    # Cochran's q and Mcnemar are equivalent for 2x2 tables
    data = table[:, 0:2]
    xtab = np.asarray(pd.crosstab(data[:, 0], data[:, 1]))
    b1 = ctab.cochrans_q(data, return_object=True)
    b2 = ctab.mcnemar(xtab, exact=False, correction=False)
    assert_allclose(b1.statistic, b2.statistic)
    assert_allclose(b1.pvalue, b2.pvalue)

    # Test for printing bunch
    assert_equal(str(b1).startswith("df          1\npvalue      0.65"), True)


class CheckStratifiedMixin:

    @classmethod
    def initialize(cls, tables, use_arr=False):
        tables1 = tables if not use_arr else np.dstack(tables)
        cls.rslt = ctab.StratifiedTable(tables1)
        cls.rslt_0 = ctab.StratifiedTable(tables, shift_zeros=True)
        tables_pandas = [pd.DataFrame(x) for x in tables]
        cls.rslt_pandas = ctab.StratifiedTable(tables_pandas)


    def test_oddsratio_pooled(self):
        assert_allclose(self.rslt.oddsratio_pooled, self.oddsratio_pooled,
                        rtol=1e-4, atol=1e-4)


    def test_logodds_pooled(self):
        assert_allclose(self.rslt.logodds_pooled, self.logodds_pooled,
                        rtol=1e-4, atol=1e-4)


    def test_null_odds(self):
        rslt = self.rslt.test_null_odds(correction=True)
        assert_allclose(rslt.statistic, self.mh_stat, rtol=1e-4, atol=1e-5)
        assert_allclose(rslt.pvalue, self.mh_pvalue, rtol=1e-4, atol=1e-4)


    def test_oddsratio_pooled_confint(self):
        lcb, ucb = self.rslt.oddsratio_pooled_confint()
        assert_allclose(lcb, self.or_lcb, rtol=1e-4, atol=1e-4)
        assert_allclose(ucb, self.or_ucb, rtol=1e-4, atol=1e-4)


    def test_logodds_pooled_confint(self):
        lcb, ucb = self.rslt.logodds_pooled_confint()
        assert_allclose(lcb, np.log(self.or_lcb), rtol=1e-4,
                        atol=1e-4)
        assert_allclose(ucb, np.log(self.or_ucb), rtol=1e-4,
                        atol=1e-4)


    def test_equal_odds(self):

        if not hasattr(self, "or_homog"):
            return

        rslt = self.rslt.test_equal_odds(adjust=False)
        assert_allclose(rslt.statistic, self.or_homog, rtol=1e-4, atol=1e-4)
        assert_allclose(rslt.pvalue, self.or_homog_p, rtol=1e-4, atol=1e-4)

        rslt = self.rslt.test_equal_odds(adjust=True)
        assert_allclose(rslt.statistic, self.or_homog_adj, rtol=1e-4, atol=1e-4)
        assert_allclose(rslt.pvalue, self.or_homog_adj_p, rtol=1e-4, atol=1e-4)


    def test_pandas(self):

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            assert_equal(self.rslt.summary().as_text(),
                         self.rslt_pandas.summary().as_text())


    def test_from_data(self):

        np.random.seed(241)
        df = pd.DataFrame(index=range(100), columns=("v1", "v2", "strat"))
        df["v1"] = np.random.randint(0, 2, 100)
        df["v2"] = np.random.randint(0, 2, 100)
        df["strat"] = np.kron(np.arange(10), np.ones(10))

        tables = []
        for k in range(10):
            ii = np.arange(10*k, 10*(k+1))
            tables.append(pd.crosstab(df.loc[ii, "v1"], df.loc[ii, "v2"]))

        rslt1 = ctab.StratifiedTable(tables)
        rslt2 = ctab.StratifiedTable.from_data("v1", "v2", "strat", df)

        assert_equal(rslt1.summary().as_text(), rslt2.summary().as_text())


class TestStratified1(CheckStratifiedMixin):
    """
    data = array(c(0, 0, 6, 5,
                   3, 0, 3, 6,
                   6, 2, 0, 4,
                   5, 6, 1, 0,
                   2, 5, 0, 0),
                   dim=c(2, 2, 5))
    rslt = mantelhaen.test(data)
    """

    @classmethod
    def setup_class(cls):
        tables = [None] * 5
        tables[0] = np.array([[0, 0], [6, 5]])
        tables[1] = np.array([[3, 0], [3, 6]])
        tables[2] = np.array([[6, 2], [0, 4]])
        tables[3] = np.array([[5, 6], [1, 0]])
        tables[4] = np.array([[2, 5], [0, 0]])

        cls.initialize(tables)

        cls.oddsratio_pooled = 7
        cls.logodds_pooled = np.log(7)
        cls.mh_stat = 3.9286
        cls.mh_pvalue = 0.04747
        cls.or_lcb = 1.026713
        cls.or_ucb = 47.725133


class TestStratified2(CheckStratifiedMixin):
    """
    library(DescTools)
    data = array(c(20, 14, 10, 24,
                   15, 12, 3, 15,
                   3, 2, 3, 2,
                   12, 3, 7, 5,
                   1, 0, 3, 2),
                   dim=c(2, 2, 5))
    rslt = mantelhaen.test(data)
    bd1 = BreslowDayTest(data, correct=FALSE)
    bd2 = BreslowDayTest(data, correct=TRUE)
    """

    @classmethod
    def setup_class(cls):
        tables = [None] * 5
        tables[0] = np.array([[20, 14], [10, 24]])
        tables[1] = np.array([[15, 12], [3, 15]])
        tables[2] = np.array([[3, 2], [3, 2]])
        tables[3] = np.array([[12, 3], [7, 5]])
        tables[4] = np.array([[1, 0], [3, 2]])

        # check array of int
        cls.initialize(tables, use_arr=True)

        cls.oddsratio_pooled = 3.5912
        cls.logodds_pooled = np.log(3.5912)

        cls.mh_stat = 11.8852
        cls.mh_pvalue = 0.0005658

        cls.or_lcb = 1.781135
        cls.or_ucb = 7.240633

        # Breslow Day test without Tarone adjustment
        cls.or_homog = 1.8438
        cls.or_homog_p = 0.7645

        # Breslow Day test with Tarone adjustment
        cls.or_homog_adj = 1.8436
        cls.or_homog_adj_p = 0.7645


class TestStratified3(CheckStratifiedMixin):
    """
    library(DescTools)
    data = array(c(313, 512, 19, 89,
                   207, 353, 8, 17,
                   205, 120, 391, 202,
                   278, 139, 244, 131,
                   138, 53, 299, 94,
                   351, 22, 317, 24),
                   dim=c(2, 2, 6))
    rslt = mantelhaen.test(data)
    bd1 = BreslowDayTest(data, correct=FALSE)
    bd2 = BreslowDayTest(data, correct=TRUE)
    """

    @classmethod
    def setup_class(cls):
        tables = [None] * 6
        tables[0] = np.array([[313, 512], [19, 89]])
        tables[1] = np.array([[207, 353], [8, 17]])
        tables[2] = np.array([[205, 120], [391, 202]])
        tables[3] = np.array([[278, 139], [244, 131]])
        tables[4] = np.array([[138, 53], [299, 94]])
        tables[5] = np.array([[351, 22], [317, 24]])

        cls.initialize(tables)

        cls.oddsratio_pooled = 1.101879
        cls.logodds_pooled = np.log(1.101879)

        cls.mh_stat = 1.3368
        cls.mh_pvalue = 0.2476

        cls.or_lcb = 0.9402012
        cls.or_ucb = 1.2913602

        # Breslow Day test without Tarone adjustment
        cls.or_homog = 18.83297
        cls.or_homog_p = 0.002064786

        # Breslow Day test with Tarone adjustment
        cls.or_homog_adj = 18.83297
        cls.or_homog_adj_p = 0.002064786

class Check2x2Mixin:
    @classmethod
    def initialize(cls):
        cls.tbl_obj = ctab.Table2x2(cls.table)
        cls.tbl_data_obj = ctab.Table2x2.from_data(cls.data)

    def test_oddsratio(self):
        assert_allclose(self.tbl_obj.oddsratio, self.oddsratio)


    def test_log_oddsratio(self):
        assert_allclose(self.tbl_obj.log_oddsratio, self.log_oddsratio)


    def test_log_oddsratio_se(self):
        assert_allclose(self.tbl_obj.log_oddsratio_se, self.log_oddsratio_se)


    def test_oddsratio_pvalue(self):
        assert_allclose(self.tbl_obj.oddsratio_pvalue(), self.oddsratio_pvalue)


    def test_oddsratio_confint(self):
        lcb1, ucb1 = self.tbl_obj.oddsratio_confint(0.05)
        lcb2, ucb2 = self.oddsratio_confint
        assert_allclose(lcb1, lcb2)
        assert_allclose(ucb1, ucb2)


    def test_riskratio(self):
        assert_allclose(self.tbl_obj.riskratio, self.riskratio)


    def test_log_riskratio(self):
        assert_allclose(self.tbl_obj.log_riskratio, self.log_riskratio)


    def test_log_riskratio_se(self):
        assert_allclose(self.tbl_obj.log_riskratio_se, self.log_riskratio_se)


    def test_riskratio_pvalue(self):
        assert_allclose(self.tbl_obj.riskratio_pvalue(), self.riskratio_pvalue)


    def test_riskratio_confint(self):
        lcb1, ucb1 = self.tbl_obj.riskratio_confint(0.05)
        lcb2, ucb2 = self.riskratio_confint
        assert_allclose(lcb1, lcb2)
        assert_allclose(ucb1, ucb2)


    def test_log_riskratio_confint(self):
        lcb1, ucb1 = self.tbl_obj.log_riskratio_confint(0.05)
        lcb2, ucb2 = self.log_riskratio_confint
        assert_allclose(lcb1, lcb2)
        assert_allclose(ucb1, ucb2)


    def test_from_data(self):
        assert_equal(self.tbl_obj.summary().as_text(),
                     self.tbl_data_obj.summary().as_text())

    def test_summary(self):

        assert_equal(self.tbl_obj.summary().as_text(),
                     self.summary_string)


class Test2x2_1(Check2x2Mixin):

    @classmethod
    def setup_class(cls):
        data = np.zeros((8, 2))
        data[:, 0] = [0, 0, 1, 1, 0, 0, 1, 1]
        data[:, 1] = [0, 1, 0, 1, 0, 1, 0, 1]
        cls.data = np.asarray(data)
        cls.table = np.asarray([[2, 2], [2, 2]])

        cls.oddsratio = 1.
        cls.log_oddsratio = 0.
        cls.log_oddsratio_se = np.sqrt(2)
        cls.oddsratio_confint = [0.062548836166112329, 15.987507702689751]
        cls.oddsratio_pvalue = 1.
        cls.riskratio = 1.
        cls.log_riskratio = 0.
        cls.log_riskratio_se = 1 / np.sqrt(2)
        cls.riskratio_pvalue = 1.
        cls.riskratio_confint = [0.25009765325990629,
                                  3.9984381579173824]
        cls.log_riskratio_confint = [-1.3859038243496782,
                                      1.3859038243496782]
        ss = [  '               Estimate   SE   LCB    UCB   p-value',
                '---------------------------------------------------',
                'Odds ratio        1.000        0.063 15.988   1.000',
                'Log odds ratio    0.000 1.414 -2.772  2.772   1.000',
                'Risk ratio        1.000        0.250  3.998   1.000',
                'Log risk ratio    0.000 0.707 -1.386  1.386   1.000',
                '---------------------------------------------------']
        cls.summary_string = '\n'.join(ss)
        cls.initialize()
