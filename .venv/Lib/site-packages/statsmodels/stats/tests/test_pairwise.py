# -*- coding: utf-8 -*-
"""

Created on Wed Mar 28 15:34:18 2012

Author: Josef Perktold
"""
from statsmodels.compat.python import asbytes

from io import BytesIO
import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_, assert_allclose, assert_almost_equal, assert_equal, \
    assert_raises

from statsmodels.stats.libqsturng import qsturng
from statsmodels.stats.multicomp import (tukeyhsd, pairwise_tukeyhsd,
                                         MultiComparison)

ss = '''\
  43.9  1   1
  39.0  1   2
  46.7  1   3
  43.8  1   4
  44.2  1   5
  47.7  1   6
  43.6  1   7
  38.9  1   8
  43.6  1   9
  40.0  1  10
  89.8  2   1
  87.1  2   2
  92.7  2   3
  90.6  2   4
  87.7  2   5
  92.4  2   6
  86.1  2   7
  88.1  2   8
  90.8  2   9
  89.1  2  10
  68.4  3   1
  69.3  3   2
  68.5  3   3
  66.4  3   4
  70.0  3   5
  68.1  3   6
  70.6  3   7
  65.2  3   8
  63.8  3   9
  69.2  3  10
  36.2  4   1
  45.2  4   2
  40.7  4   3
  40.5  4   4
  39.3  4   5
  40.3  4   6
  43.2  4   7
  38.7  4   8
  40.9  4   9
  39.7  4  10'''

#idx   Treatment StressReduction
ss2 = '''\
1     mental               2
2     mental               2
3     mental               3
4     mental               4
5     mental               4
6     mental               5
7     mental               3
8     mental               4
9     mental               4
10    mental               4
11  physical               4
12  physical               4
13  physical               3
14  physical               5
15  physical               4
16  physical               1
17  physical               1
18  physical               2
19  physical               3
20  physical               3
21   medical               1
22   medical               2
23   medical               2
24   medical               2
25   medical               3
26   medical               2
27   medical               3
28   medical               1
29   medical               3
30   medical               1'''

ss3 = '''\
1 24.5
1 23.5
1 26.4
1 27.1
1 29.9
2 28.4
2 34.2
2 29.5
2 32.2
2 30.1
3 26.1
3 28.3
3 24.3
3 26.2
3 27.8'''

ss5 = '''\
2 - 3\t4.340\t0.691\t7.989\t***
2 - 1\t4.600\t0.951\t8.249\t***
3 - 2\t-4.340\t-7.989\t-0.691\t***
3 - 1\t0.260\t-3.389\t3.909\t-
1 - 2\t-4.600\t-8.249\t-0.951\t***
1 - 3\t-0.260\t-3.909\t3.389\t-
'''

cylinders = np.array([8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 4, 6, 6, 6, 4, 4,
                    4, 4, 4, 4, 6, 8, 8, 8, 8, 4, 4, 4, 4, 8, 8, 8, 8, 6, 6, 6, 6, 4, 4, 4, 4, 6, 6,
                    6, 6, 4, 4, 4, 4, 4, 8, 4, 6, 6, 8, 8, 8, 8, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                    4, 4, 4, 4, 4, 4, 4, 6, 6, 4, 6, 4, 4, 4, 4, 4, 4, 4, 4])
cyl_labels = np.array(['USA', 'USA', 'USA', 'USA', 'USA', 'USA', 'USA', 'USA', 'USA', 'USA', 'France',
    'USA', 'USA', 'USA', 'USA', 'USA', 'USA', 'USA', 'USA', 'USA', 'Japan', 'USA', 'USA', 'USA', 'Japan',
    'Germany', 'France', 'Germany', 'Sweden', 'Germany', 'USA', 'USA', 'USA', 'USA', 'USA', 'Germany',
    'USA', 'USA', 'France', 'USA', 'USA', 'USA', 'USA', 'USA', 'USA', 'USA', 'USA', 'USA', 'USA', 'Germany',
    'Japan', 'USA', 'USA', 'USA', 'USA', 'Germany', 'Japan', 'Japan', 'USA', 'Sweden', 'USA', 'France',
    'Japan', 'Germany', 'USA', 'USA', 'USA', 'USA', 'USA', 'USA', 'USA', 'USA', 'USA', 'USA', 'USA', 'USA',
    'Germany', 'Japan', 'Japan', 'USA', 'USA', 'Japan', 'Japan', 'Japan', 'Japan', 'Japan', 'Japan', 'USA',
    'USA', 'USA', 'USA', 'Japan', 'USA', 'USA', 'USA', 'Germany', 'USA', 'USA', 'USA'])

#accommodate recfromtxt for python 3.2, requires bytes
ss = asbytes(ss)
ss2 = asbytes(ss2)
ss3 = asbytes(ss3)
ss5 = asbytes(ss5)

dta = pd.read_csv(BytesIO(ss), sep=r'\s+', header=None, engine='python')
dta.columns = "Rust", "Brand", "Replication"
dta2 = pd.read_csv(BytesIO(ss2), sep=r'\s+', header=None, engine='python')
dta2.columns = "idx", "Treatment", "StressReduction"
dta2["Treatment"] = dta2["Treatment"].map(lambda v: v.encode('utf-8'))
dta3 = pd.read_csv(BytesIO(ss3), sep=r'\s+', header=None, engine='python')
dta3.columns = ["Brand", "Relief"]
dta5 = pd.read_csv(BytesIO(ss5), sep=r'\t', header=None, engine='python')
dta5.columns = ['pair', 'mean', 'lower', 'upper', 'sig']
for col in ('pair', 'sig'):
    dta5[col] = dta5[col].map(lambda v: v.encode('utf-8'))
sas_ = dta5.iloc[[1, 3, 2]]


def get_thsd(mci, alpha=0.05):
    var_ = np.var(mci.groupstats.groupdemean(), ddof=len(mci.groupsunique))
    means = mci.groupstats.groupmean
    nobs = mci.groupstats.groupnobs
    resi = tukeyhsd(means, nobs, var_, df=None, alpha=alpha,
                    q_crit=qsturng(1-alpha, len(means), (nobs-1).sum()))
    #print resi[4]
    var2 = (mci.groupstats.groupvarwithin() * (nobs - 1.)).sum() \
                                                        / (nobs - 1.).sum()
    #print nobs, (nobs - 1).sum()
    #print mci.groupstats.groupvarwithin()
    assert_almost_equal(var_, var2, decimal=14)
    return resi

class CheckTuckeyHSDMixin:

    @classmethod
    def setup_class_(cls):
        cls.mc = MultiComparison(cls.endog, cls.groups)
        cls.res = cls.mc.tukeyhsd(alpha=cls.alpha)

    def test_multicomptukey(self):
        assert_almost_equal(self.res.meandiffs, self.meandiff2, decimal=14)
        assert_almost_equal(self.res.confint, self.confint2, decimal=2)
        assert_equal(self.res.reject, self.reject2)

    def test_group_tukey(self):
        res_t = get_thsd(self.mc, alpha=self.alpha)
        assert_almost_equal(res_t[4], self.confint2, decimal=2)

    def test_shortcut_function(self):
        #check wrapper function
        res = pairwise_tukeyhsd(self.endog, self.groups, alpha=self.alpha)
        assert_almost_equal(res.confint, self.res.confint, decimal=14)

    @pytest.mark.smoke
    @pytest.mark.matplotlib
    def test_plot_simultaneous_ci(self, close_figures):
        self.res._simultaneous_ci()
        reference = self.res.groupsunique[1]
        self.res.plot_simultaneous(comparison_name=reference)


class TestTuckeyHSD2(CheckTuckeyHSDMixin):

    @classmethod
    def setup_class(cls):
        #balanced case
        cls.endog = dta2['StressReduction']
        cls.groups = dta2['Treatment']
        cls.alpha = 0.05
        cls.setup_class_() #in super

        #from R
        tukeyhsd2s = np.array([ 1.5,1,-0.5,0.3214915,
                               -0.1785085,-1.678509,2.678509,2.178509,
                                0.6785085,0.01056279,0.1079035,0.5513904]
                              ).reshape(3,4, order='F')
        cls.meandiff2 = tukeyhsd2s[:, 0]
        cls.confint2 = tukeyhsd2s[:, 1:3]
        pvals = tukeyhsd2s[:, 3]
        cls.reject2 = pvals < 0.05

    def test_table_names_default_group_order(self):
        t = self.res._results_table
        # if the group_order parameter is not used, the groups should
        # be reported in alphabetical order
        expected_order = [(b'medical', b'mental'),
                          (b'medical', b'physical'),
                          (b'mental', b'physical')]
        for i in range(1, 4):
            first_group = t[i][0].data
            second_group = t[i][1].data
            assert_((first_group, second_group) == expected_order[i - 1])

    def test_table_names_custom_group_order(self):
        # if the group_order parameter is used, the groups should
        # be reported in the specified order
        mc = MultiComparison(self.endog, self.groups,
                             group_order=[b'physical', b'medical', b'mental'])
        res = mc.tukeyhsd(alpha=self.alpha)
        #print(res)
        t = res._results_table
        expected_order = [(b'physical',b'medical'),
                          (b'physical',b'mental'),
                          (b'medical', b'mental')]
        for i in range(1, 4):
            first_group = t[i][0].data
            second_group = t[i][1].data
            assert_((first_group, second_group) == expected_order[i - 1])


class TestTuckeyHSD2Pandas(TestTuckeyHSD2):

    @classmethod
    def setup_class(cls):
        super(TestTuckeyHSD2Pandas, cls).setup_class()

        cls.endog = pd.Series(cls.endog)
        # we are working with bytes on python 3, not with strings in this case
        cls.groups = pd.Series(cls.groups, dtype=object)

    def test_incorrect_output(self):
        # too few groups
        with pytest.raises(ValueError):
            MultiComparison(np.array([1] * 10), [1, 2] * 4)
        # too many groups
        with pytest.raises(ValueError):
            MultiComparison(np.array([1] * 10), [1, 2] * 6)
        # just one group
        with pytest.raises(ValueError):
            MultiComparison(np.array([1] * 10), [1] * 10)

        # group_order does not select all observations, only one group left
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            assert_raises(ValueError, MultiComparison, np.array([1] * 10),
                         [1, 2] * 5, group_order=[1])

        # group_order does not select all observations,
        # we do tukey_hsd with reduced set of observations
        data = np.arange(15)
        groups = np.repeat([1, 2, 3], 5)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            mod1 = MultiComparison(np.array(data), groups, group_order=[1, 2])
            assert_equal(len(w), 1)
            assert issubclass(w[0].category, UserWarning)

        res1 = mod1.tukeyhsd(alpha=0.01)
        mod2 = MultiComparison(np.array(data[:10]), groups[:10])
        res2 = mod2.tukeyhsd(alpha=0.01)

        attributes = ['confint', 'data', 'df_total', 'groups', 'groupsunique',
                     'meandiffs', 'q_crit', 'reject', 'reject2', 'std_pairs',
                     'variance']
        for att in attributes:
            err_msg = att + 'failed'
            assert_allclose(getattr(res1, att), getattr(res2, att), rtol=1e-14,
                            err_msg=err_msg)

        attributes = ['data', 'datali', 'groupintlab', 'groups', 'groupsunique',
                      'ngroups', 'nobs', 'pairindices']
        for att in attributes:
            err_msg = att + 'failed'
            assert_allclose(getattr(mod1, att), getattr(mod2, att), rtol=1e-14,
                            err_msg=err_msg)


class TestTuckeyHSD2s(CheckTuckeyHSDMixin):
    @classmethod
    def setup_class(cls):
        #unbalanced case
        cls.endog = dta2['StressReduction'][3:29]
        cls.groups = dta2['Treatment'][3:29]
        cls.alpha = 0.01
        cls.setup_class_()

        #from R
        tukeyhsd2s = np.array(
                [1.8888888888888889, 0.888888888888889, -1, 0.2658549,
                 -0.5908785, -2.587133, 3.511923, 2.368656,
                 0.5871331, 0.002837638, 0.150456, 0.1266072]
                ).reshape(3,4, order='F')
        cls.meandiff2 = tukeyhsd2s[:, 0]
        cls.confint2 = tukeyhsd2s[:, 1:3]
        pvals = tukeyhsd2s[:, 3]
        cls.reject2 = pvals < 0.01


class TestTuckeyHSD3(CheckTuckeyHSDMixin):

    @classmethod
    def setup_class(cls):
        #SAS case
        cls.endog = dta3['Relief']
        cls.groups = dta3['Brand']
        cls.alpha = 0.05
        cls.setup_class_()
        #super(cls, cls).setup_class_()
        #CheckTuckeyHSD.setup_class_()
        cls.meandiff2 = sas_['mean']
        cls.confint2 = sas_[['lower','upper']].astype(float).values.reshape((3, 2))
        cls.reject2 = sas_['sig'] == asbytes('***')


class TestTuckeyHSD4(CheckTuckeyHSDMixin):

    @classmethod
    def setup_class(cls):
        #unbalanced case verified in Matlab
        cls.endog = cylinders
        cls.groups = cyl_labels
        cls.alpha = 0.05
        cls.setup_class_()
        cls.res._simultaneous_ci()

        #from Matlab
        cls.halfwidth2 = np.array([1.5228335685980883, 0.9794949704444682, 0.78673802805533644,
                                    2.3321237694566364, 0.57355135882752939])
        cls.meandiff2 = np.array([0.22222222222222232, 0.13333333333333375, 0.0, 2.2898550724637685,
                            -0.088888888888888573, -0.22222222222222232, 2.0676328502415462,
                            -0.13333333333333375, 2.1565217391304348, 2.2898550724637685])
        cls.confint2 = np.array([-2.32022210717, 2.76466655161, -2.247517583, 2.51418424967,
                            -3.66405224956, 3.66405224956, 0.113960166573, 4.46574997835,
                            -1.87278583908, 1.6950080613, -3.529655688, 3.08521124356, 0.568180988881,
                            3.5670847116, -3.31822643175, 3.05155976508, 0.951206924521, 3.36183655374,
                             -0.74487911754, 5.32458926247]).reshape(10,2)
        cls.reject2 = np.array([False, False, False,  True, False, False,  True, False,  True, False])

    def test_hochberg_intervals(self):
        assert_almost_equal(self.res.halfwidths, self.halfwidth2, 4)
