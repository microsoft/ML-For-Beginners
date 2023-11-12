from statsmodels.compat.pandas import assert_frame_equal

import pandas as pd
import numpy as np
from statsmodels.stats.anova import AnovaRM
from numpy.testing import (assert_array_almost_equal, assert_raises,
                           assert_equal)

DV = [7, 3, 6, 6, 5, 8, 6, 7,
      7, 11, 9, 11, 10, 10, 11, 11,
      8, 14, 10, 11, 12, 10, 11, 12,
      16, 7, 11, 9, 10, 11, 8, 8,
      16, 10, 13, 10, 10, 14, 11, 12,
      24, 29, 10, 22, 25, 28, 22, 24,
      1, 3, 5, 8, 3, 5, 6, 8,
      9, 18, 19, 1, 12, 15, 2, 3,
      3, 4, 13, 21, 2, 11, 18, 2,
      12, 7, 12, 3, 19, 1, 4, 13,
      13, 14, 3, 4, 8, 19, 21, 2,
      4, 9, 12, 2, 5, 8, 2, 4]

id = [1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8]

id = ['%d' % i for i in id]

A = ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a',
     'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a',
     'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a',
     'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b',
     'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b',
     'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b',
     'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a',
     'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a',
     'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a',
     'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b',
     'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b',
     'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b']

B = ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a',
     'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b',
     'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c',
     'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a',
     'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b',
     'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c',
     'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a',
     'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b',
     'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c',
     'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a',
     'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b',
     'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c']

D = ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a',
     'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a',
     'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a',
     'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a',
     'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a',
     'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a',
     'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b',
     'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b',
     'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b',
     'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b',
     'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b',
     'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b']

data = pd.DataFrame([id, A, B, D, DV], index=['id', 'A', 'B', 'D', 'DV']).T
data['DV'] = data['DV'].astype('int')


def test_single_factor_repeated_measures_anova():
    """
    Testing single factor repeated measures anova
    Results reproduces R `ezANOVA` function from library ez
    """
    df = AnovaRM(data.iloc[:16, :], 'DV', 'id', within=['B']).fit()
    a = [[1, 7, 22.4, 0.002125452]]
    assert_array_almost_equal(df.anova_table.iloc[:, [1, 2, 0, 3]].values,
                              a, decimal=5)


def test_two_factors_repeated_measures_anova():
    """
    Testing two factors repeated measures anova
    Results reproduces R `ezANOVA` function from library ez
    """
    df = AnovaRM(data.iloc[:48, :], 'DV', 'id', within=['A', 'B']).fit()
    a = [[1, 7, 40.14159, 3.905263e-04],
         [2, 14, 29.21739, 1.007549e-05],
         [2, 14, 17.10545, 1.741322e-04]]
    assert_array_almost_equal(df.anova_table.iloc[:, [1, 2, 0, 3]].values,
                              a, decimal=5)


def test_three_factors_repeated_measures_anova():
    """
    Testing three factors repeated measures anova
    Results reproduces R `ezANOVA` function from library ez
    """
    df = AnovaRM(data, 'DV', 'id', within=['A', 'B', 'D']).fit()
    a = [[1, 7, 8.7650709, 0.021087505],
         [2, 14, 8.4985785, 0.003833921],
         [1, 7, 20.5076546, 0.002704428],
         [2, 14, 0.8457797, 0.450021759],
         [1, 7, 21.7593382, 0.002301792],
         [2, 14, 6.2416695, 0.011536846],
         [2, 14, 5.4253359, 0.018010647]]
    assert_array_almost_equal(df.anova_table.iloc[:, [1, 2, 0, 3]].values,
                              a, decimal=5)


def test_repeated_measures_invalid_factor_name():
    """
    Test with a factor name of 'C', which conflicts with patsy.
    """
    assert_raises(ValueError, AnovaRM, data.iloc[:16, :], 'DV', 'id',
                  within=['C'])


def test_repeated_measures_collinearity():
    data1 = data.iloc[:48, :].copy()
    data1['E'] = data1['A']
    assert_raises(ValueError, AnovaRM, data1, 'DV', 'id', within=['A', 'E'])


def test_repeated_measures_unbalanced_data():
    assert_raises(ValueError, AnovaRM, data.iloc[1:48, :], 'DV', 'id',
                  within=['A', 'B'])


def test_repeated_measures_aggregation():
    df1 = AnovaRM(data, 'DV', 'id', within=['A', 'B', 'D']).fit()
    double_data = pd.concat([data, data], axis=0)
    df2 = AnovaRM(double_data, 'DV', 'id', within=['A', 'B', 'D'],
                  aggregate_func=np.mean).fit()

    assert_frame_equal(df1.anova_table, df2.anova_table)


def test_repeated_measures_aggregation_one_subject_duplicated():
    df1 = AnovaRM(data, 'DV', 'id', within=['A', 'B', 'D']).fit()
    data2 = pd.concat([data, data.loc[data['id'] == '1', :]], axis=0)
    data2 = data2.reset_index()
    df2 = AnovaRM(data2,
                  'DV', 'id', within=['A', 'B', 'D'],
                  aggregate_func=np.mean).fit()

    assert_frame_equal(df1.anova_table, df2.anova_table)


def test_repeated_measures_aggregate_func():
    double_data = pd.concat([data, data], axis=0)
    assert_raises(ValueError, AnovaRM, double_data, 'DV', 'id',
                  within=['A', 'B', 'D'])

    m1 = AnovaRM(double_data, 'DV', 'id', within=['A', 'B', 'D'],
                 aggregate_func=np.mean)
    m2 = AnovaRM(double_data, 'DV', 'id', within=['A', 'B', 'D'],
                 aggregate_func=np.median)

    assert_raises(AssertionError, assert_equal,
                  m1.aggregate_func, m2.aggregate_func)
    assert_frame_equal(m1.fit().anova_table, m2.fit().anova_table)


def test_repeated_measures_aggregate_func_mean():
    double_data = pd.concat([data, data], axis=0)
    m1 = AnovaRM(double_data, 'DV', 'id', within=['A', 'B', 'D'],
                 aggregate_func=np.mean)

    m2 = AnovaRM(double_data, 'DV', 'id', within=['A', 'B', 'D'],
                 aggregate_func='mean')

    assert_equal(m1.aggregate_func, m2.aggregate_func)


def test_repeated_measures_aggregate_compare_with_ezANOVA():
    # Results should reproduces those from R's `ezANOVA` (library ez).
    ez = pd.DataFrame(
        {'F Value': [8.7650709, 8.4985785, 20.5076546, 0.8457797, 21.7593382,
                     6.2416695, 5.4253359],
         'Num DF': [1, 2, 1, 2, 1, 2, 2],
         'Den DF': [7, 14, 7, 14, 7, 14, 14],
         'Pr > F': [0.021087505, 0.003833921, 0.002704428, 0.450021759,
                    0.002301792, 0.011536846, 0.018010647]},
        index=pd.Index(['A', 'B', 'D', 'A:B', 'A:D', 'B:D', 'A:B:D']))
    ez = ez[['F Value', 'Num DF', 'Den DF', 'Pr > F']]

    double_data = pd.concat([data, data], axis=0)
    df = (AnovaRM(double_data, 'DV', 'id', within=['A', 'B', 'D'],
                  aggregate_func=np.mean)
          .fit()
          .anova_table)

    assert_frame_equal(ez, df, check_dtype=False)
