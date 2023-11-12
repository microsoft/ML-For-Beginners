from statsmodels.compat.pandas import assert_series_equal, assert_frame_equal

from io import StringIO
from textwrap import dedent

import numpy as np
import numpy.testing as npt

import numpy
from numpy.testing import assert_equal
import pandas
import pytest

from statsmodels.imputation import ros


def load_basic_data():
    raw_csv = StringIO(
        "res,qual\n2.00,=\n4.20,=\n4.62,=\n5.00,ND\n5.00,ND\n5.50,ND\n"
        "5.57,=\n5.66,=\n5.75,ND\n5.86,=\n6.65,=\n6.78,=\n6.79,=\n7.50,=\n"
        "7.50,=\n7.50,=\n8.63,=\n8.71,=\n8.99,=\n9.50,ND\n9.50,ND\n9.85,=\n"
        "10.82,=\n11.00,ND\n11.25,=\n11.25,=\n12.20,=\n14.92,=\n16.77,=\n"
        "17.81,=\n19.16,=\n19.19,=\n19.64,=\n20.18,=\n22.97,=\n"
    )
    df = pandas.read_csv(raw_csv)
    df.loc[:, 'conc'] = df['res']
    df.loc[:, 'censored'] = df['qual'] == 'ND'

    return df


def load_intermediate_data():
    df = pandas.DataFrame([
        {'censored': True, 'conc': 5.0, 'det_limit_index': 1, 'rank': 1},
        {'censored': True, 'conc': 5.0, 'det_limit_index': 1, 'rank': 2},
        {'censored': True, 'conc': 5.5, 'det_limit_index': 2, 'rank': 1},
        {'censored': True, 'conc': 5.75, 'det_limit_index': 3, 'rank': 1},
        {'censored': True, 'conc': 9.5, 'det_limit_index': 4, 'rank': 1},
        {'censored': True, 'conc': 9.5, 'det_limit_index': 4, 'rank': 2},
        {'censored': True, 'conc': 11.0, 'det_limit_index': 5, 'rank': 1},
        {'censored': False, 'conc': 2.0, 'det_limit_index': 0, 'rank': 1},
        {'censored': False, 'conc': 4.2, 'det_limit_index': 0, 'rank': 2},
        {'censored': False, 'conc': 4.62, 'det_limit_index': 0, 'rank': 3},
        {'censored': False, 'conc': 5.57, 'det_limit_index': 2, 'rank': 1},
        {'censored': False, 'conc': 5.66, 'det_limit_index': 2, 'rank': 2},
        {'censored': False, 'conc': 5.86, 'det_limit_index': 3, 'rank': 1},
        {'censored': False, 'conc': 6.65, 'det_limit_index': 3, 'rank': 2},
        {'censored': False, 'conc': 6.78, 'det_limit_index': 3, 'rank': 3},
        {'censored': False, 'conc': 6.79, 'det_limit_index': 3, 'rank': 4},
        {'censored': False, 'conc': 7.5, 'det_limit_index': 3, 'rank': 5},
        {'censored': False, 'conc': 7.5, 'det_limit_index': 3, 'rank': 6},
        {'censored': False, 'conc': 7.5, 'det_limit_index': 3, 'rank': 7},
        {'censored': False, 'conc': 8.63, 'det_limit_index': 3, 'rank': 8},
        {'censored': False, 'conc': 8.71, 'det_limit_index': 3, 'rank': 9},
        {'censored': False, 'conc': 8.99, 'det_limit_index': 3, 'rank': 10},
        {'censored': False, 'conc': 9.85, 'det_limit_index': 4, 'rank': 1},
        {'censored': False, 'conc': 10.82, 'det_limit_index': 4, 'rank': 2},
        {'censored': False, 'conc': 11.25, 'det_limit_index': 5, 'rank': 1},
        {'censored': False, 'conc': 11.25, 'det_limit_index': 5, 'rank': 2},
        {'censored': False, 'conc': 12.2, 'det_limit_index': 5, 'rank': 3},
        {'censored': False, 'conc': 14.92, 'det_limit_index': 5, 'rank': 4},
        {'censored': False, 'conc': 16.77, 'det_limit_index': 5, 'rank': 5},
        {'censored': False, 'conc': 17.81, 'det_limit_index': 5, 'rank': 6},
        {'censored': False, 'conc': 19.16, 'det_limit_index': 5, 'rank': 7},
        {'censored': False, 'conc': 19.19, 'det_limit_index': 5, 'rank': 8},
        {'censored': False, 'conc': 19.64, 'det_limit_index': 5, 'rank': 9},
        {'censored': False, 'conc': 20.18, 'det_limit_index': 5, 'rank': 10},
        {'censored': False, 'conc': 22.97, 'det_limit_index': 5, 'rank': 11}
    ])

    return df


def load_advanced_data():
    df = pandas.DataFrame([
        {'Zprelim': -1.4456202174142005, 'censored': True, 'conc': 5.0,
        'det_limit_index': 1, 'plot_pos': 0.07414187643020594, 'rank': 1},
        {'Zprelim': -1.2201035333697587, 'censored': True, 'conc': 5.0,
        'det_limit_index': 1, 'plot_pos': 0.11121281464530891, 'rank': 2},
        {'Zprelim': -1.043822530159519, 'censored': True, 'conc': 5.5,
        'det_limit_index': 2, 'plot_pos': 0.14828375286041187, 'rank': 1},
        {'Zprelim': -1.0438225301595188, 'censored': True, 'conc': 5.75,
        'det_limit_index': 3, 'plot_pos': 0.1482837528604119, 'rank': 1},
        {'Zprelim': -0.8109553641377003, 'censored': True, 'conc': 9.5,
        'det_limit_index': 4, 'plot_pos': 0.20869565217391303, 'rank': 1},
        {'Zprelim': -0.4046779045300476, 'censored': True, 'conc': 9.5,
        'det_limit_index': 4, 'plot_pos': 0.34285714285714286, 'rank': 2},
        {'Zprelim': -0.20857169501420522, 'censored': True, 'conc': 11.0,
        'det_limit_index': 5, 'plot_pos': 0.41739130434782606, 'rank': 1},
        {'Zprelim': -1.5927654676048002, 'censored': False, 'conc': 2.0,
        'det_limit_index': 0, 'plot_pos': 0.055606407322654455, 'rank': 1},
        {'Zprelim': -1.2201035333697587, 'censored': False, 'conc': 4.2,
        'det_limit_index': 0, 'plot_pos': 0.11121281464530891, 'rank': 2},
        {'Zprelim': -0.9668111610681008, 'censored': False, 'conc': 4.62,
        'det_limit_index': 0, 'plot_pos': 0.16681922196796337, 'rank': 3},
        {'Zprelim': -0.6835186393930371, 'censored': False, 'conc': 5.57,
        'det_limit_index': 2, 'plot_pos': 0.24713958810068648, 'rank': 1},
        {'Zprelim': -0.6072167256926887, 'censored': False, 'conc': 5.66,
        'det_limit_index': 2, 'plot_pos': 0.27185354691075514, 'rank': 2},
        {'Zprelim': -0.44953240276543616, 'censored': False, 'conc': 5.86,
        'det_limit_index': 3, 'plot_pos': 0.3265238194299979, 'rank': 1},
        {'Zprelim': -0.36788328223414807, 'censored': False, 'conc': 6.65,
        'det_limit_index': 3, 'plot_pos': 0.35648013313917204, 'rank': 2},
        {'Zprelim': -0.28861907892223937, 'censored': False, 'conc': 6.78,
        'det_limit_index': 3, 'plot_pos': 0.38643644684834616, 'rank': 3},
        {'Zprelim': -0.21113039741112186, 'censored': False, 'conc': 6.79,
        'det_limit_index': 3, 'plot_pos': 0.4163927605575203, 'rank': 4},
        {'Zprelim': -0.1348908823006299, 'censored': False, 'conc': 7.5,
        'det_limit_index': 3, 'plot_pos': 0.4463490742666944, 'rank': 5},
        {'Zprelim': -0.05942854708257491, 'censored': False, 'conc': 7.5,
        'det_limit_index': 3, 'plot_pos': 0.4763053879758685, 'rank': 6},
        {'Zprelim': 0.015696403006170083, 'censored': False, 'conc': 7.5,
        'det_limit_index': 3, 'plot_pos': 0.5062617016850427, 'rank': 7},
        {'Zprelim': 0.09091016994359362, 'censored': False, 'conc': 8.63,
        'det_limit_index': 3, 'plot_pos': 0.5362180153942168, 'rank': 8},
        {'Zprelim': 0.16664251178856201, 'censored': False, 'conc': 8.71,
        'det_limit_index': 3, 'plot_pos': 0.5661743291033909, 'rank': 9},
        {'Zprelim': 0.24334426739770573, 'censored': False, 'conc': 8.99,
        'det_limit_index': 3, 'plot_pos': 0.596130642812565, 'rank': 10},
        {'Zprelim': 0.3744432988606558, 'censored': False, 'conc': 9.85,
        'det_limit_index': 4, 'plot_pos': 0.6459627329192545, 'rank': 1},
        {'Zprelim': 0.4284507519609981, 'censored': False, 'conc': 10.82,
        'det_limit_index': 4, 'plot_pos': 0.6658385093167701, 'rank': 2},
        {'Zprelim': 0.5589578655042562, 'censored': False, 'conc': 11.25,
        'det_limit_index': 5, 'plot_pos': 0.7119047619047619, 'rank': 1},
        {'Zprelim': 0.6374841609623771, 'censored': False, 'conc': 11.25,
        'det_limit_index': 5, 'plot_pos': 0.7380952380952381, 'rank': 2},
        {'Zprelim': 0.7201566171385521, 'censored': False, 'conc': 12.2,
        'det_limit_index': 5, 'plot_pos': 0.7642857142857142, 'rank': 3},
        {'Zprelim': 0.8080746339118065, 'censored': False, 'conc': 14.92,
        'det_limit_index': 5, 'plot_pos': 0.7904761904761904, 'rank': 4},
        {'Zprelim': 0.9027347916438648, 'censored': False, 'conc': 16.77,
        'det_limit_index': 5, 'plot_pos': 0.8166666666666667, 'rank': 5},
        {'Zprelim': 1.0062699858608395, 'censored': False, 'conc': 17.81,
        'det_limit_index': 5, 'plot_pos': 0.8428571428571429, 'rank': 6},
        {'Zprelim': 1.1219004674623523, 'censored': False, 'conc': 19.16,
        'det_limit_index': 5, 'plot_pos': 0.8690476190476191, 'rank': 7},
        {'Zprelim': 1.2548759122271174, 'censored': False, 'conc': 19.19,
        'det_limit_index': 5, 'plot_pos': 0.8952380952380953, 'rank': 8},
        {'Zprelim': 1.414746425534976, 'censored': False, 'conc': 19.64,
        'det_limit_index': 5, 'plot_pos': 0.9214285714285714, 'rank': 9},
        {'Zprelim': 1.622193585315426, 'censored': False, 'conc': 20.18,
        'det_limit_index': 5, 'plot_pos': 0.9476190476190476, 'rank': 10},
        {'Zprelim': 1.9399896117517081, 'censored': False, 'conc': 22.97,
        'det_limit_index': 5, 'plot_pos': 0.9738095238095239, 'rank': 11}
    ])

    return df


def load_basic_cohn():
    cohn = pandas.DataFrame([
        {'lower_dl': 2.0, 'ncen_equal': 0.0, 'nobs_below': 0.0,
         'nuncen_above': 3.0, 'prob_exceedance': 1.0, 'upper_dl': 5.0},
        {'lower_dl': 5.0, 'ncen_equal': 2.0, 'nobs_below': 5.0,
         'nuncen_above': 0.0, 'prob_exceedance': 0.77757437070938218, 'upper_dl': 5.5},
        {'lower_dl': 5.5, 'ncen_equal': 1.0, 'nobs_below': 6.0,
         'nuncen_above': 2.0, 'prob_exceedance': 0.77757437070938218, 'upper_dl': 5.75},
        {'lower_dl': 5.75, 'ncen_equal': 1.0, 'nobs_below': 9.0,
         'nuncen_above': 10.0, 'prob_exceedance': 0.7034324942791762, 'upper_dl': 9.5},
        {'lower_dl': 9.5, 'ncen_equal': 2.0, 'nobs_below': 21.0,
         'nuncen_above': 2.0, 'prob_exceedance': 0.37391304347826088, 'upper_dl': 11.0},
        {'lower_dl': 11.0, 'ncen_equal': 1.0, 'nobs_below': 24.0,
         'nuncen_above': 11.0, 'prob_exceedance': 0.31428571428571428, 'upper_dl': numpy.inf},
        {'lower_dl': numpy.nan, 'ncen_equal': numpy.nan, 'nobs_below': numpy.nan,
         'nuncen_above': numpy.nan, 'prob_exceedance': 0.0, 'upper_dl': numpy.nan}
    ])
    return cohn


class Test__ros_sort:
    def setup_method(self):
        self.df = load_basic_data()

        self.expected_baseline = pandas.DataFrame([
            {'censored': True,  'conc': 5.0},   {'censored': True,  'conc': 5.0},
            {'censored': True,  'conc': 5.5},   {'censored': True,  'conc': 5.75},
            {'censored': True,  'conc': 9.5},   {'censored': True,  'conc': 9.5},
            {'censored': True,  'conc': 11.0},  {'censored': False, 'conc': 2.0},
            {'censored': False, 'conc': 4.2},   {'censored': False, 'conc': 4.62},
            {'censored': False, 'conc': 5.57},  {'censored': False, 'conc': 5.66},
            {'censored': False, 'conc': 5.86},  {'censored': False, 'conc': 6.65},
            {'censored': False, 'conc': 6.78},  {'censored': False, 'conc': 6.79},
            {'censored': False, 'conc': 7.5},   {'censored': False, 'conc': 7.5},
            {'censored': False, 'conc': 7.5},   {'censored': False, 'conc': 8.63},
            {'censored': False, 'conc': 8.71},  {'censored': False, 'conc': 8.99},
            {'censored': False, 'conc': 9.85},  {'censored': False, 'conc': 10.82},
            {'censored': False, 'conc': 11.25}, {'censored': False, 'conc': 11.25},
            {'censored': False, 'conc': 12.2},  {'censored': False, 'conc': 14.92},
            {'censored': False, 'conc': 16.77}, {'censored': False, 'conc': 17.81},
            {'censored': False, 'conc': 19.16}, {'censored': False, 'conc': 19.19},
            {'censored': False, 'conc': 19.64}, {'censored': False, 'conc': 20.18},
            {'censored': False, 'conc': 22.97},
        ])[['conc', 'censored']]

        self.expected_with_warning = self.expected_baseline.iloc[:-1]

    def test_baseline(self):
        result = ros._ros_sort(self.df, 'conc', 'censored')
        assert_frame_equal(result, self.expected_baseline)

    def test_censored_greater_than_max(self):
        df = self.df.copy()
        max_row = df['conc'].idxmax()
        df.loc[max_row, 'censored'] = True
        result = ros._ros_sort(df, 'conc', 'censored')
        assert_frame_equal(result, self.expected_with_warning)


class Test_cohn_numbers:
    def setup_method(self):
        self.df = load_basic_data()
        self.final_cols = ['lower_dl', 'upper_dl', 'nuncen_above', 'nobs_below',
                           'ncen_equal', 'prob_exceedance']

        self.expected_baseline = pandas.DataFrame([
            {'lower_dl': 2.0, 'ncen_equal': 0.0, 'nobs_below': 0.0,
             'nuncen_above': 3.0, 'prob_exceedance': 1.0, 'upper_dl': 5.0},
            {'lower_dl': 5.0, 'ncen_equal': 2.0, 'nobs_below': 5.0,
             'nuncen_above': 0.0, 'prob_exceedance': 0.77757437070938218, 'upper_dl': 5.5},
            {'lower_dl': 5.5, 'ncen_equal': 1.0, 'nobs_below': 6.0,
             'nuncen_above': 2.0, 'prob_exceedance': 0.77757437070938218, 'upper_dl': 5.75},
            {'lower_dl': 5.75, 'ncen_equal': 1.0, 'nobs_below': 9.0,
             'nuncen_above': 10.0, 'prob_exceedance': 0.7034324942791762, 'upper_dl': 9.5},
            {'lower_dl': 9.5, 'ncen_equal': 2.0, 'nobs_below': 21.0,
             'nuncen_above': 2.0, 'prob_exceedance': 0.37391304347826088, 'upper_dl': 11.0},
            {'lower_dl': 11.0, 'ncen_equal': 1.0, 'nobs_below': 24.0,
             'nuncen_above': 11.0, 'prob_exceedance': 0.31428571428571428, 'upper_dl': numpy.inf},
            {'lower_dl': numpy.nan, 'ncen_equal': numpy.nan, 'nobs_below': numpy.nan,
             'nuncen_above': numpy.nan, 'prob_exceedance': 0.0, 'upper_dl': numpy.nan}
        ])[self.final_cols]


    def test_baseline(self):
        result = ros.cohn_numbers(self.df, observations='conc', censorship='censored')
        assert_frame_equal(result, self.expected_baseline)

    def test_no_NDs(self):
        _df = self.df.copy()
        _df['qual'] = False
        result = ros.cohn_numbers(_df, observations='conc', censorship='qual')
        assert result.shape == (0, 6)


class Test__detection_limit_index:
    def setup_method(self):
        self.cohn = load_basic_cohn()
        self.empty_cohn = pandas.DataFrame(numpy.empty((0, 7)))

    def test_empty(self):
        assert_equal(ros._detection_limit_index(None, self.empty_cohn), 0)

    def test_populated(self):
        assert_equal(ros._detection_limit_index(3.5, self.cohn), 0)
        assert_equal(ros._detection_limit_index(6.0, self.cohn), 3)
        assert_equal(ros._detection_limit_index(12.0, self.cohn), 5)

    def test_out_of_bounds(self):
        with pytest.raises(IndexError):
            ros._detection_limit_index(0, self.cohn)


def test__ros_group_rank():
    df = pandas.DataFrame({
        'dl_idx': [1] * 12,
        'params': list('AABCCCDE') + list('DCBA'),
        'values': list(range(12))
    })

    result = ros._ros_group_rank(df, 'dl_idx', 'params')
    expected = pandas.Series([1, 2, 1, 1, 2, 3, 1, 1, 2, 4, 2, 3], name='rank')
    assert_series_equal(result.astype(int), expected.astype(int))


class Test__ros_plot_pos:
    def setup_method(self):
        self.cohn = load_basic_cohn()

    def test_uncensored_1(self):
        row = {'censored': False, 'det_limit_index': 2, 'rank': 1}
        result = ros._ros_plot_pos(row, 'censored', self.cohn)
        assert_equal(result, 0.24713958810068648)

    def test_uncensored_2(self):
        row = {'censored': False, 'det_limit_index': 2, 'rank': 12}
        result = ros._ros_plot_pos(row, 'censored', self.cohn)
        assert_equal(result, 0.51899313501144173)

    def test_censored_1(self):
        row = {'censored': True, 'det_limit_index': 5, 'rank': 4}
        result = ros._ros_plot_pos(row, 'censored', self.cohn)
        assert_equal(result, 1.3714285714285714)

    def test_censored_2(self):
        row = {'censored': True, 'det_limit_index': 4, 'rank': 2}
        result = ros._ros_plot_pos(row, 'censored', self.cohn)
        assert_equal(result, 0.41739130434782606)


def test__norm_plot_pos():
    result = ros._norm_plot_pos([1, 2, 3, 4])
    expected = numpy.array([ 0.159104,  0.385452,  0.614548,  0.840896])
    npt.assert_array_almost_equal(result, expected)


def test_plotting_positions():
    df = load_intermediate_data()
    cohn = load_basic_cohn()

    results = ros.plotting_positions(df, 'censored', cohn)
    expected = numpy.array([
        0.07414188,  0.11121281,  0.14828375,  0.14828375,  0.20869565,
        0.34285714,  0.4173913 ,  0.05560641,  0.11121281,  0.16681922,
        0.24713959,  0.27185355,  0.32652382,  0.35648013,  0.38643645,
        0.41639276,  0.44634907,  0.47630539,  0.5062617 ,  0.53621802,
        0.56617433,  0.59613064,  0.64596273,  0.66583851,  0.71190476,
        0.73809524,  0.76428571,  0.79047619,  0.81666667,  0.84285714,
        0.86904762,  0.8952381 ,  0.92142857,  0.94761905,  0.97380952
    ])
    npt.assert_array_almost_equal(results, expected)


def test__impute():
    expected = numpy.array([
        3.11279729,   3.60634338,   4.04602788,   4.04602788,
        4.71008116,   6.14010906,   6.97841457,   2.        ,
        4.2       ,   4.62      ,   5.57      ,   5.66      ,
        5.86      ,   6.65      ,   6.78      ,   6.79      ,
        7.5       ,   7.5       ,   7.5       ,   8.63      ,
        8.71      ,   8.99      ,   9.85      ,  10.82      ,
        11.25      ,  11.25      ,  12.2       ,  14.92      ,
        16.77      ,  17.81      ,  19.16      ,  19.19      ,
        19.64      ,  20.18      ,  22.97
    ])
    df = load_advanced_data()
    df = ros._impute(df, 'conc', 'censored', numpy.log, numpy.exp)
    result = df['final'].values
    npt.assert_array_almost_equal(result, expected)


def test__do_ros():
    expected = numpy.array([
        3.11279729,   3.60634338,   4.04602788,   4.04602788,
        4.71008116,   6.14010906,   6.97841457,   2.        ,
        4.2       ,   4.62      ,   5.57      ,   5.66      ,
        5.86      ,   6.65      ,   6.78      ,   6.79      ,
        7.5       ,   7.5       ,   7.5       ,   8.63      ,
        8.71      ,   8.99      ,   9.85      ,  10.82      ,
        11.25      ,  11.25      ,  12.2       ,  14.92      ,
        16.77      ,  17.81      ,  19.16      ,  19.19      ,
        19.64      ,  20.18      ,  22.97
    ])

    df = load_basic_data()
    df = ros._do_ros(df, 'conc', 'censored', numpy.log, numpy.exp)
    result = df['final'].values
    npt.assert_array_almost_equal(result, expected)


class CheckROSMixin:
    def test_ros_df(self):
        result = ros.impute_ros(self.rescol, self.cencol, df=self.df)
        npt.assert_array_almost_equal(
            sorted(result),
            sorted(self.expected_final),
            decimal=self.decimal
        )

    def test_ros_arrays(self):
        result = ros.impute_ros(self.df[self.rescol], self.df[self.cencol], df=None)
        npt.assert_array_almost_equal(
            sorted(result),
            sorted(self.expected_final),
            decimal=self.decimal
        )

    def test_cohn(self):
        cols = [
            'nuncen_above', 'nobs_below',
            'ncen_equal', 'prob_exceedance'
        ]
        cohn = ros.cohn_numbers(self.df, self.rescol, self.cencol)
        # Use round in place of the deprecated check_less_precise arg
        assert_frame_equal(
            np.round(cohn[cols], 3),
            np.round(self.expected_cohn[cols], 3),
        )


class Test_ROS_HelselAppendixB(CheckROSMixin):
    """
    Appendix B dataset from "Estimation of Descriptive Statists for
    Multiply Censored Water Quality Data", Water Resources Research,
    Vol 24, No 12, pp 1997 - 2004. December 1988.
    """
    decimal = 2
    res = numpy.array([
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10., 10., 10.,
        3.0, 7.0, 9.0, 12., 15., 20., 27., 33., 50.
    ])
    cen = numpy.array([
        True, True, True, True, True, True, True, True, True,
        False, False, False, False, False, False, False,
        False, False
    ])
    rescol = 'obs'
    cencol = 'cen'
    df = pandas.DataFrame({rescol: res, cencol: cen})
    expected_final = numpy.array([
        0.47,  0.85, 1.11, 1.27, 1.76, 2.34, 2.50, 3.00, 3.03,
        4.80, 7.00, 9.00, 12.0, 15.0, 20.0, 27.0, 33.0, 50.0
    ])

    expected_cohn = pandas.DataFrame({
        'nuncen_above': numpy.array([3.0, 6.0, numpy.nan]),
        'nobs_below': numpy.array([6.0, 12.0, numpy.nan]),
        'ncen_equal': numpy.array([6.0, 3.0, numpy.nan]),
        'prob_exceedance': numpy.array([0.55556, 0.33333, 0.0]),
    })


class Test_ROS_HelselArsenic(CheckROSMixin):
    """
    Oahu arsenic data from Nondetects and Data Analysis by
    Dennis R. Helsel (John Wiley, 2005)

    Plotting positions are fudged since relative to source data since
    modeled data is what matters and (source data plot positions are
    not uniformly spaced, which seems weird)
    """
    decimal = 2
    res = numpy.array([
        3.2, 2.8, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
        2.0, 2.0, 1.7, 1.5, 1.0, 1.0, 1.0, 1.0,
        0.9, 0.9, 0.7, 0.7, 0.6, 0.5, 0.5, 0.5
    ])

    cen = numpy.array([
        False, False, True, True, True, True, True,
        True, True, True, False, False, True, True,
        True, True, False, True, False, False, False,
        False, False, False
    ])
    rescol = 'obs'
    cencol = 'cen'
    df = pandas.DataFrame({rescol: res, cencol: cen})
    expected_final = numpy.array([
        3.20, 2.80, 1.42, 1.14, 0.95, 0.81, 0.68, 0.57,
        0.46, 0.35, 1.70, 1.50, 0.98, 0.76, 0.58, 0.41,
        0.90, 0.61, 0.70, 0.70, 0.60, 0.50, 0.50, 0.50
    ])

    expected_cohn = pandas.DataFrame({
        'nuncen_above': numpy.array([6.0, 1.0, 2.0, 2.0, numpy.nan]),
        'nobs_below': numpy.array([0.0, 7.0, 12.0, 22.0, numpy.nan]),
        'ncen_equal': numpy.array([0.0, 1.0, 4.0, 8.0, numpy.nan]),
        'prob_exceedance': numpy.array([1.0, 0.3125, 0.21429, 0.0833, 0.0]),
    })


class Test_ROS_RNADAdata(CheckROSMixin):
    decimal = 3
    datastring = StringIO(dedent("""\
        res cen
        0.090  True
        0.090  True
        0.090  True
        0.101 False
        0.136 False
        0.340 False
        0.457 False
        0.514 False
        0.629 False
        0.638 False
        0.774 False
        0.788 False
        0.900  True
        0.900  True
        0.900  True
        1.000  True
        1.000  True
        1.000  True
        1.000  True
        1.000  True
        1.000 False
        1.000  True
        1.000  True
        1.000  True
        1.000  True
        1.000  True
        1.000  True
        1.000  True
        1.000  True
        1.000  True
        1.000  True
        1.000  True
        1.000  True
        1.100 False
        2.000 False
        2.000 False
        2.404 False
        2.860 False
        3.000 False
        3.000 False
        3.705 False
        4.000 False
        5.000 False
        5.960 False
        6.000 False
        7.214 False
       16.000 False
       17.716 False
       25.000 False
       51.000 False"""))
    rescol = 'res'
    cencol = 'cen'
    df = pandas.read_csv(datastring, sep=r'\s+')
    expected_final = numpy.array([
        0.01907990,  0.03826254,  0.06080717,  0.10100000,  0.13600000,
        0.34000000,  0.45700000,  0.51400000,  0.62900000,  0.63800000,
        0.77400000,  0.78800000,  0.08745914,  0.25257575,  0.58544205,
        0.01711153,  0.03373885,  0.05287083,  0.07506079,  0.10081573,
        1.00000000,  0.13070334,  0.16539309,  0.20569039,  0.25257575,
        0.30725491,  0.37122555,  0.44636843,  0.53507405,  0.64042242,
        0.76644378,  0.91850581,  1.10390531,  1.10000000,  2.00000000,
        2.00000000,  2.40400000,  2.86000000,  3.00000000,  3.00000000,
        3.70500000,  4.00000000,  5.00000000,  5.96000000,  6.00000000,
        7.21400000, 16.00000000, 17.71600000, 25.00000000, 51.00000000
    ])

    expected_cohn = pandas.DataFrame({
        'nuncen_above': numpy.array([9., 0.0, 18., numpy.nan]),
        'nobs_below': numpy.array([3., 15., 32., numpy.nan]),
        'ncen_equal': numpy.array([3., 3., 17., numpy.nan]),
        'prob_exceedance': numpy.array([0.84, 0.36, 0.36, 0]),
    })


class Test_NoOp_ZeroND(CheckROSMixin):
    decimal = 2
    numpy.random.seed(0)
    N = 20
    res = numpy.random.lognormal(size=N)
    cen = [False] * N
    rescol = 'obs'
    cencol = 'cen'
    df = pandas.DataFrame({rescol: res, cencol: cen})
    expected_final = numpy.array([
        0.38, 0.43, 0.81, 0.86, 0.90, 1.13, 1.15, 1.37, 1.40,
        1.49, 1.51, 1.56, 2.14, 2.59, 2.66, 4.28, 4.46, 5.84,
        6.47, 9.4
    ])

    expected_cohn = pandas.DataFrame({
        'nuncen_above': numpy.array([]),
        'nobs_below': numpy.array([]),
        'ncen_equal': numpy.array([]),
        'prob_exceedance': numpy.array([]),
    })


class Test_ROS_OneND(CheckROSMixin):
    decimal = 3
    res = numpy.array([
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10., 10., 10.,
        3.0, 7.0, 9.0, 12., 15., 20., 27., 33., 50.
    ])
    cen = numpy.array([
        True, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False,
        False, False
    ])
    rescol = 'conc'
    cencol = 'cen'
    df = pandas.DataFrame({rescol: res, cencol: cen})
    expected_final = numpy.array([
        0.24, 1.0, 1.0, 1.0, 1.0, 1.0, 10., 10., 10.,
        3.0 , 7.0, 9.0, 12., 15., 20., 27., 33., 50.
    ])

    expected_cohn = pandas.DataFrame({
        'nuncen_above': numpy.array([17.0, numpy.nan]),
        'nobs_below': numpy.array([1.0, numpy.nan]),
        'ncen_equal': numpy.array([1.0, numpy.nan]),
        'prob_exceedance': numpy.array([0.94444, 0.0]),
    })


class Test_HalfDLs_80pctNDs(CheckROSMixin):
    decimal = 3
    res = numpy.array([
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10., 10., 10.,
        3.0, 7.0, 9.0, 12., 15., 20., 27., 33., 50.
    ])
    cen = numpy.array([
        True, True, True, True, True, True, True, True,
        True, True, True, True, True, True, True, False,
        False, False
    ])
    rescol = 'value'
    cencol = 'qual'
    df = pandas.DataFrame({rescol: res, cencol: cen})
    expected_final = numpy.array([
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 5.0, 5.0, 5.0,
        1.5, 3.5, 4.5, 6.0, 7.5, 10., 27., 33., 50.
    ])

    expected_cohn = pandas.DataFrame({
        'nuncen_above': numpy.array([0., 0., 0., 0., 0., 0., 0., 3., numpy.nan]),
        'nobs_below': numpy.array([6., 7., 8., 9., 12., 13., 14., 15., numpy.nan]),
        'ncen_equal': numpy.array([6., 1., 1., 1., 3., 1., 1., 1., numpy.nan]),
        'prob_exceedance': numpy.array([0.16667] * 8 + [0.]),
    })


class Test_HaflDLs_OneUncensored(CheckROSMixin):
    decimal = 3
    res = numpy.array([1.0, 1.0, 12., 15., ])
    cen = numpy.array([True, True, True, False ])
    rescol = 'value'
    cencol = 'qual'
    df = pandas.DataFrame({rescol: res, cencol: cen})
    expected_final = numpy.array([0.5,   0.5,   6. ,  15.])

    expected_cohn = pandas.DataFrame({
        'nuncen_above': numpy.array([0., 1., numpy.nan]),
        'nobs_below': numpy.array([2., 3., numpy.nan]),
        'ncen_equal': numpy.array([2., 1., numpy.nan]),
        'prob_exceedance': numpy.array([0.25, 0.25, 0.]),
    })


class Test_ROS_MaxCen_GT_MaxUncen(Test_ROS_HelselAppendixB):
    res = numpy.array([
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10., 10., 10.,
        3.0, 7.0, 9.0, 12., 15., 20., 27., 33., 50.,
        60, 70
    ])
    cen = numpy.array([
        True, True, True, True, True, True, True, True, True,
        False, False, False, False, False, False, False,
        False, False, True, True
    ])


class Test_ROS_OnlyDL_GT_MaxUncen(Test_NoOp_ZeroND):
    numpy.random.seed(0)
    N = 20
    res =  [
        0.38, 0.43, 0.81, 0.86, 0.90, 1.13, 1.15, 1.37, 1.40,
        1.49, 1.51, 1.56, 2.14, 2.59, 2.66, 4.28, 4.46, 5.84,
        6.47, 9.40, 10.0, 10.0
    ]
    cen = ([False] * N) + [True, True]
