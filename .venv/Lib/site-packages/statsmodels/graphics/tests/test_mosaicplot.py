from statsmodels.compat.python import lrange

from io import BytesIO
from itertools import product

import numpy as np
from numpy.testing import assert_, assert_raises
import pandas as pd
import pytest

from statsmodels.api import datasets

# utilities for the tests

try:
    import matplotlib.pyplot as plt  # noqa:F401
except ImportError:
    pass

# other functions to be tested for accuracy
# the main drawing function
from statsmodels.graphics.mosaicplot import (
    _hierarchical_split,
    _key_splitting,
    _normalize_split,
    _reduce_dict,
    _split_rect,
    mosaic,
)


@pytest.mark.matplotlib
def test_data_conversion(close_figures):
    # It will not reorder the elements
    # so the dictionary will look odd
    # as it key order has the c and b
    # keys swapped
    import pandas
    _, ax = plt.subplots(4, 4)
    data = {'ax': 1, 'bx': 2, 'cx': 3}
    mosaic(data, ax=ax[0, 0], title='basic dict', axes_label=False)
    data = pandas.Series(data)
    mosaic(data, ax=ax[0, 1], title='basic series', axes_label=False)
    data = [1, 2, 3]
    mosaic(data, ax=ax[0, 2], title='basic list', axes_label=False)
    data = np.asarray(data)
    mosaic(data, ax=ax[0, 3], title='basic array', axes_label=False)
    plt.close("all")

    data = {('ax', 'cx'): 1, ('bx', 'cx'): 2, ('ax', 'dx'): 3, ('bx', 'dx'): 4}
    mosaic(data, ax=ax[1, 0], title='compound dict', axes_label=False)
    mosaic(data, ax=ax[2, 0], title='inverted keys dict', index=[1, 0], axes_label=False)
    data = pandas.Series(data)
    mosaic(data, ax=ax[1, 1], title='compound series', axes_label=False)
    mosaic(data, ax=ax[2, 1], title='inverted keys series', index=[1, 0])
    data = [[1, 2], [3, 4]]
    mosaic(data, ax=ax[1, 2], title='compound list', axes_label=False)
    mosaic(data, ax=ax[2, 2], title='inverted keys list', index=[1, 0])
    data = np.array([[1, 2], [3, 4]])
    mosaic(data, ax=ax[1, 3], title='compound array', axes_label=False)
    mosaic(data, ax=ax[2, 3], title='inverted keys array', index=[1, 0], axes_label=False)
    plt.close("all")

    gender = ['male', 'male', 'male', 'female', 'female', 'female']
    pet = ['cat', 'dog', 'dog', 'cat', 'dog', 'cat']
    data = pandas.DataFrame({'gender': gender, 'pet': pet})
    mosaic(data, ['gender'], ax=ax[3, 0], title='dataframe by key 1', axes_label=False)
    mosaic(data, ['pet'], ax=ax[3, 1], title='dataframe by key 2', axes_label=False)
    mosaic(data, ['gender', 'pet'], ax=ax[3, 2], title='both keys', axes_label=False)
    mosaic(data, ['pet', 'gender'], ax=ax[3, 3], title='keys inverted', axes_label=False)
    plt.close("all")
    plt.suptitle('testing data conversion (plot 1 of 4)')


@pytest.mark.matplotlib
def test_mosaic_simple(close_figures):
    # display a simple plot of 4 categories of data, splitted in four
    # levels with increasing size for each group
    # creation of the levels
    key_set = (['male', 'female'], ['old', 'adult', 'young'],
               ['worker', 'unemployed'], ['healty', 'ill'])
    # the cartesian product of all the categories is
    # the complete set of categories
    keys = list(product(*key_set))
    data = dict(zip(keys, range(1, 1 + len(keys))))
    # which colours should I use for the various categories?
    # put it into a dict
    props = {}
    #males and females in blue and red
    props[('male',)] = {'color': 'b'}
    props[('female',)] = {'color': 'r'}
    # all the groups corresponding to ill groups have a different color
    for key in keys:
        if 'ill' in key:
            if 'male' in key:
                props[key] = {'color': 'BlueViolet' , 'hatch': '+'}
            else:
                props[key] = {'color': 'Crimson' , 'hatch': '+'}
    # mosaic of the data, with given gaps and colors
    mosaic(data, gap=0.05, properties=props, axes_label=False)
    plt.suptitle('syntetic data, 4 categories (plot 2 of 4)')


@pytest.mark.matplotlib
def test_mosaic(close_figures):
    # make the same analysis on a known dataset

    # load the data and clean it a bit
    affairs = datasets.fair.load_pandas()
    datas = affairs.exog
    # any time greater than 0 is cheating
    datas['cheated'] = affairs.endog > 0
    # sort by the marriage quality and give meaningful name
    # [rate_marriage, age, yrs_married, children,
    # religious, educ, occupation, occupation_husb]
    datas = datas.sort_values(['rate_marriage', 'religious'])

    num_to_desc = {1: 'awful', 2: 'bad', 3: 'intermediate',
                   4: 'good', 5: 'wonderful'}
    datas['rate_marriage'] = datas['rate_marriage'].map(num_to_desc)
    num_to_faith = {1: 'non religious', 2: 'poorly religious', 3: 'religious',
                    4: 'very religious'}
    datas['religious'] = datas['religious'].map(num_to_faith)
    num_to_cheat = {False: 'faithful', True: 'cheated'}
    datas['cheated'] = datas['cheated'].map(num_to_cheat)
    # finished cleaning
    _, ax = plt.subplots(2, 2)
    mosaic(datas, ['rate_marriage', 'cheated'], ax=ax[0, 0],
           title='by marriage happiness')
    mosaic(datas, ['religious', 'cheated'], ax=ax[0, 1],
           title='by religiosity')
    mosaic(datas, ['rate_marriage', 'religious', 'cheated'], ax=ax[1, 0],
           title='by both', labelizer=lambda k:'')
    ax[1, 0].set_xlabel('marriage rating')
    ax[1, 0].set_ylabel('religion status')
    mosaic(datas, ['religious', 'rate_marriage'], ax=ax[1, 1],
           title='inter-dependence', axes_label=False)
    plt.suptitle("extramarital affairs (plot 3 of 4)")


@pytest.mark.matplotlib
def test_mosaic_very_complex(close_figures):
    # make a scattermatrix of mosaic plots to show the correlations between
    # each pair of variable in a dataset. Could be easily converted into a
    # new function that does this automatically based on the type of data
    key_name = ['gender', 'age', 'health', 'work']
    key_base = (['male', 'female'], ['old', 'young'],
                ['healty', 'ill'], ['work', 'unemployed'])
    keys = list(product(*key_base))
    data = dict(zip(keys, range(1, 1 + len(keys))))
    props = {}
    props[('male', 'old')] = {'color': 'r'}
    props[('female',)] = {'color': 'pink'}
    L = len(key_base)
    _, axes = plt.subplots(L, L)
    for i in range(L):
        for j in range(L):
            m = set(range(L)).difference(set((i, j)))
            if i == j:
                axes[i, i].text(0.5, 0.5, key_name[i],
                                ha='center', va='center')
                axes[i, i].set_xticks([])
                axes[i, i].set_xticklabels([])
                axes[i, i].set_yticks([])
                axes[i, i].set_yticklabels([])
            else:
                ji = max(i, j)
                ij = min(i, j)
                temp_data = dict([((k[ij], k[ji]) + tuple(k[r] for r in m), v)
                                  for k, v in data.items()])

                keys = list(temp_data.keys())
                for k in keys:
                    value = _reduce_dict(temp_data, k[:2])
                    temp_data[k[:2]] = value
                    del temp_data[k]
                mosaic(temp_data, ax=axes[i, j], axes_label=False,
                       properties=props, gap=0.05, horizontal=i > j)
    plt.suptitle('old males should look bright red,  (plot 4 of 4)')


@pytest.mark.matplotlib
def test_axes_labeling(close_figures):
    from numpy.random import rand
    key_set = (['male', 'female'], ['old', 'adult', 'young'],
               ['worker', 'unemployed'], ['yes', 'no'])
    # the cartesian product of all the categories is
    # the complete set of categories
    keys = list(product(*key_set))
    data = dict(zip(keys, rand(len(keys))))
    lab = lambda k: ''.join(s[0] for s in k)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    mosaic(data, ax=ax1, labelizer=lab, horizontal=True, label_rotation=45)
    mosaic(data, ax=ax2, labelizer=lab, horizontal=False,
           label_rotation=[0, 45, 90, 0])
    #fig.tight_layout()
    fig.suptitle("correct alignment of the axes labels")


@pytest.mark.smoke
@pytest.mark.matplotlib
def test_mosaic_empty_cells(close_figures):
    # GH#2286
    import pandas as pd
    mydata = pd.DataFrame({'id2': {64: 'Angelica',
                                   65: 'DXW_UID', 66: 'casuid01',
                                   67: 'casuid01', 68: 'EC93_uid',
                                   69: 'EC93_uid', 70: 'EC93_uid',
                                   60: 'DXW_UID',  61: 'AtmosFox',
                                   62: 'DXW_UID', 63: 'DXW_UID'},
                           'id1': {64: 'TGP',
                                   65: 'Retention01', 66: 'default',
                                   67: 'default', 68: 'Musa_EC_9_3',
                                   69: 'Musa_EC_9_3', 70: 'Musa_EC_9_3',
                                   60: 'default', 61: 'default',
                                   62: 'default', 63: 'default'}})

    ct = pd.crosstab(mydata.id1, mydata.id2)
    _, vals = mosaic(ct.T.unstack())
    _, vals = mosaic(mydata, ['id1','id2'])


eq = lambda x, y: assert_(np.allclose(x, y))


def test_recursive_split():
    keys = list(product('mf'))
    data = dict(zip(keys, [1] * len(keys)))
    res = _hierarchical_split(data, gap=0)
    assert_(list(res.keys()) == keys)
    res[('m',)] = (0.0, 0.0, 0.5, 1.0)
    res[('f',)] = (0.5, 0.0, 0.5, 1.0)
    keys = list(product('mf', 'yao'))
    data = dict(zip(keys, [1] * len(keys)))
    res = _hierarchical_split(data, gap=0)
    assert_(list(res.keys()) == keys)
    res[('m', 'y')] = (0.0, 0.0, 0.5, 1 / 3)
    res[('m', 'a')] = (0.0, 1 / 3, 0.5, 1 / 3)
    res[('m', 'o')] = (0.0, 2 / 3, 0.5, 1 / 3)
    res[('f', 'y')] = (0.5, 0.0, 0.5, 1 / 3)
    res[('f', 'a')] = (0.5, 1 / 3, 0.5, 1 / 3)
    res[('f', 'o')] = (0.5, 2 / 3, 0.5, 1 / 3)


def test__reduce_dict():
    data = dict(zip(list(product('mf', 'oy', 'wn')), [1] * 8))
    eq(_reduce_dict(data, ('m',)), 4)
    eq(_reduce_dict(data, ('m', 'o')), 2)
    eq(_reduce_dict(data, ('m', 'o', 'w')), 1)
    data = dict(zip(list(product('mf', 'oy', 'wn')), lrange(8)))
    eq(_reduce_dict(data, ('m',)), 6)
    eq(_reduce_dict(data, ('m', 'o')), 1)
    eq(_reduce_dict(data, ('m', 'o', 'w')), 0)


def test__key_splitting():
    # subdivide starting with an empty tuple
    base_rect = {tuple(): (0, 0, 1, 1)}
    res = _key_splitting(base_rect, ['a', 'b'], [1, 1], tuple(), True, 0)
    assert_(list(res.keys()) == [('a',), ('b',)])
    eq(res[('a',)], (0, 0, 0.5, 1))
    eq(res[('b',)], (0.5, 0, 0.5, 1))
    # subdivide a in two sublevel
    res_bis = _key_splitting(res, ['c', 'd'], [1, 1], ('a',), False, 0)
    assert_(list(res_bis.keys()) == [('a', 'c'), ('a', 'd'), ('b',)])
    eq(res_bis[('a', 'c')], (0.0, 0.0, 0.5, 0.5))
    eq(res_bis[('a', 'd')], (0.0, 0.5, 0.5, 0.5))
    eq(res_bis[('b',)], (0.5, 0, 0.5, 1))
    # starting with a non empty tuple and uneven distribution
    base_rect = {('total',): (0, 0, 1, 1)}
    res = _key_splitting(base_rect, ['a', 'b'], [1, 2], ('total',), True, 0)
    assert_(list(res.keys()) == [('total',) + (e,) for e in ['a', 'b']])
    eq(res[('total', 'a')], (0, 0, 1 / 3, 1))
    eq(res[('total', 'b')], (1 / 3, 0, 2 / 3, 1))


def test_proportion_normalization():
    # extremes should give the whole set, as well
    # as if 0 is inserted
    eq(_normalize_split(0.), [0.0, 0.0, 1.0])
    eq(_normalize_split(1.), [0.0, 1.0, 1.0])
    eq(_normalize_split(2.), [0.0, 1.0, 1.0])
    # negative values should raise ValueError
    assert_raises(ValueError, _normalize_split, -1)
    assert_raises(ValueError, _normalize_split, [1., -1])
    assert_raises(ValueError, _normalize_split, [1., -1, 0.])
    # if everything is zero it will complain
    assert_raises(ValueError, _normalize_split, [0.])
    assert_raises(ValueError, _normalize_split, [0., 0.])
    # one-element array should return the whole interval
    eq(_normalize_split([0.5]), [0.0, 1.0])
    eq(_normalize_split([1.]), [0.0, 1.0])
    eq(_normalize_split([2.]), [0.0, 1.0])
    # simple division should give two pieces
    for x in [0.3, 0.5, 0.9]:
        eq(_normalize_split(x), [0., x, 1.0])
    # multiple division should split as the sum of the components
    for x, y in [(0.25, 0.5), (0.1, 0.8), (10., 30.)]:
        eq(_normalize_split([x, y]), [0., x / (x + y), 1.0])
    for x, y, z in [(1., 1., 1.), (0.1, 0.5, 0.7), (10., 30., 40)]:
        eq(_normalize_split(
            [x, y, z]), [0., x / (x + y + z), (x + y) / (x + y + z), 1.0])


def test_false_split():
    # if you ask it to be divided in only one piece, just return the original
    # one
    pure_square = [0., 0., 1., 1.]
    conf_h = dict(proportion=[1], gap=0.0, horizontal=True)
    conf_v = dict(proportion=[1], gap=0.0, horizontal=False)
    eq(_split_rect(*pure_square, **conf_h), pure_square)
    eq(_split_rect(*pure_square, **conf_v), pure_square)
    conf_h = dict(proportion=[1], gap=0.5, horizontal=True)
    conf_v = dict(proportion=[1], gap=0.5, horizontal=False)
    eq(_split_rect(*pure_square, **conf_h), pure_square)
    eq(_split_rect(*pure_square, **conf_v), pure_square)

    # identity on a void rectangle should not give anything strange
    null_square = [0., 0., 0., 0.]
    conf = dict(proportion=[1], gap=0.0, horizontal=True)
    eq(_split_rect(*null_square, **conf), null_square)
    conf = dict(proportion=[1], gap=1.0, horizontal=True)
    eq(_split_rect(*null_square, **conf), null_square)

    # splitting a negative rectangle should raise error
    neg_square = [0., 0., -1., 0.]
    conf = dict(proportion=[1], gap=0.0, horizontal=True)
    assert_raises(ValueError, _split_rect, *neg_square, **conf)
    conf = dict(proportion=[1, 1], gap=0.0, horizontal=True)
    assert_raises(ValueError, _split_rect, *neg_square, **conf)
    conf = dict(proportion=[1], gap=0.5, horizontal=True)
    assert_raises(ValueError, _split_rect, *neg_square, **conf)
    conf = dict(proportion=[1, 1], gap=0.5, horizontal=True)
    assert_raises(ValueError, _split_rect, *neg_square, **conf)


def test_rect_pure_split():
    pure_square = [0., 0., 1., 1.]
    # division in two equal pieces from the perfect square
    h_2split = [(0.0, 0.0, 0.5, 1.0), (0.5, 0.0, 0.5, 1.0)]
    conf_h = dict(proportion=[1, 1], gap=0.0, horizontal=True)
    eq(_split_rect(*pure_square, **conf_h), h_2split)

    v_2split = [(0.0, 0.0, 1.0, 0.5), (0.0, 0.5, 1.0, 0.5)]
    conf_v = dict(proportion=[1, 1], gap=0.0, horizontal=False)
    eq(_split_rect(*pure_square, **conf_v), v_2split)

    # division in two non-equal pieces from the perfect square
    h_2split = [(0.0, 0.0, 1 / 3, 1.0), (1 / 3, 0.0, 2 / 3, 1.0)]
    conf_h = dict(proportion=[1, 2], gap=0.0, horizontal=True)
    eq(_split_rect(*pure_square, **conf_h), h_2split)

    v_2split = [(0.0, 0.0, 1.0, 1 / 3), (0.0, 1 / 3, 1.0, 2 / 3)]
    conf_v = dict(proportion=[1, 2], gap=0.0, horizontal=False)
    eq(_split_rect(*pure_square, **conf_v), v_2split)

    # division in three equal pieces from the perfect square
    h_2split = [(0.0, 0.0, 1 / 3, 1.0), (1 / 3, 0.0, 1 / 3, 1.0), (2 / 3, 0.0,
                 1 / 3, 1.0)]
    conf_h = dict(proportion=[1, 1, 1], gap=0.0, horizontal=True)
    eq(_split_rect(*pure_square, **conf_h), h_2split)

    v_2split = [(0.0, 0.0, 1.0, 1 / 3), (0.0, 1 / 3, 1.0, 1 / 3), (0.0, 2 / 3,
                 1.0, 1 / 3)]
    conf_v = dict(proportion=[1, 1, 1], gap=0.0, horizontal=False)
    eq(_split_rect(*pure_square, **conf_v), v_2split)

    # division in three non-equal pieces from the perfect square
    h_2split = [(0.0, 0.0, 1 / 4, 1.0), (1 / 4, 0.0, 1 / 2, 1.0), (3 / 4, 0.0,
                 1 / 4, 1.0)]
    conf_h = dict(proportion=[1, 2, 1], gap=0.0, horizontal=True)
    eq(_split_rect(*pure_square, **conf_h), h_2split)

    v_2split = [(0.0, 0.0, 1.0, 1 / 4), (0.0, 1 / 4, 1.0, 1 / 2), (0.0, 3 / 4,
                 1.0, 1 / 4)]
    conf_v = dict(proportion=[1, 2, 1], gap=0.0, horizontal=False)
    eq(_split_rect(*pure_square, **conf_v), v_2split)

    # splitting on a void rectangle should give multiple void
    null_square = [0., 0., 0., 0.]
    conf = dict(proportion=[1, 1], gap=0.0, horizontal=True)
    eq(_split_rect(*null_square, **conf), [null_square, null_square])
    conf = dict(proportion=[1, 2], gap=1.0, horizontal=True)
    eq(_split_rect(*null_square, **conf), [null_square, null_square])


def test_rect_deformed_split():
    non_pure_square = [1., -1., 1., 0.5]
    # division in two equal pieces from the perfect square
    h_2split = [(1.0, -1.0, 0.5, 0.5), (1.5, -1.0, 0.5, 0.5)]
    conf_h = dict(proportion=[1, 1], gap=0.0, horizontal=True)
    eq(_split_rect(*non_pure_square, **conf_h), h_2split)

    v_2split = [(1.0, -1.0, 1.0, 0.25), (1.0, -0.75, 1.0, 0.25)]
    conf_v = dict(proportion=[1, 1], gap=0.0, horizontal=False)
    eq(_split_rect(*non_pure_square, **conf_v), v_2split)

    # division in two non-equal pieces from the perfect square
    h_2split = [(1.0, -1.0, 1 / 3, 0.5), (1 + 1 / 3, -1.0, 2 / 3, 0.5)]
    conf_h = dict(proportion=[1, 2], gap=0.0, horizontal=True)
    eq(_split_rect(*non_pure_square, **conf_h), h_2split)

    v_2split = [(1.0, -1.0, 1.0, 1 / 6), (1.0, 1 / 6 - 1, 1.0, 2 / 6)]
    conf_v = dict(proportion=[1, 2], gap=0.0, horizontal=False)
    eq(_split_rect(*non_pure_square, **conf_v), v_2split)


def test_gap_split():
    pure_square = [0., 0., 1., 1.]

    # null split
    conf_h = dict(proportion=[1], gap=1.0, horizontal=True)
    eq(_split_rect(*pure_square, **conf_h), pure_square)

    # equal split
    h_2split = [(0.0, 0.0, 0.25, 1.0), (0.75, 0.0, 0.25, 1.0)]
    conf_h = dict(proportion=[1, 1], gap=1.0, horizontal=True)
    eq(_split_rect(*pure_square, **conf_h), h_2split)

    # disequal split
    h_2split = [(0.0, 0.0, 1 / 6, 1.0), (0.5 + 1 / 6, 0.0, 1 / 3, 1.0)]
    conf_h = dict(proportion=[1, 2], gap=1.0, horizontal=True)
    eq(_split_rect(*pure_square, **conf_h), h_2split)


@pytest.mark.matplotlib
def test_default_arg_index(close_figures):
    # 2116
    df = pd.DataFrame({'size' : ['small', 'large', 'large', 'small', 'large',
                                 'small'],
                       'length' : ['long', 'short', 'short', 'long', 'long',
                                   'short']})
    assert_raises(ValueError, mosaic, data=df, title='foobar')


@pytest.mark.matplotlib
def test_missing_category(close_figures):
    # GH5639
    animal = ['dog', 'dog', 'dog', 'cat', 'dog', 'cat', 'cat',
              'dog', 'dog', 'cat']
    size = ['medium', 'large', 'medium', 'medium', 'medium', 'medium',
            'large', 'large', 'large', 'small']
    testdata = pd.DataFrame({'animal': animal, 'size': size})
    testdata['size'] = pd.Categorical(testdata['size'],
                                      categories=['small', 'medium', 'large'])
    testdata = testdata.sort_values('size')
    fig, _ = mosaic(testdata, ['animal', 'size'])
    bio = BytesIO()
    fig.savefig(bio, format='png')
