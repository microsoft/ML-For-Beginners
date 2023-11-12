from statsmodels.compat.pandas import assert_frame_equal, assert_series_equal

import numpy as np
from numpy.testing import assert_equal
import pandas as pd
import pytest
from scipy import sparse

from statsmodels.tools.grouputils import (dummy_sparse, Grouping, Group,
                                          combine_indices, group_sums)
from statsmodels.datasets import grunfeld, anes96


class CheckGrouping:

    @pytest.mark.smoke
    def test_reindex(self):
        self.grouping.reindex(self.grouping.index)

    def test_count_categories(self):
        self.grouping.count_categories(level=0)
        np.testing.assert_equal(self.grouping.counts, self.expected_counts)

    def test_sort(self):
        # data frame
        sorted_data, index = self.grouping.sort(self.data)
        expected_sorted_data = self.data.sort_index()

        assert_frame_equal(sorted_data, expected_sorted_data)
        np.testing.assert_(isinstance(sorted_data, pd.DataFrame))
        np.testing.assert_(not index.equals(self.grouping.index))

        # make sure it copied
        if hasattr(sorted_data, 'equals'): # newer pandas
            np.testing.assert_(not sorted_data.equals(self.data))

        # 2d arrays
        sorted_data, index = self.grouping.sort(self.data.values)
        np.testing.assert_array_equal(sorted_data,
                                      expected_sorted_data.values)
        np.testing.assert_(isinstance(sorted_data, np.ndarray))

        # 1d series
        series = self.data[self.data.columns[0]]
        sorted_data, index = self.grouping.sort(series)

        expected_sorted_data = series.sort_index()
        assert_series_equal(sorted_data, expected_sorted_data)
        np.testing.assert_(isinstance(sorted_data, pd.Series))
        if hasattr(sorted_data, 'equals'):
            np.testing.assert_(not sorted_data.equals(series))

        # 1d array
        array = series.values
        sorted_data, index = self.grouping.sort(array)

        expected_sorted_data = series.sort_index().values
        np.testing.assert_array_equal(sorted_data, expected_sorted_data)
        np.testing.assert_(isinstance(sorted_data, np.ndarray))

    def test_transform_dataframe(self):
        names = self.data.index.names
        transformed_dataframe = self.grouping.transform_dataframe(
                                            self.data,
                                            lambda x : x.mean(),
                                            level=0)
        cols = [names[0]] + list(self.data.columns)
        df = self.data.reset_index()[cols].set_index(names[0])
        grouped = df[self.data.columns].groupby(level=0)
        expected = grouped.apply(lambda x : x.mean())
        np.testing.assert_allclose(transformed_dataframe,
                                   expected.values)

        if len(names) > 1:
            transformed_dataframe = self.grouping.transform_dataframe(
                                            self.data, lambda x : x.mean(),
                                            level=1)
            cols = [names[1]] + list(self.data.columns)
            df = self.data.reset_index()[cols].set_index(names[1])
            grouped = df.groupby(level=0)
            expected = grouped.apply(lambda x: x.mean())[self.data.columns]
            np.testing.assert_allclose(transformed_dataframe,
                                       expected.values)

    def test_transform_array(self):
        names = self.data.index.names
        transformed_array = self.grouping.transform_array(
                                            self.data.values,
                                            lambda x : x.mean(),
                                            level=0)
        cols = [names[0]] + list(self.data.columns)
        df = self.data.reset_index()[cols].set_index(names[0])
        grouped = df[self.data.columns].groupby(level=0)
        expected = grouped.apply(lambda x: x.mean())
        np.testing.assert_allclose(transformed_array,
                                   expected.values)

        if len(names) > 1:
            transformed_array = self.grouping.transform_array(
                                            self.data.values,
                                            lambda x : x.mean(), level=1)
            cols = [names[1]] + list(self.data.columns)
            df = self.data.reset_index()[cols].set_index(names[1])
            grouped = df[self.data.columns].groupby(level=0)
            expected = grouped.apply(lambda x: x.mean())[self.data.columns]
            np.testing.assert_allclose(transformed_array,
                                       expected.values)


    def test_transform_slices(self):
        names = self.data.index.names
        transformed_slices = self.grouping.transform_slices(
                                            self.data.values,
                                            lambda x, idx : x.mean(0),  # noqa
                                            level=0)
        expected = self.data.reset_index().groupby(
            names[0])[self.data.columns].mean()
        np.testing.assert_allclose(transformed_slices, expected.values,
                                   rtol=1e-12, atol=1e-25)

        if len(names) > 1:
            transformed_slices = self.grouping.transform_slices(
                                            self.data.values,
                                            lambda x, idx : x.mean(0),  # noqa
                                            level=1)
            expected = self.data.reset_index().groupby(
                names[1])[self.data.columns].mean()
            np.testing.assert_allclose(transformed_slices, expected.values,
                                       rtol=1e-12, atol=1e-25)

    @pytest.mark.smoke
    def test_dummies_groups(self):
        # calls dummy_sparse under the hood
        self.grouping.dummies_groups()

        if len(self.grouping.group_names) > 1:
            self.grouping.dummies_groups(level=1)

    def test_dummy_sparse(self):
        data = self.data
        self.grouping.dummy_sparse()
        values = data.index.get_level_values(0).values
        expected = pd.get_dummies(pd.Series(values, dtype="category"),
                                  drop_first=False)
        np.testing.assert_equal(self.grouping._dummies.toarray(), expected)

        if len(self.grouping.group_names) > 1:
            self.grouping.dummy_sparse(level=1)
            values = data.index.get_level_values(1).values
            expected = pd.get_dummies(pd.Series(values, dtype="category"),
                                      drop_first=False)
            np.testing.assert_equal(self.grouping._dummies.toarray(),
                                    expected)


class TestMultiIndexGrouping(CheckGrouping):
    @classmethod
    def setup_class(cls):
        grun_data = grunfeld.load_pandas().data
        multi_index_data = grun_data.set_index(['firm', 'year'])
        multi_index_panel = multi_index_data.index
        cls.grouping = Grouping(multi_index_panel)
        cls.data = multi_index_data

        cls.expected_counts = [20] * 11


class TestIndexGrouping(CheckGrouping):
    @classmethod
    def setup_class(cls):
        grun_data = grunfeld.load_pandas().data
        index_data = grun_data.set_index(['firm'])
        index_group = index_data.index
        cls.grouping = Grouping(index_group)
        cls.data = index_data

        cls.expected_counts = [20] * 11


def test_init_api():
    # make a multi-index panel
    grun_data = grunfeld.load_pandas().data
    multi_index_panel = grun_data.set_index(['firm', 'year']).index
    grouping = Grouping(multi_index_panel)
    # check group_names
    np.testing.assert_array_equal(grouping.group_names, ['firm', 'year'])
    # check shape
    np.testing.assert_array_equal(grouping.index_shape, (11, 20))
    # check index_int
    np.testing.assert_array_equal(grouping.labels,
      [[ 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
         5, 5, 5, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
         8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
         4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 7, 7,
         7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
         7, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
         9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
         10, 10, 10, 10, 10, 10, 10, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
         6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 3, 3, 3, 3, 3, 3, 3,
         3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
         17, 18, 19, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
         14, 15, 16, 17, 18, 19, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
         11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 1, 2, 3, 4, 5, 6, 7,
         8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 1, 2, 3, 4,
         5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 1,
         2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
         19, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
         16, 17, 18, 19, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
         13, 14, 15, 16, 17, 18, 19, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
         10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 1, 2, 3, 4, 5, 6,
         7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 1, 2, 3,
         4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]])
    grouping = Grouping(multi_index_panel, names=['firms', 'year'])
    np.testing.assert_array_equal(grouping.group_names, ['firms', 'year'])

    # make a multi-index grouping
    anes_data = anes96.load_pandas().data
    multi_index_groups = anes_data.set_index(['educ', 'income',
                                              'TVnews']).index
    grouping = Grouping(multi_index_groups)
    np.testing.assert_array_equal(grouping.group_names,
                                  ['educ', 'income', 'TVnews'])
    np.testing.assert_array_equal(grouping.index_shape, (7, 24, 8))

    # make a list multi-index panel
    list_panel = multi_index_panel.tolist()
    grouping = Grouping(list_panel, names=['firms', 'year'])
    np.testing.assert_array_equal(grouping.group_names, ['firms', 'year'])
    np.testing.assert_array_equal(grouping.index_shape, (11, 20))

    # make a list multi-index grouping
    list_groups = multi_index_groups.tolist()
    grouping = Grouping(list_groups, names=['educ', 'income', 'TVnews'])
    np.testing.assert_array_equal(grouping.group_names,
                                  ['educ', 'income', 'TVnews'])
    np.testing.assert_array_equal(grouping.index_shape, (7, 24, 8))


    # single-variable index grouping
    index_group = multi_index_panel.get_level_values(0)
    grouping = Grouping(index_group)
    # the original multi_index_panel had it's name changed inplace above
    np.testing.assert_array_equal(grouping.group_names, ['firms'])
    np.testing.assert_array_equal(grouping.index_shape, (220,))

    # single variable list grouping
    list_group = multi_index_panel.get_level_values(0).tolist()
    grouping = Grouping(list_group)
    np.testing.assert_array_equal(grouping.group_names, ["group0"])
    np.testing.assert_array_equal(grouping.index_shape, 11*20)

    # test generic group names
    grouping = Grouping(list_groups)
    np.testing.assert_array_equal(grouping.group_names,
                                  ['group0', 'group1', 'group2'])


def test_combine_indices():
    # Moved from grouputils __main__ section
    np.random.seed(985367)
    groups = np.random.randint(0, 2, size=(10, 2))
    uv, ux, u, label = combine_indices(groups, return_labels=True)
    uv, ux, u, label = combine_indices(groups, prefix='g1,g2=', sep=',',
                                       return_labels=True)

    group0 = np.array(['sector0', 'sector1'])[groups[:, 0]]
    group1 = np.array(['region0', 'region1'])[groups[:, 1]]
    uv, ux, u, label = combine_indices((group0, group1),
                                       prefix='sector,region=',
                                       sep=',',
                                       return_labels=True)
    uv, ux, u, label = combine_indices((group0, group1), prefix='', sep='.',
                                       return_labels=True)
    group_joint = np.array(label)[uv]
    group_joint_expected = np.array(['sector1.region0', 'sector0.region1',
                                     'sector0.region0', 'sector0.region1',
                                     'sector1.region1', 'sector0.region0',
                                     'sector1.region0', 'sector1.region0',
                                     'sector0.region1', 'sector0.region0'],
                                    dtype='|U15')
    assert_equal(group_joint, group_joint_expected)


@pytest.mark.smoke
def test_group_sums():
    # Moved from grouputils __main__ section
    g = np.array([0, 0, 1, 2, 1, 1, 2, 0])

    group_sums(np.arange(len(g)*3*2).reshape(len(g), 3, 2), g,
               use_bincount=False).T
    group_sums(np.arange(len(g)*3*2).reshape(len(g), 3, 2)[:, :, 0], g)
    group_sums(np.arange(len(g)*3*2).reshape(len(g), 3, 2)[:, :, 1], g)


@pytest.mark.smoke
def test_group_class():
    # Moved from grouputils __main__ section
    g = np.array([0, 0, 1, 2, 1, 1, 2, 0])

    x = np.arange(len(g)*3).reshape(len(g), 3, order='F')
    mygroup = Group(g)

    mygroup.group_int
    mygroup.group_sums(x)
    mygroup.labels()


def test_dummy_sparse():
    # See GH#5687

    g = np.array([0, 0, 2, 1, 1, 2, 0])
    indi = dummy_sparse(g)
    assert isinstance(indi, sparse.csr_matrix)
    result = indi.todense()
    expected = np.matrix([[1, 0, 0],
                         [1, 0, 0],
                         [0, 0, 1],
                         [0, 1, 0],
                         [0, 1, 0],
                         [0, 0, 1],
                         [1, 0, 0]], dtype=np.int8)
    assert_equal(result, expected)


    # current behavior with missing groups
    g = np.array([0, 0, 2, 0, 2, 0])
    indi = dummy_sparse(g)
    result = indi.todense()
    expected = np.matrix([[1, 0, 0],
                         [1, 0, 0],
                         [0, 0, 1],
                         [1, 0, 0],
                         [0, 0, 1],
                         [1, 0, 0]], dtype=np.int8)
    assert_equal(result, expected)
