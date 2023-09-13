import numpy as np

from pandas import (
    PeriodIndex,
    factorize,
)
import pandas._testing as tm


class TestFactorize:
    def test_factorize(self):
        idx1 = PeriodIndex(
            ["2014-01", "2014-01", "2014-02", "2014-02", "2014-03", "2014-03"], freq="M"
        )

        exp_arr = np.array([0, 0, 1, 1, 2, 2], dtype=np.intp)
        exp_idx = PeriodIndex(["2014-01", "2014-02", "2014-03"], freq="M")

        arr, idx = idx1.factorize()
        tm.assert_numpy_array_equal(arr, exp_arr)
        tm.assert_index_equal(idx, exp_idx)

        arr, idx = idx1.factorize(sort=True)
        tm.assert_numpy_array_equal(arr, exp_arr)
        tm.assert_index_equal(idx, exp_idx)

        idx2 = PeriodIndex(
            ["2014-03", "2014-03", "2014-02", "2014-01", "2014-03", "2014-01"], freq="M"
        )

        exp_arr = np.array([2, 2, 1, 0, 2, 0], dtype=np.intp)
        arr, idx = idx2.factorize(sort=True)
        tm.assert_numpy_array_equal(arr, exp_arr)
        tm.assert_index_equal(idx, exp_idx)

        exp_arr = np.array([0, 0, 1, 2, 0, 2], dtype=np.intp)
        exp_idx = PeriodIndex(["2014-03", "2014-02", "2014-01"], freq="M")
        arr, idx = idx2.factorize()
        tm.assert_numpy_array_equal(arr, exp_arr)
        tm.assert_index_equal(idx, exp_idx)

    def test_factorize_complex(self):  # TODO: WTF is this test doing here?s
        # GH 17927
        array = [1, 2, 2 + 1j]
        msg = "factorize with argument that is not not a Series"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            labels, uniques = factorize(array)

        expected_labels = np.array([0, 1, 2], dtype=np.intp)
        tm.assert_numpy_array_equal(labels, expected_labels)

        # Should return a complex dtype in the future
        expected_uniques = np.array([(1 + 0j), (2 + 0j), (2 + 1j)], dtype=object)
        tm.assert_numpy_array_equal(uniques, expected_uniques)
