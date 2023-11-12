
import numpy as np
from numpy.testing import assert_equal
from statsmodels.tools.catadd import add_indep

from scipy import linalg

def test_add_indep():
    x1 = np.array([0,0,0,0,0,1,1,1,2,2,2])
    x2 = np.array([0,0,0,0,0,1,1,1,1,1,1])
    x0 = np.ones(len(x2))
    x = np.column_stack([x0, x1[:,None]*np.arange(3), x2[:,None]*np.arange(2)])
    varnames = ['const'] + ['var1_%d' %i for i in np.arange(3)] \
                         + ['var2_%d' %i for i in np.arange(2)]
    xo, vo = add_indep(x, varnames)

    assert_equal(xo, np.column_stack((x0, x1, x2)))
    assert_equal((linalg.svdvals(x) > 1e-12).sum(), 3)
    assert_equal(vo, ['const', 'var1_1', 'var2_1'])
