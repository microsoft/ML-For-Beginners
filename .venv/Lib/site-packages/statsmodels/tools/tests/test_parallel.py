import warnings
from statsmodels.tools.parallel import parallel_func
from numpy import arange, testing
from math import sqrt

def test_parallel():
    x = arange(10.)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parallel, p_func, n_jobs = parallel_func(sqrt, n_jobs=-1, verbose=0)
        y = parallel(p_func(i**2) for i in range(10))
    testing.assert_equal(x,y)
