import numpy as np
from numpy.testing import assert_array_almost_equal

from statsmodels.sandbox.distributions.gof_new import bootstrap, NewNorm


def test_loop_vectorized_batch_equivalence():
    # test equality of loop, vectorized, batch-vectorized
    nobs = 200

    np.random.seed(8765679)
    resu1 = bootstrap(NewNorm(), args=(0, 1), nobs=nobs, nrep=100,
                      value=0.576/(1 + 4./nobs - 25./nobs**2))

    np.random.seed(8765679)
    tmp = [bootstrap(NewNorm(), args=(0, 1), nobs=nobs, nrep=1)
           for _ in range(100)]
    resu2 = (np.array(tmp) > 0.576/(1 + 4./nobs - 25./nobs**2)).mean()

    np.random.seed(8765679)
    tmp = [bootstrap(NewNorm(), args=(0, 1), nobs=nobs, nrep=1,
                     value=0.576/(1 + 4./nobs - 25./nobs**2),
                     batch_size=10) for _ in range(10)]
    resu3 = np.array(tmp).mean()

    assert_array_almost_equal(resu1, resu2, 15)
    assert_array_almost_equal(resu2, resu3, 15)
