import scipy as sp
import pytest


def test_array_api_deprecations():
    X = sp.sparse.csr_array([
        [1,2,3],
        [4,0,6]
    ])
    msg = "1.13.0"

    with pytest.deprecated_call(match=msg):
        X.get_shape()

    with pytest.deprecated_call(match=msg):
        X.set_shape((2,3))

    with pytest.deprecated_call(match=msg):
        X.asfptype()

    with pytest.deprecated_call(match=msg):
        X.getmaxprint()

    with pytest.deprecated_call(match=msg):
        X.getnnz()

    with pytest.deprecated_call(match=msg):
        X.getH()

    with pytest.deprecated_call(match=msg):
        X.getcol(1).todense()

    with pytest.deprecated_call(match=msg):
        X.getrow(1).todense()
