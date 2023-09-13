import pickle

import pytest

from sklearn.utils.metaestimators import available_if


class AvailableParameterEstimator:
    """This estimator's `available` parameter toggles the presence of a method"""

    def __init__(self, available=True, return_value=1):
        self.available = available
        self.return_value = return_value

    @available_if(lambda est: est.available)
    def available_func(self):
        """This is a mock available_if function"""
        return self.return_value


def test_available_if_docstring():
    assert "This is a mock available_if function" in str(
        AvailableParameterEstimator.__dict__["available_func"].__doc__
    )
    assert "This is a mock available_if function" in str(
        AvailableParameterEstimator.available_func.__doc__
    )
    assert "This is a mock available_if function" in str(
        AvailableParameterEstimator().available_func.__doc__
    )


def test_available_if():
    assert hasattr(AvailableParameterEstimator(), "available_func")
    assert not hasattr(AvailableParameterEstimator(available=False), "available_func")


def test_available_if_unbound_method():
    # This is a non regression test for:
    # https://github.com/scikit-learn/scikit-learn/issues/20614
    # to make sure that decorated functions can be used as an unbound method,
    # for instance when monkeypatching.
    est = AvailableParameterEstimator()
    AvailableParameterEstimator.available_func(est)

    est = AvailableParameterEstimator(available=False)
    with pytest.raises(
        AttributeError,
        match="This 'AvailableParameterEstimator' has no attribute 'available_func'",
    ):
        AvailableParameterEstimator.available_func(est)


def test_available_if_methods_can_be_pickled():
    """Check that available_if methods can be pickled.

    Non-regression test for #21344.
    """
    return_value = 10
    est = AvailableParameterEstimator(available=True, return_value=return_value)
    pickled_bytes = pickle.dumps(est.available_func)
    unpickled_func = pickle.loads(pickled_bytes)
    assert unpickled_func() == return_value
