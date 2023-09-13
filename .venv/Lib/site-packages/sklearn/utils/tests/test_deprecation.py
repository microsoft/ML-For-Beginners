# Authors: Raghav RV <rvraghav93@gmail.com>
# License: BSD 3 clause


import pickle

import pytest

from sklearn.utils.deprecation import _is_deprecated, deprecated


@deprecated("qwerty")
class MockClass1:
    pass


class MockClass2:
    @deprecated("mockclass2_method")
    def method(self):
        pass

    @deprecated("n_features_ is deprecated")  # type: ignore
    @property
    def n_features_(self):
        """Number of input features."""
        return 10


class MockClass3:
    @deprecated()
    def __init__(self):
        pass


class MockClass4:
    pass


class MockClass5(MockClass1):
    """Inherit from deprecated class but does not call super().__init__."""

    def __init__(self, a):
        self.a = a


@deprecated("a message")
class MockClass6:
    """A deprecated class that overrides __new__."""

    def __new__(cls, *args, **kwargs):
        assert len(args) > 0
        return super().__new__(cls)


@deprecated()
def mock_function():
    return 10


def test_deprecated():
    with pytest.warns(FutureWarning, match="qwerty"):
        MockClass1()
    with pytest.warns(FutureWarning, match="mockclass2_method"):
        MockClass2().method()
    with pytest.warns(FutureWarning, match="deprecated"):
        MockClass3()
    with pytest.warns(FutureWarning, match="qwerty"):
        MockClass5(42)
    with pytest.warns(FutureWarning, match="a message"):
        MockClass6(42)
    with pytest.warns(FutureWarning, match="deprecated"):
        val = mock_function()
    assert val == 10


def test_is_deprecated():
    # Test if _is_deprecated helper identifies wrapping via deprecated
    # NOTE it works only for class methods and functions
    assert _is_deprecated(MockClass1.__new__)
    assert _is_deprecated(MockClass2().method)
    assert _is_deprecated(MockClass3.__init__)
    assert not _is_deprecated(MockClass4.__init__)
    assert _is_deprecated(MockClass5.__new__)
    assert _is_deprecated(mock_function)


def test_pickle():
    pickle.loads(pickle.dumps(mock_function))
