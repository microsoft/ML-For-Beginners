"""Test utilities for docstring."""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT

import pytest

from imblearn.utils import Substitution
from imblearn.utils._docstring import _n_jobs_docstring, _random_state_docstring

func_docstring = """A function.

    Parameters
    ----------
    xxx

    yyy
    """


def func(param_1, param_2):
    """A function.

    Parameters
    ----------
    {param_1}

    {param_2}
    """
    return param_1, param_2


cls_docstring = """A class.

    Parameters
    ----------
    xxx

    yyy
    """


class cls:
    """A class.

    Parameters
    ----------
    {param_1}

    {param_2}
    """

    def __init__(self, param_1, param_2):
        self.param_1 = param_1
        self.param_2 = param_2


@pytest.mark.parametrize(
    "obj, obj_docstring", [(func, func_docstring), (cls, cls_docstring)]
)
def test_docstring_inject(obj, obj_docstring):
    obj_injected_docstring = Substitution(param_1="xxx", param_2="yyy")(obj)
    assert obj_injected_docstring.__doc__ == obj_docstring


def test_docstring_template():
    assert "random_state" in _random_state_docstring
    assert "n_jobs" in _n_jobs_docstring


def test_docstring_with_python_OO():
    """Check that we don't raise a warning if the code is executed with -OO.

    Non-regression test for:
    https://github.com/scikit-learn-contrib/imbalanced-learn/issues/945
    """
    instance = cls(param_1="xxx", param_2="yyy")
    instance.__doc__ = None  # simulate -OO

    instance = Substitution(param_1="xxx", param_2="yyy")(instance)

    assert instance.__doc__ is None
