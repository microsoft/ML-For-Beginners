"""Test for the deprecation helper"""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT

import pytest

from imblearn.utils.deprecation import deprecate_parameter


class Sampler:
    def __init__(self):
        self.a = "something"
        self.b = "something"


def test_deprecate_parameter():
    with pytest.warns(FutureWarning, match="is deprecated from"):
        deprecate_parameter(Sampler(), "0.2", "a")
    with pytest.warns(FutureWarning, match="Use 'b' instead."):
        deprecate_parameter(Sampler(), "0.2", "a", "b")
