from IPython.utils.syspathcontext import appended_to_syspath
import pytest


def test_append_deprecated():
    with pytest.warns(DeprecationWarning):
        appended_to_syspath(".")
