"""
Test that our implementation of wrap_non_picklable_objects mimics
properly the loky implementation.
"""

from .._cloudpickle_wrapper import wrap_non_picklable_objects
from .._cloudpickle_wrapper import _my_wrap_non_picklable_objects


def a_function(x):
    return x


class AClass(object):

    def __call__(self, x):
        return x


def test_wrap_non_picklable_objects():
    # Mostly a smoke test: test that we can use callable in the same way
    # with both our implementation of wrap_non_picklable_objects and the
    # upstream one
    for obj in (a_function, AClass()):
        wrapped_obj = wrap_non_picklable_objects(obj)
        my_wrapped_obj = _my_wrap_non_picklable_objects(obj)
        assert wrapped_obj(1) == my_wrapped_obj(1)
