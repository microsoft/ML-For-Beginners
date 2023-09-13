"""
Small shim of loky's cloudpickle_wrapper to avoid failure when
multiprocessing is not available.
"""


from ._multiprocessing_helpers import mp


def _my_wrap_non_picklable_objects(obj, keep_wrapper=True):
    return obj


if mp is not None:
    from .externals.loky import wrap_non_picklable_objects
else:
    wrap_non_picklable_objects = _my_wrap_non_picklable_objects

__all__ = ["wrap_non_picklable_objects"]
