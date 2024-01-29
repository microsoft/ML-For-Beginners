# This file is not meant for public use and will be removed in SciPy v2.0.0.

from importlib import import_module
import warnings

__all__ = [  # noqa: F822
    'docformat', 'inherit_docstring_from', 'indentcount_lines',
    'filldoc', 'unindent_dict', 'unindent_string', 'extend_notes_in_docstring',
    'replace_notes_in_docstring'
]


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
            f"`scipy.misc.doccer` has no attribute `{name}`; furthermore, "
            f"`scipy.misc.doccer` is deprecated and will be removed in SciPy 2.0.0."
        )

    attr = getattr(import_module("scipy._lib.doccer"), name, None)

    if attr is not None:
        message = (
            f"Please import `{name}` from the `scipy._lib.doccer` namespace; "
            f"the `scipy.misc.doccer` namespace is deprecated and "
            f"will be removed in SciPy 2.0.0."
        )
    else:
        message = (
            f"`scipy.misc.doccer.{name}` is deprecated along with "
            f"the `scipy.misc.doccer` namespace. "
            f"`scipy.misc.doccer.{name}` will be removed in SciPy 1.13.0, and "
            f"the `scipy.misc.doccer` namespace will be removed in SciPy 2.0.0."
        )

    warnings.warn(message, category=DeprecationWarning, stacklevel=2)

    try:
        return getattr(import_module("scipy._lib.doccer"), name)
    except AttributeError as e:
        raise e
