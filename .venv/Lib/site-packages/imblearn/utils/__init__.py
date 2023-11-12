"""
The :mod:`imblearn.utils` module includes various utilities.
"""

from ._docstring import Substitution
from ._validation import (
    check_neighbors_object,
    check_sampling_strategy,
    check_target_type,
)

__all__ = [
    "check_neighbors_object",
    "check_sampling_strategy",
    "check_target_type",
    "Substitution",
]
