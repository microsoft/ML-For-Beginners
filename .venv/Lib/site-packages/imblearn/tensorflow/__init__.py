"""The :mod:`imblearn.tensorflow` provides utilities to deal with imbalanced
dataset in tensorflow."""

from ._generator import balanced_batch_generator

__all__ = ["balanced_batch_generator"]
