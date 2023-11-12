"""
The :mod:`imblearn.datasets` provides methods to generate
imbalanced data.
"""

from ._imbalance import make_imbalance
from ._zenodo import fetch_datasets

__all__ = ["make_imbalance", "fetch_datasets"]
