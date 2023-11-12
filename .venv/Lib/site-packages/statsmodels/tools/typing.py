from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence, Union

from packaging.version import parse
from pandas import DataFrame, Series

if TYPE_CHECKING:
    import numpy as np

    if parse(np.__version__) < parse("1.22.0"):
        raise NotImplementedError(
            "NumPy 1.22.0 or later required for type checking"
        )
    from numpy.typing import (
        ArrayLike as ArrayLike,
        DTypeLike,
        NDArray,
        _FloatLike_co,
        _UIntLike_co,
    )

    _ExtendedFloatLike_co = Union[_FloatLike_co, _UIntLike_co]
    NumericArray = NDArray[Any, np.dtype[_ExtendedFloatLike_co]]
    Float64Array = NDArray[Any, np.double]
    ArrayLike1D = Union[Sequence[Union[float, int]], NumericArray, Series]
    ArrayLike2D = Union[
        Sequence[Sequence[Union[float, int]]], NumericArray, DataFrame
    ]
else:
    ArrayLike = Any
    DTypeLike = Any
    Float64Array = Any
    NumericArray = Any
    ArrayLike1D = Any
    ArrayLike2D = Any
    NDArray = Any

__all__ = [
    "ArrayLike",
    "DTypeLike",
    "Float64Array",
    "ArrayLike1D",
    "ArrayLike2D",
    "NDArray",
    "NumericArray",
]
