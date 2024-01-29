from numpy.typing import NDArray
from typing import Any

def lu_decompose(a: NDArray[Any], lu: NDArray[Any], perm: NDArray[Any], permute_l: bool) -> None: ...  # noqa: E501

def lu_dispatcher(a: NDArray[Any], lu: NDArray[Any], perm: NDArray[Any], permute_l: bool) -> None: ...  # noqa: E501
