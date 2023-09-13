from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy.typing as npt

def nnls(
        a: npt.ArrayLike,
        mda: int,
        m: int,
        n: int,
        b: npt.ArrayLike,
        x: npt.ArrayLike,
        rnorm: float,
        w: float,
        zz: float,
        index_bn: int,
        mode: int,
        maxiter: int
) -> tuple[npt.ArrayLike, float, int]: ...
