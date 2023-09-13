import numpy as np
from scipy._lib._util import IntNumber
from typing import Literal

def _initialize_v(
    v : np.ndarray, 
    dim : IntNumber,
    bits: IntNumber
) -> None: ...

def _cscramble (
    dim : IntNumber,
    bits: IntNumber,
    ltm : np.ndarray,
    sv: np.ndarray
) -> None: ...

def _fill_p_cumulative(
    p: np.ndarray,
    p_cumulative: np.ndarray
) -> None: ...

def _draw(
    n : IntNumber,
    num_gen: IntNumber,
    dim: IntNumber,
    scale: float,
    sv: np.ndarray,
    quasi: np.ndarray,
    sample: np.ndarray
    ) -> None: ...

def _fast_forward(
    n: IntNumber,
    num_gen: IntNumber,
    dim: IntNumber,
    sv: np.ndarray,
    quasi: np.ndarray
    ) -> None: ...

def _categorize(
    draws: np.ndarray,
    p_cumulative: np.ndarray,
    result: np.ndarray
    ) -> None: ...

_MAXDIM: Literal[21201]
_MAXDEG: Literal[18]

def _test_find_index(
    p_cumulative: np.ndarray, 
    size: int, 
    value: float
    ) -> int: ...
