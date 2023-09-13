import numpy as np
from typing import (Union, overload, Callable, NamedTuple,
                    Protocol)
import numpy.typing as npt
from scipy._lib._util import SeedType
import scipy.stats as stats


ArrayLike0D = Union[bool, int, float, complex, str, bytes, np.generic]


__all__: list[str]


class UNURANError(RuntimeError):
    ...


class Method:
    @overload
    def rvs(self, size: None = ...) -> float | int: ...  # type: ignore[misc]
    @overload
    def rvs(self, size: int | tuple[int, ...] = ...) -> np.ndarray: ...
    def set_random_state(self, random_state: SeedType) -> None: ...


class TDRDist(Protocol):
    @property
    def pdf(self) -> Callable[..., float]: ...
    @property
    def dpdf(self) -> Callable[..., float]: ...
    @property
    def support(self) -> tuple[float, float]: ...


class TransformedDensityRejection(Method):
    def __init__(self,
                 dist: TDRDist,
                 *,
                 mode: None | float = ...,
                 center: None | float = ...,
                 domain: None | tuple[float, float] = ...,
                 c: float = ...,
                 construction_points: int | npt.ArrayLike = ...,
                 use_dars: bool = ...,
                 max_squeeze_hat_ratio: float = ...,
                 random_state: SeedType = ...) -> None: ...
    @property
    def squeeze_hat_ratio(self) -> float: ...
    @property
    def squeeze_area(self) -> float: ...
    @overload
    def ppf_hat(self, u: ArrayLike0D) -> float: ...  # type: ignore[misc]
    @overload
    def ppf_hat(self, u: npt.ArrayLike) -> np.ndarray: ...


class SROUDist(Protocol):
    @property
    def pdf(self) -> Callable[..., float]: ...
    @property
    def support(self) -> tuple[float, float]: ...


class SimpleRatioUniforms(Method):
    def __init__(self,
                 dist: SROUDist,
                 *,
                 mode: None | float = ...,
                 pdf_area: float = ...,
                 domain: None | tuple[float, float] = ...,
                 cdf_at_mode: float = ...,
                 random_state: SeedType = ...) -> None: ...


class UError(NamedTuple):
    max_error: float
    mean_absolute_error: float

class PINVDist(Protocol):
    @property
    def pdf(self) -> Callable[..., float]: ...
    @property
    def cdf(self) -> Callable[..., float]: ...
    @property
    def logpdf(self) -> Callable[..., float]: ...


class NumericalInversePolynomial(Method):
    def __init__(self,
                 dist: PINVDist,
                 *,
                 mode: None | float = ...,
                 center: None | float = ...,
                 domain: None | tuple[float, float] = ...,
                 order: int = ...,
                 u_resolution: float = ...,
                 random_state: SeedType = ...) -> None: ...
    @property
    def intervals(self) -> int: ...
    @overload
    def ppf(self, u: ArrayLike0D) -> float: ...  # type: ignore[misc]
    @overload
    def ppf(self, u: npt.ArrayLike) -> np.ndarray: ...
    @overload
    def cdf(self, x: ArrayLike0D) -> float: ...  # type: ignore[misc]
    @overload
    def cdf(self, x: npt.ArrayLike) -> np.ndarray: ...
    def u_error(self, sample_size: int = ...) -> UError: ...
    def qrvs(self,
             size: None | int | tuple[int, ...] = ...,
             d: None | int = ...,
             qmc_engine: None | stats.qmc.QMCEngine = ...) -> npt.ArrayLike: ...


class HINVDist(Protocol):
    @property
    def pdf(self) -> Callable[..., float]: ...
    @property
    def cdf(self) -> Callable[..., float]: ...
    @property
    def support(self) -> tuple[float, float]: ...


class NumericalInverseHermite(Method):
    def __init__(self,
                 dist: HINVDist,
                 *,
                 domain: None | tuple[float, float] = ...,
                 order: int= ...,
                 u_resolution: float = ...,
                 construction_points: None | npt.ArrayLike = ...,
                 max_intervals: int = ...,
                 random_state: SeedType = ...) -> None: ...
    @property
    def intervals(self) -> int: ...
    @overload
    def ppf(self, u: ArrayLike0D) -> float: ...  # type: ignore[misc]
    @overload
    def ppf(self, u: npt.ArrayLike) -> np.ndarray: ...
    def qrvs(self,
             size: None | int | tuple[int, ...] = ...,
             d: None | int = ...,
             qmc_engine: None | stats.qmc.QMCEngine = ...) -> npt.ArrayLike: ...
    def u_error(self, sample_size: int = ...) -> UError: ...


class DAUDist(Protocol):
    @property
    def pmf(self) -> Callable[..., float]: ...
    @property
    def support(self) -> tuple[float, float]: ...

class DiscreteAliasUrn(Method):
    def __init__(self,
                 dist: npt.ArrayLike | DAUDist,
                 *,
                 domain: None | tuple[float, float] = ...,
                 urn_factor: float = ...,
                 random_state: SeedType = ...) -> None: ...


class DGTDist(Protocol):
    @property
    def pmf(self) -> Callable[..., float]: ...
    @property
    def support(self) -> tuple[float, float]: ...

class DiscreteGuideTable(Method):
    def __init__(self,
                 dist: npt.ArrayLike | DGTDist,
                 *,
                 domain: None | tuple[float, float] = ...,
                 guide_factor: float = ...,
                 random_state: SeedType = ...) -> None: ...
    @overload
    def ppf(self, u: ArrayLike0D) -> float: ...  # type: ignore[misc]
    @overload
    def ppf(self, u: npt.ArrayLike) -> np.ndarray: ...
