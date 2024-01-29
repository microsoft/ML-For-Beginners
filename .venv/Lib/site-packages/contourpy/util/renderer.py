from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import io

    from numpy.typing import ArrayLike

    from contourpy._contourpy import CoordinateArray, FillReturn, FillType, LineReturn, LineType


class Renderer(ABC):
    """Abstract base class for renderers, defining the interface that they must implement."""

    def _grid_as_2d(self, x: ArrayLike, y: ArrayLike) -> tuple[CoordinateArray, CoordinateArray]:
        x = np.asarray(x)
        y = np.asarray(y)
        if x.ndim == 1:
            x, y = np.meshgrid(x, y)
        return x, y

        x = np.asarray(x)
        y = np.asarray(y)
        if x.ndim == 1:
            x, y = np.meshgrid(x, y)
        return x, y

    @abstractmethod
    def filled(
        self,
        filled: FillReturn,
        fill_type: FillType | str,
        ax: Any = 0,
        color: str = "C0",
        alpha: float = 0.7,
    ) -> None:
        pass

    @abstractmethod
    def grid(
        self,
        x: ArrayLike,
        y: ArrayLike,
        ax: Any = 0,
        color: str = "black",
        alpha: float = 0.1,
        point_color: str | None = None,
        quad_as_tri_alpha: float = 0,
    ) -> None:
        pass

    @abstractmethod
    def lines(
        self,
        lines: LineReturn,
        line_type: LineType | str,
        ax: Any = 0,
        color: str = "C0",
        alpha: float = 1.0,
        linewidth: float = 1,
    ) -> None:
        pass

    @abstractmethod
    def mask(
        self,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike | np.ma.MaskedArray[Any, Any],
        ax: Any = 0,
        color: str = "black",
    ) -> None:
        pass

    @abstractmethod
    def save(self, filename: str, transparent: bool = False) -> None:
        pass

    @abstractmethod
    def save_to_buffer(self) -> io.BytesIO:
        pass

    @abstractmethod
    def show(self) -> None:
        pass

    @abstractmethod
    def title(self, title: str, ax: Any = 0, color: str | None = None) -> None:
        pass

    @abstractmethod
    def z_values(
        self,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike,
        ax: Any = 0,
        color: str = "green",
        fmt: str = ".1f",
        quad_as_tri: bool = False,
    ) -> None:
        pass
