'''
Static type checking stub file for scipy/spatial/qhull.pyx
'''


import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing_extensions import final

class QhullError(RuntimeError):
    ...
    
@final
class _Qhull:
    # Read-only cython attribute that behaves, more or less, like a property
    @property
    def ndim(self) -> int: ...
    mode_option: bytes
    options: bytes
    furthest_site: bool

    def __init__(
        self,
        mode_option: bytes,
        points: NDArray[np.float64],
        options: None | bytes = ...,
        required_options: None | bytes = ...,
        furthest_site: bool = ...,
        incremental: bool = ...,
        interior_point: None | NDArray[np.float64] = ...,
    ) -> None: ...
    def check_active(self) -> None: ...
    def close(self) -> None: ...
    def get_points(self) -> NDArray[np.float64]: ...
    def add_points(
        self,
        points: ArrayLike,
        interior_point: ArrayLike = ...
    ) -> None: ...
    def get_paraboloid_shift_scale(self) -> tuple[float, float]: ...
    def volume_area(self) -> tuple[float, float]: ...
    def triangulate(self) -> None: ...
    def get_simplex_facet_array(self) -> tuple[
        NDArray[np.intc],
        NDArray[np.intc],
        NDArray[np.float64],
        NDArray[np.intc],
        NDArray[np.intc],
    ]: ...
    def get_hull_points(self) -> NDArray[np.float64]: ...
    def get_hull_facets(self) -> tuple[
        list[list[int]],
        NDArray[np.float64],
    ]: ...
    def get_voronoi_diagram(self) -> tuple[
        NDArray[np.float64],
        NDArray[np.intc],
        list[list[int]],
        list[list[int]],
        NDArray[np.intp],
    ]: ...
    def get_extremes_2d(self) -> NDArray[np.intc]: ...

def _get_barycentric_transforms(
    points: NDArray[np.float64],
    simplices: NDArray[np.intc],
    eps: float
) -> NDArray[np.float64]: ...

class _QhullUser:
    ndim: int
    npoints: int
    min_bound: NDArray[np.float64]
    max_bound: NDArray[np.float64]

    def __init__(self, qhull: _Qhull, incremental: bool = ...) -> None: ...
    def close(self) -> None: ...
    def _update(self, qhull: _Qhull) -> None: ...
    def _add_points(
        self,
        points: ArrayLike,
        restart: bool = ...,
        interior_point: ArrayLike = ...
    ) -> None: ...

class Delaunay(_QhullUser):
    furthest_site: bool
    paraboloid_scale: float
    paraboloid_shift: float
    simplices: NDArray[np.intc]
    neighbors: NDArray[np.intc]
    equations: NDArray[np.float64]
    coplanar: NDArray[np.intc]
    good: NDArray[np.intc]
    nsimplex: int
    vertices: NDArray[np.intc]

    def __init__(
        self,
        points: ArrayLike,
        furthest_site: bool = ...,
        incremental: bool = ...,
        qhull_options: None | str = ...
    ) -> None: ...
    def _update(self, qhull: _Qhull) -> None: ...
    def add_points(
        self,
        points: ArrayLike,
        restart: bool = ...
    ) -> None: ...
    @property
    def points(self) -> NDArray[np.float64]: ...
    @property
    def transform(self) -> NDArray[np.float64]: ...
    @property
    def vertex_to_simplex(self) -> NDArray[np.intc]: ...
    @property
    def vertex_neighbor_vertices(self) -> tuple[
        NDArray[np.intc],
        NDArray[np.intc],
    ]: ...
    @property
    def convex_hull(self) -> NDArray[np.intc]: ...
    def find_simplex(
        self,
        xi: ArrayLike,
        bruteforce: bool = ...,
        tol: float = ...
    ) -> NDArray[np.intc]: ...
    def plane_distance(self, xi: ArrayLike) -> NDArray[np.float64]: ...
    def lift_points(self, x: ArrayLike) -> NDArray[np.float64]: ...

def tsearch(tri: Delaunay, xi: ArrayLike) -> NDArray[np.intc]: ...
def _copy_docstr(dst: object, src: object) -> None: ...

class ConvexHull(_QhullUser):
    simplices: NDArray[np.intc]
    neighbors: NDArray[np.intc]
    equations: NDArray[np.float64]
    coplanar: NDArray[np.intc]
    good: None | NDArray[np.bool_]
    volume: float
    area: float
    nsimplex: int

    def __init__(
        self,
        points: ArrayLike,
        incremental: bool = ...,
        qhull_options: None | str = ...
    ) -> None: ...
    def _update(self, qhull: _Qhull) -> None: ...
    def add_points(self, points: ArrayLike,
                   restart: bool = ...) -> None: ...
    @property
    def points(self) -> NDArray[np.float64]: ...
    @property
    def vertices(self) -> NDArray[np.intc]: ...

class Voronoi(_QhullUser):
    vertices: NDArray[np.float64]
    ridge_points: NDArray[np.intc]
    ridge_vertices: list[list[int]]
    regions: list[list[int]]
    point_region: NDArray[np.intp]
    furthest_site: bool

    def __init__(
        self,
        points: ArrayLike,
        furthest_site: bool = ...,
        incremental: bool = ...,
        qhull_options: None | str = ...
    ) -> None: ...
    def _update(self, qhull: _Qhull) -> None: ...
    def add_points(
        self,
        points: ArrayLike,
        restart: bool = ...
    ) -> None: ...
    @property
    def points(self) -> NDArray[np.float64]: ...
    @property
    def ridge_dict(self) -> dict[tuple[int, int], list[int]]: ...

class HalfspaceIntersection(_QhullUser):
    interior_point: NDArray[np.float64]
    dual_facets: list[list[int]]
    dual_equations: NDArray[np.float64]
    dual_points: NDArray[np.float64]
    dual_volume: float
    dual_area: float
    intersections: NDArray[np.float64]
    ndim: int
    nineq: int

    def __init__(
        self,
        halfspaces: ArrayLike,
        interior_point: ArrayLike,
        incremental: bool = ...,
        qhull_options: None | str = ...
    ) -> None: ...
    def _update(self, qhull: _Qhull) -> None: ...
    def add_halfspaces(
        self,
        halfspaces: ArrayLike,
        restart: bool = ...
    ) -> None: ...
    @property
    def halfspaces(self) -> NDArray[np.float64]: ...
    @property
    def dual_vertices(self) -> NDArray[np.integer]: ...
