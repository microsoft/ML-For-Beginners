from matplotlib import _api

import mpl_toolkits.axes_grid1.axes_grid as axes_grid_orig
from .axislines import Axes


_api.warn_deprecated(
    "3.8", name=__name__, obj_type="module", alternative="axes_grid1.axes_grid")


@_api.deprecated("3.8", alternative=(
    "axes_grid1.axes_grid.Grid(..., axes_class=axislines.Axes"))
class Grid(axes_grid_orig.Grid):
    _defaultAxesClass = Axes


@_api.deprecated("3.8", alternative=(
    "axes_grid1.axes_grid.ImageGrid(..., axes_class=axislines.Axes"))
class ImageGrid(axes_grid_orig.ImageGrid):
    _defaultAxesClass = Axes


AxesGrid = ImageGrid
