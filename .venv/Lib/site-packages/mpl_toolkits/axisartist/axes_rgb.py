from matplotlib import _api
from mpl_toolkits.axes_grid1.axes_rgb import (  # noqa
    make_rgb_axes, RGBAxes as _RGBAxes)
from .axislines import Axes


_api.warn_deprecated(
    "3.8", name=__name__, obj_type="module", alternative="axes_grid1.axes_rgb")


@_api.deprecated("3.8", alternative=(
    "axes_grid1.axes_rgb.RGBAxes(..., axes_class=axislines.Axes"))
class RGBAxes(_RGBAxes):
    """
    Subclass of `~.axes_grid1.axes_rgb.RGBAxes` with
    ``_defaultAxesClass`` = `.axislines.Axes`.
    """
    _defaultAxesClass = Axes
