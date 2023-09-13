from mpl_toolkits.axes_grid1.axes_rgb import (  # noqa
    make_rgb_axes, RGBAxes as _RGBAxes)
from .axislines import Axes


class RGBAxes(_RGBAxes):
    """
    Subclass of `~.axes_grid1.axes_rgb.RGBAxes` with
    ``_defaultAxesClass`` = `.axislines.Axes`.
    """
    _defaultAxesClass = Axes
