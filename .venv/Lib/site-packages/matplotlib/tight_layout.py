from matplotlib._tight_layout import *  # noqa: F401, F403
from matplotlib import _api
_api.warn_deprecated("3.6", name=__name__, obj_type="module")


@_api.deprecated("3.6", alternative="figure.canvas.get_renderer()")
def get_renderer(fig):
    canvas = fig.canvas
    if canvas and hasattr(canvas, "get_renderer"):
        return canvas.get_renderer()
    else:
        from . import backend_bases
        return backend_bases._get_renderer(fig)
