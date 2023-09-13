from ._triplot import *  # noqa: F401, F403
from matplotlib import _api


_api.warn_deprecated(
    "3.7",
    message=f"Importing {__name__} was deprecated in Matplotlib 3.7 and will "
            f"be removed two minor releases later. All functionality is "
            f"available via the top-level module matplotlib.tri")
