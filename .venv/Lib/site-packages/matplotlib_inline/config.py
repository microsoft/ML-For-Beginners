"""Configurable for configuring the IPython inline backend

This module does not import anything from matplotlib.
"""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the BSD 3-Clause License.

from traitlets.config.configurable import SingletonConfigurable
from traitlets import (
    Dict, Instance, Set, Bool, TraitError, Unicode
)


# Configurable for inline backend options
def pil_available():
    """Test if PIL/Pillow is available"""
    out = False
    try:
        from PIL import Image  # noqa
        out = True
    except ImportError:
        pass
    return out


# Inherit from InlineBackendConfig for deprecation purposes
class InlineBackendConfig(SingletonConfigurable):
    pass


class InlineBackend(InlineBackendConfig):
    """An object to store configuration of the inline backend."""

    # While we are deprecating overriding matplotlib defaults out of the
    # box, this structure should remain here (empty) for API compatibility
    # and the use of other tools that may need it. Specifically Spyder takes
    # advantage of it.
    # See https://github.com/ipython/ipython/issues/10383 for details.
    rc = Dict(
        {},
        help="""Dict to manage matplotlib configuration defaults in the inline
        backend. As of v0.1.4 IPython/Jupyter do not override defaults out of
        the box, but third-party tools may use it to manage rc data. To change
        personal defaults for matplotlib,  use matplotlib's configuration
        tools, or customize this class in your `ipython_config.py` file for
        IPython/Jupyter-specific usage.""").tag(config=True)

    figure_formats = Set(
        {'png'},
        help="""A set of figure formats to enable: 'png',
                'retina', 'jpeg', 'svg', 'pdf'.""").tag(config=True)

    def _update_figure_formatters(self):
        if self.shell is not None:
            from IPython.core.pylabtools import select_figure_formats
            select_figure_formats(self.shell, self.figure_formats, **self.print_figure_kwargs)

    def _figure_formats_changed(self, name, old, new):
        if 'jpg' in new or 'jpeg' in new:
            if not pil_available():
                raise TraitError("Requires PIL/Pillow for JPG figures")
        self._update_figure_formatters()

    figure_format = Unicode(help="""The figure format to enable (deprecated
                                         use `figure_formats` instead)""").tag(config=True)

    def _figure_format_changed(self, name, old, new):
        if new:
            self.figure_formats = {new}

    print_figure_kwargs = Dict(
        {'bbox_inches': 'tight'},
        help="""Extra kwargs to be passed to fig.canvas.print_figure.

        Logical examples include: bbox_inches, quality (for jpeg figures), etc.
        """
    ).tag(config=True)
    _print_figure_kwargs_changed = _update_figure_formatters

    close_figures = Bool(
        True,
        help="""Close all figures at the end of each cell.

        When True, ensures that each cell starts with no active figures, but it
        also means that one must keep track of references in order to edit or
        redraw figures in subsequent cells. This mode is ideal for the notebook,
        where residual plots from other cells might be surprising.

        When False, one must call figure() to create new figures. This means
        that gcf() and getfigs() can reference figures created in other cells,
        and the active figure can continue to be edited with pylab/pyplot
        methods that reference the current active figure. This mode facilitates
        iterative editing of figures, and behaves most consistently with
        other matplotlib backends, but figure barriers between cells must
        be explicit.
        """).tag(config=True)

    shell = Instance('IPython.core.interactiveshell.InteractiveShellABC',
                     allow_none=True)
