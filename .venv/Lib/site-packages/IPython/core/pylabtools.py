# -*- coding: utf-8 -*-
"""Pylab (matplotlib) support utilities."""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.

from io import BytesIO
from binascii import b2a_base64
from functools import partial
import warnings

from IPython.core.display import _pngxy
from IPython.utils.decorators import flag_calls

# If user specifies a GUI, that dictates the backend, otherwise we read the
# user's mpl default from the mpl rc structure
backends = {
    "tk": "TkAgg",
    "gtk": "GTKAgg",
    "gtk3": "GTK3Agg",
    "gtk4": "GTK4Agg",
    "wx": "WXAgg",
    "qt4": "Qt4Agg",
    "qt5": "Qt5Agg",
    "qt6": "QtAgg",
    "qt": "QtAgg",
    "osx": "MacOSX",
    "nbagg": "nbAgg",
    "webagg": "WebAgg",
    "notebook": "nbAgg",
    "agg": "agg",
    "svg": "svg",
    "pdf": "pdf",
    "ps": "ps",
    "inline": "module://matplotlib_inline.backend_inline",
    "ipympl": "module://ipympl.backend_nbagg",
    "widget": "module://ipympl.backend_nbagg",
}

# We also need a reverse backends2guis mapping that will properly choose which
# GUI support to activate based on the desired matplotlib backend.  For the
# most part it's just a reverse of the above dict, but we also need to add a
# few others that map to the same GUI manually:
backend2gui = dict(zip(backends.values(), backends.keys()))
# In the reverse mapping, there are a few extra valid matplotlib backends that
# map to the same GUI support
backend2gui["GTK"] = backend2gui["GTKCairo"] = "gtk"
backend2gui["GTK3Cairo"] = "gtk3"
backend2gui["GTK4Cairo"] = "gtk4"
backend2gui["WX"] = "wx"
backend2gui["CocoaAgg"] = "osx"
# There needs to be a hysteresis here as the new QtAgg Matplotlib backend
# supports either Qt5 or Qt6 and the IPython qt event loop support Qt4, Qt5,
# and Qt6.
backend2gui["QtAgg"] = "qt"
backend2gui["Qt4Agg"] = "qt4"
backend2gui["Qt5Agg"] = "qt5"

# And some backends that don't need GUI integration
del backend2gui["nbAgg"]
del backend2gui["agg"]
del backend2gui["svg"]
del backend2gui["pdf"]
del backend2gui["ps"]
del backend2gui["module://matplotlib_inline.backend_inline"]
del backend2gui["module://ipympl.backend_nbagg"]

#-----------------------------------------------------------------------------
# Matplotlib utilities
#-----------------------------------------------------------------------------


def getfigs(*fig_nums):
    """Get a list of matplotlib figures by figure numbers.

    If no arguments are given, all available figures are returned.  If the
    argument list contains references to invalid figures, a warning is printed
    but the function continues pasting further figures.

    Parameters
    ----------
    figs : tuple
        A tuple of ints giving the figure numbers of the figures to return.
    """
    from matplotlib._pylab_helpers import Gcf
    if not fig_nums:
        fig_managers = Gcf.get_all_fig_managers()
        return [fm.canvas.figure for fm in fig_managers]
    else:
        figs = []
        for num in fig_nums:
            f = Gcf.figs.get(num)
            if f is None:
                print('Warning: figure %s not available.' % num)
            else:
                figs.append(f.canvas.figure)
        return figs


def figsize(sizex, sizey):
    """Set the default figure size to be [sizex, sizey].

    This is just an easy to remember, convenience wrapper that sets::

      matplotlib.rcParams['figure.figsize'] = [sizex, sizey]
    """
    import matplotlib
    matplotlib.rcParams['figure.figsize'] = [sizex, sizey]


def print_figure(fig, fmt="png", bbox_inches="tight", base64=False, **kwargs):
    """Print a figure to an image, and return the resulting file data

    Returned data will be bytes unless ``fmt='svg'``,
    in which case it will be unicode.

    Any keyword args are passed to fig.canvas.print_figure,
    such as ``quality`` or ``bbox_inches``.

    If `base64` is True, return base64-encoded str instead of raw bytes
    for binary-encoded image formats

    .. versionadded:: 7.29
        base64 argument
    """
    # When there's an empty figure, we shouldn't return anything, otherwise we
    # get big blank areas in the qt console.
    if not fig.axes and not fig.lines:
        return

    dpi = fig.dpi
    if fmt == 'retina':
        dpi = dpi * 2
        fmt = 'png'

    # build keyword args
    kw = {
        "format":fmt,
        "facecolor":fig.get_facecolor(),
        "edgecolor":fig.get_edgecolor(),
        "dpi":dpi,
        "bbox_inches":bbox_inches,
    }
    # **kwargs get higher priority
    kw.update(kwargs)

    bytes_io = BytesIO()
    if fig.canvas is None:
        from matplotlib.backend_bases import FigureCanvasBase
        FigureCanvasBase(fig)

    fig.canvas.print_figure(bytes_io, **kw)
    data = bytes_io.getvalue()
    if fmt == 'svg':
        data = data.decode('utf-8')
    elif base64:
        data = b2a_base64(data, newline=False).decode("ascii")
    return data

def retina_figure(fig, base64=False, **kwargs):
    """format a figure as a pixel-doubled (retina) PNG

    If `base64` is True, return base64-encoded str instead of raw bytes
    for binary-encoded image formats

    .. versionadded:: 7.29
        base64 argument
    """
    pngdata = print_figure(fig, fmt="retina", base64=False, **kwargs)
    # Make sure that retina_figure acts just like print_figure and returns
    # None when the figure is empty.
    if pngdata is None:
        return
    w, h = _pngxy(pngdata)
    metadata = {"width": w//2, "height":h//2}
    if base64:
        pngdata = b2a_base64(pngdata, newline=False).decode("ascii")
    return pngdata, metadata


# We need a little factory function here to create the closure where
# safe_execfile can live.
def mpl_runner(safe_execfile):
    """Factory to return a matplotlib-enabled runner for %run.

    Parameters
    ----------
    safe_execfile : function
        This must be a function with the same interface as the
        :meth:`safe_execfile` method of IPython.

    Returns
    -------
    A function suitable for use as the ``runner`` argument of the %run magic
    function.
    """

    def mpl_execfile(fname,*where,**kw):
        """matplotlib-aware wrapper around safe_execfile.

        Its interface is identical to that of the :func:`execfile` builtin.

        This is ultimately a call to execfile(), but wrapped in safeties to
        properly handle interactive rendering."""

        import matplotlib
        import matplotlib.pyplot as plt

        #print '*** Matplotlib runner ***' # dbg
        # turn off rendering until end of script
        with matplotlib.rc_context({"interactive": False}):
            safe_execfile(fname, *where, **kw)

        if matplotlib.is_interactive():
            plt.show()

        # make rendering call now, if the user tried to do it
        if plt.draw_if_interactive.called:
            plt.draw()
            plt.draw_if_interactive.called = False

        # re-draw everything that is stale
        try:
            da = plt.draw_all
        except AttributeError:
            pass
        else:
            da()

    return mpl_execfile


def _reshow_nbagg_figure(fig):
    """reshow an nbagg figure"""
    try:
        reshow = fig.canvas.manager.reshow
    except AttributeError as e:
        raise NotImplementedError() from e
    else:
        reshow()


def select_figure_formats(shell, formats, **kwargs):
    """Select figure formats for the inline backend.

    Parameters
    ----------
    shell : InteractiveShell
        The main IPython instance.
    formats : str or set
        One or a set of figure formats to enable: 'png', 'retina', 'jpeg', 'svg', 'pdf'.
    **kwargs : any
        Extra keyword arguments to be passed to fig.canvas.print_figure.
    """
    import matplotlib
    from matplotlib.figure import Figure

    svg_formatter = shell.display_formatter.formatters['image/svg+xml']
    png_formatter = shell.display_formatter.formatters['image/png']
    jpg_formatter = shell.display_formatter.formatters['image/jpeg']
    pdf_formatter = shell.display_formatter.formatters['application/pdf']

    if isinstance(formats, str):
        formats = {formats}
    # cast in case of list / tuple
    formats = set(formats)

    [ f.pop(Figure, None) for f in shell.display_formatter.formatters.values() ]
    mplbackend = matplotlib.get_backend().lower()
    if mplbackend == 'nbagg' or mplbackend == 'module://ipympl.backend_nbagg':
        formatter = shell.display_formatter.ipython_display_formatter
        formatter.for_type(Figure, _reshow_nbagg_figure)

    supported = {'png', 'png2x', 'retina', 'jpg', 'jpeg', 'svg', 'pdf'}
    bad = formats.difference(supported)
    if bad:
        bs = "%s" % ','.join([repr(f) for f in bad])
        gs = "%s" % ','.join([repr(f) for f in supported])
        raise ValueError("supported formats are: %s not %s" % (gs, bs))

    if "png" in formats:
        png_formatter.for_type(
            Figure, partial(print_figure, fmt="png", base64=True, **kwargs)
        )
    if "retina" in formats or "png2x" in formats:
        png_formatter.for_type(Figure, partial(retina_figure, base64=True, **kwargs))
    if "jpg" in formats or "jpeg" in formats:
        jpg_formatter.for_type(
            Figure, partial(print_figure, fmt="jpg", base64=True, **kwargs)
        )
    if "svg" in formats:
        svg_formatter.for_type(Figure, partial(print_figure, fmt="svg", **kwargs))
    if "pdf" in formats:
        pdf_formatter.for_type(
            Figure, partial(print_figure, fmt="pdf", base64=True, **kwargs)
        )

#-----------------------------------------------------------------------------
# Code for initializing matplotlib and importing pylab
#-----------------------------------------------------------------------------


def find_gui_and_backend(gui=None, gui_select=None):
    """Given a gui string return the gui and mpl backend.

    Parameters
    ----------
    gui : str
        Can be one of ('tk','gtk','wx','qt','qt4','inline','agg').
    gui_select : str
        Can be one of ('tk','gtk','wx','qt','qt4','inline').
        This is any gui already selected by the shell.

    Returns
    -------
    A tuple of (gui, backend) where backend is one of ('TkAgg','GTKAgg',
    'WXAgg','Qt4Agg','module://matplotlib_inline.backend_inline','agg').
    """

    import matplotlib

    has_unified_qt_backend = getattr(matplotlib, "__version_info__", (0, 0)) >= (3, 5)

    backends_ = dict(backends)
    if not has_unified_qt_backend:
        backends_["qt"] = "qt5agg"

    if gui and gui != 'auto':
        # select backend based on requested gui
        backend = backends_[gui]
        if gui == 'agg':
            gui = None
    else:
        # We need to read the backend from the original data structure, *not*
        # from mpl.rcParams, since a prior invocation of %matplotlib may have
        # overwritten that.
        # WARNING: this assumes matplotlib 1.1 or newer!!
        backend = matplotlib.rcParamsOrig['backend']
        # In this case, we need to find what the appropriate gui selection call
        # should be for IPython, so we can activate inputhook accordingly
        gui = backend2gui.get(backend, None)

        # If we have already had a gui active, we need it and inline are the
        # ones allowed.
        if gui_select and gui != gui_select:
            gui = gui_select
            backend = backends_[gui]

    return gui, backend


def activate_matplotlib(backend):
    """Activate the given backend and set interactive to True."""

    import matplotlib
    matplotlib.interactive(True)

    # Matplotlib had a bug where even switch_backend could not force
    # the rcParam to update. This needs to be set *before* the module
    # magic of switch_backend().
    matplotlib.rcParams['backend'] = backend

    # Due to circular imports, pyplot may be only partially initialised
    # when this function runs.
    # So avoid needing matplotlib attribute-lookup to access pyplot.
    from matplotlib import pyplot as plt

    plt.switch_backend(backend)

    plt.show._needmain = False
    # We need to detect at runtime whether show() is called by the user.
    # For this, we wrap it into a decorator which adds a 'called' flag.
    plt.draw_if_interactive = flag_calls(plt.draw_if_interactive)


def import_pylab(user_ns, import_all=True):
    """Populate the namespace with pylab-related values.

    Imports matplotlib, pylab, numpy, and everything from pylab and numpy.

    Also imports a few names from IPython (figsize, display, getfigs)

    """

    # Import numpy as np/pyplot as plt are conventions we're trying to
    # somewhat standardize on.  Making them available to users by default
    # will greatly help this.
    s = ("import numpy\n"
          "import matplotlib\n"
          "from matplotlib import pylab, mlab, pyplot\n"
          "np = numpy\n"
          "plt = pyplot\n"
          )
    exec(s, user_ns)

    if import_all:
        s = ("from matplotlib.pylab import *\n"
             "from numpy import *\n")
        exec(s, user_ns)

    # IPython symbols to add
    user_ns['figsize'] = figsize
    from IPython.display import display
    # Add display and getfigs to the user's namespace
    user_ns['display'] = display
    user_ns['getfigs'] = getfigs


def configure_inline_support(shell, backend):
    """
    .. deprecated:: 7.23

        use `matplotlib_inline.backend_inline.configure_inline_support()`

    Configure an IPython shell object for matplotlib use.

    Parameters
    ----------
    shell : InteractiveShell instance
    backend : matplotlib backend
    """
    warnings.warn(
        "`configure_inline_support` is deprecated since IPython 7.23, directly "
        "use `matplotlib_inline.backend_inline.configure_inline_support()`",
        DeprecationWarning,
        stacklevel=2,
    )

    from matplotlib_inline.backend_inline import (
        configure_inline_support as configure_inline_support_orig,
    )

    configure_inline_support_orig(shell, backend)
