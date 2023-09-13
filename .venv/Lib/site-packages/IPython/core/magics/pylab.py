"""Implementation of magic functions for matplotlib/pylab support.
"""
#-----------------------------------------------------------------------------
#  Copyright (c) 2012 The IPython Development Team.
#
#  Distributed under the terms of the Modified BSD License.
#
#  The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

# Our own packages
from traitlets.config.application import Application
from IPython.core import magic_arguments
from IPython.core.magic import Magics, magics_class, line_magic
from IPython.testing.skipdoctest import skip_doctest
from warnings import warn
from IPython.core.pylabtools import backends

#-----------------------------------------------------------------------------
# Magic implementation classes
#-----------------------------------------------------------------------------

magic_gui_arg = magic_arguments.argument(
        'gui', nargs='?',
        help="""Name of the matplotlib backend to use %s.
        If given, the corresponding matplotlib backend is used,
        otherwise it will be matplotlib's default
        (which you can set in your matplotlib config file).
        """ % str(tuple(sorted(backends.keys())))
)


@magics_class
class PylabMagics(Magics):
    """Magics related to matplotlib's pylab support"""

    @skip_doctest
    @line_magic
    @magic_arguments.magic_arguments()
    @magic_arguments.argument('-l', '--list', action='store_true',
                              help='Show available matplotlib backends')
    @magic_gui_arg
    def matplotlib(self, line=''):
        """Set up matplotlib to work interactively.

        This function lets you activate matplotlib interactive support
        at any point during an IPython session. It does not import anything
        into the interactive namespace.

        If you are using the inline matplotlib backend in the IPython Notebook
        you can set which figure formats are enabled using the following::

            In [1]: from matplotlib_inline.backend_inline import set_matplotlib_formats

            In [2]: set_matplotlib_formats('pdf', 'svg')

        The default for inline figures sets `bbox_inches` to 'tight'. This can
        cause discrepancies between the displayed image and the identical
        image created using `savefig`. This behavior can be disabled using the
        `%config` magic::

            In [3]: %config InlineBackend.print_figure_kwargs = {'bbox_inches':None}

        In addition, see the docstrings of
        `matplotlib_inline.backend_inline.set_matplotlib_formats` and
        `matplotlib_inline.backend_inline.set_matplotlib_close` for more information on
        changing additional behaviors of the inline backend.

        Examples
        --------
        To enable the inline backend for usage with the IPython Notebook::

            In [1]: %matplotlib inline

        In this case, where the matplotlib default is TkAgg::

            In [2]: %matplotlib
            Using matplotlib backend: TkAgg

        But you can explicitly request a different GUI backend::

            In [3]: %matplotlib qt

        You can list the available backends using the -l/--list option::

           In [4]: %matplotlib --list
           Available matplotlib backends: ['osx', 'qt4', 'qt5', 'gtk3', 'gtk4', 'notebook', 'wx', 'qt', 'nbagg',
           'gtk', 'tk', 'inline']
        """
        args = magic_arguments.parse_argstring(self.matplotlib, line)
        if args.list:
            backends_list = list(backends.keys())
            print("Available matplotlib backends: %s" % backends_list)
        else:
            gui, backend = self.shell.enable_matplotlib(args.gui.lower() if isinstance(args.gui, str) else args.gui)
            self._show_matplotlib_backend(args.gui, backend)

    @skip_doctest
    @line_magic
    @magic_arguments.magic_arguments()
    @magic_arguments.argument(
        '--no-import-all', action='store_true', default=None,
        help="""Prevent IPython from performing ``import *`` into the interactive namespace.

        You can govern the default behavior of this flag with the
        InteractiveShellApp.pylab_import_all configurable.
        """
    )
    @magic_gui_arg
    def pylab(self, line=''):
        """Load numpy and matplotlib to work interactively.

        This function lets you activate pylab (matplotlib, numpy and
        interactive support) at any point during an IPython session.

        %pylab makes the following imports::

            import numpy
            import matplotlib
            from matplotlib import pylab, mlab, pyplot
            np = numpy
            plt = pyplot

            from IPython.display import display
            from IPython.core.pylabtools import figsize, getfigs

            from pylab import *
            from numpy import *

        If you pass `--no-import-all`, the last two `*` imports will be excluded.

        See the %matplotlib magic for more details about activating matplotlib
        without affecting the interactive namespace.
        """
        args = magic_arguments.parse_argstring(self.pylab, line)
        if args.no_import_all is None:
            # get default from Application
            if Application.initialized():
                app = Application.instance()
                try:
                    import_all = app.pylab_import_all
                except AttributeError:
                    import_all = True
            else:
                # nothing specified, no app - default True
                import_all = True
        else:
            # invert no-import flag
            import_all = not args.no_import_all

        gui, backend, clobbered = self.shell.enable_pylab(args.gui, import_all=import_all)
        self._show_matplotlib_backend(args.gui, backend)
        print(
            "%pylab is deprecated, use %matplotlib inline and import the required libraries."
        )
        print("Populating the interactive namespace from numpy and matplotlib")
        if clobbered:
            warn("pylab import has clobbered these variables: %s"  % clobbered +
            "\n`%matplotlib` prevents importing * from pylab and numpy"
            )

    def _show_matplotlib_backend(self, gui, backend):
        """show matplotlib message backend message"""
        if not gui or gui == 'auto':
            print("Using matplotlib backend: %s" % backend)
