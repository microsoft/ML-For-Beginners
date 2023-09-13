"""Simple magics for display formats"""
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
from IPython.display import display, Javascript, Latex, SVG, HTML, Markdown
from IPython.core.magic import  (
    Magics, magics_class, cell_magic
)
from IPython.core import magic_arguments

#-----------------------------------------------------------------------------
# Magic implementation classes
#-----------------------------------------------------------------------------


@magics_class
class DisplayMagics(Magics):
    """Magics for displaying various output types with literals

    Defines javascript/latex/svg/html cell magics for writing
    blocks in those languages, to be rendered in the frontend.
    """

    @cell_magic
    def js(self, line, cell):
        """Run the cell block of Javascript code

        Alias of `%%javascript`

        Starting with IPython 8.0 %%javascript is pending deprecation to be replaced
        by a more flexible system

        Please See https://github.com/ipython/ipython/issues/13376
        """
        self.javascript(line, cell)

    @cell_magic
    def javascript(self, line, cell):
        """Run the cell block of Javascript code

        Starting with IPython 8.0 %%javascript is pending deprecation to be replaced
        by a more flexible system

        Please See https://github.com/ipython/ipython/issues/13376
        """
        display(Javascript(cell))


    @cell_magic
    def latex(self, line, cell):
        """Render the cell as a block of LaTeX

        The subset of LaTeX which is supported depends on the implementation in
        the client.  In the Jupyter Notebook, this magic only renders the subset
        of LaTeX defined by MathJax
        [here](https://docs.mathjax.org/en/v2.5-latest/tex.html)."""
        display(Latex(cell))

    @cell_magic
    def svg(self, line, cell):
        """Render the cell as an SVG literal"""
        display(SVG(cell))

    @magic_arguments.magic_arguments()
    @magic_arguments.argument(
        '--isolated', action='store_true', default=False,
        help="""Annotate the cell as 'isolated'.
Isolated cells are rendered inside their own <iframe> tag"""
    )
    @cell_magic
    def html(self, line, cell):
        """Render the cell as a block of HTML"""
        args = magic_arguments.parse_argstring(self.html, line)
        html = HTML(cell)
        if args.isolated:
            display(html, metadata={'text/html':{'isolated':True}})
        else:
            display(html)

    @cell_magic
    def markdown(self, line, cell):
        """Render the cell as Markdown text block"""
        display(Markdown(cell))
