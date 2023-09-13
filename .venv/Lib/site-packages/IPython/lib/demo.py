"""Module for interactive demos using IPython.

This module implements a few classes for running Python scripts interactively
in IPython for demonstrations.  With very simple markup (a few tags in
comments), you can control points where the script stops executing and returns
control to IPython.


Provided classes
----------------

The classes are (see their docstrings for further details):

 - Demo: pure python demos

 - IPythonDemo: demos with input to be processed by IPython as if it had been
   typed interactively (so magics work, as well as any other special syntax you
   may have added via input prefilters).

 - LineDemo: single-line version of the Demo class.  These demos are executed
   one line at a time, and require no markup.

 - IPythonLineDemo: IPython version of the LineDemo class (the demo is
   executed a line at a time, but processed via IPython).

 - ClearMixin: mixin to make Demo classes with less visual clutter.  It
   declares an empty marquee and a pre_cmd that clears the screen before each
   block (see Subclassing below).

 - ClearDemo, ClearIPDemo: mixin-enabled versions of the Demo and IPythonDemo
   classes.

Inheritance diagram:

.. inheritance-diagram:: IPython.lib.demo
   :parts: 3

Subclassing
-----------

The classes here all include a few methods meant to make customization by
subclassing more convenient.  Their docstrings below have some more details:

  - highlight(): format every block and optionally highlight comments and
    docstring content.

  - marquee(): generates a marquee to provide visible on-screen markers at each
    block start and end.

  - pre_cmd(): run right before the execution of each block.

  - post_cmd(): run right after the execution of each block.  If the block
    raises an exception, this is NOT called.


Operation
---------

The file is run in its own empty namespace (though you can pass it a string of
arguments as if in a command line environment, and it will see those as
sys.argv).  But at each stop, the global IPython namespace is updated with the
current internal demo namespace, so you can work interactively with the data
accumulated so far.

By default, each block of code is printed (with syntax highlighting) before
executing it and you have to confirm execution.  This is intended to show the
code to an audience first so you can discuss it, and only proceed with
execution once you agree.  There are a few tags which allow you to modify this
behavior.

The supported tags are:

# <demo> stop

  Defines block boundaries, the points where IPython stops execution of the
  file and returns to the interactive prompt.

  You can optionally mark the stop tag with extra dashes before and after the
  word 'stop', to help visually distinguish the blocks in a text editor:

  # <demo> --- stop ---


# <demo> silent

  Make a block execute silently (and hence automatically).  Typically used in
  cases where you have some boilerplate or initialization code which you need
  executed but do not want to be seen in the demo.

# <demo> auto

  Make a block execute automatically, but still being printed.  Useful for
  simple code which does not warrant discussion, since it avoids the extra
  manual confirmation.

# <demo> auto_all

  This tag can _only_ be in the first block, and if given it overrides the
  individual auto tags to make the whole demo fully automatic (no block asks
  for confirmation).  It can also be given at creation time (or the attribute
  set later) to override what's in the file.

While _any_ python file can be run as a Demo instance, if there are no stop
tags the whole file will run in a single block (no different that calling
first %pycat and then %run).  The minimal markup to make this useful is to
place a set of stop tags; the other tags are only there to let you fine-tune
the execution.

This is probably best explained with the simple example file below.  You can
copy this into a file named ex_demo.py, and try running it via::

    from IPython.lib.demo import Demo
    d = Demo('ex_demo.py')
    d()

Each time you call the demo object, it runs the next block.  The demo object
has a few useful methods for navigation, like again(), edit(), jump(), seek()
and back().  It can be reset for a new run via reset() or reloaded from disk
(in case you've edited the source) via reload().  See their docstrings below.

Note: To make this simpler to explore, a file called "demo-exercizer.py" has
been added to the "docs/examples/core" directory.  Just cd to this directory in
an IPython session, and type::

  %run demo-exercizer.py

and then follow the directions.

Example
-------

The following is a very simple example of a valid demo file.

::

    #################### EXAMPLE DEMO <ex_demo.py> ###############################
    '''A simple interactive demo to illustrate the use of IPython's Demo class.'''

    print('Hello, welcome to an interactive IPython demo.')

    # The mark below defines a block boundary, which is a point where IPython will
    # stop execution and return to the interactive prompt. The dashes are actually
    # optional and used only as a visual aid to clearly separate blocks while
    # editing the demo code.
    # <demo> stop

    x = 1
    y = 2

    # <demo> stop

    # the mark below makes this block as silent
    # <demo> silent

    print('This is a silent block, which gets executed but not printed.')

    # <demo> stop
    # <demo> auto
    print('This is an automatic block.')
    print('It is executed without asking for confirmation, but printed.')
    z = x + y

    print('z =', x)

    # <demo> stop
    # This is just another normal block.
    print('z is now:', z)

    print('bye!')
    ################### END EXAMPLE DEMO <ex_demo.py> ############################
"""


#*****************************************************************************
#     Copyright (C) 2005-2006 Fernando Perez. <Fernando.Perez@colorado.edu>
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
#
#*****************************************************************************

import os
import re
import shlex
import sys
import pygments
from pathlib import Path

from IPython.utils.text import marquee
from IPython.utils import openpy
from IPython.utils import py3compat
__all__ = ['Demo','IPythonDemo','LineDemo','IPythonLineDemo','DemoError']

class DemoError(Exception): pass

def re_mark(mark):
    return re.compile(r'^\s*#\s+<demo>\s+%s\s*$' % mark,re.MULTILINE)

class Demo(object):

    re_stop     = re_mark(r'-*\s?stop\s?-*')
    re_silent   = re_mark('silent')
    re_auto     = re_mark('auto')
    re_auto_all = re_mark('auto_all')

    def __init__(self,src,title='',arg_str='',auto_all=None, format_rst=False,
                 formatter='terminal', style='default'):
        """Make a new demo object.  To run the demo, simply call the object.

        See the module docstring for full details and an example (you can use
        IPython.Demo? in IPython to see it).

        Inputs:

          - src is either a file, or file-like object, or a
              string that can be resolved to a filename.

        Optional inputs:

          - title: a string to use as the demo name.  Of most use when the demo
            you are making comes from an object that has no filename, or if you
            want an alternate denotation distinct from the filename.

          - arg_str(''): a string of arguments, internally converted to a list
            just like sys.argv, so the demo script can see a similar
            environment.

          - auto_all(None): global flag to run all blocks automatically without
            confirmation.  This attribute overrides the block-level tags and
            applies to the whole demo.  It is an attribute of the object, and
            can be changed at runtime simply by reassigning it to a boolean
            value.

          - format_rst(False): a bool to enable comments and doc strings
            formatting with pygments rst lexer

          - formatter('terminal'): a string of pygments formatter name to be
            used. Useful values for terminals: terminal, terminal256,
            terminal16m

          - style('default'): a string of pygments style name to be used.
        """
        if hasattr(src, "read"):
             # It seems to be a file or a file-like object
            self.fname = "from a file-like object"
            if title == '':
                self.title = "from a file-like object"
            else:
                self.title = title
        else:
             # Assume it's a string or something that can be converted to one
            self.fname = src
            if title == '':
                (filepath, filename) = os.path.split(src)
                self.title = filename
            else:
                self.title = title
        self.sys_argv = [src] + shlex.split(arg_str)
        self.auto_all = auto_all
        self.src = src

        try:
            ip = get_ipython()  # this is in builtins whenever IPython is running
            self.inside_ipython = True
        except NameError:
            self.inside_ipython = False

        if self.inside_ipython:
            # get a few things from ipython.  While it's a bit ugly design-wise,
            # it ensures that things like color scheme and the like are always in
            # sync with the ipython mode being used.  This class is only meant to
            # be used inside ipython anyways,  so it's OK.
            self.ip_ns       = ip.user_ns
            self.ip_colorize = ip.pycolorize
            self.ip_showtb   = ip.showtraceback
            self.ip_run_cell = ip.run_cell
            self.shell       = ip

        self.formatter = pygments.formatters.get_formatter_by_name(formatter,
                                                                   style=style)
        self.python_lexer = pygments.lexers.get_lexer_by_name("py3")
        self.format_rst = format_rst
        if format_rst:
            self.rst_lexer = pygments.lexers.get_lexer_by_name("rst")

        # load user data and initialize data structures
        self.reload()

    def fload(self):
        """Load file object."""
        # read data and parse into blocks
        if hasattr(self, 'fobj') and self.fobj is not None:
           self.fobj.close()
        if hasattr(self.src, "read"):
             # It seems to be a file or a file-like object
            self.fobj = self.src
        else:
             # Assume it's a string or something that can be converted to one
            self.fobj = openpy.open(self.fname)

    def reload(self):
        """Reload source from disk and initialize state."""
        self.fload()

        self.src     = "".join(openpy.strip_encoding_cookie(self.fobj))
        src_b        = [b.strip() for b in self.re_stop.split(self.src) if b]
        self._silent = [bool(self.re_silent.findall(b)) for b in src_b]
        self._auto   = [bool(self.re_auto.findall(b)) for b in src_b]

        # if auto_all is not given (def. None), we read it from the file
        if self.auto_all is None:
            self.auto_all = bool(self.re_auto_all.findall(src_b[0]))
        else:
            self.auto_all = bool(self.auto_all)

        # Clean the sources from all markup so it doesn't get displayed when
        # running the demo
        src_blocks = []
        auto_strip = lambda s: self.re_auto.sub('',s)
        for i,b in enumerate(src_b):
            if self._auto[i]:
                src_blocks.append(auto_strip(b))
            else:
                src_blocks.append(b)
        # remove the auto_all marker
        src_blocks[0] = self.re_auto_all.sub('',src_blocks[0])

        self.nblocks = len(src_blocks)
        self.src_blocks = src_blocks

        # also build syntax-highlighted source
        self.src_blocks_colored = list(map(self.highlight,self.src_blocks))

        # ensure clean namespace and seek offset
        self.reset()

    def reset(self):
        """Reset the namespace and seek pointer to restart the demo"""
        self.user_ns     = {}
        self.finished    = False
        self.block_index = 0

    def _validate_index(self,index):
        if index<0 or index>=self.nblocks:
            raise ValueError('invalid block index %s' % index)

    def _get_index(self,index):
        """Get the current block index, validating and checking status.

        Returns None if the demo is finished"""

        if index is None:
            if self.finished:
                print('Demo finished.  Use <demo_name>.reset() if you want to rerun it.')
                return None
            index = self.block_index
        else:
            self._validate_index(index)
        return index

    def seek(self,index):
        """Move the current seek pointer to the given block.

        You can use negative indices to seek from the end, with identical
        semantics to those of Python lists."""
        if index<0:
            index = self.nblocks + index
        self._validate_index(index)
        self.block_index = index
        self.finished = False

    def back(self,num=1):
        """Move the seek pointer back num blocks (default is 1)."""
        self.seek(self.block_index-num)

    def jump(self,num=1):
        """Jump a given number of blocks relative to the current one.

        The offset can be positive or negative, defaults to 1."""
        self.seek(self.block_index+num)

    def again(self):
        """Move the seek pointer back one block and re-execute."""
        self.back(1)
        self()

    def edit(self,index=None):
        """Edit a block.

        If no number is given, use the last block executed.

        This edits the in-memory copy of the demo, it does NOT modify the
        original source file.  If you want to do that, simply open the file in
        an editor and use reload() when you make changes to the file.  This
        method is meant to let you change a block during a demonstration for
        explanatory purposes, without damaging your original script."""

        index = self._get_index(index)
        if index is None:
            return
        # decrease the index by one (unless we're at the very beginning), so
        # that the default demo.edit() call opens up the sblock we've last run
        if index>0:
            index -= 1

        filename = self.shell.mktempfile(self.src_blocks[index])
        self.shell.hooks.editor(filename, 1)
        with open(Path(filename), "r", encoding="utf-8") as f:
            new_block = f.read()
        # update the source and colored block
        self.src_blocks[index] = new_block
        self.src_blocks_colored[index] = self.highlight(new_block)
        self.block_index = index
        # call to run with the newly edited index
        self()

    def show(self,index=None):
        """Show a single block on screen"""

        index = self._get_index(index)
        if index is None:
            return

        print(self.marquee('<%s> block # %s (%s remaining)' %
                           (self.title,index,self.nblocks-index-1)))
        print(self.src_blocks_colored[index])
        sys.stdout.flush()

    def show_all(self):
        """Show entire demo on screen, block by block"""

        fname = self.title
        title = self.title
        nblocks = self.nblocks
        silent = self._silent
        marquee = self.marquee
        for index,block in enumerate(self.src_blocks_colored):
            if silent[index]:
                print(marquee('<%s> SILENT block # %s (%s remaining)' %
                              (title,index,nblocks-index-1)))
            else:
                print(marquee('<%s> block # %s (%s remaining)' %
                              (title,index,nblocks-index-1)))
            print(block, end=' ')
        sys.stdout.flush()

    def run_cell(self,source):
        """Execute a string with one or more lines of code"""

        exec(source, self.user_ns)

    def __call__(self,index=None):
        """run a block of the demo.

        If index is given, it should be an integer >=1 and <= nblocks.  This
        means that the calling convention is one off from typical Python
        lists.  The reason for the inconsistency is that the demo always
        prints 'Block n/N, and N is the total, so it would be very odd to use
        zero-indexing here."""

        index = self._get_index(index)
        if index is None:
            return
        try:
            marquee = self.marquee
            next_block = self.src_blocks[index]
            self.block_index += 1
            if self._silent[index]:
                print(marquee('Executing silent block # %s (%s remaining)' %
                              (index,self.nblocks-index-1)))
            else:
                self.pre_cmd()
                self.show(index)
                if self.auto_all or self._auto[index]:
                    print(marquee('output:'))
                else:
                    print(marquee('Press <q> to quit, <Enter> to execute...'), end=' ')
                    ans = py3compat.input().strip()
                    if ans:
                        print(marquee('Block NOT executed'))
                        return
            try:
                save_argv = sys.argv
                sys.argv = self.sys_argv
                self.run_cell(next_block)
                self.post_cmd()
            finally:
                sys.argv = save_argv

        except:
            if self.inside_ipython:
                self.ip_showtb(filename=self.fname)
        else:
            if self.inside_ipython:
                self.ip_ns.update(self.user_ns)

        if self.block_index == self.nblocks:
            mq1 = self.marquee('END OF DEMO')
            if mq1:
                # avoid spurious print if empty marquees are used
                print()
                print(mq1)
                print(self.marquee('Use <demo_name>.reset() if you want to rerun it.'))
            self.finished = True

    # These methods are meant to be overridden by subclasses who may wish to
    # customize the behavior of of their demos.
    def marquee(self,txt='',width=78,mark='*'):
        """Return the input string centered in a 'marquee'."""
        return marquee(txt,width,mark)

    def pre_cmd(self):
        """Method called before executing each block."""
        pass

    def post_cmd(self):
        """Method called after executing each block."""
        pass

    def highlight(self, block):
        """Method called on each block to highlight it content"""
        tokens = pygments.lex(block, self.python_lexer)
        if self.format_rst:
            from pygments.token import Token
            toks = []
            for token in tokens:
                if token[0] == Token.String.Doc and len(token[1]) > 6:
                    toks += pygments.lex(token[1][:3], self.python_lexer)
                    # parse doc string content by rst lexer
                    toks += pygments.lex(token[1][3:-3], self.rst_lexer)
                    toks += pygments.lex(token[1][-3:], self.python_lexer)
                elif token[0] == Token.Comment.Single:
                    toks.append((Token.Comment.Single, token[1][0]))
                    # parse comment content by rst lexer
                    # remove the extra newline added by rst lexer
                    toks += list(pygments.lex(token[1][1:], self.rst_lexer))[:-1]
                else:
                    toks.append(token)
            tokens = toks
        return pygments.format(tokens, self.formatter)


class IPythonDemo(Demo):
    """Class for interactive demos with IPython's input processing applied.

    This subclasses Demo, but instead of executing each block by the Python
    interpreter (via exec), it actually calls IPython on it, so that any input
    filters which may be in place are applied to the input block.

    If you have an interactive environment which exposes special input
    processing, you can use this class instead to write demo scripts which
    operate exactly as if you had typed them interactively.  The default Demo
    class requires the input to be valid, pure Python code.
    """

    def run_cell(self,source):
        """Execute a string with one or more lines of code"""

        self.shell.run_cell(source)

class LineDemo(Demo):
    """Demo where each line is executed as a separate block.

    The input script should be valid Python code.

    This class doesn't require any markup at all, and it's meant for simple
    scripts (with no nesting or any kind of indentation) which consist of
    multiple lines of input to be executed, one at a time, as if they had been
    typed in the interactive prompt.

    Note: the input can not have *any* indentation, which means that only
    single-lines of input are accepted, not even function definitions are
    valid."""

    def reload(self):
        """Reload source from disk and initialize state."""
        # read data and parse into blocks
        self.fload()
        lines           = self.fobj.readlines()
        src_b           = [l for l in lines if l.strip()]
        nblocks         = len(src_b)
        self.src        = ''.join(lines)
        self._silent    = [False]*nblocks
        self._auto      = [True]*nblocks
        self.auto_all   = True
        self.nblocks    = nblocks
        self.src_blocks = src_b

        # also build syntax-highlighted source
        self.src_blocks_colored = list(map(self.highlight,self.src_blocks))

        # ensure clean namespace and seek offset
        self.reset()


class IPythonLineDemo(IPythonDemo,LineDemo):
    """Variant of the LineDemo class whose input is processed by IPython."""
    pass


class ClearMixin(object):
    """Use this mixin to make Demo classes with less visual clutter.

    Demos using this mixin will clear the screen before every block and use
    blank marquees.

    Note that in order for the methods defined here to actually override those
    of the classes it's mixed with, it must go /first/ in the inheritance
    tree.  For example:

        class ClearIPDemo(ClearMixin,IPythonDemo): pass

    will provide an IPythonDemo class with the mixin's features.
    """

    def marquee(self,txt='',width=78,mark='*'):
        """Blank marquee that returns '' no matter what the input."""
        return ''

    def pre_cmd(self):
        """Method called before executing each block.

        This one simply clears the screen."""
        from IPython.utils.terminal import _term_clear
        _term_clear()

class ClearDemo(ClearMixin,Demo):
    pass


class ClearIPDemo(ClearMixin,IPythonDemo):
    pass


def slide(file_path, noclear=False, format_rst=True, formatter="terminal",
          style="native", auto_all=False, delimiter='...'):
    if noclear:
        demo_class = Demo
    else:
        demo_class = ClearDemo
    demo = demo_class(file_path, format_rst=format_rst, formatter=formatter,
                      style=style, auto_all=auto_all)
    while not demo.finished:
        demo()
        try:
            py3compat.input('\n' + delimiter)
        except KeyboardInterrupt:
            exit(1)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run python demos')
    parser.add_argument('--noclear', '-C', action='store_true',
                        help='Do not clear terminal on each slide')
    parser.add_argument('--rst', '-r', action='store_true',
                        help='Highlight comments and dostrings as rst')
    parser.add_argument('--formatter', '-f', default='terminal',
                        help='pygments formatter name could be: terminal, '
                        'terminal256, terminal16m')
    parser.add_argument('--style', '-s', default='default',
                        help='pygments style name')
    parser.add_argument('--auto', '-a', action='store_true',
                        help='Run all blocks automatically without'
                        'confirmation')
    parser.add_argument('--delimiter', '-d', default='...',
                        help='slides delimiter added after each slide run')
    parser.add_argument('file', nargs=1,
                        help='python demo file')
    args = parser.parse_args()
    slide(args.file[0], noclear=args.noclear, format_rst=args.rst,
          formatter=args.formatter, style=args.style, auto_all=args.auto,
          delimiter=args.delimiter)
