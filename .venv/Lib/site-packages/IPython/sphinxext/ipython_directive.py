# -*- coding: utf-8 -*-
"""
Sphinx directive to support embedded IPython code.

IPython provides an extension for `Sphinx <http://www.sphinx-doc.org/>`_ to
highlight and run code.

This directive allows pasting of entire interactive IPython sessions, prompts
and all, and their code will actually get re-executed at doc build time, with
all prompts renumbered sequentially. It also allows you to input code as a pure
python input by giving the argument python to the directive. The output looks
like an interactive ipython section.

Here is an example of how the IPython directive can
**run** python code, at build time.

.. ipython::

   In [1]: 1+1

   In [1]: import datetime
      ...: datetime.date.fromisoformat('2022-02-22')

It supports IPython construct that plain
Python does not understand (like magics):

.. ipython::

   In [0]: import time

   In [0]: %pdoc time.sleep

This will also support top-level async when using IPython 7.0+

.. ipython::

   In [2]: import asyncio
      ...: print('before')
      ...: await asyncio.sleep(1)
      ...: print('after')


The namespace will persist across multiple code chucks, Let's define a variable:

.. ipython::

   In [0]: who = "World"

And now say hello:

.. ipython::

   In [0]: print('Hello,', who)

If the current section raises an exception, you can add the ``:okexcept:`` flag
to the current block, otherwise the build will fail.

.. ipython::
   :okexcept:

   In [1]: 1/0

IPython Sphinx directive module
===============================

To enable this directive, simply list it in your Sphinx ``conf.py`` file
(making sure the directory where you placed it is visible to sphinx, as is
needed for all Sphinx directives). For example, to enable syntax highlighting
and the IPython directive::

    extensions = ['IPython.sphinxext.ipython_console_highlighting',
                  'IPython.sphinxext.ipython_directive']

The IPython directive outputs code-blocks with the language 'ipython'. So
if you do not have the syntax highlighting extension enabled as well, then
all rendered code-blocks will be uncolored. By default this directive assumes
that your prompts are unchanged IPython ones, but this can be customized.
The configurable options that can be placed in conf.py are:

ipython_savefig_dir:
    The directory in which to save the figures. This is relative to the
    Sphinx source directory. The default is `html_static_path`.
ipython_rgxin:
    The compiled regular expression to denote the start of IPython input
    lines. The default is ``re.compile('In \\[(\\d+)\\]:\\s?(.*)\\s*')``. You
    shouldn't need to change this.
ipython_warning_is_error: [default to True]
    Fail the build if something unexpected happen, for example if a block raise
    an exception but does not have the `:okexcept:` flag. The exact behavior of
    what is considered strict, may change between the sphinx directive version.
ipython_rgxout:
    The compiled regular expression to denote the start of IPython output
    lines. The default is ``re.compile('Out\\[(\\d+)\\]:\\s?(.*)\\s*')``. You
    shouldn't need to change this.
ipython_promptin:
    The string to represent the IPython input prompt in the generated ReST.
    The default is ``'In [%d]:'``. This expects that the line numbers are used
    in the prompt.
ipython_promptout:
    The string to represent the IPython prompt in the generated ReST. The
    default is ``'Out [%d]:'``. This expects that the line numbers are used
    in the prompt.
ipython_mplbackend:
    The string which specifies if the embedded Sphinx shell should import
    Matplotlib and set the backend. The value specifies a backend that is
    passed to `matplotlib.use()` before any lines in `ipython_execlines` are
    executed. If not specified in conf.py, then the default value of 'agg' is
    used. To use the IPython directive without matplotlib as a dependency, set
    the value to `None`. It may end up that matplotlib is still imported
    if the user specifies so in `ipython_execlines` or makes use of the
    @savefig pseudo decorator.
ipython_execlines:
    A list of strings to be exec'd in the embedded Sphinx shell. Typical
    usage is to make certain packages always available. Set this to an empty
    list if you wish to have no imports always available. If specified in
    ``conf.py`` as `None`, then it has the effect of making no imports available.
    If omitted from conf.py altogether, then the default value of
    ['import numpy as np', 'import matplotlib.pyplot as plt'] is used.
ipython_holdcount
    When the @suppress pseudo-decorator is used, the execution count can be
    incremented or not. The default behavior is to hold the execution count,
    corresponding to a value of `True`. Set this to `False` to increment
    the execution count after each suppressed command.

As an example, to use the IPython directive when `matplotlib` is not available,
one sets the backend to `None`::

    ipython_mplbackend = None

An example usage of the directive is:

.. code-block:: rst

    .. ipython::

        In [1]: x = 1

        In [2]: y = x**2

        In [3]: print(y)

See http://matplotlib.org/sampledoc/ipython_directive.html for additional
documentation.

Pseudo-Decorators
=================

Note: Only one decorator is supported per input. If more than one decorator
is specified, then only the last one is used.

In addition to the Pseudo-Decorators/options described at the above link,
several enhancements have been made. The directive will emit a message to the
console at build-time if code-execution resulted in an exception or warning.
You can suppress these on a per-block basis by specifying the :okexcept:
or :okwarning: options:

.. code-block:: rst

    .. ipython::
        :okexcept:
        :okwarning:

        In [1]: 1/0
        In [2]: # raise warning.

To Do
=====

- Turn the ad-hoc test() function into a real test suite.
- Break up ipython-specific functionality from matplotlib stuff into better
  separated code.

"""

# Authors
# =======
# 
# - John D Hunter: original author.
# - Fernando Perez: refactoring, documentation, cleanups, port to 0.11.
# - VáclavŠmilauer <eudoxos-AT-arcig.cz>: Prompt generalizations.
# - Skipper Seabold, refactoring, cleanups, pure python addition

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

# Stdlib
import atexit
import errno
import os
import pathlib
import re
import sys
import tempfile
import ast
import warnings
import shutil
from io import StringIO

# Third-party
from docutils.parsers.rst import directives
from docutils.parsers.rst import Directive
from sphinx.util import logging

# Our own
from traitlets.config import Config
from IPython import InteractiveShell
from IPython.core.profiledir import ProfileDir

use_matplotlib = False
try:
    import matplotlib
    use_matplotlib = True
except Exception:
    pass

#-----------------------------------------------------------------------------
# Globals
#-----------------------------------------------------------------------------
# for tokenizing blocks
COMMENT, INPUT, OUTPUT =  range(3)

PSEUDO_DECORATORS = ["suppress", "verbatim", "savefig", "doctest"]

#-----------------------------------------------------------------------------
# Functions and class declarations
#-----------------------------------------------------------------------------

def block_parser(part, rgxin, rgxout, fmtin, fmtout):
    """
    part is a string of ipython text, comprised of at most one
    input, one output, comments, and blank lines.  The block parser
    parses the text into a list of::

      blocks = [ (TOKEN0, data0), (TOKEN1, data1), ...]

    where TOKEN is one of [COMMENT | INPUT | OUTPUT ] and
    data is, depending on the type of token::

      COMMENT : the comment string

      INPUT: the (DECORATOR, INPUT_LINE, REST) where
         DECORATOR: the input decorator (or None)
         INPUT_LINE: the input as string (possibly multi-line)
         REST : any stdout generated by the input line (not OUTPUT)

      OUTPUT: the output string, possibly multi-line

    """
    block = []
    lines = part.split('\n')
    N = len(lines)
    i = 0
    decorator = None
    while 1:

        if i==N:
            # nothing left to parse -- the last line
            break

        line = lines[i]
        i += 1
        line_stripped = line.strip()
        if line_stripped.startswith('#'):
            block.append((COMMENT, line))
            continue

        if any(
            line_stripped.startswith("@" + pseudo_decorator)
            for pseudo_decorator in PSEUDO_DECORATORS
        ):
            if decorator:
                raise RuntimeError(
                    "Applying multiple pseudo-decorators on one line is not supported"
                )
            else:
                decorator = line_stripped
                continue

        # does this look like an input line?
        matchin = rgxin.match(line)
        if matchin:
            lineno, inputline = int(matchin.group(1)), matchin.group(2)

            # the ....: continuation string
            continuation = '   %s:'%''.join(['.']*(len(str(lineno))+2))
            Nc = len(continuation)
            # input lines can continue on for more than one line, if
            # we have a '\' line continuation char or a function call
            # echo line 'print'.  The input line can only be
            # terminated by the end of the block or an output line, so
            # we parse out the rest of the input line if it is
            # multiline as well as any echo text

            rest = []
            while i<N:

                # look ahead; if the next line is blank, or a comment, or
                # an output line, we're done

                nextline = lines[i]
                matchout = rgxout.match(nextline)
                #print "nextline=%s, continuation=%s, starts=%s"%(nextline, continuation, nextline.startswith(continuation))
                if matchout or nextline.startswith('#'):
                    break
                elif nextline.startswith(continuation):
                    # The default ipython_rgx* treat the space following the colon as optional.
                    # However, If the space is there we must consume it or code
                    # employing the cython_magic extension will fail to execute.
                    #
                    # This works with the default ipython_rgx* patterns,
                    # If you modify them, YMMV.
                    nextline = nextline[Nc:]
                    if nextline and nextline[0] == ' ':
                        nextline = nextline[1:]

                    inputline += '\n' +  nextline
                else:
                    rest.append(nextline)
                i+= 1

            block.append((INPUT, (decorator, inputline, '\n'.join(rest))))
            continue

        # if it looks like an output line grab all the text to the end
        # of the block
        matchout = rgxout.match(line)
        if matchout:
            lineno, output = int(matchout.group(1)), matchout.group(2)
            if i<N-1:
                output = '\n'.join([output] + lines[i:])

            block.append((OUTPUT, output))
            break

    return block


class EmbeddedSphinxShell(object):
    """An embedded IPython instance to run inside Sphinx"""

    def __init__(self, exec_lines=None):

        self.cout = StringIO()

        if exec_lines is None:
            exec_lines = []

        # Create config object for IPython
        config = Config()
        config.HistoryManager.hist_file = ':memory:'
        config.InteractiveShell.autocall = False
        config.InteractiveShell.autoindent = False
        config.InteractiveShell.colors = 'NoColor'

        # create a profile so instance history isn't saved
        tmp_profile_dir = tempfile.mkdtemp(prefix='profile_')
        profname = 'auto_profile_sphinx_build'
        pdir = os.path.join(tmp_profile_dir,profname)
        profile = ProfileDir.create_profile_dir(pdir)

        # Create and initialize global ipython, but don't start its mainloop.
        # This will persist across different EmbeddedSphinxShell instances.
        IP = InteractiveShell.instance(config=config, profile_dir=profile)
        atexit.register(self.cleanup)

        # Store a few parts of IPython we'll need.
        self.IP = IP
        self.user_ns = self.IP.user_ns
        self.user_global_ns = self.IP.user_global_ns

        self.input = ''
        self.output = ''
        self.tmp_profile_dir = tmp_profile_dir

        self.is_verbatim = False
        self.is_doctest = False
        self.is_suppress = False

        # Optionally, provide more detailed information to shell.
        # this is assigned by the SetUp method of IPythonDirective
        # to point at itself.
        #
        # So, you can access handy things at self.directive.state
        self.directive = None

        # on the first call to the savefig decorator, we'll import
        # pyplot as plt so we can make a call to the plt.gcf().savefig
        self._pyplot_imported = False

        # Prepopulate the namespace.
        for line in exec_lines:
            self.process_input_line(line, store_history=False)

    def cleanup(self):
        shutil.rmtree(self.tmp_profile_dir, ignore_errors=True)

    def clear_cout(self):
        self.cout.seek(0)
        self.cout.truncate(0)

    def process_input_line(self, line, store_history):
        return self.process_input_lines([line], store_history=store_history)

    def process_input_lines(self, lines, store_history=True):
        """process the input, capturing stdout"""
        stdout = sys.stdout
        source_raw = '\n'.join(lines)
        try:
            sys.stdout = self.cout
            self.IP.run_cell(source_raw, store_history=store_history)
        finally:
            sys.stdout = stdout

    def process_image(self, decorator):
        """
        # build out an image directive like
        # .. image:: somefile.png
        #    :width 4in
        #
        # from an input like
        # savefig somefile.png width=4in
        """
        savefig_dir = self.savefig_dir
        source_dir = self.source_dir
        saveargs = decorator.split(' ')
        filename = saveargs[1]
        # insert relative path to image file in source 
        # as absolute path for Sphinx
        # sphinx expects a posix path, even on Windows
        path = pathlib.Path(savefig_dir, filename)
        outfile = '/' + path.relative_to(source_dir).as_posix()

        imagerows = ['.. image:: %s' % outfile]

        for kwarg in saveargs[2:]:
            arg, val = kwarg.split('=')
            arg = arg.strip()
            val = val.strip()
            imagerows.append('   :%s: %s'%(arg, val))

        image_file = os.path.basename(outfile) # only return file name
        image_directive = '\n'.join(imagerows)
        return image_file, image_directive

    # Callbacks for each type of token
    def process_input(self, data, input_prompt, lineno):
        """
        Process data block for INPUT token.

        """
        decorator, input, rest = data
        image_file = None
        image_directive = None

        is_verbatim = decorator=='@verbatim' or self.is_verbatim
        is_doctest = (decorator is not None and \
                     decorator.startswith('@doctest')) or self.is_doctest
        is_suppress = decorator=='@suppress' or self.is_suppress
        is_okexcept = decorator=='@okexcept' or self.is_okexcept
        is_okwarning = decorator=='@okwarning' or self.is_okwarning
        is_savefig = decorator is not None and \
                     decorator.startswith('@savefig')

        input_lines = input.split('\n')
        if len(input_lines) > 1:
            if input_lines[-1] != "":
                input_lines.append('') # make sure there's a blank line
                                       # so splitter buffer gets reset

        continuation = '   %s:'%''.join(['.']*(len(str(lineno))+2))

        if is_savefig:
            image_file, image_directive = self.process_image(decorator)

        ret = []
        is_semicolon = False

        # Hold the execution count, if requested to do so.
        if is_suppress and self.hold_count:
            store_history = False
        else:
            store_history = True

        # Note: catch_warnings is not thread safe
        with warnings.catch_warnings(record=True) as ws:
            if input_lines[0].endswith(';'):
                is_semicolon = True
            #for i, line in enumerate(input_lines):

            # process the first input line
            if is_verbatim:
                self.process_input_lines([''])
                self.IP.execution_count += 1 # increment it anyway
            else:
                # only submit the line in non-verbatim mode
                self.process_input_lines(input_lines, store_history=store_history)

        if not is_suppress:
            for i, line in enumerate(input_lines):
                if i == 0:
                    formatted_line = '%s %s'%(input_prompt, line)
                else:
                    formatted_line = '%s %s'%(continuation, line)
                ret.append(formatted_line)

        if not is_suppress and len(rest.strip()) and is_verbatim:
            # The "rest" is the standard output of the input. This needs to be
            # added when in verbatim mode. If there is no "rest", then we don't
            # add it, as the new line will be added by the processed output.
            ret.append(rest)

        # Fetch the processed output. (This is not the submitted output.)
        self.cout.seek(0)
        processed_output = self.cout.read()
        if not is_suppress and not is_semicolon:
            #
            # In IPythonDirective.run, the elements of `ret` are eventually
            # combined such that '' entries correspond to newlines. So if
            # `processed_output` is equal to '', then the adding it to `ret`
            # ensures that there is a blank line between consecutive inputs
            # that have no outputs, as in:
            #
            #    In [1]: x = 4
            #
            #    In [2]: x = 5
            #
            # When there is processed output, it has a '\n' at the tail end. So
            # adding the output to `ret` will provide the necessary spacing
            # between consecutive input/output blocks, as in:
            #
            #   In [1]: x
            #   Out[1]: 5
            #
            #   In [2]: x
            #   Out[2]: 5
            #
            # When there is stdout from the input, it also has a '\n' at the
            # tail end, and so this ensures proper spacing as well. E.g.:
            #
            #   In [1]: print x
            #   5
            #
            #   In [2]: x = 5
            #
            # When in verbatim mode, `processed_output` is empty (because
            # nothing was passed to IP. Sometimes the submitted code block has
            # an Out[] portion and sometimes it does not. When it does not, we
            # need to ensure proper spacing, so we have to add '' to `ret`.
            # However, if there is an Out[] in the submitted code, then we do
            # not want to add a newline as `process_output` has stuff to add.
            # The difficulty is that `process_input` doesn't know if
            # `process_output` will be called---so it doesn't know if there is
            # Out[] in the code block. The requires that we include a hack in
            # `process_block`. See the comments there.
            #
            ret.append(processed_output)
        elif is_semicolon:
            # Make sure there is a newline after the semicolon.
            ret.append('')

        # context information
        filename = "Unknown"
        lineno = 0
        if self.directive.state:
            filename = self.directive.state.document.current_source
            lineno = self.directive.state.document.current_line

        # Use sphinx logger for warnings
        logger = logging.getLogger(__name__)

        # output any exceptions raised during execution to stdout
        # unless :okexcept: has been specified.
        if not is_okexcept and (
            ("Traceback" in processed_output) or ("SyntaxError" in processed_output)
        ):
            s = "\n>>>" + ("-" * 73) + "\n"
            s += "Exception in %s at block ending on line %s\n" % (filename, lineno)
            s += "Specify :okexcept: as an option in the ipython:: block to suppress this message\n"
            s += processed_output + "\n"
            s += "<<<" + ("-" * 73)
            logger.warning(s)
            if self.warning_is_error:
                raise RuntimeError(
                    "Unexpected exception in `{}` line {}".format(filename, lineno)
                )

        # output any warning raised during execution to stdout
        # unless :okwarning: has been specified.
        if not is_okwarning:
            for w in ws:
                s = "\n>>>" + ("-" * 73) + "\n"
                s += "Warning in %s at block ending on line %s\n" % (filename, lineno)
                s += "Specify :okwarning: as an option in the ipython:: block to suppress this message\n"
                s += ("-" * 76) + "\n"
                s += warnings.formatwarning(
                    w.message, w.category, w.filename, w.lineno, w.line
                )
                s += "<<<" + ("-" * 73)
                logger.warning(s)
                if self.warning_is_error:
                    raise RuntimeError(
                        "Unexpected warning in `{}` line {}".format(filename, lineno)
                    )

        self.clear_cout()
        return (ret, input_lines, processed_output,
                is_doctest, decorator, image_file, image_directive)


    def process_output(self, data, output_prompt, input_lines, output,
                       is_doctest, decorator, image_file):
        """
        Process data block for OUTPUT token.

        """
        # Recall: `data` is the submitted output, and `output` is the processed
        # output from `input_lines`.

        TAB = ' ' * 4

        if is_doctest and output is not None:

            found = output # This is the processed output
            found = found.strip()
            submitted = data.strip()

            if self.directive is None:
                source = 'Unavailable'
                content = 'Unavailable'
            else:
                source = self.directive.state.document.current_source
                content = self.directive.content
                # Add tabs and join into a single string.
                content = '\n'.join([TAB + line for line in content])

            # Make sure the output contains the output prompt.
            ind = found.find(output_prompt)
            if ind < 0:
                e = ('output does not contain output prompt\n\n'
                     'Document source: {0}\n\n'
                     'Raw content: \n{1}\n\n'
                     'Input line(s):\n{TAB}{2}\n\n'
                     'Output line(s):\n{TAB}{3}\n\n')
                e = e.format(source, content, '\n'.join(input_lines),
                             repr(found), TAB=TAB)
                raise RuntimeError(e)
            found = found[len(output_prompt):].strip()

            # Handle the actual doctest comparison.
            if decorator.strip() == '@doctest':
                # Standard doctest
                if found != submitted:
                    e = ('doctest failure\n\n'
                         'Document source: {0}\n\n'
                         'Raw content: \n{1}\n\n'
                         'On input line(s):\n{TAB}{2}\n\n'
                         'we found output:\n{TAB}{3}\n\n'
                         'instead of the expected:\n{TAB}{4}\n\n')
                    e = e.format(source, content, '\n'.join(input_lines),
                                 repr(found), repr(submitted), TAB=TAB)
                    raise RuntimeError(e)
            else:
                self.custom_doctest(decorator, input_lines, found, submitted)

        # When in verbatim mode, this holds additional submitted output
        # to be written in the final Sphinx output.
        # https://github.com/ipython/ipython/issues/5776
        out_data = []

        is_verbatim = decorator=='@verbatim' or self.is_verbatim
        if is_verbatim and data.strip():
            # Note that `ret` in `process_block` has '' as its last element if
            # the code block was in verbatim mode. So if there is no submitted
            # output, then we will have proper spacing only if we do not add
            # an additional '' to `out_data`. This is why we condition on
            # `and data.strip()`.

            # The submitted output has no output prompt. If we want the
            # prompt and the code to appear, we need to join them now
            # instead of adding them separately---as this would create an
            # undesired newline. How we do this ultimately depends on the
            # format of the output regex. I'll do what works for the default
            # prompt for now, and we might have to adjust if it doesn't work
            # in other cases. Finally, the submitted output does not have
            # a trailing newline, so we must add it manually.
            out_data.append("{0} {1}\n".format(output_prompt, data))

        return out_data

    def process_comment(self, data):
        """Process data fPblock for COMMENT token."""
        if not self.is_suppress:
            return [data]

    def save_image(self, image_file):
        """
        Saves the image file to disk.
        """
        self.ensure_pyplot()
        command = 'plt.gcf().savefig("%s")'%image_file
        #print 'SAVEFIG', command  # dbg
        self.process_input_line('bookmark ipy_thisdir', store_history=False)
        self.process_input_line('cd -b ipy_savedir', store_history=False)
        self.process_input_line(command, store_history=False)
        self.process_input_line('cd -b ipy_thisdir', store_history=False)
        self.process_input_line('bookmark -d ipy_thisdir', store_history=False)
        self.clear_cout()

    def process_block(self, block):
        """
        process block from the block_parser and return a list of processed lines
        """
        ret = []
        output = None
        input_lines = None
        lineno = self.IP.execution_count

        input_prompt = self.promptin % lineno
        output_prompt = self.promptout % lineno
        image_file = None
        image_directive = None

        found_input = False
        for token, data in block:
            if token == COMMENT:
                out_data = self.process_comment(data)
            elif token == INPUT:
                found_input = True
                (out_data, input_lines, output, is_doctest,
                 decorator, image_file, image_directive) = \
                          self.process_input(data, input_prompt, lineno)
            elif token == OUTPUT:
                if not found_input:

                    TAB = ' ' * 4
                    linenumber = 0
                    source = 'Unavailable'
                    content = 'Unavailable'
                    if self.directive:
                        linenumber = self.directive.state.document.current_line
                        source = self.directive.state.document.current_source
                        content = self.directive.content
                        # Add tabs and join into a single string.
                        content = '\n'.join([TAB + line for line in content])

                    e = ('\n\nInvalid block: Block contains an output prompt '
                         'without an input prompt.\n\n'
                         'Document source: {0}\n\n'
                         'Content begins at line {1}: \n\n{2}\n\n'
                         'Problematic block within content: \n\n{TAB}{3}\n\n')
                    e = e.format(source, linenumber, content, block, TAB=TAB)

                    # Write, rather than include in exception, since Sphinx
                    # will truncate tracebacks.
                    sys.stdout.write(e)
                    raise RuntimeError('An invalid block was detected.')
                out_data = \
                    self.process_output(data, output_prompt, input_lines,
                                        output, is_doctest, decorator,
                                        image_file)
                if out_data:
                    # Then there was user submitted output in verbatim mode.
                    # We need to remove the last element of `ret` that was
                    # added in `process_input`, as it is '' and would introduce
                    # an undesirable newline.
                    assert(ret[-1] == '')
                    del ret[-1]

            if out_data:
                ret.extend(out_data)

        # save the image files
        if image_file is not None:
            self.save_image(image_file)

        return ret, image_directive

    def ensure_pyplot(self):
        """
        Ensures that pyplot has been imported into the embedded IPython shell.

        Also, makes sure to set the backend appropriately if not set already.

        """
        # We are here if the @figure pseudo decorator was used. Thus, it's
        # possible that we could be here even if python_mplbackend were set to
        # `None`. That's also strange and perhaps worthy of raising an
        # exception, but for now, we just set the backend to 'agg'.

        if not self._pyplot_imported:
            if 'matplotlib.backends' not in sys.modules:
                # Then ipython_matplotlib was set to None but there was a
                # call to the @figure decorator (and ipython_execlines did
                # not set a backend).
                #raise Exception("No backend was set, but @figure was used!")
                import matplotlib
                matplotlib.use('agg')

            # Always import pyplot into embedded shell.
            self.process_input_line('import matplotlib.pyplot as plt',
                                    store_history=False)
            self._pyplot_imported = True

    def process_pure_python(self, content):
        """
        content is a list of strings. it is unedited directive content

        This runs it line by line in the InteractiveShell, prepends
        prompts as needed capturing stderr and stdout, then returns
        the content as a list as if it were ipython code
        """
        output = []
        savefig = False # keep up with this to clear figure
        multiline = False # to handle line continuation
        multiline_start = None
        fmtin = self.promptin

        ct = 0

        for lineno, line in enumerate(content):

            line_stripped = line.strip()
            if not len(line):
                output.append(line)
                continue

            # handle pseudo-decorators, whilst ensuring real python decorators are treated as input
            if any(
                line_stripped.startswith("@" + pseudo_decorator)
                for pseudo_decorator in PSEUDO_DECORATORS
            ):
                output.extend([line])
                if 'savefig' in line:
                    savefig = True # and need to clear figure
                continue

            # handle comments
            if line_stripped.startswith('#'):
                output.extend([line])
                continue

            # deal with lines checking for multiline
            continuation  = u'   %s:'% ''.join(['.']*(len(str(ct))+2))
            if not multiline:
                modified = u"%s %s" % (fmtin % ct, line_stripped)
                output.append(modified)
                ct += 1
                try:
                    ast.parse(line_stripped)
                    output.append(u'')
                except Exception: # on a multiline
                    multiline = True
                    multiline_start = lineno
            else: # still on a multiline
                modified = u'%s %s' % (continuation, line)
                output.append(modified)

                # if the next line is indented, it should be part of multiline
                if len(content) > lineno + 1:
                    nextline = content[lineno + 1]
                    if len(nextline) - len(nextline.lstrip()) > 3:
                        continue
                try:
                    mod = ast.parse(
                            '\n'.join(content[multiline_start:lineno+1]))
                    if isinstance(mod.body[0], ast.FunctionDef):
                        # check to see if we have the whole function
                        for element in mod.body[0].body:
                            if isinstance(element, ast.Return):
                                multiline = False
                    else:
                        output.append(u'')
                        multiline = False
                except Exception:
                    pass

            if savefig: # clear figure if plotted
                self.ensure_pyplot()
                self.process_input_line('plt.clf()', store_history=False)
                self.clear_cout()
                savefig = False

        return output

    def custom_doctest(self, decorator, input_lines, found, submitted):
        """
        Perform a specialized doctest.

        """
        from .custom_doctests import doctests

        args = decorator.split()
        doctest_type = args[1]
        if doctest_type in doctests:
            doctests[doctest_type](self, args, input_lines, found, submitted)
        else:
            e = "Invalid option to @doctest: {0}".format(doctest_type)
            raise Exception(e)


class IPythonDirective(Directive):

    has_content = True
    required_arguments = 0
    optional_arguments = 4 # python, suppress, verbatim, doctest
    final_argumuent_whitespace = True
    option_spec = { 'python': directives.unchanged,
                    'suppress' : directives.flag,
                    'verbatim' : directives.flag,
                    'doctest' : directives.flag,
                    'okexcept': directives.flag,
                    'okwarning': directives.flag
                  }

    shell = None

    seen_docs = set()

    def get_config_options(self):
        # contains sphinx configuration variables
        config = self.state.document.settings.env.config

        # get config variables to set figure output directory
        savefig_dir = config.ipython_savefig_dir
        source_dir = self.state.document.settings.env.srcdir
        savefig_dir = os.path.join(source_dir, savefig_dir)

        # get regex and prompt stuff
        rgxin      = config.ipython_rgxin
        rgxout     = config.ipython_rgxout
        warning_is_error= config.ipython_warning_is_error
        promptin   = config.ipython_promptin
        promptout  = config.ipython_promptout
        mplbackend = config.ipython_mplbackend
        exec_lines = config.ipython_execlines
        hold_count = config.ipython_holdcount

        return (savefig_dir, source_dir, rgxin, rgxout,
                promptin, promptout, mplbackend, exec_lines, hold_count, warning_is_error)

    def setup(self):
        # Get configuration values.
        (savefig_dir, source_dir, rgxin, rgxout, promptin, promptout,
         mplbackend, exec_lines, hold_count, warning_is_error) = self.get_config_options()

        try:
            os.makedirs(savefig_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        if self.shell is None:
            # We will be here many times.  However, when the
            # EmbeddedSphinxShell is created, its interactive shell member
            # is the same for each instance.

            if mplbackend and 'matplotlib.backends' not in sys.modules and use_matplotlib:
                import matplotlib
                matplotlib.use(mplbackend)

            # Must be called after (potentially) importing matplotlib and
            # setting its backend since exec_lines might import pylab.
            self.shell = EmbeddedSphinxShell(exec_lines)

            # Store IPython directive to enable better error messages
            self.shell.directive = self

        # reset the execution count if we haven't processed this doc
        #NOTE: this may be borked if there are multiple seen_doc tmp files
        #check time stamp?
        if not self.state.document.current_source in self.seen_docs:
            self.shell.IP.history_manager.reset()
            self.shell.IP.execution_count = 1
            self.seen_docs.add(self.state.document.current_source)

        # and attach to shell so we don't have to pass them around
        self.shell.rgxin = rgxin
        self.shell.rgxout = rgxout
        self.shell.promptin = promptin
        self.shell.promptout = promptout
        self.shell.savefig_dir = savefig_dir
        self.shell.source_dir = source_dir
        self.shell.hold_count = hold_count
        self.shell.warning_is_error = warning_is_error

        # setup bookmark for saving figures directory
        self.shell.process_input_line(
            'bookmark ipy_savedir "%s"' % savefig_dir, store_history=False
        )
        self.shell.clear_cout()

        return rgxin, rgxout, promptin, promptout

    def teardown(self):
        # delete last bookmark
        self.shell.process_input_line('bookmark -d ipy_savedir',
                                      store_history=False)
        self.shell.clear_cout()

    def run(self):
        debug = False

        #TODO, any reason block_parser can't be a method of embeddable shell
        # then we wouldn't have to carry these around
        rgxin, rgxout, promptin, promptout = self.setup()

        options = self.options
        self.shell.is_suppress = 'suppress' in options
        self.shell.is_doctest = 'doctest' in options
        self.shell.is_verbatim = 'verbatim' in options
        self.shell.is_okexcept = 'okexcept' in options
        self.shell.is_okwarning = 'okwarning' in options

        # handle pure python code
        if 'python' in self.arguments:
            content = self.content
            self.content = self.shell.process_pure_python(content)

        # parts consists of all text within the ipython-block.
        # Each part is an input/output block.
        parts = '\n'.join(self.content).split('\n\n')

        lines = ['.. code-block:: ipython', '']
        figures = []

        # Use sphinx logger for warnings
        logger = logging.getLogger(__name__)

        for part in parts:
            block = block_parser(part, rgxin, rgxout, promptin, promptout)
            if len(block):
                rows, figure = self.shell.process_block(block)
                for row in rows:
                    lines.extend(['   {0}'.format(line)
                                  for line in row.split('\n')])

                if figure is not None:
                    figures.append(figure)
            else:
                message = 'Code input with no code at {}, line {}'\
                            .format(
                                self.state.document.current_source,
                                self.state.document.current_line)
                if self.shell.warning_is_error:
                    raise RuntimeError(message)
                else:
                    logger.warning(message)

        for figure in figures:
            lines.append('')
            lines.extend(figure.split('\n'))
            lines.append('')

        if len(lines) > 2:
            if debug:
                print('\n'.join(lines))
            else:
                # This has to do with input, not output. But if we comment
                # these lines out, then no IPython code will appear in the
                # final output.
                self.state_machine.insert_input(
                    lines, self.state_machine.input_lines.source(0))

        # cleanup
        self.teardown()

        return []

# Enable as a proper Sphinx directive
def setup(app):
    setup.app = app

    app.add_directive('ipython', IPythonDirective)
    app.add_config_value('ipython_savefig_dir', 'savefig', 'env')
    app.add_config_value('ipython_warning_is_error', True, 'env')
    app.add_config_value('ipython_rgxin',
                         re.compile(r'In \[(\d+)\]:\s?(.*)\s*'), 'env')
    app.add_config_value('ipython_rgxout',
                         re.compile(r'Out\[(\d+)\]:\s?(.*)\s*'), 'env')
    app.add_config_value('ipython_promptin', 'In [%d]:', 'env')
    app.add_config_value('ipython_promptout', 'Out[%d]:', 'env')

    # We could just let matplotlib pick whatever is specified as the default
    # backend in the matplotlibrc file, but this would cause issues if the
    # backend didn't work in headless environments. For this reason, 'agg'
    # is a good default backend choice.
    app.add_config_value('ipython_mplbackend', 'agg', 'env')

    # If the user sets this config value to `None`, then EmbeddedSphinxShell's
    # __init__ method will treat it as [].
    execlines = ['import numpy as np']
    if use_matplotlib:
        execlines.append('import matplotlib.pyplot as plt')
    app.add_config_value('ipython_execlines', execlines, 'env')

    app.add_config_value('ipython_holdcount', True, 'env')

    metadata = {'parallel_read_safe': True, 'parallel_write_safe': True}
    return metadata

# Simple smoke test, needs to be converted to a proper automatic test.
def test():

    examples = [
        r"""
In [9]: pwd
Out[9]: '/home/jdhunter/py4science/book'

In [10]: cd bookdata/
/home/jdhunter/py4science/book/bookdata

In [2]: from pylab import *

In [2]: ion()

In [3]: im = imread('stinkbug.png')

@savefig mystinkbug.png width=4in
In [4]: imshow(im)
Out[4]: <matplotlib.image.AxesImage object at 0x39ea850>

""",
        r"""

In [1]: x = 'hello world'

# string methods can be
# used to alter the string
@doctest
In [2]: x.upper()
Out[2]: 'HELLO WORLD'

@verbatim
In [3]: x.st<TAB>
x.startswith  x.strip
""",
    r"""

In [130]: url = 'http://ichart.finance.yahoo.com/table.csv?s=CROX\
   .....: &d=9&e=22&f=2009&g=d&a=1&br=8&c=2006&ignore=.csv'

In [131]: print url.split('&')
['http://ichart.finance.yahoo.com/table.csv?s=CROX', 'd=9', 'e=22', 'f=2009', 'g=d', 'a=1', 'b=8', 'c=2006', 'ignore=.csv']

In [60]: import urllib

""",
    r"""\

In [133]: import numpy.random

@suppress
In [134]: numpy.random.seed(2358)

@doctest
In [135]: numpy.random.rand(10,2)
Out[135]:
array([[ 0.64524308,  0.59943846],
       [ 0.47102322,  0.8715456 ],
       [ 0.29370834,  0.74776844],
       [ 0.99539577,  0.1313423 ],
       [ 0.16250302,  0.21103583],
       [ 0.81626524,  0.1312433 ],
       [ 0.67338089,  0.72302393],
       [ 0.7566368 ,  0.07033696],
       [ 0.22591016,  0.77731835],
       [ 0.0072729 ,  0.34273127]])

""",

    r"""
In [106]: print x
jdh

In [109]: for i in range(10):
   .....:     print i
   .....:
   .....:
0
1
2
3
4
5
6
7
8
9
""",

        r"""

In [144]: from pylab import *

In [145]: ion()

# use a semicolon to suppress the output
@savefig test_hist.png width=4in
In [151]: hist(np.random.randn(10000), 100);


@savefig test_plot.png width=4in
In [151]: plot(np.random.randn(10000), 'o');
   """,

        r"""
# use a semicolon to suppress the output
In [151]: plt.clf()

@savefig plot_simple.png width=4in
In [151]: plot([1,2,3])

@savefig hist_simple.png width=4in
In [151]: hist(np.random.randn(10000), 100);

""",
     r"""
# update the current fig
In [151]: ylabel('number')

In [152]: title('normal distribution')


@savefig hist_with_text.png
In [153]: grid(True)

@doctest float
In [154]: 0.1 + 0.2
Out[154]: 0.3

@doctest float
In [155]: np.arange(16).reshape(4,4)
Out[155]:
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11],
       [12, 13, 14, 15]])

In [1]: x = np.arange(16, dtype=float).reshape(4,4)

In [2]: x[0,0] = np.inf

In [3]: x[0,1] = np.nan

@doctest float
In [4]: x
Out[4]:
array([[ inf,  nan,   2.,   3.],
       [  4.,   5.,   6.,   7.],
       [  8.,   9.,  10.,  11.],
       [ 12.,  13.,  14.,  15.]])


        """,
        ]
    # skip local-file depending first example:
    examples = examples[1:]

    #ipython_directive.DEBUG = True  # dbg
    #options = dict(suppress=True)  # dbg
    options = {}
    for example in examples:
        content = example.split('\n')
        IPythonDirective('debug', arguments=None, options=options,
                          content=content, lineno=0,
                          content_offset=None, block_text=None,
                          state=None, state_machine=None,
                          )

# Run test suite as a script
if __name__=='__main__':
    if not os.path.isdir('_static'):
        os.mkdir('_static')
    test()
    print('All OK? Check figures in _static/')
