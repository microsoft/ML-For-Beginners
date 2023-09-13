"""Extra magics for terminal use."""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.


from logging import error
import os
import sys

from IPython.core.error import TryNext, UsageError
from IPython.core.magic import Magics, magics_class, line_magic
from IPython.lib.clipboard import ClipboardEmpty
from IPython.testing.skipdoctest import skip_doctest
from IPython.utils.text import SList, strip_email_quotes
from IPython.utils import py3compat

def get_pasted_lines(sentinel, l_input=py3compat.input, quiet=False):
    """ Yield pasted lines until the user enters the given sentinel value.
    """
    if not quiet:
        print("Pasting code; enter '%s' alone on the line to stop or use Ctrl-D." \
              % sentinel)
        prompt = ":"
    else:
        prompt = ""
    while True:
        try:
            l = l_input(prompt)
            if l == sentinel:
                return
            else:
                yield l
        except EOFError:
            print('<EOF>')
            return


@magics_class
class TerminalMagics(Magics):
    def __init__(self, shell):
        super(TerminalMagics, self).__init__(shell)

    def store_or_execute(self, block, name, store_history=False):
        """ Execute a block, or store it in a variable, per the user's request.
        """
        if name:
            # If storing it for further editing
            self.shell.user_ns[name] = SList(block.splitlines())
            print("Block assigned to '%s'" % name)
        else:
            b = self.preclean_input(block)
            self.shell.user_ns['pasted_block'] = b
            self.shell.using_paste_magics = True
            try:
                self.shell.run_cell(b, store_history)
            finally:
                self.shell.using_paste_magics = False

    def preclean_input(self, block):
        lines = block.splitlines()
        while lines and not lines[0].strip():
            lines = lines[1:]
        return strip_email_quotes('\n'.join(lines))

    def rerun_pasted(self, name='pasted_block'):
        """ Rerun a previously pasted command.
        """
        b = self.shell.user_ns.get(name)

        # Sanity checks
        if b is None:
            raise UsageError('No previous pasted block available')
        if not isinstance(b, str):
            raise UsageError(
                "Variable 'pasted_block' is not a string, can't execute")

        print("Re-executing '%s...' (%d chars)"% (b.split('\n',1)[0], len(b)))
        self.shell.run_cell(b)

    @line_magic
    def autoindent(self, parameter_s = ''):
        """Toggle autoindent on/off (deprecated)"""
        self.shell.set_autoindent()
        print("Automatic indentation is:",['OFF','ON'][self.shell.autoindent])

    @skip_doctest
    @line_magic
    def cpaste(self, parameter_s=''):
        """Paste & execute a pre-formatted code block from clipboard.

        You must terminate the block with '--' (two minus-signs) or Ctrl-D
        alone on the line. You can also provide your own sentinel with '%paste
        -s %%' ('%%' is the new sentinel for this operation).

        The block is dedented prior to execution to enable execution of method
        definitions. '>' and '+' characters at the beginning of a line are
        ignored, to allow pasting directly from e-mails, diff files and
        doctests (the '...' continuation prompt is also stripped).  The
        executed block is also assigned to variable named 'pasted_block' for
        later editing with '%edit pasted_block'.

        You can also pass a variable name as an argument, e.g. '%cpaste foo'.
        This assigns the pasted block to variable 'foo' as string, without
        dedenting or executing it (preceding >>> and + is still stripped)

        '%cpaste -r' re-executes the block previously entered by cpaste.
        '%cpaste -q' suppresses any additional output messages.

        Do not be alarmed by garbled output on Windows (it's a readline bug).
        Just press enter and type -- (and press enter again) and the block
        will be what was just pasted.

        Shell escapes are not supported (yet).

        See Also
        --------
        paste : automatically pull code from clipboard.

        Examples
        --------
        ::

          In [8]: %cpaste
          Pasting code; enter '--' alone on the line to stop.
          :>>> a = ["world!", "Hello"]
          :>>> print(" ".join(sorted(a)))
          :--
          Hello world!

        ::
          In [8]: %cpaste
          Pasting code; enter '--' alone on the line to stop.
          :>>> %alias_magic t timeit
          :>>> %t -n1 pass
          :--
          Created `%t` as an alias for `%timeit`.
          Created `%%t` as an alias for `%%timeit`.
          354 ns ± 224 ns per loop (mean ± std. dev. of 7 runs, 1 loop each)
        """
        opts, name = self.parse_options(parameter_s, 'rqs:', mode='string')
        if 'r' in opts:
            self.rerun_pasted()
            return

        quiet = ('q' in opts)

        sentinel = opts.get('s', u'--')
        block = '\n'.join(get_pasted_lines(sentinel, quiet=quiet))
        self.store_or_execute(block, name, store_history=True)

    @line_magic
    def paste(self, parameter_s=''):
        """Paste & execute a pre-formatted code block from clipboard.

        The text is pulled directly from the clipboard without user
        intervention and printed back on the screen before execution (unless
        the -q flag is given to force quiet mode).

        The block is dedented prior to execution to enable execution of method
        definitions. '>' and '+' characters at the beginning of a line are
        ignored, to allow pasting directly from e-mails, diff files and
        doctests (the '...' continuation prompt is also stripped).  The
        executed block is also assigned to variable named 'pasted_block' for
        later editing with '%edit pasted_block'.

        You can also pass a variable name as an argument, e.g. '%paste foo'.
        This assigns the pasted block to variable 'foo' as string, without
        executing it (preceding >>> and + is still stripped).

        Options:

          -r: re-executes the block previously entered by cpaste.

          -q: quiet mode: do not echo the pasted text back to the terminal.

        IPython statements (magics, shell escapes) are not supported (yet).

        See Also
        --------
        cpaste : manually paste code into terminal until you mark its end.
        """
        opts, name = self.parse_options(parameter_s, 'rq', mode='string')
        if 'r' in opts:
            self.rerun_pasted()
            return
        try:
            block = self.shell.hooks.clipboard_get()
        except TryNext as clipboard_exc:
            message = getattr(clipboard_exc, 'args')
            if message:
                error(message[0])
            else:
                error('Could not get text from the clipboard.')
            return
        except ClipboardEmpty as e:
            raise UsageError("The clipboard appears to be empty") from e

        # By default, echo back to terminal unless quiet mode is requested
        if 'q' not in opts:
            sys.stdout.write(self.shell.pycolorize(block))
            if not block.endswith("\n"):
                sys.stdout.write("\n")
            sys.stdout.write("## -- End pasted text --\n")

        self.store_or_execute(block, name, store_history=True)

    # Class-level: add a '%cls' magic only on Windows
    if sys.platform == 'win32':
        @line_magic
        def cls(self, s):
            """Clear screen.
            """
            os.system("cls")
