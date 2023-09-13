"""Implementation of code management magic functions.
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

# Stdlib
import inspect
import io
import os
import re
import sys
import ast
from itertools import chain
from urllib.request import Request, urlopen
from urllib.parse import urlencode
from pathlib import Path

# Our own packages
from IPython.core.error import TryNext, StdinNotImplementedError, UsageError
from IPython.core.macro import Macro
from IPython.core.magic import Magics, magics_class, line_magic
from IPython.core.oinspect import find_file, find_source_lines
from IPython.core.release import version
from IPython.testing.skipdoctest import skip_doctest
from IPython.utils.contexts import preserve_keys
from IPython.utils.path import get_py_filename
from warnings import warn
from logging import error
from IPython.utils.text import get_text_list

#-----------------------------------------------------------------------------
# Magic implementation classes
#-----------------------------------------------------------------------------

# Used for exception handling in magic_edit
class MacroToEdit(ValueError): pass

ipython_input_pat = re.compile(r"<ipython\-input\-(\d+)-[a-z\d]+>$")

# To match, e.g. 8-10 1:5 :10 3-
range_re = re.compile(r"""
(?P<start>\d+)?
((?P<sep>[\-:])
 (?P<end>\d+)?)?
$""", re.VERBOSE)


def extract_code_ranges(ranges_str):
    """Turn a string of range for %%load into 2-tuples of (start, stop)
    ready to use as a slice of the content split by lines.

    Examples
    --------
    list(extract_input_ranges("5-10 2"))
    [(4, 10), (1, 2)]
    """
    for range_str in ranges_str.split():
        rmatch = range_re.match(range_str)
        if not rmatch:
            continue
        sep = rmatch.group("sep")
        start = rmatch.group("start")
        end = rmatch.group("end")

        if sep == '-':
            start = int(start) - 1 if start else None
            end = int(end) if end else None
        elif sep == ':':
            start = int(start) - 1 if start else None
            end = int(end) - 1 if end else None
        else:
            end = int(start)
            start = int(start) - 1
        yield (start, end)


def extract_symbols(code, symbols):
    """
    Return a tuple  (blocks, not_found)
    where ``blocks`` is a list of code fragments
    for each symbol parsed from code, and ``not_found`` are
    symbols not found in the code.

    For example::

        In [1]: code = '''a = 10
           ...: def b(): return 42
           ...: class A: pass'''

        In [2]: extract_symbols(code, 'A,b,z')
        Out[2]: (['class A: pass\\n', 'def b(): return 42\\n'], ['z'])
    """
    symbols = symbols.split(',')

    # this will raise SyntaxError if code isn't valid Python
    py_code = ast.parse(code)

    marks = [(getattr(s, 'name', None), s.lineno) for s in py_code.body]
    code = code.split('\n')

    symbols_lines = {}
    
    # we already know the start_lineno of each symbol (marks). 
    # To find each end_lineno, we traverse in reverse order until each 
    # non-blank line
    end = len(code)  
    for name, start in reversed(marks):
        while not code[end - 1].strip():
            end -= 1
        if name:
            symbols_lines[name] = (start - 1, end)
        end = start - 1

    # Now symbols_lines is a map
    # {'symbol_name': (start_lineno, end_lineno), ...}
    
    # fill a list with chunks of codes for each requested symbol
    blocks = []
    not_found = []
    for symbol in symbols:
        if symbol in symbols_lines:
            start, end = symbols_lines[symbol]
            blocks.append('\n'.join(code[start:end]) + '\n')
        else:
            not_found.append(symbol)

    return blocks, not_found

def strip_initial_indent(lines):
    """For %load, strip indent from lines until finding an unindented line.

    https://github.com/ipython/ipython/issues/9775
    """
    indent_re = re.compile(r'\s+')

    it = iter(lines)
    first_line = next(it)
    indent_match = indent_re.match(first_line)

    if indent_match:
        # First line was indented
        indent = indent_match.group()
        yield first_line[len(indent):]

        for line in it:
            if line.startswith(indent):
                yield line[len(indent):]
            else:
                # Less indented than the first line - stop dedenting
                yield line
                break
    else:
        yield first_line

    # Pass the remaining lines through without dedenting
    for line in it:
        yield line


class InteractivelyDefined(Exception):
    """Exception for interactively defined variable in magic_edit"""
    def __init__(self, index):
        self.index = index


@magics_class
class CodeMagics(Magics):
    """Magics related to code management (loading, saving, editing, ...)."""

    def __init__(self, *args, **kwargs):
        self._knowntemps = set()
        super(CodeMagics, self).__init__(*args, **kwargs)

    @line_magic
    def save(self, parameter_s=''):
        """Save a set of lines or a macro to a given filename.

        Usage:\\
          %save [options] filename [history]

        Options:

          -r: use 'raw' input.  By default, the 'processed' history is used,
          so that magics are loaded in their transformed version to valid
          Python.  If this option is given, the raw input as typed as the
          command line is used instead.
          
          -f: force overwrite.  If file exists, %save will prompt for overwrite
          unless -f is given.

          -a: append to the file instead of overwriting it.

        The history argument uses the same syntax as %history for input ranges,
        then saves the lines to the filename you specify.

        If no ranges are specified, saves history of the current session up to
        this point.

        It adds a '.py' extension to the file if you don't do so yourself, and
        it asks for confirmation before overwriting existing files.

        If `-r` option is used, the default extension is `.ipy`.
        """

        opts,args = self.parse_options(parameter_s,'fra',mode='list')
        if not args:
            raise UsageError('Missing filename.')
        raw = 'r' in opts
        force = 'f' in opts
        append = 'a' in opts
        mode = 'a' if append else 'w'
        ext = '.ipy' if raw else '.py'
        fname, codefrom = args[0], " ".join(args[1:])
        if not fname.endswith(('.py','.ipy')):
            fname += ext
        fname = os.path.expanduser(fname)
        file_exists = os.path.isfile(fname)
        if file_exists and not force and not append:
            try:
                overwrite = self.shell.ask_yes_no('File `%s` exists. Overwrite (y/[N])? ' % fname, default='n')
            except StdinNotImplementedError:
                print("File `%s` exists. Use `%%save -f %s` to force overwrite" % (fname, parameter_s))
                return
            if not overwrite :
                print('Operation cancelled.')
                return
        try:
            cmds = self.shell.find_user_code(codefrom,raw)
        except (TypeError, ValueError) as e:
            print(e.args[0])
            return
        with io.open(fname, mode, encoding="utf-8") as f:
            if not file_exists or not append:
                f.write("# coding: utf-8\n")
            f.write(cmds)
            # make sure we end on a newline
            if not cmds.endswith('\n'):
                f.write('\n')
        print('The following commands were written to file `%s`:' % fname)
        print(cmds)

    @line_magic
    def pastebin(self, parameter_s=''):
        """Upload code to dpaste.com, returning the URL.

        Usage:\\
          %pastebin [-d "Custom description"][-e 24] 1-7

        The argument can be an input history range, a filename, or the name of a
        string or macro.

        If no arguments are given, uploads the history of this session up to
        this point.

        Options:

          -d: Pass a custom description. The default will say
              "Pasted from IPython".
          -e: Pass number of days for the link to be expired.
              The default will be 7 days.
        """
        opts, args = self.parse_options(parameter_s, "d:e:")

        try:
            code = self.shell.find_user_code(args)
        except (ValueError, TypeError) as e:
            print(e.args[0])
            return

        expiry_days = 7
        try:
            expiry_days = int(opts.get("e", 7))
        except ValueError as e:
            print(e.args[0].capitalize())
            return
        if expiry_days < 1 or expiry_days > 365:
            print("Expiry days should be in range of 1 to 365")
            return

        post_data = urlencode(
            {
                "title": opts.get("d", "Pasted from IPython"),
                "syntax": "python",
                "content": code,
                "expiry_days": expiry_days,
            }
        ).encode("utf-8")

        request = Request(
            "https://dpaste.com/api/v2/",
            headers={"User-Agent": "IPython v{}".format(version)},
        )
        response = urlopen(request, post_data)
        return response.headers.get('Location')

    @line_magic
    def loadpy(self, arg_s):
        """Alias of `%load`

        `%loadpy` has gained some flexibility and dropped the requirement of a `.py`
        extension. So it has been renamed simply into %load. You can look at
        `%load`'s docstring for more info.
        """
        self.load(arg_s)

    @line_magic
    def load(self, arg_s):
        """Load code into the current frontend.

        Usage:\\
          %load [options] source

          where source can be a filename, URL, input history range, macro, or
          element in the user namespace

        If no arguments are given, loads the history of this session up to this
        point.

        Options:

          -r <lines>: Specify lines or ranges of lines to load from the source.
          Ranges could be specified as x-y (x..y) or in python-style x:y 
          (x..(y-1)). Both limits x and y can be left blank (meaning the 
          beginning and end of the file, respectively).

          -s <symbols>: Specify function or classes to load from python source. 

          -y : Don't ask confirmation for loading source above 200 000 characters.

          -n : Include the user's namespace when searching for source code.

        This magic command can either take a local filename, a URL, an history
        range (see %history) or a macro as argument, it will prompt for
        confirmation before loading source with more than 200 000 characters, unless
        -y flag is passed or if the frontend does not support raw_input::

        %load
        %load myscript.py
        %load 7-27
        %load myMacro
        %load http://www.example.com/myscript.py
        %load -r 5-10 myscript.py
        %load -r 10-20,30,40: foo.py
        %load -s MyClass,wonder_function myscript.py
        %load -n MyClass
        %load -n my_module.wonder_function
        """
        opts,args = self.parse_options(arg_s,'yns:r:')
        search_ns = 'n' in opts
        contents = self.shell.find_user_code(args, search_ns=search_ns)

        if 's' in opts:
            try:
                blocks, not_found = extract_symbols(contents, opts['s'])
            except SyntaxError:
                # non python code
                error("Unable to parse the input as valid Python code")
                return

            if len(not_found) == 1:
                warn('The symbol `%s` was not found' % not_found[0])
            elif len(not_found) > 1:
                warn('The symbols %s were not found' % get_text_list(not_found,
                                                                     wrap_item_with='`')
                )

            contents = '\n'.join(blocks)

        if 'r' in opts:
            ranges = opts['r'].replace(',', ' ')
            lines = contents.split('\n')
            slices = extract_code_ranges(ranges)
            contents = [lines[slice(*slc)] for slc in slices]
            contents = '\n'.join(strip_initial_indent(chain.from_iterable(contents)))

        l = len(contents)

        # 200 000 is ~ 2500 full 80 character lines
        # so in average, more than 5000 lines
        if l > 200000 and 'y' not in opts:
            try:
                ans = self.shell.ask_yes_no(("The text you're trying to load seems pretty big"\
                " (%d characters). Continue (y/[N]) ?" % l), default='n' )
            except StdinNotImplementedError:
                #assume yes if raw input not implemented
                ans = True

            if ans is False :
                print('Operation cancelled.')
                return

        contents = "# %load {}\n".format(arg_s) + contents

        self.shell.set_next_input(contents, replace=True)

    @staticmethod
    def _find_edit_target(shell, args, opts, last_call):
        """Utility method used by magic_edit to find what to edit."""

        def make_filename(arg):
            "Make a filename from the given args"
            try:
                filename = get_py_filename(arg)
            except IOError:
                # If it ends with .py but doesn't already exist, assume we want
                # a new file.
                if arg.endswith('.py'):
                    filename = arg
                else:
                    filename = None
            return filename

        # Set a few locals from the options for convenience:
        opts_prev = 'p' in opts
        opts_raw = 'r' in opts

        # custom exceptions
        class DataIsObject(Exception): pass

        # Default line number value
        lineno = opts.get('n',None)

        if opts_prev:
            args = '_%s' % last_call[0]
            if args not in shell.user_ns:
                args = last_call[1]

        # by default this is done with temp files, except when the given
        # arg is a filename
        use_temp = True

        data = ''

        # First, see if the arguments should be a filename.
        filename = make_filename(args)
        if filename:
            use_temp = False
        elif args:
            # Mode where user specifies ranges of lines, like in %macro.
            data = shell.extract_input_lines(args, opts_raw)
            if not data:
                try:
                    # Load the parameter given as a variable. If not a string,
                    # process it as an object instead (below)

                    #print '*** args',args,'type',type(args)  # dbg
                    data = eval(args, shell.user_ns)
                    if not isinstance(data, str):
                        raise DataIsObject

                except (NameError,SyntaxError):
                    # given argument is not a variable, try as a filename
                    filename = make_filename(args)
                    if filename is None:
                        warn("Argument given (%s) can't be found as a variable "
                             "or as a filename." % args)
                        return (None, None, None)
                    use_temp = False

                except DataIsObject as e:
                    # macros have a special edit function
                    if isinstance(data, Macro):
                        raise MacroToEdit(data) from e

                    # For objects, try to edit the file where they are defined
                    filename = find_file(data)
                    if filename:
                        if 'fakemodule' in filename.lower() and \
                            inspect.isclass(data):
                            # class created by %edit? Try to find source
                            # by looking for method definitions instead, the
                            # __module__ in those classes is FakeModule.
                            attrs = [getattr(data, aname) for aname in dir(data)]
                            for attr in attrs:
                                if not inspect.ismethod(attr):
                                    continue
                                filename = find_file(attr)
                                if filename and \
                                  'fakemodule' not in filename.lower():
                                    # change the attribute to be the edit
                                    # target instead
                                    data = attr
                                    break
                        
                        m = ipython_input_pat.match(os.path.basename(filename))
                        if m:
                            raise InteractivelyDefined(int(m.groups()[0])) from e

                        datafile = 1
                    if filename is None:
                        filename = make_filename(args)
                        datafile = 1
                        if filename is not None:
                            # only warn about this if we get a real name
                            warn('Could not find file where `%s` is defined.\n'
                             'Opening a file named `%s`' % (args, filename))
                    # Now, make sure we can actually read the source (if it was
                    # in a temp file it's gone by now).
                    if datafile:
                        if lineno is None:
                            lineno = find_source_lines(data)
                        if lineno is None:
                            filename = make_filename(args)
                            if filename is None:
                                warn('The file where `%s` was defined '
                                     'cannot be read or found.' % data)
                                return (None, None, None)
                    use_temp = False

        if use_temp:
            filename = shell.mktempfile(data)
            print('IPython will make a temporary file named:',filename)

        # use last_call to remember the state of the previous call, but don't
        # let it be clobbered by successive '-p' calls.
        try:
            last_call[0] = shell.displayhook.prompt_count
            if not opts_prev:
                last_call[1] = args
        except:
            pass


        return filename, lineno, use_temp

    def _edit_macro(self,mname,macro):
        """open an editor with the macro data in a file"""
        filename = self.shell.mktempfile(macro.value)
        self.shell.hooks.editor(filename)

        # and make a new macro object, to replace the old one
        mvalue = Path(filename).read_text(encoding="utf-8")
        self.shell.user_ns[mname] = Macro(mvalue)

    @skip_doctest
    @line_magic
    def edit(self, parameter_s='',last_call=['','']):
        """Bring up an editor and execute the resulting code.

        Usage:
          %edit [options] [args]

        %edit runs IPython's editor hook. The default version of this hook is
        set to call the editor specified by your $EDITOR environment variable.
        If this isn't found, it will default to vi under Linux/Unix and to
        notepad under Windows. See the end of this docstring for how to change
        the editor hook.

        You can also set the value of this editor via the
        ``TerminalInteractiveShell.editor`` option in your configuration file.
        This is useful if you wish to use a different editor from your typical
        default with IPython (and for Windows users who typically don't set
        environment variables).

        This command allows you to conveniently edit multi-line code right in
        your IPython session.

        If called without arguments, %edit opens up an empty editor with a
        temporary file and will execute the contents of this file when you
        close it (don't forget to save it!).


        Options:

        -n <number>: open the editor at a specified line number.  By default,
        the IPython editor hook uses the unix syntax 'editor +N filename', but
        you can configure this by providing your own modified hook if your
        favorite editor supports line-number specifications with a different
        syntax.

        -p: this will call the editor with the same data as the previous time
        it was used, regardless of how long ago (in your current session) it
        was.

        -r: use 'raw' input.  This option only applies to input taken from the
        user's history.  By default, the 'processed' history is used, so that
        magics are loaded in their transformed version to valid Python.  If
        this option is given, the raw input as typed as the command line is
        used instead.  When you exit the editor, it will be executed by
        IPython's own processor.

        -x: do not execute the edited code immediately upon exit. This is
        mainly useful if you are editing programs which need to be called with
        command line arguments, which you can then do using %run.


        Arguments:

        If arguments are given, the following possibilities exist:

        - If the argument is a filename, IPython will load that into the
          editor. It will execute its contents with execfile() when you exit,
          loading any code in the file into your interactive namespace.

        - The arguments are ranges of input history,  e.g. "7 ~1/4-6".
          The syntax is the same as in the %history magic.

        - If the argument is a string variable, its contents are loaded
          into the editor. You can thus edit any string which contains
          python code (including the result of previous edits).

        - If the argument is the name of an object (other than a string),
          IPython will try to locate the file where it was defined and open the
          editor at the point where it is defined. You can use `%edit function`
          to load an editor exactly at the point where 'function' is defined,
          edit it and have the file be executed automatically.

        - If the object is a macro (see %macro for details), this opens up your
          specified editor with a temporary file containing the macro's data.
          Upon exit, the macro is reloaded with the contents of the file.

        Note: opening at an exact line is only supported under Unix, and some
        editors (like kedit and gedit up to Gnome 2.8) do not understand the
        '+NUMBER' parameter necessary for this feature. Good editors like
        (X)Emacs, vi, jed, pico and joe all do.

        After executing your code, %edit will return as output the code you
        typed in the editor (except when it was an existing file). This way
        you can reload the code in further invocations of %edit as a variable,
        via _<NUMBER> or Out[<NUMBER>], where <NUMBER> is the prompt number of
        the output.

        Note that %edit is also available through the alias %ed.

        This is an example of creating a simple function inside the editor and
        then modifying it. First, start up the editor::

          In [1]: edit
          Editing... done. Executing edited code...
          Out[1]: 'def foo():\\n    print "foo() was defined in an editing
          session"\\n'

        We can then call the function foo()::

          In [2]: foo()
          foo() was defined in an editing session

        Now we edit foo.  IPython automatically loads the editor with the
        (temporary) file where foo() was previously defined::

          In [3]: edit foo
          Editing... done. Executing edited code...

        And if we call foo() again we get the modified version::

          In [4]: foo()
          foo() has now been changed!

        Here is an example of how to edit a code snippet successive
        times. First we call the editor::

          In [5]: edit
          Editing... done. Executing edited code...
          hello
          Out[5]: "print 'hello'\\n"

        Now we call it again with the previous output (stored in _)::

          In [6]: edit _
          Editing... done. Executing edited code...
          hello world
          Out[6]: "print 'hello world'\\n"

        Now we call it with the output #8 (stored in _8, also as Out[8])::

          In [7]: edit _8
          Editing... done. Executing edited code...
          hello again
          Out[7]: "print 'hello again'\\n"


        Changing the default editor hook:

        If you wish to write your own editor hook, you can put it in a
        configuration file which you load at startup time.  The default hook
        is defined in the IPython.core.hooks module, and you can use that as a
        starting example for further modifications.  That file also has
        general instructions on how to set a new hook for use once you've
        defined it."""
        opts,args = self.parse_options(parameter_s,'prxn:')

        try:
            filename, lineno, is_temp = self._find_edit_target(self.shell, 
                                                       args, opts, last_call)
        except MacroToEdit as e:
            self._edit_macro(args, e.args[0])
            return
        except InteractivelyDefined as e:
            print("Editing In[%i]" % e.index)
            args = str(e.index)
            filename, lineno, is_temp = self._find_edit_target(self.shell, 
                                                       args, opts, last_call)
        if filename is None:
            # nothing was found, warnings have already been issued,
            # just give up.
            return

        if is_temp:
            self._knowntemps.add(filename)
        elif (filename in self._knowntemps):
            is_temp = True


        # do actual editing here
        print('Editing...', end=' ')
        sys.stdout.flush()
        filepath = Path(filename)
        try:
            # Quote filenames that may have spaces in them when opening
            # the editor
            quoted = filename = str(filepath.absolute())
            if " " in quoted:
                quoted = "'%s'" % quoted
            self.shell.hooks.editor(quoted, lineno)
        except TryNext:
            warn('Could not open editor')
            return

        # XXX TODO: should this be generalized for all string vars?
        # For now, this is special-cased to blocks created by cpaste
        if args.strip() == "pasted_block":
            self.shell.user_ns["pasted_block"] = filepath.read_text(encoding="utf-8")

        if 'x' in opts:  # -x prevents actual execution
            print()
        else:
            print('done. Executing edited code...')
            with preserve_keys(self.shell.user_ns, '__file__'):
                if not is_temp:
                    self.shell.user_ns["__file__"] = filename
                if "r" in opts:  # Untranslated IPython code
                    source = filepath.read_text(encoding="utf-8")
                    self.shell.run_cell(source, store_history=False)
                else:
                    self.shell.safe_execfile(filename, self.shell.user_ns,
                                             self.shell.user_ns)

        if is_temp:
            try:
                return filepath.read_text(encoding="utf-8")
            except IOError as msg:
                if Path(msg.filename) == filepath:
                    warn('File not found. Did you forget to save?')
                    return
                else:
                    self.shell.showtraceback()
