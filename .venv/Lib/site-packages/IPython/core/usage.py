# -*- coding: utf-8 -*-
"""Usage information for the main IPython applications.
"""
#-----------------------------------------------------------------------------
#  Copyright (C) 2008-2011  The IPython Development Team
#  Copyright (C) 2001-2007 Fernando Perez. <fperez@colorado.edu>
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
#-----------------------------------------------------------------------------

import sys
from IPython.core import release

cl_usage = """\
=========
 IPython
=========

Tools for Interactive Computing in Python
=========================================

    A Python shell with automatic history (input and output), dynamic object
    introspection, easier configuration, command completion, access to the
    system shell and more.  IPython can also be embedded in running programs.


Usage

    ipython [subcommand] [options] [-c cmd | -m mod | file] [--] [arg] ...

    If invoked with no options, it executes the file and exits, passing the
    remaining arguments to the script, just as if you had specified the same
    command with python. You may need to specify `--` before args to be passed
    to the script, to prevent IPython from attempting to parse them. If you
    specify the option `-i` before the filename, it will enter an interactive
    IPython session after running the script, rather than exiting. Files ending
    in .py will be treated as normal Python, but files ending in .ipy can
    contain special IPython syntax (magic commands, shell expansions, etc.).

    Almost all configuration in IPython is available via the command-line. Do
    `ipython --help-all` to see all available options.  For persistent
    configuration, look into your `ipython_config.py` configuration file for
    details.

    This file is typically installed in the `IPYTHONDIR` directory, and there
    is a separate configuration directory for each profile. The default profile
    directory will be located in $IPYTHONDIR/profile_default. IPYTHONDIR
    defaults to to `$HOME/.ipython`.  For Windows users, $HOME resolves to
    C:\\Users\\YourUserName in most instances.

    To initialize a profile with the default configuration file, do::

      $> ipython profile create

    and start editing `IPYTHONDIR/profile_default/ipython_config.py`

    In IPython's documentation, we will refer to this directory as
    `IPYTHONDIR`, you can change its default location by creating an
    environment variable with this name and setting it to the desired path.

    For more information, see the manual available in HTML and PDF in your
    installation, or online at https://ipython.org/documentation.html.
"""

interactive_usage = """
IPython -- An enhanced Interactive Python
=========================================

IPython offers a fully compatible replacement for the standard Python
interpreter, with convenient shell features, special commands, command
history mechanism and output results caching.

At your system command line, type 'ipython -h' to see the command line
options available. This document only describes interactive features.

GETTING HELP
------------

Within IPython you have various way to access help:

  ?         -> Introduction and overview of IPython's features (this screen).
  object?   -> Details about 'object'.
  object??  -> More detailed, verbose information about 'object'.
  %quickref -> Quick reference of all IPython specific syntax and magics.
  help      -> Access Python's own help system.

If you are in terminal IPython you can quit this screen by pressing `q`.


MAIN FEATURES
-------------

* Access to the standard Python help with object docstrings and the Python
  manuals. Simply type 'help' (no quotes) to invoke it.

* Magic commands: type %magic for information on the magic subsystem.

* System command aliases, via the %alias command or the configuration file(s).

* Dynamic object information:

  Typing ?word or word? prints detailed information about an object. Certain
  long strings (code, etc.) get snipped in the center for brevity.

  Typing ??word or word?? gives access to the full information without
  snipping long strings. Strings that are longer than the screen are printed
  through the less pager.

  The ?/?? system gives access to the full source code for any object (if
  available), shows function prototypes and other useful information.

  If you just want to see an object's docstring, type '%pdoc object' (without
  quotes, and without % if you have automagic on).

* Tab completion in the local namespace:

  At any time, hitting tab will complete any available python commands or
  variable names, and show you a list of the possible completions if there's
  no unambiguous one. It will also complete filenames in the current directory.

* Search previous command history in multiple ways:

  - Start typing, and then use arrow keys up/down or (Ctrl-p/Ctrl-n) to search
    through the history items that match what you've typed so far.

  - Hit Ctrl-r: opens a search prompt. Begin typing and the system searches
    your history for lines that match what you've typed so far, completing as
    much as it can.

  - %hist: search history by index.

* Persistent command history across sessions.

* Logging of input with the ability to save and restore a working session.

* System shell with !. Typing !ls will run 'ls' in the current directory.

* The reload command does a 'deep' reload of a module: changes made to the
  module since you imported will actually be available without having to exit.

* Verbose and colored exception traceback printouts. See the magic xmode and
  xcolor functions for details (just type %magic).

* Input caching system:

  IPython offers numbered prompts (In/Out) with input and output caching. All
  input is saved and can be retrieved as variables (besides the usual arrow
  key recall).

  The following GLOBAL variables always exist (so don't overwrite them!):
  _i: stores previous input.
  _ii: next previous.
  _iii: next-next previous.
  _ih : a list of all input _ih[n] is the input from line n.

  Additionally, global variables named _i<n> are dynamically created (<n>
  being the prompt counter), such that _i<n> == _ih[<n>]

  For example, what you typed at prompt 14 is available as _i14 and _ih[14].

  You can create macros which contain multiple input lines from this history,
  for later re-execution, with the %macro function.

  The history function %hist allows you to see any part of your input history
  by printing a range of the _i variables. Note that inputs which contain
  magic functions (%) appear in the history with a prepended comment. This is
  because they aren't really valid Python code, so you can't exec them.

* Output caching system:

  For output that is returned from actions, a system similar to the input
  cache exists but using _ instead of _i. Only actions that produce a result
  (NOT assignments, for example) are cached. If you are familiar with
  Mathematica, IPython's _ variables behave exactly like Mathematica's %
  variables.

  The following GLOBAL variables always exist (so don't overwrite them!):
  _ (one underscore): previous output.
  __ (two underscores): next previous.
  ___ (three underscores): next-next previous.

  Global variables named _<n> are dynamically created (<n> being the prompt
  counter), such that the result of output <n> is always available as _<n>.

  Finally, a global dictionary named _oh exists with entries for all lines
  which generated output.

* Directory history:

  Your history of visited directories is kept in the global list _dh, and the
  magic %cd command can be used to go to any entry in that list.

* Auto-parentheses and auto-quotes (adapted from Nathan Gray's LazyPython)

  1. Auto-parentheses
        
     Callable objects (i.e. functions, methods, etc) can be invoked like
     this (notice the commas between the arguments)::
       
         In [1]: callable_ob arg1, arg2, arg3
       
     and the input will be translated to this::
       
         callable_ob(arg1, arg2, arg3)
       
     This feature is off by default (in rare cases it can produce
     undesirable side-effects), but you can activate it at the command-line
     by starting IPython with `--autocall 1`, set it permanently in your
     configuration file, or turn on at runtime with `%autocall 1`.

     You can force auto-parentheses by using '/' as the first character
     of a line.  For example::
       
          In [1]: /globals             # becomes 'globals()'
       
     Note that the '/' MUST be the first character on the line!  This
     won't work::
       
          In [2]: print /globals    # syntax error

     In most cases the automatic algorithm should work, so you should
     rarely need to explicitly invoke /. One notable exception is if you
     are trying to call a function with a list of tuples as arguments (the
     parenthesis will confuse IPython)::
       
          In [1]: zip (1,2,3),(4,5,6)  # won't work
       
     but this will work::
       
          In [2]: /zip (1,2,3),(4,5,6)
          ------> zip ((1,2,3),(4,5,6))
          Out[2]= [(1, 4), (2, 5), (3, 6)]

     IPython tells you that it has altered your command line by
     displaying the new command line preceded by -->.  e.g.::
       
          In [18]: callable list
          -------> callable (list)

  2. Auto-Quoting
    
     You can force auto-quoting of a function's arguments by using ',' as
     the first character of a line.  For example::
       
          In [1]: ,my_function /home/me   # becomes my_function("/home/me")

     If you use ';' instead, the whole argument is quoted as a single
     string (while ',' splits on whitespace)::
       
          In [2]: ,my_function a b c   # becomes my_function("a","b","c")
          In [3]: ;my_function a b c   # becomes my_function("a b c")

     Note that the ',' MUST be the first character on the line!  This
     won't work::
       
          In [4]: x = ,my_function /home/me    # syntax error
"""

interactive_usage_min =  """\
An enhanced console for Python.
Some of its features are:
- Tab completion in the local namespace.
- Logging of input, see command-line options.
- System shell escape via ! , eg !ls.
- Magic commands, starting with a % (like %ls, %pwd, %cd, etc.)
- Keeps track of locally defined variables via %who, %whos.
- Show object information with a ? eg ?x or x? (use ?? for more info).
"""

quick_reference = r"""
IPython -- An enhanced Interactive Python - Quick Reference Card
================================================================

obj?, obj??      : Get help, or more help for object (also works as
                   ?obj, ??obj).
?foo.*abc*       : List names in 'foo' containing 'abc' in them.
%magic           : Information about IPython's 'magic' % functions.

Magic functions are prefixed by % or %%, and typically take their arguments
without parentheses, quotes or even commas for convenience.  Line magics take a
single % and cell magics are prefixed with two %%.

Example magic function calls:

%alias d ls -F   : 'd' is now an alias for 'ls -F'
alias d ls -F    : Works if 'alias' not a python name
alist = %alias   : Get list of aliases to 'alist'
cd /usr/share    : Obvious. cd -<tab> to choose from visited dirs.
%cd??            : See help AND source for magic %cd
%timeit x=10     : time the 'x=10' statement with high precision.
%%timeit x=2**100
x**100           : time 'x**100' with a setup of 'x=2**100'; setup code is not
                   counted.  This is an example of a cell magic.

System commands:

!cp a.txt b/     : System command escape, calls os.system()
cp a.txt b/      : after %rehashx, most system commands work without !
cp ${f}.txt $bar : Variable expansion in magics and system commands
files = !ls /usr : Capture system command output
files.s, files.l, files.n: "a b c", ['a','b','c'], 'a\nb\nc'

History:

_i, _ii, _iii    : Previous, next previous, next next previous input
_i4, _ih[2:5]    : Input history line 4, lines 2-4
exec(_i81)       : Execute input history line #81 again
%rep 81          : Edit input history line #81
_, __, ___       : previous, next previous, next next previous output
_dh              : Directory history
_oh              : Output history
%hist            : Command history of current session.
%hist -g foo     : Search command history of (almost) all sessions for 'foo'.
%hist -g         : Command history of (almost) all sessions.
%hist 1/2-8      : Command history containing lines 2-8 of session 1.
%hist 1/ ~2/     : Command history of session 1 and 2 sessions before current.
%hist ~8/1-~6/5  : Command history from line 1 of 8 sessions ago to
                   line 5 of 6 sessions ago.
%edit 0/         : Open editor to execute code with history of current session.

Autocall:

f 1,2            : f(1,2)  # Off by default, enable with %autocall magic.
/f 1,2           : f(1,2) (forced autoparen)
,f 1 2           : f("1","2")
;f 1 2           : f("1 2")

Remember: TAB completion works in many contexts, not just file names
or python names.

The following magic functions are currently available:

"""

default_banner_parts = ["Python %s\n"%sys.version.split("\n")[0],
    "Type 'copyright', 'credits' or 'license' for more information\n" ,
    "IPython {version} -- An enhanced Interactive Python. Type '?' for help.\n".format(version=release.version),
]

default_banner = ''.join(default_banner_parts)
