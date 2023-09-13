"""Implementation of magic functions for interaction with the OS.

Note: this module is named 'osm' instead of 'os' to avoid a collision with the
builtin.
"""
# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.

import io
import os
import pathlib
import re
import sys
from pprint import pformat

from IPython.core import magic_arguments
from IPython.core import oinspect
from IPython.core import page
from IPython.core.alias import AliasError, Alias
from IPython.core.error import UsageError
from IPython.core.magic import  (
    Magics, compress_dhist, magics_class, line_magic, cell_magic, line_cell_magic
)
from IPython.testing.skipdoctest import skip_doctest
from IPython.utils.openpy import source_to_unicode
from IPython.utils.process import abbrev_cwd
from IPython.utils.terminal import set_term_title
from traitlets import Bool
from warnings import warn


@magics_class
class OSMagics(Magics):
    """Magics to interact with the underlying OS (shell-type functionality).
    """

    cd_force_quiet = Bool(False,
        help="Force %cd magic to be quiet even if -q is not passed."
    ).tag(config=True)

    def __init__(self, shell=None, **kwargs):

        # Now define isexec in a cross platform manner.
        self.is_posix = False
        self.execre = None
        if os.name == 'posix':
            self.is_posix = True
        else:
            try:
                winext = os.environ['pathext'].replace(';','|').replace('.','')
            except KeyError:
                winext = 'exe|com|bat|py'
            try:
                self.execre = re.compile(r'(.*)\.(%s)$' % winext,re.IGNORECASE)
            except re.error:
                warn("Seems like your pathext environmental "
                     "variable is malformed. Please check it to "
                     "enable a proper handle of file extensions "
                     "managed for your system")
                winext = 'exe|com|bat|py'
                self.execre = re.compile(r'(.*)\.(%s)$' % winext,re.IGNORECASE)

        # call up the chain
        super().__init__(shell=shell, **kwargs)


    def _isexec_POSIX(self, file):
        """
        Test for executable on a POSIX system
        """
        if os.access(file.path, os.X_OK):
            # will fail on maxOS if access is not X_OK
            return file.is_file()
        return False


    
    def _isexec_WIN(self, file):
        """
        Test for executable file on non POSIX system
        """
        return file.is_file() and self.execre.match(file.name) is not None

    def isexec(self, file):
        """
        Test for executable file on non POSIX system
        """
        if self.is_posix:
            return self._isexec_POSIX(file)
        else:
            return self._isexec_WIN(file)


    @skip_doctest
    @line_magic
    def alias(self, parameter_s=''):
        """Define an alias for a system command.

        '%alias alias_name cmd' defines 'alias_name' as an alias for 'cmd'

        Then, typing 'alias_name params' will execute the system command 'cmd
        params' (from your underlying operating system).

        Aliases have lower precedence than magic functions and Python normal
        variables, so if 'foo' is both a Python variable and an alias, the
        alias can not be executed until 'del foo' removes the Python variable.

        You can use the %l specifier in an alias definition to represent the
        whole line when the alias is called.  For example::

          In [2]: alias bracket echo "Input in brackets: <%l>"
          In [3]: bracket hello world
          Input in brackets: <hello world>

        You can also define aliases with parameters using %s specifiers (one
        per parameter)::

          In [1]: alias parts echo first %s second %s
          In [2]: %parts A B
          first A second B
          In [3]: %parts A
          Incorrect number of arguments: 2 expected.
          parts is an alias to: 'echo first %s second %s'

        Note that %l and %s are mutually exclusive.  You can only use one or
        the other in your aliases.

        Aliases expand Python variables just like system calls using ! or !!
        do: all expressions prefixed with '$' get expanded.  For details of
        the semantic rules, see PEP-215:
        https://peps.python.org/pep-0215/.  This is the library used by
        IPython for variable expansion.  If you want to access a true shell
        variable, an extra $ is necessary to prevent its expansion by
        IPython::

          In [6]: alias show echo
          In [7]: PATH='A Python string'
          In [8]: show $PATH
          A Python string
          In [9]: show $$PATH
          /usr/local/lf9560/bin:/usr/local/intel/compiler70/ia32/bin:...

        You can use the alias facility to access all of $PATH.  See the %rehashx
        function, which automatically creates aliases for the contents of your
        $PATH.

        If called with no parameters, %alias prints the current alias table
        for your system.  For posix systems, the default aliases are 'cat',
        'cp', 'mv', 'rm', 'rmdir', and 'mkdir', and other platform-specific
        aliases are added.  For windows-based systems, the default aliases are
        'copy', 'ddir', 'echo', 'ls', 'ldir', 'mkdir', 'ren', and 'rmdir'.

        You can see the definition of alias by adding a question mark in the
        end::

          In [1]: cat?
          Repr: <alias cat for 'cat'>"""

        par = parameter_s.strip()
        if not par:
            aliases = sorted(self.shell.alias_manager.aliases)
            # stored = self.shell.db.get('stored_aliases', {} )
            # for k, v in stored:
            #     atab.append(k, v[0])

            print("Total number of aliases:", len(aliases))
            sys.stdout.flush()
            return aliases

        # Now try to define a new one
        try:
            alias,cmd = par.split(None, 1)
        except TypeError:
            print(oinspect.getdoc(self.alias))
            return
        
        try:
            self.shell.alias_manager.define_alias(alias, cmd)
        except AliasError as e:
            print(e)
    # end magic_alias

    @line_magic
    def unalias(self, parameter_s=''):
        """Remove an alias"""

        aname = parameter_s.strip()
        try:
            self.shell.alias_manager.undefine_alias(aname)
        except ValueError as e:
            print(e)
            return
        
        stored = self.shell.db.get('stored_aliases', {} )
        if aname in stored:
            print("Removing %stored alias",aname)
            del stored[aname]
            self.shell.db['stored_aliases'] = stored

    @line_magic
    def rehashx(self, parameter_s=''):
        """Update the alias table with all executable files in $PATH.

        rehashx explicitly checks that every entry in $PATH is a file
        with execute access (os.X_OK).

        Under Windows, it checks executability as a match against a
        '|'-separated string of extensions, stored in the IPython config
        variable win_exec_ext.  This defaults to 'exe|com|bat'.

        This function also resets the root module cache of module completer,
        used on slow filesystems.
        """
        from IPython.core.alias import InvalidAliasError

        # for the benefit of module completer in ipy_completers.py
        del self.shell.db['rootmodules_cache']

        path = [os.path.abspath(os.path.expanduser(p)) for p in
            os.environ.get('PATH','').split(os.pathsep)]

        syscmdlist = []
        savedir = os.getcwd()

        # Now walk the paths looking for executables to alias.
        try:
            # write the whole loop for posix/Windows so we don't have an if in
            # the innermost part
            if self.is_posix:
                for pdir in path:
                    try:
                        os.chdir(pdir)
                    except OSError:
                        continue

                    # for python 3.6+ rewrite to: with os.scandir(pdir) as dirlist:
                    dirlist = os.scandir(path=pdir)
                    for ff in dirlist:
                        if self.isexec(ff):
                            fname = ff.name
                            try:
                                # Removes dots from the name since ipython
                                # will assume names with dots to be python.
                                if not self.shell.alias_manager.is_alias(fname):
                                    self.shell.alias_manager.define_alias(
                                        fname.replace('.',''), fname)
                            except InvalidAliasError:
                                pass
                            else:
                                syscmdlist.append(fname)
            else:
                no_alias = Alias.blacklist
                for pdir in path:
                    try:
                        os.chdir(pdir)
                    except OSError:
                        continue

                    # for python 3.6+ rewrite to: with os.scandir(pdir) as dirlist:
                    dirlist = os.scandir(pdir)
                    for ff in dirlist:
                        fname = ff.name
                        base, ext = os.path.splitext(fname)
                        if self.isexec(ff) and base.lower() not in no_alias:
                            if ext.lower() == '.exe':
                                fname = base
                                try:
                                    # Removes dots from the name since ipython
                                    # will assume names with dots to be python.
                                    self.shell.alias_manager.define_alias(
                                        base.lower().replace('.',''), fname)
                                except InvalidAliasError:
                                    pass
                                syscmdlist.append(fname)

            self.shell.db['syscmdlist'] = syscmdlist
        finally:
            os.chdir(savedir)

    @skip_doctest
    @line_magic
    def pwd(self, parameter_s=''):
        """Return the current working directory path.

        Examples
        --------
        ::

          In [9]: pwd
          Out[9]: '/home/tsuser/sprint/ipython'
        """
        try:
            return os.getcwd()
        except FileNotFoundError as e:
            raise UsageError("CWD no longer exists - please use %cd to change directory.") from e

    @skip_doctest
    @line_magic
    def cd(self, parameter_s=''):
        """Change the current working directory.

        This command automatically maintains an internal list of directories
        you visit during your IPython session, in the variable ``_dh``. The
        command :magic:`%dhist` shows this history nicely formatted. You can
        also do ``cd -<tab>`` to see directory history conveniently.
        Usage:

          - ``cd 'dir'``: changes to directory 'dir'.
          - ``cd -``: changes to the last visited directory.
          - ``cd -<n>``: changes to the n-th directory in the directory history.
          - ``cd --foo``: change to directory that matches 'foo' in history
          - ``cd -b <bookmark_name>``: jump to a bookmark set by %bookmark
          - Hitting a tab key after ``cd -b`` allows you to tab-complete
            bookmark names.

          .. note::
            ``cd <bookmark_name>`` is enough if there is no directory
            ``<bookmark_name>``, but a bookmark with the name exists.

        Options:

        -q               Be quiet. Do not print the working directory after the
                         cd command is executed. By default IPython's cd
                         command does print this directory, since the default
                         prompts do not display path information.

        .. note::
           Note that ``!cd`` doesn't work for this purpose because the shell
           where ``!command`` runs is immediately discarded after executing
           'command'.

        Examples
        --------
        ::

          In [10]: cd parent/child
          /home/tsuser/parent/child
        """

        try:
            oldcwd = os.getcwd()
        except FileNotFoundError:
            # Happens if the CWD has been deleted.
            oldcwd = None

        numcd = re.match(r'(-)(\d+)$',parameter_s)
        # jump in directory history by number
        if numcd:
            nn = int(numcd.group(2))
            try:
                ps = self.shell.user_ns['_dh'][nn]
            except IndexError:
                print('The requested directory does not exist in history.')
                return
            else:
                opts = {}
        elif parameter_s.startswith('--'):
            ps = None
            fallback = None
            pat = parameter_s[2:]
            dh = self.shell.user_ns['_dh']
            # first search only by basename (last component)
            for ent in reversed(dh):
                if pat in os.path.basename(ent) and os.path.isdir(ent):
                    ps = ent
                    break

                if fallback is None and pat in ent and os.path.isdir(ent):
                    fallback = ent

            # if we have no last part match, pick the first full path match
            if ps is None:
                ps = fallback

            if ps is None:
                print("No matching entry in directory history")
                return
            else:
                opts = {}


        else:
            opts, ps = self.parse_options(parameter_s, 'qb', mode='string')
        # jump to previous
        if ps == '-':
            try:
                ps = self.shell.user_ns['_dh'][-2]
            except IndexError as e:
                raise UsageError('%cd -: No previous directory to change to.') from e
        # jump to bookmark if needed
        else:
            if not os.path.isdir(ps) or 'b' in opts:
                bkms = self.shell.db.get('bookmarks', {})

                if ps in bkms:
                    target = bkms[ps]
                    print('(bookmark:%s) -> %s' % (ps, target))
                    ps = target
                else:
                    if 'b' in opts:
                        raise UsageError("Bookmark '%s' not found.  "
                              "Use '%%bookmark -l' to see your bookmarks." % ps)

        # at this point ps should point to the target dir
        if ps:
            try:
                os.chdir(os.path.expanduser(ps))
                if hasattr(self.shell, 'term_title') and self.shell.term_title:
                    set_term_title(self.shell.term_title_format.format(cwd=abbrev_cwd()))
            except OSError:
                print(sys.exc_info()[1])
            else:
                cwd = pathlib.Path.cwd()
                dhist = self.shell.user_ns['_dh']
                if oldcwd != cwd:
                    dhist.append(cwd)
                    self.shell.db['dhist'] = compress_dhist(dhist)[-100:]

        else:
            os.chdir(self.shell.home_dir)
            if hasattr(self.shell, 'term_title') and self.shell.term_title:
                set_term_title(self.shell.term_title_format.format(cwd="~"))
            cwd = pathlib.Path.cwd()
            dhist = self.shell.user_ns['_dh']

            if oldcwd != cwd:
                dhist.append(cwd)
                self.shell.db['dhist'] = compress_dhist(dhist)[-100:]
        if not 'q' in opts and not self.cd_force_quiet and self.shell.user_ns['_dh']:
            print(self.shell.user_ns['_dh'][-1])

    @line_magic
    def env(self, parameter_s=''):
        """Get, set, or list environment variables.

        Usage:\\

          :``%env``: lists all environment variables/values
          :``%env var``: get value for var
          :``%env var val``: set value for var
          :``%env var=val``: set value for var
          :``%env var=$val``: set value for var, using python expansion if possible
        """
        if parameter_s.strip():
            split = '=' if '=' in parameter_s else ' '
            bits = parameter_s.split(split)
            if len(bits) == 1:
                key = parameter_s.strip()
                if key in os.environ:
                    return os.environ[key]
                else:
                    err = "Environment does not have key: {0}".format(key)
                    raise UsageError(err)
            if len(bits) > 1:
                return self.set_env(parameter_s)
        env = dict(os.environ)
        # hide likely secrets when printing the whole environment
        for key in list(env):
            if any(s in key.lower() for s in ('key', 'token', 'secret')):
                env[key] = '<hidden>'

        return env

    @line_magic
    def set_env(self, parameter_s):
        """Set environment variables.  Assumptions are that either "val" is a
        name in the user namespace, or val is something that evaluates to a
        string.

        Usage:\\
          :``%set_env var val``: set value for var
          :``%set_env var=val``: set value for var
          :``%set_env var=$val``: set value for var, using python expansion if possible
        """
        split = '=' if '=' in parameter_s else ' '
        bits = parameter_s.split(split, 1)
        if not parameter_s.strip() or len(bits)<2:
            raise UsageError("usage is 'set_env var=val'")
        var = bits[0].strip()
        val = bits[1].strip()
        if re.match(r'.*\s.*', var):
            # an environment variable with whitespace is almost certainly
            # not what the user intended.  what's more likely is the wrong
            # split was chosen, ie for "set_env cmd_args A=B", we chose
            # '=' for the split and should have chosen ' '.  to get around
            # this, users should just assign directly to os.environ or use
            # standard magic {var} expansion.
            err = "refusing to set env var with whitespace: '{0}'"
            err = err.format(val)
            raise UsageError(err)
        os.environ[var] = val
        print('env: {0}={1}'.format(var,val))

    @line_magic
    def pushd(self, parameter_s=''):
        """Place the current dir on stack and change directory.

        Usage:\\
          %pushd ['dirname']
        """

        dir_s = self.shell.dir_stack
        tgt = os.path.expanduser(parameter_s)
        cwd = os.getcwd().replace(self.shell.home_dir,'~')
        if tgt:
            self.cd(parameter_s)
        dir_s.insert(0,cwd)
        return self.shell.run_line_magic('dirs', '')

    @line_magic
    def popd(self, parameter_s=''):
        """Change to directory popped off the top of the stack.
        """
        if not self.shell.dir_stack:
            raise UsageError("%popd on empty stack")
        top = self.shell.dir_stack.pop(0)
        self.cd(top)
        print("popd ->",top)

    @line_magic
    def dirs(self, parameter_s=''):
        """Return the current directory stack."""

        return self.shell.dir_stack

    @line_magic
    def dhist(self, parameter_s=''):
        """Print your history of visited directories.

        %dhist       -> print full history\\
        %dhist n     -> print last n entries only\\
        %dhist n1 n2 -> print entries between n1 and n2 (n2 not included)\\

        This history is automatically maintained by the %cd command, and
        always available as the global list variable _dh. You can use %cd -<n>
        to go to directory number <n>.

        Note that most of time, you should view directory history by entering
        cd -<TAB>.

        """

        dh = self.shell.user_ns['_dh']
        if parameter_s:
            try:
                args = map(int,parameter_s.split())
            except:
                self.arg_err(self.dhist)
                return
            if len(args) == 1:
                ini,fin = max(len(dh)-(args[0]),0),len(dh)
            elif len(args) == 2:
                ini,fin = args
                fin = min(fin, len(dh))
            else:
                self.arg_err(self.dhist)
                return
        else:
            ini,fin = 0,len(dh)
        print('Directory history (kept in _dh)')
        for i in range(ini, fin):
            print("%d: %s" % (i, dh[i]))

    @skip_doctest
    @line_magic
    def sc(self, parameter_s=''):
        """Shell capture - run shell command and capture output (DEPRECATED use !).

        DEPRECATED. Suboptimal, retained for backwards compatibility.

        You should use the form 'var = !command' instead. Example:

         "%sc -l myfiles = ls ~" should now be written as

         "myfiles = !ls ~"

        myfiles.s, myfiles.l and myfiles.n still apply as documented
        below.

        --
        %sc [options] varname=command

        IPython will run the given command using commands.getoutput(), and
        will then update the user's interactive namespace with a variable
        called varname, containing the value of the call.  Your command can
        contain shell wildcards, pipes, etc.

        The '=' sign in the syntax is mandatory, and the variable name you
        supply must follow Python's standard conventions for valid names.

        (A special format without variable name exists for internal use)

        Options:

          -l: list output.  Split the output on newlines into a list before
          assigning it to the given variable.  By default the output is stored
          as a single string.

          -v: verbose.  Print the contents of the variable.

        In most cases you should not need to split as a list, because the
        returned value is a special type of string which can automatically
        provide its contents either as a list (split on newlines) or as a
        space-separated string.  These are convenient, respectively, either
        for sequential processing or to be passed to a shell command.

        For example::

            # Capture into variable a
            In [1]: sc a=ls *py

            # a is a string with embedded newlines
            In [2]: a
            Out[2]: 'setup.py\\nwin32_manual_post_install.py'

            # which can be seen as a list:
            In [3]: a.l
            Out[3]: ['setup.py', 'win32_manual_post_install.py']

            # or as a whitespace-separated string:
            In [4]: a.s
            Out[4]: 'setup.py win32_manual_post_install.py'

            # a.s is useful to pass as a single command line:
            In [5]: !wc -l $a.s
              146 setup.py
              130 win32_manual_post_install.py
              276 total

            # while the list form is useful to loop over:
            In [6]: for f in a.l:
               ...:      !wc -l $f
               ...:
            146 setup.py
            130 win32_manual_post_install.py

        Similarly, the lists returned by the -l option are also special, in
        the sense that you can equally invoke the .s attribute on them to
        automatically get a whitespace-separated string from their contents::

            In [7]: sc -l b=ls *py

            In [8]: b
            Out[8]: ['setup.py', 'win32_manual_post_install.py']

            In [9]: b.s
            Out[9]: 'setup.py win32_manual_post_install.py'

        In summary, both the lists and strings used for output capture have
        the following special attributes::

            .l (or .list) : value as list.
            .n (or .nlstr): value as newline-separated string.
            .s (or .spstr): value as space-separated string.
        """

        opts,args = self.parse_options(parameter_s, 'lv')
        # Try to get a variable name and command to run
        try:
            # the variable name must be obtained from the parse_options
            # output, which uses shlex.split to strip options out.
            var,_ = args.split('=', 1)
            var = var.strip()
            # But the command has to be extracted from the original input
            # parameter_s, not on what parse_options returns, to avoid the
            # quote stripping which shlex.split performs on it.
            _,cmd = parameter_s.split('=', 1)
        except ValueError:
            var,cmd = '',''
        # If all looks ok, proceed
        split = 'l' in opts
        out = self.shell.getoutput(cmd, split=split)
        if 'v' in opts:
            print('%s ==\n%s' % (var, pformat(out)))
        if var:
            self.shell.user_ns.update({var:out})
        else:
            return out

    @line_cell_magic
    def sx(self, line='', cell=None):
        """Shell execute - run shell command and capture output (!! is short-hand).

        %sx command

        IPython will run the given command using commands.getoutput(), and
        return the result formatted as a list (split on '\\n').  Since the
        output is _returned_, it will be stored in ipython's regular output
        cache Out[N] and in the '_N' automatic variables.

        Notes:

        1) If an input line begins with '!!', then %sx is automatically
        invoked.  That is, while::

          !ls

        causes ipython to simply issue system('ls'), typing::

          !!ls

        is a shorthand equivalent to::

          %sx ls

        2) %sx differs from %sc in that %sx automatically splits into a list,
        like '%sc -l'.  The reason for this is to make it as easy as possible
        to process line-oriented shell output via further python commands.
        %sc is meant to provide much finer control, but requires more
        typing.

        3) Just like %sc -l, this is a list with special attributes:
        ::

          .l (or .list) : value as list.
          .n (or .nlstr): value as newline-separated string.
          .s (or .spstr): value as whitespace-separated string.

        This is very useful when trying to use such lists as arguments to
        system commands."""
        
        if cell is None:
            # line magic
            return self.shell.getoutput(line)
        else:
            opts,args = self.parse_options(line, '', 'out=')
            output = self.shell.getoutput(cell)
            out_name = opts.get('out', opts.get('o'))
            if out_name:
                self.shell.user_ns[out_name] = output
            else:
                return output

    system = line_cell_magic('system')(sx)
    bang = cell_magic('!')(sx)

    @line_magic
    def bookmark(self, parameter_s=''):
        """Manage IPython's bookmark system.

        %bookmark <name>       - set bookmark to current dir
        %bookmark <name> <dir> - set bookmark to <dir>
        %bookmark -l           - list all bookmarks
        %bookmark -d <name>    - remove bookmark
        %bookmark -r           - remove all bookmarks

        You can later on access a bookmarked folder with::

          %cd -b <name>

        or simply '%cd <name>' if there is no directory called <name> AND
        there is such a bookmark defined.

        Your bookmarks persist through IPython sessions, but they are
        associated with each profile."""

        opts,args = self.parse_options(parameter_s,'drl',mode='list')
        if len(args) > 2:
            raise UsageError("%bookmark: too many arguments")

        bkms = self.shell.db.get('bookmarks',{})

        if 'd' in opts:
            try:
                todel = args[0]
            except IndexError as e:
                raise UsageError(
                    "%bookmark -d: must provide a bookmark to delete") from e
            else:
                try:
                    del bkms[todel]
                except KeyError as e:
                    raise UsageError(
                        "%%bookmark -d: Can't delete bookmark '%s'" % todel) from e

        elif 'r' in opts:
            bkms = {}
        elif 'l' in opts:
            bks = sorted(bkms)
            if bks:
                size = max(map(len, bks))
            else:
                size = 0
            fmt = '%-'+str(size)+'s -> %s'
            print('Current bookmarks:')
            for bk in bks:
                print(fmt % (bk, bkms[bk]))
        else:
            if not args:
                raise UsageError("%bookmark: You must specify the bookmark name")
            elif len(args)==1:
                bkms[args[0]] = os.getcwd()
            elif len(args)==2:
                bkms[args[0]] = args[1]
        self.shell.db['bookmarks'] = bkms

    @line_magic
    def pycat(self, parameter_s=''):
        """Show a syntax-highlighted file through a pager.

        This magic is similar to the cat utility, but it will assume the file
        to be Python source and will show it with syntax highlighting.

        This magic command can either take a local filename, an url,
        an history range (see %history) or a macro as argument.

        If no parameter is given, prints out history of current session up to
        this point. ::

        %pycat myscript.py
        %pycat 7-27
        %pycat myMacro
        %pycat http://www.example.com/myscript.py
        """
        try:
            cont = self.shell.find_user_code(parameter_s, skip_encoding_cookie=False)
        except (ValueError, IOError):
            print("Error: no such file, variable, URL, history range or macro")
            return

        page.page(self.shell.pycolorize(source_to_unicode(cont)))

    @magic_arguments.magic_arguments()
    @magic_arguments.argument(
        '-a', '--append', action='store_true', default=False,
        help='Append contents of the cell to an existing file. '
             'The file will be created if it does not exist.'
    )
    @magic_arguments.argument(
        'filename', type=str,
        help='file to write'
    )
    @cell_magic
    def writefile(self, line, cell):
        """Write the contents of the cell to a file.

        The file will be overwritten unless the -a (--append) flag is specified.
        """
        args = magic_arguments.parse_argstring(self.writefile, line)
        if re.match(r'^(\'.*\')|(".*")$', args.filename):
            filename = os.path.expanduser(args.filename[1:-1])
        else:
            filename = os.path.expanduser(args.filename)
            
        if os.path.exists(filename):
            if args.append:
                print("Appending to %s" % filename)
            else:
                print("Overwriting %s" % filename)
        else:
            print("Writing %s" % filename)
        
        mode = 'a' if args.append else 'w'
        with io.open(filename, mode, encoding='utf-8') as f:
            f.write(cell)
