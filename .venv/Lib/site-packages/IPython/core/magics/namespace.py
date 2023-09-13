"""Implementation of namespace-related magic functions.
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
import gc
import re
import sys

# Our own packages
from IPython.core import page
from IPython.core.error import StdinNotImplementedError, UsageError
from IPython.core.magic import Magics, magics_class, line_magic
from IPython.testing.skipdoctest import skip_doctest
from IPython.utils.encoding import DEFAULT_ENCODING
from IPython.utils.openpy import read_py_file
from IPython.utils.path import get_py_filename

#-----------------------------------------------------------------------------
# Magic implementation classes
#-----------------------------------------------------------------------------

@magics_class
class NamespaceMagics(Magics):
    """Magics to manage various aspects of the user's namespace.

    These include listing variables, introspecting into them, etc.
    """

    @line_magic
    def pinfo(self, parameter_s='', namespaces=None):
        """Provide detailed information about an object.

        '%pinfo object' is just a synonym for object? or ?object."""

        #print 'pinfo par: <%s>' % parameter_s  # dbg
        # detail_level: 0 -> obj? , 1 -> obj??
        detail_level = 0
        # We need to detect if we got called as 'pinfo pinfo foo', which can
        # happen if the user types 'pinfo foo?' at the cmd line.
        pinfo,qmark1,oname,qmark2 = \
               re.match(r'(pinfo )?(\?*)(.*?)(\??$)',parameter_s).groups()
        if pinfo or qmark1 or qmark2:
            detail_level = 1
        if "*" in oname:
            self.psearch(oname)
        else:
            self.shell._inspect('pinfo', oname, detail_level=detail_level,
                                namespaces=namespaces)

    @line_magic
    def pinfo2(self, parameter_s='', namespaces=None):
        """Provide extra detailed information about an object.

        '%pinfo2 object' is just a synonym for object?? or ??object."""
        self.shell._inspect('pinfo', parameter_s, detail_level=1,
                            namespaces=namespaces)

    @skip_doctest
    @line_magic
    def pdef(self, parameter_s='', namespaces=None):
        """Print the call signature for any callable object.

        If the object is a class, print the constructor information.

        Examples
        --------
        ::

          In [3]: %pdef urllib.urlopen
          urllib.urlopen(url, data=None, proxies=None)
        """
        self.shell._inspect('pdef',parameter_s, namespaces)

    @line_magic
    def pdoc(self, parameter_s='', namespaces=None):
        """Print the docstring for an object.

        If the given object is a class, it will print both the class and the
        constructor docstrings."""
        self.shell._inspect('pdoc',parameter_s, namespaces)

    @line_magic
    def psource(self, parameter_s='', namespaces=None):
        """Print (or run through pager) the source code for an object."""
        if not parameter_s:
            raise UsageError('Missing object name.')
        self.shell._inspect('psource',parameter_s, namespaces)

    @line_magic
    def pfile(self, parameter_s='', namespaces=None):
        """Print (or run through pager) the file where an object is defined.

        The file opens at the line where the object definition begins. IPython
        will honor the environment variable PAGER if set, and otherwise will
        do its best to print the file in a convenient form.

        If the given argument is not an object currently defined, IPython will
        try to interpret it as a filename (automatically adding a .py extension
        if needed). You can thus use %pfile as a syntax highlighting code
        viewer."""

        # first interpret argument as an object name
        out = self.shell._inspect('pfile',parameter_s, namespaces)
        # if not, try the input as a filename
        if out == 'not found':
            try:
                filename = get_py_filename(parameter_s)
            except IOError as msg:
                print(msg)
                return
            page.page(self.shell.pycolorize(read_py_file(filename, skip_encoding_cookie=False)))

    @line_magic
    def psearch(self, parameter_s=''):
        """Search for object in namespaces by wildcard.

        %psearch [options] PATTERN [OBJECT TYPE]

        Note: ? can be used as a synonym for %psearch, at the beginning or at
        the end: both a*? and ?a* are equivalent to '%psearch a*'.  Still, the
        rest of the command line must be unchanged (options come first), so
        for example the following forms are equivalent

        %psearch -i a* function
        -i a* function?
        ?-i a* function

        Arguments:

          PATTERN

          where PATTERN is a string containing * as a wildcard similar to its
          use in a shell.  The pattern is matched in all namespaces on the
          search path. By default objects starting with a single _ are not
          matched, many IPython generated objects have a single
          underscore. The default is case insensitive matching. Matching is
          also done on the attributes of objects and not only on the objects
          in a module.

          [OBJECT TYPE]

          Is the name of a python type from the types module. The name is
          given in lowercase without the ending type, ex. StringType is
          written string. By adding a type here only objects matching the
          given type are matched. Using all here makes the pattern match all
          types (this is the default).

        Options:

          -a: makes the pattern match even objects whose names start with a
          single underscore.  These names are normally omitted from the
          search.

          -i/-c: make the pattern case insensitive/sensitive.  If neither of
          these options are given, the default is read from your configuration
          file, with the option ``InteractiveShell.wildcards_case_sensitive``.
          If this option is not specified in your configuration file, IPython's
          internal default is to do a case sensitive search.

          -e/-s NAMESPACE: exclude/search a given namespace.  The pattern you
          specify can be searched in any of the following namespaces:
          'builtin', 'user', 'user_global','internal', 'alias', where
          'builtin' and 'user' are the search defaults.  Note that you should
          not use quotes when specifying namespaces.

          -l: List all available object types for object matching. This function
          can be used without arguments.

          'Builtin' contains the python module builtin, 'user' contains all
          user data, 'alias' only contain the shell aliases and no python
          objects, 'internal' contains objects used by IPython.  The
          'user_global' namespace is only used by embedded IPython instances,
          and it contains module-level globals.  You can add namespaces to the
          search with -s or exclude them with -e (these options can be given
          more than once).

        Examples
        --------
        ::

          %psearch a*            -> objects beginning with an a
          %psearch -e builtin a* -> objects NOT in the builtin space starting in a
          %psearch a* function   -> all functions beginning with an a
          %psearch re.e*         -> objects beginning with an e in module re
          %psearch r*.e*         -> objects that start with e in modules starting in r
          %psearch r*.* string   -> all strings in modules beginning with r

        Case sensitive search::

          %psearch -c a*         list all object beginning with lower case a

        Show objects beginning with a single _::

          %psearch -a _*         list objects beginning with a single underscore

        List available objects::

          %psearch -l            list all available object types
        """
        # default namespaces to be searched
        def_search = ['user_local', 'user_global', 'builtin']

        # Process options/args
        opts,args = self.parse_options(parameter_s,'cias:e:l',list_all=True)
        opt = opts.get
        shell = self.shell
        psearch = shell.inspector.psearch
        
        # select list object types
        list_types = False
        if 'l' in opts:
            list_types = True

        # select case options
        if 'i' in opts:
            ignore_case = True
        elif 'c' in opts:
            ignore_case = False
        else:
            ignore_case = not shell.wildcards_case_sensitive

        # Build list of namespaces to search from user options
        def_search.extend(opt('s',[]))
        ns_exclude = ns_exclude=opt('e',[])
        ns_search = [nm for nm in def_search if nm not in ns_exclude]

        # Call the actual search
        try:
            psearch(args,shell.ns_table,ns_search,
                    show_all=opt('a'),ignore_case=ignore_case, list_types=list_types)
        except:
            shell.showtraceback()

    @skip_doctest
    @line_magic
    def who_ls(self, parameter_s=''):
        """Return a sorted list of all interactive variables.

        If arguments are given, only variables of types matching these
        arguments are returned.

        Examples
        --------
        Define two variables and list them with who_ls::

          In [1]: alpha = 123

          In [2]: beta = 'test'

          In [3]: %who_ls
          Out[3]: ['alpha', 'beta']

          In [4]: %who_ls int
          Out[4]: ['alpha']

          In [5]: %who_ls str
          Out[5]: ['beta']
        """

        user_ns = self.shell.user_ns
        user_ns_hidden = self.shell.user_ns_hidden
        nonmatching = object()  # This can never be in user_ns
        out = [ i for i in user_ns
                if not i.startswith('_') \
                and (user_ns[i] is not user_ns_hidden.get(i, nonmatching)) ]

        typelist = parameter_s.split()
        if typelist:
            typeset = set(typelist)
            out = [i for i in out if type(user_ns[i]).__name__ in typeset]

        out.sort()
        return out

    @skip_doctest
    @line_magic
    def who(self, parameter_s=''):
        """Print all interactive variables, with some minimal formatting.

        If any arguments are given, only variables whose type matches one of
        these are printed.  For example::

          %who function str

        will only list functions and strings, excluding all other types of
        variables.  To find the proper type names, simply use type(var) at a
        command line to see how python prints type names.  For example:

        ::

          In [1]: type('hello')\\
          Out[1]: <type 'str'>

        indicates that the type name for strings is 'str'.

        ``%who`` always excludes executed names loaded through your configuration
        file and things which are internal to IPython.

        This is deliberate, as typically you may load many modules and the
        purpose of %who is to show you only what you've manually defined.

        Examples
        --------

        Define two variables and list them with who::

          In [1]: alpha = 123

          In [2]: beta = 'test'

          In [3]: %who
          alpha   beta

          In [4]: %who int
          alpha

          In [5]: %who str
          beta
        """

        varlist = self.who_ls(parameter_s)
        if not varlist:
            if parameter_s:
                print('No variables match your requested type.')
            else:
                print('Interactive namespace is empty.')
            return

        # if we have variables, move on...
        count = 0
        for i in varlist:
            print(i+'\t', end=' ')
            count += 1
            if count > 8:
                count = 0
                print()
        print()

    @skip_doctest
    @line_magic
    def whos(self, parameter_s=''):
        """Like %who, but gives some extra information about each variable.

        The same type filtering of %who can be applied here.

        For all variables, the type is printed. Additionally it prints:

          - For {},[],(): their length.

          - For numpy arrays, a summary with shape, number of
            elements, typecode and size in memory.

          - Everything else: a string representation, snipping their middle if
            too long.

        Examples
        --------
        Define two variables and list them with whos::

          In [1]: alpha = 123

          In [2]: beta = 'test'

          In [3]: %whos
          Variable   Type        Data/Info
          --------------------------------
          alpha      int         123
          beta       str         test
        """

        varnames = self.who_ls(parameter_s)
        if not varnames:
            if parameter_s:
                print('No variables match your requested type.')
            else:
                print('Interactive namespace is empty.')
            return

        # if we have variables, move on...

        # for these types, show len() instead of data:
        seq_types = ['dict', 'list', 'tuple']

        # for numpy arrays, display summary info
        ndarray_type = None
        if 'numpy' in sys.modules:
            try:
                from numpy import ndarray
            except ImportError:
                pass
            else:
                ndarray_type = ndarray.__name__

        # Find all variable names and types so we can figure out column sizes

        # some types are well known and can be shorter
        abbrevs = {'IPython.core.macro.Macro' : 'Macro'}
        def type_name(v):
            tn = type(v).__name__
            return abbrevs.get(tn,tn)

        varlist = [self.shell.user_ns[n] for n in varnames]

        typelist = []
        for vv in varlist:
            tt = type_name(vv)

            if tt=='instance':
                typelist.append( abbrevs.get(str(vv.__class__),
                                             str(vv.__class__)))
            else:
                typelist.append(tt)

        # column labels and # of spaces as separator
        varlabel = 'Variable'
        typelabel = 'Type'
        datalabel = 'Data/Info'
        colsep = 3
        # variable format strings
        vformat    = "{0:<{varwidth}}{1:<{typewidth}}"
        aformat    = "%s: %s elems, type `%s`, %s bytes"
        # find the size of the columns to format the output nicely
        varwidth = max(max(map(len,varnames)), len(varlabel)) + colsep
        typewidth = max(max(map(len,typelist)), len(typelabel)) + colsep
        # table header
        print(varlabel.ljust(varwidth) + typelabel.ljust(typewidth) + \
              ' '+datalabel+'\n' + '-'*(varwidth+typewidth+len(datalabel)+1))
        # and the table itself
        kb = 1024
        Mb = 1048576  # kb**2
        for vname,var,vtype in zip(varnames,varlist,typelist):
            print(vformat.format(vname, vtype, varwidth=varwidth, typewidth=typewidth), end=' ')
            if vtype in seq_types:
                print("n="+str(len(var)))
            elif vtype == ndarray_type:
                vshape = str(var.shape).replace(',','').replace(' ','x')[1:-1]
                if vtype==ndarray_type:
                    # numpy
                    vsize  = var.size
                    vbytes = vsize*var.itemsize
                    vdtype = var.dtype

                if vbytes < 100000:
                    print(aformat % (vshape, vsize, vdtype, vbytes))
                else:
                    print(aformat % (vshape, vsize, vdtype, vbytes), end=' ')
                    if vbytes < Mb:
                        print('(%s kb)' % (vbytes/kb,))
                    else:
                        print('(%s Mb)' % (vbytes/Mb,))
            else:
                try:
                    vstr = str(var)
                except UnicodeEncodeError:
                    vstr = var.encode(DEFAULT_ENCODING,
                                      'backslashreplace')
                except:
                    vstr = "<object with id %d (str() failed)>" % id(var)
                vstr = vstr.replace('\n', '\\n')
                if len(vstr) < 50:
                    print(vstr)
                else:
                    print(vstr[:25] + "<...>" + vstr[-25:])

    @line_magic
    def reset(self, parameter_s=''):
        """Resets the namespace by removing all names defined by the user, if
        called without arguments, or by removing some types of objects, such
        as everything currently in IPython's In[] and Out[] containers (see
        the parameters for details).

        Parameters
        ----------
        -f
            force reset without asking for confirmation.
        -s
            'Soft' reset: Only clears your namespace, leaving history intact.
            References to objects may be kept. By default (without this option),
            we do a 'hard' reset, giving you a new session and removing all
            references to objects from the current session.
        --aggressive
            Try to aggressively remove modules from sys.modules ; this
            may allow you to reimport Python modules that have been updated and
            pick up changes, but can have unintended consequences.

        in
            reset input history
        out
            reset output history
        dhist
            reset directory history
        array
            reset only variables that are NumPy arrays

        See Also
        --------
        reset_selective : invoked as ``%reset_selective``

        Examples
        --------
        ::

          In [6]: a = 1

          In [7]: a
          Out[7]: 1

          In [8]: 'a' in get_ipython().user_ns
          Out[8]: True

          In [9]: %reset -f

          In [1]: 'a' in get_ipython().user_ns
          Out[1]: False

          In [2]: %reset -f in
          Flushing input history

          In [3]: %reset -f dhist in
          Flushing directory history
          Flushing input history

        Notes
        -----
        Calling this magic from clients that do not implement standard input,
        such as the ipython notebook interface, will reset the namespace
        without confirmation.
        """
        opts, args = self.parse_options(parameter_s, "sf", "aggressive", mode="list")
        if "f" in opts:
            ans = True
        else:
            try:
                ans = self.shell.ask_yes_no(
                "Once deleted, variables cannot be recovered. Proceed (y/[n])?",
                default='n')
            except StdinNotImplementedError:
                ans = True
        if not ans:
            print('Nothing done.')
            return

        if 's' in opts:                     # Soft reset
            user_ns = self.shell.user_ns
            for i in self.who_ls():
                del(user_ns[i])
        elif len(args) == 0:                # Hard reset
            self.shell.reset(new_session=False, aggressive=("aggressive" in opts))

        # reset in/out/dhist/array: previously extensinions/clearcmd.py
        ip = self.shell
        user_ns = self.shell.user_ns  # local lookup, heavily used

        for target in args:
            target = target.lower() # make matches case insensitive
            if target == 'out':
                print("Flushing output cache (%d entries)" % len(user_ns['_oh']))
                self.shell.displayhook.flush()

            elif target == 'in':
                print("Flushing input history")
                pc = self.shell.displayhook.prompt_count + 1
                for n in range(1, pc):
                    key = '_i'+repr(n)
                    user_ns.pop(key,None)
                user_ns.update(dict(_i=u'',_ii=u'',_iii=u''))
                hm = ip.history_manager
                # don't delete these, as %save and %macro depending on the
                # length of these lists to be preserved
                hm.input_hist_parsed[:] = [''] * pc
                hm.input_hist_raw[:] = [''] * pc
                # hm has internal machinery for _i,_ii,_iii, clear it out
                hm._i = hm._ii = hm._iii = hm._i00 =  u''

            elif target == 'array':
                # Support cleaning up numpy arrays
                try:
                    from numpy import ndarray
                    # This must be done with items and not iteritems because
                    # we're going to modify the dict in-place.
                    for x,val in list(user_ns.items()):
                        if isinstance(val,ndarray):
                            del user_ns[x]
                except ImportError:
                    print("reset array only works if Numpy is available.")

            elif target == 'dhist':
                print("Flushing directory history")
                del user_ns['_dh'][:]

            else:
                print("Don't know how to reset ", end=' ')
                print(target + ", please run `%reset?` for details")

        gc.collect()

    @line_magic
    def reset_selective(self, parameter_s=''):
        """Resets the namespace by removing names defined by the user.

        Input/Output history are left around in case you need them.

        %reset_selective [-f] regex

        No action is taken if regex is not included

        Options
          -f : force reset without asking for confirmation.

        See Also
        --------
        reset : invoked as ``%reset``

        Examples
        --------
        We first fully reset the namespace so your output looks identical to
        this example for pedagogical reasons; in practice you do not need a
        full reset::

          In [1]: %reset -f

        Now, with a clean namespace we can make a few variables and use
        ``%reset_selective`` to only delete names that match our regexp::

          In [2]: a=1; b=2; c=3; b1m=4; b2m=5; b3m=6; b4m=7; b2s=8

          In [3]: who_ls
          Out[3]: ['a', 'b', 'b1m', 'b2m', 'b2s', 'b3m', 'b4m', 'c']

          In [4]: %reset_selective -f b[2-3]m

          In [5]: who_ls
          Out[5]: ['a', 'b', 'b1m', 'b2s', 'b4m', 'c']

          In [6]: %reset_selective -f d

          In [7]: who_ls
          Out[7]: ['a', 'b', 'b1m', 'b2s', 'b4m', 'c']

          In [8]: %reset_selective -f c

          In [9]: who_ls
          Out[9]: ['a', 'b', 'b1m', 'b2s', 'b4m']

          In [10]: %reset_selective -f b

          In [11]: who_ls
          Out[11]: ['a']

        Notes
        -----
        Calling this magic from clients that do not implement standard input,
        such as the ipython notebook interface, will reset the namespace
        without confirmation.
        """

        opts, regex = self.parse_options(parameter_s,'f')

        if 'f' in opts:
            ans = True
        else:
            try:
                ans = self.shell.ask_yes_no(
                "Once deleted, variables cannot be recovered. Proceed (y/[n])? ",
                default='n')
            except StdinNotImplementedError:
                ans = True
        if not ans:
            print('Nothing done.')
            return
        user_ns = self.shell.user_ns
        if not regex:
            print('No regex pattern specified. Nothing done.')
            return
        else:
            try:
                m = re.compile(regex)
            except TypeError as e:
                raise TypeError('regex must be a string or compiled pattern') from e
            for i in self.who_ls():
                if m.search(i):
                    del(user_ns[i])

    @line_magic
    def xdel(self, parameter_s=''):
        """Delete a variable, trying to clear it from anywhere that
        IPython's machinery has references to it. By default, this uses
        the identity of the named object in the user namespace to remove
        references held under other names. The object is also removed
        from the output history.

        Options
          -n : Delete the specified name from all namespaces, without
          checking their identity.
        """
        opts, varname = self.parse_options(parameter_s,'n')
        try:
            self.shell.del_var(varname, ('n' in opts))
        except (NameError, ValueError) as e:
            print(type(e).__name__ +": "+ str(e))
