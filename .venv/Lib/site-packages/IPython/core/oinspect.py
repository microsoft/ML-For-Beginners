# -*- coding: utf-8 -*-
"""Tools for inspecting Python objects.

Uses syntax highlighting for presenting the various information elements.

Similar in spirit to the inspect module, but all calls take a name argument to
reference the name under which an object is being read.
"""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.

__all__ = ['Inspector','InspectColors']

# stdlib modules
from dataclasses import dataclass
from inspect import signature
from textwrap import dedent
import ast
import html
import inspect
import io as stdlib_io
import linecache
import os
import sys
import types
import warnings

from typing import Any, Optional, Dict, Union, List, Tuple

from typing import TypeAlias

# IPython's own
from IPython.core import page
from IPython.lib.pretty import pretty
from IPython.testing.skipdoctest import skip_doctest
from IPython.utils import PyColorize
from IPython.utils import openpy
from IPython.utils.dir2 import safe_hasattr
from IPython.utils.path import compress_user
from IPython.utils.text import indent
from IPython.utils.wildcard import list_namespace
from IPython.utils.wildcard import typestr2type
from IPython.utils.coloransi import TermColors, ColorScheme, ColorSchemeTable
from IPython.utils.py3compat import cast_unicode
from IPython.utils.colorable import Colorable
from IPython.utils.decorators import undoc

from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter

HOOK_NAME = "__custom_documentations__"


UnformattedBundle: TypeAlias = Dict[str, List[Tuple[str, str]]]  # List of (title, body)
Bundle: TypeAlias = Dict[str, str]


@dataclass
class OInfo:
    ismagic: bool
    isalias: bool
    found: bool
    namespace: Optional[str]
    parent: Any
    obj: Any

    def get(self, field):
        """Get a field from the object for backward compatibility with before 8.12

        see https://github.com/h5py/h5py/issues/2253
        """
        # We need to deprecate this at some point, but the warning will show in completion.
        # Let's comment this for now and uncomment end of 2023 ish
        #        warnings.warn(
        #            f"OInfo dataclass with fields access since IPython 8.12 please use OInfo.{field} instead."
        #            "OInfo used to be a dict but a dataclass provide static fields verification with mypy."
        #            "This warning and backward compatibility `get()` method were added in 8.13.",
        #            DeprecationWarning,
        #            stacklevel=2,
        #        )
        return getattr(self, field)


def pylight(code):
    return highlight(code, PythonLexer(), HtmlFormatter(noclasses=True))

# builtin docstrings to ignore
_func_call_docstring = types.FunctionType.__call__.__doc__
_object_init_docstring = object.__init__.__doc__
_builtin_type_docstrings = {
    inspect.getdoc(t) for t in (types.ModuleType, types.MethodType,
                                types.FunctionType, property)
}

_builtin_func_type = type(all)
_builtin_meth_type = type(str.upper)  # Bound methods have the same type as builtin functions
#****************************************************************************
# Builtin color schemes

Colors = TermColors  # just a shorthand

InspectColors = PyColorize.ANSICodeColors

#****************************************************************************
# Auxiliary functions and objects

# See the messaging spec for the definition of all these fields.  This list
# effectively defines the order of display
info_fields = ['type_name', 'base_class', 'string_form', 'namespace',
               'length', 'file', 'definition', 'docstring', 'source',
               'init_definition', 'class_docstring', 'init_docstring',
               'call_def', 'call_docstring',
               # These won't be printed but will be used to determine how to
               # format the object
               'ismagic', 'isalias', 'isclass', 'found', 'name'
               ]


def object_info(**kw):
    """Make an object info dict with all fields present."""
    infodict = {k:None for k in info_fields}
    infodict.update(kw)
    return infodict


def get_encoding(obj):
    """Get encoding for python source file defining obj

    Returns None if obj is not defined in a sourcefile.
    """
    ofile = find_file(obj)
    # run contents of file through pager starting at line where the object
    # is defined, as long as the file isn't binary and is actually on the
    # filesystem.
    if ofile is None:
        return None
    elif ofile.endswith(('.so', '.dll', '.pyd')):
        return None
    elif not os.path.isfile(ofile):
        return None
    else:
        # Print only text files, not extension binaries.  Note that
        # getsourcelines returns lineno with 1-offset and page() uses
        # 0-offset, so we must adjust.
        with stdlib_io.open(ofile, 'rb') as buffer:   # Tweaked to use io.open for Python 2
            encoding, lines = openpy.detect_encoding(buffer.readline)
        return encoding

def getdoc(obj) -> Union[str,None]:
    """Stable wrapper around inspect.getdoc.

    This can't crash because of attribute problems.

    It also attempts to call a getdoc() method on the given object.  This
    allows objects which provide their docstrings via non-standard mechanisms
    (like Pyro proxies) to still be inspected by ipython's ? system.
    """
    # Allow objects to offer customized documentation via a getdoc method:
    try:
        ds = obj.getdoc()
    except Exception:
        pass
    else:
        if isinstance(ds, str):
            return inspect.cleandoc(ds)
    docstr = inspect.getdoc(obj)
    return docstr


def getsource(obj, oname='') -> Union[str,None]:
    """Wrapper around inspect.getsource.

    This can be modified by other projects to provide customized source
    extraction.

    Parameters
    ----------
    obj : object
        an object whose source code we will attempt to extract
    oname : str
        (optional) a name under which the object is known

    Returns
    -------
    src : unicode or None

    """

    if isinstance(obj, property):
        sources = []
        for attrname in ['fget', 'fset', 'fdel']:
            fn = getattr(obj, attrname)
            if fn is not None:
                encoding = get_encoding(fn)
                oname_prefix = ('%s.' % oname) if oname else ''
                sources.append(''.join(('# ', oname_prefix, attrname)))
                if inspect.isfunction(fn):
                    _src = getsource(fn)
                    if _src:
                        # assert _src is not None, "please mypy"
                        sources.append(dedent(_src))
                else:
                    # Default str/repr only prints function name,
                    # pretty.pretty prints module name too.
                    sources.append(
                        '%s%s = %s\n' % (oname_prefix, attrname, pretty(fn))
                    )
        if sources:
            return '\n'.join(sources)
        else:
            return None

    else:
        # Get source for non-property objects.

        obj = _get_wrapped(obj)

        try:
            src = inspect.getsource(obj)
        except TypeError:
            # The object itself provided no meaningful source, try looking for
            # its class definition instead.
            try:
                src = inspect.getsource(obj.__class__)
            except (OSError, TypeError):
                return None
        except OSError:
            return None

        return src


def is_simple_callable(obj):
    """True if obj is a function ()"""
    return (inspect.isfunction(obj) or inspect.ismethod(obj) or \
            isinstance(obj, _builtin_func_type) or isinstance(obj, _builtin_meth_type))

@undoc
def getargspec(obj):
    """Wrapper around :func:`inspect.getfullargspec`

    In addition to functions and methods, this can also handle objects with a
    ``__call__`` attribute.

    DEPRECATED: Deprecated since 7.10. Do not use, will be removed.
    """

    warnings.warn('`getargspec` function is deprecated as of IPython 7.10'
                  'and will be removed in future versions.', DeprecationWarning, stacklevel=2)

    if safe_hasattr(obj, '__call__') and not is_simple_callable(obj):
        obj = obj.__call__

    return inspect.getfullargspec(obj)

@undoc
def format_argspec(argspec):
    """Format argspect, convenience wrapper around inspect's.

    This takes a dict instead of ordered arguments and calls
    inspect.format_argspec with the arguments in the necessary order.

    DEPRECATED (since 7.10): Do not use; will be removed in future versions.
    """
    
    warnings.warn('`format_argspec` function is deprecated as of IPython 7.10'
                  'and will be removed in future versions.', DeprecationWarning, stacklevel=2)


    return inspect.formatargspec(argspec['args'], argspec['varargs'],
                                 argspec['varkw'], argspec['defaults'])

@undoc
def call_tip(oinfo, format_call=True):
    """DEPRECATED since 6.0. Extract call tip data from an oinfo dict."""
    warnings.warn(
        "`call_tip` function is deprecated as of IPython 6.0"
        "and will be removed in future versions.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Get call definition
    argspec = oinfo.get('argspec')
    if argspec is None:
        call_line = None
    else:
        # Callable objects will have 'self' as their first argument, prune
        # it out if it's there for clarity (since users do *not* pass an
        # extra first argument explicitly).
        try:
            has_self = argspec['args'][0] == 'self'
        except (KeyError, IndexError):
            pass
        else:
            if has_self:
                argspec['args'] = argspec['args'][1:]

        call_line = oinfo['name']+format_argspec(argspec)

    # Now get docstring.
    # The priority is: call docstring, constructor docstring, main one.
    doc = oinfo.get('call_docstring')
    if doc is None:
        doc = oinfo.get('init_docstring')
    if doc is None:
        doc = oinfo.get('docstring','')

    return call_line, doc


def _get_wrapped(obj):
    """Get the original object if wrapped in one or more @decorators

    Some objects automatically construct similar objects on any unrecognised
    attribute access (e.g. unittest.mock.call). To protect against infinite loops,
    this will arbitrarily cut off after 100 levels of obj.__wrapped__
    attribute access. --TK, Jan 2016
    """
    orig_obj = obj
    i = 0
    while safe_hasattr(obj, '__wrapped__'):
        obj = obj.__wrapped__
        i += 1
        if i > 100:
            # __wrapped__ is probably a lie, so return the thing we started with
            return orig_obj
    return obj

def find_file(obj) -> str:
    """Find the absolute path to the file where an object was defined.

    This is essentially a robust wrapper around `inspect.getabsfile`.

    Returns None if no file can be found.

    Parameters
    ----------
    obj : any Python object

    Returns
    -------
    fname : str
        The absolute path to the file where the object was defined.
    """
    obj = _get_wrapped(obj)

    fname = None
    try:
        fname = inspect.getabsfile(obj)
    except TypeError:
        # For an instance, the file that matters is where its class was
        # declared.
        try:
            fname = inspect.getabsfile(obj.__class__)
        except (OSError, TypeError):
            # Can happen for builtins
            pass
    except OSError:
        pass

    return cast_unicode(fname)


def find_source_lines(obj):
    """Find the line number in a file where an object was defined.

    This is essentially a robust wrapper around `inspect.getsourcelines`.

    Returns None if no file can be found.

    Parameters
    ----------
    obj : any Python object

    Returns
    -------
    lineno : int
        The line number where the object definition starts.
    """
    obj = _get_wrapped(obj)

    try:
        lineno = inspect.getsourcelines(obj)[1]
    except TypeError:
        # For instances, try the class object like getsource() does
        try:
            lineno = inspect.getsourcelines(obj.__class__)[1]
        except (OSError, TypeError):
            return None
    except OSError:
        return None

    return lineno

class Inspector(Colorable):

    def __init__(self, color_table=InspectColors,
                 code_color_table=PyColorize.ANSICodeColors,
                 scheme=None,
                 str_detail_level=0,
                 parent=None, config=None):
        super(Inspector, self).__init__(parent=parent, config=config)
        self.color_table = color_table
        self.parser = PyColorize.Parser(out='str', parent=self, style=scheme)
        self.format = self.parser.format
        self.str_detail_level = str_detail_level
        self.set_active_scheme(scheme)

    def _getdef(self,obj,oname='') -> Union[str,None]:
        """Return the call signature for any callable object.

        If any exception is generated, None is returned instead and the
        exception is suppressed."""
        if not callable(obj):
            return None
        try:
            return _render_signature(signature(obj), oname)
        except:
            return None

    def __head(self,h) -> str:
        """Return a header string with proper colors."""
        return '%s%s%s' % (self.color_table.active_colors.header,h,
                           self.color_table.active_colors.normal)

    def set_active_scheme(self, scheme):
        if scheme is not None:
            self.color_table.set_active_scheme(scheme)
            self.parser.color_table.set_active_scheme(scheme)

    def noinfo(self, msg, oname):
        """Generic message when no information is found."""
        print('No %s found' % msg, end=' ')
        if oname:
            print('for %s' % oname)
        else:
            print()

    def pdef(self, obj, oname=''):
        """Print the call signature for any callable object.

        If the object is a class, print the constructor information."""

        if not callable(obj):
            print('Object is not callable.')
            return

        header = ''

        if inspect.isclass(obj):
            header = self.__head('Class constructor information:\n')


        output = self._getdef(obj,oname)
        if output is None:
            self.noinfo('definition header',oname)
        else:
            print(header,self.format(output), end=' ')

    # In Python 3, all classes are new-style, so they all have __init__.
    @skip_doctest
    def pdoc(self, obj, oname='', formatter=None):
        """Print the docstring for any object.

        Optional:
        -formatter: a function to run the docstring through for specially
        formatted docstrings.

        Examples
        --------
        In [1]: class NoInit:
           ...:     pass

        In [2]: class NoDoc:
           ...:     def __init__(self):
           ...:         pass

        In [3]: %pdoc NoDoc
        No documentation found for NoDoc

        In [4]: %pdoc NoInit
        No documentation found for NoInit

        In [5]: obj = NoInit()

        In [6]: %pdoc obj
        No documentation found for obj

        In [5]: obj2 = NoDoc()

        In [6]: %pdoc obj2
        No documentation found for obj2
        """

        head = self.__head  # For convenience
        lines = []
        ds = getdoc(obj)
        if formatter:
            ds = formatter(ds).get('plain/text', ds)
        if ds:
            lines.append(head("Class docstring:"))
            lines.append(indent(ds))
        if inspect.isclass(obj) and hasattr(obj, '__init__'):
            init_ds = getdoc(obj.__init__)
            if init_ds is not None:
                lines.append(head("Init docstring:"))
                lines.append(indent(init_ds))
        elif hasattr(obj,'__call__'):
            call_ds = getdoc(obj.__call__)
            if call_ds:
                lines.append(head("Call docstring:"))
                lines.append(indent(call_ds))

        if not lines:
            self.noinfo('documentation',oname)
        else:
            page.page('\n'.join(lines))

    def psource(self, obj, oname=''):
        """Print the source code for an object."""

        # Flush the source cache because inspect can return out-of-date source
        linecache.checkcache()
        try:
            src = getsource(obj, oname=oname)
        except Exception:
            src = None

        if src is None:
            self.noinfo('source', oname)
        else:
            page.page(self.format(src))

    def pfile(self, obj, oname=''):
        """Show the whole file where an object was defined."""

        lineno = find_source_lines(obj)
        if lineno is None:
            self.noinfo('file', oname)
            return

        ofile = find_file(obj)
        # run contents of file through pager starting at line where the object
        # is defined, as long as the file isn't binary and is actually on the
        # filesystem.
        if ofile.endswith(('.so', '.dll', '.pyd')):
            print('File %r is binary, not printing.' % ofile)
        elif not os.path.isfile(ofile):
            print('File %r does not exist, not printing.' % ofile)
        else:
            # Print only text files, not extension binaries.  Note that
            # getsourcelines returns lineno with 1-offset and page() uses
            # 0-offset, so we must adjust.
            page.page(self.format(openpy.read_py_file(ofile, skip_encoding_cookie=False)), lineno - 1)


    def _mime_format(self, text:str, formatter=None) -> dict:
        """Return a mime bundle representation of the input text.

        - if `formatter` is None, the returned mime bundle has
           a ``text/plain`` field, with the input text.
           a ``text/html`` field with a ``<pre>`` tag containing the input text.

        - if ``formatter`` is not None, it must be a callable transforming the
          input text into a mime bundle. Default values for ``text/plain`` and
          ``text/html`` representations are the ones described above.

        Note:

        Formatters returning strings are supported but this behavior is deprecated.

        """
        defaults = {
            "text/plain": text,
            "text/html": f"<pre>{html.escape(text)}</pre>",
        }

        if formatter is None:
            return defaults
        else:
            formatted = formatter(text)

            if not isinstance(formatted, dict):
                # Handle the deprecated behavior of a formatter returning
                # a string instead of a mime bundle.
                return {"text/plain": formatted, "text/html": f"<pre>{formatted}</pre>"}

            else:
                return dict(defaults, **formatted)

    def format_mime(self, bundle: UnformattedBundle) -> Bundle:
        """Format a mimebundle being created by _make_info_unformatted into a real mimebundle"""
        # Format text/plain mimetype
        assert isinstance(bundle["text/plain"], list)
        for item in bundle["text/plain"]:
            assert isinstance(item, tuple)

        new_b: Bundle = {}
        lines = []
        _len = max(len(h) for h, _ in bundle["text/plain"])

        for head, body in bundle["text/plain"]:
            body = body.strip("\n")
            delim = "\n" if "\n" in body else " "
            lines.append(
                f"{self.__head(head+':')}{(_len - len(head))*' '}{delim}{body}"
            )

        new_b["text/plain"] = "\n".join(lines)

        if "text/html" in bundle:
            assert isinstance(bundle["text/html"], list)
            for item in bundle["text/html"]:
                assert isinstance(item, tuple)
            # Format the text/html mimetype
            if isinstance(bundle["text/html"], (list, tuple)):
                # bundle['text/html'] is a list of (head, formatted body) pairs
                new_b["text/html"] = "\n".join(
                    (f"<h1>{head}</h1>\n{body}" for (head, body) in bundle["text/html"])
                )

        for k in bundle.keys():
            if k in ("text/html", "text/plain"):
                continue
            else:
                new_b = bundle[k]  # type:ignore
        return new_b

    def _append_info_field(
        self,
        bundle: UnformattedBundle,
        title: str,
        key: str,
        info,
        omit_sections,
        formatter,
    ):
        """Append an info value to the unformatted mimebundle being constructed by _make_info_unformatted"""
        if title in omit_sections or key in omit_sections:
            return
        field = info[key]
        if field is not None:
            formatted_field = self._mime_format(field, formatter)
            bundle["text/plain"].append((title, formatted_field["text/plain"]))
            bundle["text/html"].append((title, formatted_field["text/html"]))

    def _make_info_unformatted(
        self, obj, info, formatter, detail_level, omit_sections
    ) -> UnformattedBundle:
        """Assemble the mimebundle as unformatted lists of information"""
        bundle: UnformattedBundle = {
            "text/plain": [],
            "text/html": [],
        }

        # A convenience function to simplify calls below
        def append_field(
            bundle: UnformattedBundle, title: str, key: str, formatter=None
        ):
            self._append_info_field(
                bundle,
                title=title,
                key=key,
                info=info,
                omit_sections=omit_sections,
                formatter=formatter,
            )

        def code_formatter(text) -> Bundle:
            return {
                'text/plain': self.format(text),
                'text/html': pylight(text)
            }

        if info["isalias"]:
            append_field(bundle, "Repr", "string_form")

        elif info['ismagic']:
            if detail_level > 0:
                append_field(bundle, "Source", "source", code_formatter)
            else:
                append_field(bundle, "Docstring", "docstring", formatter)
            append_field(bundle, "File", "file")

        elif info['isclass'] or is_simple_callable(obj):
            # Functions, methods, classes
            append_field(bundle, "Signature", "definition", code_formatter)
            append_field(bundle, "Init signature", "init_definition", code_formatter)
            append_field(bundle, "Docstring", "docstring", formatter)
            if detail_level > 0 and info["source"]:
                append_field(bundle, "Source", "source", code_formatter)
            else:
                append_field(bundle, "Init docstring", "init_docstring", formatter)

            append_field(bundle, "File", "file")
            append_field(bundle, "Type", "type_name")
            append_field(bundle, "Subclasses", "subclasses")

        else:
            # General Python objects
            append_field(bundle, "Signature", "definition", code_formatter)
            append_field(bundle, "Call signature", "call_def", code_formatter)
            append_field(bundle, "Type", "type_name")
            append_field(bundle, "String form", "string_form")

            # Namespace
            if info["namespace"] != "Interactive":
                append_field(bundle, "Namespace", "namespace")

            append_field(bundle, "Length", "length")
            append_field(bundle, "File", "file")

            # Source or docstring, depending on detail level and whether
            # source found.
            if detail_level > 0 and info["source"]:
                append_field(bundle, "Source", "source", code_formatter)
            else:
                append_field(bundle, "Docstring", "docstring", formatter)

            append_field(bundle, "Class docstring", "class_docstring", formatter)
            append_field(bundle, "Init docstring", "init_docstring", formatter)
            append_field(bundle, "Call docstring", "call_docstring", formatter)
        return bundle


    def _get_info(
        self,
        obj: Any,
        oname: str = "",
        formatter=None,
        info: Optional[OInfo] = None,
        detail_level=0,
        omit_sections=(),
    ) -> Bundle:
        """Retrieve an info dict and format it.

        Parameters
        ----------
        obj : any
            Object to inspect and return info from
        oname : str (default: ''):
            Name of the variable pointing to `obj`.
        formatter : callable
        info
            already computed information
        detail_level : integer
            Granularity of detail level, if set to 1, give more information.
        omit_sections : container[str]
            Titles or keys to omit from output (can be set, tuple, etc., anything supporting `in`)
        """

        info_dict = self.info(obj, oname=oname, info=info, detail_level=detail_level)
        bundle = self._make_info_unformatted(
            obj,
            info_dict,
            formatter,
            detail_level=detail_level,
            omit_sections=omit_sections,
        )
        return self.format_mime(bundle)

    def pinfo(
        self,
        obj,
        oname="",
        formatter=None,
        info: Optional[OInfo] = None,
        detail_level=0,
        enable_html_pager=True,
        omit_sections=(),
    ):
        """Show detailed information about an object.

        Optional arguments:

        - oname: name of the variable pointing to the object.

        - formatter: callable (optional)
              A special formatter for docstrings.

              The formatter is a callable that takes a string as an input
              and returns either a formatted string or a mime type bundle
              in the form of a dictionary.

              Although the support of custom formatter returning a string
              instead of a mime type bundle is deprecated.

        - info: a structure with some information fields which may have been
          precomputed already.

        - detail_level: if set to 1, more information is given.

        - omit_sections: set of section keys and titles to omit
        """
        assert info is not None
        info_b: Bundle = self._get_info(
            obj, oname, formatter, info, detail_level, omit_sections=omit_sections
        )
        if not enable_html_pager:
            del info_b["text/html"]
        page.page(info_b)

    def _info(self, obj, oname="", info=None, detail_level=0):
        """
        Inspector.info() was likely improperly marked as deprecated
        while only a parameter was deprecated. We "un-deprecate" it.
        """

        warnings.warn(
            "The `Inspector.info()` method has been un-deprecated as of 8.0 "
            "and the `formatter=` keyword removed. `Inspector._info` is now "
            "an alias, and you can just call `.info()` directly.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.info(obj, oname=oname, info=info, detail_level=detail_level)

    def info(self, obj, oname="", info=None, detail_level=0) -> Dict[str, Any]:
        """Compute a dict with detailed information about an object.

        Parameters
        ----------
        obj : any
            An object to find information about
        oname : str (default: '')
            Name of the variable pointing to `obj`.
        info : (default: None)
            A struct (dict like with attr access) with some information fields
            which may have been precomputed already.
        detail_level : int (default:0)
            If set to 1, more information is given.

        Returns
        -------
        An object info dict with known fields from `info_fields`. Keys are
        strings, values are string or None.
        """

        if info is None:
            ismagic = False
            isalias = False
            ospace = ''
        else:
            ismagic = info.ismagic
            isalias = info.isalias
            ospace = info.namespace

        # Get docstring, special-casing aliases:
        att_name = oname.split(".")[-1]
        parents_docs = None
        prelude = ""
        if info and info.parent is not None and hasattr(info.parent, HOOK_NAME):
            parents_docs_dict = getattr(info.parent, HOOK_NAME)
            parents_docs = parents_docs_dict.get(att_name, None)
        out = dict(
            name=oname, found=True, isalias=isalias, ismagic=ismagic, subclasses=None
        )

        if parents_docs:
            ds = parents_docs
        elif isalias:
            if not callable(obj):
                try:
                    ds = "Alias to the system command:\n  %s" % obj[1]
                except:
                    ds = "Alias: " + str(obj)
            else:
                ds = "Alias to " + str(obj)
                if obj.__doc__:
                    ds += "\nDocstring:\n" + obj.__doc__
        else:
            ds_or_None = getdoc(obj)
            if ds_or_None is None:
                ds = '<no docstring>'
            else:
                ds = ds_or_None

        ds = prelude + ds

        # store output in a dict, we initialize it here and fill it as we go

        string_max = 200 # max size of strings to show (snipped if longer)
        shalf = int((string_max - 5) / 2)

        if ismagic:
            out['type_name'] = 'Magic function'
        elif isalias:
            out['type_name'] = 'System alias'
        else:
            out['type_name'] = type(obj).__name__

        try:
            bclass = obj.__class__
            out['base_class'] = str(bclass)
        except:
            pass

        # String form, but snip if too long in ? form (full in ??)
        if detail_level >= self.str_detail_level:
            try:
                ostr = str(obj)
                str_head = 'string_form'
                if not detail_level and len(ostr)>string_max:
                    ostr = ostr[:shalf] + ' <...> ' + ostr[-shalf:]
                    ostr = ("\n" + " " * len(str_head.expandtabs())).\
                            join(q.strip() for q in ostr.split("\n"))
                out[str_head] = ostr
            except:
                pass

        if ospace:
            out['namespace'] = ospace

        # Length (for strings and lists)
        try:
            out['length'] = str(len(obj))
        except Exception:
            pass

        # Filename where object was defined
        binary_file = False
        fname = find_file(obj)
        if fname is None:
            # if anything goes wrong, we don't want to show source, so it's as
            # if the file was binary
            binary_file = True
        else:
            if fname.endswith(('.so', '.dll', '.pyd')):
                binary_file = True
            elif fname.endswith('<string>'):
                fname = 'Dynamically generated function. No source code available.'
            out['file'] = compress_user(fname)

        # Original source code for a callable, class or property.
        if detail_level:
            # Flush the source cache because inspect can return out-of-date
            # source
            linecache.checkcache()
            try:
                if isinstance(obj, property) or not binary_file:
                    src = getsource(obj, oname)
                    if src is not None:
                        src = src.rstrip()
                    out['source'] = src

            except Exception:
                pass

        # Add docstring only if no source is to be shown (avoid repetitions).
        if ds and not self._source_contains_docstring(out.get('source'), ds):
            out['docstring'] = ds

        # Constructor docstring for classes
        if inspect.isclass(obj):
            out['isclass'] = True

            # get the init signature:
            try:
                init_def = self._getdef(obj, oname)
            except AttributeError:
                init_def = None

            # get the __init__ docstring
            try:
                obj_init = obj.__init__
            except AttributeError:
                init_ds = None
            else:
                if init_def is None:
                    # Get signature from init if top-level sig failed.
                    # Can happen for built-in types (list, etc.).
                    try:
                        init_def = self._getdef(obj_init, oname)
                    except AttributeError:
                        pass
                init_ds = getdoc(obj_init)
                # Skip Python's auto-generated docstrings
                if init_ds == _object_init_docstring:
                    init_ds = None

            if init_def:
                out['init_definition'] = init_def

            if init_ds:
                out['init_docstring'] = init_ds

            names = [sub.__name__ for sub in type.__subclasses__(obj)]
            if len(names) < 10:
                all_names = ', '.join(names)
            else:
                all_names = ', '.join(names[:10]+['...'])
            out['subclasses'] = all_names
        # and class docstring for instances:
        else:
            # reconstruct the function definition and print it:
            defln = self._getdef(obj, oname)
            if defln:
                out['definition'] = defln

            # First, check whether the instance docstring is identical to the
            # class one, and print it separately if they don't coincide.  In
            # most cases they will, but it's nice to print all the info for
            # objects which use instance-customized docstrings.
            if ds:
                try:
                    cls = getattr(obj,'__class__')
                except:
                    class_ds = None
                else:
                    class_ds = getdoc(cls)
                # Skip Python's auto-generated docstrings
                if class_ds in _builtin_type_docstrings:
                    class_ds = None
                if class_ds and ds != class_ds:
                    out['class_docstring'] = class_ds

            # Next, try to show constructor docstrings
            try:
                init_ds = getdoc(obj.__init__)
                # Skip Python's auto-generated docstrings
                if init_ds == _object_init_docstring:
                    init_ds = None
            except AttributeError:
                init_ds = None
            if init_ds:
                out['init_docstring'] = init_ds

            # Call form docstring for callable instances
            if safe_hasattr(obj, '__call__') and not is_simple_callable(obj):
                call_def = self._getdef(obj.__call__, oname)
                if call_def and (call_def != out.get('definition')):
                    # it may never be the case that call def and definition differ,
                    # but don't include the same signature twice
                    out['call_def'] = call_def
                call_ds = getdoc(obj.__call__)
                # Skip Python's auto-generated docstrings
                if call_ds == _func_call_docstring:
                    call_ds = None
                if call_ds:
                    out['call_docstring'] = call_ds

        return object_info(**out)

    @staticmethod
    def _source_contains_docstring(src, doc):
        """
        Check whether the source *src* contains the docstring *doc*.

        This is is helper function to skip displaying the docstring if the
        source already contains it, avoiding repetition of information.
        """
        try:
            (def_node,) = ast.parse(dedent(src)).body
            return ast.get_docstring(def_node) == doc  # type: ignore[arg-type]
        except Exception:
            # The source can become invalid or even non-existent (because it
            # is re-fetched from the source file) so the above code fail in
            # arbitrary ways.
            return False

    def psearch(self,pattern,ns_table,ns_search=[],
                ignore_case=False,show_all=False, *, list_types=False):
        """Search namespaces with wildcards for objects.

        Arguments:

        - pattern: string containing shell-like wildcards to use in namespace
          searches and optionally a type specification to narrow the search to
          objects of that type.

        - ns_table: dict of name->namespaces for search.

        Optional arguments:

          - ns_search: list of namespace names to include in search.

          - ignore_case(False): make the search case-insensitive.

          - show_all(False): show all names, including those starting with
            underscores.

          - list_types(False): list all available object types for object matching.
        """
        #print 'ps pattern:<%r>' % pattern # dbg

        # defaults
        type_pattern = 'all'
        filter = ''

        # list all object types
        if list_types:
            page.page('\n'.join(sorted(typestr2type)))
            return

        cmds = pattern.split()
        len_cmds  =  len(cmds)
        if len_cmds == 1:
            # Only filter pattern given
            filter = cmds[0]
        elif len_cmds == 2:
            # Both filter and type specified
            filter,type_pattern = cmds
        else:
            raise ValueError('invalid argument string for psearch: <%s>' %
                             pattern)

        # filter search namespaces
        for name in ns_search:
            if name not in ns_table:
                raise ValueError('invalid namespace <%s>. Valid names: %s' %
                                 (name,ns_table.keys()))

        #print 'type_pattern:',type_pattern # dbg
        search_result, namespaces_seen = set(), set()
        for ns_name in ns_search:
            ns = ns_table[ns_name]
            # Normally, locals and globals are the same, so we just check one.
            if id(ns) in namespaces_seen:
                continue
            namespaces_seen.add(id(ns))
            tmp_res = list_namespace(ns, type_pattern, filter,
                                    ignore_case=ignore_case, show_all=show_all)
            search_result.update(tmp_res)

        page.page('\n'.join(sorted(search_result)))


def _render_signature(obj_signature, obj_name) -> str:
    """
    This was mostly taken from inspect.Signature.__str__.
    Look there for the comments.
    The only change is to add linebreaks when this gets too long.
    """
    result = []
    pos_only = False
    kw_only = True
    for param in obj_signature.parameters.values():
        if param.kind == inspect.Parameter.POSITIONAL_ONLY:
            pos_only = True
        elif pos_only:
            result.append('/')
            pos_only = False

        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            kw_only = False
        elif param.kind == inspect.Parameter.KEYWORD_ONLY and kw_only:
            result.append('*')
            kw_only = False

        result.append(str(param))

    if pos_only:
        result.append('/')

    # add up name, parameters, braces (2), and commas
    if len(obj_name) + sum(len(r) + 2 for r in result) > 75:
        # This doesn’t fit behind “Signature: ” in an inspect window.
        rendered = '{}(\n{})'.format(obj_name, ''.join(
            '    {},\n'.format(r) for r in result)
        )
    else:
        rendered = '{}({})'.format(obj_name, ', '.join(result))

    if obj_signature.return_annotation is not inspect._empty:
        anno = inspect.formatannotation(obj_signature.return_annotation)
        rendered += ' -> {}'.format(anno)

    return rendered
