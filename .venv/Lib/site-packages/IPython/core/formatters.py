# -*- coding: utf-8 -*-
"""Display formatters.

Inheritance diagram:

.. inheritance-diagram:: IPython.core.formatters
   :parts: 3
"""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.

import abc
import sys
import traceback
import warnings
from io import StringIO

from decorator import decorator

from traitlets.config.configurable import Configurable
from .getipython import get_ipython
from ..utils.sentinel import Sentinel
from ..utils.dir2 import get_real_method
from ..lib import pretty
from traitlets import (
    Bool, Dict, Integer, Unicode, CUnicode, ObjectName, List,
    ForwardDeclaredInstance,
    default, observe,
)

from typing import Any


class DisplayFormatter(Configurable):

    active_types = List(Unicode(),
        help="""List of currently active mime-types to display.
        You can use this to set a white-list for formats to display.
        
        Most users will not need to change this value.
        """).tag(config=True)

    @default('active_types')
    def _active_types_default(self):
        return self.format_types

    @observe('active_types')
    def _active_types_changed(self, change):
        for key, formatter in self.formatters.items():
            if key in change['new']:
                formatter.enabled = True
            else:
                formatter.enabled = False

    ipython_display_formatter = ForwardDeclaredInstance('FormatterABC')
    @default('ipython_display_formatter')
    def _default_formatter(self):
        return IPythonDisplayFormatter(parent=self)

    mimebundle_formatter = ForwardDeclaredInstance('FormatterABC')
    @default('mimebundle_formatter')
    def _default_mime_formatter(self):
        return MimeBundleFormatter(parent=self)

    # A dict of formatter whose keys are format types (MIME types) and whose
    # values are subclasses of BaseFormatter.
    formatters = Dict()
    @default('formatters')
    def _formatters_default(self):
        """Activate the default formatters."""
        formatter_classes = [
            PlainTextFormatter,
            HTMLFormatter,
            MarkdownFormatter,
            SVGFormatter,
            PNGFormatter,
            PDFFormatter,
            JPEGFormatter,
            LatexFormatter,
            JSONFormatter,
            JavascriptFormatter
        ]
        d = {}
        for cls in formatter_classes:
            f = cls(parent=self)
            d[f.format_type] = f
        return d

    def format(self, obj, include=None, exclude=None):
        """Return a format data dict for an object.

        By default all format types will be computed.

        The following MIME types are usually implemented:

        * text/plain
        * text/html
        * text/markdown
        * text/latex
        * application/json
        * application/javascript
        * application/pdf
        * image/png
        * image/jpeg
        * image/svg+xml

        Parameters
        ----------
        obj : object
            The Python object whose format data will be computed.
        include : list, tuple or set; optional
            A list of format type strings (MIME types) to include in the
            format data dict. If this is set *only* the format types included
            in this list will be computed.
        exclude : list, tuple or set; optional
            A list of format type string (MIME types) to exclude in the format
            data dict. If this is set all format types will be computed,
            except for those included in this argument.
            Mimetypes present in exclude will take precedence over the ones in include

        Returns
        -------
        (format_dict, metadata_dict) : tuple of two dicts
            format_dict is a dictionary of key/value pairs, one of each format that was
            generated for the object. The keys are the format types, which
            will usually be MIME type strings and the values and JSON'able
            data structure containing the raw data for the representation in
            that format.

            metadata_dict is a dictionary of metadata about each mime-type output.
            Its keys will be a strict subset of the keys in format_dict.

        Notes
        -----
            If an object implement `_repr_mimebundle_` as well as various
            `_repr_*_`, the data returned by `_repr_mimebundle_` will take
            precedence and the corresponding `_repr_*_` for this mimetype will
            not be called.

        """
        format_dict = {}
        md_dict = {}
        
        if self.ipython_display_formatter(obj):
            # object handled itself, don't proceed
            return {}, {}

        format_dict, md_dict = self.mimebundle_formatter(obj, include=include, exclude=exclude)

        if format_dict or md_dict:
            if include:
                format_dict = {k:v for k,v in format_dict.items() if k in include}
                md_dict = {k:v for k,v in md_dict.items() if k in include}
            if exclude:
                format_dict = {k:v for k,v in format_dict.items() if k not in exclude}
                md_dict = {k:v for k,v in md_dict.items() if k not in exclude}

        for format_type, formatter in self.formatters.items():
            if format_type in format_dict:
                # already got it from mimebundle, maybe don't render again.
                # exception: manually registered per-mime renderer
                # check priority:
                # 1. user-registered per-mime formatter
                # 2. mime-bundle (user-registered or repr method)
                # 3. default per-mime formatter (e.g. repr method)
                try:
                    formatter.lookup(obj)
                except KeyError:
                    # no special formatter, use mime-bundle-provided value
                    continue
            if include and format_type not in include:
                continue
            if exclude and format_type in exclude:
                continue
            
            md = None
            try:
                data = formatter(obj)
            except:
                # FIXME: log the exception
                raise
            
            # formatters can return raw data or (data, metadata)
            if isinstance(data, tuple) and len(data) == 2:
                data, md = data
            
            if data is not None:
                format_dict[format_type] = data
            if md is not None:
                md_dict[format_type] = md
        return format_dict, md_dict

    @property
    def format_types(self):
        """Return the format types (MIME types) of the active formatters."""
        return list(self.formatters.keys())


#-----------------------------------------------------------------------------
# Formatters for specific format types (text, html, svg, etc.)
#-----------------------------------------------------------------------------


def _safe_repr(obj):
    """Try to return a repr of an object

    always returns a string, at least.
    """
    try:
        return repr(obj)
    except Exception as e:
        return "un-repr-able object (%r)" % e


class FormatterWarning(UserWarning):
    """Warning class for errors in formatters"""

@decorator
def catch_format_error(method, self, *args, **kwargs):
    """show traceback on failed format call"""
    try:
        r = method(self, *args, **kwargs)
    except NotImplementedError:
        # don't warn on NotImplementedErrors
        return self._check_return(None, args[0])
    except Exception:
        exc_info = sys.exc_info()
        ip = get_ipython()
        if ip is not None:
            ip.showtraceback(exc_info)
        else:
            traceback.print_exception(*exc_info)
        return self._check_return(None, args[0])
    return self._check_return(r, args[0])


class FormatterABC(metaclass=abc.ABCMeta):
    """ Abstract base class for Formatters.

    A formatter is a callable class that is responsible for computing the
    raw format data for a particular format type (MIME type). For example,
    an HTML formatter would have a format type of `text/html` and would return
    the HTML representation of the object when called.
    """

    # The format type of the data returned, usually a MIME type.
    format_type = 'text/plain'

    # Is the formatter enabled...
    enabled = True
    
    @abc.abstractmethod
    def __call__(self, obj):
        """Return a JSON'able representation of the object.

        If the object cannot be formatted by this formatter,
        warn and return None.
        """
        return repr(obj)


def _mod_name_key(typ):
    """Return a (__module__, __name__) tuple for a type.

    Used as key in Formatter.deferred_printers.
    """
    module = getattr(typ, '__module__', None)
    name = getattr(typ, '__name__', None)
    return (module, name)


def _get_type(obj):
    """Return the type of an instance (old and new-style)"""
    return getattr(obj, '__class__', None) or type(obj)


_raise_key_error = Sentinel('_raise_key_error', __name__, 
"""
Special value to raise a KeyError

Raise KeyError in `BaseFormatter.pop` if passed as the default value to `pop`
""")


class BaseFormatter(Configurable):
    """A base formatter class that is configurable.

    This formatter should usually be used as the base class of all formatters.
    It is a traited :class:`Configurable` class and includes an extensible
    API for users to determine how their objects are formatted. The following
    logic is used to find a function to format an given object.

    1. The object is introspected to see if it has a method with the name
       :attr:`print_method`. If is does, that object is passed to that method
       for formatting.
    2. If no print method is found, three internal dictionaries are consulted
       to find print method: :attr:`singleton_printers`, :attr:`type_printers`
       and :attr:`deferred_printers`.

    Users should use these dictionaries to register functions that will be
    used to compute the format data for their objects (if those objects don't
    have the special print methods). The easiest way of using these
    dictionaries is through the :meth:`for_type` and :meth:`for_type_by_name`
    methods.

    If no function/callable is found to compute the format data, ``None`` is
    returned and this format type is not used.
    """

    format_type = Unicode("text/plain")
    _return_type: Any = str

    enabled = Bool(True).tag(config=True)

    print_method = ObjectName('__repr__')

    # The singleton printers.
    # Maps the IDs of the builtin singleton objects to the format functions.
    singleton_printers = Dict().tag(config=True)

    # The type-specific printers.
    # Map type objects to the format functions.
    type_printers = Dict().tag(config=True)

    # The deferred-import type-specific printers.
    # Map (modulename, classname) pairs to the format functions.
    deferred_printers = Dict().tag(config=True)
    
    @catch_format_error
    def __call__(self, obj):
        """Compute the format for an object."""
        if self.enabled:
            # lookup registered printer
            try:
                printer = self.lookup(obj)
            except KeyError:
                pass
            else:
                return printer(obj)
            # Finally look for special method names
            method = get_real_method(obj, self.print_method)
            if method is not None:
                return method()
            return None
        else:
            return None
    
    def __contains__(self, typ):
        """map in to lookup_by_type"""
        try:
            self.lookup_by_type(typ)
        except KeyError:
            return False
        else:
            return True
    
    def _check_return(self, r, obj):
        """Check that a return value is appropriate

        Return the value if so, None otherwise, warning if invalid.
        """
        if r is None or isinstance(r, self._return_type) or \
            (isinstance(r, tuple) and r and isinstance(r[0], self._return_type)):
            return r
        else:
            warnings.warn(
                "%s formatter returned invalid type %s (expected %s) for object: %s" % \
                (self.format_type, type(r), self._return_type, _safe_repr(obj)),
                FormatterWarning
            )
    
    def lookup(self, obj):
        """Look up the formatter for a given instance.

        Parameters
        ----------
        obj : object instance

        Returns
        -------
        f : callable
            The registered formatting callable for the type.

        Raises
        ------
        KeyError if the type has not been registered.
        """
        # look for singleton first
        obj_id = id(obj)
        if obj_id in self.singleton_printers:
            return self.singleton_printers[obj_id]
        # then lookup by type
        return self.lookup_by_type(_get_type(obj))
    
    def lookup_by_type(self, typ):
        """Look up the registered formatter for a type.

        Parameters
        ----------
        typ : type or '__module__.__name__' string for a type

        Returns
        -------
        f : callable
            The registered formatting callable for the type.

        Raises
        ------
        KeyError if the type has not been registered.
        """
        if isinstance(typ, str):
            typ_key = tuple(typ.rsplit('.',1))
            if typ_key not in self.deferred_printers:
                # We may have it cached in the type map. We will have to
                # iterate over all of the types to check.
                for cls in self.type_printers:
                    if _mod_name_key(cls) == typ_key:
                        return self.type_printers[cls]
            else:
                return self.deferred_printers[typ_key]
        else:
            for cls in pretty._get_mro(typ):
                if cls in self.type_printers or self._in_deferred_types(cls):
                    return self.type_printers[cls]
        
        # If we have reached here, the lookup failed.
        raise KeyError("No registered printer for {0!r}".format(typ))

    def for_type(self, typ, func=None):
        """Add a format function for a given type.

        Parameters
        ----------
        typ : type or '__module__.__name__' string for a type
            The class of the object that will be formatted using `func`.

        func : callable
            A callable for computing the format data.
            `func` will be called with the object to be formatted,
            and will return the raw data in this formatter's format.
            Subclasses may use a different call signature for the
            `func` argument.

            If `func` is None or not specified, there will be no change,
            only returning the current value.

        Returns
        -------
        oldfunc : callable
            The currently registered callable.
            If you are registering a new formatter,
            this will be the previous value (to enable restoring later).
        """
        # if string given, interpret as 'pkg.module.class_name'
        if isinstance(typ, str):
            type_module, type_name = typ.rsplit('.', 1)
            return self.for_type_by_name(type_module, type_name, func)
        
        try:
            oldfunc = self.lookup_by_type(typ)
        except KeyError:
            oldfunc = None
        
        if func is not None:
            self.type_printers[typ] = func
        
        return oldfunc

    def for_type_by_name(self, type_module, type_name, func=None):
        """Add a format function for a type specified by the full dotted
        module and name of the type, rather than the type of the object.

        Parameters
        ----------
        type_module : str
            The full dotted name of the module the type is defined in, like
            ``numpy``.

        type_name : str
            The name of the type (the class name), like ``dtype``

        func : callable
            A callable for computing the format data.
            `func` will be called with the object to be formatted,
            and will return the raw data in this formatter's format.
            Subclasses may use a different call signature for the
            `func` argument.

            If `func` is None or unspecified, there will be no change,
            only returning the current value.

        Returns
        -------
        oldfunc : callable
            The currently registered callable.
            If you are registering a new formatter,
            this will be the previous value (to enable restoring later).
        """
        key = (type_module, type_name)
        
        try:
            oldfunc = self.lookup_by_type("%s.%s" % key)
        except KeyError:
            oldfunc = None
        
        if func is not None:
            self.deferred_printers[key] = func
        return oldfunc
    
    def pop(self, typ, default=_raise_key_error):
        """Pop a formatter for the given type.

        Parameters
        ----------
        typ : type or '__module__.__name__' string for a type
        default : object
            value to be returned if no formatter is registered for typ.

        Returns
        -------
        obj : object
            The last registered object for the type.

        Raises
        ------
        KeyError if the type is not registered and default is not specified.
        """
        
        if isinstance(typ, str):
            typ_key = tuple(typ.rsplit('.',1))
            if typ_key not in self.deferred_printers:
                # We may have it cached in the type map. We will have to
                # iterate over all of the types to check.
                for cls in self.type_printers:
                    if _mod_name_key(cls) == typ_key:
                        old = self.type_printers.pop(cls)
                        break
                else:
                    old = default
            else:
                old = self.deferred_printers.pop(typ_key)
        else:
            if typ in self.type_printers:
                old = self.type_printers.pop(typ)
            else:
                old = self.deferred_printers.pop(_mod_name_key(typ), default)
        if old is _raise_key_error:
            raise KeyError("No registered value for {0!r}".format(typ))
        return old

    def _in_deferred_types(self, cls):
        """
        Check if the given class is specified in the deferred type registry.

        Successful matches will be moved to the regular type registry for future use.
        """
        mod = getattr(cls, '__module__', None)
        name = getattr(cls, '__name__', None)
        key = (mod, name)
        if key in self.deferred_printers:
            # Move the printer over to the regular registry.
            printer = self.deferred_printers.pop(key)
            self.type_printers[cls] = printer
            return True
        return False


class PlainTextFormatter(BaseFormatter):
    """The default pretty-printer.

    This uses :mod:`IPython.lib.pretty` to compute the format data of
    the object. If the object cannot be pretty printed, :func:`repr` is used.
    See the documentation of :mod:`IPython.lib.pretty` for details on
    how to write pretty printers.  Here is a simple example::

        def dtype_pprinter(obj, p, cycle):
            if cycle:
                return p.text('dtype(...)')
            if hasattr(obj, 'fields'):
                if obj.fields is None:
                    p.text(repr(obj))
                else:
                    p.begin_group(7, 'dtype([')
                    for i, field in enumerate(obj.descr):
                        if i > 0:
                            p.text(',')
                            p.breakable()
                        p.pretty(field)
                    p.end_group(7, '])')
    """

    # The format type of data returned.
    format_type = Unicode('text/plain')

    # This subclass ignores this attribute as it always need to return
    # something.
    enabled = Bool(True).tag(config=False)
    
    max_seq_length = Integer(pretty.MAX_SEQ_LENGTH,
        help="""Truncate large collections (lists, dicts, tuples, sets) to this size.
        
        Set to 0 to disable truncation.
        """
    ).tag(config=True)
    
    # Look for a _repr_pretty_ methods to use for pretty printing.
    print_method = ObjectName('_repr_pretty_')

    # Whether to pretty-print or not.
    pprint = Bool(True).tag(config=True)

    # Whether to be verbose or not.
    verbose = Bool(False).tag(config=True)

    # The maximum width.
    max_width = Integer(79).tag(config=True)

    # The newline character.
    newline = Unicode('\n').tag(config=True)

    # format-string for pprinting floats
    float_format = Unicode('%r')
    # setter for float precision, either int or direct format-string
    float_precision = CUnicode('').tag(config=True)

    @observe('float_precision')
    def _float_precision_changed(self, change):
        """float_precision changed, set float_format accordingly.

        float_precision can be set by int or str.
        This will set float_format, after interpreting input.
        If numpy has been imported, numpy print precision will also be set.

        integer `n` sets format to '%.nf', otherwise, format set directly.

        An empty string returns to defaults (repr for float, 8 for numpy).

        This parameter can be set via the '%precision' magic.
        """
        new = change['new']
        if '%' in new:
            # got explicit format string
            fmt = new
            try:
                fmt%3.14159
            except Exception as e:
                raise ValueError("Precision must be int or format string, not %r"%new) from e
        elif new:
            # otherwise, should be an int
            try:
                i = int(new)
                assert i >= 0
            except ValueError as e:
                raise ValueError("Precision must be int or format string, not %r"%new) from e
            except AssertionError as e:
                raise ValueError("int precision must be non-negative, not %r"%i) from e

            fmt = '%%.%if'%i
            if 'numpy' in sys.modules:
                # set numpy precision if it has been imported
                import numpy
                numpy.set_printoptions(precision=i)
        else:
            # default back to repr
            fmt = '%r'
            if 'numpy' in sys.modules:
                import numpy
                # numpy default is 8
                numpy.set_printoptions(precision=8)
        self.float_format = fmt

    # Use the default pretty printers from IPython.lib.pretty.
    @default('singleton_printers')
    def _singleton_printers_default(self):
        return pretty._singleton_pprinters.copy()

    @default('type_printers')
    def _type_printers_default(self):
        d = pretty._type_pprinters.copy()
        d[float] = lambda obj,p,cycle: p.text(self.float_format%obj)
        # if NumPy is used, set precision for its float64 type
        if "numpy" in sys.modules:
            import numpy

            d[numpy.float64] = lambda obj, p, cycle: p.text(self.float_format % obj)
        return d

    @default('deferred_printers')
    def _deferred_printers_default(self):
        return pretty._deferred_type_pprinters.copy()

    #### FormatterABC interface ####

    @catch_format_error
    def __call__(self, obj):
        """Compute the pretty representation of the object."""
        if not self.pprint:
            return repr(obj)
        else:
            stream = StringIO()
            printer = pretty.RepresentationPrinter(stream, self.verbose,
                self.max_width, self.newline,
                max_seq_length=self.max_seq_length,
                singleton_pprinters=self.singleton_printers,
                type_pprinters=self.type_printers,
                deferred_pprinters=self.deferred_printers)
            printer.pretty(obj)
            printer.flush()
            return stream.getvalue()


class HTMLFormatter(BaseFormatter):
    """An HTML formatter.

    To define the callables that compute the HTML representation of your
    objects, define a :meth:`_repr_html_` method or use the :meth:`for_type`
    or :meth:`for_type_by_name` methods to register functions that handle
    this.

    The return value of this formatter should be a valid HTML snippet that
    could be injected into an existing DOM. It should *not* include the
    ```<html>`` or ```<body>`` tags.
    """
    format_type = Unicode('text/html')

    print_method = ObjectName('_repr_html_')


class MarkdownFormatter(BaseFormatter):
    """A Markdown formatter.

    To define the callables that compute the Markdown representation of your
    objects, define a :meth:`_repr_markdown_` method or use the :meth:`for_type`
    or :meth:`for_type_by_name` methods to register functions that handle
    this.

    The return value of this formatter should be a valid Markdown.
    """
    format_type = Unicode('text/markdown')

    print_method = ObjectName('_repr_markdown_')

class SVGFormatter(BaseFormatter):
    """An SVG formatter.

    To define the callables that compute the SVG representation of your
    objects, define a :meth:`_repr_svg_` method or use the :meth:`for_type`
    or :meth:`for_type_by_name` methods to register functions that handle
    this.

    The return value of this formatter should be valid SVG enclosed in
    ```<svg>``` tags, that could be injected into an existing DOM. It should
    *not* include the ```<html>`` or ```<body>`` tags.
    """
    format_type = Unicode('image/svg+xml')

    print_method = ObjectName('_repr_svg_')


class PNGFormatter(BaseFormatter):
    """A PNG formatter.

    To define the callables that compute the PNG representation of your
    objects, define a :meth:`_repr_png_` method or use the :meth:`for_type`
    or :meth:`for_type_by_name` methods to register functions that handle
    this.

    The return value of this formatter should be raw PNG data, *not*
    base64 encoded.
    """
    format_type = Unicode('image/png')

    print_method = ObjectName('_repr_png_')
    
    _return_type = (bytes, str)


class JPEGFormatter(BaseFormatter):
    """A JPEG formatter.

    To define the callables that compute the JPEG representation of your
    objects, define a :meth:`_repr_jpeg_` method or use the :meth:`for_type`
    or :meth:`for_type_by_name` methods to register functions that handle
    this.

    The return value of this formatter should be raw JPEG data, *not*
    base64 encoded.
    """
    format_type = Unicode('image/jpeg')

    print_method = ObjectName('_repr_jpeg_')

    _return_type = (bytes, str)


class LatexFormatter(BaseFormatter):
    """A LaTeX formatter.

    To define the callables that compute the LaTeX representation of your
    objects, define a :meth:`_repr_latex_` method or use the :meth:`for_type`
    or :meth:`for_type_by_name` methods to register functions that handle
    this.

    The return value of this formatter should be a valid LaTeX equation,
    enclosed in either ```$```, ```$$``` or another LaTeX equation
    environment.
    """
    format_type = Unicode('text/latex')

    print_method = ObjectName('_repr_latex_')


class JSONFormatter(BaseFormatter):
    """A JSON string formatter.

    To define the callables that compute the JSONable representation of
    your objects, define a :meth:`_repr_json_` method or use the :meth:`for_type`
    or :meth:`for_type_by_name` methods to register functions that handle
    this.

    The return value of this formatter should be a JSONable list or dict.
    JSON scalars (None, number, string) are not allowed, only dict or list containers.
    """
    format_type = Unicode('application/json')
    _return_type = (list, dict)

    print_method = ObjectName('_repr_json_')
    
    def _check_return(self, r, obj):
        """Check that a return value is appropriate

        Return the value if so, None otherwise, warning if invalid.
        """
        if r is None:
            return
        md = None
        if isinstance(r, tuple):
            # unpack data, metadata tuple for type checking on first element
            r, md = r

        assert not isinstance(
            r, str
        ), "JSON-as-string has been deprecated since IPython < 3"

        if md is not None:
            # put the tuple back together
            r = (r, md)
        return super(JSONFormatter, self)._check_return(r, obj)


class JavascriptFormatter(BaseFormatter):
    """A Javascript formatter.

    To define the callables that compute the Javascript representation of
    your objects, define a :meth:`_repr_javascript_` method or use the
    :meth:`for_type` or :meth:`for_type_by_name` methods to register functions
    that handle this.

    The return value of this formatter should be valid Javascript code and
    should *not* be enclosed in ```<script>``` tags.
    """
    format_type = Unicode('application/javascript')

    print_method = ObjectName('_repr_javascript_')


class PDFFormatter(BaseFormatter):
    """A PDF formatter.

    To define the callables that compute the PDF representation of your
    objects, define a :meth:`_repr_pdf_` method or use the :meth:`for_type`
    or :meth:`for_type_by_name` methods to register functions that handle
    this.

    The return value of this formatter should be raw PDF data, *not*
    base64 encoded.
    """
    format_type = Unicode('application/pdf')

    print_method = ObjectName('_repr_pdf_')

    _return_type = (bytes, str)

class IPythonDisplayFormatter(BaseFormatter):
    """An escape-hatch Formatter for objects that know how to display themselves.
    
    To define the callables that compute the representation of your
    objects, define a :meth:`_ipython_display_` method or use the :meth:`for_type`
    or :meth:`for_type_by_name` methods to register functions that handle
    this. Unlike mime-type displays, this method should not return anything,
    instead calling any appropriate display methods itself.
    
    This display formatter has highest priority.
    If it fires, no other display formatter will be called.

    Prior to IPython 6.1, `_ipython_display_` was the only way to display custom mime-types
    without registering a new Formatter.
    
    IPython 6.1 introduces `_repr_mimebundle_` for displaying custom mime-types,
    so `_ipython_display_` should only be used for objects that require unusual
    display patterns, such as multiple display calls.
    """
    print_method = ObjectName('_ipython_display_')
    _return_type = (type(None), bool)

    @catch_format_error
    def __call__(self, obj):
        """Compute the format for an object."""
        if self.enabled:
            # lookup registered printer
            try:
                printer = self.lookup(obj)
            except KeyError:
                pass
            else:
                printer(obj)
                return True
            # Finally look for special method names
            method = get_real_method(obj, self.print_method)
            if method is not None:
                method()
                return True


class MimeBundleFormatter(BaseFormatter):
    """A Formatter for arbitrary mime-types.

    Unlike other `_repr_<mimetype>_` methods,
    `_repr_mimebundle_` should return mime-bundle data,
    either the mime-keyed `data` dictionary or the tuple `(data, metadata)`.
    Any mime-type is valid.

    To define the callables that compute the mime-bundle representation of your
    objects, define a :meth:`_repr_mimebundle_` method or use the :meth:`for_type`
    or :meth:`for_type_by_name` methods to register functions that handle
    this.

    .. versionadded:: 6.1
    """
    print_method = ObjectName('_repr_mimebundle_')
    _return_type = dict
    
    def _check_return(self, r, obj):
        r = super(MimeBundleFormatter, self)._check_return(r, obj)
        # always return (data, metadata):
        if r is None:
            return {}, {}
        if not isinstance(r, tuple):
            return r, {}
        return r

    @catch_format_error
    def __call__(self, obj, include=None, exclude=None):
        """Compute the format for an object.

        Identical to parent's method but we pass extra parameters to the method.

        Unlike other _repr_*_ `_repr_mimebundle_` should allow extra kwargs, in
        particular `include` and `exclude`.
        """
        if self.enabled:
            # lookup registered printer
            try:
                printer = self.lookup(obj)
            except KeyError:
                pass
            else:
                return printer(obj)
            # Finally look for special method names
            method = get_real_method(obj, self.print_method)

            if method is not None:
                return method(include=include, exclude=exclude)
            return None
        else:
            return None


FormatterABC.register(BaseFormatter)
FormatterABC.register(PlainTextFormatter)
FormatterABC.register(HTMLFormatter)
FormatterABC.register(MarkdownFormatter)
FormatterABC.register(SVGFormatter)
FormatterABC.register(PNGFormatter)
FormatterABC.register(PDFFormatter)
FormatterABC.register(JPEGFormatter)
FormatterABC.register(LatexFormatter)
FormatterABC.register(JSONFormatter)
FormatterABC.register(JavascriptFormatter)
FormatterABC.register(IPythonDisplayFormatter)
FormatterABC.register(MimeBundleFormatter)


def format_display_data(obj, include=None, exclude=None):
    """Return a format data dict for an object.

    By default all format types will be computed.

    Parameters
    ----------
    obj : object
        The Python object whose format data will be computed.

    Returns
    -------
    format_dict : dict
        A dictionary of key/value pairs, one or each format that was
        generated for the object. The keys are the format types, which
        will usually be MIME type strings and the values and JSON'able
        data structure containing the raw data for the representation in
        that format.
    include : list or tuple, optional
        A list of format type strings (MIME types) to include in the
        format data dict. If this is set *only* the format types included
        in this list will be computed.
    exclude : list or tuple, optional
        A list of format type string (MIME types) to exclude in the format
        data dict. If this is set all format types will be computed,
        except for those included in this argument.
    """
    from .interactiveshell import InteractiveShell

    return InteractiveShell.instance().display_formatter.format(
        obj,
        include,
        exclude
    )
