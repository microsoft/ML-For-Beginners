# -*- coding: utf-8 -*-
"""
Python advanced pretty printer.  This pretty printer is intended to
replace the old `pprint` python module which does not allow developers
to provide their own pretty print callbacks.

This module is based on ruby's `prettyprint.rb` library by `Tanaka Akira`.


Example Usage
-------------

To directly print the representation of an object use `pprint`::

    from pretty import pprint
    pprint(complex_object)

To get a string of the output use `pretty`::

    from pretty import pretty
    string = pretty(complex_object)


Extending
---------

The pretty library allows developers to add pretty printing rules for their
own objects.  This process is straightforward.  All you have to do is to
add a `_repr_pretty_` method to your object and call the methods on the
pretty printer passed::

    class MyObject(object):

        def _repr_pretty_(self, p, cycle):
            ...

Here's an example for a class with a simple constructor::

    class MySimpleObject:

        def __init__(self, a, b, *, c=None):
            self.a = a
            self.b = b
            self.c = c

        def _repr_pretty_(self, p, cycle):
            ctor = CallExpression.factory(self.__class__.__name__)
            if self.c is None:
                p.pretty(ctor(a, b))
            else:
                p.pretty(ctor(a, b, c=c))

Here is an example implementation of a `_repr_pretty_` method for a list
subclass::

    class MyList(list):

        def _repr_pretty_(self, p, cycle):
            if cycle:
                p.text('MyList(...)')
            else:
                with p.group(8, 'MyList([', '])'):
                    for idx, item in enumerate(self):
                        if idx:
                            p.text(',')
                            p.breakable()
                        p.pretty(item)

The `cycle` parameter is `True` if pretty detected a cycle.  You *have* to
react to that or the result is an infinite loop.  `p.text()` just adds
non breaking text to the output, `p.breakable()` either adds a whitespace
or breaks here.  If you pass it an argument it's used instead of the
default space.  `p.pretty` prettyprints another object using the pretty print
method.

The first parameter to the `group` function specifies the extra indentation
of the next line.  In this example the next item will either be on the same
line (if the items are short enough) or aligned with the right edge of the
opening bracket of `MyList`.

If you just want to indent something you can use the group function
without open / close parameters.  You can also use this code::

    with p.indent(2):
        ...

Inheritance diagram:

.. inheritance-diagram:: IPython.lib.pretty
   :parts: 3

:copyright: 2007 by Armin Ronacher.
            Portions (c) 2009 by Robert Kern.
:license: BSD License.
"""

from contextlib import contextmanager
import datetime
import os
import re
import sys
import types
from collections import deque
from inspect import signature
from io import StringIO
from warnings import warn

from IPython.utils.decorators import undoc
from IPython.utils.py3compat import PYPY

__all__ = ['pretty', 'pprint', 'PrettyPrinter', 'RepresentationPrinter',
    'for_type', 'for_type_by_name', 'RawText', 'RawStringLiteral', 'CallExpression']


MAX_SEQ_LENGTH = 1000
_re_pattern_type = type(re.compile(''))

def _safe_getattr(obj, attr, default=None):
    """Safe version of getattr.

    Same as getattr, but will return ``default`` on any Exception,
    rather than raising.
    """
    try:
        return getattr(obj, attr, default)
    except Exception:
        return default

@undoc
class CUnicodeIO(StringIO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warn(("CUnicodeIO is deprecated since IPython 6.0. "
              "Please use io.StringIO instead."),
             DeprecationWarning, stacklevel=2)

def _sorted_for_pprint(items):
    """
    Sort the given items for pretty printing. Since some predictable
    sorting is better than no sorting at all, we sort on the string
    representation if normal sorting fails.
    """
    items = list(items)
    try:
        return sorted(items)
    except Exception:
        try:
            return sorted(items, key=str)
        except Exception:
            return items

def pretty(obj, verbose=False, max_width=79, newline='\n', max_seq_length=MAX_SEQ_LENGTH):
    """
    Pretty print the object's representation.
    """
    stream = StringIO()
    printer = RepresentationPrinter(stream, verbose, max_width, newline, max_seq_length=max_seq_length)
    printer.pretty(obj)
    printer.flush()
    return stream.getvalue()


def pprint(obj, verbose=False, max_width=79, newline='\n', max_seq_length=MAX_SEQ_LENGTH):
    """
    Like `pretty` but print to stdout.
    """
    printer = RepresentationPrinter(sys.stdout, verbose, max_width, newline, max_seq_length=max_seq_length)
    printer.pretty(obj)
    printer.flush()
    sys.stdout.write(newline)
    sys.stdout.flush()

class _PrettyPrinterBase(object):

    @contextmanager
    def indent(self, indent):
        """with statement support for indenting/dedenting."""
        self.indentation += indent
        try:
            yield
        finally:
            self.indentation -= indent

    @contextmanager
    def group(self, indent=0, open='', close=''):
        """like begin_group / end_group but for the with statement."""
        self.begin_group(indent, open)
        try:
            yield
        finally:
            self.end_group(indent, close)

class PrettyPrinter(_PrettyPrinterBase):
    """
    Baseclass for the `RepresentationPrinter` prettyprinter that is used to
    generate pretty reprs of objects.  Contrary to the `RepresentationPrinter`
    this printer knows nothing about the default pprinters or the `_repr_pretty_`
    callback method.
    """

    def __init__(self, output, max_width=79, newline='\n', max_seq_length=MAX_SEQ_LENGTH):
        self.output = output
        self.max_width = max_width
        self.newline = newline
        self.max_seq_length = max_seq_length
        self.output_width = 0
        self.buffer_width = 0
        self.buffer = deque()

        root_group = Group(0)
        self.group_stack = [root_group]
        self.group_queue = GroupQueue(root_group)
        self.indentation = 0

    def _break_one_group(self, group):
        while group.breakables:
            x = self.buffer.popleft()
            self.output_width = x.output(self.output, self.output_width)
            self.buffer_width -= x.width
        while self.buffer and isinstance(self.buffer[0], Text):
            x = self.buffer.popleft()
            self.output_width = x.output(self.output, self.output_width)
            self.buffer_width -= x.width

    def _break_outer_groups(self):
        while self.max_width < self.output_width + self.buffer_width:
            group = self.group_queue.deq()
            if not group:
                return
            self._break_one_group(group)

    def text(self, obj):
        """Add literal text to the output."""
        width = len(obj)
        if self.buffer:
            text = self.buffer[-1]
            if not isinstance(text, Text):
                text = Text()
                self.buffer.append(text)
            text.add(obj, width)
            self.buffer_width += width
            self._break_outer_groups()
        else:
            self.output.write(obj)
            self.output_width += width

    def breakable(self, sep=' '):
        """
        Add a breakable separator to the output.  This does not mean that it
        will automatically break here.  If no breaking on this position takes
        place the `sep` is inserted which default to one space.
        """
        width = len(sep)
        group = self.group_stack[-1]
        if group.want_break:
            self.flush()
            self.output.write(self.newline)
            self.output.write(' ' * self.indentation)
            self.output_width = self.indentation
            self.buffer_width = 0
        else:
            self.buffer.append(Breakable(sep, width, self))
            self.buffer_width += width
            self._break_outer_groups()

    def break_(self):
        """
        Explicitly insert a newline into the output, maintaining correct indentation.
        """
        group = self.group_queue.deq()
        if group:
            self._break_one_group(group)
        self.flush()
        self.output.write(self.newline)
        self.output.write(' ' * self.indentation)
        self.output_width = self.indentation
        self.buffer_width = 0


    def begin_group(self, indent=0, open=''):
        """
        Begin a group.
        The first parameter specifies the indentation for the next line (usually
        the width of the opening text), the second the opening text.  All
        parameters are optional.
        """
        if open:
            self.text(open)
        group = Group(self.group_stack[-1].depth + 1)
        self.group_stack.append(group)
        self.group_queue.enq(group)
        self.indentation += indent

    def _enumerate(self, seq):
        """like enumerate, but with an upper limit on the number of items"""
        for idx, x in enumerate(seq):
            if self.max_seq_length and idx >= self.max_seq_length:
                self.text(',')
                self.breakable()
                self.text('...')
                return
            yield idx, x

    def end_group(self, dedent=0, close=''):
        """End a group. See `begin_group` for more details."""
        self.indentation -= dedent
        group = self.group_stack.pop()
        if not group.breakables:
            self.group_queue.remove(group)
        if close:
            self.text(close)

    def flush(self):
        """Flush data that is left in the buffer."""
        for data in self.buffer:
            self.output_width += data.output(self.output, self.output_width)
        self.buffer.clear()
        self.buffer_width = 0


def _get_mro(obj_class):
    """ Get a reasonable method resolution order of a class and its superclasses
    for both old-style and new-style classes.
    """
    if not hasattr(obj_class, '__mro__'):
        # Old-style class. Mix in object to make a fake new-style class.
        try:
            obj_class = type(obj_class.__name__, (obj_class, object), {})
        except TypeError:
            # Old-style extension type that does not descend from object.
            # FIXME: try to construct a more thorough MRO.
            mro = [obj_class]
        else:
            mro = obj_class.__mro__[1:-1]
    else:
        mro = obj_class.__mro__
    return mro


class RepresentationPrinter(PrettyPrinter):
    """
    Special pretty printer that has a `pretty` method that calls the pretty
    printer for a python object.

    This class stores processing data on `self` so you must *never* use
    this class in a threaded environment.  Always lock it or reinstanciate
    it.

    Instances also have a verbose flag callbacks can access to control their
    output.  For example the default instance repr prints all attributes and
    methods that are not prefixed by an underscore if the printer is in
    verbose mode.
    """

    def __init__(self, output, verbose=False, max_width=79, newline='\n',
        singleton_pprinters=None, type_pprinters=None, deferred_pprinters=None,
        max_seq_length=MAX_SEQ_LENGTH):

        PrettyPrinter.__init__(self, output, max_width, newline, max_seq_length=max_seq_length)
        self.verbose = verbose
        self.stack = []
        if singleton_pprinters is None:
            singleton_pprinters = _singleton_pprinters.copy()
        self.singleton_pprinters = singleton_pprinters
        if type_pprinters is None:
            type_pprinters = _type_pprinters.copy()
        self.type_pprinters = type_pprinters
        if deferred_pprinters is None:
            deferred_pprinters = _deferred_type_pprinters.copy()
        self.deferred_pprinters = deferred_pprinters

    def pretty(self, obj):
        """Pretty print the given object."""
        obj_id = id(obj)
        cycle = obj_id in self.stack
        self.stack.append(obj_id)
        self.begin_group()
        try:
            obj_class = _safe_getattr(obj, '__class__', None) or type(obj)
            # First try to find registered singleton printers for the type.
            try:
                printer = self.singleton_pprinters[obj_id]
            except (TypeError, KeyError):
                pass
            else:
                return printer(obj, self, cycle)
            # Next walk the mro and check for either:
            #   1) a registered printer
            #   2) a _repr_pretty_ method
            for cls in _get_mro(obj_class):
                if cls in self.type_pprinters:
                    # printer registered in self.type_pprinters
                    return self.type_pprinters[cls](obj, self, cycle)
                else:
                    # deferred printer
                    printer = self._in_deferred_types(cls)
                    if printer is not None:
                        return printer(obj, self, cycle)
                    else:
                        # Finally look for special method names.
                        # Some objects automatically create any requested
                        # attribute. Try to ignore most of them by checking for
                        # callability.
                        if '_repr_pretty_' in cls.__dict__:
                            meth = cls._repr_pretty_
                            if callable(meth):
                                return meth(obj, self, cycle)
                        if cls is not object \
                                and callable(cls.__dict__.get('__repr__')):
                            return _repr_pprint(obj, self, cycle)

            return _default_pprint(obj, self, cycle)
        finally:
            self.end_group()
            self.stack.pop()

    def _in_deferred_types(self, cls):
        """
        Check if the given class is specified in the deferred type registry.

        Returns the printer from the registry if it exists, and None if the
        class is not in the registry. Successful matches will be moved to the
        regular type registry for future use.
        """
        mod = _safe_getattr(cls, '__module__', None)
        name = _safe_getattr(cls, '__name__', None)
        key = (mod, name)
        printer = None
        if key in self.deferred_pprinters:
            # Move the printer over to the regular registry.
            printer = self.deferred_pprinters.pop(key)
            self.type_pprinters[cls] = printer
        return printer


class Printable(object):

    def output(self, stream, output_width):
        return output_width


class Text(Printable):

    def __init__(self):
        self.objs = []
        self.width = 0

    def output(self, stream, output_width):
        for obj in self.objs:
            stream.write(obj)
        return output_width + self.width

    def add(self, obj, width):
        self.objs.append(obj)
        self.width += width


class Breakable(Printable):

    def __init__(self, seq, width, pretty):
        self.obj = seq
        self.width = width
        self.pretty = pretty
        self.indentation = pretty.indentation
        self.group = pretty.group_stack[-1]
        self.group.breakables.append(self)

    def output(self, stream, output_width):
        self.group.breakables.popleft()
        if self.group.want_break:
            stream.write(self.pretty.newline)
            stream.write(' ' * self.indentation)
            return self.indentation
        if not self.group.breakables:
            self.pretty.group_queue.remove(self.group)
        stream.write(self.obj)
        return output_width + self.width


class Group(Printable):

    def __init__(self, depth):
        self.depth = depth
        self.breakables = deque()
        self.want_break = False


class GroupQueue(object):

    def __init__(self, *groups):
        self.queue = []
        for group in groups:
            self.enq(group)

    def enq(self, group):
        depth = group.depth
        while depth > len(self.queue) - 1:
            self.queue.append([])
        self.queue[depth].append(group)

    def deq(self):
        for stack in self.queue:
            for idx, group in enumerate(reversed(stack)):
                if group.breakables:
                    del stack[idx]
                    group.want_break = True
                    return group
            for group in stack:
                group.want_break = True
            del stack[:]

    def remove(self, group):
        try:
            self.queue[group.depth].remove(group)
        except ValueError:
            pass


class RawText:
    """ Object such that ``p.pretty(RawText(value))`` is the same as ``p.text(value)``.

    An example usage of this would be to show a list as binary numbers, using
    ``p.pretty([RawText(bin(i)) for i in integers])``.
    """
    def __init__(self, value):
        self.value = value

    def _repr_pretty_(self, p, cycle):
        p.text(self.value)


class CallExpression:
    """ Object which emits a line-wrapped call expression in the form `__name(*args, **kwargs)` """
    def __init__(__self, __name, *args, **kwargs):
        # dunders are to avoid clashes with kwargs, as python's name manging
        # will kick in.
        self = __self
        self.name = __name
        self.args = args
        self.kwargs = kwargs

    @classmethod
    def factory(cls, name):
        def inner(*args, **kwargs):
            return cls(name, *args, **kwargs)
        return inner

    def _repr_pretty_(self, p, cycle):
        # dunders are to avoid clashes with kwargs, as python's name manging
        # will kick in.

        started = False
        def new_item():
            nonlocal started
            if started:
                p.text(",")
                p.breakable()
            started = True

        prefix = self.name + "("
        with p.group(len(prefix), prefix, ")"):
            for arg in self.args:
                new_item()
                p.pretty(arg)
            for arg_name, arg in self.kwargs.items():
                new_item()
                arg_prefix = arg_name + "="
                with p.group(len(arg_prefix), arg_prefix):
                    p.pretty(arg)


class RawStringLiteral:
    """ Wrapper that shows a string with a `r` prefix """
    def __init__(self, value):
        self.value = value

    def _repr_pretty_(self, p, cycle):
        base_repr = repr(self.value)
        if base_repr[:1] in 'uU':
            base_repr = base_repr[1:]
            prefix = 'ur'
        else:
            prefix = 'r'
        base_repr = prefix + base_repr.replace('\\\\', '\\')
        p.text(base_repr)


def _default_pprint(obj, p, cycle):
    """
    The default print function.  Used if an object does not provide one and
    it's none of the builtin objects.
    """
    klass = _safe_getattr(obj, '__class__', None) or type(obj)
    if _safe_getattr(klass, '__repr__', None) is not object.__repr__:
        # A user-provided repr. Find newlines and replace them with p.break_()
        _repr_pprint(obj, p, cycle)
        return
    p.begin_group(1, '<')
    p.pretty(klass)
    p.text(' at 0x%x' % id(obj))
    if cycle:
        p.text(' ...')
    elif p.verbose:
        first = True
        for key in dir(obj):
            if not key.startswith('_'):
                try:
                    value = getattr(obj, key)
                except AttributeError:
                    continue
                if isinstance(value, types.MethodType):
                    continue
                if not first:
                    p.text(',')
                p.breakable()
                p.text(key)
                p.text('=')
                step = len(key) + 1
                p.indentation += step
                p.pretty(value)
                p.indentation -= step
                first = False
    p.end_group(1, '>')


def _seq_pprinter_factory(start, end):
    """
    Factory that returns a pprint function useful for sequences.  Used by
    the default pprint for tuples and lists.
    """
    def inner(obj, p, cycle):
        if cycle:
            return p.text(start + '...' + end)
        step = len(start)
        p.begin_group(step, start)
        for idx, x in p._enumerate(obj):
            if idx:
                p.text(',')
                p.breakable()
            p.pretty(x)
        if len(obj) == 1 and isinstance(obj, tuple):
            # Special case for 1-item tuples.
            p.text(',')
        p.end_group(step, end)
    return inner


def _set_pprinter_factory(start, end):
    """
    Factory that returns a pprint function useful for sets and frozensets.
    """
    def inner(obj, p, cycle):
        if cycle:
            return p.text(start + '...' + end)
        if len(obj) == 0:
            # Special case.
            p.text(type(obj).__name__ + '()')
        else:
            step = len(start)
            p.begin_group(step, start)
            # Like dictionary keys, we will try to sort the items if there aren't too many
            if not (p.max_seq_length and len(obj) >= p.max_seq_length):
                items = _sorted_for_pprint(obj)
            else:
                items = obj
            for idx, x in p._enumerate(items):
                if idx:
                    p.text(',')
                    p.breakable()
                p.pretty(x)
            p.end_group(step, end)
    return inner


def _dict_pprinter_factory(start, end):
    """
    Factory that returns a pprint function used by the default pprint of
    dicts and dict proxies.
    """
    def inner(obj, p, cycle):
        if cycle:
            return p.text('{...}')
        step = len(start)
        p.begin_group(step, start)
        keys = obj.keys()
        for idx, key in p._enumerate(keys):
            if idx:
                p.text(',')
                p.breakable()
            p.pretty(key)
            p.text(': ')
            p.pretty(obj[key])
        p.end_group(step, end)
    return inner


def _super_pprint(obj, p, cycle):
    """The pprint for the super type."""
    p.begin_group(8, '<super: ')
    p.pretty(obj.__thisclass__)
    p.text(',')
    p.breakable()
    if PYPY: # In PyPy, super() objects don't have __self__ attributes
        dself = obj.__repr__.__self__
        p.pretty(None if dself is obj else dself)
    else:
        p.pretty(obj.__self__)
    p.end_group(8, '>')



class _ReFlags:
    def __init__(self, value):
        self.value = value

    def _repr_pretty_(self, p, cycle):
        done_one = False
        for flag in ('TEMPLATE', 'IGNORECASE', 'LOCALE', 'MULTILINE', 'DOTALL',
            'UNICODE', 'VERBOSE', 'DEBUG'):
            if self.value & getattr(re, flag):
                if done_one:
                    p.text('|')
                p.text('re.' + flag)
                done_one = True


def _re_pattern_pprint(obj, p, cycle):
    """The pprint function for regular expression patterns."""
    re_compile = CallExpression.factory('re.compile')
    if obj.flags:
        p.pretty(re_compile(RawStringLiteral(obj.pattern), _ReFlags(obj.flags)))
    else:
        p.pretty(re_compile(RawStringLiteral(obj.pattern)))


def _types_simplenamespace_pprint(obj, p, cycle):
    """The pprint function for types.SimpleNamespace."""
    namespace = CallExpression.factory('namespace')
    if cycle:
        p.pretty(namespace(RawText("...")))
    else:
        p.pretty(namespace(**obj.__dict__))


def _type_pprint(obj, p, cycle):
    """The pprint for classes and types."""
    # Heap allocated types might not have the module attribute,
    # and others may set it to None.

    # Checks for a __repr__ override in the metaclass. Can't compare the
    # type(obj).__repr__ directly because in PyPy the representation function
    # inherited from type isn't the same type.__repr__
    if [m for m in _get_mro(type(obj)) if "__repr__" in vars(m)][:1] != [type]:
        _repr_pprint(obj, p, cycle)
        return

    mod = _safe_getattr(obj, '__module__', None)
    try:
        name = obj.__qualname__
        if not isinstance(name, str):
            # This can happen if the type implements __qualname__ as a property
            # or other descriptor in Python 2.
            raise Exception("Try __name__")
    except Exception:
        name = obj.__name__
        if not isinstance(name, str):
            name = '<unknown type>'

    if mod in (None, '__builtin__', 'builtins', 'exceptions'):
        p.text(name)
    else:
        p.text(mod + '.' + name)


def _repr_pprint(obj, p, cycle):
    """A pprint that just redirects to the normal repr function."""
    # Find newlines and replace them with p.break_()
    output = repr(obj)
    lines = output.splitlines()
    with p.group():
        for idx, output_line in enumerate(lines):
            if idx:
                p.break_()
            p.text(output_line)


def _function_pprint(obj, p, cycle):
    """Base pprint for all functions and builtin functions."""
    name = _safe_getattr(obj, '__qualname__', obj.__name__)
    mod = obj.__module__
    if mod and mod not in ('__builtin__', 'builtins', 'exceptions'):
        name = mod + '.' + name
    try:
       func_def = name + str(signature(obj))
    except ValueError:
       func_def = name
    p.text('<function %s>' % func_def)


def _exception_pprint(obj, p, cycle):
    """Base pprint for all exceptions."""
    name = getattr(obj.__class__, '__qualname__', obj.__class__.__name__)
    if obj.__class__.__module__ not in ('exceptions', 'builtins'):
        name = '%s.%s' % (obj.__class__.__module__, name)

    p.pretty(CallExpression(name, *getattr(obj, 'args', ())))


#: the exception base
try:
    _exception_base = BaseException
except NameError:
    _exception_base = Exception


#: printers for builtin types
_type_pprinters = {
    int:                        _repr_pprint,
    float:                      _repr_pprint,
    str:                        _repr_pprint,
    tuple:                      _seq_pprinter_factory('(', ')'),
    list:                       _seq_pprinter_factory('[', ']'),
    dict:                       _dict_pprinter_factory('{', '}'),
    set:                        _set_pprinter_factory('{', '}'),
    frozenset:                  _set_pprinter_factory('frozenset({', '})'),
    super:                      _super_pprint,
    _re_pattern_type:           _re_pattern_pprint,
    type:                       _type_pprint,
    types.FunctionType:         _function_pprint,
    types.BuiltinFunctionType:  _function_pprint,
    types.MethodType:           _repr_pprint,
    types.SimpleNamespace:      _types_simplenamespace_pprint,
    datetime.datetime:          _repr_pprint,
    datetime.timedelta:         _repr_pprint,
    _exception_base:            _exception_pprint
}

# render os.environ like a dict
_env_type = type(os.environ)
# future-proof in case os.environ becomes a plain dict?
if _env_type is not dict:
    _type_pprinters[_env_type] = _dict_pprinter_factory('environ{', '}')

try:
    # In PyPy, types.DictProxyType is dict, setting the dictproxy printer
    # using dict.setdefault avoids overwriting the dict printer
    _type_pprinters.setdefault(types.DictProxyType,
                               _dict_pprinter_factory('dict_proxy({', '})'))
    _type_pprinters[types.ClassType] = _type_pprint
    _type_pprinters[types.SliceType] = _repr_pprint
except AttributeError: # Python 3
    _type_pprinters[types.MappingProxyType] = \
        _dict_pprinter_factory('mappingproxy({', '})')
    _type_pprinters[slice] = _repr_pprint

_type_pprinters[range] = _repr_pprint
_type_pprinters[bytes] = _repr_pprint

#: printers for types specified by name
_deferred_type_pprinters = {
}

def for_type(typ, func):
    """
    Add a pretty printer for a given type.
    """
    oldfunc = _type_pprinters.get(typ, None)
    if func is not None:
        # To support easy restoration of old pprinters, we need to ignore Nones.
        _type_pprinters[typ] = func
    return oldfunc

def for_type_by_name(type_module, type_name, func):
    """
    Add a pretty printer for a type specified by the module and name of a type
    rather than the type object itself.
    """
    key = (type_module, type_name)
    oldfunc = _deferred_type_pprinters.get(key, None)
    if func is not None:
        # To support easy restoration of old pprinters, we need to ignore Nones.
        _deferred_type_pprinters[key] = func
    return oldfunc


#: printers for the default singletons
_singleton_pprinters = dict.fromkeys(map(id, [None, True, False, Ellipsis,
                                      NotImplemented]), _repr_pprint)


def _defaultdict_pprint(obj, p, cycle):
    cls_ctor = CallExpression.factory(obj.__class__.__name__)
    if cycle:
        p.pretty(cls_ctor(RawText("...")))
    else:
        p.pretty(cls_ctor(obj.default_factory, dict(obj)))

def _ordereddict_pprint(obj, p, cycle):
    cls_ctor = CallExpression.factory(obj.__class__.__name__)
    if cycle:
        p.pretty(cls_ctor(RawText("...")))
    elif len(obj):
        p.pretty(cls_ctor(list(obj.items())))
    else:
        p.pretty(cls_ctor())

def _deque_pprint(obj, p, cycle):
    cls_ctor = CallExpression.factory(obj.__class__.__name__)
    if cycle:
        p.pretty(cls_ctor(RawText("...")))
    elif obj.maxlen is not None:
        p.pretty(cls_ctor(list(obj), maxlen=obj.maxlen))
    else:
        p.pretty(cls_ctor(list(obj)))

def _counter_pprint(obj, p, cycle):
    cls_ctor = CallExpression.factory(obj.__class__.__name__)
    if cycle:
        p.pretty(cls_ctor(RawText("...")))
    elif len(obj):
        p.pretty(cls_ctor(dict(obj.most_common())))
    else:
        p.pretty(cls_ctor())


def _userlist_pprint(obj, p, cycle):
    cls_ctor = CallExpression.factory(obj.__class__.__name__)
    if cycle:
        p.pretty(cls_ctor(RawText("...")))
    else:
        p.pretty(cls_ctor(obj.data))


for_type_by_name('collections', 'defaultdict', _defaultdict_pprint)
for_type_by_name('collections', 'OrderedDict', _ordereddict_pprint)
for_type_by_name('collections', 'deque', _deque_pprint)
for_type_by_name('collections', 'Counter', _counter_pprint)
for_type_by_name("collections", "UserList", _userlist_pprint)

if __name__ == '__main__':
    from random import randrange
    class Foo(object):
        def __init__(self):
            self.foo = 1
            self.bar = re.compile(r'\s+')
            self.blub = dict.fromkeys(range(30), randrange(1, 40))
            self.hehe = 23424.234234
            self.list = ["blub", "blah", self]

        def get_foo(self):
            print("foo")

    pprint(Foo(), verbose=True)
