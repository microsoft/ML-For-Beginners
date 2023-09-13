"""Miscellaneous utility functions and classes.

This module is used internally by Tornado.  It is not necessarily expected
that the functions and classes defined here will be useful to other
applications, but they are documented here in case they are.

The one public-facing part of this module is the `Configurable` class
and its `~Configurable.configure` method, which becomes a part of the
interface of its subclasses, including `.AsyncHTTPClient`, `.IOLoop`,
and `.Resolver`.
"""

import array
import asyncio
import atexit
from inspect import getfullargspec
import os
import re
import typing
import zlib

from typing import (
    Any,
    Optional,
    Dict,
    Mapping,
    List,
    Tuple,
    Match,
    Callable,
    Type,
    Sequence,
)

if typing.TYPE_CHECKING:
    # Additional imports only used in type comments.
    # This lets us make these imports lazy.
    import datetime  # noqa: F401
    from types import TracebackType  # noqa: F401
    from typing import Union  # noqa: F401
    import unittest  # noqa: F401

# Aliases for types that are spelled differently in different Python
# versions. bytes_type is deprecated and no longer used in Tornado
# itself but is left in case anyone outside Tornado is using it.
bytes_type = bytes
unicode_type = str
basestring_type = str

try:
    from sys import is_finalizing
except ImportError:
    # Emulate it
    def _get_emulated_is_finalizing() -> Callable[[], bool]:
        L = []  # type: List[None]
        atexit.register(lambda: L.append(None))

        def is_finalizing() -> bool:
            # Not referencing any globals here
            return L != []

        return is_finalizing

    is_finalizing = _get_emulated_is_finalizing()


# versionchanged:: 6.2
# no longer our own TimeoutError, use standard asyncio class
TimeoutError = asyncio.TimeoutError


class ObjectDict(Dict[str, Any]):
    """Makes a dictionary behave like an object, with attribute-style access."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


class GzipDecompressor(object):
    """Streaming gzip decompressor.

    The interface is like that of `zlib.decompressobj` (without some of the
    optional arguments, but it understands gzip headers and checksums.
    """

    def __init__(self) -> None:
        # Magic parameter makes zlib module understand gzip header
        # http://stackoverflow.com/questions/1838699/how-can-i-decompress-a-gzip-stream-with-zlib
        # This works on cpython and pypy, but not jython.
        self.decompressobj = zlib.decompressobj(16 + zlib.MAX_WBITS)

    def decompress(self, value: bytes, max_length: int = 0) -> bytes:
        """Decompress a chunk, returning newly-available data.

        Some data may be buffered for later processing; `flush` must
        be called when there is no more input data to ensure that
        all data was processed.

        If ``max_length`` is given, some input data may be left over
        in ``unconsumed_tail``; you must retrieve this value and pass
        it back to a future call to `decompress` if it is not empty.
        """
        return self.decompressobj.decompress(value, max_length)

    @property
    def unconsumed_tail(self) -> bytes:
        """Returns the unconsumed portion left over"""
        return self.decompressobj.unconsumed_tail

    def flush(self) -> bytes:
        """Return any remaining buffered data not yet returned by decompress.

        Also checks for errors such as truncated input.
        No other methods may be called on this object after `flush`.
        """
        return self.decompressobj.flush()


def import_object(name: str) -> Any:
    """Imports an object by name.

    ``import_object('x')`` is equivalent to ``import x``.
    ``import_object('x.y.z')`` is equivalent to ``from x.y import z``.

    >>> import tornado.escape
    >>> import_object('tornado.escape') is tornado.escape
    True
    >>> import_object('tornado.escape.utf8') is tornado.escape.utf8
    True
    >>> import_object('tornado') is tornado
    True
    >>> import_object('tornado.missing_module')
    Traceback (most recent call last):
        ...
    ImportError: No module named missing_module
    """
    if name.count(".") == 0:
        return __import__(name)

    parts = name.split(".")
    obj = __import__(".".join(parts[:-1]), fromlist=[parts[-1]])
    try:
        return getattr(obj, parts[-1])
    except AttributeError:
        raise ImportError("No module named %s" % parts[-1])


def exec_in(
    code: Any, glob: Dict[str, Any], loc: Optional[Optional[Mapping[str, Any]]] = None
) -> None:
    if isinstance(code, str):
        # exec(string) inherits the caller's future imports; compile
        # the string first to prevent that.
        code = compile(code, "<string>", "exec", dont_inherit=True)
    exec(code, glob, loc)


def raise_exc_info(
    exc_info: Tuple[Optional[type], Optional[BaseException], Optional["TracebackType"]]
) -> typing.NoReturn:
    try:
        if exc_info[1] is not None:
            raise exc_info[1].with_traceback(exc_info[2])
        else:
            raise TypeError("raise_exc_info called with no exception")
    finally:
        # Clear the traceback reference from our stack frame to
        # minimize circular references that slow down GC.
        exc_info = (None, None, None)


def errno_from_exception(e: BaseException) -> Optional[int]:
    """Provides the errno from an Exception object.

    There are cases that the errno attribute was not set so we pull
    the errno out of the args but if someone instantiates an Exception
    without any args you will get a tuple error. So this function
    abstracts all that behavior to give you a safe way to get the
    errno.
    """

    if hasattr(e, "errno"):
        return e.errno  # type: ignore
    elif e.args:
        return e.args[0]
    else:
        return None


_alphanum = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")


def _re_unescape_replacement(match: Match[str]) -> str:
    group = match.group(1)
    if group[0] in _alphanum:
        raise ValueError("cannot unescape '\\\\%s'" % group[0])
    return group


_re_unescape_pattern = re.compile(r"\\(.)", re.DOTALL)


def re_unescape(s: str) -> str:
    r"""Unescape a string escaped by `re.escape`.

    May raise ``ValueError`` for regular expressions which could not
    have been produced by `re.escape` (for example, strings containing
    ``\d`` cannot be unescaped).

    .. versionadded:: 4.4
    """
    return _re_unescape_pattern.sub(_re_unescape_replacement, s)


class Configurable(object):
    """Base class for configurable interfaces.

    A configurable interface is an (abstract) class whose constructor
    acts as a factory function for one of its implementation subclasses.
    The implementation subclass as well as optional keyword arguments to
    its initializer can be set globally at runtime with `configure`.

    By using the constructor as the factory method, the interface
    looks like a normal class, `isinstance` works as usual, etc.  This
    pattern is most useful when the choice of implementation is likely
    to be a global decision (e.g. when `~select.epoll` is available,
    always use it instead of `~select.select`), or when a
    previously-monolithic class has been split into specialized
    subclasses.

    Configurable subclasses must define the class methods
    `configurable_base` and `configurable_default`, and use the instance
    method `initialize` instead of ``__init__``.

    .. versionchanged:: 5.0

       It is now possible for configuration to be specified at
       multiple levels of a class hierarchy.

    """

    # Type annotations on this class are mostly done with comments
    # because they need to refer to Configurable, which isn't defined
    # until after the class definition block. These can use regular
    # annotations when our minimum python version is 3.7.
    #
    # There may be a clever way to use generics here to get more
    # precise types (i.e. for a particular Configurable subclass T,
    # all the types are subclasses of T, not just Configurable).
    __impl_class = None  # type: Optional[Type[Configurable]]
    __impl_kwargs = None  # type: Dict[str, Any]

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        base = cls.configurable_base()
        init_kwargs = {}  # type: Dict[str, Any]
        if cls is base:
            impl = cls.configured_class()
            if base.__impl_kwargs:
                init_kwargs.update(base.__impl_kwargs)
        else:
            impl = cls
        init_kwargs.update(kwargs)
        if impl.configurable_base() is not base:
            # The impl class is itself configurable, so recurse.
            return impl(*args, **init_kwargs)
        instance = super(Configurable, cls).__new__(impl)
        # initialize vs __init__ chosen for compatibility with AsyncHTTPClient
        # singleton magic.  If we get rid of that we can switch to __init__
        # here too.
        instance.initialize(*args, **init_kwargs)
        return instance

    @classmethod
    def configurable_base(cls):
        # type: () -> Type[Configurable]
        """Returns the base class of a configurable hierarchy.

        This will normally return the class in which it is defined.
        (which is *not* necessarily the same as the ``cls`` classmethod
        parameter).

        """
        raise NotImplementedError()

    @classmethod
    def configurable_default(cls):
        # type: () -> Type[Configurable]
        """Returns the implementation class to be used if none is configured."""
        raise NotImplementedError()

    def _initialize(self) -> None:
        pass

    initialize = _initialize  # type: Callable[..., None]
    """Initialize a `Configurable` subclass instance.

    Configurable classes should use `initialize` instead of ``__init__``.

    .. versionchanged:: 4.2
       Now accepts positional arguments in addition to keyword arguments.
    """

    @classmethod
    def configure(cls, impl, **kwargs):
        # type: (Union[None, str, Type[Configurable]], Any) -> None
        """Sets the class to use when the base class is instantiated.

        Keyword arguments will be saved and added to the arguments passed
        to the constructor.  This can be used to set global defaults for
        some parameters.
        """
        base = cls.configurable_base()
        if isinstance(impl, str):
            impl = typing.cast(Type[Configurable], import_object(impl))
        if impl is not None and not issubclass(impl, cls):
            raise ValueError("Invalid subclass of %s" % cls)
        base.__impl_class = impl
        base.__impl_kwargs = kwargs

    @classmethod
    def configured_class(cls):
        # type: () -> Type[Configurable]
        """Returns the currently configured class."""
        base = cls.configurable_base()
        # Manually mangle the private name to see whether this base
        # has been configured (and not another base higher in the
        # hierarchy).
        if base.__dict__.get("_Configurable__impl_class") is None:
            base.__impl_class = cls.configurable_default()
        if base.__impl_class is not None:
            return base.__impl_class
        else:
            # Should be impossible, but mypy wants an explicit check.
            raise ValueError("configured class not found")

    @classmethod
    def _save_configuration(cls):
        # type: () -> Tuple[Optional[Type[Configurable]], Dict[str, Any]]
        base = cls.configurable_base()
        return (base.__impl_class, base.__impl_kwargs)

    @classmethod
    def _restore_configuration(cls, saved):
        # type: (Tuple[Optional[Type[Configurable]], Dict[str, Any]]) -> None
        base = cls.configurable_base()
        base.__impl_class = saved[0]
        base.__impl_kwargs = saved[1]


class ArgReplacer(object):
    """Replaces one value in an ``args, kwargs`` pair.

    Inspects the function signature to find an argument by name
    whether it is passed by position or keyword.  For use in decorators
    and similar wrappers.
    """

    def __init__(self, func: Callable, name: str) -> None:
        self.name = name
        try:
            self.arg_pos = self._getargnames(func).index(name)  # type: Optional[int]
        except ValueError:
            # Not a positional parameter
            self.arg_pos = None

    def _getargnames(self, func: Callable) -> List[str]:
        try:
            return getfullargspec(func).args
        except TypeError:
            if hasattr(func, "func_code"):
                # Cython-generated code has all the attributes needed
                # by inspect.getfullargspec, but the inspect module only
                # works with ordinary functions. Inline the portion of
                # getfullargspec that we need here. Note that for static
                # functions the @cython.binding(True) decorator must
                # be used (for methods it works out of the box).
                code = func.func_code  # type: ignore
                return code.co_varnames[: code.co_argcount]
            raise

    def get_old_value(
        self, args: Sequence[Any], kwargs: Dict[str, Any], default: Any = None
    ) -> Any:
        """Returns the old value of the named argument without replacing it.

        Returns ``default`` if the argument is not present.
        """
        if self.arg_pos is not None and len(args) > self.arg_pos:
            return args[self.arg_pos]
        else:
            return kwargs.get(self.name, default)

    def replace(
        self, new_value: Any, args: Sequence[Any], kwargs: Dict[str, Any]
    ) -> Tuple[Any, Sequence[Any], Dict[str, Any]]:
        """Replace the named argument in ``args, kwargs`` with ``new_value``.

        Returns ``(old_value, args, kwargs)``.  The returned ``args`` and
        ``kwargs`` objects may not be the same as the input objects, or
        the input objects may be mutated.

        If the named argument was not found, ``new_value`` will be added
        to ``kwargs`` and None will be returned as ``old_value``.
        """
        if self.arg_pos is not None and len(args) > self.arg_pos:
            # The arg to replace is passed positionally
            old_value = args[self.arg_pos]
            args = list(args)  # *args is normally a tuple
            args[self.arg_pos] = new_value
        else:
            # The arg to replace is either omitted or passed by keyword.
            old_value = kwargs.get(self.name)
            kwargs[self.name] = new_value
        return old_value, args, kwargs


def timedelta_to_seconds(td):
    # type: (datetime.timedelta) -> float
    """Equivalent to ``td.total_seconds()`` (introduced in Python 2.7)."""
    return td.total_seconds()


def _websocket_mask_python(mask: bytes, data: bytes) -> bytes:
    """Websocket masking function.

    `mask` is a `bytes` object of length 4; `data` is a `bytes` object of any length.
    Returns a `bytes` object of the same length as `data` with the mask applied
    as specified in section 5.3 of RFC 6455.

    This pure-python implementation may be replaced by an optimized version when available.
    """
    mask_arr = array.array("B", mask)
    unmasked_arr = array.array("B", data)
    for i in range(len(data)):
        unmasked_arr[i] = unmasked_arr[i] ^ mask_arr[i % 4]
    return unmasked_arr.tobytes()


if os.environ.get("TORNADO_NO_EXTENSION") or os.environ.get("TORNADO_EXTENSION") == "0":
    # These environment variables exist to make it easier to do performance
    # comparisons; they are not guaranteed to remain supported in the future.
    _websocket_mask = _websocket_mask_python
else:
    try:
        from tornado.speedups import websocket_mask as _websocket_mask
    except ImportError:
        if os.environ.get("TORNADO_EXTENSION") == "1":
            raise
        _websocket_mask = _websocket_mask_python


def doctests():
    # type: () -> unittest.TestSuite
    import doctest

    return doctest.DocTestSuite()
