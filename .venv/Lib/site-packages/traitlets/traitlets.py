"""
A lightweight Traits like module.

This is designed to provide a lightweight, simple, pure Python version of
many of the capabilities of enthought.traits.  This includes:

* Validation
* Type specification with defaults
* Static and dynamic notification
* Basic predefined types
* An API that is similar to enthought.traits

We don't support:

* Delegation
* Automatic GUI generation
* A full set of trait types.  Most importantly, we don't provide container
  traits (list, dict, tuple) that can trigger notifications if their
  contents change.
* API compatibility with enthought.traits

There are also some important difference in our design:

* enthought.traits does not validate default values.  We do.

We choose to create this module because we need these capabilities, but
we need them to be pure Python so they work in all Python implementations,
including Jython and IronPython.

Inheritance diagram:

.. inheritance-diagram:: traitlets.traitlets
   :parts: 3
"""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.
#
# Adapted from enthought.traits, Copyright (c) Enthought, Inc.,
# also under the terms of the Modified BSD License.

from __future__ import annotations

import contextlib
import enum
import inspect
import os
import re
import sys
import types
import typing as t
from ast import literal_eval

from .utils.bunch import Bunch
from .utils.descriptions import add_article, class_of, describe, repr_type
from .utils.getargspec import getargspec
from .utils.importstring import import_item
from .utils.sentinel import Sentinel
from .utils.warnings import deprecated_method, should_warn, warn

SequenceTypes = (list, tuple, set, frozenset)

# backward compatibility, use to differ between Python 2 and 3.
ClassTypes = (type,)

if t.TYPE_CHECKING:
    from typing_extensions import TypeVar
else:
    from typing import TypeVar

# exports:

__all__ = [
    "All",
    "Any",
    "BaseDescriptor",
    "Bool",
    "Bytes",
    "CBool",
    "CBytes",
    "CComplex",
    "CFloat",
    "CInt",
    "CLong",
    "CRegExp",
    "CUnicode",
    "Callable",
    "CaselessStrEnum",
    "ClassBasedTraitType",
    "Complex",
    "Container",
    "DefaultHandler",
    "Dict",
    "DottedObjectName",
    "Enum",
    "EventHandler",
    "Float",
    "ForwardDeclaredInstance",
    "ForwardDeclaredMixin",
    "ForwardDeclaredType",
    "FuzzyEnum",
    "HasDescriptors",
    "HasTraits",
    "Instance",
    "Int",
    "Integer",
    "List",
    "Long",
    "MetaHasDescriptors",
    "MetaHasTraits",
    "ObjectName",
    "ObserveHandler",
    "Set",
    "TCPAddress",
    "This",
    "TraitError",
    "TraitType",
    "Tuple",
    "Type",
    "Unicode",
    "Undefined",
    "Union",
    "UseEnum",
    "ValidateHandler",
    "default",
    "directional_link",
    "dlink",
    "link",
    "observe",
    "observe_compat",
    "parse_notifier_name",
    "validate",
]

# any TraitType subclass (that doesn't start with _) will be added automatically

# -----------------------------------------------------------------------------
# Basic classes
# -----------------------------------------------------------------------------


Undefined = Sentinel(
    "Undefined",
    "traitlets",
    """
Used in Traitlets to specify that no defaults are set in kwargs
""",
)

All = Sentinel(
    "All",
    "traitlets",
    """
Used in Traitlets to listen to all types of notification or to notifications
from all trait attributes.
""",
)

# Deprecated alias
NoDefaultSpecified = Undefined


class TraitError(Exception):
    pass


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def isidentifier(s: t.Any) -> bool:
    return t.cast(bool, s.isidentifier())


def _safe_literal_eval(s: str) -> t.Any:
    """Safely evaluate an expression

    Returns original string if eval fails.

    Use only where types are ambiguous.
    """
    try:
        return literal_eval(s)
    except (NameError, SyntaxError, ValueError):
        return s


def is_trait(t: t.Any) -> bool:
    """Returns whether the given value is an instance or subclass of TraitType."""
    return isinstance(t, TraitType) or (isinstance(t, type) and issubclass(t, TraitType))


def parse_notifier_name(names: Sentinel | str | t.Iterable[Sentinel | str]) -> t.Iterable[t.Any]:
    """Convert the name argument to a list of names.

    Examples
    --------
    >>> parse_notifier_name([])
    [traitlets.All]
    >>> parse_notifier_name("a")
    ['a']
    >>> parse_notifier_name(["a", "b"])
    ['a', 'b']
    >>> parse_notifier_name(All)
    [traitlets.All]
    """
    if names is All or isinstance(names, str):
        return [names]
    elif isinstance(names, Sentinel):
        raise TypeError("`names` must be either `All`, a str, or a list of strs.")
    else:
        if not names or All in names:
            return [All]
        for n in names:
            if not isinstance(n, str):
                raise TypeError(f"names must be strings, not {type(n).__name__}({n!r})")
        return names


class _SimpleTest:
    def __init__(self, value: t.Any) -> None:
        self.value = value

    def __call__(self, test: t.Any) -> bool:
        return bool(test == self.value)

    def __repr__(self) -> str:
        return "<SimpleTest(%r)" % self.value

    def __str__(self) -> str:
        return self.__repr__()


def getmembers(object: t.Any, predicate: t.Any = None) -> list[tuple[str, t.Any]]:
    """A safe version of inspect.getmembers that handles missing attributes.

    This is useful when there are descriptor based attributes that for
    some reason raise AttributeError even though they exist.  This happens
    in zope.interface with the __provides__ attribute.
    """
    results = []
    for key in dir(object):
        try:
            value = getattr(object, key)
        except AttributeError:
            pass
        else:
            if not predicate or predicate(value):
                results.append((key, value))
    results.sort()
    return results


def _validate_link(*tuples: t.Any) -> None:
    """Validate arguments for traitlet link functions"""
    for tup in tuples:
        if not len(tup) == 2:
            raise TypeError(
                "Each linked traitlet must be specified as (HasTraits, 'trait_name'), not %r" % t
            )
        obj, trait_name = tup
        if not isinstance(obj, HasTraits):
            raise TypeError("Each object must be HasTraits, not %r" % type(obj))
        if trait_name not in obj.traits():
            raise TypeError(f"{obj!r} has no trait {trait_name!r}")


class link:
    """Link traits from different objects together so they remain in sync.

    Parameters
    ----------
    source : (object / attribute name) pair
    target : (object / attribute name) pair
    transform: iterable with two callables (optional)
        Data transformation between source and target and target and source.

    Examples
    --------
    >>> class X(HasTraits):
    ...     value = Int()

    >>> src = X(value=1)
    >>> tgt = X(value=42)
    >>> c = link((src, "value"), (tgt, "value"))

    Setting source updates target objects:
    >>> src.value = 5
    >>> tgt.value
    5
    """

    updating = False

    def __init__(self, source: t.Any, target: t.Any, transform: t.Any = None) -> None:
        _validate_link(source, target)
        self.source, self.target = source, target
        self._transform, self._transform_inv = transform if transform else (lambda x: x,) * 2

        self.link()

    def link(self) -> None:
        try:
            setattr(
                self.target[0],
                self.target[1],
                self._transform(getattr(self.source[0], self.source[1])),
            )

        finally:
            self.source[0].observe(self._update_target, names=self.source[1])
            self.target[0].observe(self._update_source, names=self.target[1])

    @contextlib.contextmanager
    def _busy_updating(self) -> t.Any:
        self.updating = True
        try:
            yield
        finally:
            self.updating = False

    def _update_target(self, change: t.Any) -> None:
        if self.updating:
            return
        with self._busy_updating():
            setattr(self.target[0], self.target[1], self._transform(change.new))
            if getattr(self.source[0], self.source[1]) != change.new:
                raise TraitError(
                    f"Broken link {self}: the source value changed while updating " "the target."
                )

    def _update_source(self, change: t.Any) -> None:
        if self.updating:
            return
        with self._busy_updating():
            setattr(self.source[0], self.source[1], self._transform_inv(change.new))
            if getattr(self.target[0], self.target[1]) != change.new:
                raise TraitError(
                    f"Broken link {self}: the target value changed while updating " "the source."
                )

    def unlink(self) -> None:
        self.source[0].unobserve(self._update_target, names=self.source[1])
        self.target[0].unobserve(self._update_source, names=self.target[1])


class directional_link:
    """Link the trait of a source object with traits of target objects.

    Parameters
    ----------
    source : (object, attribute name) pair
    target : (object, attribute name) pair
    transform: callable (optional)
        Data transformation between source and target.

    Examples
    --------
    >>> class X(HasTraits):
    ...     value = Int()

    >>> src = X(value=1)
    >>> tgt = X(value=42)
    >>> c = directional_link((src, "value"), (tgt, "value"))

    Setting source updates target objects:
    >>> src.value = 5
    >>> tgt.value
    5

    Setting target does not update source object:
    >>> tgt.value = 6
    >>> src.value
    5

    """

    updating = False

    def __init__(self, source: t.Any, target: t.Any, transform: t.Any = None) -> None:
        self._transform = transform if transform else lambda x: x
        _validate_link(source, target)
        self.source, self.target = source, target
        self.link()

    def link(self) -> None:
        try:
            setattr(
                self.target[0],
                self.target[1],
                self._transform(getattr(self.source[0], self.source[1])),
            )
        finally:
            self.source[0].observe(self._update, names=self.source[1])

    @contextlib.contextmanager
    def _busy_updating(self) -> t.Any:
        self.updating = True
        try:
            yield
        finally:
            self.updating = False

    def _update(self, change: t.Any) -> None:
        if self.updating:
            return
        with self._busy_updating():
            setattr(self.target[0], self.target[1], self._transform(change.new))

    def unlink(self) -> None:
        self.source[0].unobserve(self._update, names=self.source[1])


dlink = directional_link


# -----------------------------------------------------------------------------
# Base Descriptor Class
# -----------------------------------------------------------------------------


class BaseDescriptor:
    """Base descriptor class

    Notes
    -----
    This implements Python's descriptor protocol.

    This class is the base class for all such descriptors.  The
    only magic we use is a custom metaclass for the main :class:`HasTraits`
    class that does the following:

    1. Sets the :attr:`name` attribute of every :class:`BaseDescriptor`
       instance in the class dict to the name of the attribute.
    2. Sets the :attr:`this_class` attribute of every :class:`BaseDescriptor`
       instance in the class dict to the *class* that declared the trait.
       This is used by the :class:`This` trait to allow subclasses to
       accept superclasses for :class:`This` values.
    """

    name: str | None = None
    this_class: type[HasTraits] | None = None

    def class_init(self, cls: type[HasTraits], name: str | None) -> None:
        """Part of the initialization which may depend on the underlying
        HasDescriptors class.

        It is typically overloaded for specific types.

        This method is called by :meth:`MetaHasDescriptors.__init__`
        passing the class (`cls`) and `name` under which the descriptor
        has been assigned.
        """
        self.this_class = cls
        self.name = name

    def subclass_init(self, cls: type[HasTraits]) -> None:
        # Instead of HasDescriptors.setup_instance calling
        # every instance_init, we opt in by default.
        # This gives descriptors a change to opt out for
        # performance reasons.
        # Because most traits do not need instance_init,
        # and it will otherwise be called for every HasTrait instance
        # being created, this otherwise gives a significant performance
        # pentalty. Most TypeTraits in traitlets opt out.
        cls._instance_inits.append(self.instance_init)

    def instance_init(self, obj: t.Any) -> None:
        """Part of the initialization which may depend on the underlying
        HasDescriptors instance.

        It is typically overloaded for specific types.

        This method is called by :meth:`HasTraits.__new__` and in the
        :meth:`BaseDescriptor.instance_init` method of descriptors holding
        other descriptors.
        """


G = TypeVar("G")
S = TypeVar("S")
T = TypeVar("T")


# Self from typing extension doesn't work well with mypy https://github.com/python/mypy/pull/14041
# see https://peps.python.org/pep-0673/#use-in-generic-classes
# Self = t.TypeVar("Self", bound="TraitType[Any, Any]")
if t.TYPE_CHECKING:
    from typing_extensions import Literal, Self

    K = TypeVar("K", default=str)
    V = TypeVar("V", default=t.Any)


# We use a type for the getter (G) and setter (G) because we allow
# for traits to cast (for instance CInt will use G=int, S=t.Any)
class TraitType(BaseDescriptor, t.Generic[G, S]):
    """A base class for all trait types."""

    metadata: dict[str, t.Any] = {}
    allow_none: bool = False
    read_only: bool = False
    info_text: str = "any value"
    default_value: t.Any = Undefined

    def __init__(
        self: TraitType[G, S],
        default_value: t.Any = Undefined,
        allow_none: bool = False,
        read_only: bool | None = None,
        help: str | None = None,
        config: t.Any = None,
        **kwargs: t.Any,
    ) -> None:
        """Declare a traitlet.

        If *allow_none* is True, None is a valid value in addition to any
        values that are normally valid. The default is up to the subclass.
        For most trait types, the default value for ``allow_none`` is False.

        If *read_only* is True, attempts to directly modify a trait attribute raises a TraitError.

        If *help* is a string, it documents the attribute's purpose.

        Extra metadata can be associated with the traitlet using the .tag() convenience method
        or by using the traitlet instance's .metadata dictionary.
        """
        if default_value is not Undefined:
            self.default_value = default_value
        if allow_none:
            self.allow_none = allow_none
        if read_only is not None:
            self.read_only = read_only
        self.help = help if help is not None else ""
        if self.help:
            # define __doc__ so that inspectors like autodoc find traits
            self.__doc__ = self.help

        if len(kwargs) > 0:
            stacklevel = 1
            f = inspect.currentframe()
            # count supers to determine stacklevel for warning
            assert f is not None
            while f.f_code.co_name == "__init__":
                stacklevel += 1
                f = f.f_back
                assert f is not None
            mod = f.f_globals.get("__name__") or ""
            pkg = mod.split(".", 1)[0]
            key = ("metadata-tag", pkg, *sorted(kwargs))
            if should_warn(key):
                warn(
                    f"metadata {kwargs} was set from the constructor. "
                    "With traitlets 4.1, metadata should be set using the .tag() method, "
                    "e.g., Int().tag(key1='value1', key2='value2')",
                    DeprecationWarning,
                    stacklevel=stacklevel,
                )
            if len(self.metadata) > 0:
                self.metadata = self.metadata.copy()
                self.metadata.update(kwargs)
            else:
                self.metadata = kwargs
        else:
            self.metadata = self.metadata.copy()
        if config is not None:
            self.metadata["config"] = config

        # We add help to the metadata during a deprecation period so that
        # code that looks for the help string there can find it.
        if help is not None:
            self.metadata["help"] = help

    def from_string(self, s: str) -> G | None:
        """Get a value from a config string

        such as an environment variable or CLI arguments.

        Traits can override this method to define their own
        parsing of config strings.

        .. seealso:: item_from_string

        .. versionadded:: 5.0
        """
        if self.allow_none and s == "None":
            return None
        return s  # type:ignore[return-value]

    def default(self, obj: t.Any = None) -> G | None:
        """The default generator for this trait

        Notes
        -----
        This method is registered to HasTraits classes during ``class_init``
        in the same way that dynamic defaults defined by ``@default`` are.
        """
        if self.default_value is not Undefined:
            return t.cast(G, self.default_value)
        elif hasattr(self, "make_dynamic_default"):
            return t.cast(G, self.make_dynamic_default())
        else:
            # Undefined will raise in TraitType.get
            return t.cast(G, self.default_value)

    def get_default_value(self) -> G | None:
        """DEPRECATED: Retrieve the static default value for this trait.
        Use self.default_value instead
        """
        warn(
            "get_default_value is deprecated in traitlets 4.0: use the .default_value attribute",
            DeprecationWarning,
            stacklevel=2,
        )
        return t.cast(G, self.default_value)

    def init_default_value(self, obj: t.Any) -> G | None:
        """DEPRECATED: Set the static default value for the trait type."""
        warn(
            "init_default_value is deprecated in traitlets 4.0, and may be removed in the future",
            DeprecationWarning,
            stacklevel=2,
        )
        value = self._validate(obj, self.default_value)
        obj._trait_values[self.name] = value
        return value

    def get(self, obj: HasTraits, cls: type[t.Any] | None = None) -> G | None:
        assert self.name is not None
        try:
            value = obj._trait_values[self.name]
        except KeyError:
            # Check for a dynamic initializer.
            default = obj.trait_defaults(self.name)
            if default is Undefined:
                warn(
                    "Explicit using of Undefined as the default value "
                    "is deprecated in traitlets 5.0, and may cause "
                    "exceptions in the future.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            # Using a context manager has a large runtime overhead, so we
            # write out the obj.cross_validation_lock call here.
            _cross_validation_lock = obj._cross_validation_lock
            try:
                obj._cross_validation_lock = True
                value = self._validate(obj, default)
            finally:
                obj._cross_validation_lock = _cross_validation_lock
            obj._trait_values[self.name] = value
            obj._notify_observers(
                Bunch(
                    name=self.name,
                    value=value,
                    owner=obj,
                    type="default",
                )
            )
            return t.cast(G, value)
        except Exception as e:
            # This should never be reached.
            raise TraitError("Unexpected error in TraitType: default value not set properly") from e
        else:
            return t.cast(G, value)

    @t.overload
    def __get__(self, obj: None, cls: type[t.Any]) -> Self:
        ...

    @t.overload
    def __get__(self, obj: t.Any, cls: type[t.Any]) -> G:
        ...

    def __get__(self, obj: HasTraits | None, cls: type[t.Any]) -> Self | G:
        """Get the value of the trait by self.name for the instance.

        Default values are instantiated when :meth:`HasTraits.__new__`
        is called.  Thus by the time this method gets called either the
        default value or a user defined value (they called :meth:`__set__`)
        is in the :class:`HasTraits` instance.
        """
        if obj is None:
            return self
        else:
            return t.cast(G, self.get(obj, cls))  # the G should encode the Optional

    def set(self, obj: HasTraits, value: S) -> None:
        new_value = self._validate(obj, value)
        assert self.name is not None
        try:
            old_value = obj._trait_values[self.name]
        except KeyError:
            old_value = self.default_value

        obj._trait_values[self.name] = new_value
        try:
            silent = bool(old_value == new_value)
        except Exception:
            # if there is an error in comparing, default to notify
            silent = False
        if silent is not True:
            # we explicitly compare silent to True just in case the equality
            # comparison above returns something other than True/False
            obj._notify_trait(self.name, old_value, new_value)

    def __set__(self, obj: HasTraits, value: S) -> None:
        """Set the value of the trait by self.name for the instance.

        Values pass through a validation stage where errors are raised when
        impropper types, or types that cannot be coerced, are encountered.
        """
        if self.read_only:
            raise TraitError('The "%s" trait is read-only.' % self.name)
        self.set(obj, value)

    def _validate(self, obj: t.Any, value: t.Any) -> G | None:
        if value is None and self.allow_none:
            return value
        if hasattr(self, "validate"):
            value = self.validate(obj, value)
        if obj._cross_validation_lock is False:
            value = self._cross_validate(obj, value)
        return t.cast(G, value)

    def _cross_validate(self, obj: t.Any, value: t.Any) -> G | None:
        if self.name in obj._trait_validators:
            proposal = Bunch({"trait": self, "value": value, "owner": obj})
            value = obj._trait_validators[self.name](obj, proposal)
        elif hasattr(obj, "_%s_validate" % self.name):
            meth_name = "_%s_validate" % self.name
            cross_validate = getattr(obj, meth_name)
            deprecated_method(
                cross_validate,
                obj.__class__,
                meth_name,
                "use @validate decorator instead.",
            )
            value = cross_validate(value, self)
        return t.cast(G, value)

    def __or__(self, other: TraitType[t.Any, t.Any]) -> Union:
        if isinstance(other, Union):
            return Union([self, *other.trait_types])
        else:
            return Union([self, other])

    def info(self) -> str:
        return self.info_text

    def error(
        self,
        obj: HasTraits | None,
        value: t.Any,
        error: Exception | None = None,
        info: str | None = None,
    ) -> t.NoReturn:
        """Raise a TraitError

        Parameters
        ----------
        obj : HasTraits or None
            The instance which owns the trait. If not
            object is given, then an object agnostic
            error will be raised.
        value : any
            The value that caused the error.
        error : Exception (default: None)
            An error that was raised by a child trait.
            The arguments of this exception should be
            of the form ``(value, info, *traits)``.
            Where the ``value`` and ``info`` are the
            problem value, and string describing the
            expected value. The ``traits`` are a series
            of :class:`TraitType` instances that are
            "children" of this one (the first being
            the deepest).
        info : str (default: None)
            A description of the expected value. By
            default this is inferred from this trait's
            ``info`` method.
        """
        if error is not None:
            # handle nested error
            error.args += (self,)
            if self.name is not None:
                # this is the root trait that must format the final message
                chain = " of ".join(describe("a", t) for t in error.args[2:])
                if obj is not None:
                    error.args = (
                        "The '{}' trait of {} instance contains {} which "
                        "expected {}, not {}.".format(
                            self.name,
                            describe("an", obj),
                            chain,
                            error.args[1],
                            describe("the", error.args[0]),
                        ),
                    )
                else:
                    error.args = (
                        "The '{}' trait contains {} which " "expected {}, not {}.".format(
                            self.name,
                            chain,
                            error.args[1],
                            describe("the", error.args[0]),
                        ),
                    )
            raise error

        # this trait caused an error
        if self.name is None:
            # this is not the root trait
            raise TraitError(value, info or self.info(), self)

        # this is the root trait
        if obj is not None:
            e = "The '{}' trait of {} instance expected {}, not {}.".format(
                self.name,
                class_of(obj),
                info or self.info(),
                describe("the", value),
            )
        else:
            e = "The '{}' trait expected {}, not {}.".format(
                self.name,
                info or self.info(),
                describe("the", value),
            )
        raise TraitError(e)

    def get_metadata(self, key: str, default: t.Any = None) -> t.Any:
        """DEPRECATED: Get a metadata value.

        Use .metadata[key] or .metadata.get(key, default) instead.
        """
        if key == "help":
            msg = "use the instance .help string directly, like x.help"
        else:
            msg = "use the instance .metadata dictionary directly, like x.metadata[key] or x.metadata.get(key, default)"
        warn("Deprecated in traitlets 4.1, " + msg, DeprecationWarning, stacklevel=2)
        return self.metadata.get(key, default)

    def set_metadata(self, key: str, value: t.Any) -> None:
        """DEPRECATED: Set a metadata key/value.

        Use .metadata[key] = value instead.
        """
        if key == "help":
            msg = "use the instance .help string directly, like x.help = value"
        else:
            msg = "use the instance .metadata dictionary directly, like x.metadata[key] = value"
        warn("Deprecated in traitlets 4.1, " + msg, DeprecationWarning, stacklevel=2)
        self.metadata[key] = value

    def tag(self, **metadata: t.Any) -> Self:
        """Sets metadata and returns self.

        This allows convenient metadata tagging when initializing the trait, such as:

        Examples
        --------
        >>> Int(0).tag(config=True, sync=True)
        <traitlets.traitlets.Int object at ...>

        """
        maybe_constructor_keywords = set(metadata.keys()).intersection(
            {"help", "allow_none", "read_only", "default_value"}
        )
        if maybe_constructor_keywords:
            warn(
                "The following attributes are set in using `tag`, but seem to be constructor keywords arguments: %s "
                % maybe_constructor_keywords,
                UserWarning,
                stacklevel=2,
            )

        self.metadata.update(metadata)
        return self

    def default_value_repr(self) -> str:
        return repr(self.default_value)


# -----------------------------------------------------------------------------
# The HasTraits implementation
# -----------------------------------------------------------------------------


class _CallbackWrapper:
    """An object adapting a on_trait_change callback into an observe callback.

    The comparison operator __eq__ is implemented to enable removal of wrapped
    callbacks.
    """

    def __init__(self, cb: t.Any) -> None:
        self.cb = cb
        # Bound methods have an additional 'self' argument.
        offset = -1 if isinstance(self.cb, types.MethodType) else 0
        self.nargs = len(getargspec(cb)[0]) + offset
        if self.nargs > 4:
            raise TraitError("a trait changed callback must have 0-4 arguments.")

    def __eq__(self, other: object) -> bool:
        # The wrapper is equal to the wrapped element
        if isinstance(other, _CallbackWrapper):
            return bool(self.cb == other.cb)
        else:
            return bool(self.cb == other)

    def __call__(self, change: Bunch) -> None:
        # The wrapper is callable
        if self.nargs == 0:
            self.cb()
        elif self.nargs == 1:
            self.cb(change.name)
        elif self.nargs == 2:
            self.cb(change.name, change.new)
        elif self.nargs == 3:
            self.cb(change.name, change.old, change.new)
        elif self.nargs == 4:
            self.cb(change.name, change.old, change.new, change.owner)


def _callback_wrapper(cb: t.Any) -> _CallbackWrapper:
    if isinstance(cb, _CallbackWrapper):
        return cb
    else:
        return _CallbackWrapper(cb)


class MetaHasDescriptors(type):
    """A metaclass for HasDescriptors.

    This metaclass makes sure that any TraitType class attributes are
    instantiated and sets their name attribute.
    """

    def __new__(
        mcls: type[MetaHasDescriptors],
        name: str,
        bases: tuple[type, ...],
        classdict: dict[str, t.Any],
        **kwds: t.Any,
    ) -> MetaHasDescriptors:
        """Create the HasDescriptors class."""
        for k, v in classdict.items():
            # ----------------------------------------------------------------
            # Support of deprecated behavior allowing for TraitType types
            # to be used instead of TraitType instances.
            if inspect.isclass(v) and issubclass(v, TraitType):
                warn(
                    "Traits should be given as instances, not types (for example, `Int()`, not `Int`)."
                    " Passing types is deprecated in traitlets 4.1.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                classdict[k] = v()
            # ----------------------------------------------------------------

        return super().__new__(mcls, name, bases, classdict, **kwds)

    def __init__(
        cls, name: str, bases: tuple[type, ...], classdict: dict[str, t.Any], **kwds: t.Any
    ) -> None:
        """Finish initializing the HasDescriptors class."""
        super().__init__(name, bases, classdict, **kwds)
        cls.setup_class(classdict)

    def setup_class(cls: MetaHasDescriptors, classdict: dict[str, t.Any]) -> None:
        """Setup descriptor instance on the class

        This sets the :attr:`this_class` and :attr:`name` attributes of each
        BaseDescriptor in the class dict of the newly created ``cls`` before
        calling their :attr:`class_init` method.
        """
        cls._descriptors = []
        cls._instance_inits: list[t.Any] = []
        for k, v in classdict.items():
            if isinstance(v, BaseDescriptor):
                v.class_init(cls, k)  # type:ignore[arg-type]

        for _, v in getmembers(cls):
            if isinstance(v, BaseDescriptor):
                v.subclass_init(cls)  # type:ignore[arg-type]
                cls._descriptors.append(v)


class MetaHasTraits(MetaHasDescriptors):
    """A metaclass for HasTraits."""

    def setup_class(cls: MetaHasTraits, classdict: dict[str, t.Any]) -> None:
        # for only the current class
        cls._trait_default_generators: dict[str, t.Any] = {}
        # also looking at base classes
        cls._all_trait_default_generators = {}
        cls._traits = {}
        cls._static_immutable_initial_values = {}

        super().setup_class(classdict)

        mro = cls.mro()

        for name in dir(cls):
            # Some descriptors raise AttributeError like zope.interface's
            # __provides__ attributes even though they exist.  This causes
            # AttributeErrors even though they are listed in dir(cls).
            try:
                value = getattr(cls, name)
            except AttributeError:
                continue
            if isinstance(value, TraitType):
                cls._traits[name] = value
                trait = value
                default_method_name = "_%s_default" % name
                mro_trait = mro
                try:
                    mro_trait = mro[: mro.index(trait.this_class) + 1]  # type:ignore[arg-type]
                except ValueError:
                    # this_class not in mro
                    pass
                for c in mro_trait:
                    if default_method_name in c.__dict__:
                        cls._all_trait_default_generators[name] = c.__dict__[default_method_name]
                        break
                    if name in c.__dict__.get("_trait_default_generators", {}):
                        cls._all_trait_default_generators[name] = c._trait_default_generators[name]  # type: ignore[attr-defined]
                        break
                else:
                    # We don't have a dynamic default generator using @default etc.
                    # Now if the default value is not dynamic and immutable (string, number)
                    # and does not require any validation, we keep them in a dict
                    # of initial values to speed up instance creation.
                    # This is a very specific optimization, but a very common scenario in
                    # for instance ipywidgets.
                    none_ok = trait.default_value is None and trait.allow_none
                    if (
                        type(trait) in [CInt, Int]
                        and trait.min is None  # type: ignore[attr-defined]
                        and trait.max is None  # type: ignore[attr-defined]
                        and (isinstance(trait.default_value, int) or none_ok)
                    ):
                        cls._static_immutable_initial_values[name] = trait.default_value
                    elif (
                        type(trait) in [CFloat, Float]
                        and trait.min is None  # type: ignore[attr-defined]
                        and trait.max is None  # type: ignore[attr-defined]
                        and (isinstance(trait.default_value, float) or none_ok)
                    ):
                        cls._static_immutable_initial_values[name] = trait.default_value
                    elif type(trait) in [CBool, Bool] and (
                        isinstance(trait.default_value, bool) or none_ok
                    ):
                        cls._static_immutable_initial_values[name] = trait.default_value
                    elif type(trait) in [CUnicode, Unicode] and (
                        isinstance(trait.default_value, str) or none_ok
                    ):
                        cls._static_immutable_initial_values[name] = trait.default_value
                    elif type(trait) == Any and (
                        isinstance(trait.default_value, (str, int, float, bool)) or none_ok
                    ):
                        cls._static_immutable_initial_values[name] = trait.default_value
                    elif type(trait) == Union and trait.default_value is None:
                        cls._static_immutable_initial_values[name] = None
                    elif (
                        isinstance(trait, Instance)
                        and trait.default_args is None
                        and trait.default_kwargs is None
                        and trait.allow_none
                    ):
                        cls._static_immutable_initial_values[name] = None

                    # we always add it, because a class may change when we call add_trait
                    # and then the instance may not have all the _static_immutable_initial_values
                    cls._all_trait_default_generators[name] = trait.default


def observe(*names: Sentinel | str, type: str = "change") -> ObserveHandler:
    """A decorator which can be used to observe Traits on a class.

    The handler passed to the decorator will be called with one ``change``
    dict argument. The change dictionary at least holds a 'type' key and a
    'name' key, corresponding respectively to the type of notification and the
    name of the attribute that triggered the notification.

    Other keys may be passed depending on the value of 'type'. In the case
    where type is 'change', we also have the following keys:
    * ``owner`` : the HasTraits instance
    * ``old`` : the old value of the modified trait attribute
    * ``new`` : the new value of the modified trait attribute
    * ``name`` : the name of the modified trait attribute.

    Parameters
    ----------
    *names
        The str names of the Traits to observe on the object.
    type : str, kwarg-only
        The type of event to observe (e.g. 'change')
    """
    if not names:
        raise TypeError("Please specify at least one trait name to observe.")
    for name in names:
        if name is not All and not isinstance(name, str):
            raise TypeError("trait names to observe must be strings or All, not %r" % name)
    return ObserveHandler(names, type=type)


def observe_compat(func: FuncT) -> FuncT:
    """Backward-compatibility shim decorator for observers

    Use with:

    @observe('name')
    @observe_compat
    def _foo_changed(self, change):
        ...

    With this, `super()._foo_changed(self, name, old, new)` in subclasses will still work.
    Allows adoption of new observer API without breaking subclasses that override and super.
    """

    def compatible_observer(
        self: t.Any, change_or_name: str, old: t.Any = Undefined, new: t.Any = Undefined
    ) -> t.Any:
        if isinstance(change_or_name, dict):  # type:ignore[unreachable]
            change = Bunch(change_or_name)  # type:ignore[unreachable]
        else:
            clsname = self.__class__.__name__
            warn(
                f"A parent of {clsname}._{change_or_name}_changed has adopted the new (traitlets 4.1) @observe(change) API",
                DeprecationWarning,
                stacklevel=2,
            )
            change = Bunch(
                type="change",
                old=old,
                new=new,
                name=change_or_name,
                owner=self,
            )
        return func(self, change)

    return t.cast(FuncT, compatible_observer)


def validate(*names: Sentinel | str) -> ValidateHandler:
    """A decorator to register cross validator of HasTraits object's state
    when a Trait is set.

    The handler passed to the decorator must have one ``proposal`` dict argument.
    The proposal dictionary must hold the following keys:

    * ``owner`` : the HasTraits instance
    * ``value`` : the proposed value for the modified trait attribute
    * ``trait`` : the TraitType instance associated with the attribute

    Parameters
    ----------
    *names
        The str names of the Traits to validate.

    Notes
    -----
    Since the owner has access to the ``HasTraits`` instance via the 'owner' key,
    the registered cross validator could potentially make changes to attributes
    of the ``HasTraits`` instance. However, we recommend not to do so. The reason
    is that the cross-validation of attributes may run in arbitrary order when
    exiting the ``hold_trait_notifications`` context, and such changes may not
    commute.
    """
    if not names:
        raise TypeError("Please specify at least one trait name to validate.")
    for name in names:
        if name is not All and not isinstance(name, str):
            raise TypeError("trait names to validate must be strings or All, not %r" % name)
    return ValidateHandler(names)


def default(name: str) -> DefaultHandler:
    """A decorator which assigns a dynamic default for a Trait on a HasTraits object.

    Parameters
    ----------
    name
        The str name of the Trait on the object whose default should be generated.

    Notes
    -----
    Unlike observers and validators which are properties of the HasTraits
    instance, default value generators are class-level properties.

    Besides, default generators are only invoked if they are registered in
    subclasses of `this_type`.

    ::

        class A(HasTraits):
            bar = Int()

            @default('bar')
            def get_bar_default(self):
                return 11

        class B(A):
            bar = Float()  # This trait ignores the default generator defined in
                           # the base class A

        class C(B):

            @default('bar')
            def some_other_default(self):  # This default generator should not be
                return 3.0                 # ignored since it is defined in a
                                           # class derived from B.a.this_class.
    """
    if not isinstance(name, str):
        raise TypeError("Trait name must be a string or All, not %r" % name)
    return DefaultHandler(name)


FuncT = t.TypeVar("FuncT", bound=t.Callable[..., t.Any])


class EventHandler(BaseDescriptor):
    def _init_call(self, func: FuncT) -> EventHandler:
        self.func = func
        return self

    @t.overload
    def __call__(self, func: FuncT, *args: t.Any, **kwargs: t.Any) -> FuncT:
        ...

    @t.overload
    def __call__(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        ...

    def __call__(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        """Pass `*args` and `**kwargs` to the handler's function if it exists."""
        if hasattr(self, "func"):
            return self.func(*args, **kwargs)
        else:
            return self._init_call(*args, **kwargs)

    def __get__(self, inst: t.Any, cls: t.Any = None) -> types.MethodType | EventHandler:
        if inst is None:
            return self
        return types.MethodType(self.func, inst)


class ObserveHandler(EventHandler):
    def __init__(self, names: tuple[Sentinel | str, ...], type: str = "") -> None:
        self.trait_names = names
        self.type = type

    def instance_init(self, inst: HasTraits) -> None:
        inst.observe(self, self.trait_names, type=self.type)


class ValidateHandler(EventHandler):
    def __init__(self, names: tuple[Sentinel | str, ...]) -> None:
        self.trait_names = names

    def instance_init(self, inst: HasTraits) -> None:
        inst._register_validator(self, self.trait_names)


class DefaultHandler(EventHandler):
    def __init__(self, name: str) -> None:
        self.trait_name = name

    def class_init(self, cls: type[HasTraits], name: str | None) -> None:
        super().class_init(cls, name)
        cls._trait_default_generators[self.trait_name] = self


class HasDescriptors(metaclass=MetaHasDescriptors):
    """The base class for all classes that have descriptors."""

    def __new__(*args: t.Any, **kwargs: t.Any) -> t.Any:
        # Pass cls as args[0] to allow "cls" as keyword argument
        cls = args[0]
        args = args[1:]

        # This is needed because object.__new__ only accepts
        # the cls argument.
        new_meth = super(HasDescriptors, cls).__new__
        if new_meth is object.__new__:
            inst = new_meth(cls)
        else:
            inst = new_meth(cls, *args, **kwargs)
        inst.setup_instance(*args, **kwargs)
        return inst

    def setup_instance(*args: t.Any, **kwargs: t.Any) -> None:
        """
        This is called **before** self.__init__ is called.
        """
        # Pass self as args[0] to allow "self" as keyword argument
        self = args[0]
        args = args[1:]

        self._cross_validation_lock = False
        cls = self.__class__
        # Let descriptors performance initialization when a HasDescriptor
        # instance is created. This allows registration of observers and
        # default creations or other bookkeepings.
        # Note that descriptors can opt-out of this behavior by overriding
        # subclass_init.
        for init in cls._instance_inits:
            init(self)


class HasTraits(HasDescriptors, metaclass=MetaHasTraits):
    _trait_values: dict[str, t.Any]
    _static_immutable_initial_values: dict[str, t.Any]
    _trait_notifiers: dict[str | Sentinel, t.Any]
    _trait_validators: dict[str | Sentinel, t.Any]
    _cross_validation_lock: bool
    _traits: dict[str, t.Any]
    _all_trait_default_generators: dict[str, t.Any]

    def setup_instance(*args: t.Any, **kwargs: t.Any) -> None:
        # Pass self as args[0] to allow "self" as keyword argument
        self = args[0]
        args = args[1:]

        # although we'd prefer to set only the initial values not present
        # in kwargs, we will overwrite them in `__init__`, and simply making
        # a copy of a dict is faster than checking for each key.
        self._trait_values = self._static_immutable_initial_values.copy()
        self._trait_notifiers = {}
        self._trait_validators = {}
        self._cross_validation_lock = False
        super(HasTraits, self).setup_instance(*args, **kwargs)

    def __init__(self, *args: t.Any, **kwargs: t.Any) -> None:
        # Allow trait values to be set using keyword arguments.
        # We need to use setattr for this to trigger validation and
        # notifications.
        super_args = args
        super_kwargs = {}

        if kwargs:
            # this is a simplified (and faster) version of
            # the hold_trait_notifications(self) context manager
            def ignore(change: Bunch) -> None:
                pass

            self.notify_change = ignore  # type:ignore[method-assign]
            self._cross_validation_lock = True
            changes = {}
            for key, value in kwargs.items():
                if self.has_trait(key):
                    setattr(self, key, value)
                    changes[key] = Bunch(
                        name=key,
                        old=None,
                        new=value,
                        owner=self,
                        type="change",
                    )
                else:
                    # passthrough args that don't set traits to super
                    super_kwargs[key] = value
            # notify and cross validate all trait changes that were set in kwargs
            changed = set(kwargs) & set(self._traits)
            for key in changed:
                value = self._traits[key]._cross_validate(self, getattr(self, key))
                self.set_trait(key, value)
                changes[key]["new"] = value
            self._cross_validation_lock = False
            # Restore method retrieval from class
            del self.notify_change
            for key in changed:
                self.notify_change(changes[key])

        try:
            super().__init__(*super_args, **super_kwargs)
        except TypeError as e:
            arg_s_list = [repr(arg) for arg in super_args]
            for k, v in super_kwargs.items():
                arg_s_list.append(f"{k}={v!r}")
            arg_s = ", ".join(arg_s_list)
            warn(
                "Passing unrecognized arguments to super({classname}).__init__({arg_s}).\n"
                "{error}\n"
                "This is deprecated in traitlets 4.2."
                "This error will be raised in a future release of traitlets.".format(
                    arg_s=arg_s,
                    classname=self.__class__.__name__,
                    error=e,
                ),
                DeprecationWarning,
                stacklevel=2,
            )

    def __getstate__(self) -> dict[str, t.Any]:
        d = self.__dict__.copy()
        # event handlers stored on an instance are
        # expected to be reinstantiated during a
        # recall of instance_init during __setstate__
        d["_trait_notifiers"] = {}
        d["_trait_validators"] = {}
        d["_trait_values"] = self._trait_values.copy()
        d["_cross_validation_lock"] = False  # FIXME: raise if cloning locked!

        return d

    def __setstate__(self, state: dict[str, t.Any]) -> None:
        self.__dict__ = state.copy()

        # event handlers are reassigned to self
        cls = self.__class__
        for key in dir(cls):
            # Some descriptors raise AttributeError like zope.interface's
            # __provides__ attributes even though they exist.  This causes
            # AttributeErrors even though they are listed in dir(cls).
            try:
                value = getattr(cls, key)
            except AttributeError:
                pass
            else:
                if isinstance(value, EventHandler):
                    value.instance_init(self)

    @property
    @contextlib.contextmanager
    def cross_validation_lock(self) -> t.Any:
        """
        A contextmanager for running a block with our cross validation lock set
        to True.

        At the end of the block, the lock's value is restored to its value
        prior to entering the block.
        """
        if self._cross_validation_lock:
            yield
            return
        else:
            try:
                self._cross_validation_lock = True
                yield
            finally:
                self._cross_validation_lock = False

    @contextlib.contextmanager
    def hold_trait_notifications(self) -> t.Any:
        """Context manager for bundling trait change notifications and cross
        validation.

        Use this when doing multiple trait assignments (init, config), to avoid
        race conditions in trait notifiers requesting other trait values.
        All trait notifications will fire after all values have been assigned.
        """
        if self._cross_validation_lock:
            yield
            return
        else:
            cache: dict[str, list[Bunch]] = {}

            def compress(past_changes: list[Bunch] | None, change: Bunch) -> list[Bunch]:
                """Merges the provided change with the last if possible."""
                if past_changes is None:
                    return [change]
                else:
                    if past_changes[-1]["type"] == "change" and change.type == "change":
                        past_changes[-1]["new"] = change.new
                    else:
                        # In case of changes other than 'change', append the notification.
                        past_changes.append(change)
                    return past_changes

            def hold(change: Bunch) -> None:
                name = change.name
                cache[name] = compress(cache.get(name), change)

            try:
                # Replace notify_change with `hold`, caching and compressing
                # notifications, disable cross validation and yield.
                self.notify_change = hold  # type:ignore[method-assign]
                self._cross_validation_lock = True
                yield
                # Cross validate final values when context is released.
                for name in list(cache.keys()):
                    trait = getattr(self.__class__, name)
                    value = trait._cross_validate(self, getattr(self, name))
                    self.set_trait(name, value)
            except TraitError as e:
                # Roll back in case of TraitError during final cross validation.
                self.notify_change = lambda x: None  # type:ignore[method-assign, assignment]  # noqa: ARG005
                for name, changes in cache.items():
                    for change in changes[::-1]:
                        # TODO: Separate in a rollback function per notification type.
                        if change.type == "change":
                            if change.old is not Undefined:
                                self.set_trait(name, change.old)
                            else:
                                self._trait_values.pop(name)
                cache = {}
                raise e
            finally:
                self._cross_validation_lock = False
                # Restore method retrieval from class
                del self.notify_change

                # trigger delayed notifications
                for changes in cache.values():
                    for change in changes:
                        self.notify_change(change)

    def _notify_trait(self, name: str, old_value: t.Any, new_value: t.Any) -> None:
        self.notify_change(
            Bunch(
                name=name,
                old=old_value,
                new=new_value,
                owner=self,
                type="change",
            )
        )

    def notify_change(self, change: Bunch) -> None:
        """Notify observers of a change event"""
        return self._notify_observers(change)

    def _notify_observers(self, event: Bunch) -> None:
        """Notify observers of any event"""
        if not isinstance(event, Bunch):
            # cast to bunch if given a dict
            event = Bunch(event)  # type:ignore[unreachable]
        name, type = event["name"], event["type"]

        callables = []
        if name in self._trait_notifiers:
            callables.extend(self._trait_notifiers.get(name, {}).get(type, []))
            callables.extend(self._trait_notifiers.get(name, {}).get(All, []))
        if All in self._trait_notifiers:
            callables.extend(self._trait_notifiers.get(All, {}).get(type, []))
            callables.extend(self._trait_notifiers.get(All, {}).get(All, []))

        # Now static ones
        magic_name = "_%s_changed" % name
        if event["type"] == "change" and hasattr(self, magic_name):
            class_value = getattr(self.__class__, magic_name)
            if not isinstance(class_value, ObserveHandler):
                deprecated_method(
                    class_value,
                    self.__class__,
                    magic_name,
                    "use @observe and @unobserve instead.",
                )
                cb = getattr(self, magic_name)
                # Only append the magic method if it was not manually registered
                if cb not in callables:
                    callables.append(_callback_wrapper(cb))

        # Call them all now
        # Traits catches and logs errors here.  I allow them to raise
        for c in callables:
            # Bound methods have an additional 'self' argument.

            if isinstance(c, _CallbackWrapper):
                c = c.__call__
            elif isinstance(c, EventHandler) and c.name is not None:
                c = getattr(self, c.name)

            c(event)

    def _add_notifiers(
        self, handler: t.Callable[..., t.Any], name: Sentinel | str, type: str | Sentinel
    ) -> None:
        if name not in self._trait_notifiers:
            nlist: list[t.Any] = []
            self._trait_notifiers[name] = {type: nlist}
        else:
            if type not in self._trait_notifiers[name]:
                nlist = []
                self._trait_notifiers[name][type] = nlist
            else:
                nlist = self._trait_notifiers[name][type]
        if handler not in nlist:
            nlist.append(handler)

    def _remove_notifiers(
        self, handler: t.Callable[..., t.Any] | None, name: Sentinel | str, type: str | Sentinel
    ) -> None:
        try:
            if handler is None:
                del self._trait_notifiers[name][type]
            else:
                self._trait_notifiers[name][type].remove(handler)
        except KeyError:
            pass

    def on_trait_change(
        self,
        handler: EventHandler | None = None,
        name: Sentinel | str | None = None,
        remove: bool = False,
    ) -> None:
        """DEPRECATED: Setup a handler to be called when a trait changes.

        This is used to setup dynamic notifications of trait changes.

        Static handlers can be created by creating methods on a HasTraits
        subclass with the naming convention '_[traitname]_changed'.  Thus,
        to create static handler for the trait 'a', create the method
        _a_changed(self, name, old, new) (fewer arguments can be used, see
        below).

        If `remove` is True and `handler` is not specified, all change
        handlers for the specified name are uninstalled.

        Parameters
        ----------
        handler : callable, None
            A callable that is called when a trait changes.  Its
            signature can be handler(), handler(name), handler(name, new),
            handler(name, old, new), or handler(name, old, new, self).
        name : list, str, None
            If None, the handler will apply to all traits.  If a list
            of str, handler will apply to all names in the list.  If a
            str, the handler will apply just to that name.
        remove : bool
            If False (the default), then install the handler.  If True
            then unintall it.
        """
        warn(
            "on_trait_change is deprecated in traitlets 4.1: use observe instead",
            DeprecationWarning,
            stacklevel=2,
        )
        if name is None:
            name = All
        if remove:
            self.unobserve(_callback_wrapper(handler), names=name)
        else:
            self.observe(_callback_wrapper(handler), names=name)

    def observe(
        self,
        handler: t.Callable[..., t.Any],
        names: Sentinel | str | t.Iterable[Sentinel | str] = All,
        type: Sentinel | str = "change",
    ) -> None:
        """Setup a handler to be called when a trait changes.

        This is used to setup dynamic notifications of trait changes.

        Parameters
        ----------
        handler : callable
            A callable that is called when a trait changes. Its
            signature should be ``handler(change)``, where ``change`` is a
            dictionary. The change dictionary at least holds a 'type' key.
            * ``type``: the type of notification.
            Other keys may be passed depending on the value of 'type'. In the
            case where type is 'change', we also have the following keys:
            * ``owner`` : the HasTraits instance
            * ``old`` : the old value of the modified trait attribute
            * ``new`` : the new value of the modified trait attribute
            * ``name`` : the name of the modified trait attribute.
        names : list, str, All
            If names is All, the handler will apply to all traits.  If a list
            of str, handler will apply to all names in the list.  If a
            str, the handler will apply just to that name.
        type : str, All (default: 'change')
            The type of notification to filter by. If equal to All, then all
            notifications are passed to the observe handler.
        """
        for name in parse_notifier_name(names):
            self._add_notifiers(handler, name, type)

    def unobserve(
        self,
        handler: t.Callable[..., t.Any],
        names: Sentinel | str | t.Iterable[Sentinel | str] = All,
        type: Sentinel | str = "change",
    ) -> None:
        """Remove a trait change handler.

        This is used to unregister handlers to trait change notifications.

        Parameters
        ----------
        handler : callable
            The callable called when a trait attribute changes.
        names : list, str, All (default: All)
            The names of the traits for which the specified handler should be
            uninstalled. If names is All, the specified handler is uninstalled
            from the list of notifiers corresponding to all changes.
        type : str or All (default: 'change')
            The type of notification to filter by. If All, the specified handler
            is uninstalled from the list of notifiers corresponding to all types.
        """
        for name in parse_notifier_name(names):
            self._remove_notifiers(handler, name, type)

    def unobserve_all(self, name: str | t.Any = All) -> None:
        """Remove trait change handlers of any type for the specified name.
        If name is not specified, removes all trait notifiers."""
        if name is All:
            self._trait_notifiers = {}
        else:
            try:
                del self._trait_notifiers[name]
            except KeyError:
                pass

    def _register_validator(
        self, handler: t.Callable[..., None], names: tuple[str | Sentinel, ...]
    ) -> None:
        """Setup a handler to be called when a trait should be cross validated.

        This is used to setup dynamic notifications for cross-validation.

        If a validator is already registered for any of the provided names, a
        TraitError is raised and no new validator is registered.

        Parameters
        ----------
        handler : callable
            A callable that is called when the given trait is cross-validated.
            Its signature is handler(proposal), where proposal is a Bunch (dictionary with attribute access)
            with the following attributes/keys:
                * ``owner`` : the HasTraits instance
                * ``value`` : the proposed value for the modified trait attribute
                * ``trait`` : the TraitType instance associated with the attribute
        names : List of strings
            The names of the traits that should be cross-validated
        """
        for name in names:
            magic_name = "_%s_validate" % name
            if hasattr(self, magic_name):
                class_value = getattr(self.__class__, magic_name)
                if not isinstance(class_value, ValidateHandler):
                    deprecated_method(
                        class_value,
                        self.__class__,
                        magic_name,
                        "use @validate decorator instead.",
                    )
        for name in names:
            self._trait_validators[name] = handler

    def add_traits(self, **traits: t.Any) -> None:
        """Dynamically add trait attributes to the HasTraits instance."""
        cls = self.__class__
        attrs = {"__module__": cls.__module__}
        if hasattr(cls, "__qualname__"):
            # __qualname__ introduced in Python 3.3 (see PEP 3155)
            attrs["__qualname__"] = cls.__qualname__
        attrs.update(traits)
        self.__class__ = type(cls.__name__, (cls,), attrs)
        for trait in traits.values():
            trait.instance_init(self)

    def set_trait(self, name: str, value: t.Any) -> None:
        """Forcibly sets trait attribute, including read-only attributes."""
        cls = self.__class__
        if not self.has_trait(name):
            raise TraitError(f"Class {cls.__name__} does not have a trait named {name}")
        getattr(cls, name).set(self, value)

    @classmethod
    def class_trait_names(cls: type[HasTraits], **metadata: t.Any) -> list[str]:
        """Get a list of all the names of this class' traits.

        This method is just like the :meth:`trait_names` method,
        but is unbound.
        """
        return list(cls.class_traits(**metadata))

    @classmethod
    def class_traits(cls: type[HasTraits], **metadata: t.Any) -> dict[str, TraitType[t.Any, t.Any]]:
        """Get a ``dict`` of all the traits of this class.  The dictionary
        is keyed on the name and the values are the TraitType objects.

        This method is just like the :meth:`traits` method, but is unbound.

        The TraitTypes returned don't know anything about the values
        that the various HasTrait's instances are holding.

        The metadata kwargs allow functions to be passed in which
        filter traits based on metadata values.  The functions should
        take a single value as an argument and return a boolean.  If
        any function returns False, then the trait is not included in
        the output.  If a metadata key doesn't exist, None will be passed
        to the function.
        """
        traits = cls._traits.copy()

        if len(metadata) == 0:
            return traits

        result = {}
        for name, trait in traits.items():
            for meta_name, meta_eval in metadata.items():
                if not callable(meta_eval):
                    meta_eval = _SimpleTest(meta_eval)
                if not meta_eval(trait.metadata.get(meta_name, None)):
                    break
            else:
                result[name] = trait

        return result

    @classmethod
    def class_own_traits(
        cls: type[HasTraits], **metadata: t.Any
    ) -> dict[str, TraitType[t.Any, t.Any]]:
        """Get a dict of all the traitlets defined on this class, not a parent.

        Works like `class_traits`, except for excluding traits from parents.
        """
        sup = super(cls, cls)
        return {
            n: t
            for (n, t) in cls.class_traits(**metadata).items()
            if getattr(sup, n, None) is not t
        }

    def has_trait(self, name: str) -> bool:
        """Returns True if the object has a trait with the specified name."""
        return name in self._traits

    def trait_has_value(self, name: str) -> bool:
        """Returns True if the specified trait has a value.

        This will return false even if ``getattr`` would return a
        dynamically generated default value. These default values
        will be recognized as existing only after they have been
        generated.

        Example

        .. code-block:: python

            class MyClass(HasTraits):
                i = Int()


            mc = MyClass()
            assert not mc.trait_has_value("i")
            mc.i  # generates a default value
            assert mc.trait_has_value("i")
        """
        return name in self._trait_values

    def trait_values(self, **metadata: t.Any) -> dict[str, t.Any]:
        """A ``dict`` of trait names and their values.

        The metadata kwargs allow functions to be passed in which
        filter traits based on metadata values.  The functions should
        take a single value as an argument and return a boolean.  If
        any function returns False, then the trait is not included in
        the output.  If a metadata key doesn't exist, None will be passed
        to the function.

        Returns
        -------
        A ``dict`` of trait names and their values.

        Notes
        -----
        Trait values are retrieved via ``getattr``, any exceptions raised
        by traits or the operations they may trigger will result in the
        absence of a trait value in the result ``dict``.
        """
        return {name: getattr(self, name) for name in self.trait_names(**metadata)}

    def _get_trait_default_generator(self, name: str) -> t.Any:
        """Return default generator for a given trait

        Walk the MRO to resolve the correct default generator according to inheritance.
        """
        method_name = "_%s_default" % name
        if method_name in self.__dict__:
            return getattr(self, method_name)
        if method_name in self.__class__.__dict__:
            return getattr(self.__class__, method_name)
        return self._all_trait_default_generators[name]

    def trait_defaults(self, *names: str, **metadata: t.Any) -> dict[str, t.Any] | Sentinel:
        """Return a trait's default value or a dictionary of them

        Notes
        -----
        Dynamically generated default values may
        depend on the current state of the object."""
        for n in names:
            if not self.has_trait(n):
                raise TraitError(f"'{n}' is not a trait of '{type(self).__name__}' instances")

        if len(names) == 1 and len(metadata) == 0:
            return t.cast(Sentinel, self._get_trait_default_generator(names[0])(self))

        trait_names = self.trait_names(**metadata)
        trait_names.extend(names)

        defaults = {}
        for n in trait_names:
            defaults[n] = self._get_trait_default_generator(n)(self)
        return defaults

    def trait_names(self, **metadata: t.Any) -> list[str]:
        """Get a list of all the names of this class' traits."""
        return list(self.traits(**metadata))

    def traits(self, **metadata: t.Any) -> dict[str, TraitType[t.Any, t.Any]]:
        """Get a ``dict`` of all the traits of this class.  The dictionary
        is keyed on the name and the values are the TraitType objects.

        The TraitTypes returned don't know anything about the values
        that the various HasTrait's instances are holding.

        The metadata kwargs allow functions to be passed in which
        filter traits based on metadata values.  The functions should
        take a single value as an argument and return a boolean.  If
        any function returns False, then the trait is not included in
        the output.  If a metadata key doesn't exist, None will be passed
        to the function.
        """
        traits = self._traits.copy()

        if len(metadata) == 0:
            return traits

        result = {}
        for name, trait in traits.items():
            for meta_name, meta_eval in metadata.items():
                if not callable(meta_eval):
                    meta_eval = _SimpleTest(meta_eval)
                if not meta_eval(trait.metadata.get(meta_name, None)):
                    break
            else:
                result[name] = trait

        return result

    def trait_metadata(self, traitname: str, key: str, default: t.Any = None) -> t.Any:
        """Get metadata values for trait by key."""
        try:
            trait = getattr(self.__class__, traitname)
        except AttributeError as e:
            raise TraitError(
                f"Class {self.__class__.__name__} does not have a trait named {traitname}"
            ) from e
        metadata_name = "_" + traitname + "_metadata"
        if hasattr(self, metadata_name) and key in getattr(self, metadata_name):
            return getattr(self, metadata_name).get(key, default)
        else:
            return trait.metadata.get(key, default)

    @classmethod
    def class_own_trait_events(cls: type[HasTraits], name: str) -> dict[str, EventHandler]:
        """Get a dict of all event handlers defined on this class, not a parent.

        Works like ``event_handlers``, except for excluding traits from parents.
        """
        sup = super(cls, cls)
        return {
            n: e
            for (n, e) in cls.events(name).items()  # type:ignore[attr-defined]
            if getattr(sup, n, None) is not e
        }

    @classmethod
    def trait_events(cls: type[HasTraits], name: str | None = None) -> dict[str, EventHandler]:
        """Get a ``dict`` of all the event handlers of this class.

        Parameters
        ----------
        name : str (default: None)
            The name of a trait of this class. If name is ``None`` then all
            the event handlers of this class will be returned instead.

        Returns
        -------
        The event handlers associated with a trait name, or all event handlers.
        """
        events = {}
        for k, v in getmembers(cls):
            if isinstance(v, EventHandler):
                if name is None:
                    events[k] = v
                elif name in v.trait_names:  # type:ignore[attr-defined]
                    events[k] = v
                elif hasattr(v, "tags"):
                    if cls.trait_names(**v.tags):
                        events[k] = v
        return events


# -----------------------------------------------------------------------------
# Actual TraitTypes implementations/subclasses
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# TraitTypes subclasses for handling classes and instances of classes
# -----------------------------------------------------------------------------


class ClassBasedTraitType(TraitType[G, S]):
    """
    A trait with error reporting and string -> type resolution for Type,
    Instance and This.
    """

    def _resolve_string(self, string: str) -> t.Any:
        """
        Resolve a string supplied for a type into an actual object.
        """
        return import_item(string)


class Type(ClassBasedTraitType[G, S]):
    """A trait whose value must be a subclass of a specified class."""

    if t.TYPE_CHECKING:

        @t.overload
        def __init__(
            self: Type[type, type],
            default_value: Sentinel | None | str = ...,
            klass: None | str = ...,
            allow_none: Literal[False] = ...,
            read_only: bool | None = ...,
            help: str | None = ...,
            config: t.Any | None = ...,
            **kwargs: t.Any,
        ) -> None:
            ...

        @t.overload
        def __init__(
            self: Type[type | None, type | None],
            default_value: Sentinel | None | str = ...,
            klass: None | str = ...,
            allow_none: Literal[True] = ...,
            read_only: bool | None = ...,
            help: str | None = ...,
            config: t.Any | None = ...,
            **kwargs: t.Any,
        ) -> None:
            ...

        @t.overload
        def __init__(
            self: Type[S, S],
            default_value: S = ...,
            klass: S = ...,
            allow_none: Literal[False] = ...,
            read_only: bool | None = ...,
            help: str | None = ...,
            config: t.Any | None = ...,
            **kwargs: t.Any,
        ) -> None:
            ...

        @t.overload
        def __init__(
            self: Type[S | None, S | None],
            default_value: S | None = ...,
            klass: S = ...,
            allow_none: Literal[True] = ...,
            read_only: bool | None = ...,
            help: str | None = ...,
            config: t.Any | None = ...,
            **kwargs: t.Any,
        ) -> None:
            ...

    def __init__(
        self,
        default_value: t.Any = Undefined,
        klass: t.Any = None,
        allow_none: bool = False,
        read_only: bool | None = None,
        help: str | None = None,
        config: t.Any | None = None,
        **kwargs: t.Any,
    ) -> None:
        """Construct a Type trait

        A Type trait specifies that its values must be subclasses of
        a particular class.

        If only ``default_value`` is given, it is used for the ``klass`` as
        well. If neither are given, both default to ``object``.

        Parameters
        ----------
        default_value : class, str or None
            The default value must be a subclass of klass.  If an str,
            the str must be a fully specified class name, like 'foo.bar.Bah'.
            The string is resolved into real class, when the parent
            :class:`HasTraits` class is instantiated.
        klass : class, str [ default object ]
            Values of this trait must be a subclass of klass.  The klass
            may be specified in a string like: 'foo.bar.MyClass'.
            The string is resolved into real class, when the parent
            :class:`HasTraits` class is instantiated.
        allow_none : bool [ default False ]
            Indicates whether None is allowed as an assignable value.
        **kwargs
            extra kwargs passed to `ClassBasedTraitType`
        """
        if default_value is Undefined:
            new_default_value = object if (klass is None) else klass
        else:
            new_default_value = default_value

        if klass is None:
            if (default_value is None) or (default_value is Undefined):
                klass = object
            else:
                klass = default_value

        if not (inspect.isclass(klass) or isinstance(klass, str)):
            raise TraitError("A Type trait must specify a class.")

        self.klass = klass

        super().__init__(
            new_default_value,
            allow_none=allow_none,
            read_only=read_only,
            help=help,
            config=config,
            **kwargs,
        )

    def validate(self, obj: t.Any, value: t.Any) -> G:
        """Validates that the value is a valid object instance."""
        if isinstance(value, str):
            try:
                value = self._resolve_string(value)
            except ImportError as e:
                raise TraitError(
                    f"The '{self.name}' trait of {obj} instance must be a type, but "
                    f"{value!r} could not be imported"
                ) from e
        try:
            if issubclass(value, self.klass):  # type:ignore[arg-type]
                return t.cast(G, value)
        except Exception:
            pass

        self.error(obj, value)

    def info(self) -> str:
        """Returns a description of the trait."""
        if isinstance(self.klass, str):
            klass = self.klass
        else:
            klass = self.klass.__module__ + "." + self.klass.__name__
        result = "a subclass of '%s'" % klass
        if self.allow_none:
            return result + " or None"
        return result

    def instance_init(self, obj: t.Any) -> None:
        # we can't do this in subclass_init because that
        # might be called before all imports are done.
        self._resolve_classes()

    def _resolve_classes(self) -> None:
        if isinstance(self.klass, str):
            self.klass = self._resolve_string(self.klass)
        if isinstance(self.default_value, str):
            self.default_value = self._resolve_string(self.default_value)

    def default_value_repr(self) -> str:
        value = self.default_value
        assert value is not None
        if isinstance(value, str):
            return repr(value)
        else:
            return repr(f"{value.__module__}.{value.__name__}")


class Instance(ClassBasedTraitType[T, T]):
    """A trait whose value must be an instance of a specified class.

    The value can also be an instance of a subclass of the specified class.

    Subclasses can declare default classes by overriding the klass attribute
    """

    klass: str | type[T] | None = None

    if t.TYPE_CHECKING:

        @t.overload
        def __init__(
            self: Instance[T],
            klass: type[T] = ...,
            args: tuple[t.Any, ...] | None = ...,
            kw: dict[str, t.Any] | None = ...,
            allow_none: Literal[False] = ...,
            read_only: bool | None = ...,
            help: str | None = ...,
            **kwargs: t.Any,
        ) -> None:
            ...

        @t.overload
        def __init__(
            self: Instance[T | None],
            klass: type[T] = ...,
            args: tuple[t.Any, ...] | None = ...,
            kw: dict[str, t.Any] | None = ...,
            allow_none: Literal[True] = ...,
            read_only: bool | None = ...,
            help: str | None = ...,
            **kwargs: t.Any,
        ) -> None:
            ...

        @t.overload
        def __init__(
            self: Instance[t.Any],
            klass: str | None = ...,
            args: tuple[t.Any, ...] | None = ...,
            kw: dict[str, t.Any] | None = ...,
            allow_none: Literal[False] = ...,
            read_only: bool | None = ...,
            help: str | None = ...,
            **kwargs: t.Any,
        ) -> None:
            ...

        @t.overload
        def __init__(
            self: Instance[t.Any | None],
            klass: str | None = ...,
            args: tuple[t.Any, ...] | None = ...,
            kw: dict[str, t.Any] | None = ...,
            allow_none: Literal[True] = ...,
            read_only: bool | None = ...,
            help: str | None = ...,
            **kwargs: t.Any,
        ) -> None:
            ...

    def __init__(
        self,
        klass: str | type[T] | None = None,
        args: tuple[t.Any, ...] | None = None,
        kw: dict[str, t.Any] | None = None,
        allow_none: bool = False,
        read_only: bool | None = None,
        help: str | None = None,
        **kwargs: t.Any,
    ) -> None:
        """Construct an Instance trait.

        This trait allows values that are instances of a particular
        class or its subclasses.  Our implementation is quite different
        from that of enthough.traits as we don't allow instances to be used
        for klass and we handle the ``args`` and ``kw`` arguments differently.

        Parameters
        ----------
        klass : class, str
            The class that forms the basis for the trait.  Class names
            can also be specified as strings, like 'foo.bar.Bar'.
        args : tuple
            Positional arguments for generating the default value.
        kw : dict
            Keyword arguments for generating the default value.
        allow_none : bool [ default False ]
            Indicates whether None is allowed as a value.
        **kwargs
            Extra kwargs passed to `ClassBasedTraitType`

        Notes
        -----
        If both ``args`` and ``kw`` are None, then the default value is None.
        If ``args`` is a tuple and ``kw`` is a dict, then the default is
        created as ``klass(*args, **kw)``.  If exactly one of ``args`` or ``kw`` is
        None, the None is replaced by ``()`` or ``{}``, respectively.
        """
        if klass is None:
            klass = self.klass

        if (klass is not None) and (inspect.isclass(klass) or isinstance(klass, str)):
            self.klass = klass
        else:
            raise TraitError("The klass attribute must be a class not: %r" % klass)

        if (kw is not None) and not isinstance(kw, dict):
            raise TraitError("The 'kw' argument must be a dict or None.")
        if (args is not None) and not isinstance(args, tuple):
            raise TraitError("The 'args' argument must be a tuple or None.")

        self.default_args = args
        self.default_kwargs = kw

        super().__init__(allow_none=allow_none, read_only=read_only, help=help, **kwargs)

    def validate(self, obj: t.Any, value: t.Any) -> T | None:
        assert self.klass is not None
        if self.allow_none and value is None:
            return value
        if isinstance(value, self.klass):  # type:ignore[arg-type]
            return t.cast(T, value)
        else:
            self.error(obj, value)

    def info(self) -> str:
        if isinstance(self.klass, str):
            result = add_article(self.klass)
        else:
            result = describe("a", self.klass)
        if self.allow_none:
            result += " or None"
        return result

    def instance_init(self, obj: t.Any) -> None:
        # we can't do this in subclass_init because that
        # might be called before all imports are done.
        self._resolve_classes()

    def _resolve_classes(self) -> None:
        if isinstance(self.klass, str):
            self.klass = self._resolve_string(self.klass)

    def make_dynamic_default(self) -> T | None:
        if (self.default_args is None) and (self.default_kwargs is None):
            return None
        assert self.klass is not None
        return self.klass(*(self.default_args or ()), **(self.default_kwargs or {}))  # type:ignore[operator]

    def default_value_repr(self) -> str:
        return repr(self.make_dynamic_default())

    def from_string(self, s: str) -> T | None:
        return t.cast(T, _safe_literal_eval(s))


class ForwardDeclaredMixin:
    """
    Mixin for forward-declared versions of Instance and Type.
    """

    def _resolve_string(self, string: str) -> t.Any:
        """
        Find the specified class name by looking for it in the module in which
        our this_class attribute was defined.
        """
        modname = self.this_class.__module__  # type:ignore[attr-defined]
        return import_item(".".join([modname, string]))


class ForwardDeclaredType(ForwardDeclaredMixin, Type[G, S]):
    """
    Forward-declared version of Type.
    """


class ForwardDeclaredInstance(ForwardDeclaredMixin, Instance[T]):
    """
    Forward-declared version of Instance.
    """


class This(ClassBasedTraitType[t.Optional[T], t.Optional[T]]):
    """A trait for instances of the class containing this trait.

    Because how how and when class bodies are executed, the ``This``
    trait can only have a default value of None.  This, and because we
    always validate default values, ``allow_none`` is *always* true.
    """

    info_text = "an instance of the same type as the receiver or None"

    def __init__(self, **kwargs: t.Any) -> None:
        super().__init__(None, **kwargs)

    def validate(self, obj: t.Any, value: t.Any) -> HasTraits | None:
        # What if value is a superclass of obj.__class__?  This is
        # complicated if it was the superclass that defined the This
        # trait.
        assert self.this_class is not None
        if isinstance(value, self.this_class) or (value is None):
            return value
        else:
            self.error(obj, value)


class Union(TraitType[t.Any, t.Any]):
    """A trait type representing a Union type."""

    def __init__(self, trait_types: t.Any, **kwargs: t.Any) -> None:
        """Construct a Union  trait.

        This trait allows values that are allowed by at least one of the
        specified trait types. A Union traitlet cannot have metadata on
        its own, besides the metadata of the listed types.

        Parameters
        ----------
        trait_types : sequence
            The list of trait types of length at least 1.
        **kwargs
            Extra kwargs passed to `TraitType`

        Notes
        -----
        Union([Float(), Bool(), Int()]) attempts to validate the provided values
        with the validation function of Float, then Bool, and finally Int.

        Parsing from string is ambiguous for container types which accept other
        collection-like literals (e.g. List accepting both `[]` and `()`
        precludes Union from ever parsing ``Union([List(), Tuple()])`` as a tuple;
        you can modify behaviour of too permissive container traits by overriding
        ``_literal_from_string_pairs`` in subclasses.
        Similarly, parsing unions of numeric types is only unambiguous if
        types are provided in order of increasing permissiveness, e.g.
        ``Union([Int(), Float()])`` (since floats accept integer-looking values).
        """
        self.trait_types = list(trait_types)
        self.info_text = " or ".join([tt.info() for tt in self.trait_types])
        super().__init__(**kwargs)

    def default(self, obj: t.Any = None) -> t.Any:
        default = super().default(obj)
        for trait in self.trait_types:
            if default is Undefined:
                default = trait.default(obj)
            else:
                break
        return default

    def class_init(self, cls: type[HasTraits], name: str | None) -> None:
        for trait_type in reversed(self.trait_types):
            trait_type.class_init(cls, None)
        super().class_init(cls, name)

    def subclass_init(self, cls: type[t.Any]) -> None:
        for trait_type in reversed(self.trait_types):
            trait_type.subclass_init(cls)
        # explicitly not calling super().subclass_init(cls)
        # to opt out of instance_init

    def validate(self, obj: t.Any, value: t.Any) -> t.Any:
        with obj.cross_validation_lock:
            for trait_type in self.trait_types:
                try:
                    v = trait_type._validate(obj, value)
                    # In the case of an element trait, the name is None
                    if self.name is not None:
                        setattr(obj, "_" + self.name + "_metadata", trait_type.metadata)
                    return v
                except TraitError:
                    continue
        self.error(obj, value)

    def __or__(self, other: t.Any) -> Union:
        if isinstance(other, Union):
            return Union(self.trait_types + other.trait_types)
        else:
            return Union([*self.trait_types, other])

    def from_string(self, s: str) -> t.Any:
        for trait_type in self.trait_types:
            try:
                v = trait_type.from_string(s)
                return trait_type.validate(None, v)
            except (TraitError, ValueError):
                continue
        return super().from_string(s)


# -----------------------------------------------------------------------------
# Basic TraitTypes implementations/subclasses
# -----------------------------------------------------------------------------


class Any(TraitType[t.Optional[t.Any], t.Optional[t.Any]]):
    """A trait which allows any value."""

    if t.TYPE_CHECKING:

        @t.overload
        def __init__(
            self: Any,
            default_value: t.Any = ...,
            *,
            allow_none: Literal[False],
            read_only: bool | None = ...,
            help: str | None = ...,
            config: t.Any | None = ...,
            **kwargs: t.Any,
        ) -> None:
            ...

        @t.overload
        def __init__(
            self: Any,
            default_value: t.Any = ...,
            *,
            allow_none: Literal[True],
            read_only: bool | None = ...,
            help: str | None = ...,
            config: t.Any | None = ...,
            **kwargs: t.Any,
        ) -> None:
            ...

        @t.overload
        def __init__(
            self: Any,
            default_value: t.Any = ...,
            *,
            allow_none: Literal[True, False] = ...,
            help: str | None = ...,
            read_only: bool | None = False,
            config: t.Any = None,
            **kwargs: t.Any,
        ) -> None:
            ...

        def __init__(
            self: Any,
            default_value: t.Any = ...,
            *,
            allow_none: bool = False,
            help: str | None = "",
            read_only: bool | None = False,
            config: t.Any = None,
            **kwargs: t.Any,
        ) -> None:
            ...

        @t.overload
        def __get__(self, obj: None, cls: type[t.Any]) -> Any:
            ...

        @t.overload
        def __get__(self, obj: t.Any, cls: type[t.Any]) -> t.Any:
            ...

        def __get__(self, obj: t.Any | None, cls: type[t.Any]) -> t.Any | Any:
            ...

    default_value: t.Any | None = None
    allow_none = True
    info_text = "any value"

    def subclass_init(self, cls: type[t.Any]) -> None:
        pass  # fully opt out of instance_init


def _validate_bounds(
    trait: Int[t.Any, t.Any] | Float[t.Any, t.Any], obj: t.Any, value: t.Any
) -> t.Any:
    """
    Validate that a number to be applied to a trait is between bounds.

    If value is not between min_bound and max_bound, this raises a
    TraitError with an error message appropriate for this trait.
    """
    if trait.min is not None and value < trait.min:
        raise TraitError(
            f"The value of the '{trait.name}' trait of {class_of(obj)} instance should "
            f"not be less than {trait.min}, but a value of {value} was "
            "specified"
        )
    if trait.max is not None and value > trait.max:
        raise TraitError(
            f"The value of the '{trait.name}' trait of {class_of(obj)} instance should "
            f"not be greater than {trait.max}, but a value of {value} was "
            "specified"
        )
    return value


# I = t.TypeVar('I', t.Optional[int], int)


class Int(TraitType[G, S]):
    """An int trait."""

    default_value = 0
    info_text = "an int"

    @t.overload
    def __init__(
        self: Int[int, int],
        default_value: int | Sentinel = ...,
        allow_none: Literal[False] = ...,
        read_only: bool | None = ...,
        help: str | None = ...,
        config: t.Any | None = ...,
        **kwargs: t.Any,
    ) -> None:
        ...

    @t.overload
    def __init__(
        self: Int[int | None, int | None],
        default_value: int | Sentinel | None = ...,
        allow_none: Literal[True] = ...,
        read_only: bool | None = ...,
        help: str | None = ...,
        config: t.Any | None = ...,
        **kwargs: t.Any,
    ) -> None:
        ...

    def __init__(
        self,
        default_value: t.Any = Undefined,
        allow_none: bool = False,
        read_only: bool | None = None,
        help: str | None = None,
        config: t.Any | None = None,
        **kwargs: t.Any,
    ) -> None:
        self.min = kwargs.pop("min", None)
        self.max = kwargs.pop("max", None)
        super().__init__(
            default_value=default_value,
            allow_none=allow_none,
            read_only=read_only,
            help=help,
            config=config,
            **kwargs,
        )

    def validate(self, obj: t.Any, value: t.Any) -> G:
        if not isinstance(value, int):
            self.error(obj, value)
        return t.cast(G, _validate_bounds(self, obj, value))

    def from_string(self, s: str) -> G:
        if self.allow_none and s == "None":
            return t.cast(G, None)
        return t.cast(G, int(s))

    def subclass_init(self, cls: type[t.Any]) -> None:
        pass  # fully opt out of instance_init


class CInt(Int[G, S]):
    """A casting version of the int trait."""

    if t.TYPE_CHECKING:

        @t.overload
        def __init__(
            self: CInt[int, t.Any],
            default_value: t.Any | Sentinel = ...,
            allow_none: Literal[False] = ...,
            read_only: bool | None = ...,
            help: str | None = ...,
            config: t.Any | None = ...,
            **kwargs: t.Any,
        ) -> None:
            ...

        @t.overload
        def __init__(
            self: CInt[int | None, t.Any],
            default_value: t.Any | Sentinel | None = ...,
            allow_none: Literal[True] = ...,
            read_only: bool | None = ...,
            help: str | None = ...,
            config: t.Any | None = ...,
            **kwargs: t.Any,
        ) -> None:
            ...

        def __init__(
            self: CInt[int | None, t.Any],
            default_value: t.Any | Sentinel | None = ...,
            allow_none: bool = ...,
            read_only: bool | None = ...,
            help: str | None = ...,
            config: t.Any | None = ...,
            **kwargs: t.Any,
        ) -> None:
            ...

    def validate(self, obj: t.Any, value: t.Any) -> G:
        try:
            value = int(value)
        except Exception:
            self.error(obj, value)
        return t.cast(G, _validate_bounds(self, obj, value))


Long, CLong = Int, CInt
Integer = Int


class Float(TraitType[G, S]):
    """A float trait."""

    default_value = 0.0
    info_text = "a float"

    @t.overload
    def __init__(
        self: Float[float, int | float],
        default_value: float | Sentinel = ...,
        allow_none: Literal[False] = ...,
        read_only: bool | None = ...,
        help: str | None = ...,
        config: t.Any | None = ...,
        **kwargs: t.Any,
    ) -> None:
        ...

    @t.overload
    def __init__(
        self: Float[int | None, int | float | None],
        default_value: float | Sentinel | None = ...,
        allow_none: Literal[True] = ...,
        read_only: bool | None = ...,
        help: str | None = ...,
        config: t.Any | None = ...,
        **kwargs: t.Any,
    ) -> None:
        ...

    def __init__(
        self: Float[int | None, int | float | None],
        default_value: float | Sentinel | None = Undefined,
        allow_none: bool = False,
        read_only: bool | None = False,
        help: str | None = None,
        config: t.Any | None = None,
        **kwargs: t.Any,
    ) -> None:
        self.min = kwargs.pop("min", -float("inf"))
        self.max = kwargs.pop("max", float("inf"))
        super().__init__(
            default_value=default_value,
            allow_none=allow_none,
            read_only=read_only,
            help=help,
            config=config,
            **kwargs,
        )

    def validate(self, obj: t.Any, value: t.Any) -> G:
        if isinstance(value, int):
            value = float(value)
        if not isinstance(value, float):
            self.error(obj, value)
        return t.cast(G, _validate_bounds(self, obj, value))

    def from_string(self, s: str) -> G:
        if self.allow_none and s == "None":
            return t.cast(G, None)
        return t.cast(G, float(s))

    def subclass_init(self, cls: type[t.Any]) -> None:
        pass  # fully opt out of instance_init


class CFloat(Float[G, S]):
    """A casting version of the float trait."""

    if t.TYPE_CHECKING:

        @t.overload
        def __init__(
            self: CFloat[float, t.Any],
            default_value: t.Any = ...,
            allow_none: Literal[False] = ...,
            read_only: bool | None = ...,
            help: str | None = ...,
            config: t.Any | None = ...,
            **kwargs: t.Any,
        ) -> None:
            ...

        @t.overload
        def __init__(
            self: CFloat[float | None, t.Any],
            default_value: t.Any = ...,
            allow_none: Literal[True] = ...,
            read_only: bool | None = ...,
            help: str | None = ...,
            config: t.Any | None = ...,
            **kwargs: t.Any,
        ) -> None:
            ...

        def __init__(
            self: CFloat[float | None, t.Any],
            default_value: t.Any = ...,
            allow_none: bool = ...,
            read_only: bool | None = ...,
            help: str | None = ...,
            config: t.Any | None = ...,
            **kwargs: t.Any,
        ) -> None:
            ...

    def validate(self, obj: t.Any, value: t.Any) -> G:
        try:
            value = float(value)
        except Exception:
            self.error(obj, value)
        return t.cast(G, _validate_bounds(self, obj, value))


class Complex(TraitType[complex, t.Union[complex, float, int]]):
    """A trait for complex numbers."""

    default_value = 0.0 + 0.0j
    info_text = "a complex number"

    def validate(self, obj: t.Any, value: t.Any) -> complex | None:
        if isinstance(value, complex):
            return value
        if isinstance(value, (float, int)):
            return complex(value)
        self.error(obj, value)

    def from_string(self, s: str) -> complex | None:
        if self.allow_none and s == "None":
            return None
        return complex(s)

    def subclass_init(self, cls: type[t.Any]) -> None:
        pass  # fully opt out of instance_init


class CComplex(Complex, TraitType[complex, t.Any]):
    """A casting version of the complex number trait."""

    def validate(self, obj: t.Any, value: t.Any) -> complex | None:
        try:
            return complex(value)
        except Exception:
            self.error(obj, value)


# We should always be explicit about whether we're using bytes or unicode, both
# for Python 3 conversion and for reliable unicode behaviour on Python 2. So
# we don't have a Str type.
class Bytes(TraitType[bytes, bytes]):
    """A trait for byte strings."""

    default_value = b""
    info_text = "a bytes object"

    def validate(self, obj: t.Any, value: t.Any) -> bytes | None:
        if isinstance(value, bytes):
            return value
        self.error(obj, value)

    def from_string(self, s: str) -> bytes | None:
        if self.allow_none and s == "None":
            return None
        if len(s) >= 3:
            # handle deprecated b"string"
            for quote in ('"', "'"):
                if s[:2] == f"b{quote}" and s[-1] == quote:
                    old_s = s
                    s = s[2:-1]
                    warn(
                        "Supporting extra quotes around Bytes is deprecated in traitlets 5.0. "
                        f"Use {s!r} instead of {old_s!r}.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                    break
        return s.encode("utf8")

    def subclass_init(self, cls: type[t.Any]) -> None:
        pass  # fully opt out of instance_init


class CBytes(Bytes, TraitType[bytes, t.Any]):
    """A casting version of the byte string trait."""

    def validate(self, obj: t.Any, value: t.Any) -> bytes | None:
        try:
            return bytes(value)
        except Exception:
            self.error(obj, value)


class Unicode(TraitType[G, S]):
    """A trait for unicode strings."""

    default_value = ""
    info_text = "a unicode string"

    if t.TYPE_CHECKING:

        @t.overload
        def __init__(
            self: Unicode[str, str | bytes],
            default_value: str | Sentinel = ...,
            allow_none: Literal[False] = ...,
            read_only: bool | None = ...,
            help: str | None = ...,
            config: t.Any = ...,
            **kwargs: t.Any,
        ) -> None:
            ...

        @t.overload
        def __init__(
            self: Unicode[str | None, str | bytes | None],
            default_value: str | Sentinel | None = ...,
            allow_none: Literal[True] = ...,
            read_only: bool | None = ...,
            help: str | None = ...,
            config: t.Any = ...,
            **kwargs: t.Any,
        ) -> None:
            ...

        def __init__(
            self: Unicode[str | None, str | bytes | None],
            default_value: str | Sentinel | None = ...,
            allow_none: bool = ...,
            read_only: bool | None = ...,
            help: str | None = ...,
            config: t.Any = ...,
            **kwargs: t.Any,
        ) -> None:
            ...

    def validate(self, obj: t.Any, value: t.Any) -> G:
        if isinstance(value, str):
            return t.cast(G, value)
        if isinstance(value, bytes):
            try:
                return t.cast(G, value.decode("ascii", "strict"))
            except UnicodeDecodeError as e:
                msg = "Could not decode {!r} for unicode trait '{}' of {} instance."
                raise TraitError(msg.format(value, self.name, class_of(obj))) from e
        self.error(obj, value)

    def from_string(self, s: str) -> G:
        if self.allow_none and s == "None":
            return t.cast(G, None)
        s = os.path.expanduser(s)
        if len(s) >= 2:
            # handle deprecated "1"
            for c in ('"', "'"):
                if s[0] == s[-1] == c:
                    old_s = s
                    s = s[1:-1]
                    warn(
                        "Supporting extra quotes around strings is deprecated in traitlets 5.0. "
                        f"You can use {s!r} instead of {old_s!r} if you require traitlets >=5.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
        return t.cast(G, s)

    def subclass_init(self, cls: type[t.Any]) -> None:
        pass  # fully opt out of instance_init


class CUnicode(Unicode[G, S], TraitType[str, t.Any]):
    """A casting version of the unicode trait."""

    if t.TYPE_CHECKING:

        @t.overload
        def __init__(
            self: CUnicode[str, t.Any],
            default_value: str | Sentinel = ...,
            allow_none: Literal[False] = ...,
            read_only: bool | None = ...,
            help: str | None = ...,
            config: t.Any = ...,
            **kwargs: t.Any,
        ) -> None:
            ...

        @t.overload
        def __init__(
            self: CUnicode[str | None, t.Any],
            default_value: str | Sentinel | None = ...,
            allow_none: Literal[True] = ...,
            read_only: bool | None = ...,
            help: str | None = ...,
            config: t.Any = ...,
            **kwargs: t.Any,
        ) -> None:
            ...

        def __init__(
            self: CUnicode[str | None, t.Any],
            default_value: str | Sentinel | None = ...,
            allow_none: bool = ...,
            read_only: bool | None = ...,
            help: str | None = ...,
            config: t.Any = ...,
            **kwargs: t.Any,
        ) -> None:
            ...

    def validate(self, obj: t.Any, value: t.Any) -> G:
        try:
            return t.cast(G, str(value))
        except Exception:
            self.error(obj, value)


class ObjectName(TraitType[str, str]):
    """A string holding a valid object name in this version of Python.

    This does not check that the name exists in any scope."""

    info_text = "a valid object identifier in Python"

    coerce_str = staticmethod(lambda _, s: s)

    def validate(self, obj: t.Any, value: t.Any) -> str:
        value = self.coerce_str(obj, value)

        if isinstance(value, str) and isidentifier(value):
            return value
        self.error(obj, value)

    def from_string(self, s: str) -> str | None:
        if self.allow_none and s == "None":
            return None
        return s


class DottedObjectName(ObjectName):
    """A string holding a valid dotted object name in Python, such as A.b3._c"""

    def validate(self, obj: t.Any, value: t.Any) -> str:
        value = self.coerce_str(obj, value)

        if isinstance(value, str) and all(isidentifier(a) for a in value.split(".")):
            return value
        self.error(obj, value)


class Bool(TraitType[G, S]):
    """A boolean (True, False) trait."""

    default_value = False
    info_text = "a boolean"

    if t.TYPE_CHECKING:

        @t.overload
        def __init__(
            self: Bool[bool, bool | int],
            default_value: bool | Sentinel = ...,
            allow_none: Literal[False] = ...,
            read_only: bool | None = ...,
            help: str | None = ...,
            config: t.Any = ...,
            **kwargs: t.Any,
        ) -> None:
            ...

        @t.overload
        def __init__(
            self: Bool[bool | None, bool | int | None],
            default_value: bool | Sentinel | None = ...,
            allow_none: Literal[True] = ...,
            read_only: bool | None = ...,
            help: str | None = ...,
            config: t.Any = ...,
            **kwargs: t.Any,
        ) -> None:
            ...

        def __init__(
            self: Bool[bool | None, bool | int | None],
            default_value: bool | Sentinel | None = ...,
            allow_none: bool = ...,
            read_only: bool | None = ...,
            help: str | None = ...,
            config: t.Any = ...,
            **kwargs: t.Any,
        ) -> None:
            ...

    def validate(self, obj: t.Any, value: t.Any) -> G:
        if isinstance(value, bool):
            return t.cast(G, value)
        elif isinstance(value, int):
            if value == 1:
                return t.cast(G, True)
            elif value == 0:
                return t.cast(G, False)
        self.error(obj, value)

    def from_string(self, s: str) -> G:
        if self.allow_none and s == "None":
            return t.cast(G, None)
        s = s.lower()
        if s in {"true", "1"}:
            return t.cast(G, True)
        elif s in {"false", "0"}:
            return t.cast(G, False)
        else:
            raise ValueError("%r is not 1, 0, true, or false")

    def subclass_init(self, cls: type[t.Any]) -> None:
        pass  # fully opt out of instance_init

    def argcompleter(self, **kwargs: t.Any) -> list[str]:
        """Completion hints for argcomplete"""
        completions = ["true", "1", "false", "0"]
        if self.allow_none:
            completions.append("None")
        return completions


class CBool(Bool[G, S]):
    """A casting version of the boolean trait."""

    if t.TYPE_CHECKING:

        @t.overload
        def __init__(
            self: CBool[bool, t.Any],
            default_value: bool | Sentinel = ...,
            allow_none: Literal[False] = ...,
            read_only: bool | None = ...,
            help: str | None = ...,
            config: t.Any = ...,
            **kwargs: t.Any,
        ) -> None:
            ...

        @t.overload
        def __init__(
            self: CBool[bool | None, t.Any],
            default_value: bool | Sentinel | None = ...,
            allow_none: Literal[True] = ...,
            read_only: bool | None = ...,
            help: str | None = ...,
            config: t.Any = ...,
            **kwargs: t.Any,
        ) -> None:
            ...

        def __init__(
            self: CBool[bool | None, t.Any],
            default_value: bool | Sentinel | None = ...,
            allow_none: bool = ...,
            read_only: bool | None = ...,
            help: str | None = ...,
            config: t.Any = ...,
            **kwargs: t.Any,
        ) -> None:
            ...

    def validate(self, obj: t.Any, value: t.Any) -> G:
        try:
            return t.cast(G, bool(value))
        except Exception:
            self.error(obj, value)


class Enum(TraitType[G, G]):
    """An enum whose value must be in a given sequence."""

    if t.TYPE_CHECKING:

        @t.overload
        def __init__(
            self: Enum[G],
            values: t.Sequence[G],
            default_value: G | Sentinel = ...,
            allow_none: Literal[False] = ...,
            read_only: bool | None = ...,
            help: str | None = ...,
            config: t.Any = ...,
            **kwargs: t.Any,
        ) -> None:
            ...

        @t.overload
        def __init__(
            self: Enum[G | None],
            values: t.Sequence[G] | None,
            default_value: G | Sentinel | None = ...,
            allow_none: Literal[True] = ...,
            read_only: bool | None = ...,
            help: str | None = ...,
            config: t.Any = ...,
            **kwargs: t.Any,
        ) -> None:
            ...

    def __init__(
        self: Enum[G],
        values: t.Sequence[G] | None,
        default_value: G | Sentinel | None = Undefined,
        allow_none: bool = False,
        read_only: bool | None = None,
        help: str | None = None,
        config: t.Any = None,
        **kwargs: t.Any,
    ) -> None:
        self.values = values
        if allow_none is True and default_value is Undefined:
            default_value = None
        kwargs["allow_none"] = allow_none
        kwargs["read_only"] = read_only
        kwargs["help"] = help
        kwargs["config"] = config
        super().__init__(default_value, **kwargs)

    def validate(self, obj: t.Any, value: t.Any) -> G:
        if self.values and value in self.values:
            return t.cast(G, value)
        self.error(obj, value)

    def _choices_str(self, as_rst: bool = False) -> str:
        """Returns a description of the trait choices (not none)."""
        choices = self.values or []
        if as_rst:
            choice_str = "|".join("``%r``" % x for x in choices)
        else:
            choice_str = repr(list(choices))
        return choice_str

    def _info(self, as_rst: bool = False) -> str:
        """Returns a description of the trait."""
        none = " or %s" % ("`None`" if as_rst else "None") if self.allow_none else ""
        return f"any of {self._choices_str(as_rst)}{none}"

    def info(self) -> str:
        return self._info(as_rst=False)

    def info_rst(self) -> str:
        return self._info(as_rst=True)

    def from_string(self, s: str) -> G:
        try:
            return self.validate(None, s)
        except TraitError:
            return t.cast(G, _safe_literal_eval(s))

    def subclass_init(self, cls: type[t.Any]) -> None:
        pass  # fully opt out of instance_init

    def argcompleter(self, **kwargs: t.Any) -> list[str]:
        """Completion hints for argcomplete"""
        return [str(v) for v in self.values or []]


class CaselessStrEnum(Enum[G]):
    """An enum of strings where the case should be ignored."""

    def __init__(
        self: CaselessStrEnum[t.Any],
        values: t.Any,
        default_value: t.Any = Undefined,
        **kwargs: t.Any,
    ) -> None:
        super().__init__(values, default_value=default_value, **kwargs)

    def validate(self, obj: t.Any, value: t.Any) -> G:
        if not isinstance(value, str):
            self.error(obj, value)

        for v in self.values or []:
            assert isinstance(v, str)
            if v.lower() == value.lower():
                return t.cast(G, v)
        self.error(obj, value)

    def _info(self, as_rst: bool = False) -> str:
        """Returns a description of the trait."""
        none = " or %s" % ("`None`" if as_rst else "None") if self.allow_none else ""
        return f"any of {self._choices_str(as_rst)} (case-insensitive){none}"

    def info(self) -> str:
        return self._info(as_rst=False)

    def info_rst(self) -> str:
        return self._info(as_rst=True)


class FuzzyEnum(Enum[G]):
    """An case-ignoring enum matching choices by unique prefixes/substrings."""

    case_sensitive = False
    #: If True, choices match anywhere in the string, otherwise match prefixes.
    substring_matching = False

    def __init__(
        self: FuzzyEnum[t.Any],
        values: t.Any,
        default_value: t.Any = Undefined,
        case_sensitive: bool = False,
        substring_matching: bool = False,
        **kwargs: t.Any,
    ) -> None:
        self.case_sensitive = case_sensitive
        self.substring_matching = substring_matching
        super().__init__(values, default_value=default_value, **kwargs)

    def validate(self, obj: t.Any, value: t.Any) -> G:
        if not isinstance(value, str):
            self.error(obj, value)

        conv_func = (lambda c: c) if self.case_sensitive else lambda c: c.lower()
        substring_matching = self.substring_matching
        match_func = (lambda v, c: v in c) if substring_matching else (lambda v, c: c.startswith(v))
        value = conv_func(value)  # type:ignore[no-untyped-call]
        choices = self.values or []
        matches = [match_func(value, conv_func(c)) for c in choices]  # type:ignore[no-untyped-call]
        if sum(matches) == 1:
            for v, m in zip(choices, matches):
                if m:
                    return v

        self.error(obj, value)

    def _info(self, as_rst: bool = False) -> str:
        """Returns a description of the trait."""
        none = " or %s" % ("`None`" if as_rst else "None") if self.allow_none else ""
        case = "sensitive" if self.case_sensitive else "insensitive"
        substr = "substring" if self.substring_matching else "prefix"
        return f"any case-{case} {substr} of {self._choices_str(as_rst)}{none}"

    def info(self) -> str:
        return self._info(as_rst=False)

    def info_rst(self) -> str:
        return self._info(as_rst=True)


class Container(Instance[T]):
    """An instance of a container (list, set, etc.)

    To be subclassed by overriding klass.
    """

    klass: type[T] | None = None
    _cast_types: t.Any = ()
    _valid_defaults = SequenceTypes
    _trait: t.Any = None
    _literal_from_string_pairs: t.Any = ("[]", "()")

    @t.overload
    def __init__(
        self: Container[T],
        *,
        allow_none: Literal[False],
        read_only: bool | None = ...,
        help: str | None = ...,
        config: t.Any | None = ...,
        **kwargs: t.Any,
    ) -> None:
        ...

    @t.overload
    def __init__(
        self: Container[T | None],
        *,
        allow_none: Literal[True],
        read_only: bool | None = ...,
        help: str | None = ...,
        config: t.Any | None = ...,
        **kwargs: t.Any,
    ) -> None:
        ...

    @t.overload
    def __init__(
        self: Container[T],
        *,
        trait: t.Any = ...,
        default_value: t.Any = ...,
        help: str = ...,
        read_only: bool = ...,
        config: t.Any = ...,
        **kwargs: t.Any,
    ) -> None:
        ...

    def __init__(
        self,
        trait: t.Any | None = None,
        default_value: t.Any = Undefined,
        help: str | None = None,
        read_only: bool | None = None,
        config: t.Any | None = None,
        **kwargs: t.Any,
    ) -> None:
        """Create a container trait type from a list, set, or tuple.

        The default value is created by doing ``List(default_value)``,
        which creates a copy of the ``default_value``.

        ``trait`` can be specified, which restricts the type of elements
        in the container to that TraitType.

        If only one arg is given and it is not a Trait, it is taken as
        ``default_value``:

        ``c = List([1, 2, 3])``

        Parameters
        ----------
        trait : TraitType [ optional ]
            the type for restricting the contents of the Container.  If unspecified,
            types are not checked.
        default_value : SequenceType [ optional ]
            The default value for the Trait.  Must be list/tuple/set, and
            will be cast to the container type.
        allow_none : bool [ default False ]
            Whether to allow the value to be None
        **kwargs : any
            further keys for extensions to the Trait (e.g. config)

        """

        # allow List([values]):
        if trait is not None and default_value is Undefined and not is_trait(trait):
            default_value = trait
            trait = None

        if default_value is None and not kwargs.get("allow_none", False):
            # improve backward-compatibility for possible subclasses
            # specifying default_value=None as default,
            # keeping 'unspecified' behavior (i.e. empty container)
            warn(
                f"Specifying {self.__class__.__name__}(default_value=None)"
                " for no default is deprecated in traitlets 5.0.5."
                " Use default_value=Undefined",
                DeprecationWarning,
                stacklevel=2,
            )
            default_value = Undefined

        if default_value is Undefined:
            args: t.Any = ()
        elif default_value is None:
            # default_value back on kwargs for super() to handle
            args = ()
            kwargs["default_value"] = None
        elif isinstance(default_value, self._valid_defaults):
            args = (default_value,)
        else:
            raise TypeError(f"default value of {self.__class__.__name__} was {default_value}")

        if is_trait(trait):
            if isinstance(trait, type):
                warn(
                    "Traits should be given as instances, not types (for example, `Int()`, not `Int`)."
                    " Passing types is deprecated in traitlets 4.1.",
                    DeprecationWarning,
                    stacklevel=3,
                )
            self._trait = trait() if isinstance(trait, type) else trait
        elif trait is not None:
            raise TypeError("`trait` must be a Trait or None, got %s" % repr_type(trait))

        super().__init__(
            klass=self.klass, args=args, help=help, read_only=read_only, config=config, **kwargs
        )

    def validate(self, obj: t.Any, value: t.Any) -> T | None:
        if isinstance(value, self._cast_types):
            assert self.klass is not None
            value = self.klass(value)  # type:ignore[call-arg]
        value = super().validate(obj, value)
        if value is None:
            return value

        value = self.validate_elements(obj, value)

        return t.cast(T, value)

    def validate_elements(self, obj: t.Any, value: t.Any) -> T | None:
        validated = []
        if self._trait is None or isinstance(self._trait, Any):
            return t.cast(T, value)
        for v in value:
            try:
                v = self._trait._validate(obj, v)
            except TraitError as error:
                self.error(obj, v, error)
            else:
                validated.append(v)
        assert self.klass is not None
        return self.klass(validated)  # type:ignore[call-arg]

    def class_init(self, cls: type[t.Any], name: str | None) -> None:
        if isinstance(self._trait, TraitType):
            self._trait.class_init(cls, None)
        super().class_init(cls, name)

    def subclass_init(self, cls: type[t.Any]) -> None:
        if isinstance(self._trait, TraitType):
            self._trait.subclass_init(cls)
        # explicitly not calling super().subclass_init(cls)
        # to opt out of instance_init

    def from_string(self, s: str) -> T | None:
        """Load value from a single string"""
        if not isinstance(s, str):
            raise TraitError(f"Expected string, got {s!r}")
        try:
            test = literal_eval(s)
        except Exception:
            test = None
        return self.validate(None, test)

    def from_string_list(self, s_list: list[str]) -> T | None:
        """Return the value from a list of config strings

        This is where we parse CLI configuration
        """
        assert self.klass is not None
        if len(s_list) == 1:
            # check for deprecated --Class.trait="['a', 'b', 'c']"
            r = s_list[0]
            if r == "None" and self.allow_none:
                return None
            if len(r) >= 2 and any(
                r.startswith(start) and r.endswith(end)
                for start, end in self._literal_from_string_pairs
            ):
                if self.this_class:
                    clsname = self.this_class.__name__ + "."
                else:
                    clsname = ""
                assert self.name is not None
                warn(
                    "--{0}={1} for containers is deprecated in traitlets 5.0. "
                    "You can pass `--{0} item` ... multiple times to add items to a list.".format(
                        clsname + self.name, r
                    ),
                    DeprecationWarning,
                    stacklevel=2,
                )
                return self.klass(literal_eval(r))  # type:ignore[call-arg]
        sig = inspect.signature(self.item_from_string)
        if "index" in sig.parameters:
            item_from_string = self.item_from_string
        else:
            # backward-compat: allow item_from_string to ignore index arg
            def item_from_string(s: str, index: int | None = None) -> T | str:
                return t.cast(T, self.item_from_string(s))

        return self.klass(  # type:ignore[call-arg]
            [item_from_string(s, index=idx) for idx, s in enumerate(s_list)]
        )

    def item_from_string(self, s: str, index: int | None = None) -> T | str:
        """Cast a single item from a string

        Evaluated when parsing CLI configuration from a string
        """
        if self._trait:
            return t.cast(T, self._trait.from_string(s))
        else:
            return s


class List(Container[t.List[T]]):
    """An instance of a Python list."""

    klass = list  # type:ignore[assignment]
    _cast_types: t.Any = (tuple,)

    def __init__(
        self,
        trait: t.List[T] | t.Tuple[T] | t.Set[T] | Sentinel | TraitType[T, t.Any] | None = None,
        default_value: t.List[T] | t.Tuple[T] | t.Set[T] | Sentinel | None = Undefined,
        minlen: int = 0,
        maxlen: int = sys.maxsize,
        **kwargs: t.Any,
    ) -> None:
        """Create a List trait type from a list, set, or tuple.

        The default value is created by doing ``list(default_value)``,
        which creates a copy of the ``default_value``.

        ``trait`` can be specified, which restricts the type of elements
        in the container to that TraitType.

        If only one arg is given and it is not a Trait, it is taken as
        ``default_value``:

        ``c = List([1, 2, 3])``

        Parameters
        ----------
        trait : TraitType [ optional ]
            the type for restricting the contents of the Container.
            If unspecified, types are not checked.
        default_value : SequenceType [ optional ]
            The default value for the Trait.  Must be list/tuple/set, and
            will be cast to the container type.
        minlen : Int [ default 0 ]
            The minimum length of the input list
        maxlen : Int [ default sys.maxsize ]
            The maximum length of the input list
        """
        self._maxlen = maxlen
        self._minlen = minlen
        super().__init__(trait=trait, default_value=default_value, **kwargs)

    def length_error(self, obj: t.Any, value: t.Any) -> None:
        e = (
            "The '%s' trait of %s instance must be of length %i <= L <= %i, but a value of %s was specified."
            % (self.name, class_of(obj), self._minlen, self._maxlen, value)
        )
        raise TraitError(e)

    def validate_elements(self, obj: t.Any, value: t.Any) -> t.Any:
        length = len(value)
        if length < self._minlen or length > self._maxlen:
            self.length_error(obj, value)

        return super().validate_elements(obj, value)

    def set(self, obj: t.Any, value: t.Any) -> None:
        if isinstance(value, str):
            return super().set(obj, [value])  # type:ignore[list-item]
        else:
            return super().set(obj, value)


class Set(Container[t.Set[t.Any]]):
    """An instance of a Python set."""

    klass = set
    _cast_types = (tuple, list)

    _literal_from_string_pairs = ("[]", "()", "{}")

    # Redefine __init__ just to make the docstring more accurate.
    def __init__(
        self,
        trait: t.Any = None,
        default_value: t.Any = Undefined,
        minlen: int = 0,
        maxlen: int = sys.maxsize,
        **kwargs: t.Any,
    ) -> None:
        """Create a Set trait type from a list, set, or tuple.

        The default value is created by doing ``set(default_value)``,
        which creates a copy of the ``default_value``.

        ``trait`` can be specified, which restricts the type of elements
        in the container to that TraitType.

        If only one arg is given and it is not a Trait, it is taken as
        ``default_value``:

        ``c = Set({1, 2, 3})``

        Parameters
        ----------
        trait : TraitType [ optional ]
            the type for restricting the contents of the Container.
            If unspecified, types are not checked.
        default_value : SequenceType [ optional ]
            The default value for the Trait.  Must be list/tuple/set, and
            will be cast to the container type.
        minlen : Int [ default 0 ]
            The minimum length of the input list
        maxlen : Int [ default sys.maxsize ]
            The maximum length of the input list
        """
        self._maxlen = maxlen
        self._minlen = minlen
        super().__init__(trait=trait, default_value=default_value, **kwargs)

    def length_error(self, obj: t.Any, value: t.Any) -> None:
        e = (
            "The '%s' trait of %s instance must be of length %i <= L <= %i, but a value of %s was specified."
            % (self.name, class_of(obj), self._minlen, self._maxlen, value)
        )
        raise TraitError(e)

    def validate_elements(self, obj: t.Any, value: t.Any) -> t.Any:
        length = len(value)
        if length < self._minlen or length > self._maxlen:
            self.length_error(obj, value)

        return super().validate_elements(obj, value)

    def set(self, obj: t.Any, value: t.Any) -> None:
        if isinstance(value, str):
            return super().set(
                obj,
                set(
                    value,
                ),
            )
        else:
            return super().set(obj, value)

    def default_value_repr(self) -> str:
        # Ensure default value is sorted for a reproducible build
        list_repr = repr(sorted(self.make_dynamic_default() or []))
        if list_repr == "[]":
            return "set()"
        return "{" + list_repr[1:-1] + "}"


class Tuple(Container[t.Tuple[t.Any, ...]]):
    """An instance of a Python tuple."""

    klass = tuple
    _cast_types = (list,)

    def __init__(self, *traits: t.Any, **kwargs: t.Any) -> None:
        """Create a tuple from a list, set, or tuple.

        Create a fixed-type tuple with Traits:

        ``t = Tuple(Int(), Str(), CStr())``

        would be length 3, with Int,Str,CStr for each element.

        If only one arg is given and it is not a Trait, it is taken as
        default_value:

        ``t = Tuple((1, 2, 3))``

        Otherwise, ``default_value`` *must* be specified by keyword.

        Parameters
        ----------
        *traits : TraitTypes [ optional ]
            the types for restricting the contents of the Tuple.  If unspecified,
            types are not checked. If specified, then each positional argument
            corresponds to an element of the tuple.  Tuples defined with traits
            are of fixed length.
        default_value : SequenceType [ optional ]
            The default value for the Tuple.  Must be list/tuple/set, and
            will be cast to a tuple. If ``traits`` are specified,
            ``default_value`` must conform to the shape and type they specify.
        **kwargs
            Other kwargs passed to `Container`
        """
        default_value = kwargs.pop("default_value", Undefined)
        # allow Tuple((values,)):
        if len(traits) == 1 and default_value is Undefined and not is_trait(traits[0]):
            default_value = traits[0]
            traits = ()

        if default_value is None and not kwargs.get("allow_none", False):
            # improve backward-compatibility for possible subclasses
            # specifying default_value=None as default,
            # keeping 'unspecified' behavior (i.e. empty container)
            warn(
                f"Specifying {self.__class__.__name__}(default_value=None)"
                " for no default is deprecated in traitlets 5.0.5."
                " Use default_value=Undefined",
                DeprecationWarning,
                stacklevel=2,
            )
            default_value = Undefined

        if default_value is Undefined:
            args: t.Any = ()
        elif default_value is None:
            # default_value back on kwargs for super() to handle
            args = ()
            kwargs["default_value"] = None
        elif isinstance(default_value, self._valid_defaults):
            args = (default_value,)
        else:
            raise TypeError(f"default value of {self.__class__.__name__} was {default_value}")

        self._traits = []
        for trait in traits:
            if isinstance(trait, type):
                warn(
                    "Traits should be given as instances, not types (for example, `Int()`, not `Int`)"
                    " Passing types is deprecated in traitlets 4.1.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                trait = trait()
            self._traits.append(trait)

        if self._traits and (default_value is None or default_value is Undefined):
            # don't allow default to be an empty container if length is specified
            args = None
        super(Container, self).__init__(klass=self.klass, args=args, **kwargs)

    def item_from_string(self, s: str, index: int) -> t.Any:  # type:ignore[override]
        """Cast a single item from a string

        Evaluated when parsing CLI configuration from a string
        """
        if not self._traits or index >= len(self._traits):
            # return s instead of raising index error
            # length errors will be raised later on validation
            return s
        return self._traits[index].from_string(s)

    def validate_elements(self, obj: t.Any, value: t.Any) -> t.Any:
        if not self._traits:
            # nothing to validate
            return value
        if len(value) != len(self._traits):
            e = (
                "The '%s' trait of %s instance requires %i elements, but a value of %s was specified."
                % (self.name, class_of(obj), len(self._traits), repr_type(value))
            )
            raise TraitError(e)

        validated = []
        for trait, v in zip(self._traits, value):
            try:
                v = trait._validate(obj, v)
            except TraitError as error:
                self.error(obj, v, error)
            else:
                validated.append(v)
        return tuple(validated)

    def class_init(self, cls: type[t.Any], name: str | None) -> None:
        for trait in self._traits:
            if isinstance(trait, TraitType):
                trait.class_init(cls, None)
        super(Container, self).class_init(cls, name)

    def subclass_init(self, cls: type[t.Any]) -> None:
        for trait in self._traits:
            if isinstance(trait, TraitType):
                trait.subclass_init(cls)
        # explicitly not calling super().subclass_init(cls)
        # to opt out of instance_init


class Dict(Instance["dict[K, V]"]):
    """An instance of a Python dict.

    One or more traits can be passed to the constructor
    to validate the keys and/or values of the dict.
    If you need more detailed validation,
    you may use a custom validator method.

    .. versionchanged:: 5.0
        Added key_trait for validating dict keys.

    .. versionchanged:: 5.0
        Deprecated ambiguous ``trait``, ``traits`` args in favor of ``value_trait``, ``per_key_traits``.
    """

    _value_trait = None
    _key_trait = None

    def __init__(
        self,
        value_trait: TraitType[t.Any, t.Any] | dict[K, V] | Sentinel | None = None,
        per_key_traits: t.Any = None,
        key_trait: TraitType[t.Any, t.Any] | None = None,
        default_value: dict[K, V] | Sentinel | None = Undefined,
        **kwargs: t.Any,
    ) -> None:
        """Create a dict trait type from a Python dict.

        The default value is created by doing ``dict(default_value)``,
        which creates a copy of the ``default_value``.

        Parameters
        ----------
        value_trait : TraitType [ optional ]
            The specified trait type to check and use to restrict the values of
            the dict. If unspecified, values are not checked.
        per_key_traits : Dictionary of {keys:trait types} [ optional, keyword-only ]
            A Python dictionary containing the types that are valid for
            restricting the values of the dict on a per-key basis.
            Each value in this dict should be a Trait for validating
        key_trait : TraitType [ optional, keyword-only ]
            The type for restricting the keys of the dict. If
            unspecified, the types of the keys are not checked.
        default_value : SequenceType [ optional, keyword-only ]
            The default value for the Dict.  Must be dict, tuple, or None, and
            will be cast to a dict if not None. If any key or value traits are specified,
            the `default_value` must conform to the constraints.

        Examples
        --------
        a dict whose values must be text
        >>> d = Dict(Unicode())

        d2['n'] must be an integer
        d2['s'] must be text
        >>> d2 = Dict(per_key_traits={"n": Integer(), "s": Unicode()})

        d3's keys must be text
        d3's values must be integers
        >>> d3 = Dict(value_trait=Integer(), key_trait=Unicode())

        """

        # handle deprecated keywords
        trait = kwargs.pop("trait", None)
        if trait is not None:
            if value_trait is not None:
                raise TypeError(
                    "Found a value for both `value_trait` and its deprecated alias `trait`."
                )
            value_trait = trait
            warn(
                "Keyword `trait` is deprecated in traitlets 5.0, use `value_trait` instead",
                DeprecationWarning,
                stacklevel=2,
            )
        traits = kwargs.pop("traits", None)
        if traits is not None:
            if per_key_traits is not None:
                raise TypeError(
                    "Found a value for both `per_key_traits` and its deprecated alias `traits`."
                )
            per_key_traits = traits
            warn(
                "Keyword `traits` is deprecated in traitlets 5.0, use `per_key_traits` instead",
                DeprecationWarning,
                stacklevel=2,
            )

        # Handling positional arguments
        if default_value is Undefined and value_trait is not None:
            if not is_trait(value_trait):
                assert not isinstance(value_trait, TraitType)
                default_value = value_trait
                value_trait = None

        if key_trait is None and per_key_traits is not None:
            if is_trait(per_key_traits):
                key_trait = per_key_traits
                per_key_traits = None

        # Handling default value
        if default_value is Undefined:
            default_value = {}
        if default_value is None:
            args: t.Any = None
        elif isinstance(default_value, dict):
            args = (default_value,)
        elif isinstance(default_value, SequenceTypes):
            args = (default_value,)
        else:
            raise TypeError("default value of Dict was %s" % default_value)

        # Case where a type of TraitType is provided rather than an instance
        if is_trait(value_trait):
            if isinstance(value_trait, type):
                warn(  # type:ignore[unreachable]
                    "Traits should be given as instances, not types (for example, `Int()`, not `Int`)"
                    " Passing types is deprecated in traitlets 4.1.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                value_trait = value_trait()
            self._value_trait = value_trait
        elif value_trait is not None:
            raise TypeError(
                "`value_trait` must be a Trait or None, got %s" % repr_type(value_trait)
            )

        if is_trait(key_trait):
            if isinstance(key_trait, type):
                warn(  # type:ignore[unreachable]
                    "Traits should be given as instances, not types (for example, `Int()`, not `Int`)"
                    " Passing types is deprecated in traitlets 4.1.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                key_trait = key_trait()
            self._key_trait = key_trait
        elif key_trait is not None:
            raise TypeError("`key_trait` must be a Trait or None, got %s" % repr_type(key_trait))

        self._per_key_traits = per_key_traits

        super().__init__(klass=dict, args=args, **kwargs)

    def element_error(
        self, obj: t.Any, element: t.Any, validator: t.Any, side: str = "Values"
    ) -> None:
        e = (
            side
            + f" of the '{self.name}' trait of {class_of(obj)} instance must be {validator.info()}, but a value of {repr_type(element)} was specified."
        )
        raise TraitError(e)

    def validate(self, obj: t.Any, value: t.Any) -> dict[K, V] | None:
        value = super().validate(obj, value)
        if value is None:
            return value
        return self.validate_elements(obj, value)

    def validate_elements(self, obj: t.Any, value: dict[t.Any, t.Any]) -> dict[K, V] | None:
        per_key_override = self._per_key_traits or {}
        key_trait = self._key_trait
        value_trait = self._value_trait
        if not (key_trait or value_trait or per_key_override):
            return value

        validated = {}
        for key in value:
            v = value[key]
            if key_trait:
                try:
                    key = key_trait._validate(obj, key)
                except TraitError:
                    self.element_error(obj, key, key_trait, "Keys")
            active_value_trait = per_key_override.get(key, value_trait)
            if active_value_trait:
                try:
                    v = active_value_trait._validate(obj, v)
                except TraitError:
                    self.element_error(obj, v, active_value_trait, "Values")
            validated[key] = v

        return self.klass(validated)  # type:ignore[misc,operator]

    def class_init(self, cls: type[t.Any], name: str | None) -> None:
        if isinstance(self._value_trait, TraitType):
            self._value_trait.class_init(cls, None)
        if isinstance(self._key_trait, TraitType):
            self._key_trait.class_init(cls, None)
        if self._per_key_traits is not None:
            for trait in self._per_key_traits.values():
                trait.class_init(cls, None)
        super().class_init(cls, name)

    def subclass_init(self, cls: type[t.Any]) -> None:
        if isinstance(self._value_trait, TraitType):
            self._value_trait.subclass_init(cls)
        if isinstance(self._key_trait, TraitType):
            self._key_trait.subclass_init(cls)
        if self._per_key_traits is not None:
            for trait in self._per_key_traits.values():
                trait.subclass_init(cls)
        # explicitly not calling super().subclass_init(cls)
        # to opt out of instance_init

    def from_string(self, s: str) -> dict[K, V] | None:
        """Load value from a single string"""
        if not isinstance(s, str):
            raise TypeError(f"from_string expects a string, got {s!r} of type {type(s)}")
        try:
            return t.cast("dict[K, V]", self.from_string_list([s]))
        except Exception:
            test = _safe_literal_eval(s)
            if isinstance(test, dict):
                return test
            raise

    def from_string_list(self, s_list: list[str]) -> t.Any:
        """Return a dict from a list of config strings.

        This is where we parse CLI configuration.

        Each item should have the form ``"key=value"``.

        item parsing is done in :meth:`.item_from_string`.
        """
        if len(s_list) == 1 and s_list[0] == "None" and self.allow_none:
            return None
        if len(s_list) == 1 and s_list[0].startswith("{") and s_list[0].endswith("}"):
            warn(
                f"--{self.name}={s_list[0]} for dict-traits is deprecated in traitlets 5.0. "
                f"You can pass --{self.name} <key=value> ... multiple times to add items to a dict.",
                DeprecationWarning,
                stacklevel=2,
            )

            return literal_eval(s_list[0])

        combined = {}
        for d in [self.item_from_string(s) for s in s_list]:
            combined.update(d)
        return combined

    def item_from_string(self, s: str) -> dict[K, V]:
        """Cast a single-key dict from a string.

        Evaluated when parsing CLI configuration from a string.

        Dicts expect strings of the form key=value.

        Returns a one-key dictionary,
        which will be merged in :meth:`.from_string_list`.
        """

        if "=" not in s:
            raise TraitError(
                f"'{self.__class__.__name__}' options must have the form 'key=value', got {s!r}"
            )
        key, value = s.split("=", 1)

        # cast key with key trait, if defined
        if self._key_trait:
            key = self._key_trait.from_string(key)

        # cast value with value trait, if defined (per-key or global)
        value_trait = (self._per_key_traits or {}).get(key, self._value_trait)
        if value_trait:
            value = value_trait.from_string(value)
        return t.cast("dict[K, V]", {key: value})


class TCPAddress(TraitType[G, S]):
    """A trait for an (ip, port) tuple.

    This allows for both IPv4 IP addresses as well as hostnames.
    """

    default_value = ("127.0.0.1", 0)
    info_text = "an (ip, port) tuple"

    if t.TYPE_CHECKING:

        @t.overload
        def __init__(
            self: TCPAddress[tuple[str, int], tuple[str, int]],
            default_value: bool | Sentinel = ...,
            allow_none: Literal[False] = ...,
            read_only: bool | None = ...,
            help: str | None = ...,
            config: t.Any = ...,
            **kwargs: t.Any,
        ) -> None:
            ...

        @t.overload
        def __init__(
            self: TCPAddress[tuple[str, int] | None, tuple[str, int] | None],
            default_value: bool | None | Sentinel = ...,
            allow_none: Literal[True] = ...,
            read_only: bool | None = ...,
            help: str | None = ...,
            config: t.Any = ...,
            **kwargs: t.Any,
        ) -> None:
            ...

        def __init__(
            self: TCPAddress[tuple[str, int] | None, tuple[str, int] | None]
            | TCPAddress[tuple[str, int], tuple[str, int]],
            default_value: bool | None | Sentinel = Undefined,
            allow_none: Literal[True, False] = False,
            read_only: bool | None = None,
            help: str | None = None,
            config: t.Any = None,
            **kwargs: t.Any,
        ) -> None:
            ...

    def validate(self, obj: t.Any, value: t.Any) -> G:
        if isinstance(value, tuple):
            if len(value) == 2:
                if isinstance(value[0], str) and isinstance(value[1], int):
                    port = value[1]
                    if port >= 0 and port <= 65535:
                        return t.cast(G, value)
        self.error(obj, value)

    def from_string(self, s: str) -> G:
        if self.allow_none and s == "None":
            return t.cast(G, None)
        if ":" not in s:
            raise ValueError("Require `ip:port`, got %r" % s)
        ip, port_str = s.split(":", 1)
        port = int(port_str)
        return t.cast(G, (ip, port))


class CRegExp(TraitType["re.Pattern[t.Any]", t.Union["re.Pattern[t.Any]", str]]):
    """A casting compiled regular expression trait.

    Accepts both strings and compiled regular expressions. The resulting
    attribute will be a compiled regular expression."""

    info_text = "a regular expression"

    def validate(self, obj: t.Any, value: t.Any) -> re.Pattern[t.Any] | None:
        try:
            return re.compile(value)
        except Exception:
            self.error(obj, value)


class UseEnum(TraitType[t.Any, t.Any]):
    """Use a Enum class as model for the data type description.
    Note that if no default-value is provided, the first enum-value is used
    as default-value.

    .. sourcecode:: python

        # -- SINCE: Python 3.4 (or install backport: pip install enum34)
        import enum
        from traitlets import HasTraits, UseEnum


        class Color(enum.Enum):
            red = 1  # -- IMPLICIT: default_value
            blue = 2
            green = 3


        class MyEntity(HasTraits):
            color = UseEnum(Color, default_value=Color.blue)


        entity = MyEntity(color=Color.red)
        entity.color = Color.green  # USE: Enum-value (preferred)
        entity.color = "green"  # USE: name (as string)
        entity.color = "Color.green"  # USE: scoped-name (as string)
        entity.color = 3  # USE: number (as int)
        assert entity.color is Color.green
    """

    default_value: enum.Enum | None = None
    info_text = "Trait type adapter to a Enum class"

    def __init__(
        self, enum_class: type[t.Any], default_value: t.Any = None, **kwargs: t.Any
    ) -> None:
        assert issubclass(enum_class, enum.Enum), "REQUIRE: enum.Enum, but was: %r" % enum_class
        allow_none = kwargs.get("allow_none", False)
        if default_value is None and not allow_none:
            default_value = next(iter(enum_class.__members__.values()))
        super().__init__(default_value=default_value, **kwargs)
        self.enum_class = enum_class
        self.name_prefix = enum_class.__name__ + "."

    def select_by_number(self, value: int, default: t.Any = Undefined) -> t.Any:
        """Selects enum-value by using its number-constant."""
        assert isinstance(value, int)
        enum_members = self.enum_class.__members__
        for enum_item in enum_members.values():
            if enum_item.value == value:
                return enum_item
        # -- NOT FOUND:
        return default

    def select_by_name(self, value: str, default: t.Any = Undefined) -> t.Any:
        """Selects enum-value by using its name or scoped-name."""
        assert isinstance(value, str)
        if value.startswith(self.name_prefix):
            # -- SUPPORT SCOPED-NAMES, like: "Color.red" => "red"
            value = value.replace(self.name_prefix, "", 1)
        return self.enum_class.__members__.get(value, default)

    def validate(self, obj: t.Any, value: t.Any) -> t.Any:
        if isinstance(value, self.enum_class):
            return value
        elif isinstance(value, int):
            # -- CONVERT: number => enum_value (item)
            value2 = self.select_by_number(value)
            if value2 is not Undefined:
                return value2
        elif isinstance(value, str):
            # -- CONVERT: name or scoped_name (as string) => enum_value (item)
            value2 = self.select_by_name(value)
            if value2 is not Undefined:
                return value2
        elif value is None:
            if self.allow_none:
                return None
            else:
                return self.default_value
        self.error(obj, value)

    def _choices_str(self, as_rst: bool = False) -> str:
        """Returns a description of the trait choices (not none)."""
        choices = self.enum_class.__members__.keys()
        if as_rst:
            return "|".join("``%r``" % x for x in choices)
        else:
            return repr(list(choices))  # Listify because py3.4- prints odict-class

    def _info(self, as_rst: bool = False) -> str:
        """Returns a description of the trait."""
        none = " or %s" % ("`None`" if as_rst else "None") if self.allow_none else ""
        return f"any of {self._choices_str(as_rst)}{none}"

    def info(self) -> str:
        return self._info(as_rst=False)

    def info_rst(self) -> str:
        return self._info(as_rst=True)


class Callable(TraitType[t.Callable[..., t.Any], t.Callable[..., t.Any]]):
    """A trait which is callable.

    Notes
    -----
    Classes are callable, as are instances
    with a __call__() method."""

    info_text = "a callable"

    def validate(self, obj: t.Any, value: t.Any) -> t.Any:
        if callable(value):
            return value
        else:
            self.error(obj, value)
