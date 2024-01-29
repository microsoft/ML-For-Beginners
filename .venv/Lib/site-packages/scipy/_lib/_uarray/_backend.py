import typing
import types
import inspect
import functools
from . import _uarray
import copyreg
import pickle
import contextlib

from ._uarray import (  # type: ignore
    BackendNotImplementedError,
    _Function,
    _SkipBackendContext,
    _SetBackendContext,
    _BackendState,
)

__all__ = [
    "set_backend",
    "set_global_backend",
    "skip_backend",
    "register_backend",
    "determine_backend",
    "determine_backend_multi",
    "clear_backends",
    "create_multimethod",
    "generate_multimethod",
    "_Function",
    "BackendNotImplementedError",
    "Dispatchable",
    "wrap_single_convertor",
    "wrap_single_convertor_instance",
    "all_of_type",
    "mark_as",
    "set_state",
    "get_state",
    "reset_state",
    "_BackendState",
    "_SkipBackendContext",
    "_SetBackendContext",
]

ArgumentExtractorType = typing.Callable[..., tuple["Dispatchable", ...]]
ArgumentReplacerType = typing.Callable[
    [tuple, dict, tuple], tuple[tuple, dict]
]

def unpickle_function(mod_name, qname, self_):
    import importlib

    try:
        module = importlib.import_module(mod_name)
        qname = qname.split(".")
        func = module
        for q in qname:
            func = getattr(func, q)

        if self_ is not None:
            func = types.MethodType(func, self_)

        return func
    except (ImportError, AttributeError) as e:
        from pickle import UnpicklingError

        raise UnpicklingError from e


def pickle_function(func):
    mod_name = getattr(func, "__module__", None)
    qname = getattr(func, "__qualname__", None)
    self_ = getattr(func, "__self__", None)

    try:
        test = unpickle_function(mod_name, qname, self_)
    except pickle.UnpicklingError:
        test = None

    if test is not func:
        raise pickle.PicklingError(
            f"Can't pickle {func}: it's not the same object as {test}"
        )

    return unpickle_function, (mod_name, qname, self_)


def pickle_state(state):
    return _uarray._BackendState._unpickle, state._pickle()


def pickle_set_backend_context(ctx):
    return _SetBackendContext, ctx._pickle()


def pickle_skip_backend_context(ctx):
    return _SkipBackendContext, ctx._pickle()


copyreg.pickle(_Function, pickle_function)
copyreg.pickle(_uarray._BackendState, pickle_state)
copyreg.pickle(_SetBackendContext, pickle_set_backend_context)
copyreg.pickle(_SkipBackendContext, pickle_skip_backend_context)


def get_state():
    """
    Returns an opaque object containing the current state of all the backends.

    Can be used for synchronization between threads/processes.

    See Also
    --------
    set_state
        Sets the state returned by this function.
    """
    return _uarray.get_state()


@contextlib.contextmanager
def reset_state():
    """
    Returns a context manager that resets all state once exited.

    See Also
    --------
    set_state
        Context manager that sets the backend state.
    get_state
        Gets a state to be set by this context manager.
    """
    with set_state(get_state()):
        yield


@contextlib.contextmanager
def set_state(state):
    """
    A context manager that sets the state of the backends to one returned by :obj:`get_state`.

    See Also
    --------
    get_state
        Gets a state to be set by this context manager.
    """  # noqa: E501
    old_state = get_state()
    _uarray.set_state(state)
    try:
        yield
    finally:
        _uarray.set_state(old_state, True)


def create_multimethod(*args, **kwargs):
    """
    Creates a decorator for generating multimethods.

    This function creates a decorator that can be used with an argument
    extractor in order to generate a multimethod. Other than for the
    argument extractor, all arguments are passed on to
    :obj:`generate_multimethod`.

    See Also
    --------
    generate_multimethod
        Generates a multimethod.
    """

    def wrapper(a):
        return generate_multimethod(a, *args, **kwargs)

    return wrapper


def generate_multimethod(
    argument_extractor: ArgumentExtractorType,
    argument_replacer: ArgumentReplacerType,
    domain: str,
    default: typing.Optional[typing.Callable] = None,
):
    """
    Generates a multimethod.

    Parameters
    ----------
    argument_extractor : ArgumentExtractorType
        A callable which extracts the dispatchable arguments. Extracted arguments
        should be marked by the :obj:`Dispatchable` class. It has the same signature
        as the desired multimethod.
    argument_replacer : ArgumentReplacerType
        A callable with the signature (args, kwargs, dispatchables), which should also
        return an (args, kwargs) pair with the dispatchables replaced inside the
        args/kwargs.
    domain : str
        A string value indicating the domain of this multimethod.
    default: Optional[Callable], optional
        The default implementation of this multimethod, where ``None`` (the default)
        specifies there is no default implementation.

    Examples
    --------
    In this example, ``a`` is to be dispatched over, so we return it, while marking it
    as an ``int``.
    The trailing comma is needed because the args have to be returned as an iterable.

    >>> def override_me(a, b):
    ...   return Dispatchable(a, int),

    Next, we define the argument replacer that replaces the dispatchables inside
    args/kwargs with the supplied ones.

    >>> def override_replacer(args, kwargs, dispatchables):
    ...     return (dispatchables[0], args[1]), {}

    Next, we define the multimethod.

    >>> overridden_me = generate_multimethod(
    ...     override_me, override_replacer, "ua_examples"
    ... )

    Notice that there's no default implementation, unless you supply one.

    >>> overridden_me(1, "a")
    Traceback (most recent call last):
        ...
    uarray.BackendNotImplementedError: ...

    >>> overridden_me2 = generate_multimethod(
    ...     override_me, override_replacer, "ua_examples", default=lambda x, y: (x, y)
    ... )
    >>> overridden_me2(1, "a")
    (1, 'a')

    See Also
    --------
    uarray
        See the module documentation for how to override the method by creating
        backends.
    """
    kw_defaults, arg_defaults, opts = get_defaults(argument_extractor)
    ua_func = _Function(
        argument_extractor,
        argument_replacer,
        domain,
        arg_defaults,
        kw_defaults,
        default,
    )

    return functools.update_wrapper(ua_func, argument_extractor)


def set_backend(backend, coerce=False, only=False):
    """
    A context manager that sets the preferred backend.

    Parameters
    ----------
    backend
        The backend to set.
    coerce
        Whether or not to coerce to a specific backend's types. Implies ``only``.
    only
        Whether or not this should be the last backend to try.

    See Also
    --------
    skip_backend: A context manager that allows skipping of backends.
    set_global_backend: Set a single, global backend for a domain.
    """
    try:
        return backend.__ua_cache__["set", coerce, only]
    except AttributeError:
        backend.__ua_cache__ = {}
    except KeyError:
        pass

    ctx = _SetBackendContext(backend, coerce, only)
    backend.__ua_cache__["set", coerce, only] = ctx
    return ctx


def skip_backend(backend):
    """
    A context manager that allows one to skip a given backend from processing
    entirely. This allows one to use another backend's code in a library that
    is also a consumer of the same backend.

    Parameters
    ----------
    backend
        The backend to skip.

    See Also
    --------
    set_backend: A context manager that allows setting of backends.
    set_global_backend: Set a single, global backend for a domain.
    """
    try:
        return backend.__ua_cache__["skip"]
    except AttributeError:
        backend.__ua_cache__ = {}
    except KeyError:
        pass

    ctx = _SkipBackendContext(backend)
    backend.__ua_cache__["skip"] = ctx
    return ctx


def get_defaults(f):
    sig = inspect.signature(f)
    kw_defaults = {}
    arg_defaults = []
    opts = set()
    for k, v in sig.parameters.items():
        if v.default is not inspect.Parameter.empty:
            kw_defaults[k] = v.default
        if v.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            arg_defaults.append(v.default)
        opts.add(k)

    return kw_defaults, tuple(arg_defaults), opts


def set_global_backend(backend, coerce=False, only=False, *, try_last=False):
    """
    This utility method replaces the default backend for permanent use. It
    will be tried in the list of backends automatically, unless the
    ``only`` flag is set on a backend. This will be the first tried
    backend outside the :obj:`set_backend` context manager.

    Note that this method is not thread-safe.

    .. warning::
        We caution library authors against using this function in
        their code. We do *not* support this use-case. This function
        is meant to be used only by users themselves, or by a reference
        implementation, if one exists.

    Parameters
    ----------
    backend
        The backend to register.
    coerce : bool
        Whether to coerce input types when trying this backend.
    only : bool
        If ``True``, no more backends will be tried if this fails.
        Implied by ``coerce=True``.
    try_last : bool
        If ``True``, the global backend is tried after registered backends.

    See Also
    --------
    set_backend: A context manager that allows setting of backends.
    skip_backend: A context manager that allows skipping of backends.
    """
    _uarray.set_global_backend(backend, coerce, only, try_last)


def register_backend(backend):
    """
    This utility method sets registers backend for permanent use. It
    will be tried in the list of backends automatically, unless the
    ``only`` flag is set on a backend.

    Note that this method is not thread-safe.

    Parameters
    ----------
    backend
        The backend to register.
    """
    _uarray.register_backend(backend)


def clear_backends(domain, registered=True, globals=False):
    """
    This utility method clears registered backends.

    .. warning::
        We caution library authors against using this function in
        their code. We do *not* support this use-case. This function
        is meant to be used only by users themselves.

    .. warning::
        Do NOT use this method inside a multimethod call, or the
        program is likely to crash.

    Parameters
    ----------
    domain : Optional[str]
        The domain for which to de-register backends. ``None`` means
        de-register for all domains.
    registered : bool
        Whether or not to clear registered backends. See :obj:`register_backend`.
    globals : bool
        Whether or not to clear global backends. See :obj:`set_global_backend`.

    See Also
    --------
    register_backend : Register a backend globally.
    set_global_backend : Set a global backend.
    """
    _uarray.clear_backends(domain, registered, globals)


class Dispatchable:
    """
    A utility class which marks an argument with a specific dispatch type.


    Attributes
    ----------
    value
        The value of the Dispatchable.

    type
        The type of the Dispatchable.

    Examples
    --------
    >>> x = Dispatchable(1, str)
    >>> x
    <Dispatchable: type=<class 'str'>, value=1>

    See Also
    --------
    all_of_type
        Marks all unmarked parameters of a function.

    mark_as
        Allows one to create a utility function to mark as a given type.
    """

    def __init__(self, value, dispatch_type, coercible=True):
        self.value = value
        self.type = dispatch_type
        self.coercible = coercible

    def __getitem__(self, index):
        return (self.type, self.value)[index]

    def __str__(self):
        return f"<{type(self).__name__}: type={self.type!r}, value={self.value!r}>"

    __repr__ = __str__


def mark_as(dispatch_type):
    """
    Creates a utility function to mark something as a specific type.

    Examples
    --------
    >>> mark_int = mark_as(int)
    >>> mark_int(1)
    <Dispatchable: type=<class 'int'>, value=1>
    """
    return functools.partial(Dispatchable, dispatch_type=dispatch_type)


def all_of_type(arg_type):
    """
    Marks all unmarked arguments as a given type.

    Examples
    --------
    >>> @all_of_type(str)
    ... def f(a, b):
    ...     return a, Dispatchable(b, int)
    >>> f('a', 1)
    (<Dispatchable: type=<class 'str'>, value='a'>,
     <Dispatchable: type=<class 'int'>, value=1>)
    """

    def outer(func):
        @functools.wraps(func)
        def inner(*args, **kwargs):
            extracted_args = func(*args, **kwargs)
            return tuple(
                Dispatchable(arg, arg_type)
                if not isinstance(arg, Dispatchable)
                else arg
                for arg in extracted_args
            )

        return inner

    return outer


def wrap_single_convertor(convert_single):
    """
    Wraps a ``__ua_convert__`` defined for a single element to all elements.
    If any of them return ``NotImplemented``, the operation is assumed to be
    undefined.

    Accepts a signature of (value, type, coerce).
    """

    @functools.wraps(convert_single)
    def __ua_convert__(dispatchables, coerce):
        converted = []
        for d in dispatchables:
            c = convert_single(d.value, d.type, coerce and d.coercible)

            if c is NotImplemented:
                return NotImplemented

            converted.append(c)

        return converted

    return __ua_convert__


def wrap_single_convertor_instance(convert_single):
    """
    Wraps a ``__ua_convert__`` defined for a single element to all elements.
    If any of them return ``NotImplemented``, the operation is assumed to be
    undefined.

    Accepts a signature of (value, type, coerce).
    """

    @functools.wraps(convert_single)
    def __ua_convert__(self, dispatchables, coerce):
        converted = []
        for d in dispatchables:
            c = convert_single(self, d.value, d.type, coerce and d.coercible)

            if c is NotImplemented:
                return NotImplemented

            converted.append(c)

        return converted

    return __ua_convert__


def determine_backend(value, dispatch_type, *, domain, only=True, coerce=False):
    """Set the backend to the first active backend that supports ``value``

    This is useful for functions that call multimethods without any dispatchable
    arguments. You can use :func:`determine_backend` to ensure the same backend
    is used everywhere in a block of multimethod calls.

    Parameters
    ----------
    value
        The value being tested
    dispatch_type
        The dispatch type associated with ``value``, aka
        ":ref:`marking <MarkingGlossary>`".
    domain: string
        The domain to query for backends and set.
    coerce: bool
        Whether or not to allow coercion to the backend's types. Implies ``only``.
    only: bool
        Whether or not this should be the last backend to try.

    See Also
    --------
    set_backend: For when you know which backend to set

    Notes
    -----

    Support is determined by the ``__ua_convert__`` protocol. Backends not
    supporting the type must return ``NotImplemented`` from their
    ``__ua_convert__`` if they don't support input of that type.

    Examples
    --------

    Suppose we have two backends ``BackendA`` and ``BackendB`` each supporting
    different types, ``TypeA`` and ``TypeB``. Neither supporting the other type:

    >>> with ua.set_backend(ex.BackendA):
    ...     ex.call_multimethod(ex.TypeB(), ex.TypeB())
    Traceback (most recent call last):
        ...
    uarray.BackendNotImplementedError: ...

    Now consider a multimethod that creates a new object of ``TypeA``, or
    ``TypeB`` depending on the active backend.

    >>> with ua.set_backend(ex.BackendA), ua.set_backend(ex.BackendB):
    ...         res = ex.creation_multimethod()
    ...         ex.call_multimethod(res, ex.TypeA())
    Traceback (most recent call last):
        ...
    uarray.BackendNotImplementedError: ...

    ``res`` is an object of ``TypeB`` because ``BackendB`` is set in the
    innermost with statement. So, ``call_multimethod`` fails since the types
    don't match.

    Instead, we need to first find a backend suitable for all of our objects.

    >>> with ua.set_backend(ex.BackendA), ua.set_backend(ex.BackendB):
    ...     x = ex.TypeA()
    ...     with ua.determine_backend(x, "mark", domain="ua_examples"):
    ...         res = ex.creation_multimethod()
    ...         ex.call_multimethod(res, x)
    TypeA

    """
    dispatchables = (Dispatchable(value, dispatch_type, coerce),)
    backend = _uarray.determine_backend(domain, dispatchables, coerce)

    return set_backend(backend, coerce=coerce, only=only)


def determine_backend_multi(
    dispatchables, *, domain, only=True, coerce=False, **kwargs
):
    """Set a backend supporting all ``dispatchables``

    This is useful for functions that call multimethods without any dispatchable
    arguments. You can use :func:`determine_backend_multi` to ensure the same
    backend is used everywhere in a block of multimethod calls involving
    multiple arrays.

    Parameters
    ----------
    dispatchables: Sequence[Union[uarray.Dispatchable, Any]]
        The dispatchables that must be supported
    domain: string
        The domain to query for backends and set.
    coerce: bool
        Whether or not to allow coercion to the backend's types. Implies ``only``.
    only: bool
        Whether or not this should be the last backend to try.
    dispatch_type: Optional[Any]
        The default dispatch type associated with ``dispatchables``, aka
        ":ref:`marking <MarkingGlossary>`".

    See Also
    --------
    determine_backend: For a single dispatch value
    set_backend: For when you know which backend to set

    Notes
    -----

    Support is determined by the ``__ua_convert__`` protocol. Backends not
    supporting the type must return ``NotImplemented`` from their
    ``__ua_convert__`` if they don't support input of that type.

    Examples
    --------

    :func:`determine_backend` allows the backend to be set from a single
    object. :func:`determine_backend_multi` allows multiple objects to be
    checked simultaneously for support in the backend. Suppose we have a
    ``BackendAB`` which supports ``TypeA`` and ``TypeB`` in the same call,
    and a ``BackendBC`` that doesn't support ``TypeA``.

    >>> with ua.set_backend(ex.BackendAB), ua.set_backend(ex.BackendBC):
    ...     a, b = ex.TypeA(), ex.TypeB()
    ...     with ua.determine_backend_multi(
    ...         [ua.Dispatchable(a, "mark"), ua.Dispatchable(b, "mark")],
    ...         domain="ua_examples"
    ...     ):
    ...         res = ex.creation_multimethod()
    ...         ex.call_multimethod(res, a, b)
    TypeA

    This won't call ``BackendBC`` because it doesn't support ``TypeA``.

    We can also use leave out the ``ua.Dispatchable`` if we specify the
    default ``dispatch_type`` for the ``dispatchables`` argument.

    >>> with ua.set_backend(ex.BackendAB), ua.set_backend(ex.BackendBC):
    ...     a, b = ex.TypeA(), ex.TypeB()
    ...     with ua.determine_backend_multi(
    ...         [a, b], dispatch_type="mark", domain="ua_examples"
    ...     ):
    ...         res = ex.creation_multimethod()
    ...         ex.call_multimethod(res, a, b)
    TypeA

    """
    if "dispatch_type" in kwargs:
        disp_type = kwargs.pop("dispatch_type")
        dispatchables = tuple(
            d if isinstance(d, Dispatchable) else Dispatchable(d, disp_type)
            for d in dispatchables
        )
    else:
        dispatchables = tuple(dispatchables)
        if not all(isinstance(d, Dispatchable) for d in dispatchables):
            raise TypeError("dispatchables must be instances of uarray.Dispatchable")

    if len(kwargs) != 0:
        raise TypeError(f"Received unexpected keyword arguments: {kwargs}")

    backend = _uarray.determine_backend(domain, dispatchables, coerce)

    return set_backend(backend, coerce=coerce, only=only)
