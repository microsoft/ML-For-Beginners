"""
Metadata Routing Utility

In order to better understand the components implemented in this file, one
needs to understand their relationship to one another.

The only relevant public API for end users are the ``set_{method}_request``,
e.g. ``estimator.set_fit_request(sample_weight=True)``. However, third-party
developers and users who implement custom meta-estimators, need to deal with
the objects implemented in this file.

All estimators (should) implement a ``get_metadata_routing`` method, returning
the routing requests set for the estimator. This method is automatically
implemented via ``BaseEstimator`` for all simple estimators, but needs a custom
implementation for meta-estimators.

In non-routing consumers, i.e. the simplest case, e.g. ``SVM``,
``get_metadata_routing`` returns a ``MetadataRequest`` object.

In routers, e.g. meta-estimators and a multi metric scorer,
``get_metadata_routing`` returns a ``MetadataRouter`` object.

An object which is both a router and a consumer, e.g. a meta-estimator which
consumes ``sample_weight`` and routes ``sample_weight`` to its sub-estimators,
routing information includes both information about the object itself (added
via ``MetadataRouter.add_self_request``), as well as the routing information
for its sub-estimators.

A ``MetadataRequest`` instance includes one ``MethodMetadataRequest`` per
method in ``METHODS``, which includes ``fit``, ``score``, etc.

Request values are added to the routing mechanism by adding them to
``MethodMetadataRequest`` instances, e.g.
``metadatarequest.fit.add(param="sample_weight", alias="my_weights")``. This is
used in ``set_{method}_request`` which are automatically generated, so users
and developers almost never need to directly call methods on a
``MethodMetadataRequest``.

The ``alias`` above in the ``add`` method has to be either a string (an alias),
or a {True (requested), False (unrequested), None (error if passed)}``. There
are some other special values such as ``UNUSED`` and ``WARN`` which are used
for purposes such as warning of removing a metadata in a child class, but not
used by the end users.

``MetadataRouter`` includes information about sub-objects' routing and how
methods are mapped together. For instance, the information about which methods
of a sub-estimator are called in which methods of the meta-estimator are all
stored here. Conceptually, this information looks like:

```
{
    "sub_estimator1": (
        mapping=[(caller="fit", callee="transform"), ...],
        router=MetadataRequest(...),  # or another MetadataRouter
    ),
    ...
}
```

To give the above representation some structure, we use the following objects:

- ``(caller, callee)`` is a namedtuple called ``MethodPair``

- The list of ``MethodPair`` stored in the ``mapping`` field is a
  ``MethodMapping`` object

- ``(mapping=..., router=...)`` is a namedtuple called ``RouterMappingPair``

The ``set_{method}_request`` methods are dynamically generated for estimators
which inherit from the ``BaseEstimator``. This is done by attaching instances
of the ``RequestMethod`` descriptor to classes, which is done in the
``_MetadataRequester`` class, and ``BaseEstimator`` inherits from this mixin.
This mixin also implements the ``get_metadata_routing``, which meta-estimators
need to override, but it works for simple consumers as is.
"""

# Author: Adrin Jalali <adrin.jalali@gmail.com>
# License: BSD 3 clause

import inspect
from collections import namedtuple
from copy import deepcopy
from typing import TYPE_CHECKING, Optional, Union
from warnings import warn

from .. import get_config
from ..exceptions import UnsetMetadataPassedError
from ._bunch import Bunch

# Only the following methods are supported in the routing mechanism. Adding new
# methods at the moment involves monkeypatching this list.
# Note that if this list is changed or monkeypatched, the corresponding method
# needs to be added under a TYPE_CHECKING condition like the one done here in
# _MetadataRequester
SIMPLE_METHODS = [
    "fit",
    "partial_fit",
    "predict",
    "predict_proba",
    "predict_log_proba",
    "decision_function",
    "score",
    "split",
    "transform",
    "inverse_transform",
]

# These methods are a composite of other methods and one cannot set their
# requests directly. Instead they should be set by setting the requests of the
# simple methods which make the composite ones.
COMPOSITE_METHODS = {
    "fit_transform": ["fit", "transform"],
    "fit_predict": ["fit", "predict"],
}

METHODS = SIMPLE_METHODS + list(COMPOSITE_METHODS.keys())


def _routing_enabled():
    """Return whether metadata routing is enabled.

    .. versionadded:: 1.3

    Returns
    -------
    enabled : bool
        Whether metadata routing is enabled. If the config is not set, it
        defaults to False.
    """
    return get_config().get("enable_metadata_routing", False)


def _raise_for_params(params, owner, method):
    """Raise an error if metadata routing is not enabled and params are passed.

    .. versionadded:: 1.4

    Parameters
    ----------
    params : dict
        The metadata passed to a method.

    owner : object
        The object to which the method belongs.

    method : str
        The name of the method, e.g. "fit".

    Raises
    ------
    ValueError
        If metadata routing is not enabled and params are passed.
    """
    caller = (
        f"{owner.__class__.__name__}.{method}" if method else owner.__class__.__name__
    )
    if not _routing_enabled() and params:
        raise ValueError(
            f"Passing extra keyword arguments to {caller} is only supported if"
            " enable_metadata_routing=True, which you can set using"
            " `sklearn.set_config`. See the User Guide"
            " <https://scikit-learn.org/stable/metadata_routing.html> for more"
            f" details. Extra parameters passed are: {set(params)}"
        )


def _raise_for_unsupported_routing(obj, method, **kwargs):
    """Raise when metadata routing is enabled and metadata is passed.

    This is used in meta-estimators which have not implemented metadata routing
    to prevent silent bugs. There is no need to use this function if the
    meta-estimator is not accepting any metadata, especially in `fit`, since
    if a meta-estimator accepts any metadata, they would do that in `fit` as
    well.

    Parameters
    ----------
    obj : estimator
        The estimator for which we're raising the error.

    method : str
        The method where the error is raised.

    **kwargs : dict
        The metadata passed to the method.
    """
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    if _routing_enabled() and kwargs:
        cls_name = obj.__class__.__name__
        raise NotImplementedError(
            f"{cls_name}.{method} cannot accept given metadata ({set(kwargs.keys())})"
            f" since metadata routing is not yet implemented for {cls_name}."
        )


class _RoutingNotSupportedMixin:
    """A mixin to be used to remove the default `get_metadata_routing`.

    This is used in meta-estimators where metadata routing is not yet
    implemented.

    This also makes it clear in our rendered documentation that this method
    cannot be used.
    """

    def get_metadata_routing(self):
        """Raise `NotImplementedError`.

        This estimator does not support metadata routing yet."""
        raise NotImplementedError(
            f"{self.__class__.__name__} has not implemented metadata routing yet."
        )


# Request values
# ==============
# Each request value needs to be one of the following values, or an alias.

# this is used in `__metadata_request__*` attributes to indicate that a
# metadata is not present even though it may be present in the
# corresponding method's signature.
UNUSED = "$UNUSED$"

# this is used whenever a default value is changed, and therefore the user
# should explicitly set the value, otherwise a warning is shown. An example
# is when a meta-estimator is only a router, but then becomes also a
# consumer in a new release.
WARN = "$WARN$"

# this is the default used in `set_{method}_request` methods to indicate no
# change requested by the user.
UNCHANGED = "$UNCHANGED$"

VALID_REQUEST_VALUES = [False, True, None, UNUSED, WARN]


def request_is_alias(item):
    """Check if an item is a valid alias.

    Values in ``VALID_REQUEST_VALUES`` are not considered aliases in this
    context. Only a string which is a valid identifier is.

    Parameters
    ----------
    item : object
        The given item to be checked if it can be an alias.

    Returns
    -------
    result : bool
        Whether the given item is a valid alias.
    """
    if item in VALID_REQUEST_VALUES:
        return False

    # item is only an alias if it's a valid identifier
    return isinstance(item, str) and item.isidentifier()


def request_is_valid(item):
    """Check if an item is a valid request value (and not an alias).

    Parameters
    ----------
    item : object
        The given item to be checked.

    Returns
    -------
    result : bool
        Whether the given item is valid.
    """
    return item in VALID_REQUEST_VALUES


# Metadata Request for Simple Consumers
# =====================================
# This section includes MethodMetadataRequest and MetadataRequest which are
# used in simple consumers.


class MethodMetadataRequest:
    """A prescription of how metadata is to be passed to a single method.

    Refer to :class:`MetadataRequest` for how this class is used.

    .. versionadded:: 1.3

    Parameters
    ----------
    owner : str
        A display name for the object owning these requests.

    method : str
        The name of the method to which these requests belong.

    requests : dict of {str: bool, None or str}, default=None
        The initial requests for this method.
    """

    def __init__(self, owner, method, requests=None):
        self._requests = requests or dict()
        self.owner = owner
        self.method = method

    @property
    def requests(self):
        """Dictionary of the form: ``{key: alias}``."""
        return self._requests

    def add_request(
        self,
        *,
        param,
        alias,
    ):
        """Add request info for a metadata.

        Parameters
        ----------
        param : str
            The property for which a request is set.

        alias : str, or {True, False, None}
            Specifies which metadata should be routed to `param`

            - str: the name (or alias) of metadata given to a meta-estimator that
              should be routed to this parameter.

            - True: requested

            - False: not requested

            - None: error if passed
        """
        if not request_is_alias(alias) and not request_is_valid(alias):
            raise ValueError(
                f"The alias you're setting for `{param}` should be either a "
                "valid identifier or one of {None, True, False}, but given "
                f"value is: `{alias}`"
            )

        if alias == param:
            alias = True

        if alias == UNUSED:
            if param in self._requests:
                del self._requests[param]
            else:
                raise ValueError(
                    f"Trying to remove parameter {param} with UNUSED which doesn't"
                    " exist."
                )
        else:
            self._requests[param] = alias

        return self

    def _get_param_names(self, return_alias):
        """Get names of all metadata that can be consumed or routed by this method.

        This method returns the names of all metadata, even the ``False``
        ones.

        Parameters
        ----------
        return_alias : bool
            Controls whether original or aliased names should be returned. If
            ``False``, aliases are ignored and original names are returned.

        Returns
        -------
        names : set of str
            A set of strings with the names of all parameters.
        """
        return set(
            alias if return_alias and not request_is_valid(alias) else prop
            for prop, alias in self._requests.items()
            if not request_is_valid(alias) or alias is not False
        )

    def _check_warnings(self, *, params):
        """Check whether metadata is passed which is marked as WARN.

        If any metadata is passed which is marked as WARN, a warning is raised.

        Parameters
        ----------
        params : dict
            The metadata passed to a method.
        """
        params = {} if params is None else params
        warn_params = {
            prop
            for prop, alias in self._requests.items()
            if alias == WARN and prop in params
        }
        for param in warn_params:
            warn(
                f"Support for {param} has recently been added to this class. "
                "To maintain backward compatibility, it is ignored now. "
                "You can set the request value to False to silence this "
                "warning, or to True to consume and use the metadata."
            )

    def _route_params(self, params):
        """Prepare the given parameters to be passed to the method.

        The output of this method can be used directly as the input to the
        corresponding method as extra props.

        Parameters
        ----------
        params : dict
            A dictionary of provided metadata.

        Returns
        -------
        params : Bunch
            A :class:`~sklearn.utils.Bunch` of {prop: value} which can be given to the
            corresponding method.
        """
        self._check_warnings(params=params)
        unrequested = dict()
        args = {arg: value for arg, value in params.items() if value is not None}
        res = Bunch()
        for prop, alias in self._requests.items():
            if alias is False or alias == WARN:
                continue
            elif alias is True and prop in args:
                res[prop] = args[prop]
            elif alias is None and prop in args:
                unrequested[prop] = args[prop]
            elif alias in args:
                res[prop] = args[alias]
        if unrequested:
            raise UnsetMetadataPassedError(
                message=(
                    f"[{', '.join([key for key in unrequested])}] are passed but are"
                    " not explicitly set as requested or not for"
                    f" {self.owner}.{self.method}"
                ),
                unrequested_params=unrequested,
                routed_params=res,
            )
        return res

    def _consumes(self, params):
        """Check whether the given parameters are consumed by this method.

        Parameters
        ----------
        params : iterable of str
            An iterable of parameters to check.

        Returns
        -------
        consumed : set of str
            A set of parameters which are consumed by this method.
        """
        params = set(params)
        res = set()
        for prop, alias in self._requests.items():
            if alias is True and prop in params:
                res.add(prop)
            elif isinstance(alias, str) and alias in params:
                res.add(alias)
        return res

    def _serialize(self):
        """Serialize the object.

        Returns
        -------
        obj : dict
            A serialized version of the instance in the form of a dictionary.
        """
        return self._requests

    def __repr__(self):
        return str(self._serialize())

    def __str__(self):
        return str(repr(self))


class MetadataRequest:
    """Contains the metadata request info of a consumer.

    Instances of `MethodMetadataRequest` are used in this class for each
    available method under `metadatarequest.{method}`.

    Consumer-only classes such as simple estimators return a serialized
    version of this class as the output of `get_metadata_routing()`.

    .. versionadded:: 1.3

    Parameters
    ----------
    owner : str
        The name of the object to which these requests belong.
    """

    # this is here for us to use this attribute's value instead of doing
    # `isinstance` in our checks, so that we avoid issues when people vendor
    # this file instead of using it directly from scikit-learn.
    _type = "metadata_request"

    def __init__(self, owner):
        self.owner = owner
        for method in SIMPLE_METHODS:
            setattr(
                self,
                method,
                MethodMetadataRequest(owner=owner, method=method),
            )

    def consumes(self, method, params):
        """Check whether the given parameters are consumed by the given method.

        .. versionadded:: 1.4

        Parameters
        ----------
        method : str
            The name of the method to check.

        params : iterable of str
            An iterable of parameters to check.

        Returns
        -------
        consumed : set of str
            A set of parameters which are consumed by the given method.
        """
        return getattr(self, method)._consumes(params=params)

    def __getattr__(self, name):
        # Called when the default attribute access fails with an AttributeError
        # (either __getattribute__() raises an AttributeError because name is
        # not an instance attribute or an attribute in the class tree for self;
        # or __get__() of a name property raises AttributeError). This method
        # should either return the (computed) attribute value or raise an
        # AttributeError exception.
        # https://docs.python.org/3/reference/datamodel.html#object.__getattr__
        if name not in COMPOSITE_METHODS:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )

        requests = {}
        for method in COMPOSITE_METHODS[name]:
            mmr = getattr(self, method)
            existing = set(requests.keys())
            upcoming = set(mmr.requests.keys())
            common = existing & upcoming
            conflicts = [key for key in common if requests[key] != mmr._requests[key]]
            if conflicts:
                raise ValueError(
                    f"Conflicting metadata requests for {', '.join(conflicts)} while"
                    f" composing the requests for {name}. Metadata with the same name"
                    f" for methods {', '.join(COMPOSITE_METHODS[name])} should have the"
                    " same request value."
                )
            requests.update(mmr._requests)
        return MethodMetadataRequest(owner=self.owner, method=name, requests=requests)

    def _get_param_names(self, method, return_alias, ignore_self_request=None):
        """Get names of all metadata that can be consumed or routed by specified \
            method.

        This method returns the names of all metadata, even the ``False``
        ones.

        Parameters
        ----------
        method : str
            The name of the method for which metadata names are requested.

        return_alias : bool
            Controls whether original or aliased names should be returned. If
            ``False``, aliases are ignored and original names are returned.

        ignore_self_request : bool
            Ignored. Present for API compatibility.

        Returns
        -------
        names : set of str
            A set of strings with the names of all parameters.
        """
        return getattr(self, method)._get_param_names(return_alias=return_alias)

    def _route_params(self, *, method, params):
        """Prepare the given parameters to be passed to the method.

        The output of this method can be used directly as the input to the
        corresponding method as extra keyword arguments to pass metadata.

        Parameters
        ----------
        method : str
            The name of the method for which the parameters are requested and
            routed.

        params : dict
            A dictionary of provided metadata.

        Returns
        -------
        params : Bunch
            A :class:`~sklearn.utils.Bunch` of {prop: value} which can be given to the
            corresponding method.
        """
        return getattr(self, method)._route_params(params=params)

    def _check_warnings(self, *, method, params):
        """Check whether metadata is passed which is marked as WARN.

        If any metadata is passed which is marked as WARN, a warning is raised.

        Parameters
        ----------
        method : str
            The name of the method for which the warnings should be checked.

        params : dict
            The metadata passed to a method.
        """
        getattr(self, method)._check_warnings(params=params)

    def _serialize(self):
        """Serialize the object.

        Returns
        -------
        obj : dict
            A serialized version of the instance in the form of a dictionary.
        """
        output = dict()
        for method in SIMPLE_METHODS:
            mmr = getattr(self, method)
            if len(mmr.requests):
                output[method] = mmr._serialize()
        return output

    def __repr__(self):
        return str(self._serialize())

    def __str__(self):
        return str(repr(self))


# Metadata Request for Routers
# ============================
# This section includes all objects required for MetadataRouter which is used
# in routers, returned by their ``get_metadata_routing``.

# This namedtuple is used to store a (mapping, routing) pair. Mapping is a
# MethodMapping object, and routing is the output of `get_metadata_routing`.
# MetadataRouter stores a collection of these namedtuples.
RouterMappingPair = namedtuple("RouterMappingPair", ["mapping", "router"])

# A namedtuple storing a single method route. A collection of these namedtuples
# is stored in a MetadataRouter.
MethodPair = namedtuple("MethodPair", ["callee", "caller"])


class MethodMapping:
    """Stores the mapping between callee and caller methods for a router.

    This class is primarily used in a ``get_metadata_routing()`` of a router
    object when defining the mapping between a sub-object (a sub-estimator or a
    scorer) to the router's methods. It stores a collection of ``Route``
    namedtuples.

    Iterating through an instance of this class will yield named
    ``MethodPair(callee, caller)`` tuples.

    .. versionadded:: 1.3
    """

    def __init__(self):
        self._routes = []

    def __iter__(self):
        return iter(self._routes)

    def add(self, *, callee, caller):
        """Add a method mapping.

        Parameters
        ----------
        callee : str
            Child object's method name. This method is called in ``caller``.

        caller : str
            Parent estimator's method name in which the ``callee`` is called.

        Returns
        -------
        self : MethodMapping
            Returns self.
        """
        if callee not in METHODS:
            raise ValueError(
                f"Given callee:{callee} is not a valid method. Valid methods are:"
                f" {METHODS}"
            )
        if caller not in METHODS:
            raise ValueError(
                f"Given caller:{caller} is not a valid method. Valid methods are:"
                f" {METHODS}"
            )
        self._routes.append(MethodPair(callee=callee, caller=caller))
        return self

    def _serialize(self):
        """Serialize the object.

        Returns
        -------
        obj : list
            A serialized version of the instance in the form of a list.
        """
        result = list()
        for route in self._routes:
            result.append({"callee": route.callee, "caller": route.caller})
        return result

    @classmethod
    def from_str(cls, route):
        """Construct an instance from a string.

        Parameters
        ----------
        route : str
            A string representing the mapping, it can be:

              - `"one-to-one"`: a one to one mapping for all methods.
              - `"method"`: the name of a single method, such as ``fit``,
                ``transform``, ``score``, etc.

        Returns
        -------
        obj : MethodMapping
            A :class:`~sklearn.utils.metadata_routing.MethodMapping` instance
            constructed from the given string.
        """
        routing = cls()
        if route == "one-to-one":
            for method in METHODS:
                routing.add(callee=method, caller=method)
        elif route in METHODS:
            routing.add(callee=route, caller=route)
        else:
            raise ValueError("route should be 'one-to-one' or a single method!")
        return routing

    def __repr__(self):
        return str(self._serialize())

    def __str__(self):
        return str(repr(self))


class MetadataRouter:
    """Stores and handles metadata routing for a router object.

    This class is used by router objects to store and handle metadata routing.
    Routing information is stored as a dictionary of the form ``{"object_name":
    RouteMappingPair(method_mapping, routing_info)}``, where ``method_mapping``
    is an instance of :class:`~sklearn.utils.metadata_routing.MethodMapping` and
    ``routing_info`` is either a
    :class:`~sklearn.utils.metadata_routing.MetadataRequest` or a
    :class:`~sklearn.utils.metadata_routing.MetadataRouter` instance.

    .. versionadded:: 1.3

    Parameters
    ----------
    owner : str
        The name of the object to which these requests belong.
    """

    # this is here for us to use this attribute's value instead of doing
    # `isinstance`` in our checks, so that we avoid issues when people vendor
    # this file instead of using it directly from scikit-learn.
    _type = "metadata_router"

    def __init__(self, owner):
        self._route_mappings = dict()
        # `_self_request` is used if the router is also a consumer.
        # _self_request, (added using `add_self_request()`) is treated
        # differently from the other objects which are stored in
        # _route_mappings.
        self._self_request = None
        self.owner = owner

    def add_self_request(self, obj):
        """Add `self` (as a consumer) to the routing.

        This method is used if the router is also a consumer, and hence the
        router itself needs to be included in the routing. The passed object
        can be an estimator or a
        :class:`~sklearn.utils.metadata_routing.MetadataRequest`.

        A router should add itself using this method instead of `add` since it
        should be treated differently than the other objects to which metadata
        is routed by the router.

        Parameters
        ----------
        obj : object
            This is typically the router instance, i.e. `self` in a
            ``get_metadata_routing()`` implementation. It can also be a
            ``MetadataRequest`` instance.

        Returns
        -------
        self : MetadataRouter
            Returns `self`.
        """
        if getattr(obj, "_type", None) == "metadata_request":
            self._self_request = deepcopy(obj)
        elif hasattr(obj, "_get_metadata_request"):
            self._self_request = deepcopy(obj._get_metadata_request())
        else:
            raise ValueError(
                "Given `obj` is neither a `MetadataRequest` nor does it implement the"
                " required API. Inheriting from `BaseEstimator` implements the required"
                " API."
            )
        return self

    def add(self, *, method_mapping, **objs):
        """Add named objects with their corresponding method mapping.

        Parameters
        ----------
        method_mapping : MethodMapping or str
            The mapping between the child and the parent's methods. If str, the
            output of :func:`~sklearn.utils.metadata_routing.MethodMapping.from_str`
            is used.

        **objs : dict
            A dictionary of objects from which metadata is extracted by calling
            :func:`~sklearn.utils.metadata_routing.get_routing_for_object` on them.

        Returns
        -------
        self : MetadataRouter
            Returns `self`.
        """
        if isinstance(method_mapping, str):
            method_mapping = MethodMapping.from_str(method_mapping)
        else:
            method_mapping = deepcopy(method_mapping)

        for name, obj in objs.items():
            self._route_mappings[name] = RouterMappingPair(
                mapping=method_mapping, router=get_routing_for_object(obj)
            )
        return self

    def consumes(self, method, params):
        """Check whether the given parameters are consumed by the given method.

        .. versionadded:: 1.4

        Parameters
        ----------
        method : str
            The name of the method to check.

        params : iterable of str
            An iterable of parameters to check.

        Returns
        -------
        consumed : set of str
            A set of parameters which are consumed by the given method.
        """
        res = set()
        if self._self_request:
            res = res | self._self_request.consumes(method=method, params=params)

        for _, route_mapping in self._route_mappings.items():
            for callee, caller in route_mapping.mapping:
                if caller == method:
                    res = res | route_mapping.router.consumes(
                        method=callee, params=params
                    )

        return res

    def _get_param_names(self, *, method, return_alias, ignore_self_request):
        """Get names of all metadata that can be consumed or routed by specified \
            method.

        This method returns the names of all metadata, even the ``False``
        ones.

        Parameters
        ----------
        method : str
            The name of the method for which metadata names are requested.

        return_alias : bool
            Controls whether original or aliased names should be returned,
            which only applies to the stored `self`. If no `self` routing
            object is stored, this parameter has no effect.

        ignore_self_request : bool
            If `self._self_request` should be ignored. This is used in `_route_params`.
            If ``True``, ``return_alias`` has no effect.

        Returns
        -------
        names : set of str
            A set of strings with the names of all parameters.
        """
        res = set()
        if self._self_request and not ignore_self_request:
            res = res.union(
                self._self_request._get_param_names(
                    method=method, return_alias=return_alias
                )
            )

        for name, route_mapping in self._route_mappings.items():
            for callee, caller in route_mapping.mapping:
                if caller == method:
                    res = res.union(
                        route_mapping.router._get_param_names(
                            method=callee, return_alias=True, ignore_self_request=False
                        )
                    )
        return res

    def _route_params(self, *, params, method):
        """Prepare the given parameters to be passed to the method.

        This is used when a router is used as a child object of another router.
        The parent router then passes all parameters understood by the child
        object to it and delegates their validation to the child.

        The output of this method can be used directly as the input to the
        corresponding method as extra props.

        Parameters
        ----------
        method : str
            The name of the method for which the parameters are requested and
            routed.

        params : dict
            A dictionary of provided metadata.

        Returns
        -------
        params : Bunch
            A :class:`~sklearn.utils.Bunch` of {prop: value} which can be given to the
            corresponding method.
        """
        res = Bunch()
        if self._self_request:
            res.update(self._self_request._route_params(params=params, method=method))

        param_names = self._get_param_names(
            method=method, return_alias=True, ignore_self_request=True
        )
        child_params = {
            key: value for key, value in params.items() if key in param_names
        }
        for key in set(res.keys()).intersection(child_params.keys()):
            # conflicts are okay if the passed objects are the same, but it's
            # an issue if they're different objects.
            if child_params[key] is not res[key]:
                raise ValueError(
                    f"In {self.owner}, there is a conflict on {key} between what is"
                    " requested for this estimator and what is requested by its"
                    " children. You can resolve this conflict by using an alias for"
                    " the child estimator(s) requested metadata."
                )

        res.update(child_params)
        return res

    def route_params(self, *, caller, params):
        """Return the input parameters requested by child objects.

        The output of this method is a bunch, which includes the inputs for all
        methods of each child object that are used in the router's `caller`
        method.

        If the router is also a consumer, it also checks for warnings of
        `self`'s/consumer's requested metadata.

        Parameters
        ----------
        caller : str
            The name of the method for which the parameters are requested and
            routed. If called inside the :term:`fit` method of a router, it
            would be `"fit"`.

        params : dict
            A dictionary of provided metadata.

        Returns
        -------
        params : Bunch
            A :class:`~sklearn.utils.Bunch` of the form
            ``{"object_name": {"method_name": {prop: value}}}`` which can be
            used to pass the required metadata to corresponding methods or
            corresponding child objects.
        """
        if self._self_request:
            self._self_request._check_warnings(params=params, method=caller)

        res = Bunch()
        for name, route_mapping in self._route_mappings.items():
            router, mapping = route_mapping.router, route_mapping.mapping

            res[name] = Bunch()
            for _callee, _caller in mapping:
                if _caller == caller:
                    res[name][_callee] = router._route_params(
                        params=params, method=_callee
                    )
        return res

    def validate_metadata(self, *, method, params):
        """Validate given metadata for a method.

        This raises a ``TypeError`` if some of the passed metadata are not
        understood by child objects.

        Parameters
        ----------
        method : str
            The name of the method for which the parameters are requested and
            routed. If called inside the :term:`fit` method of a router, it
            would be `"fit"`.

        params : dict
            A dictionary of provided metadata.
        """
        param_names = self._get_param_names(
            method=method, return_alias=False, ignore_self_request=False
        )
        if self._self_request:
            self_params = self._self_request._get_param_names(
                method=method, return_alias=False
            )
        else:
            self_params = set()
        extra_keys = set(params.keys()) - param_names - self_params
        if extra_keys:
            raise TypeError(
                f"{self.owner}.{method} got unexpected argument(s) {extra_keys}, which"
                " are not requested metadata in any object."
            )

    def _serialize(self):
        """Serialize the object.

        Returns
        -------
        obj : dict
            A serialized version of the instance in the form of a dictionary.
        """
        res = dict()
        if self._self_request:
            res["$self_request"] = self._self_request._serialize()
        for name, route_mapping in self._route_mappings.items():
            res[name] = dict()
            res[name]["mapping"] = route_mapping.mapping._serialize()
            res[name]["router"] = route_mapping.router._serialize()

        return res

    def __iter__(self):
        if self._self_request:
            yield "$self_request", RouterMappingPair(
                mapping=MethodMapping.from_str("one-to-one"), router=self._self_request
            )
        for name, route_mapping in self._route_mappings.items():
            yield (name, route_mapping)

    def __repr__(self):
        return str(self._serialize())

    def __str__(self):
        return str(repr(self))


def get_routing_for_object(obj=None):
    """Get a ``Metadata{Router, Request}`` instance from the given object.

    This function returns a
    :class:`~sklearn.utils.metadata_routing.MetadataRouter` or a
    :class:`~sklearn.utils.metadata_routing.MetadataRequest` from the given input.

    This function always returns a copy or an instance constructed from the
    input, such that changing the output of this function will not change the
    original object.

    .. versionadded:: 1.3

    Parameters
    ----------
    obj : object
        - If the object is already a
            :class:`~sklearn.utils.metadata_routing.MetadataRequest` or a
            :class:`~sklearn.utils.metadata_routing.MetadataRouter`, return a copy
            of that.
        - If the object provides a `get_metadata_routing` method, return a copy
            of the output of that method.
        - Returns an empty :class:`~sklearn.utils.metadata_routing.MetadataRequest`
            otherwise.

    Returns
    -------
    obj : MetadataRequest or MetadataRouting
        A ``MetadataRequest`` or a ``MetadataRouting`` taken or created from
        the given object.
    """
    # doing this instead of a try/except since an AttributeError could be raised
    # for other reasons.
    if hasattr(obj, "get_metadata_routing"):
        return deepcopy(obj.get_metadata_routing())

    elif getattr(obj, "_type", None) in ["metadata_request", "metadata_router"]:
        return deepcopy(obj)

    return MetadataRequest(owner=None)


# Request method
# ==============
# This section includes what's needed for the request method descriptor and
# their dynamic generation in a meta class.

# These strings are used to dynamically generate the docstrings for
# set_{method}_request methods.
REQUESTER_DOC = """        Request metadata passed to the ``{method}`` method.

        Note that this method is only relevant if
        ``enable_metadata_routing=True`` (see :func:`sklearn.set_config`).
        Please see :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        The options for each parameter are:

        - ``True``: metadata is requested, and \
passed to ``{method}`` if provided. The request is ignored if \
metadata is not provided.

        - ``False``: metadata is not requested and the meta-estimator \
will not pass it to ``{method}``.

        - ``None``: metadata is not requested, and the meta-estimator \
will raise an error if the user provides it.

        - ``str``: metadata should be passed to the meta-estimator with \
this given alias instead of the original name.

        The default (``sklearn.utils.metadata_routing.UNCHANGED``) retains the
        existing request. This allows you to change the request for some
        parameters and not others.

        .. versionadded:: 1.3

        .. note::
            This method is only relevant if this estimator is used as a
            sub-estimator of a meta-estimator, e.g. used inside a
            :class:`~sklearn.pipeline.Pipeline`. Otherwise it has no effect.

        Parameters
        ----------
"""
REQUESTER_DOC_PARAM = """        {metadata} : str, True, False, or None, \
                    default=sklearn.utils.metadata_routing.UNCHANGED
            Metadata routing for ``{metadata}`` parameter in ``{method}``.

"""
REQUESTER_DOC_RETURN = """        Returns
        -------
        self : object
            The updated object.
"""


class RequestMethod:
    """
    A descriptor for request methods.

    .. versionadded:: 1.3

    Parameters
    ----------
    name : str
        The name of the method for which the request function should be
        created, e.g. ``"fit"`` would create a ``set_fit_request`` function.

    keys : list of str
        A list of strings which are accepted parameters by the created
        function, e.g. ``["sample_weight"]`` if the corresponding method
        accepts it as a metadata.

    validate_keys : bool, default=True
        Whether to check if the requested parameters fit the actual parameters
        of the method.

    Notes
    -----
    This class is a descriptor [1]_ and uses PEP-362 to set the signature of
    the returned function [2]_.

    References
    ----------
    .. [1] https://docs.python.org/3/howto/descriptor.html

    .. [2] https://www.python.org/dev/peps/pep-0362/
    """

    def __init__(self, name, keys, validate_keys=True):
        self.name = name
        self.keys = keys
        self.validate_keys = validate_keys

    def __get__(self, instance, owner):
        # we would want to have a method which accepts only the expected args
        def func(**kw):
            """Updates the request for provided parameters

            This docstring is overwritten below.
            See REQUESTER_DOC for expected functionality
            """
            if not _routing_enabled():
                raise RuntimeError(
                    "This method is only available when metadata routing is enabled."
                    " You can enable it using"
                    " sklearn.set_config(enable_metadata_routing=True)."
                )

            if self.validate_keys and (set(kw) - set(self.keys)):
                raise TypeError(
                    f"Unexpected args: {set(kw) - set(self.keys)}. Accepted arguments"
                    f" are: {set(self.keys)}"
                )

            requests = instance._get_metadata_request()
            method_metadata_request = getattr(requests, self.name)

            for prop, alias in kw.items():
                if alias is not UNCHANGED:
                    method_metadata_request.add_request(param=prop, alias=alias)
            instance._metadata_request = requests

            return instance

        # Now we set the relevant attributes of the function so that it seems
        # like a normal method to the end user, with known expected arguments.
        func.__name__ = f"set_{self.name}_request"
        params = [
            inspect.Parameter(
                name="self",
                kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=owner,
            )
        ]
        params.extend(
            [
                inspect.Parameter(
                    k,
                    inspect.Parameter.KEYWORD_ONLY,
                    default=UNCHANGED,
                    annotation=Optional[Union[bool, None, str]],
                )
                for k in self.keys
            ]
        )
        func.__signature__ = inspect.Signature(
            params,
            return_annotation=owner,
        )
        doc = REQUESTER_DOC.format(method=self.name)
        for metadata in self.keys:
            doc += REQUESTER_DOC_PARAM.format(metadata=metadata, method=self.name)
        doc += REQUESTER_DOC_RETURN
        func.__doc__ = doc
        return func


class _MetadataRequester:
    """Mixin class for adding metadata request functionality.

    ``BaseEstimator`` inherits from this Mixin.

    .. versionadded:: 1.3
    """

    if TYPE_CHECKING:  # pragma: no cover
        # This code is never run in runtime, but it's here for type checking.
        # Type checkers fail to understand that the `set_{method}_request`
        # methods are dynamically generated, and they complain that they are
        # not defined. We define them here to make type checkers happy.
        # During type checking analyzers assume this to be True.
        # The following list of defined methods mirrors the list of methods
        # in SIMPLE_METHODS.
        # fmt: off
        def set_fit_request(self, **kwargs): pass
        def set_partial_fit_request(self, **kwargs): pass
        def set_predict_request(self, **kwargs): pass
        def set_predict_proba_request(self, **kwargs): pass
        def set_predict_log_proba_request(self, **kwargs): pass
        def set_decision_function_request(self, **kwargs): pass
        def set_score_request(self, **kwargs): pass
        def set_split_request(self, **kwargs): pass
        def set_transform_request(self, **kwargs): pass
        def set_inverse_transform_request(self, **kwargs): pass
        # fmt: on

    def __init_subclass__(cls, **kwargs):
        """Set the ``set_{method}_request`` methods.

        This uses PEP-487 [1]_ to set the ``set_{method}_request`` methods. It
        looks for the information available in the set default values which are
        set using ``__metadata_request__*`` class attributes, or inferred
        from method signatures.

        The ``__metadata_request__*`` class attributes are used when a method
        does not explicitly accept a metadata through its arguments or if the
        developer would like to specify a request value for those metadata
        which are different from the default ``None``.

        References
        ----------
        .. [1] https://www.python.org/dev/peps/pep-0487
        """
        try:
            requests = cls._get_default_requests()
        except Exception:
            # if there are any issues in the default values, it will be raised
            # when ``get_metadata_routing`` is called. Here we are going to
            # ignore all the issues such as bad defaults etc.
            super().__init_subclass__(**kwargs)
            return

        for method in SIMPLE_METHODS:
            mmr = getattr(requests, method)
            # set ``set_{method}_request``` methods
            if not len(mmr.requests):
                continue
            setattr(
                cls,
                f"set_{method}_request",
                RequestMethod(method, sorted(mmr.requests.keys())),
            )
        super().__init_subclass__(**kwargs)

    @classmethod
    def _build_request_for_signature(cls, router, method):
        """Build the `MethodMetadataRequest` for a method using its signature.

        This method takes all arguments from the method signature and uses
        ``None`` as their default request value, except ``X``, ``y``, ``Y``,
        ``Xt``, ``yt``, ``*args``, and ``**kwargs``.

        Parameters
        ----------
        router : MetadataRequest
            The parent object for the created `MethodMetadataRequest`.
        method : str
            The name of the method.

        Returns
        -------
        method_request : MethodMetadataRequest
            The prepared request using the method's signature.
        """
        mmr = MethodMetadataRequest(owner=cls.__name__, method=method)
        # Here we use `isfunction` instead of `ismethod` because calling `getattr`
        # on a class instead of an instance returns an unbound function.
        if not hasattr(cls, method) or not inspect.isfunction(getattr(cls, method)):
            return mmr
        # ignore the first parameter of the method, which is usually "self"
        params = list(inspect.signature(getattr(cls, method)).parameters.items())[1:]
        for pname, param in params:
            if pname in {"X", "y", "Y", "Xt", "yt"}:
                continue
            if param.kind in {param.VAR_POSITIONAL, param.VAR_KEYWORD}:
                continue
            mmr.add_request(
                param=pname,
                alias=None,
            )
        return mmr

    @classmethod
    def _get_default_requests(cls):
        """Collect default request values.

        This method combines the information present in ``__metadata_request__*``
        class attributes, as well as determining request keys from method
        signatures.
        """
        requests = MetadataRequest(owner=cls.__name__)

        for method in SIMPLE_METHODS:
            setattr(
                requests,
                method,
                cls._build_request_for_signature(router=requests, method=method),
            )

        # Then overwrite those defaults with the ones provided in
        # __metadata_request__* attributes. Defaults set in
        # __metadata_request__* attributes take precedence over signature
        # sniffing.

        # need to go through the MRO since this is a class attribute and
        # ``vars`` doesn't report the parent class attributes. We go through
        # the reverse of the MRO so that child classes have precedence over
        # their parents.
        defaults = dict()
        for base_class in reversed(inspect.getmro(cls)):
            base_defaults = {
                attr: value
                for attr, value in vars(base_class).items()
                if "__metadata_request__" in attr
            }
            defaults.update(base_defaults)
        defaults = dict(sorted(defaults.items()))

        for attr, value in defaults.items():
            # we don't check for attr.startswith() since python prefixes attrs
            # starting with __ with the `_ClassName`.
            substr = "__metadata_request__"
            method = attr[attr.index(substr) + len(substr) :]
            for prop, alias in value.items():
                getattr(requests, method).add_request(param=prop, alias=alias)

        return requests

    def _get_metadata_request(self):
        """Get requested data properties.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        Returns
        -------
        request : MetadataRequest
            A :class:`~sklearn.utils.metadata_routing.MetadataRequest` instance.
        """
        if hasattr(self, "_metadata_request"):
            requests = get_routing_for_object(self._metadata_request)
        else:
            requests = self._get_default_requests()

        return requests

    def get_metadata_routing(self):
        """Get metadata routing of this object.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        Returns
        -------
        routing : MetadataRequest
            A :class:`~sklearn.utils.metadata_routing.MetadataRequest` encapsulating
            routing information.
        """
        return self._get_metadata_request()


# Process Routing in Routers
# ==========================
# This is almost always the only method used in routers to process and route
# given metadata. This is to minimize the boilerplate required in routers.


# Here the first two arguments are positional only which makes everything
# passed as keyword argument a metadata. The first two args also have an `_`
# prefix to reduce the chances of name collisions with the passed metadata, and
# since they're positional only, users will never type those underscores.
def process_routing(_obj, _method, /, **kwargs):
    """Validate and route input parameters.

    This function is used inside a router's method, e.g. :term:`fit`,
    to validate the metadata and handle the routing.

    Assuming this signature: ``fit(self, X, y, sample_weight=None, **fit_params)``,
    a call to this function would be:
    ``process_routing(self, sample_weight=sample_weight, **fit_params)``.

    Note that if routing is not enabled and ``kwargs`` is empty, then it
    returns an empty routing where ``process_routing(...).ANYTHING.ANY_METHOD``
    is always an empty dictionary.

    .. versionadded:: 1.3

    Parameters
    ----------
    _obj : object
        An object implementing ``get_metadata_routing``. Typically a
        meta-estimator.

    _method : str
        The name of the router's method in which this function is called.

    **kwargs : dict
        Metadata to be routed.

    Returns
    -------
    routed_params : Bunch
        A :class:`~sklearn.utils.Bunch` of the form ``{"object_name": {"method_name":
        {prop: value}}}`` which can be used to pass the required metadata to
        corresponding methods or corresponding child objects. The object names
        are those defined in `obj.get_metadata_routing()`.
    """
    if not _routing_enabled() and not kwargs:
        # If routing is not enabled and kwargs are empty, then we don't have to
        # try doing any routing, we can simply return a structure which returns
        # an empty dict on routed_params.ANYTHING.ANY_METHOD.
        class EmptyRequest:
            def get(self, name, default=None):
                return default if default else {}

            def __getitem__(self, name):
                return Bunch(**{method: dict() for method in METHODS})

            def __getattr__(self, name):
                return Bunch(**{method: dict() for method in METHODS})

        return EmptyRequest()

    if not (hasattr(_obj, "get_metadata_routing") or isinstance(_obj, MetadataRouter)):
        raise AttributeError(
            f"The given object ({repr(_obj.__class__.__name__)}) needs to either"
            " implement the routing method `get_metadata_routing` or be a"
            " `MetadataRouter` instance."
        )
    if _method not in METHODS:
        raise TypeError(
            f"Can only route and process input on these methods: {METHODS}, "
            f"while the passed method is: {_method}."
        )

    request_routing = get_routing_for_object(_obj)
    request_routing.validate_metadata(params=kwargs, method=_method)
    routed_params = request_routing.route_params(params=kwargs, caller=_method)

    return routed_params
