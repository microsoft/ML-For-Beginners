"""Useful decorators for Traitlets users."""

import copy
from inspect import Parameter, Signature, signature
from typing import Type, TypeVar

from ..traitlets import HasTraits, Undefined


def _get_default(value):
    """Get default argument value, given the trait default value."""
    return Parameter.empty if value == Undefined else value


T = TypeVar("T", bound=HasTraits)


def signature_has_traits(cls: Type[T]) -> Type[T]:
    """Return a decorated class with a constructor signature that contain Trait names as kwargs."""
    traits = [
        (name, _get_default(value.default_value))
        for name, value in cls.class_traits().items()
        if not name.startswith("_")
    ]

    # Taking the __init__ signature, as the cls signature is not initialized yet
    old_signature = signature(cls.__init__)
    old_parameter_names = list(old_signature.parameters)

    old_positional_parameters = []
    old_var_positional_parameter = None  # This won't be None if the old signature contains *args
    old_keyword_only_parameters = []
    old_var_keyword_parameter = None  # This won't be None if the old signature contains **kwargs

    for parameter_name in old_signature.parameters:
        # Copy the parameter
        parameter = copy.copy(old_signature.parameters[parameter_name])

        if (
            parameter.kind is Parameter.POSITIONAL_ONLY
            or parameter.kind is Parameter.POSITIONAL_OR_KEYWORD
        ):
            old_positional_parameters.append(parameter)

        elif parameter.kind is Parameter.VAR_POSITIONAL:
            old_var_positional_parameter = parameter

        elif parameter.kind is Parameter.KEYWORD_ONLY:
            old_keyword_only_parameters.append(parameter)

        elif parameter.kind is Parameter.VAR_KEYWORD:
            old_var_keyword_parameter = parameter

    # Unfortunately, if the old signature does not contain **kwargs, we can't do anything,
    # because it can't accept traits as keyword arguments
    if old_var_keyword_parameter is None:
        raise RuntimeError(
            "The {} constructor does not take **kwargs, which means that the signature can not be expanded with trait names".format(
                cls
            )
        )

    new_parameters = []

    # Append the old positional parameters (except `self` which is the first parameter)
    new_parameters += old_positional_parameters[1:]

    # Append *args if the old signature had it
    if old_var_positional_parameter is not None:
        new_parameters.append(old_var_positional_parameter)

    # Append the old keyword only parameters
    new_parameters += old_keyword_only_parameters

    # Append trait names as keyword only parameters in the signature
    new_parameters += [
        Parameter(name, kind=Parameter.KEYWORD_ONLY, default=default)
        for name, default in traits
        if name not in old_parameter_names
    ]

    # Append **kwargs
    new_parameters.append(old_var_keyword_parameter)

    cls.__signature__ = Signature(new_parameters)  # type:ignore[attr-defined]

    return cls
