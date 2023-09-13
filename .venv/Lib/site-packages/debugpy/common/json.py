# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root
# for license information.

"""Improved JSON serialization.
"""

import builtins
import json
import numbers
import operator


JsonDecoder = json.JSONDecoder


class JsonEncoder(json.JSONEncoder):
    """Customizable JSON encoder.

    If the object implements __getstate__, then that method is invoked, and its
    result is serialized instead of the object itself.
    """

    def default(self, value):
        try:
            get_state = value.__getstate__
        except AttributeError:
            pass
        else:
            return get_state()
        return super().default(value)


class JsonObject(object):
    """A wrapped Python object that formats itself as JSON when asked for a string
    representation via str() or format().
    """

    json_encoder_factory = JsonEncoder
    """Used by __format__ when format_spec is not empty."""

    json_encoder = json_encoder_factory(indent=4)
    """The default encoder used by __format__ when format_spec is empty."""

    def __init__(self, value):
        assert not isinstance(value, JsonObject)
        self.value = value

    def __getstate__(self):
        raise NotImplementedError

    def __repr__(self):
        return builtins.repr(self.value)

    def __str__(self):
        return format(self)

    def __format__(self, format_spec):
        """If format_spec is empty, uses self.json_encoder to serialize self.value
        as a string. Otherwise, format_spec is treated as an argument list to be
        passed to self.json_encoder_factory - which defaults to JSONEncoder - and
        then the resulting formatter is used to serialize self.value as a string.

        Example::

            format("{0} {0:indent=4,sort_keys=True}", json.repr(x))
        """
        if format_spec:
            # At this point, format_spec is a string that looks something like
            # "indent=4,sort_keys=True". What we want is to build a function call
            # from that which looks like:
            #
            #   json_encoder_factory(indent=4,sort_keys=True)
            #
            # which we can then eval() to create our encoder instance.
            make_encoder = "json_encoder_factory(" + format_spec + ")"
            encoder = eval(
                make_encoder, {"json_encoder_factory": self.json_encoder_factory}
            )
        else:
            encoder = self.json_encoder
        return encoder.encode(self.value)


# JSON property validators, for use with MessageDict.
#
# A validator is invoked with the actual value of the JSON property passed to it as
# the sole argument; or if the property is missing in JSON, then () is passed. Note
# that None represents an actual null in JSON, while () is a missing value.
#
# The validator must either raise TypeError or ValueError describing why the property
# value is invalid, or else return the value of the property, possibly after performing
# some substitutions - e.g. replacing () with some default value.


def _converter(value, classinfo):
    """Convert value (str) to number, otherwise return None if is not possible"""
    for one_info in classinfo:
        if issubclass(one_info, numbers.Number):
            try:
                return one_info(value)
            except ValueError:
                pass


def of_type(*classinfo, **kwargs):
    """Returns a validator for a JSON property that requires it to have a value of
    the specified type. If optional=True, () is also allowed.

    The meaning of classinfo is the same as for isinstance().
    """

    assert len(classinfo)
    optional = kwargs.pop("optional", False)
    assert not len(kwargs)

    def validate(value):
        if (optional and value == ()) or isinstance(value, classinfo):
            return value
        else:
            converted_value = _converter(value, classinfo)
            if converted_value:
                return converted_value

            if not optional and value == ():
                raise ValueError("must be specified")
            raise TypeError("must be " + " or ".join(t.__name__ for t in classinfo))

    return validate


def default(default):
    """Returns a validator for a JSON property with a default value.

    The validator will only allow property values that have the same type as the
    specified default value.
    """

    def validate(value):
        if value == ():
            return default
        elif isinstance(value, type(default)):
            return value
        else:
            raise TypeError("must be {0}".format(type(default).__name__))

    return validate


def enum(*values, **kwargs):
    """Returns a validator for a JSON enum.

    The validator will only allow the property to have one of the specified values.

    If optional=True, and the property is missing, the first value specified is used
    as the default.
    """

    assert len(values)
    optional = kwargs.pop("optional", False)
    assert not len(kwargs)

    def validate(value):
        if optional and value == ():
            return values[0]
        elif value in values:
            return value
        else:
            raise ValueError("must be one of: {0!r}".format(list(values)))

    return validate


def array(validate_item=False, vectorize=False, size=None):
    """Returns a validator for a JSON array.

    If the property is missing, it is treated as if it were []. Otherwise, it must
    be a list.

    If validate_item=False, it's treated as if it were (lambda x: x) - i.e. any item
    is considered valid, and is unchanged. If validate_item is a type or a tuple,
    it's treated as if it were json.of_type(validate).

    Every item in the list is replaced with validate_item(item) in-place, propagating
    any exceptions raised by the latter. If validate_item is a type or a tuple, it is
    treated as if it were json.of_type(validate_item).

    If vectorize=True, and the value is neither a list nor a dict, it is treated as
    if it were a single-element list containing that single value - e.g. "foo" is
    then the same as ["foo"]; but {} is an error, and not [{}].

    If size is not None, it can be an int, a tuple of one int, a tuple of two ints,
    or a set. If it's an int, the array must have exactly that many elements. If it's
    a tuple of one int, it's the minimum length. If it's a tuple of two ints, they
    are the minimum and the maximum lengths. If it's a set, it's the set of sizes that
    are valid - e.g. for {2, 4}, the array can be either 2 or 4 elements long.
    """

    if not validate_item:
        validate_item = lambda x: x
    elif isinstance(validate_item, type) or isinstance(validate_item, tuple):
        validate_item = of_type(validate_item)

    if size is None:
        validate_size = lambda _: True
    elif isinstance(size, set):
        size = {operator.index(n) for n in size}
        validate_size = lambda value: (
            len(value) in size
            or "must have {0} elements".format(
                " or ".join(str(n) for n in sorted(size))
            )
        )
    elif isinstance(size, tuple):
        assert 1 <= len(size) <= 2
        size = tuple(operator.index(n) for n in size)
        min_len, max_len = (size + (None,))[0:2]
        validate_size = lambda value: (
            "must have at least {0} elements".format(min_len)
            if len(value) < min_len
            else "must have at most {0} elements".format(max_len)
            if max_len is not None and len(value) < max_len
            else True
        )
    else:
        size = operator.index(size)
        validate_size = lambda value: (
            len(value) == size or "must have {0} elements".format(size)
        )

    def validate(value):
        if value == ():
            value = []
        elif vectorize and not isinstance(value, (list, dict)):
            value = [value]

        of_type(list)(value)

        size_err = validate_size(value)  # True if valid, str if error
        if size_err is not True:
            raise ValueError(size_err)

        for i, item in enumerate(value):
            try:
                value[i] = validate_item(item)
            except (TypeError, ValueError) as exc:
                raise type(exc)(f"[{repr(i)}] {exc}")
        return value

    return validate


def object(validate_value=False):
    """Returns a validator for a JSON object.

    If the property is missing, it is treated as if it were {}. Otherwise, it must
    be a dict.

    If validate_value=False, it's treated as if it were (lambda x: x) - i.e. any
    value is considered valid, and is unchanged. If validate_value is a type or a
    tuple, it's treated as if it were json.of_type(validate_value).

    Every value in the dict is replaced with validate_value(value) in-place, propagating
    any exceptions raised by the latter. If validate_value is a type or a tuple, it is
    treated as if it were json.of_type(validate_value). Keys are not affected.
    """

    if isinstance(validate_value, type) or isinstance(validate_value, tuple):
        validate_value = of_type(validate_value)

    def validate(value):
        if value == ():
            return {}

        of_type(dict)(value)
        if validate_value:
            for k, v in value.items():
                try:
                    value[k] = validate_value(v)
                except (TypeError, ValueError) as exc:
                    raise type(exc)(f"[{repr(k)}] {exc}")
        return value

    return validate


def repr(value):
    return JsonObject(value)


dumps = json.dumps
loads = json.loads
