# SPDX-License-Identifier: Apache-2.0

from .utils_checking import check_signature

# This dictionary defines the converters which can be invoked in the
# conversion framework defined in _topology.py. A key in this dictionary
# is an operator's unique ID (e.g., string and type) while the
# associated value is the callable object used to convert the
# operator specified by the key.
_converter_pool = {}


class RegisteredConverter:
    def __init__(self, fct, options):
        self._fct = fct
        self._options = options

    def __call__(self, *args):
        if (
            len(args) == 3
            and hasattr(args[2], "_get_allowed_options")
            and hasattr(args[1], "raw_operator")
        ):
            # Checks that the user did not specify a wrong option.
            if args[1].raw_operator is not None:
                args[2]._get_allowed_options(args[1].raw_operator)
        return self._fct(*args)

    def get_allowed_options(self):
        return self._options


# This dictionary defines the shape calculators which can be invoked in
# the conversion framework defined in _topology.py. A key in this
# dictionary is an operator's unique ID (e.g., string and type) while
# the associated value is the callable object used to infer the output
# shape(s) for the operator specified by the key.
_shape_calculator_pool = {}


def register_converter(
    operator_name, conversion_function, overwrite=False, options=None
):
    """
    :param operator_name: A unique operator ID. It is usually a string
                          but you can use a type as well
    :param conversion_function: A callable object
    :param overwrite: By default, we raise an exception if the caller of
                      this function is trying to assign an existing
                      key (i.e., operator_name) a new value
                      (i.e., conversion_function). Set this flag to True
                      to enable overwriting.
    :param options: supported options for this converter
        (dictionary {name: supported values or None})
    """
    if conversion_function is None:
        raise ValueError("A converter cannot be None for %r." % operator_name)
    if not overwrite and operator_name in _converter_pool:
        raise ValueError("We do not overwrite registered converter " "by default")
    if len(_converter_pool) > 0:
        key = next(iter(_converter_pool))
        check_signature(
            conversion_function, _converter_pool[key]._fct, skip=("operator",)
        )
    _converter_pool[operator_name] = RegisteredConverter(conversion_function, options)


def get_converter(operator_name):
    if operator_name not in _converter_pool:
        msg = "Unsupported conversion for operator %s (%d registered)" % (
            operator_name,
            len(_converter_pool),
        )
        raise ValueError(msg)
    return _converter_pool[operator_name]


def register_shape_calculator(operator_name, calculator_function, overwrite=False):
    """
    :param operator_name: A unique operator ID. It is usually a string
                          but you can use a type as well
    :param calculator_function: A callable object
    :param overwrite: By default, we raise an exception if the caller
                      of this function is trying to assign an existing
                      key (i.e., operator_name) a new value
                      (i.e., calculator_function). Set this flag to True
                      to enable overwriting.
    """
    if calculator_function is None:
        raise ValueError("A shape calculator cannot be None for %r." % operator_name)
    if not overwrite and operator_name in _shape_calculator_pool:
        raise ValueError(
            "We do not overwrite registrated shape calculator " "by default"
        )
    if calculator_function is not None and len(_shape_calculator_pool) > 0:
        key = next(iter(_shape_calculator_pool))
        check_signature(
            calculator_function, _shape_calculator_pool[key], skip=("operator",)
        )
    _shape_calculator_pool[operator_name] = calculator_function


def get_shape_calculator(operator_name):
    if operator_name not in _shape_calculator_pool:
        msg = "Unsupported shape calculator for operator " "'%s'." % operator_name
        raise ValueError(msg)
    return _shape_calculator_pool[operator_name]
