# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################

# This dictionary defines the converters which can be invoked in the conversion framework defined in topology.py. A key
# in this dictionary is an operator's unique ID (e.g., string and type) while the associated value is the callable
# object used to convert the operator specified by the key.
_converter_pool = {}

# This dictionary defines the shape calculators which can be invoked in the conversion framework defined in
# topology.py. A key in this dictionary is an operator's unique ID (e.g., string and type) while the associated value
# is the callable object used to infer the output shape(s) for the operator specified by the key.
_shape_calculator_pool = {}


def register_converter(operator_name, conversion_function, overwrite=False):
    '''
    :param operator_name: A unique operator ID. It is usually a string but you can use a type as well
    :param conversion_function: A callable object
    :param overwrite: By default, we raise an exception if the caller of this function is trying to assign an existing
    key (i.e., operator_name) a new value (i.e., conversion_function). Set this flag to True to enable overwriting.
    '''
    if not overwrite and operator_name in _converter_pool:
        raise ValueError('We do not overwrite registrated converter by default')
    _converter_pool[operator_name] = conversion_function


def get_converter(operator_name):
    '''
    Given an Operator object (named operator) defined in topology.py, we can retrieve its conversion function.
    >>> from onnxmltools.convert.common._topology import Operator
    >>> operator = Operator('dummy_name', 'dummy_scope', 'dummy_operator_type', None)
    >>> get_converter(operator.type)  # Use 'dummy_operator_type' for dictionary looking-up

    :param operator_name: An operator ID
    :return: a conversion function for a specific Operator object
    '''
    if operator_name not in _converter_pool:
        raise ValueError('Unsupported conversion for operator %s' % operator_name)
    return _converter_pool[operator_name]


def register_shape_calculator(operator_name, calculator_function, overwrite=False):
    '''
    :param operator_name: A unique operator ID. It is usually a string but you can use a type as well
    :param calculator_function: A callable object
    :param overwrite:  By default, we raise an exception if the caller of this function is trying to assign an existing
    key (i.e., operator_name) a new value (i.e., calculator_function). Set this flag to True to enable overwriting.
    '''
    if not overwrite and operator_name in _shape_calculator_pool:
        raise ValueError('We do not overwrite registrated shape calculator by default')
    _shape_calculator_pool[operator_name] = calculator_function


def get_shape_calculator(operator_name):
    '''
    Given an Operator object (named operator) defined in topology.py, we can retrieve its shape calculation function.
    >>> from onnxmltools.convert.common._topology import Operator
    >>> operator = Operator('dummy_name', 'dummy_scope', 'dummy_operator_type', None)
    >>> get_shape_calculator(operator.type)  # Use 'dummy_operator_type' for dictionary looking-up

    :param operator_name: An operator ID
    :return: a shape calculation function for a specific Operator object
    '''
    if operator_name not in _shape_calculator_pool:
        raise ValueError('Unsupported shape calculation for operator %s' % operator_name)
    return _shape_calculator_pool[operator_name]
