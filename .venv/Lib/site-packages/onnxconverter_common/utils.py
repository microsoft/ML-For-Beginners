# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################

import numbers
import numpy as np
import warnings
import packaging.version as pv


def sparkml_installed():
    """
    Checks that *spark* is available.
    """
    try:
        import pyspark  # noqa F401
        return True
    except ImportError:
        return False


def sklearn_installed():
    """
    Checks that *scikit-learn* is available.
    """
    try:
        import sklearn  # noqa F401
        return True
    except ImportError:
        return False


def skl2onnx_installed():
    """
    Checks that *skl2onnx* converter is available.
    """
    try:
        import skl2onnx  # noqa F401
        return True
    except ImportError:
        return False


def coreml_installed():
    """
    Checks that *coremltools* is available.
    """
    try:
        import coremltools  # noqa F401
        return True
    except ImportError:
        return False


def keras2onnx_installed():
    """
    Checks that *keras2onnx* is available.
    """
    try:
        import keras2onnx  # noqa F401
        return True
    except ImportError:
        return False


def torch_installed():
    """
    Checks that *pytorch* is available.
    """
    try:
        import torch  # noqa F401
        return True
    except ImportError:
        return False


def caffe2_installed():
    """
    Checks that *caffe* is available.
    """
    try:
        import caffe2  # noqa F401
        return True
    except ImportError:
        return False


def libsvm_installed():
    """
    Checks that *libsvm* is available.
    """
    try:
        import svm  # noqa F401
        import svmutil  # noqa F401
        return True
    except ImportError:
        return False


def lightgbm_installed():
    """
    Checks that *lightgbm* is available.
    """
    try:
        import lightgbm  # noqa F401
        return True
    except ImportError:
        return False


def xgboost_installed():
    """
    Checks that *xgboost* is available.
    """
    try:
        import xgboost  # noqa F401
    except ImportError:
        return False
    from xgboost.core import _LIB
    try:
        _LIB.XGBoosterDumpModelEx
    except AttributeError:
        # The version is not recent enough even though it is version 0.6.
        # You need to install xgboost from github and not from pypi.
        return False
    from xgboost import __version__
    vers = pv.Version(__version__)
    allowed = pv.Version('0.7')
    if vers < allowed:
        warnings.warn('The converter works for xgboost >= 0.7. Earlier versions might not.')
    return True


def h2o_installed():
    """
    Checks that *h2o* is available.
    """
    try:
        import h2o  # noqa F401
    except ImportError:
        return False
    return True


def hummingbird_installed():
    """
    Checks that *Hummingbird* is available.
    """
    try:
        import hummingbird.ml  # noqa: F401

        return True
    except ImportError:
        return False


def get_producer():
    """
    Internal helper function to return the producer
    """
    from . import __producer__
    return __producer__


def get_producer_version():
    """
    Internal helper function to return the producer version
    """
    from . import __producer_version__
    return __producer_version__


def get_domain():
    """
    Internal helper function to return the model domain
    """
    from . import __domain__
    return __domain__


def get_model_version():
    """
    Internal helper function to return the model version
    """
    from . import __model_version__
    return __model_version__


def is_numeric_type(item):
    numeric_types = (int, float, complex)
    types = numeric_types

    if isinstance(item, list):
        return all(isinstance(i, types) for i in item)
    if isinstance(item, np.ndarray):
        return np.issubdtype(item.dtype, np.number)
    return isinstance(item, types)


def is_string_type(item):
    if isinstance(item, list):
        return all(isinstance(i, str) for i in item)
    if isinstance(item, np.ndarray):
        return np.issubdtype(item.dtype, np.str_)
    return isinstance(item, str)


def cast_list(type, items):
    return [type(item) for item in items]


def convert_to_python_value(var):
    if isinstance(var, numbers.Integral):
        return int(var)
    elif isinstance(var, numbers.Real):
        return float(var)
    elif isinstance(var, str):
        return str(var)
    else:
        raise TypeError('Unable to convert {0} to python type'.format(type(var)))


def convert_to_python_default_value(var):
    if isinstance(var, numbers.Integral):
        return int()
    elif isinstance(var, numbers.Real):
        return float()
    elif isinstance(var, str):
        return str()
    else:
        raise TypeError('Unable to find default python value for type {0}'.format(type(var)))


def convert_to_list(var):
    if isinstance(var, numbers.Real) or isinstance(var, str):
        return [convert_to_python_value(var)]
    elif isinstance(var, np.ndarray) and len(var.shape) == 1:
        return [convert_to_python_value(v) for v in var]
    elif isinstance(var, list):
        flattened = []
        if all(isinstance(ele, np.ndarray) and len(ele.shape) == 1 for ele in var):
            max_classes = max([ele.shape[0] for ele in var])
            flattened_one = []
            for ele in var:
                for i in range(max_classes):
                    if i < ele.shape[0]:
                        flattened_one.append(convert_to_python_value(ele[i]))
                    else:
                        flattened_one.append(convert_to_python_default_value(ele[0]))
            flattened += flattened_one
            return flattened
        elif all(isinstance(v, numbers.Real) or isinstance(v, str) for v in var):
            return [convert_to_python_value(v) for v in var]
        else:
            raise TypeError('Unable to flatten variable')
    else:
        raise TypeError('Unable to flatten variable')


def check_input_and_output_numbers(operator, input_count_range=None, output_count_range=None):
    '''
    Check if the number of input(s)/output(s) is correct

    :param operator: A Operator object
    :param input_count_range: A list of two integers or an integer. If it's a list the first/second element is the
    minimal/maximal number of inputs. If it's an integer, it is equivalent to specify that number twice in a list. For
    infinite ranges like 5 to infinity, you need to use [5, None].
    :param output_count_range: A list of two integers or an integer. See input_count_range for its format.
    '''
    if isinstance(input_count_range, list):
        min_input_count = input_count_range[0]
        max_input_count = input_count_range[1]
    elif isinstance(input_count_range, int) or input_count_range is None:
        min_input_count = input_count_range
        max_input_count = input_count_range
    else:
        raise RuntimeError('input_count_range must be a list or an integer')

    if isinstance(output_count_range, list):
        min_output_count = output_count_range[0]
        max_output_count = output_count_range[1]
    elif isinstance(output_count_range, int) or output_count_range is None:
        min_output_count = output_count_range
        max_output_count = output_count_range
    else:
        raise RuntimeError('output_count_range must be a list or an integer')

    if min_input_count is not None and len(operator.inputs) < min_input_count:
        raise RuntimeError(
            'For operator %s (type: %s), at least %s input(s) is(are) required but we got %s input(s) which are %s'
            % (operator.full_name, operator.type, min_input_count, len(operator.inputs), operator.input_full_names))

    if max_input_count is not None and len(operator.inputs) > max_input_count:
        raise RuntimeError(
            'For operator %s (type: %s), at most %s input(s) is(are) supported but we got %s input(s) which are %s'
            % (operator.full_name, operator.type, max_input_count, len(operator.inputs), operator.input_full_names))

    if min_output_count is not None and len(operator.outputs) < min_output_count:
        raise RuntimeError(
            'For operator %s (type: %s), at least %s output(s) is(are) produced but we got %s output(s) which are %s'
            % (operator.full_name, operator.type, min_output_count, len(operator.outputs), operator.output_full_names))

    if max_output_count is not None and len(operator.outputs) > max_output_count:
        raise RuntimeError(
            'For operator %s (type: %s), at most %s outputs(s) is(are) supported but we got %s output(s) which are %s'
            % (operator.full_name, operator.type, max_output_count, len(operator.outputs), operator.output_full_names))


def check_input_and_output_types(operator, good_input_types=None, good_output_types=None):
    '''
    Check if the type(s) of input(s)/output(s) is(are) correct

    :param operator: A Operator object
    :param good_input_types: A list of allowed input types (e.g., [FloatTensorType, Int64TensorType]) or None. None
    means that we skip the check of the input types.
    :param good_output_types: A list of allowed output types. See good_input_types for its format.
    '''
    if good_input_types is not None:
        for variable in operator.inputs:
            if type(variable.type) not in good_input_types:
                raise RuntimeError('Operator %s (type: %s) got an input %s with a wrong type %s. Only %s are allowed'
                                   % (operator.full_name, operator.type, variable.full_name, type(variable.type),
                                      good_input_types))

    if good_output_types is not None:
        for variable in operator.outputs:
            if type(variable.type) not in good_output_types:
                raise RuntimeError('Operator %s (type: %s) got an output %s with a wrong type %s. Only %s are allowed'
                                   % (operator.full_name, operator.type, variable.full_name, type(variable.type),
                                      good_output_types))
