# SPDX-License-Identifier: Apache-2.0

import pprint
from collections import OrderedDict
import hashlib
import numpy as np
from onnx.numpy_helper import from_array
from onnxconverter_common.utils import sklearn_installed, skl2onnx_installed  # noqa
from onnxconverter_common.utils import is_numeric_type, is_string_type  # noqa
from onnxconverter_common.utils import cast_list, convert_to_python_value  # noqa
from onnxconverter_common.utils import convert_to_python_default_value  # noqa
from onnxconverter_common.utils import convert_to_list  # noqa
from onnxconverter_common.utils import check_input_and_output_numbers  # noqa
from onnxconverter_common.utils import check_input_and_output_types  # noqa
from .data_types import TensorType

_unique_index = {"subgraph": 0}


def get_unique_subgraph():
    "Returns a unique identifier integer for subgraph."
    global _unique_index
    _unique_index["subgraph"] += 1
    return _unique_index["subgraph"]


def get_producer():
    """
    Internal helper function to return the producer
    """
    from .. import __producer__

    return __producer__


def get_producer_version():
    """
    Internal helper function to return the producer version
    """
    from .. import __producer_version__

    return __producer_version__


def get_domain():
    """
    Internal helper function to return the model domain
    """
    from .. import __domain__

    return __domain__


def get_model_version():
    """
    Internal helper function to return the model version
    """
    from .. import __model_version__

    return __model_version__


def get_column_index(i, inputs):
    """
    Returns a tuples (variable index, column index in that variable).
    The function has two different behaviours, one when *i* (column index)
    is an integer, another one when *i* is a string (column name).
    If *i* is a string, the function looks for input name with
    this name and returns (index, 0).
    If *i* is an integer, let's assume first we have two inputs
    *I0 = FloatTensorType([None, 2])* and *I1 = FloatTensorType([None, 3])*,
    in this case, here are the results:

    ::

        get_column_index(0, inputs) -> (0, 0)
        get_column_index(1, inputs) -> (0, 1)
        get_column_index(2, inputs) -> (1, 0)
        get_column_index(3, inputs) -> (1, 1)
        get_column_index(4, inputs) -> (1, 2)
    """
    if isinstance(i, int):
        if i == 0:
            # Useful shortcut, skips the case when end is None
            # (unknown dimension)
            return 0, 0
        vi = 0
        pos = 0
        end = inputs[0].type.shape[1] if isinstance(inputs[0].type, TensorType) else 1
        if end is None:
            raise RuntimeError(
                "Cannot extract a specific column {0} when "
                "one input ('{1}') has unknown "
                "dimension.".format(i, inputs[0])
            )
        while True:
            if pos <= i < end:
                return (vi, i - pos)
            vi += 1
            pos = end
            if vi >= len(inputs):
                raise RuntimeError(
                    "Input {} (i={}, end={}) is not available in\n{}".format(
                        vi, i, end, pprint.pformat(inputs)
                    )
                )
            rel_end = (
                inputs[vi].type.shape[1]
                if isinstance(inputs[vi].type, TensorType)
                else 1
            )
            if rel_end is None:
                raise RuntimeError(
                    "Cannot extract a specific column {0} when "
                    "one input ('{1}') has unknown "
                    "dimension.".format(i, inputs[vi])
                )
            end += rel_end
    else:
        for ind, inp in enumerate(inputs):
            if inp.raw_name == i:
                return ind, 0
        raise RuntimeError(
            "Unable to find column name %r among names %r. "
            "Make sure the input names specified with parameter "
            "initial_types fits the column names specified in the "
            "pipeline to convert. This may happen because a "
            "ColumnTransformer follows a transformer without "
            "any mapped converter in a pipeline." % (i, [n.raw_name for n in inputs])
        )


def get_column_indices(indices, inputs, multiple):
    """
    Returns the requested graph inpudes based on their
    indices or names. See :func:`get_column_index`.

    :param indices: variables indices or names
    :param inputs: graph inputs
    :param multiple: allows column to come from multiple variables
    :return: a tuple *(variable name, list of requested indices)* if
        *multiple* is False, a dictionary *{ var_index: [ list of
        requested indices ] }*
        if *multiple* is True
    """
    if multiple:
        res = OrderedDict()
        for p in indices:
            ov, onnx_i = get_column_index(p, inputs)
            if ov not in res:
                res[ov] = []
            res[ov].append(onnx_i)
        return res
    else:
        onnx_var = None
        onnx_is = []
        for p in indices:
            ov, onnx_i = get_column_index(p, inputs)
            onnx_is.append(onnx_i)
            if onnx_var is None:
                onnx_var = ov
            elif onnx_var != ov:
                cols = [onnx_var, ov]
                raise NotImplementedError(
                    "sklearn-onnx is not able to merge multiple columns from "
                    "multiple variables ({0}). You should think about merging "
                    "initial types.".format(cols)
                )
        return onnx_var, onnx_is


def hash_array(value, length=15):
    "Computes a hash identifying the value."
    try:
        onx = from_array(value)
    except (AttributeError, TypeError) as e:
        # sparse matrix for example
        if hasattr(value, "tocoo"):
            coo = value.tocoo()
            arrs = [coo.data, coo.row, coo.col, np.array(coo.shape)]
            m = hashlib.sha256()
            for arr in arrs:
                m.update(from_array(arr).SerializeToString())
            return m.hexdigest()[:length]

        raise ValueError(
            "Unable to compute hash for type %r (value=%r)." % (type(value), value)
        ) from e
    except RuntimeError as ee:
        # cannot be serialized
        if isinstance(value, (np.ndarray, list)):
            b = str(value).encode("utf-8")
            m = hashlib.sha256()
            m.update(b)
            return m.hexdigest()[:length]
        raise RuntimeError(
            "Unable to convert value type %r, (value=%r)." % (type(value), value)
        ) from ee

    m = hashlib.sha256()
    m.update(onx.SerializeToString())
    return m.hexdigest()[:length]
