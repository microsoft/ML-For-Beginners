# SPDX-License-Identifier: Apache-2.0

import numpy as np

try:
    from pandas import DataFrame
except ImportError:
    DataFrame = None
from scipy.sparse import coo_matrix
from ..proto import TensorProto, ValueInfoProto
from ..common._topology import Variable
from ..common.data_types import (
    _guess_numpy_type,
    _guess_type_proto,
    BooleanTensorType,
    DataType,
    DoubleType,
    DoubleTensorType,
    FloatType,
    FloatTensorType,
    Int64Type,
    Int64TensorType,
    Int32TensorType,
    StringTensorType,
)
from ..common.data_types import Int8TensorType, UInt8TensorType, UInt8Type, Int8Type


def _guess_type(given_type):
    """
    Returns the proper type of an input.
    """

    def _guess_dim(value):
        if value == 0:
            return None
        return value

    if isinstance(given_type, (np.ndarray, coo_matrix)):
        shape = list(given_type.shape)
        if len(shape) == 0:
            # a number
            return _guess_numpy_type(given_type.dtype, tuple())
        shape[0] = None
        return _guess_numpy_type(given_type.dtype, shape)
    if isinstance(
        given_type,
        (
            FloatTensorType,
            Int64TensorType,
            Int32TensorType,
            StringTensorType,
            BooleanTensorType,
            DoubleTensorType,
            Int8TensorType,
            UInt8TensorType,
        ),
    ):
        return given_type
    if isinstance(given_type, Variable):
        return given_type.type
    if isinstance(given_type, DataType):
        return given_type
    if isinstance(given_type, TensorProto):
        return _guess_type_proto(given_type.data_type, given_type.dims)
    if isinstance(given_type, ValueInfoProto):
        ttype = given_type.type.tensor_type
        dims = [
            _guess_dim(ttype.shape.dim[i].dim_value)
            for i in range(len(ttype.shape.dim))
        ]
        return _guess_type_proto(ttype.elem_type, dims)
    if isinstance(given_type, np.int64):
        return Int64Type()
    if isinstance(given_type, np.float32):
        return FloatType()
    if isinstance(given_type, np.float64):
        return DoubleType()
    if isinstance(given_type, np.int8):
        return Int8Type()
    if isinstance(given_type, np.uint8):
        return UInt8Type()
    if given_type.__class__.__name__.endswith("Categorical"):
        # pandas Categorical without important pandas
        return Int64TensorType()
    raise NotImplementedError(
        "Unsupported type '{}'. You may raise an issue "
        "at https://github.com/onnx/sklearn-onnx/issues."
        "".format(type(given_type))
    )


def guess_initial_types(X, initial_types):
    if X is None and initial_types is None:
        raise NotImplementedError("Initial types must be specified.")
    if initial_types is None:
        if isinstance(X, np.ndarray):
            X = X[:1]
            gt = _guess_type(X)
            initial_types = [("X", gt)]
        elif DataFrame is not None and isinstance(X, DataFrame):
            X = X[:1]
            initial_types = []
            for c in X.columns:
                if isinstance(X[c].values[0], (str, np.str_)):
                    g = StringTensorType()
                else:
                    g = _guess_type(X[c].values)
                g.shape = [None, 1]
                initial_types.append((c, g))
        elif isinstance(X, list):
            initial_types = X
        else:
            raise TypeError("Unexpected type %r, unable to guess type." % type(X))
    return initial_types
