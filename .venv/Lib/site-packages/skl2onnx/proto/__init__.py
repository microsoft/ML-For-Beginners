# SPDX-License-Identifier: Apache-2.0


# Rather than using ONNX protobuf definition throughout our codebase,
# we import ONNX protobuf definition here so that we can conduct quick
# fixes by overwriting ONNX functions without changing any lines
# elsewhere.
from onnx import onnx_pb as onnx_proto  # noqa
from onnx import defs  # noqa

# Overwrite the make_tensor defined in onnx.helper because of a bug
# (string tensor get assigned twice)
from onnx import mapping
from onnx.onnx_pb import TensorProto, ValueInfoProto  # noqa

try:
    from onnx.onnx_pb import SparseTensorProto  # noqa
except ImportError:
    # onnx is too old.
    pass
from onnx.helper import split_complex_to_pairs


def make_tensor_fixed(name, data_type, dims, vals, raw=False):
    """
    Make a TensorProto with specified arguments.  If raw is False, this
    function will choose the corresponding proto field to store the
    values based on data_type. If raw is True, use "raw_data" proto
    field to store the values, and values should be of type bytes in
    this case.
    """
    tensor = TensorProto()
    tensor.data_type = data_type
    tensor.name = name

    if data_type == TensorProto.COMPLEX64 or data_type == TensorProto.COMPLEX128:
        vals = split_complex_to_pairs(vals)
    if raw:
        tensor.raw_data = vals
    else:
        field = mapping.STORAGE_TENSOR_TYPE_TO_FIELD[
            mapping.TENSOR_TYPE_TO_STORAGE_TENSOR_TYPE[data_type]
        ]
        getattr(tensor, field).extend(vals)

    tensor.dims.extend(dims)
    return tensor


def get_opset_number_from_onnx():
    """
    Returns the latest opset version supported
    by the *onnx* package.
    """
    return defs.onnx_opset_version()


def get_latest_tested_opset_version():
    """
    This module relies on *onnxruntime* to test every
    converter. The function returns the most recent
    target opset tested with *onnxruntime* or the opset
    version specified by *onnx* package if this one is lower
    (return by `onnx.defs.onnx_opset_version()`).
    """
    from .. import __max_supported_opset__

    return min(__max_supported_opset__, get_opset_number_from_onnx())
