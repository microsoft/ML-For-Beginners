# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Union

import onnx
import onnx.onnx_cpp2py_export.printer as C  # noqa: N812


def to_text(proto: Union[onnx.ModelProto, onnx.FunctionProto, onnx.GraphProto]) -> str:
    if isinstance(proto, onnx.ModelProto):
        return C.model_to_text(proto.SerializeToString())
    if isinstance(proto, onnx.FunctionProto):
        return C.function_to_text(proto.SerializeToString())
    if isinstance(proto, onnx.GraphProto):
        return C.graph_to_text(proto.SerializeToString())
    raise TypeError("Unsupported argument type.")
