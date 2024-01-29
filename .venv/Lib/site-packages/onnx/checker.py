# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Graph utilities for checking whether an ONNX proto message is legal."""

from __future__ import annotations

__all__ = [
    "check_attribute",
    "check_function",
    "check_graph",
    "check_model",
    "check_node",
    "check_sparse_tensor",
    "check_tensor",
    "check_value_info",
    "DEFAULT_CONTEXT",
    "ValidationError",
    "C",
    "MAXIMUM_PROTOBUF",
]

import os
import sys
from typing import Any, Callable, TypeVar

from google.protobuf.message import Message

import onnx.defs
import onnx.onnx_cpp2py_export.checker as C  # noqa: N812
import onnx.shape_inference
from onnx import (
    IR_VERSION,
    AttributeProto,
    FunctionProto,
    GraphProto,
    ModelProto,
    NodeProto,
    SparseTensorProto,
    TensorProto,
    ValueInfoProto,
    helper,
)

# Limitation of single protobuf file is 2GB
MAXIMUM_PROTOBUF = 2000000000

# TODO: This thing where we reserialize the protobuf back into the
# string, only to deserialize it at the call site, is really goofy.
# Stop doing that.


# NB: Please don't edit this context!
DEFAULT_CONTEXT = C.CheckerContext()
DEFAULT_CONTEXT.ir_version = IR_VERSION
# TODO: Maybe ONNX-ML should also be defaulted?
DEFAULT_CONTEXT.opset_imports = {"": onnx.defs.onnx_opset_version()}


FuncType = TypeVar("FuncType", bound=Callable[..., Any])


def _ensure_proto_type(proto: Message, proto_type: type[Message]) -> None:
    if not isinstance(proto, proto_type):
        raise TypeError(
            f"The proto message needs to be of type '{proto_type.__name__}'"
        )


def check_value_info(
    value_info: ValueInfoProto, ctx: C.CheckerContext = DEFAULT_CONTEXT
) -> None:
    _ensure_proto_type(value_info, ValueInfoProto)
    return C.check_value_info(value_info.SerializeToString(), ctx)


def check_tensor(tensor: TensorProto, ctx: C.CheckerContext = DEFAULT_CONTEXT) -> None:
    _ensure_proto_type(tensor, TensorProto)
    return C.check_tensor(tensor.SerializeToString(), ctx)


def check_attribute(
    attr: AttributeProto, ctx: C.CheckerContext = DEFAULT_CONTEXT
) -> None:
    _ensure_proto_type(attr, AttributeProto)
    return C.check_attribute(attr.SerializeToString(), ctx)


def check_node(node: NodeProto, ctx: C.CheckerContext = DEFAULT_CONTEXT) -> None:
    _ensure_proto_type(node, NodeProto)
    return C.check_node(node.SerializeToString(), ctx)


def check_function(
    function: FunctionProto, ctx: C.CheckerContext | None = None
) -> None:
    _ensure_proto_type(function, FunctionProto)
    if ctx is None:
        ctx = C.CheckerContext()
        ctx.ir_version = helper.find_min_ir_version_for(
            list(function.opset_import), True
        )
        function_opset_dic = {}
        for domain_version in function.opset_import:
            function_opset_dic[domain_version.domain] = domain_version.version
        ctx.opset_imports = function_opset_dic
    C.check_function(function.SerializeToString(), ctx)


def check_graph(graph: GraphProto, ctx: C.CheckerContext = DEFAULT_CONTEXT) -> None:
    _ensure_proto_type(graph, GraphProto)
    return C.check_graph(graph.SerializeToString(), ctx)


def check_sparse_tensor(
    sparse: SparseTensorProto, ctx: C.CheckerContext = DEFAULT_CONTEXT
) -> None:
    _ensure_proto_type(sparse, SparseTensorProto)
    C.check_sparse_tensor(sparse.SerializeToString(), ctx)


def check_model(
    model: ModelProto | str | bytes | os.PathLike,
    full_check: bool = False,
    skip_opset_compatibility_check: bool = False,
) -> None:
    """Check the consistency of a model. An exception is raised if the test fails.

    Args:
        model: Model to check.
        full_check: If True, the function also checks for shapes that can be inferred.
        skip_opset_compatibility_check: If True, the function skips the check for
            opset compatibility.
    """
    # If model is a path instead of ModelProto
    if isinstance(model, (str, os.PathLike)):
        C.check_model_path(os.fspath(model), full_check, skip_opset_compatibility_check)
    else:
        protobuf_string = (
            model if isinstance(model, bytes) else model.SerializeToString()
        )
        # If the protobuf is larger than 2GB,
        # remind users should use the model path to check
        if sys.getsizeof(protobuf_string) > MAXIMUM_PROTOBUF:
            raise ValueError(
                "This protobuf of onnx model is too large (>2GB). Call check_model with model path instead."
            )
        C.check_model(protobuf_string, full_check, skip_opset_compatibility_check)


ValidationError = C.ValidationError
