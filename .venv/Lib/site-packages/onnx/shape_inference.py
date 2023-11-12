# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""onnx shape inference. Shape inference is not guaranteed to be
complete.

"""

from typing import Dict, List, Optional, Sequence, Union

import onnx
import onnx.onnx_cpp2py_export.shape_inference as C  # noqa: N812
from onnx import AttributeProto, FunctionProto, ModelProto, TypeProto


def infer_shapes(
    model: Union[ModelProto, bytes],
    check_type: bool = False,
    strict_mode: bool = False,
    data_prop: bool = False,
) -> ModelProto:
    """Apply shape inference to the provided ModelProto.

    Inferred shapes are added to the value_info field of the graph.

    If the inferred values conflict with values already provided in the
    graph, that means that the provided values are invalid (or there is a
    bug in shape inference), and the result is unspecified.

    Arguments:
        model (Union[ModelProto, bytes], bool, bool, bool) -> ModelProto
        check_type (bool): Checks the type-equality for input and output
        strict_mode (bool): Stricter shape inference, it will throw errors if any;
            Otherwise, simply stop if any error
        data_prop (bool): Enables data propagation for limited operators to perform shape computation

    Returns:
        (ModelProto) model with inferred shape information
    """
    if isinstance(model, (ModelProto, bytes)):
        model_str = model if isinstance(model, bytes) else model.SerializeToString()
        inferred_model_str = C.infer_shapes(
            model_str, check_type, strict_mode, data_prop
        )
        return onnx.load_from_string(inferred_model_str)
    if isinstance(model, str):
        raise TypeError(
            "infer_shapes only accepts ModelProto or bytes,"
            "you can use infer_shapes_path for the model path (String)."
        )

    raise TypeError(
        f"infer_shapes only accepts ModelProto or bytes, incorrect type: {type(model)}"
    )


def infer_shapes_path(
    model_path: str,
    output_path: str = "",
    check_type: bool = False,
    strict_mode: bool = False,
    data_prop: bool = False,
) -> None:
    """
    Take model path for shape_inference same as infer_shape; it support >2GB models
    Directly output the inferred model to the output_path; Default is the original model path
    """
    if isinstance(model_path, ModelProto):
        raise TypeError(
            "infer_shapes_path only accepts model Path (String),"
            "you can use infer_shapes for the ModelProto."
        )
    # Directly output the inferred model into the specified path, return nothing
    if isinstance(model_path, str):
        # If output_path is not defined, default output_path would be the original model path
        if output_path == "":
            output_path = model_path
        C.infer_shapes_path(model_path, output_path, check_type, strict_mode, data_prop)
    else:
        raise TypeError(
            "infer_shapes_path only accepts model path (String), "
            f"incorrect type: {type(model_path)}"
        )


def infer_node_outputs(
    schema: onnx.defs.OpSchema,
    node: onnx.NodeProto,
    input_types: Dict[str, onnx.TypeProto],
    input_data: Optional[Dict[str, onnx.TensorProto]] = None,
    input_sparse_data: Optional[Dict[str, onnx.SparseTensorProto]] = None,
    opset_imports: Optional[List[onnx.OperatorSetIdProto]] = None,
    ir_version: int = onnx.IR_VERSION,
) -> Dict[str, onnx.TypeProto]:
    if not schema.has_type_and_shape_inference_function:  # type: ignore
        return {}
    if input_data is None:
        input_data = {}
    if input_sparse_data is None:
        input_sparse_data = {}
    if opset_imports is None:
        passed_opset_imports = {}
    else:
        passed_opset_imports = {opset.domain: opset.version for opset in opset_imports}

    # catch KeyError if node's input does not exist in input_types
    passed_input_types = {
        key: input_types[key].SerializeToString() for key in node.input
    }
    # input_types will also be used as outer_scope_value_types so do not filter by node's input here
    for key in input_types:
        if key not in passed_input_types:
            passed_input_types[key] = input_types[key].SerializeToString()
    passed_input_data = {
        key: input_data[key].SerializeToString()
        for key in node.input
        if key in input_data
    }
    passed_sparse_input_data = {
        key: input_sparse_data[key].SerializeToString()
        for key in node.input
        if key in input_sparse_data
    }

    outputs = schema._infer_node_outputs(  # pylint: disable=protected-access
        node.SerializeToString(),
        passed_input_types,
        passed_input_data,
        passed_sparse_input_data,
        passed_opset_imports,
        ir_version,
    )  # type: ignore[call-arg]
    return {key: onnx.TypeProto.FromString(out) for key, out in outputs.items()}


def infer_function_output_types(
    function: FunctionProto,
    input_types: Sequence[TypeProto],
    attributes: Sequence[AttributeProto],
) -> List[TypeProto]:
    """
    Apply type-and-shape-inference to given function body, with given input types
    and given input attribute values.
    """
    result = C.infer_function_output_types(
        function.SerializeToString(),
        [x.SerializeToString() for x in input_types],
        [x.SerializeToString() for x in attributes],
    )

    def to_type_proto(x) -> TypeProto:
        type_proto = onnx.TypeProto()
        type_proto.ParseFromString(x)
        return type_proto

    return [to_type_proto(x) for x in result]


InferenceError = C.InferenceError
