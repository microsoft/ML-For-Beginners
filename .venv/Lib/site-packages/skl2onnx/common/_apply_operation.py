# SPDX-License-Identifier: Apache-2.0

import numpy as np

try:
    from onnx.helper import np_dtype_to_tensor_dtype
except ImportError:
    from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE

    def np_dtype_to_tensor_dtype(dtype):
        return NP_TYPE_TO_TENSOR_TYPE[dtype]


from onnxconverter_common.onnx_ops import *  # noqa
from ..proto import onnx_proto


def apply_normalizer(scope, inputs, outputs, container, norm, use_float):
    """
    Adds operator Normalizer if *use_float* is true,
    otherwise, uses *ReduceSum* + *Div*. *Normalizer*
    always produces float according to ONNX speciciations.
    """
    input = inputs[0] if isinstance(inputs, list) else inputs
    output = outputs[0] if isinstance(outputs, list) else outputs
    use_normalizer = container.is_allowed({"Normalizer"})

    if use_normalizer and use_float:
        container.add_node(
            "Normalizer",
            input,
            output,
            op_domain="ai.onnx.ml",
            norm=norm,
            name=scope.get_unique_operator_name("Normalizer"),
        )
    else:
        # Normalizer only produces floats.
        if norm == "L1":
            norm = scope.get_unique_variable_name("norm")
            norm_abs = scope.get_unique_variable_name("norm_abs")
            container.add_node(
                "Abs", input, norm_abs, name=scope.get_unique_operator_name("Abs")
            )

            if container.target_opset < 13:
                container.add_node(
                    "ReduceSum",
                    norm_abs,
                    norm,
                    axes=[1],
                    keepdims=1,
                    name=scope.get_unique_operator_name("ReduceSum"),
                )
            else:
                axis_name = scope.get_unique_variable_name("axis")
                container.add_initializer(
                    axis_name, onnx_proto.TensorProto.INT64, [1], [1]
                )
                container.add_node(
                    "ReduceSum",
                    [norm_abs, axis_name],
                    norm,
                    keepdims=1,
                    name=scope.get_unique_operator_name("ReduceSum"),
                )
            apply_div(  # noqa
                scope,
                [input, norm],
                output,
                container,
                operator_name=scope.get_unique_operator_name("NormalizerNorm"),
            )
        elif norm == "L2":
            norm = scope.get_unique_variable_name("norm")
            norm2 = scope.get_unique_variable_name("norm2")
            if container.target_opset < 18:
                container.add_node(
                    "ReduceSumSquare",
                    input,
                    norm,
                    axes=[1],
                    keepdims=1,
                    name=scope.get_unique_operator_name("ReduceSumSquare"),
                )
            else:
                axis_name = scope.get_unique_variable_name("axis")
                container.add_initializer(
                    axis_name, onnx_proto.TensorProto.INT64, [1], [1]
                )
                container.add_node(
                    "ReduceSumSquare",
                    [input, axis_name],
                    norm,
                    keepdims=1,
                    name=scope.get_unique_operator_name("ReduceSumSquare"),
                )
            container.add_node(
                "Sqrt", [norm], norm2, name=scope.get_unique_operator_name("Sqrt")
            )
            apply_div(  # noqa
                scope,
                [input, norm2],
                output,
                container,
                operator_name=scope.get_unique_operator_name("NormalizerNorm"),
            )
        else:
            raise NotImplementedError(
                "Normalization not implemented for norm %r." % norm
            )


def _create_name_or_use_existing_one(scope, op_type, name):
    if name is None:
        return scope.get_unique_operator_name(op_type)
    return name


def apply_clip(
    scope, input_name, output_name, container, operator_name=None, max=None, min=None
):
    name = _create_name_or_use_existing_one(scope, "Clip", operator_name)
    attrs = {"name": name}

    if container.target_opset < 11:
        if max is not None:
            attrs["max"] = float(max)
        if min is not None:
            attrs["min"] = float(min)

        if container.target_opset < 6:
            attrs["consumed_inputs"] = [0]
            op_version = 1
        else:
            op_version = 6

        container.add_node(
            "Clip", input_name, output_name, op_version=op_version, **attrs
        )
    else:
        if container.target_opset < 12:
            op_version = 11
        else:
            op_version = 12
        if min is None and max is not None:
            raise RuntimeError("Operator 'Clip': min must be specified if max is.")
        inputs = [input_name]

        if min is not None:
            if isinstance(
                min,
                (np.ndarray, float, int, np.float32, np.float64, np.int64, np.int32),
            ):
                # add initializer
                if isinstance(min, np.ndarray):
                    if len(min.shape) == 0:
                        min = [min]
                    elif min.shape == (1,):
                        min = list(min[0]) if hasattr(min[0], "__iter__") else list(min)
                    else:
                        raise RuntimeError("min must be an array of one element.")
                else:
                    min = [min]

                # container in sklearn-onnx stores the computation type in
                # container.dtype.
                min_name = scope.get_unique_variable_name("clip_min")
                if op_version < 12:
                    min = np.array(min, dtype=getattr(container, "dtype", np.float32))
                    container.add_initializer(
                        min_name,
                        getattr(container, "proto_dtype", onnx_proto.TensorProto.FLOAT),
                        [],
                        [min[0]],
                    )
                else:
                    min = np.array(min)
                    container.add_initializer(
                        min_name, np_dtype_to_tensor_dtype(min.dtype), [], [min[0]]
                    )
                min = min_name
            if isinstance(min, str):
                inputs.append(min)
            else:
                raise RuntimeError("Parameter 'min' must be a string or a float.")

        if max is not None:
            if min is None:
                raise RuntimeError("Parameter 'min' must be specified if 'max' is.")
            if isinstance(
                max,
                (np.ndarray, float, int, np.float32, np.float64, np.int64, np.int32),
            ):
                # add initializer
                if isinstance(max, np.ndarray):
                    if len(max.shape) == 0:
                        max = [max]
                    elif max.shape == (1,):
                        max = list(max[0]) if hasattr(max[0], "__iter__") else list(max)
                    else:
                        raise RuntimeError("max must be an array of one element.")
                else:
                    max = [max]

                max_name = scope.get_unique_variable_name("clip_max")
                if op_version < 12:
                    max = np.array(max, dtype=getattr(container, "dtype", np.float32))
                    container.add_initializer(
                        max_name,
                        getattr(container, "proto_dtype", onnx_proto.TensorProto.FLOAT),
                        [],
                        [max[0]],
                    )
                else:
                    max = np.array(max)
                    container.add_initializer(
                        max_name, np_dtype_to_tensor_dtype(max.dtype), [], [max[0]]
                    )
                max = max_name
            if isinstance(max, str):
                inputs.append(max)
            else:
                raise RuntimeError("Parameter 'max' must be a string or a float.")

        container.add_node("Clip", inputs, output_name, op_version=op_version, **attrs)
