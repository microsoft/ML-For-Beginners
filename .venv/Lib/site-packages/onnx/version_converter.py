# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
"""onnx version converter

This enables users to convert their models between different opsets within the
default domain ("" or "ai.onnx").
"""

import onnx
import onnx.onnx_cpp2py_export.version_converter as C  # noqa: N812
from onnx import ModelProto


def convert_version(model: ModelProto, target_version: int) -> ModelProto:
    """Apply the version conversion on the serialized ModelProto.

    Arguments:
        input (ModelProto): model
        target_version (int): target opset version

    Returns:
        return (ModelProto) converted model

    Raises Exceptions:
        RuntimeError when some necessary conversion is not supported

    Supported adapters:
        - Add from Opset 7 to Opset 6
        - Add from Opset 6 to Opset 5
        - Add from Opset 6 to Opset 7
        - Add from Opset 5 to Opset 6
        - Mul from Opset 6 to Opset 7
        - Mul from Opset 7 to Opset 6
        - Mul from Opset 6 to Opset 5
        - Mul from Opset 5 to Opset 6
        - Gemm from Opset 7 to Opset 6
        - Gemm from Opset 6 to Opset 5
        - Gemm from Opset 6 to Opset 7
        - Gemm from Opset 5 to Opset 6
        - Relu from Opset 6 to Opset 5
        - Relu from Opset 5 to Opset 6
        - BatchNorm from Opset 7 to Opset 6
        - BatchNorm from Opset 6 to Opset 7
        - BatchNorm from Opset 6 to Opset 5
        - BatchNorm from Opset 5 to Opset 6
        - Concat from Opset 4 to Opset 3
        - Concat from Opset 3 to Opset 4
        - Reshape from Opset 5 to Opset 4
        - Reshape from Opset 4 to Opset 5
        - Sum from Opset 7 to Opset 8
        - Sum from Opset 8 to Opset 7
        - Sum from Opset 6 to Opset 5
        - Sum from Opset 5 to Opset 6
        - MaxPool from Opset 8 to Opset 7
        - MaxPool from Opset 7 to Opset 8
        - AveragePool from Opset 7 to Opset 6
        - AveragePool from Opset 6 to Opset 7
        - Dropout from Opset 7 to Opset 6
        - Dropout from Opset 6 to Opset 5
        - Dropout from Opset 6 to Opset 7
        - Dropout from Opset 5 to Opset 6
        - RNN from Opset 13 to Opset 14
        - RNN from Opset 14 to Opset 13
        - GRU from Opset 13 to Opset 14
        - GRU from Opset 14 to Opset 13
        - LSTM from Opset 13 to Opset 14
        - LSTM from Opset 14 to Opset 13

    Unsupported adapters:
        - Min from Opset 8 to Opset 7
        - Min from Opset 7 to Opset 8
        - Min from Opset 6 to Opset 5
        - Min from Opset 5 to Opset 6
        - Mean from Opset 8 to Opset 7
        - Mean from Opset 7 to Opset 8
        - Mean from Opset 6 to Opset 5
        - Mean from Opset 5 to Opset 6
        - Max from Opset 8 to Opset 7
        - Max from Opset 7 to Opset 8
        - Max from Opset 6 to Opset 5
        - Max from Opset 5 to Opset 6
        - Xor from Opset 6 to Opset 7
        - Xor from Opset 7 to Opset 6
        - Upsample from Opset 6 to Opset 7
        - Upsample from Opset 7 to Opset 6
        - Sub from Opset 6 to Opset 7
        - Sub from Opset 7 to Opset 6
        - Sub from Opset 6 to Opset 5
        - Sub from Opset 5 to Opset 6
        - RNN from Opset 6 to Opset 7
        - RNN from Opset 7 to Opset 6
        - Pow from Opset 6 to Opset 7
        - Pow from Opset 7 to Opset 6
        - PRelu from Opset 6 to Opset 7
        - PRelu from Opset 7 to Opset 6
        - PRelu from Opset 6 to Opset 5
        - PRelu from Opset 5 to Opset 6
        - Or from Opset 6 to Opset 7
        - Or from Opset 7 to Opset 6
        - Less from Opset 6 to Opset 7
        - Less from Opset 7 to Opset 6
        - LSTM from Opset 6 to Opset 7
        - LSTM from Opset 7 to Opset 6
        - Greater from Opset 6 to Opset 7
        - Greater from Opset 7 to Opset 6
        - GRU from Opset 6 to Opset 7
        - GRU from Opset 7 to Opset 6
        - GRU from Opset 3 to Opset 2
        - GRU from Opset 2 to Opset 3
        - Equal from Opset 6 to Opset 7
        - Equal from Opset 7 to Opset 6
        - Div from Opset 6 to Opset 7
        - Div from Opset 7 to Opset 6
        - Div from Opset 6 to Opset 5
        - Div from Opset 5 to Opset 6
        - And from Opset 6 to Opset 7
        - And from Opset 7 to Opset 6
        - And from Opset 6 to Opset 5
        - And from Opset 5 to Opset 6
        - Tile from Opset 6 to Opset 5
        - Tile from Opset 5 to Opset 6
        - Sqrt from Opset 6 to Opset 5
        - Sqrt from Opset 5 to Opset 6
        - Sigmoid from opset 6 to opset 5
        - Sigmoid from opset 5 to opset 6
        - Selu from opset 6 to opset 5
        - Selu from opset 5 to opset 6
        - Reciprocal from opset 6 to opset 5
        - Reciprocal from opset 5 to opset 6
        - Neg from opset 6 to opset 5
        - Neg from opset 5 to opset 6
        - Log from opset 6 to opset 5
        - Log from opset 5 to opset 6
        - LeakyRelu from opset 6 to opset 5
        - LeakyRelu from opset 5 to opset 6
        - InstanceNormalization from opset 6 to opset 5
        - InstanceNormalization from opset 5 to opset 6
        - HardSigmoid from opset 6 to opset 5
        - HardSigmoid from opset 5 to opset 6
        - Floor from opset 6 to opset 5
        - Floor from opset 5 to opset 6
        - Exp from opset 6 to opset 5
        - Exp from opset 5 to opset 6
        - Elu from opset 6 to opset 5
        - Elu from opset 5 to opset 6
        - Clip from opset 6 to opset 5
        - Clip from opset 5 to opset 6
        - Ceil from opset 6 to opset 5
        - Ceil from opset 5 to opset 6
        - Cast from opset 6 to opset 5
        - Cast from opset 5 to opset 6
        - Abs from opset 6 to opset 5
        - Abs from opset 5 to opset 6
        - Split from opset 2 to opset 1
        - Split from opset 1 to opset 2
        - Pad from opset 2 to opset 1
        - Pad from opset 1 to opset 2
        - LpPool from opset 2 to opset 1
        - LpPool from opset 1 to opset 2
        - GlobalLpPool from opset 2 to opset 1
        - GlobalLpPool from opset 1 to opset 2
    """
    if not isinstance(model, ModelProto):
        raise ValueError(
            f"VersionConverter only accepts ModelProto as model, incorrect type: {type(model)}"
        )
    if not isinstance(target_version, int):
        raise ValueError(
            f"VersionConverter only accepts int as target_version, incorrect type: {type(target_version)}"
        )
    model_str = model.SerializeToString()
    converted_model_str = C.convert_version(model_str, target_version)
    return onnx.load_from_string(converted_model_str)


ConvertError = C.ConvertError
