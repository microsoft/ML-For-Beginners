# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import sys

import numpy as np

import onnx
from onnx import TensorProto, helper
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
from onnx.helper import (
    float32_to_float8e4m3,
    float32_to_float8e5m2,
    make_tensor,
    tensor_dtype_to_field,
)
from onnx.numpy_helper import float8e4m3_to_float32, float8e5m2_to_float32


class Cast(Base):
    @staticmethod
    def export() -> None:
        shape = (3, 4)
        test_cases = [
            ("FLOAT", "FLOAT16"),
            ("FLOAT", "DOUBLE"),
            ("FLOAT16", "FLOAT"),
            ("FLOAT16", "DOUBLE"),
            ("DOUBLE", "FLOAT"),
            ("DOUBLE", "FLOAT16"),
            ("FLOAT", "STRING"),
            ("STRING", "FLOAT"),
            ("FLOAT", "BFLOAT16"),
            ("BFLOAT16", "FLOAT"),
            ("FLOAT", "FLOAT8E4M3FN"),
            ("FLOAT16", "FLOAT8E4M3FN"),
            ("FLOAT", "FLOAT8E4M3FNUZ"),
            ("FLOAT16", "FLOAT8E4M3FNUZ"),
            ("FLOAT8E4M3FN", "FLOAT"),
            ("FLOAT8E4M3FN", "FLOAT16"),
            ("FLOAT8E4M3FNUZ", "FLOAT"),
            ("FLOAT8E4M3FNUZ", "FLOAT16"),
            ("FLOAT", "FLOAT8E5M2"),
            ("FLOAT16", "FLOAT8E5M2"),
            ("FLOAT", "FLOAT8E5M2FNUZ"),
            ("FLOAT16", "FLOAT8E5M2FNUZ"),
            ("FLOAT8E5M2", "FLOAT"),
            ("FLOAT8E5M2", "FLOAT16"),
            ("FLOAT8E5M2FNUZ", "FLOAT"),
            ("FLOAT8E5M2FNUZ", "FLOAT16"),
        ]

        vect_float32_to_float8e4m3 = np.vectorize(float32_to_float8e4m3)
        vect_float32_to_float8e5m2 = np.vectorize(float32_to_float8e5m2)
        f8_types = ("FLOAT8E4M3FN", "FLOAT8E4M3FNUZ", "FLOAT8E5M2", "FLOAT8E5M2FNUZ")

        for from_type, to_type in test_cases:
            input_type_proto = None
            output_type_proto = None
            if from_type == "BFLOAT16" or to_type == "BFLOAT16":
                np_fp32 = np.array(
                    [
                        "0.47892547",
                        "0.48033667",
                        "0.49968487",
                        "0.81910545",
                        "0.47031248",
                        "0.816468",
                        "0.21087195",
                        "0.7229038",
                        "NaN",
                        "INF",
                        "+INF",
                        "-INF",
                    ],
                    dtype=np.float32,
                )
                little_endisan = sys.byteorder == "little"
                np_uint16_view = np_fp32.view(dtype=np.uint16)
                np_bfp16 = (
                    np_uint16_view[1::2] if little_endisan else np_uint16_view[0::2]
                )
                if to_type == "BFLOAT16":
                    assert from_type == "FLOAT"
                    input = np_fp32.reshape([3, 4])
                    output = np_bfp16.reshape([3, 4])
                    input_type_proto = onnx.helper.make_tensor_type_proto(
                        int(TensorProto.FLOAT), input.shape
                    )
                    output_type_proto = onnx.helper.make_tensor_type_proto(
                        int(TensorProto.BFLOAT16), output.shape
                    )
                else:
                    assert to_type == "FLOAT"
                    input = np_bfp16.reshape([3, 4])
                    # convert bfloat to FLOAT
                    np_fp32_zeros = np.zeros((len(np_bfp16) * 2,), dtype=np.uint16)
                    if little_endisan:
                        np_fp32_zeros[1::2] = np_bfp16
                    else:
                        np_fp32_zeros[0::2] = np_bfp16
                    np_fp32_from_bfloat = np_fp32_zeros.view(dtype=np.float32)
                    output = np_fp32_from_bfloat.reshape([3, 4])
                    input_type_proto = onnx.helper.make_tensor_type_proto(
                        int(TensorProto.BFLOAT16), input.shape
                    )
                    output_type_proto = onnx.helper.make_tensor_type_proto(
                        int(TensorProto.FLOAT), output.shape
                    )
            elif from_type in f8_types or to_type in f8_types:
                np_fp32 = np.array(
                    [
                        "0.47892547",
                        "0.48033667",
                        "0.49968487",
                        "0.81910545",
                        "0.47031248",
                        "0.7229038",
                        "1000000",
                        "1e-7",
                        "NaN",
                        "INF",
                        "+INF",
                        "-INF",
                    ],
                    dtype=np.float32,
                )

                if from_type == "FLOAT":
                    input_values = np_fp32
                    input = make_tensor(
                        "x", TensorProto.FLOAT, [3, 4], np_fp32.tolist()
                    )
                elif from_type == "FLOAT16":
                    input_values = np_fp32.astype(np.float16).astype(np.float32)
                    input = make_tensor(
                        "x", TensorProto.FLOAT16, [3, 4], input_values.tolist()
                    )
                elif from_type == "FLOAT8E4M3FN":
                    input_values = float8e4m3_to_float32(
                        vect_float32_to_float8e4m3(np_fp32)
                    )
                    input = make_tensor(
                        "x", TensorProto.FLOAT8E4M3FN, [3, 4], input_values.tolist()
                    )
                elif from_type == "FLOAT8E4M3FNUZ":
                    input_values = float8e4m3_to_float32(
                        vect_float32_to_float8e4m3(np_fp32, uz=True), uz=True
                    )
                    input = make_tensor(
                        "x", TensorProto.FLOAT8E4M3FNUZ, [3, 4], input_values.tolist()
                    )
                elif from_type == "FLOAT8E5M2":
                    input_values = float8e5m2_to_float32(
                        vect_float32_to_float8e5m2(np_fp32)
                    )
                    input = make_tensor(
                        "x", TensorProto.FLOAT8E5M2, [3, 4], input_values.tolist()
                    )
                elif from_type == "FLOAT8E5M2FNUZ":
                    input_values = float8e5m2_to_float32(
                        vect_float32_to_float8e5m2(np_fp32, fn=True, uz=True),
                        fn=True,
                        uz=True,
                    )
                    input = make_tensor(
                        "x", TensorProto.FLOAT8E5M2FNUZ, [3, 4], input_values.tolist()
                    )
                else:
                    raise ValueError(
                        "Conversion from {from_type} to {to_type} is not tested."
                    )

                if to_type == "FLOAT8E4M3FN":
                    expected = float8e4m3_to_float32(
                        vect_float32_to_float8e4m3(input_values)
                    )
                elif to_type == "FLOAT8E4M3FNUZ":
                    expected = float8e4m3_to_float32(
                        vect_float32_to_float8e4m3(input_values, uz=True), uz=True
                    )
                elif to_type == "FLOAT8E5M2":
                    expected = float8e5m2_to_float32(
                        vect_float32_to_float8e5m2(input_values)
                    )
                elif to_type == "FLOAT8E5M2FNUZ":
                    expected = float8e5m2_to_float32(
                        vect_float32_to_float8e5m2(input_values, fn=True, uz=True),
                        fn=True,
                        uz=True,
                    )
                elif to_type == "FLOAT16":
                    expected = input_values.astype(np.float16).astype(np.float32)
                elif to_type == "FLOAT":
                    expected = input_values
                else:
                    raise ValueError(
                        "Conversion from {from_type} to {to_type} is not tested."
                    )
                expected_tensor = make_tensor(
                    "x", getattr(TensorProto, to_type), [3, 4], expected.tolist()
                )
                output = expected_tensor

            elif from_type != "STRING":
                input = np.random.random_sample(shape).astype(
                    helper.tensor_dtype_to_np_dtype(getattr(TensorProto, from_type))
                )
                if to_type == "STRING":
                    # Converting input to str, then give it object dtype for generating script
                    ss = []
                    for i in input.flatten():
                        s = str(i).encode("utf-8")
                        su = s.decode("utf-8")
                        ss.append(su)

                    output = np.array(ss).astype(object).reshape([3, 4])
                else:
                    output = input.astype(
                        helper.tensor_dtype_to_np_dtype(getattr(TensorProto, to_type))
                    )
            else:
                input = np.array(
                    [
                        "0.47892547",
                        "0.48033667",
                        "0.49968487",
                        "0.81910545",
                        "0.47031248",
                        "0.816468",
                        "0.21087195",
                        "0.7229038",
                        "NaN",
                        "INF",
                        "+INF",
                        "-INF",
                    ],
                    dtype=np.dtype(object),
                ).reshape([3, 4])
                output = input.astype(
                    helper.tensor_dtype_to_np_dtype(getattr(TensorProto, to_type))
                )
            node = onnx.helper.make_node(
                "Cast",
                inputs=["input"],
                outputs=["output"],
                to=getattr(TensorProto, to_type),
            )
            if input_type_proto and output_type_proto:
                expect(
                    node,
                    inputs=[input],
                    outputs=[output],
                    name="test_cast_" + from_type + "_to_" + to_type,
                    input_type_protos=[input_type_proto],
                    output_type_protos=[output_type_proto],
                )
            else:
                expect(
                    node,
                    inputs=[input],
                    outputs=[output],
                    name="test_cast_" + from_type + "_to_" + to_type,
                )

    @staticmethod
    def export_saturate_false() -> None:
        test_cases = [
            ("FLOAT", "FLOAT8E4M3FN"),
            ("FLOAT16", "FLOAT8E4M3FN"),
            ("FLOAT", "FLOAT8E4M3FNUZ"),
            ("FLOAT16", "FLOAT8E4M3FNUZ"),
            ("FLOAT", "FLOAT8E5M2"),
            ("FLOAT16", "FLOAT8E5M2"),
            ("FLOAT", "FLOAT8E5M2FNUZ"),
            ("FLOAT16", "FLOAT8E5M2FNUZ"),
        ]
        vect_float32_to_float8e4m3 = np.vectorize(float32_to_float8e4m3)
        vect_float32_to_float8e5m2 = np.vectorize(float32_to_float8e5m2)

        for from_type, to_type in test_cases:
            np_fp32 = np.array(
                [
                    "0.47892547",
                    "0.48033667",
                    "0.49968487",
                    "0.81910545",
                    "0.47031248",
                    "0.7229038",
                    "1000000",
                    "1e-7",
                    "NaN",
                    "INF",
                    "+INF",
                    "-INF",
                ],
                dtype=np.float32,
            )

            if from_type == "FLOAT":
                input_values = np_fp32
                input = make_tensor("x", TensorProto.FLOAT, [3, 4], np_fp32.tolist())
            elif from_type == "FLOAT16":
                input_values = np_fp32.astype(np.float16).astype(np.float32)
                input = make_tensor(
                    "x", TensorProto.FLOAT16, [3, 4], input_values.tolist()
                )
            else:
                raise ValueError(
                    "Conversion from {from_type} to {to_type} is not tested."
                )

            if to_type == "FLOAT8E4M3FN":
                expected = vect_float32_to_float8e4m3(input_values, saturate=False)
            elif to_type == "FLOAT8E4M3FNUZ":
                expected = vect_float32_to_float8e4m3(
                    input_values, uz=True, saturate=False
                )
            elif to_type == "FLOAT8E5M2":
                expected = vect_float32_to_float8e5m2(input_values, saturate=False)
            elif to_type == "FLOAT8E5M2FNUZ":
                expected = vect_float32_to_float8e5m2(
                    input_values, fn=True, uz=True, saturate=False
                )
            else:
                raise ValueError(
                    "Conversion from {from_type} to {to_type} is not tested."
                )

            ivals = bytes([int(i) for i in expected])
            tensor = TensorProto()
            tensor.data_type = getattr(TensorProto, to_type)
            tensor.name = "x"
            tensor.dims.extend([3, 4])
            field = tensor_dtype_to_field(tensor.data_type)
            getattr(tensor, field).extend(ivals)

            output = tensor

            node = onnx.helper.make_node(
                "Cast",
                inputs=["input"],
                outputs=["output"],
                to=getattr(TensorProto, to_type),
                saturate=0,
            )
            expect(
                node,
                inputs=[input],
                outputs=[output],
                name="test_cast_no_saturate_" + from_type + "_to_" + to_type,
            )
