# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import sys

import numpy as np

import onnx
from onnx import TensorProto, helper
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
from onnx.helper import float32_to_float8e4m3, float32_to_float8e5m2, make_tensor
from onnx.numpy_helper import float8e4m3_to_float32, float8e5m2_to_float32


class CastLike(Base):
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
            ("FLOAT", "FLOAT8E4M3FNUZ"),
            ("FLOAT8E4M3FN", "FLOAT"),
            ("FLOAT8E4M3FNUZ", "FLOAT"),
            ("FLOAT", "FLOAT8E5M2"),
            ("FLOAT", "FLOAT8E5M2FNUZ"),
            ("FLOAT8E5M2", "FLOAT"),
            ("FLOAT8E5M2FNUZ", "FLOAT"),
        ]

        vect_float32_to_float8e4m3 = np.vectorize(float32_to_float8e4m3)
        vect_float32_to_float8e5m2 = np.vectorize(float32_to_float8e5m2)

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
                like = output.flatten()[0:1]
            elif from_type in (
                "FLOAT8E4M3FN",
                "FLOAT8E4M3FNUZ",
                "FLOAT8E5M2",
                "FLOAT8E5M2FNUZ",
            ) or to_type in (
                "FLOAT8E4M3FN",
                "FLOAT8E4M3FNUZ",
                "FLOAT8E5M2",
                "FLOAT8E5M2FNUZ",
            ):
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
                if to_type == "FLOAT8E4M3FN":
                    expected = float8e4m3_to_float32(
                        vect_float32_to_float8e4m3(np_fp32)
                    )
                    expected_tensor = make_tensor(
                        "x", TensorProto.FLOAT8E4M3FN, [3, 4], expected.tolist()
                    )
                    like_tensor = make_tensor(
                        "x", TensorProto.FLOAT8E4M3FN, [1], expected[:1]
                    )
                elif to_type == "FLOAT8E4M3FNUZ":
                    expected = float8e4m3_to_float32(
                        vect_float32_to_float8e4m3(np_fp32, uz=True), uz=True
                    )
                    expected_tensor = make_tensor(
                        "x", TensorProto.FLOAT8E4M3FNUZ, [3, 4], expected.tolist()
                    )
                    like_tensor = make_tensor(
                        "x", TensorProto.FLOAT8E4M3FNUZ, [1], expected[:1]
                    )
                elif to_type == "FLOAT8E5M2":
                    expected = float8e5m2_to_float32(
                        vect_float32_to_float8e5m2(np_fp32)
                    )
                    expected_tensor = make_tensor(
                        "x", TensorProto.FLOAT8E5M2, [3, 4], expected.tolist()
                    )
                    like_tensor = make_tensor(
                        "x", TensorProto.FLOAT8E5M2, [1], expected[:1]
                    )
                elif to_type == "FLOAT8E5M2FNUZ":
                    expected = float8e5m2_to_float32(
                        vect_float32_to_float8e5m2(np_fp32, fn=True, uz=True),
                        fn=True,
                        uz=True,
                    )
                    expected_tensor = make_tensor(
                        "x", TensorProto.FLOAT8E5M2FNUZ, [3, 4], expected.tolist()
                    )
                    like_tensor = make_tensor(
                        "x", TensorProto.FLOAT8E5M2FNUZ, [1], expected[:1]
                    )
                if from_type == "FLOAT":
                    input = np_fp32.reshape((3, 4))
                    output = expected_tensor
                    like = like_tensor
                else:
                    assert to_type == "FLOAT"
                    input = expected_tensor
                    output = expected.reshape((3, 4))
                    like = output.flatten()[:1]
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
                like = output.flatten()[0:1]
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
                like = output.flatten()[0:1]
            node = onnx.helper.make_node(
                "CastLike",
                inputs=["input", "like"],
                outputs=["output"],
            )
            if input_type_proto and output_type_proto:
                like_type_proto = onnx.helper.make_tensor_type_proto(
                    output_type_proto.tensor_type.elem_type, like.shape
                )

                expect(
                    node,
                    inputs=[input, like],
                    outputs=[output],
                    name="test_castlike_" + from_type + "_to_" + to_type,
                    input_type_protos=[input_type_proto, like_type_proto],
                    output_type_protos=[output_type_proto],
                )
            else:
                expect(
                    node,
                    inputs=[input, like],
                    outputs=[output],
                    name="test_castlike_" + from_type + "_to_" + to_type,
                )
