# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


from typing import Sequence

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.model import expect


class NormalizeStrings(Base):
    @staticmethod
    def export() -> None:
        def make_graph(
            node: onnx.helper.NodeProto,
            input_shape: Sequence[int],
            output_shape: Sequence[int],
        ) -> onnx.helper.GraphProto:
            graph = onnx.helper.make_graph(
                nodes=[node],
                name="StringNormalizer",
                inputs=[
                    onnx.helper.make_tensor_value_info(
                        "x", onnx.TensorProto.STRING, input_shape
                    )
                ],
                outputs=[
                    onnx.helper.make_tensor_value_info(
                        "y", onnx.TensorProto.STRING, output_shape
                    )
                ],
            )
            return graph

        # 1st model_monday_casesensintive_nochangecase
        stopwords = ["monday"]
        node = onnx.helper.make_node(
            "StringNormalizer",
            inputs=["x"],
            outputs=["y"],
            is_case_sensitive=1,
            stopwords=stopwords,
        )

        x = np.array(["monday", "tuesday", "wednesday", "thursday"]).astype(object)
        y = np.array(["tuesday", "wednesday", "thursday"]).astype(object)

        graph = make_graph(node, [4], [3])
        model = onnx.helper.make_model_gen_version(
            graph,
            producer_name="backend-test",
            opset_imports=[onnx.helper.make_opsetid("", 10)],
        )
        expect(
            model,
            inputs=[x],
            outputs=[y],
            name="test_strnorm_model_monday_casesensintive_nochangecase",
        )

        # 2nd model_nostopwords_nochangecase
        node = onnx.helper.make_node(
            "StringNormalizer", inputs=["x"], outputs=["y"], is_case_sensitive=1
        )

        x = np.array(["monday", "tuesday"]).astype(object)
        y = x

        graph = make_graph(node, [2], [2])
        model = onnx.helper.make_model_gen_version(
            graph,
            producer_name="backend-test",
            opset_imports=[onnx.helper.make_opsetid("", 10)],
        )
        expect(
            model,
            inputs=[x],
            outputs=[y],
            name="test_strnorm_model_nostopwords_nochangecase",
        )

        # 3rd model_monday_casesensintive_lower
        stopwords = ["monday"]
        node = onnx.helper.make_node(
            "StringNormalizer",
            inputs=["x"],
            outputs=["y"],
            case_change_action="LOWER",
            is_case_sensitive=1,
            stopwords=stopwords,
        )

        x = np.array(["monday", "tuesday", "wednesday", "thursday"]).astype(object)
        y = np.array(["tuesday", "wednesday", "thursday"]).astype(object)

        graph = make_graph(node, [4], [3])
        model = onnx.helper.make_model_gen_version(
            graph,
            producer_name="backend-test",
            opset_imports=[onnx.helper.make_opsetid("", 10)],
        )
        expect(
            model,
            inputs=[x],
            outputs=[y],
            name="test_strnorm_model_monday_casesensintive_lower",
        )

        # 4 model_monday_casesensintive_upper
        stopwords = ["monday"]
        node = onnx.helper.make_node(
            "StringNormalizer",
            inputs=["x"],
            outputs=["y"],
            case_change_action="UPPER",
            is_case_sensitive=1,
            stopwords=stopwords,
        )

        x = np.array(["monday", "tuesday", "wednesday", "thursday"]).astype(object)
        y = np.array(["TUESDAY", "WEDNESDAY", "THURSDAY"]).astype(object)

        graph = make_graph(node, [4], [3])
        model = onnx.helper.make_model_gen_version(
            graph,
            producer_name="backend-test",
            opset_imports=[onnx.helper.make_opsetid("", 10)],
        )
        expect(
            model,
            inputs=[x],
            outputs=[y],
            name="test_strnorm_model_monday_casesensintive_upper",
        )

        # 5 monday_insensintive_upper_twodim
        stopwords = ["monday"]
        node = onnx.helper.make_node(
            "StringNormalizer",
            inputs=["x"],
            outputs=["y"],
            case_change_action="UPPER",
            stopwords=stopwords,
        )

        input_shape = [1, 6]
        output_shape = [1, 4]
        x = (
            np.array(
                ["Monday", "tuesday", "wednesday", "Monday", "tuesday", "wednesday"]
            )
            .astype(object)
            .reshape(input_shape)
        )
        y = (
            np.array(["TUESDAY", "WEDNESDAY", "TUESDAY", "WEDNESDAY"])
            .astype(object)
            .reshape(output_shape)
        )

        graph = make_graph(node, input_shape, output_shape)
        model = onnx.helper.make_model_gen_version(
            graph,
            producer_name="backend-test",
            opset_imports=[onnx.helper.make_opsetid("", 10)],
        )
        expect(
            model,
            inputs=[x],
            outputs=[y],
            name="test_strnorm_model_monday_insensintive_upper_twodim",
        )

        # 6 monday_empty_output
        stopwords = ["monday"]
        node = onnx.helper.make_node(
            "StringNormalizer",
            inputs=["x"],
            outputs=["y"],
            case_change_action="UPPER",
            is_case_sensitive=0,
            stopwords=stopwords,
        )

        x = np.array(["monday", "monday"]).astype(object)
        y = np.array([""]).astype(object)

        graph = make_graph(node, [2], [1])
        model = onnx.helper.make_model_gen_version(
            graph,
            producer_name="backend-test",
            opset_imports=[onnx.helper.make_opsetid("", 10)],
        )
        expect(
            model,
            inputs=[x],
            outputs=[y],
            name="test_strnorm_model_monday_empty_output",
        )
