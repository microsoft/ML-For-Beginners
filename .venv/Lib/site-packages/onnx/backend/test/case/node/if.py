# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


def compute_if_outputs(x, cond):  # type: ignore
    if cond:
        return []
    else:
        return x


class If(Base):
    @staticmethod
    def export_if() -> None:
        # Given a bool scalar input cond.
        # return constant tensor x if cond is True, otherwise return constant tensor y.

        then_out = onnx.helper.make_tensor_value_info(
            "then_out", onnx.TensorProto.FLOAT, [5]
        )
        else_out = onnx.helper.make_tensor_value_info(
            "else_out", onnx.TensorProto.FLOAT, [5]
        )

        x = np.array([1, 2, 3, 4, 5]).astype(np.float32)
        y = np.array([5, 4, 3, 2, 1]).astype(np.float32)

        then_const_node = onnx.helper.make_node(
            "Constant",
            inputs=[],
            outputs=["then_out"],
            value=onnx.numpy_helper.from_array(x),
        )

        else_const_node = onnx.helper.make_node(
            "Constant",
            inputs=[],
            outputs=["else_out"],
            value=onnx.numpy_helper.from_array(y),
        )

        then_body = onnx.helper.make_graph(
            [then_const_node], "then_body", [], [then_out]
        )

        else_body = onnx.helper.make_graph(
            [else_const_node], "else_body", [], [else_out]
        )

        if_node = onnx.helper.make_node(
            "If",
            inputs=["cond"],
            outputs=["res"],
            then_branch=then_body,
            else_branch=else_body,
        )

        cond = np.array(1).astype(bool)
        res = x if cond else y
        expect(
            if_node,
            inputs=[cond],
            outputs=[res],
            name="test_if",
            opset_imports=[onnx.helper.make_opsetid("", 11)],
        )

    @staticmethod
    def export_if_seq() -> None:
        # Given a bool scalar input cond.
        # return constant sequence x if cond is True, otherwise return constant sequence y.

        then_out = onnx.helper.make_tensor_sequence_value_info(
            "then_out", onnx.TensorProto.FLOAT, shape=[5]
        )
        else_out = onnx.helper.make_tensor_sequence_value_info(
            "else_out", onnx.TensorProto.FLOAT, shape=[5]
        )

        x = [np.array([1, 2, 3, 4, 5]).astype(np.float32)]
        y = [np.array([5, 4, 3, 2, 1]).astype(np.float32)]

        then_const_node = onnx.helper.make_node(
            "Constant",
            inputs=[],
            outputs=["x"],
            value=onnx.numpy_helper.from_array(x[0]),
        )

        then_seq_node = onnx.helper.make_node(
            "SequenceConstruct", inputs=["x"], outputs=["then_out"]
        )

        else_const_node = onnx.helper.make_node(
            "Constant",
            inputs=[],
            outputs=["y"],
            value=onnx.numpy_helper.from_array(y[0]),
        )

        else_seq_node = onnx.helper.make_node(
            "SequenceConstruct", inputs=["y"], outputs=["else_out"]
        )

        then_body = onnx.helper.make_graph(
            [then_const_node, then_seq_node], "then_body", [], [then_out]
        )

        else_body = onnx.helper.make_graph(
            [else_const_node, else_seq_node], "else_body", [], [else_out]
        )

        if_node = onnx.helper.make_node(
            "If",
            inputs=["cond"],
            outputs=["res"],
            then_branch=then_body,
            else_branch=else_body,
        )

        cond = np.array(1).astype(bool)
        res = x if cond else y
        expect(
            if_node,
            inputs=[cond],
            outputs=[res],
            name="test_if_seq",
            opset_imports=[onnx.helper.make_opsetid("", 13)],
        )

    @staticmethod
    def export_if_optional() -> None:
        # Given a bool scalar input cond, return an empty optional sequence of
        # tensor if True, return an optional sequence with value x
        # (the input optional sequence) otherwise.

        ten_in_tp = onnx.helper.make_tensor_type_proto(
            onnx.TensorProto.FLOAT, shape=[5]
        )
        seq_in_tp = onnx.helper.make_sequence_type_proto(ten_in_tp)

        then_out_tensor_tp = onnx.helper.make_tensor_type_proto(
            onnx.TensorProto.FLOAT, shape=[5]
        )
        then_out_seq_tp = onnx.helper.make_sequence_type_proto(then_out_tensor_tp)
        then_out_opt_tp = onnx.helper.make_optional_type_proto(then_out_seq_tp)
        then_out = onnx.helper.make_value_info("optional_empty", then_out_opt_tp)

        else_out_tensor_tp = onnx.helper.make_tensor_type_proto(
            onnx.TensorProto.FLOAT, shape=[5]
        )
        else_out_seq_tp = onnx.helper.make_sequence_type_proto(else_out_tensor_tp)
        else_out_opt_tp = onnx.helper.make_optional_type_proto(else_out_seq_tp)
        else_out = onnx.helper.make_value_info("else_opt", else_out_opt_tp)

        x = [np.array([1, 2, 3, 4, 5]).astype(np.float32)]
        cond = np.array(0).astype(bool)
        res = compute_if_outputs(x, cond)

        opt_empty_in = onnx.helper.make_node(
            "Optional", inputs=[], outputs=["optional_empty"], type=seq_in_tp
        )

        then_body = onnx.helper.make_graph([opt_empty_in], "then_body", [], [then_out])

        else_const_node = onnx.helper.make_node(
            "Constant",
            inputs=[],
            outputs=["x"],
            value=onnx.numpy_helper.from_array(x[0]),
        )

        else_seq_node = onnx.helper.make_node(
            "SequenceConstruct", inputs=["x"], outputs=["else_seq"]
        )

        else_optional_seq_node = onnx.helper.make_node(
            "Optional", inputs=["else_seq"], outputs=["else_opt"]
        )

        else_body = onnx.helper.make_graph(
            [else_const_node, else_seq_node, else_optional_seq_node],
            "else_body",
            [],
            [else_out],
        )

        if_node = onnx.helper.make_node(
            "If",
            inputs=["cond"],
            outputs=["sequence"],
            then_branch=then_body,
            else_branch=else_body,
        )

        expect(
            if_node,
            inputs=[cond],
            outputs=[res],
            name="test_if_opt",
            output_type_protos=[else_out_opt_tp],
            opset_imports=[onnx.helper.make_opsetid("", 16)],
        )
