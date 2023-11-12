# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, List

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


def compute_loop_outputs(x, seq, trip_count):  # type: ignore
    for i in range(trip_count):
        if seq is None:
            seq = []
        seq += [x[: int(i + 1)]]
    return seq


class Loop(Base):
    @staticmethod
    def export_loop_11() -> None:
        # Given a tensor x of values [x1, ..., xN], and initial tensor y
        # sum up its elements using a scan
        # returning the final state (y+x1+x2+...+xN) as well the scan_output
        # [y+x1, y+x1+x2, ..., y+x1+x2+...+xN]

        y_in = onnx.helper.make_tensor_value_info("y_in", onnx.TensorProto.FLOAT, [1])
        y_out = onnx.helper.make_tensor_value_info("y_out", onnx.TensorProto.FLOAT, [1])
        scan_out = onnx.helper.make_tensor_value_info(
            "scan_out", onnx.TensorProto.FLOAT, [1]
        )
        cond_in = onnx.helper.make_tensor_value_info(
            "cond_in", onnx.TensorProto.BOOL, []
        )
        cond_out = onnx.helper.make_tensor_value_info(
            "cond_out", onnx.TensorProto.BOOL, []
        )
        iter_count = onnx.helper.make_tensor_value_info(
            "iter_count", onnx.TensorProto.INT64, []
        )

        x = np.array([1, 2, 3, 4, 5]).astype(np.float32)
        y = np.array([-2]).astype(np.float32)

        x_const_node = onnx.helper.make_node(
            "Constant",
            inputs=[],
            outputs=["x"],
            value=onnx.helper.make_tensor(
                name="const_tensor_x",
                data_type=onnx.TensorProto.FLOAT,
                dims=x.shape,
                vals=x.flatten().astype(float),
            ),
        )

        one_const_node = onnx.helper.make_node(
            "Constant",
            inputs=[],
            outputs=["one"],
            value=onnx.helper.make_tensor(
                name="const_tensor_one",
                data_type=onnx.TensorProto.INT64,
                dims=(),
                vals=[1],
            ),
        )

        i_add_node = onnx.helper.make_node(
            "Add", inputs=["iter_count", "one"], outputs=["end"]
        )

        start_unsqueeze_node = onnx.helper.make_node(
            "Unsqueeze", inputs=["iter_count"], outputs=["slice_start"], axes=[0]
        )

        end_unsqueeze_node = onnx.helper.make_node(
            "Unsqueeze", inputs=["end"], outputs=["slice_end"], axes=[0]
        )

        slice_node = onnx.helper.make_node(
            "Slice", inputs=["x", "slice_start", "slice_end"], outputs=["slice_out"]
        )

        y_add_node = onnx.helper.make_node(
            "Add", inputs=["y_in", "slice_out"], outputs=["y_out"]
        )

        identity_node = onnx.helper.make_node(
            "Identity", inputs=["cond_in"], outputs=["cond_out"]
        )

        scan_identity_node = onnx.helper.make_node(
            "Identity", inputs=["y_out"], outputs=["scan_out"]
        )

        loop_body = onnx.helper.make_graph(
            [
                identity_node,
                x_const_node,
                one_const_node,
                i_add_node,
                start_unsqueeze_node,
                end_unsqueeze_node,
                slice_node,
                y_add_node,
                scan_identity_node,
            ],
            "loop_body",
            [iter_count, cond_in, y_in],
            [cond_out, y_out, scan_out],
        )

        node = onnx.helper.make_node(
            "Loop",
            inputs=["trip_count", "cond", "y"],
            outputs=["res_y", "res_scan"],
            body=loop_body,
        )

        trip_count = np.array(5).astype(np.int64)
        res_y = np.array([13]).astype(np.float32)
        cond = np.array(1).astype(bool)
        res_scan = np.array([-1, 1, 4, 8, 13]).astype(np.float32).reshape((5, 1))
        expect(
            node,
            inputs=[trip_count, cond, y],
            outputs=[res_y, res_scan],
            name="test_loop11",
            opset_imports=[onnx.helper.make_opsetid("", 11)],
        )

    @staticmethod
    def export_loop_13() -> None:
        # Given a tensor x of values [x1, ..., xN],
        # Return a sequence of tensors of
        #   [[x1], [x1, x2], ..., [x1, ..., xN]]

        seq_in = onnx.helper.make_tensor_sequence_value_info(
            "seq_in", onnx.TensorProto.FLOAT, None
        )
        seq_out = onnx.helper.make_tensor_sequence_value_info(
            "seq_out", onnx.TensorProto.FLOAT, None
        )
        cond_in = onnx.helper.make_tensor_value_info(
            "cond_in", onnx.TensorProto.BOOL, []
        )
        cond_out = onnx.helper.make_tensor_value_info(
            "cond_out", onnx.TensorProto.BOOL, []
        )
        iter_count = onnx.helper.make_tensor_value_info(
            "iter_count", onnx.TensorProto.INT64, []
        )

        x = np.array([1, 2, 3, 4, 5]).astype(np.float32)

        x_const_node = onnx.helper.make_node(
            "Constant",
            inputs=[],
            outputs=["x"],
            value=onnx.helper.make_tensor(
                name="const_tensor_x",
                data_type=onnx.TensorProto.FLOAT,
                dims=x.shape,
                vals=x.flatten().astype(float),
            ),
        )

        one_const_node = onnx.helper.make_node(
            "Constant",
            inputs=[],
            outputs=["one"],
            value=onnx.helper.make_tensor(
                name="const_tensor_one",
                data_type=onnx.TensorProto.INT64,
                dims=(),
                vals=[1],
            ),
        )

        zero_const_node = onnx.helper.make_node(
            "Constant",
            inputs=[],
            outputs=["slice_start"],
            value=onnx.helper.make_tensor(
                name="const_tensor_zero",
                data_type=onnx.TensorProto.INT64,
                dims=(1,),
                vals=[0],
            ),
        )

        axes_node = onnx.helper.make_node(
            "Constant",
            inputs=[],
            outputs=["axes"],
            value=onnx.helper.make_tensor(
                name="const_tensor_axes",
                data_type=onnx.TensorProto.INT64,
                dims=(),
                vals=[0],
            ),
        )

        add_node = onnx.helper.make_node(
            "Add", inputs=["iter_count", "one"], outputs=["end"]
        )

        end_unsqueeze_node = onnx.helper.make_node(
            "Unsqueeze", inputs=["end", "axes"], outputs=["slice_end"]
        )

        slice_node = onnx.helper.make_node(
            "Slice", inputs=["x", "slice_start", "slice_end"], outputs=["slice_out"]
        )

        insert_node = onnx.helper.make_node(
            "SequenceInsert", inputs=["seq_in", "slice_out"], outputs=["seq_out"]
        )

        identity_node = onnx.helper.make_node(
            "Identity", inputs=["cond_in"], outputs=["cond_out"]
        )

        loop_body = onnx.helper.make_graph(
            [
                identity_node,
                x_const_node,
                one_const_node,
                zero_const_node,
                add_node,
                axes_node,
                end_unsqueeze_node,
                slice_node,
                insert_node,
            ],
            "loop_body",
            [iter_count, cond_in, seq_in],
            [cond_out, seq_out],
        )

        node = onnx.helper.make_node(
            "Loop",
            inputs=["trip_count", "cond", "seq_empty"],
            outputs=["seq_res"],
            body=loop_body,
        )

        trip_count = np.array(5).astype(np.int64)
        seq_empty: List[Any] = []
        seq_res = [x[: int(i)] for i in x]
        cond = np.array(1).astype(bool)
        expect(
            node,
            inputs=[trip_count, cond, seq_empty],
            outputs=[seq_res],
            name="test_loop13_seq",
            opset_imports=[onnx.helper.make_opsetid("", 13)],
            input_type_protos=[
                onnx.helper.make_tensor_type_proto(
                    onnx.TensorProto.INT64, trip_count.shape
                ),
                onnx.helper.make_tensor_type_proto(onnx.TensorProto.BOOL, cond.shape),
                onnx.helper.make_sequence_type_proto(
                    onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, [])
                ),
            ],
        )

    @staticmethod
    def export_loop_16_none() -> None:
        # Given a tensor sequence of values [x1, ..., xN], and an initial optional sequence of tensors [x0],
        # Return a concatenated sequence of tensors of
        #   [x0, [x1], [x1, x2], ..., [x1, ..., xN]]

        ten_in_tp = onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, [])
        seq_in_tp = onnx.helper.make_sequence_type_proto(ten_in_tp)
        opt_in_tp = onnx.helper.make_optional_type_proto(seq_in_tp)
        opt_in = onnx.helper.make_value_info("opt_seq_in", opt_in_tp)
        seq_out = onnx.helper.make_tensor_sequence_value_info(
            "seq_out", onnx.TensorProto.FLOAT, []
        )
        cond_in = onnx.helper.make_tensor_value_info(
            "cond_in", onnx.TensorProto.BOOL, []
        )
        cond_out = onnx.helper.make_tensor_value_info(
            "cond_out", onnx.TensorProto.BOOL, []
        )
        iter_count = onnx.helper.make_tensor_value_info(
            "iter_count", onnx.TensorProto.INT64, []
        )

        x0 = np.array(0).astype(np.float32)
        x = np.array([1, 2, 3, 4, 5]).astype(np.float32)

        optional_has_elem_node = onnx.helper.make_node(
            "OptionalHasElement", inputs=["opt_seq_in"], outputs=["optional_has_elem"]
        )

        optional_is_none = onnx.helper.make_node(
            "Not", inputs=["optional_has_elem"], outputs=["optional_is_none"]
        )

        optional_get_elem = onnx.helper.make_node(
            "OptionalGetElement", inputs=["opt_seq_in"], outputs=["seq_in"]
        )

        constant_in = onnx.helper.make_node(
            "Constant",
            inputs=[],
            outputs=["constant_in"],
            value=onnx.helper.make_tensor(
                name="const_tensor", data_type=onnx.TensorProto.FLOAT, dims=(), vals=[0]
            ),
        )

        seq_const_in = onnx.helper.make_node(
            "SequenceConstruct", inputs=["constant_in"], outputs=["init_seq_in"]
        )

        then_seq_out = onnx.helper.make_tensor_sequence_value_info(
            "init_seq_in", onnx.TensorProto.FLOAT, []
        )
        then_body = onnx.helper.make_graph(
            [constant_in, seq_const_in], "then_body", [], [then_seq_out]
        )

        else_seq_out = onnx.helper.make_tensor_sequence_value_info(
            "seq_in", onnx.TensorProto.FLOAT, []
        )
        else_body = onnx.helper.make_graph(
            [optional_get_elem], "else_body", [], [else_seq_out]
        )

        if_node = onnx.helper.make_node(
            "If",
            inputs=["optional_is_none"],
            outputs=["sequence"],
            then_branch=then_body,
            else_branch=else_body,
        )

        x_const_node = onnx.helper.make_node(
            "Constant",
            inputs=[],
            outputs=["x"],
            value=onnx.helper.make_tensor(
                name="const_tensor_x",
                data_type=onnx.TensorProto.FLOAT,
                dims=x.shape,
                vals=x.flatten().astype(float),
            ),
        )

        one_const_node = onnx.helper.make_node(
            "Constant",
            inputs=[],
            outputs=["one"],
            value=onnx.helper.make_tensor(
                name="const_tensor_one",
                data_type=onnx.TensorProto.INT64,
                dims=(),
                vals=[1],
            ),
        )

        zero_const_node = onnx.helper.make_node(
            "Constant",
            inputs=[],
            outputs=["slice_start"],
            value=onnx.helper.make_tensor(
                name="const_tensor_zero",
                data_type=onnx.TensorProto.INT64,
                dims=(1,),
                vals=[0],
            ),
        )

        axes_node = onnx.helper.make_node(
            "Constant",
            inputs=[],
            outputs=["axes"],
            value=onnx.helper.make_tensor(
                name="const_tensor_axes",
                data_type=onnx.TensorProto.INT64,
                dims=(),
                vals=[0],
            ),
        )

        add_node = onnx.helper.make_node(
            "Add", inputs=["iter_count", "one"], outputs=["end"]
        )

        end_unsqueeze_node = onnx.helper.make_node(
            "Unsqueeze", inputs=["end", "axes"], outputs=["slice_end"]
        )

        slice_node = onnx.helper.make_node(
            "Slice", inputs=["x", "slice_start", "slice_end"], outputs=["slice_out"]
        )

        insert_node = onnx.helper.make_node(
            "SequenceInsert", inputs=["sequence", "slice_out"], outputs=["seq_out"]
        )

        identity_node = onnx.helper.make_node(
            "Identity", inputs=["cond_in"], outputs=["cond_out"]
        )

        loop_body = onnx.helper.make_graph(
            [
                identity_node,
                optional_has_elem_node,
                optional_is_none,
                if_node,
                x_const_node,
                one_const_node,
                zero_const_node,
                add_node,
                axes_node,
                end_unsqueeze_node,
                slice_node,
                insert_node,
            ],
            "loop_body",
            [iter_count, cond_in, opt_in],
            [cond_out, seq_out],
        )

        node = onnx.helper.make_node(
            "Loop",
            inputs=["trip_count", "cond", "opt_seq"],
            outputs=["seq_res"],
            body=loop_body,
        )

        trip_count = np.array(5).astype(np.int64)
        cond = np.array(1).astype(bool)
        seq_res = compute_loop_outputs(x, [x0], trip_count)
        opt_seq_in: List[Any] = [x0]
        expect(
            node,
            inputs=[trip_count, cond, opt_seq_in],
            outputs=[seq_res],
            name="test_loop16_seq_none",
            opset_imports=[onnx.helper.make_opsetid("", 16)],
            input_type_protos=[
                onnx.helper.make_tensor_type_proto(
                    onnx.TensorProto.INT64, trip_count.shape
                ),
                onnx.helper.make_tensor_type_proto(onnx.TensorProto.BOOL, cond.shape),
                opt_in_tp,
            ],
        )
