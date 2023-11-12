# SPDX-License-Identifier: Apache-2.0


import numpy as np

from ..proto import onnx_proto
from ..common._apply_operation import (
    apply_cast,
    apply_concat,
    apply_reshape,
    apply_mul,
    apply_add,
)
from ..common._registration import register_converter
from ..common._topology import Scope, Operator
from ..common._container import ModelComponentContainer


def convert_sklearn_k_bins_discretiser(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    op = operator.raw_operator

    if op.encode == "onehot":
        raise RuntimeError(
            "onehot encoding not supported. "
            "ONNX does not support sparse tensors. "
            "with opset < 11. You may raise an isue at "
            "https://github.com/onnx/sklearn-onnx/issues."
        )

    ranges = list(
        map(
            lambda e: e[1:-1] if len(e) > 2 else [np.finfo(np.float32).max],
            op.bin_edges_,
        )
    )
    digitised_output_name = [None] * len(ranges)
    last_column_name = None

    for i, item in enumerate(ranges):
        digitised_output_name[i] = scope.get_unique_variable_name(
            "digitised_output_{}".format(i)
        )
        column_index_name = scope.get_unique_variable_name("column_index")
        range_column_name = scope.get_unique_variable_name("range_column")
        column_name = scope.get_unique_variable_name("column")
        cast_column_name = scope.get_unique_variable_name("cast_column")
        less_result_name = scope.get_unique_variable_name("less_result")
        cast_result_name = scope.get_unique_variable_name("cast_result")
        concatenated_array_name = scope.get_unique_variable_name("concatenated_array")
        argmax_output_name = scope.get_unique_variable_name("argmax_output")

        container.add_initializer(
            column_index_name, onnx_proto.TensorProto.INT64, [], [i]
        )
        container.add_initializer(
            range_column_name, onnx_proto.TensorProto.FLOAT, [len(item)], item
        )

        container.add_node(
            "ArrayFeatureExtractor",
            [operator.inputs[0].full_name, column_index_name],
            column_name,
            name=scope.get_unique_operator_name("ArrayFeatureExtractor"),
            op_domain="ai.onnx.ml",
        )
        apply_cast(
            scope,
            column_name,
            cast_column_name,
            container,
            to=onnx_proto.TensorProto.FLOAT,
        )
        container.add_node(
            "Less",
            [cast_column_name, range_column_name],
            less_result_name,
            name=scope.get_unique_operator_name("Less"),
        )
        apply_cast(
            scope,
            less_result_name,
            cast_result_name,
            container,
            to=onnx_proto.TensorProto.FLOAT,
        )

        if last_column_name is None:
            last_column_name = scope.get_unique_variable_name("last_column")
            zero_float = scope.get_unique_variable_name("zero_float")
            one_float = scope.get_unique_variable_name("one_float")
            zero_column = scope.get_unique_variable_name("zero_column")
            container.add_initializer(
                one_float, onnx_proto.TensorProto.FLOAT, [1], np.ones(1)
            )
            container.add_initializer(
                zero_float, onnx_proto.TensorProto.FLOAT, [1], np.zeros(1)
            )
            apply_mul(
                scope,
                [cast_column_name, zero_float],
                zero_column,
                container,
                broadcast=1,
            )
            apply_add(
                scope,
                [zero_column, one_float],
                last_column_name,
                container,
                broadcast=1,
            )

        apply_concat(
            scope,
            [cast_result_name, last_column_name],
            concatenated_array_name,
            container,
            axis=1,
        )
        container.add_node(
            "ArgMax",
            concatenated_array_name,
            argmax_output_name,
            axis=1,
            name=scope.get_unique_operator_name("ArgMax"),
        )
        if op.encode == "onehot-dense":
            onehot_result_name = scope.get_unique_variable_name("onehot_result")

            container.add_node(
                "OneHotEncoder",
                argmax_output_name,
                onehot_result_name,
                name=scope.get_unique_operator_name("OneHotEncoder"),
                cats_int64s=list(range(op.n_bins_[i])),
                op_domain="ai.onnx.ml",
            )
            apply_reshape(
                scope,
                onehot_result_name,
                digitised_output_name[i],
                container,
                desired_shape=(-1, op.n_bins_[i]),
            )
        else:
            apply_cast(
                scope,
                argmax_output_name,
                digitised_output_name[i],
                container,
                to=onnx_proto.TensorProto.FLOAT,
            )
    apply_concat(
        scope, digitised_output_name, operator.outputs[0].full_name, container, axis=1
    )


register_converter("SklearnKBinsDiscretizer", convert_sklearn_k_bins_discretiser)
