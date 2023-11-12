# SPDX-License-Identifier: Apache-2.0


from onnx.helper import make_tensor
from onnx import TensorProto
from ..proto import onnx_proto
from ..common._apply_operation import apply_concat, apply_cast
from ..common._registration import register_converter
from ..common._topology import Scope, Operator
from ..common._container import ModelComponentContainer


def convert_sklearn_polynomial_features(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    op = operator.raw_operator
    transformed_columns = [None] * (op.n_output_features_)

    n_features = op.n_features_in_ if hasattr(op, "n_features_in_") else op.n_features_
    if hasattr(op, "_min_degree"):
        # scikit-learn >= 1.0
        combinations = op._combinations(
            n_features,
            op._min_degree,
            op._max_degree,
            op.interaction_only,
            op.include_bias,
        )
    else:
        combinations = op._combinations(
            n_features, op.degree, op.interaction_only, op.include_bias
        )

    unit_name = None
    last_feat = None
    for i, comb in enumerate(combinations):
        if len(comb) == 0:
            unit_name = scope.get_unique_variable_name("unit")
            transformed_columns[i] = unit_name
        else:
            comb_name = scope.get_unique_variable_name("comb")
            col_name = scope.get_unique_variable_name("col")
            prod_name = scope.get_unique_variable_name("prod")

            container.add_initializer(
                comb_name, onnx_proto.TensorProto.INT64, [len(comb)], list(comb)
            )

            container.add_node(
                "ArrayFeatureExtractor",
                [operator.inputs[0].full_name, comb_name],
                col_name,
                name=scope.get_unique_operator_name("ArrayFeatureExtractor"),
                op_domain="ai.onnx.ml",
            )
            reduce_prod_input = col_name
            if (
                operator.inputs[0].type._get_element_onnx_type()
                == onnx_proto.TensorProto.INT64
            ):
                float_col_name = scope.get_unique_variable_name("col")

                apply_cast(
                    scope,
                    col_name,
                    float_col_name,
                    container,
                    to=onnx_proto.TensorProto.FLOAT,
                )
                reduce_prod_input = float_col_name

            if container.target_opset >= 18:
                axis_name = scope.get_unique_variable_name("axis")
                container.add_initializer(
                    axis_name, onnx_proto.TensorProto.INT64, [1], [1]
                )
                container.add_node(
                    "ReduceProd",
                    [reduce_prod_input, axis_name],
                    prod_name,
                    name=scope.get_unique_operator_name("ReduceProd"),
                )
            else:
                container.add_node(
                    "ReduceProd",
                    reduce_prod_input,
                    prod_name,
                    axes=[1],
                    name=scope.get_unique_operator_name("ReduceProd"),
                )
            transformed_columns[i] = prod_name
            last_feat = prod_name

    if unit_name is not None:
        shape_name = scope.get_unique_variable_name("shape")
        container.add_node("Shape", last_feat, shape_name)
        container.add_node(
            "ConstantOfShape",
            shape_name,
            unit_name,
            value=make_tensor("ONE", TensorProto.FLOAT, [1], [1.0]),
            op_version=9,
        )

    if operator.inputs[0].type._get_element_onnx_type() == onnx_proto.TensorProto.INT64:
        concat_result_name = scope.get_unique_variable_name("concat_result")

        apply_concat(
            scope,
            [t for t in transformed_columns],
            concat_result_name,
            container,
            axis=1,
        )
        apply_cast(
            scope,
            concat_result_name,
            operator.outputs[0].full_name,
            container,
            to=onnx_proto.TensorProto.INT64,
        )
    else:
        apply_concat(
            scope,
            [t for t in transformed_columns],
            operator.outputs[0].full_name,
            container,
            axis=1,
        )


register_converter("SklearnPolynomialFeatures", convert_sklearn_polynomial_features)
