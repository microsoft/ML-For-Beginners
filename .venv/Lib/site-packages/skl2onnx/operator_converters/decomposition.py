# SPDX-License-Identifier: Apache-2.0


from ..proto import onnx_proto
from ..common._apply_operation import apply_cast, apply_div, apply_sqrt, apply_sub
from ..common._registration import register_converter
from ..common._topology import Scope, Operator
from ..common._container import ModelComponentContainer
from ..common.data_types import (
    Int64TensorType,
    DoubleTensorType,
    FloatTensorType,
    guess_proto_type,
)


def convert_truncated_svd(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    # Create alias for the scikit-learn truncated SVD model we
    # are going to convert
    svd = operator.raw_operator
    if isinstance(operator.inputs[0].type, DoubleTensorType):
        proto_dtype = guess_proto_type(operator.inputs[0].type)
    else:
        proto_dtype = guess_proto_type(FloatTensorType())
    # Transpose [K, C] matrix to [C, K], where C/K is the
    # input/transformed feature dimension
    transform_matrix = svd.components_.transpose()
    transform_matrix_name = scope.get_unique_variable_name("transform_matrix")
    # Put the transformation into an ONNX tensor
    container.add_initializer(
        transform_matrix_name,
        proto_dtype,
        transform_matrix.shape,
        transform_matrix.flatten(),
    )

    input_name = operator.inputs[0].full_name
    if isinstance(operator.inputs[0].type, Int64TensorType):
        cast_output_name = scope.get_unique_variable_name("cast_output")

        apply_cast(
            scope,
            input_name,
            cast_output_name,
            container,
            to=onnx_proto.TensorProto.FLOAT,
        )
        input_name = cast_output_name

    if operator.type == "SklearnTruncatedSVD":
        # Create the major operator, a matrix multiplication.
        container.add_node(
            "MatMul",
            [input_name, transform_matrix_name],
            operator.outputs[0].full_name,
            name=operator.full_name,
        )
    else:  # PCA
        if svd.mean_ is not None:
            mean_name = scope.get_unique_variable_name("mean")
            sub_result_name = scope.get_unique_variable_name("sub_result")

            container.add_initializer(
                mean_name, proto_dtype, svd.mean_.shape, svd.mean_
            )

            # Subtract mean from input tensor
            apply_sub(
                scope, [input_name, mean_name], sub_result_name, container, broadcast=1
            )
        else:
            sub_result_name = input_name
        if svd.whiten:
            explained_variance_name = scope.get_unique_variable_name(
                "explained_variance"
            )
            explained_variance_root_name = scope.get_unique_variable_name(
                "explained_variance_root"
            )
            matmul_result_name = scope.get_unique_variable_name("matmul_result")

            container.add_initializer(
                explained_variance_name,
                proto_dtype,
                svd.explained_variance_.shape,
                svd.explained_variance_,
            )

            container.add_node(
                "MatMul",
                [sub_result_name, transform_matrix_name],
                matmul_result_name,
                name=scope.get_unique_operator_name("MatMul"),
            )
            apply_sqrt(
                scope, explained_variance_name, explained_variance_root_name, container
            )
            apply_div(
                scope,
                [matmul_result_name, explained_variance_root_name],
                operator.outputs[0].full_name,
                container,
                broadcast=1,
            )
        else:
            container.add_node(
                "MatMul",
                [sub_result_name, transform_matrix_name],
                operator.outputs[0].full_name,
                name=scope.get_unique_operator_name("MatMul"),
            )


register_converter("SklearnIncrementalPCA", convert_truncated_svd)
register_converter("SklearnPCA", convert_truncated_svd)
register_converter("SklearnTruncatedSVD", convert_truncated_svd)
