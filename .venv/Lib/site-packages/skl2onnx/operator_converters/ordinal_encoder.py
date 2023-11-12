# SPDX-License-Identifier: Apache-2.0


import numpy as np
from ..common._apply_operation import apply_cast, apply_concat, apply_reshape
from ..common.data_types import Int64TensorType, StringTensorType
from ..common._registration import register_converter
from ..common._topology import Scope, Operator
from ..common._container import ModelComponentContainer
from ..proto import onnx_proto


def convert_sklearn_ordinal_encoder(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    ordinal_op = operator.raw_operator
    result = []
    concatenated_input_name = operator.inputs[0].full_name
    concat_result_name = scope.get_unique_variable_name("concat_result")

    if len(operator.inputs) > 1:
        concatenated_input_name = scope.get_unique_variable_name("concatenated_input")
        if all(
            isinstance(inp.type, type(operator.inputs[0].type))
            for inp in operator.inputs
        ):
            input_names = list(map(lambda x: x.full_name, operator.inputs))
        else:
            input_names = []
            for inp in operator.inputs:
                if isinstance(inp.type, Int64TensorType):
                    input_names.append(scope.get_unique_variable_name("cast_input"))
                    apply_cast(
                        scope,
                        inp.full_name,
                        input_names[-1],
                        container,
                        to=onnx_proto.TensorProto.STRING,
                    )
                elif isinstance(inp.type, StringTensorType):
                    input_names.append(inp.full_name)
                else:
                    raise NotImplementedError(
                        "{} input datatype not yet supported. "
                        "You may raise an issue at "
                        "https://github.com/onnx/sklearn-onnx/issues"
                        "".format(type(inp.type))
                    )

        apply_concat(scope, input_names, concatenated_input_name, container, axis=1)
    if len(ordinal_op.categories_) == 0:
        raise RuntimeError(
            "No categories found in type=%r, encoder=%r."
            % (type(ordinal_op), ordinal_op)
        )
    for index, categories in enumerate(ordinal_op.categories_):
        attrs = {"name": scope.get_unique_operator_name("LabelEncoder")}
        if len(categories) > 0:
            if (
                np.issubdtype(categories.dtype, np.floating)
                or categories.dtype == np.bool_
            ):
                attrs["keys_floats"] = categories
            elif np.issubdtype(categories.dtype, np.signedinteger):
                attrs["keys_int64s"] = categories
            else:
                attrs["keys_strings"] = np.array(
                    [str(s).encode("utf-8") for s in categories]
                )
            attrs["values_int64s"] = np.arange(len(categories)).astype(np.int64)

            index_name = scope.get_unique_variable_name("index")
            feature_column_name = scope.get_unique_variable_name("feature_column")
            result.append(scope.get_unique_variable_name("ordinal_output"))
            label_encoder_output = scope.get_unique_variable_name("label_encoder")

            container.add_initializer(
                index_name, onnx_proto.TensorProto.INT64, [], [index]
            )

            container.add_node(
                "ArrayFeatureExtractor",
                [concatenated_input_name, index_name],
                feature_column_name,
                op_domain="ai.onnx.ml",
                name=scope.get_unique_operator_name("ArrayFeatureExtractor"),
            )

            container.add_node(
                "LabelEncoder",
                feature_column_name,
                label_encoder_output,
                op_domain="ai.onnx.ml",
                op_version=2,
                **attrs
            )
            apply_reshape(
                scope,
                label_encoder_output,
                result[-1],
                container,
                desired_shape=(-1, 1),
            )
    apply_concat(scope, result, concat_result_name, container, axis=1)
    cast_type = (
        onnx_proto.TensorProto.FLOAT
        if np.issubdtype(ordinal_op.dtype, np.floating)
        else onnx_proto.TensorProto.INT64
    )
    apply_cast(
        scope, concat_result_name, operator.output_full_names, container, to=cast_type
    )


register_converter("SklearnOrdinalEncoder", convert_sklearn_ordinal_encoder)
