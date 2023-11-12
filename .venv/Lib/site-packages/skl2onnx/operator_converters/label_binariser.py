# SPDX-License-Identifier: Apache-2.0


import numpy as np
from ..proto import onnx_proto
from ..common._apply_operation import apply_cast, apply_reshape
from ..common._registration import register_converter
from ..common._topology import Scope, Operator
from ..common._container import ModelComponentContainer


def convert_sklearn_label_binariser(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    """Converts Scikit Label Binariser model to onnx format."""
    binariser_op = operator.raw_operator
    classes = binariser_op.classes_
    if hasattr(binariser_op, "sparse_input_") and binariser_op.sparse_input_:
        raise RuntimeError("sparse is not supported for LabelBinarizer.")
    if (
        hasattr(binariser_op, "y_type_")
        and binariser_op.y_type_ == "multilabel-indicator"
    ):
        if binariser_op.pos_label != 1:
            raise RuntimeError("pos_label != 1 is not supported " "for LabelBinarizer.")
        if list(classes) != list(range(len(classes))):
            raise RuntimeError(
                "classes != [0, 1, ..., n_classes] is not "
                "supported for LabelBinarizer."
            )
        container.add_node(
            "Identity",
            operator.inputs[0].full_name,
            operator.output_full_names,
            name=scope.get_unique_operator_name("identity"),
        )
    else:
        zeros_tensor = np.full(
            (1, len(classes)), binariser_op.neg_label, dtype=np.float32
        )
        unit_tensor = np.full(
            (1, len(classes)), binariser_op.pos_label, dtype=np.float32
        )

        classes_tensor_name = scope.get_unique_variable_name("classes_tensor")
        equal_condition_tensor_name = scope.get_unique_variable_name(
            "equal_condition_tensor"
        )
        zeros_tensor_name = scope.get_unique_variable_name("zero_tensor")
        unit_tensor_name = scope.get_unique_variable_name("unit_tensor")
        where_result_name = scope.get_unique_variable_name("where_result")

        class_dtype = onnx_proto.TensorProto.STRING
        if (
            np.issubdtype(binariser_op.classes_.dtype, np.signedinteger)
            or binariser_op.classes_.dtype == np.bool_
        ):
            class_dtype = onnx_proto.TensorProto.INT64
        else:
            classes = np.array([s.encode("utf-8") for s in classes])

        container.add_initializer(
            classes_tensor_name, class_dtype, [len(classes)], classes
        )
        container.add_initializer(
            zeros_tensor_name,
            onnx_proto.TensorProto.FLOAT,
            zeros_tensor.shape,
            zeros_tensor.ravel(),
        )
        container.add_initializer(
            unit_tensor_name,
            onnx_proto.TensorProto.FLOAT,
            unit_tensor.shape,
            unit_tensor.ravel(),
        )

        reshaped_input_name = scope.get_unique_variable_name("reshaped_input")
        apply_reshape(
            scope,
            operator.inputs[0].full_name,
            reshaped_input_name,
            container,
            desired_shape=[-1, 1],
        )

        # Models with classes_/inputs of string type would fail in the
        # following step as Equal op does not support string comparison.
        container.add_node(
            "Equal",
            [classes_tensor_name, reshaped_input_name],
            equal_condition_tensor_name,
            name=scope.get_unique_operator_name("equal"),
        )
        container.add_node(
            "Where",
            [equal_condition_tensor_name, unit_tensor_name, zeros_tensor_name],
            where_result_name,
            name=scope.get_unique_operator_name("where"),
        )
        where_res = where_result_name

        if len(binariser_op.classes_) == 2:
            array_f_name = scope.get_unique_variable_name(
                "array_feature_extractor_result"
            )
            pos_class_index_name = scope.get_unique_variable_name("pos_class_index")

            container.add_initializer(
                pos_class_index_name, onnx_proto.TensorProto.INT64, [], [1]
            )

            container.add_node(
                "ArrayFeatureExtractor",
                [where_result_name, pos_class_index_name],
                array_f_name,
                op_domain="ai.onnx.ml",
                name=scope.get_unique_operator_name("ArrayFeatureExtractor"),
            )
            where_res = array_f_name
        apply_cast(
            scope,
            where_res,
            operator.output_full_names,
            container,
            to=onnx_proto.TensorProto.INT64,
        )


register_converter("SklearnLabelBinarizer", convert_sklearn_label_binariser)
