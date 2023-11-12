# SPDX-License-Identifier: Apache-2.0


import numpy as np
from ._apply_operation import apply_cast, apply_reshape
from ..proto import onnx_proto


def get_label_classes(scope, op, node_names=False):
    """
    Extracts the model classes,
    handles option ``nocl`` and ``zipmap=='columns'``
    """
    options = scope.get_options(op, dict(nocl=False))
    if options["nocl"]:
        if len(op.classes_.shape) > 1 and op.classes_.shape[1] > 1:
            raise RuntimeError(
                "Options 'nocl=True' is not implemented for multi-label "
                "classification (class: {}).".format(op.__class__.__name__)
            )
        classes = np.arange(0, len(op.classes_))
    elif node_names:
        try:
            options = scope.get_options(op, dict(zipmap=False))
            zipcol = options["zipmap"] == "columns"
        except NameError:
            zipcol = False
        if zipcol:
            clnames = op.classes_.ravel()
            if np.issubdtype(clnames.dtype, np.integer) or clnames.dtype == np.bool_:
                classes = np.array(["i%d" % c for c in clnames])
            else:
                classes = np.array(["s%s" % c for c in clnames])
        else:
            classes = op.classes_
    elif hasattr(op, "classes_"):
        classes = op.classes_
    elif hasattr(op, "intercept_"):
        classes = len(op.intercept_)
    elif hasattr(op, "y_"):
        # _ConstantPredictor
        classes = np.array(list(sorted(set(op.y_))))
    else:
        raise RuntimeError(
            "No known ways to retrieve the number of classes for class %r."
            "" % type(op)
        )
    return classes


def _finalize_converter_classes(
    scope, argmax_output_name, output_full_name, container, classes, proto_dtype
):
    """
    See :func:`convert_voting_classifier`.
    """
    if np.issubdtype(classes.dtype, np.floating) or classes.dtype == np.bool_:
        class_type = onnx_proto.TensorProto.INT32
        classes = np.array(list(map(lambda x: int(x), classes)))
    elif np.issubdtype(classes.dtype, np.signedinteger):
        class_type = onnx_proto.TensorProto.INT32
    else:
        classes = np.array([s.encode("utf-8") for s in classes])
        class_type = onnx_proto.TensorProto.STRING

    classes_name = scope.get_unique_variable_name("classes")
    container.add_initializer(classes_name, class_type, classes.shape, classes)

    array_feature_extractor_result_name = scope.get_unique_variable_name(
        "array_feature_extractor_result"
    )
    container.add_node(
        "ArrayFeatureExtractor",
        [classes_name, argmax_output_name],
        array_feature_extractor_result_name,
        op_domain="ai.onnx.ml",
        name=scope.get_unique_operator_name("ArrayFeatureExtractor"),
    )

    output_shape = (-1,)
    if class_type == onnx_proto.TensorProto.INT32:
        cast2_result_name = scope.get_unique_variable_name("cast2_result")
        reshaped_result_name = scope.get_unique_variable_name("reshaped_result")
        apply_cast(
            scope,
            array_feature_extractor_result_name,
            cast2_result_name,
            container,
            to=proto_dtype,
        )
        apply_reshape(
            scope,
            cast2_result_name,
            reshaped_result_name,
            container,
            desired_shape=output_shape,
        )
        apply_cast(
            scope,
            reshaped_result_name,
            output_full_name,
            container,
            to=onnx_proto.TensorProto.INT64,
        )
    else:  # string labels
        apply_reshape(
            scope,
            array_feature_extractor_result_name,
            output_full_name,
            container,
            desired_shape=output_shape,
        )
