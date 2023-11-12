# SPDX-License-Identifier: Apache-2.0
import numpy as np
from ..proto import onnx_proto
from ..common._registration import register_converter
from ..common._topology import Scope, Operator
from ..common._container import ModelComponentContainer


def convert_sklearn_class_labels(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    if getattr(operator, "is_multi_output", False):
        classes = operator.classes
        if not isinstance(classes, list):
            raise RuntimeError(
                "classes must be a list of numpy arrays but is %r." "" % type(classes)
            )
        names = []
        if classes[0].dtype in (np.int64, np.int32):
            for i, cl in enumerate(classes):
                cla = np.array(cl)
                name = scope.get_unique_variable_name(
                    operator.outputs[0].full_name + "_cst_%d" % i
                )
                container.add_initializer(
                    name, onnx_proto.TensorProto.INT64, list(cla.shape), cla.tolist()
                )
                names.append(name)
        else:
            for i, cl in enumerate(classes):
                name = scope.get_unique_variable_name(
                    operator.outputs[0].full_name + "_cst_%d" % i
                )
                clids = np.arange(len(cl), dtype=np.int64)
                container.add_initializer(
                    name,
                    onnx_proto.TensorProto.INT64,
                    list(clids.shape),
                    clids.tolist(),
                )
                namele = scope.get_unique_variable_name(
                    operator.outputs[0].full_name + "_le_%d" % i
                )
                container.add_node(
                    "LabelEncoder",
                    name,
                    namele,
                    op_domain="ai.onnx.ml",
                    op_version=2,
                    default_string="0",
                    keys_int64s=clids,
                    values_strings=cl.tolist(),
                    name=scope.get_unique_operator_name("class_labels_le_%d" % i),
                )
                names.append(namele)

        container.add_node(
            "SequenceConstruct",
            names,
            operator.outputs[0].full_name,
            name=scope.get_unique_operator_name("class_labels_seq"),
        )
    else:
        classes = np.array(operator.classes)
        name = scope.get_unique_variable_name(operator.outputs[0].full_name + "_cst")

        if classes.dtype in (np.int64, np.int32):
            container.add_initializer(
                name,
                onnx_proto.TensorProto.INT64,
                list(classes.shape),
                classes.tolist(),
            )
        else:
            clids = np.arange(len(classes), dtype=np.int64)
            container.add_initializer(
                name, onnx_proto.TensorProto.INT64, list(clids.shape), clids.tolist()
            )
            namele = scope.get_unique_variable_name(
                operator.outputs[0].full_name + "_le"
            )
            container.add_node(
                "LabelEncoder",
                name,
                namele,
                op_domain="ai.onnx.ml",
                op_version=2,
                default_string="0",
                keys_int64s=clids,
                values_strings=classes.tolist(),
                name=scope.get_unique_operator_name("class_labels_le"),
            )
            name = namele

        container.add_node("Identity", name, operator.outputs[0].full_name)


register_converter("SklearnClassLabels", convert_sklearn_class_labels)
