# SPDX-License-Identifier: Apache-2.0


from ..proto import onnx_proto
from ..common.data_types import DoubleTensorType
from ..common._registration import register_converter
from ..common._topology import Scope, Operator
from ..common._container import ModelComponentContainer
from .common import concatenate_variables


def convert_sklearn_binarizer(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    feature_name = concatenate_variables(scope, operator.inputs, container)

    if isinstance(operator.inputs[0].type, DoubleTensorType):
        name0 = scope.get_unique_variable_name("cst0")
        name1 = scope.get_unique_variable_name("cst1")
        thres = scope.get_unique_variable_name("th")
        container.add_initializer(name0, onnx_proto.TensorProto.DOUBLE, [], [0.0])
        container.add_initializer(name1, onnx_proto.TensorProto.DOUBLE, [], [1.0])
        container.add_initializer(
            thres,
            onnx_proto.TensorProto.DOUBLE,
            [],
            [float(operator.raw_operator.threshold)],
        )
        binbool = scope.get_unique_variable_name("binbool")
        container.add_node(
            "Less",
            [feature_name, thres],
            binbool,
            name=scope.get_unique_operator_name("Less"),
        )
        container.add_node(
            "Where", [binbool, name0, name1], operator.output_full_names, name="Where"
        )
        return

    op_type = "Binarizer"
    attrs = {
        "name": scope.get_unique_operator_name(op_type),
        "threshold": float(operator.raw_operator.threshold),
    }
    container.add_node(
        op_type,
        feature_name,
        operator.output_full_names,
        op_domain="ai.onnx.ml",
        **attrs
    )


register_converter("SklearnBinarizer", convert_sklearn_binarizer)
