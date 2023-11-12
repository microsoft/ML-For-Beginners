# SPDX-License-Identifier: Apache-2.0

from ..proto import onnx_proto
from ..common._registration import register_converter
from ..common._topology import Scope, Operator
from ..common._container import ModelComponentContainer


def convert_sklearn_sequence_at(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    i_index = operator.index
    index_name = scope.get_unique_variable_name("seq_at%d" % i_index)
    container.add_initializer(index_name, onnx_proto.TensorProto.INT64, [], [i_index])
    container.add_node(
        "SequenceAt",
        [operator.inputs[0].full_name, index_name],
        operator.outputs[0].full_name,
        name=scope.get_unique_operator_name("SequenceAt%d" % i_index),
    )


def convert_sklearn_sequence_construct(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    container.add_node(
        "SequenceConstruct",
        [i.full_name for i in operator.inputs],
        operator.outputs[0].full_name,
        name=scope.get_unique_operator_name("SequenceConstruct"),
    )


register_converter("SklearnSequenceAt", convert_sklearn_sequence_at)
register_converter("SklearnSequenceConstruct", convert_sklearn_sequence_construct)
