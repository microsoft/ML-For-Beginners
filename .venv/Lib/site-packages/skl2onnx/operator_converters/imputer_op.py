# SPDX-License-Identifier: Apache-2.0


import numpy as np
from ..proto import onnx_proto
from ..common._registration import register_converter
from ..common.data_types import Int64TensorType, StringTensorType
from ..common._topology import Scope, Operator
from ..common._container import ModelComponentContainer
from ..common._apply_operation import apply_concat
from .common import concatenate_variables


def convert_sklearn_imputer(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    op_type = "Imputer"
    attrs = {"name": scope.get_unique_operator_name(op_type)}
    op = operator.raw_operator
    if (
        hasattr(op, "fill_value")
        and isinstance(op.fill_value, str)
        and op.fill_value.lower() != "nan"
    ):
        raise RuntimeError(
            "Imputer cannot fill missing values with a " "string '%s'." % op.fill_value
        )
    if not hasattr(op, "statistics_"):
        raise RuntimeError(
            "Member statistics_ is not present, was the " "model fitted?"
        )

    if isinstance(operator.inputs[0].type, StringTensorType):
        if not isinstance(op.missing_values, (str, np.str_)):
            raise NotImplementedError(
                "The converter is implemented when the missing values "
                "are string not %r." % type(op.missing_values)
            )

        zero = scope.get_unique_variable_name("zero")
        container.add_initializer(zero, onnx_proto.TensorProto.INT64, [1], [0])

        concatenated_feature = concatenate_variables(scope, operator.inputs, container)
        names = []
        for i in range(op.statistics_.size):
            # loop on features
            fill_value = scope.get_unique_variable_name("fillvalue")
            if op.fill_value is None:
                skl_fill_value = op.statistics_[i]
            else:
                skl_fill_value = op.fill_value
            container.add_node(
                "LabelEncoder",
                [zero],
                [fill_value],
                keys_int64s=[0],
                values_strings=[op.statistics_[i]],
                default_string=skl_fill_value,
                op_domain="ai.onnx.ml",
                op_version=2,
            )
            init = scope.get_unique_variable_name("i%d" % i)
            container.add_initializer(init, onnx_proto.TensorProto.INT64, [1], [i])
            name = scope.get_unique_variable_name("impi%d" % i)
            container.add_node(
                "ArrayFeatureExtractor",
                [concatenated_feature, init],
                [name],
                op_domain="ai.onnx.ml",
            )
            cond = scope.get_unique_variable_name("impc%d" % i)
            container.add_node(
                "LabelEncoder",
                [name],
                [cond],
                keys_strings=[str(op.missing_values)],
                values_int64s=[1],
                default_int64=0,
                op_domain="ai.onnx.ml",
                op_version=2,
            )
            condb = scope.get_unique_variable_name("impc%d" % i)
            container.add_node("Cast", [cond], [condb], to=onnx_proto.TensorProto.BOOL)

            repli = scope.get_unique_variable_name("nomiss%d" % i)
            container.add_node("Where", [condb, fill_value, name], [repli])
            names.append(repli)

        apply_concat(scope, names, operator.outputs[0].full_name, container, axis=1)
    else:
        if isinstance(operator.inputs[0].type, Int64TensorType):
            attrs["imputed_value_int64s"] = op.statistics_.astype(np.int64)
            use_int = True
            delta = np.max(np.abs(attrs["imputed_value_int64s"] - op.statistics_))
            if delta != 0:
                raise RuntimeError(
                    "SimpleImputer takes integer as input but nan values are "
                    "replaced by float {} != {}.".format(
                        attrs["imputed_value_int64s"], op.statistics_
                    )
                )
        else:
            attrs["imputed_value_floats"] = op.statistics_.astype(np.float32)
            use_int = False

        if isinstance(op.missing_values, str) and op.missing_values == "NaN":
            attrs["replaced_value_float"] = np.NaN
        elif isinstance(op.missing_values, float):
            if use_int:
                ar = np.array([op.missing_values]).astype(np.int64)
                attrs["replaced_value_int64"] = ar[0]
            else:
                attrs["replaced_value_float"] = float(op.missing_values)
        else:
            raise RuntimeError(
                "Unsupported proposed value '{0}'. You may raise an issue at "
                "https://github.com/onnx/sklearn-onnx/issues."
                "".format(op.missing_values)
            )

        concatenated_feature = concatenate_variables(scope, operator.inputs, container)
        container.add_node(
            op_type,
            concatenated_feature,
            operator.outputs[0].full_name,
            op_domain="ai.onnx.ml",
            **attrs
        )


register_converter("SklearnImputer", convert_sklearn_imputer)
register_converter("SklearnSimpleImputer", convert_sklearn_imputer)
