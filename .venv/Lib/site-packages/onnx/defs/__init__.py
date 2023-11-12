# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import List

import onnx.onnx_cpp2py_export.defs as C  # noqa: N812
from onnx import AttributeProto, FunctionProto

ONNX_DOMAIN = ""
ONNX_ML_DOMAIN = "ai.onnx.ml"
AI_ONNX_PREVIEW_TRAINING_DOMAIN = "ai.onnx.preview.training"


has = C.has_schema
get_schema = C.get_schema
get_all_schemas = C.get_all_schemas
get_all_schemas_with_history = C.get_all_schemas_with_history


def onnx_opset_version() -> int:
    """
    Return current opset for domain `ai.onnx`.
    """

    return C.schema_version_map()[ONNX_DOMAIN][1]


@property  # type: ignore
def _function_proto(self):  # type: ignore
    func_proto = FunctionProto()
    func_proto.ParseFromString(self._function_body)  # pylint: disable=protected-access
    return func_proto


OpSchema = C.OpSchema  # type: ignore
C.OpSchema.function_body = _function_proto  # type: ignore


@property  # type: ignore
def _attribute_default_value(self):  # type: ignore
    attr = AttributeProto()
    attr.ParseFromString(self._default_value)  # pylint: disable=protected-access
    return attr


OpSchema.Attribute.default_value = _attribute_default_value  # type: ignore


def get_function_ops() -> List[OpSchema]:
    """
    Return operators defined as functions.
    """

    schemas = C.get_all_schemas()
    return [schema for schema in schemas if schema.has_function or schema.has_context_dependent_function]  # type: ignore


SchemaError = C.SchemaError
