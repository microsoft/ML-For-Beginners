# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

__all__ = [
    "C",
    "ONNX_DOMAIN",
    "ONNX_ML_DOMAIN",
    "AI_ONNX_PREVIEW_TRAINING_DOMAIN",
    "has",
    "get_schema",
    "get_all_schemas",
    "get_all_schemas_with_history",
    "onnx_opset_version",
    "get_function_ops",
    "OpSchema",
    "SchemaError",
]

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
    func_proto.ParseFromString(self._function_body)
    return func_proto


OpSchema = C.OpSchema  # type: ignore
OpSchema.function_body = _function_proto  # type: ignore


@property  # type: ignore
def _attribute_default_value(self):  # type: ignore
    attr = AttributeProto()
    attr.ParseFromString(self._default_value)
    return attr


OpSchema.Attribute.default_value = _attribute_default_value  # type: ignore


def _op_schema_repr(self) -> str:
    return f"""\
OpSchema(
    name={self.name!r},
    domain={self.domain!r},
    since_version={self.since_version!r},
    doc={self.doc!r},
    type_constraints={self.type_constraints!r},
    inputs={self.inputs!r},
    outputs={self.outputs!r},
    attributes={self.attributes!r}
)"""


OpSchema.__repr__ = _op_schema_repr  # type: ignore


def _op_schema_formal_parameter_repr(self) -> str:
    return (
        f"OpSchema.FormalParameter(name={self.name!r}, type_str={self.type_str!r}, "
        f"description={self.description!r}, param_option={self.option!r}, "
        f"is_homogeneous={self.is_homogeneous!r}, min_arity={self.min_arity!r}, "
        f"differentiation_category={self.differentiation_category!r})"
    )


OpSchema.FormalParameter.__repr__ = _op_schema_formal_parameter_repr  # type: ignore


def _op_schema_type_constraint_param_repr(self) -> str:
    return (
        f"OpSchema.TypeConstraintParam(type_param_str={self.type_param_str!r}, "
        f"allowed_type_strs={self.allowed_type_strs!r}, description={self.description!r})"
    )


OpSchema.TypeConstraintParam.__repr__ = _op_schema_type_constraint_param_repr  # type: ignore


def _op_schema_attribute_repr(self) -> str:
    return (
        f"OpSchema.Attribute(name={self.name!r}, type={self.type!r}, description={self.description!r}, "
        f"default_value={self.default_value!r}, required={self.required!r})"
    )


OpSchema.Attribute.__repr__ = _op_schema_attribute_repr  # type: ignore


def get_function_ops() -> List[OpSchema]:
    """
    Return operators defined as functions.
    """

    schemas = C.get_all_schemas()
    return [schema for schema in schemas if schema.has_function or schema.has_context_dependent_function]  # type: ignore


SchemaError = C.SchemaError
