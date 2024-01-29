/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <functional>

#include "onnx/defs/generator/utils.h"
#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {
static const char* Constant_ver13_doc = R"DOC(
This operator produces a constant tensor. Exactly one of the provided attributes, either value, sparse_value,
or value_* must be specified.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Constant,
    13,
    OpSchema()
        .SetDoc(Constant_ver13_doc)
        .Attr("value", "The value for the elements of the output tensor.", AttributeProto::TENSOR, false)
        .Attr(
            "sparse_value",
            "The value for the elements of the output tensor in sparse format.",
            AttributeProto::SPARSE_TENSOR,
            false)
        .Attr(
            "value_int",
            "The value for the sole element for the scalar, int64, output tensor.",
            AttributeProto::INT,
            false)
        .Attr(
            "value_ints",
            "The values for the elements for the 1D, int64, output tensor.",
            AttributeProto::INTS,
            false)
        .Attr(
            "value_float",
            "The value for the sole element for the scalar, float32, output tensor.",
            AttributeProto::FLOAT,
            false)
        .Attr(
            "value_floats",
            "The values for the elements for the 1D, float32, output tensor.",
            AttributeProto::FLOATS,
            false)
        .Attr(
            "value_string",
            "The value for the sole element for the scalar, UTF-8 string, output tensor.",
            AttributeProto::STRING,
            false)
        .Attr(
            "value_strings",
            "The values for the elements for the 1D, UTF-8 string, output tensor.",
            AttributeProto::STRINGS,
            false)
        .Output(0, "output", "Output tensor containing the same value of the provided tensor.", "T")
        .TypeConstraint("T", OpSchema::all_tensor_types_ir4(), "Constrain input and output types to all tensor types.")
        .TypeAndShapeInferenceFunction(ConstantOpInference));

static const char* Constant_ver12_doc = R"DOC(
This operator produces a constant tensor. Exactly one of the provided attributes, either value, sparse_value,
or value_* must be specified.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Constant,
    12,
    OpSchema()
        .SetDoc(Constant_ver12_doc)
        .Attr("value", "The value for the elements of the output tensor.", AttributeProto::TENSOR, false)
        .Attr(
            "sparse_value",
            "The value for the elements of the output tensor in sparse format.",
            AttributeProto::SPARSE_TENSOR,
            false)
        .Attr(
            "value_int",
            "The value for the sole element for the scalar, int64, output tensor.",
            AttributeProto::INT,
            false)
        .Attr(
            "value_ints",
            "The values for the elements for the 1D, int64, output tensor.",
            AttributeProto::INTS,
            false)
        .Attr(
            "value_float",
            "The value for the sole element for the scalar, float32, output tensor.",
            AttributeProto::FLOAT,
            false)
        .Attr(
            "value_floats",
            "The values for the elements for the 1D, float32, output tensor.",
            AttributeProto::FLOATS,
            false)
        .Attr(
            "value_string",
            "The value for the sole element for the scalar, UTF-8 string, output tensor.",
            AttributeProto::STRING,
            false)
        .Attr(
            "value_strings",
            "The values for the elements for the 1D, UTF-8 string, output tensor.",
            AttributeProto::STRINGS,
            false)
        .Output(0, "output", "Output tensor containing the same value of the provided tensor.", "T")
        .TypeConstraint("T", OpSchema::all_tensor_types(), "Constrain input and output types to all tensor types.")
        .TypeAndShapeInferenceFunction(ConstantOpInference));

static const char* Constant_ver1_doc = R"DOC(A constant tensor.)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Constant,
    1,
    OpSchema()
        .SetDoc(Constant_ver1_doc)
        .Attr("value", "The value for the elements of the output tensor.", AttributeProto::TENSOR)
        .Output(0, "output", "Output tensor containing the same value of the provided tensor.", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          auto attr_proto = ctx.getAttribute("value");
          if (nullptr == attr_proto)
            return; // attribute not present
          if (!attr_proto->has_t())
            return; // attribute has no tensor value
          const TensorProto& tensor_proto = attr_proto->t();
          updateOutputElemType(ctx, 0, tensor_proto.data_type());
          updateOutputShape(ctx, 0, tensor_proto);
        }));

static const char* Constant_ver9_doc = R"DOC(A constant tensor.)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Constant,
    9,
    OpSchema()
        .SetDoc(Constant_ver9_doc)
        .Attr("value", "The value for the elements of the output tensor.", AttributeProto::TENSOR)
        .Output(0, "output", "Output tensor containing the same value of the provided tensor.", "T")
        .TypeConstraint("T", OpSchema::all_tensor_types(), "Constrain input and output types to all tensor types.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          auto attr_proto = ctx.getAttribute("value");
          if (nullptr == attr_proto || !attr_proto->has_t())
            fail_shape_inference("Attribute 'value' of Constant node must exist with 'Tensor' data.");
          const TensorProto& tensor_proto = attr_proto->t();
          updateOutputElemType(ctx, 0, tensor_proto.data_type());
          updateOutputShape(ctx, 0, tensor_proto);
        }));

static const char* Constant_ver11_doc = R"DOC(
A constant tensor. Exactly one of the two attributes, either value or sparse_value,
must be specified.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Constant,
    11,
    OpSchema()
        .SetDoc(Constant_ver11_doc)
        .Attr("value", "The value for the elements of the output tensor.", AttributeProto::TENSOR, false)
        .Attr(
            "sparse_value",
            "The value for the elements of the output tensor in sparse format.",
            AttributeProto::SPARSE_TENSOR,
            false)
        .Output(0, "output", "Output tensor containing the same value of the provided tensor.", "T")
        .TypeConstraint("T", OpSchema::all_tensor_types(), "Constrain input and output types to all tensor types.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          auto* value = ctx.getAttribute("value");
          auto* sparse_value = ctx.getAttribute("sparse_value");

          if ((nullptr != value) && (nullptr != sparse_value))
            fail_shape_inference(
                "Only one of the attributes 'value' or 'sparse_value' must be specified for a Constant node.");

          if (nullptr != value) {
            // OpSchema::Verify check ensures that the attribute value has_t():
            const TensorProto& tensor_proto = value->t();
            updateOutputElemType(ctx, 0, tensor_proto.data_type());
            updateOutputShape(ctx, 0, tensor_proto);
            return;
          }

          if (nullptr != sparse_value) {
            // OpSchema::Verify check ensures that the attribute value
            // has_sparse_tensor():
            const SparseTensorProto& sparse = sparse_value->sparse_tensor();
            // checker.cc::check_sparse_tensor checks that the sparse-value is
            // well-formed
            updateOutputElemType(ctx, 0, sparse.values().data_type());
            auto* output_shape = getOutputShape(ctx, 0);
            for (int i = 0; i < sparse.dims_size(); ++i)
              appendDim(output_shape, sparse.dims(i));
            return;
          }
          fail_shape_inference(
              "One of the attributes 'value' or 'sparse_value' must be specified for a Constant node.");
        }));

static const char* ConstantOfShape_ver9_doc = R"DOC(
Generate a tensor with given value and shape.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    ConstantOfShape,
    9,
    OpSchema()
        .SetDoc(ConstantOfShape_ver9_doc)
        .Attr(
            "value",
            "(Optional) The value of the output elements."
            "Should be a one-element tensor. If not specified, it defaults to a tensor of value 0 and datatype float32",
            AttributeProto::TENSOR,
            OPTIONAL_VALUE)
        .Input(
            0,
            "input",
            "1D tensor. The shape of the expected output tensor. If empty tensor is given, the output would be a scalar."
            " All values must be >= 0.",
            "T1")
        .Output(
            0,
            "output",
            "Output tensor of shape specified by 'input'."
            "If attribute 'value' is specified, the value and datatype of the output tensor is taken from 'value'."
            "If attribute 'value' is not specified, the value in the output defaults to 0, and the datatype "
            "defaults to float32.",
            "T2")
        .TypeConstraint("T1", {"tensor(int64)"}, "Constrain input types.")
        .TypeConstraint(
            "T2",
            {"tensor(float16)",
             "tensor(float)",
             "tensor(double)",
             "tensor(int8)",
             "tensor(int16)",
             "tensor(int32)",
             "tensor(int64)",
             "tensor(uint8)",
             "tensor(uint16)",
             "tensor(uint32)",
             "tensor(uint64)",
             "tensor(bool)"},
            "Constrain output types to be numerics.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          if (ctx.getAttribute("value") != nullptr) {
            propagateElemTypeFromDtypeToOutput(ctx, ctx.getAttribute("value"), 0);
          } else {
            propagateElemTypeFromDtypeToOutput(ctx, TensorProto::FLOAT, 0);
          }

          bool found = false;
          TensorShapeProto output_shape = getShapeInput(ctx, 0, found);
          if (found) {
            *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape() = output_shape;
          }
        }));

} // namespace ONNX_NAMESPACE
