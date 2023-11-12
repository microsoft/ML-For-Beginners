/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "onnx/defs/schema.h"

using namespace ONNX_NAMESPACE;

namespace ONNX_NAMESPACE {

std::function<void(OpSchema&)> BinaryLogicDocGenerator_opset12(const char* name) {
  return [=](OpSchema& schema) {
    std::string doc;
    POPULATE_OP_DOC_STR(doc = R"DOC(
Returns the tensor resulted from performing the `{name}` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

{broadcast_doc}
)DOC";
                        ReplaceAll(doc, "{name}", name);
                        ReplaceAll(doc, "{broadcast_doc}", GenerateBroadcastingDocMul().c_str()););
    schema.SetDoc(doc);
    schema.Input(0, "A", "First input operand for the logical operator.", "T");
    schema.Input(1, "B", "Second input operand for the logical operator.", "T");
    schema.Output(0, "C", "Result tensor.", "T1");
    schema.TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      // Type inference
      updateOutputElemType(ctx, 0, TensorProto::BOOL);
      // Shape inference
      if (hasNInputShapes(ctx, 2))
        bidirectionalBroadcastShapeInference(
            ctx.getInputType(0)->tensor_type().shape(),
            ctx.getInputType(1)->tensor_type().shape(),
            *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
    });
  };
}

ONNX_OPERATOR_SET_SCHEMA(
    Greater,
    9,
    OpSchema()
        .FillUsing(BinaryLogicDocGenerator_opset12("greater"))
        .TypeConstraint("T", OpSchema::all_numeric_types(), "Constrain input types to all numeric tensors.")
        .TypeConstraint("T1", {"tensor(bool)"}, "Constrain output to boolean tensor."));

ONNX_OPERATOR_SET_SCHEMA(
    Less,
    9,
    OpSchema()
        .FillUsing(BinaryLogicDocGenerator_opset12("less"))
        .TypeConstraint("T", OpSchema::all_numeric_types(), "Constrain input types to all numeric tensors.")
        .TypeConstraint("T1", {"tensor(bool)"}, "Constrain output to boolean tensor."));

ONNX_OPERATOR_SET_SCHEMA(
    Equal,
    11,
    OpSchema()
        .FillUsing(BinaryLogicDocGenerator_opset12("equal"))
        .TypeConstraint(
            "T",
            {"tensor(bool)",
             "tensor(uint8)",
             "tensor(uint16)",
             "tensor(uint32)",
             "tensor(uint64)",
             "tensor(int8)",
             "tensor(int16)",
             "tensor(int32)",
             "tensor(int64)",
             "tensor(float16)",
             "tensor(float)",
             "tensor(double)"},
            "Constrain input types to all numeric tensors.")
        .TypeConstraint("T1", {"tensor(bool)"}, "Constrain output to boolean tensor."));

inline void logicalOpInference_opset1(InferenceContext& ctx) {
  updateOutputElemType(ctx, 0, TensorProto::BOOL);
  if (hasInputShape(ctx, 0)) {
    propagateShapeFromInputToOutput(ctx, 0, 0);
  }
}

std::function<void(OpSchema&)> BinaryLogicDocGenerator_opset1(const char* name) {
  return [=](OpSchema& schema) {
    std::string doc;
    POPULATE_OP_DOC_STR(doc = R"DOC(
Returns the tensor resulted from performing the `{name}` logical operation
elementwise on the input tensors `A` and `B`.

If broadcasting is enabled, the right-hand-side argument will be broadcasted
to match the shape of left-hand-side argument. See the doc of `Add` for a
detailed description of the broadcasting rules.
)DOC";
                        ReplaceAll(doc, "{name}", name););
    schema.SetDoc(doc);
    schema.Attr("broadcast", "Enable broadcasting", AttributeProto::INT, static_cast<int64_t>(0));
    schema.Attr("axis", "If set, defines the broadcast dimensions.", AttributeProto::INT, OPTIONAL_VALUE);
    schema.Input(0, "A", "Left input tensor for the logical operator.", "T");
    schema.Input(1, "B", "Right input tensor for the logical operator.", "T");
    schema.Output(0, "C", "Result tensor.", "T1");
    schema.TypeAndShapeInferenceFunction(logicalOpInference_opset1);
  };
}

std::function<void(OpSchema&)> BinaryLogicDocGenerator_opset7(const char* name) {
  return [=](OpSchema& schema) {
    std::string doc;
    POPULATE_OP_DOC_STR(doc = R"DOC(
Returns the tensor resulted from performing the `{name}` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

{broadcast_doc}
)DOC";
                        ReplaceAll(doc, "{name}", name);
                        ReplaceAll(doc, "{broadcast_doc}", GenerateBroadcastingDocMul().c_str()););
    schema.SetDoc(doc);
    schema.Input(0, "A", "First input operand for the logical operator.", "T");
    schema.Input(1, "B", "Second input operand for the logical operator.", "T");
    schema.Output(0, "C", "Result tensor.", "T1");
    schema.TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      updateOutputElemType(ctx, 0, TensorProto::BOOL);
      if (hasNInputShapes(ctx, 2))
        bidirectionalBroadcastShapeInference(
            ctx.getInputType(0)->tensor_type().shape(),
            ctx.getInputType(1)->tensor_type().shape(),
            *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
    });
  };
}

ONNX_OPERATOR_SET_SCHEMA(
    And,
    1,
    OpSchema()
        .FillUsing(BinaryLogicDocGenerator_opset1("and"))
        .TypeConstraint("T", {"tensor(bool)"}, "Constrain input to boolean tensor.")
        .TypeConstraint("T1", {"tensor(bool)"}, "Constrain output to boolean tensor."));

ONNX_OPERATOR_SET_SCHEMA(
    Or,
    1,
    OpSchema()
        .FillUsing(BinaryLogicDocGenerator_opset1("or"))
        .TypeConstraint("T", {"tensor(bool)"}, "Constrain input to boolean tensor.")
        .TypeConstraint("T1", {"tensor(bool)"}, "Constrain output to boolean tensor."));

ONNX_OPERATOR_SET_SCHEMA(
    Xor,
    1,
    OpSchema()
        .FillUsing(BinaryLogicDocGenerator_opset1("xor"))
        .TypeConstraint("T", {"tensor(bool)"}, "Constrain input to boolean tensor.")
        .TypeConstraint("T1", {"tensor(bool)"}, "Constrain output to boolean tensor."));

ONNX_OPERATOR_SET_SCHEMA(
    Greater,
    1,
    OpSchema()
        .FillUsing(BinaryLogicDocGenerator_opset1("greater"))
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input to float tensors.")
        .TypeConstraint("T1", {"tensor(bool)"}, "Constrain output to boolean tensor."));

ONNX_OPERATOR_SET_SCHEMA(
    Less,
    1,
    OpSchema()
        .FillUsing(BinaryLogicDocGenerator_opset1("less"))
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input to float tensors.")
        .TypeConstraint("T1", {"tensor(bool)"}, "Constrain output to boolean tensor."));

ONNX_OPERATOR_SET_SCHEMA(
    Equal,
    1,
    OpSchema()
        .FillUsing(BinaryLogicDocGenerator_opset1("equal"))
        .TypeConstraint("T", {"tensor(bool)", "tensor(int32)", "tensor(int64)"}, "Constrain input to integral tensors.")
        .TypeConstraint("T1", {"tensor(bool)"}, "Constrain output to boolean tensor."));

ONNX_OPERATOR_SET_SCHEMA(
    Equal,
    7,
    OpSchema()
        .FillUsing(BinaryLogicDocGenerator_opset7("equal"))
        .TypeConstraint("T", {"tensor(bool)", "tensor(int32)", "tensor(int64)"}, "Constrain input to integral tensors.")
        .TypeConstraint("T1", {"tensor(bool)"}, "Constrain output to boolean tensor."));

ONNX_OPERATOR_SET_SCHEMA(
    Greater,
    7,
    OpSchema()
        .FillUsing(BinaryLogicDocGenerator_opset7("greater"))
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input to float tensors.")
        .TypeConstraint("T1", {"tensor(bool)"}, "Constrain output to boolean tensor."));

ONNX_OPERATOR_SET_SCHEMA(
    Less,
    7,
    OpSchema()
        .FillUsing(BinaryLogicDocGenerator_opset7("less"))
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input to float tensors.")
        .TypeConstraint("T1", {"tensor(bool)"}, "Constrain output to boolean tensor."));

// Shares same doc generator as newer opset 16 version.
extern std::function<void(OpSchema&)> BinaryLogicDocGenerator(const char* name);

ONNX_OPERATOR_SET_SCHEMA(
    LessOrEqual,
    12,
    OpSchema()
        .FillUsing(BinaryLogicDocGenerator("less_equal"))
        .TypeConstraint("T", OpSchema::all_numeric_types(), "Constrain input types to all numeric tensors.")
        .TypeConstraint("T1", {"tensor(bool)"}, "Constrain output to boolean tensor.")
        .TypeAndShapeInferenceFunction(InferenceFunction())
        .FunctionBody(R"ONNX(
        {
            O1 = Less (A, B)
            O2 = Equal (A, B)
            C = Or (O1, O2)
        }
        )ONNX"));

ONNX_OPERATOR_SET_SCHEMA(
    GreaterOrEqual,
    12,
    OpSchema()
        .FillUsing(BinaryLogicDocGenerator("greater_equal"))
        .TypeConstraint("T", OpSchema::all_numeric_types(), "Constrain input types to all numeric tensors.")
        .TypeConstraint("T1", {"tensor(bool)"}, "Constrain output to boolean tensor.")
        .TypeAndShapeInferenceFunction(InferenceFunction())
        .FunctionBody(R"ONNX(
        {
            O1 = Greater (A, B)
            O2 = Equal (A, B)
            C = Or (O1, O2)
        }
        )ONNX"));

ONNX_OPERATOR_SET_SCHEMA(
    Equal,
    13,
    OpSchema()
        .FillUsing(BinaryLogicDocGenerator("equal"))
        .TypeConstraint(
            "T",
            {"tensor(bool)",
             "tensor(uint8)",
             "tensor(uint16)",
             "tensor(uint32)",
             "tensor(uint64)",
             "tensor(int8)",
             "tensor(int16)",
             "tensor(int32)",
             "tensor(int64)",
             "tensor(float16)",
             "tensor(float)",
             "tensor(double)",
             "tensor(bfloat16)"},
            "Constrain input types to all numeric tensors.")
        .TypeConstraint("T1", {"tensor(bool)"}, "Constrain output to boolean tensor."));

} // namespace ONNX_NAMESPACE
