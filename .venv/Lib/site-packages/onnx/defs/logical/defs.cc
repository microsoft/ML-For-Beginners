/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "onnx/defs/function.h"
#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {

inline void unaryLogicalOpInference(InferenceContext& ctx) {
  // Type inference
  updateOutputElemType(ctx, 0, TensorProto::BOOL);
  // Shape inference
  if (hasInputShape(ctx, 0)) {
    propagateShapeFromInputToOutput(ctx, 0, 0);
  }
}

std::function<void(OpSchema&)> BinaryLogicDocGenerator(const char* name) {
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
    schema.Input(
        0,
        "A",
        "First input operand for the logical operator.",
        "T",
        OpSchema::Single,
        true,
        1,
        OpSchema::NonDifferentiable);
    schema.Input(
        1,
        "B",
        "Second input operand for the logical operator.",
        "T",
        OpSchema::Single,
        true,
        1,
        OpSchema::NonDifferentiable);
    schema.Output(0, "C", "Result tensor.", "T1", OpSchema::Single, true, 1, OpSchema::NonDifferentiable);
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
    And,
    7,
    OpSchema()
        .FillUsing(BinaryLogicDocGenerator("and"))
        .TypeConstraint("T", {"tensor(bool)"}, "Constrain input to boolean tensor.")
        .TypeConstraint("T1", {"tensor(bool)"}, "Constrain output to boolean tensor."));

ONNX_OPERATOR_SET_SCHEMA(
    Or,
    7,
    OpSchema()
        .FillUsing(BinaryLogicDocGenerator("or"))
        .TypeConstraint("T", {"tensor(bool)"}, "Constrain input to boolean tensor.")
        .TypeConstraint("T1", {"tensor(bool)"}, "Constrain output to boolean tensor."));

ONNX_OPERATOR_SET_SCHEMA(
    Xor,
    7,
    OpSchema()
        .FillUsing(BinaryLogicDocGenerator("xor"))
        .TypeConstraint("T", {"tensor(bool)"}, "Constrain input to boolean tensor.")
        .TypeConstraint("T1", {"tensor(bool)"}, "Constrain output to boolean tensor."));

ONNX_OPERATOR_SET_SCHEMA(
    Greater,
    13,
    OpSchema()
        .FillUsing(BinaryLogicDocGenerator("greater"))
        .TypeConstraint("T", OpSchema::all_numeric_types_ir4(), "Constrain input types to all numeric tensors.")
        .TypeConstraint("T1", {"tensor(bool)"}, "Constrain output to boolean tensor."));

ONNX_OPERATOR_SET_SCHEMA(
    Less,
    13,
    OpSchema()
        .FillUsing(BinaryLogicDocGenerator("less"))
        .TypeConstraint("T", OpSchema::all_numeric_types_ir4(), "Constrain input types to all numeric tensors.")
        .TypeConstraint("T1", {"tensor(bool)"}, "Constrain output to boolean tensor."));

ONNX_OPERATOR_SET_SCHEMA(
    Equal,
    19,
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
             "tensor(bfloat16)",
             "tensor(string)"},
            "Constrain input types to all (non-complex) tensors.")
        .TypeConstraint("T1", {"tensor(bool)"}, "Constrain output to boolean tensor."));

static const char* Not_ver1_doc = R"DOC(
Returns the negation of the input tensor element-wise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Not,
    1,
    OpSchema()
        .SetDoc(Not_ver1_doc)
        .Input(0, "X", "Input tensor", "T", OpSchema::Single, true, 1, OpSchema::NonDifferentiable)
        .Output(0, "Y", "Output tensor", "T", OpSchema::Single, true, 1, OpSchema::NonDifferentiable)
        .TypeConstraint("T", {"tensor(bool)"}, "Constrain input/output to boolean tensors.")
        .TypeAndShapeInferenceFunction(unaryLogicalOpInference));

static const char* BitShift_ver11_doc = R"DOC(
Bitwise shift operator performs element-wise operation. For each input element, if the
attribute "direction" is "RIGHT", this operator moves its binary representation toward
the right side so that the input value is effectively decreased. If the attribute "direction"
is "LEFT", bits of binary representation moves toward the left side, which results the
increase of its actual value. The input X is the tensor to be shifted and another input
Y specifies the amounts of shifting. For example, if "direction" is "Right", X is [1, 4],
and S is [1, 1], the corresponding output Z would be [0, 2]. If "direction" is "LEFT" with
X=[1, 2] and S=[1, 2], the corresponding output Y would be [2, 8].

Because this operator supports Numpy-style broadcasting, X's and Y's shapes are
not necessarily identical.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    BitShift,
    11,
    OpSchema()
        .SetDoc(GET_OP_DOC_STR(std::string(BitShift_ver11_doc) + GenerateBroadcastingDocMul()))
        .Input(
            0,
            "X",
            "First operand, input to be shifted.",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Input(1, "Y", "Second operand, amounts of shift.", "T", OpSchema::Single, true, 1, OpSchema::NonDifferentiable)
        .Output(0, "Z", "Output tensor", "T", OpSchema::Single, true, 1, OpSchema::NonDifferentiable)
        .TypeConstraint(
            "T",
            {"tensor(uint8)", "tensor(uint16)", "tensor(uint32)", "tensor(uint64)"},
            "Constrain input and output types to integer tensors.")
        .Attr(
            "direction",
            "Direction of moving bits. It can be either \"RIGHT\" (for right shift) "
            "or \"LEFT\" (for left shift).",
            AttributeProto::STRING)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          // Type inference
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          // Shape inference
          if (hasNInputShapes(ctx, 2))
            bidirectionalBroadcastShapeInference(
                ctx.getInputType(0)->tensor_type().shape(),
                ctx.getInputType(1)->tensor_type().shape(),
                *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
        }));

ONNX_OPERATOR_SET_SCHEMA(
    LessOrEqual,
    16,
    OpSchema()
        .FillUsing(BinaryLogicDocGenerator("less_equal"))
        .TypeConstraint("T", OpSchema::all_numeric_types_ir4(), "Constrain input types to all numeric tensors.")
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
    16,
    OpSchema()
        .FillUsing(BinaryLogicDocGenerator("greater_equal"))
        .TypeConstraint("T", OpSchema::all_numeric_types_ir4(), "Constrain input types to all numeric tensors.")
        .TypeConstraint("T1", {"tensor(bool)"}, "Constrain output to boolean tensor.")
        .TypeAndShapeInferenceFunction(InferenceFunction())
        .FunctionBody(R"ONNX(
        {
            O1 = Greater (A, B)
            O2 = Equal (A, B)
            C = Or (O1, O2)
        }
        )ONNX"));

static const char* BitwiseNot_ver18_doc = R"DOC(
Returns the bitwise not of the input tensor element-wise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    BitwiseNot,
    18,
    OpSchema()
        .SetDoc(BitwiseNot_ver18_doc)
        .Input(0, "X", "Input tensor", "T", OpSchema::Single, true, 1, OpSchema::NonDifferentiable)
        .Output(0, "Y", "Output tensor", "T", OpSchema::Single, true, 1, OpSchema::NonDifferentiable)
        .TypeConstraint(
            "T",
            {"tensor(uint8)",
             "tensor(uint16)",
             "tensor(uint32)",
             "tensor(uint64)",
             "tensor(int8)",
             "tensor(int16)",
             "tensor(int32)",
             "tensor(int64)"},
            "Constrain input/output to integer tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

std::function<void(OpSchema&)> BinaryBitwiseDocGenerator(const char* name) {
  return [=](OpSchema& schema) {
    std::string doc;
    POPULATE_OP_DOC_STR(doc = R"DOC(
Returns the tensor resulting from performing the bitwise `{name}` operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

{broadcast_doc}
)DOC";
                        ReplaceAll(doc, "{name}", name);
                        ReplaceAll(doc, "{broadcast_doc}", GenerateBroadcastingDocMul().c_str()););
    schema.SetDoc(doc);
    schema.Input(
        0,
        "A",
        "First input operand for the bitwise operator.",
        "T",
        OpSchema::Single,
        true,
        1,
        OpSchema::NonDifferentiable);
    schema.Input(
        1,
        "B",
        "Second input operand for the bitwise operator.",
        "T",
        OpSchema::Single,
        true,
        1,
        OpSchema::NonDifferentiable);
    schema.Output(0, "C", "Result tensor.", "T", OpSchema::Single, true, 1, OpSchema::NonDifferentiable);
    schema.TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      // Type inference
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
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
    BitwiseAnd,
    18,
    OpSchema()
        .FillUsing(BinaryBitwiseDocGenerator("and"))
        .TypeConstraint(
            "T",
            {"tensor(uint8)",
             "tensor(uint16)",
             "tensor(uint32)",
             "tensor(uint64)",
             "tensor(int8)",
             "tensor(int16)",
             "tensor(int32)",
             "tensor(int64)"},
            "Constrain input to integer tensors."));

ONNX_OPERATOR_SET_SCHEMA(
    BitwiseOr,
    18,
    OpSchema()
        .FillUsing(BinaryBitwiseDocGenerator("or"))
        .TypeConstraint(
            "T",
            {"tensor(uint8)",
             "tensor(uint16)",
             "tensor(uint32)",
             "tensor(uint64)",
             "tensor(int8)",
             "tensor(int16)",
             "tensor(int32)",
             "tensor(int64)"},
            "Constrain input to integer tensors."));

ONNX_OPERATOR_SET_SCHEMA(
    BitwiseXor,
    18,
    OpSchema()
        .FillUsing(BinaryBitwiseDocGenerator("xor"))
        .TypeConstraint(
            "T",
            {"tensor(uint8)",
             "tensor(uint16)",
             "tensor(uint32)",
             "tensor(uint64)",
             "tensor(int8)",
             "tensor(int16)",
             "tensor(int32)",
             "tensor(int64)"},
            "Constrain input to integer tensors."));

} // namespace ONNX_NAMESPACE
