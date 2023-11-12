/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <functional>
#include "onnx/defs/function.h"
#include "onnx/defs/math/utils.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/tensor_proto_util.h"

namespace ONNX_NAMESPACE {

std::function<void(OpSchema&)> MathDocGenerator_opset13(const char* name) {
  return [=](OpSchema& schema) {
    std::string doc;
    POPULATE_OP_DOC_STR(doc = R"DOC(
Performs element-wise binary {name} (with Numpy-style broadcasting support).

{broadcast_doc}
)DOC";
                        ReplaceAll(doc, "{name}", name);
                        ReplaceAll(doc, "{broadcast_doc}", GenerateBroadcastingDocMul().c_str()););
    schema.SetDoc(doc);
    schema.Input(0, "A", "First operand.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable);
    schema.Input(1, "B", "Second operand.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable);
    schema.Output(
        0,
        "C",
        "Result, has same element type as two inputs",
        "T",
        OpSchema::Single,
        true,
        1,
        OpSchema::Differentiable);
    schema.TypeConstraint(
        "T",
        OpSchema::numeric_types_for_math_reduction_ir4(),
        "Constrain input and output types to high-precision numeric tensors.");
    schema.TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      if (hasNInputShapes(ctx, 2))
        bidirectionalBroadcastShapeInference(
            ctx.getInputType(0)->tensor_type().shape(),
            ctx.getInputType(1)->tensor_type().shape(),
            *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
    });
  };
}

ONNX_OPERATOR_SET_SCHEMA(Add, 13, OpSchema().FillUsing(MathDocGenerator_opset13("addition")));

ONNX_OPERATOR_SET_SCHEMA(Sub, 13, OpSchema().FillUsing(MathDocGenerator_opset13("subtraction")));

ONNX_OPERATOR_SET_SCHEMA(Mul, 13, OpSchema().FillUsing(MathDocGenerator_opset13("multiplication")));

ONNX_OPERATOR_SET_SCHEMA(Div, 13, OpSchema().FillUsing(MathDocGenerator_opset13("division")));

std::function<void(OpSchema&)> MathDocGenerator_opset_7(const char* name) {
  return [=](OpSchema& schema) {
    std::string doc;
    POPULATE_OP_DOC_STR(doc = R"DOC(
Performs element-wise binary {name} (with Numpy-style broadcasting support).

{broadcast_doc}
)DOC";
                        ReplaceAll(doc, "{name}", name);
                        ReplaceAll(doc, "{broadcast_doc}", GenerateBroadcastingDocMul().c_str()););
    schema.SetDoc(doc);
    schema.Input(0, "A", "First operand.", "T");
    schema.Input(1, "B", "Second operand.", "T");
    schema.Output(0, "C", "Result, has same element type as two inputs", "T");
    schema.TypeConstraint(
        "T",
        OpSchema::numeric_types_for_math_reduction(),
        "Constrain input and output types to high-precision numeric tensors.");
    schema.TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      if (hasNInputShapes(ctx, 2))
        bidirectionalBroadcastShapeInference(
            ctx.getInputType(0)->tensor_type().shape(),
            ctx.getInputType(1)->tensor_type().shape(),
            *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
    });
  };
}

ONNX_OPERATOR_SET_SCHEMA(Add, 7, OpSchema().FillUsing(MathDocGenerator_opset_7("addition")));

ONNX_OPERATOR_SET_SCHEMA(Sub, 7, OpSchema().FillUsing(MathDocGenerator_opset_7("subtraction")));

ONNX_OPERATOR_SET_SCHEMA(Mul, 7, OpSchema().FillUsing(MathDocGenerator_opset_7("multiplication")));

ONNX_OPERATOR_SET_SCHEMA(Div, 7, OpSchema().FillUsing(MathDocGenerator_opset_7("division")));

std::function<void(OpSchema&)> SoftmaxFamilyDocGenerator_opset_11(const char* name, const char* description) {
  return [=](OpSchema& schema) {
    std::string doc;
    POPULATE_OP_DOC_STR(doc = R"DOC(
The operator computes the {name} ({description}) values for each layer in the batch
 of the given input.

The input does not need to explicitly be a 2D vector; rather, it will be
coerced into one. For an arbitrary n-dimensional tensor
input \in [a_0, a_1, ..., a_{k-1}, a_k, ..., a_{n-1}] and k is
the axis provided, then input will be coerced into a 2-dimensional tensor with
dimensions [a_0 * ... * a_{k-1}, a_k * ... * a_{n-1}]. For the default
case where axis=1, this means the input tensor will be coerced into a 2D tensor
of dimensions [a_0, a_1 * ... * a_{n-1}], where a_0 is often the batch size.
In this situation, we must have a_0 = N and a_1 * ... * a_{n-1} = D.
Each of these dimensions must be matched correctly, or else the operator
will throw errors. The output tensor has the same shape
and contains the {name} values of the corresponding input.
)DOC";
                        ReplaceAll(doc, "{name}", name);
                        ReplaceAll(doc, "{description}", description););
    schema.SetDoc(doc);
    schema.Attr(
        "axis",
        "Describes the axis of the inputs when coerced "
        "to 2D; defaults to one because the 0th axis most likely describes "
        "the batch_size. Negative value means counting dimensions "
        "from the back. Accepted range is [-r, r-1] where r = rank(input).",
        AttributeProto::INT,
        static_cast<int64_t>(1));
    schema.Input(
        0,
        "input",
        "The input tensor that's coerced into a 2D matrix of size (NxD) "
        "as described above.",
        "T");
    schema.Output(
        0,
        "output",
        "The output values with the same "
        "shape as input tensor (the original size without coercion).",
        "T");
    schema.TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");
    schema.TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      // Type inference
      propagateElemTypeFromInputToOutput(ctx, 0, 0);

      // Shape inference starts
      if (!hasNInputShapes(ctx, 1)) {
        return;
      }

      // Validate the value of 'axis'
      const TensorShapeProto& input_shape = ctx.getInputType(0)->tensor_type().shape();
      int r = input_shape.dim_size();
      int axis = static_cast<int>(getAttribute(ctx, "axis", 1));
      if (axis < -r || axis >= r) {
        fail_shape_inference("'axis' must be in [", -r, " , ", (r - 1), "]. Its actual value is: ", axis);
      }

      // Shape inference
      propagateShapeFromInputToOutput(ctx, 0, 0);
    });
  };
}

ONNX_OPERATOR_SET_SCHEMA(
    Softmax,
    11,
    OpSchema().FillUsing(SoftmaxFamilyDocGenerator_opset_11("softmax", "normalized exponential")));

ONNX_OPERATOR_SET_SCHEMA(
    LogSoftmax,
    11,
    OpSchema().FillUsing(SoftmaxFamilyDocGenerator_opset_11("logsoftmax", "log of softmax")));

ONNX_OPERATOR_SET_SCHEMA(
    Hardmax,
    11,
    OpSchema().FillUsing(
        SoftmaxFamilyDocGenerator_opset_11("hardmax", "1 for the first maximum value, and 0 for all others")));

static const char* Mod_doc_10 = R"DOC(
  Performs element-wise binary modulus (with Numpy-style broadcasting support).
    The sign of the remainder is the same as that of the Divisor.

    Mod operator can also behave like C fmod() or numpy.fmod. In this case, the sign of the remainder however, will be the same as the Dividend
    (in contrast to integer mod). To force a behavior like numpy.fmod() an 'fmod' Attribute is provided.
    This attribute is set to 0 by default causing the behavior to be like integer mod.
    Setting this attribute to 1 causes the remainder to be calculated similar to that of numpy.fmod().

    If the input type is floating point, then `fmod` attribute must be set to 1.

    In case of dividend being zero, the results will be platform dependent.

  This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Mod,
    10,
    OpSchema()
        .SetDoc(Mod_doc_10)
        .Attr(
            "fmod",
            "Whether the operator should behave like fmod (default=0 meaning it will do integer mods); Set this to 1 to force fmod treatment",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Input(0, "A", "Dividend tensor", "T")
        .Input(1, "B", "Divisor tensor", "T")
        .Output(0, "C", "Remainder tensor", "T")
        .TypeConstraint(
            "T",
            OpSchema::all_numeric_types(),
            "Constrain input and output types to high-precision numeric tensors.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (hasNInputShapes(ctx, 2))
            bidirectionalBroadcastShapeInference(
                ctx.getInputType(0)->tensor_type().shape(),
                ctx.getInputType(1)->tensor_type().shape(),
                *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
        }));

static const char* Neg_ver6_doc = R"DOC(
Neg takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where each element flipped sign, y = -x, is applied to
the tensor elementwise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Neg,
    6,
    OpSchema()
        .SetDoc(Neg_ver6_doc)
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint(
            "T",
            {"tensor(float)",
             "tensor(int32)",
             "tensor(int8)",
             "tensor(int16)",
             "tensor(int64)",
             "tensor(float16)",
             "tensor(double)"},
            "Constrain input and output types to signed numeric tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Abs_ver6_doc = R"DOC(
Absolute takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the absolute is, y = abs(x), is applied to
the tensor elementwise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Abs,
    6,
    OpSchema()
        .SetDoc(Abs_ver6_doc)
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint("T", OpSchema::all_numeric_types(), "Constrain input and output types to all numeric tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Reciprocal_ver6_doc = R"DOC(
Reciprocal takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the reciprocal is, y = 1/x, is applied to
the tensor elementwise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Reciprocal,
    6,
    OpSchema()
        .SetDoc(Reciprocal_ver6_doc)
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Floor_ver6_doc = R"DOC(
Floor takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the floor is, y = floor(x), is applied to
the tensor elementwise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Floor,
    6,
    OpSchema()
        .SetDoc(Floor_ver6_doc)
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Ceil_ver6_doc = R"DOC(
Ceil takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the ceil is, y = ceil(x), is applied to
the tensor elementwise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Ceil,
    6,
    OpSchema()
        .SetDoc(Ceil_ver6_doc)
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Sqrt_ver6_doc = R"DOC(
Square root takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the square root is, y = x^0.5, is applied to
the tensor elementwise. If x is negative, then it will return NaN.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Sqrt,
    6,
    OpSchema()
        .SetDoc(Sqrt_ver6_doc)
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Relu_ver6_doc = R"DOC(
Relu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the rectified linear function, y = max(0, x), is applied to
the tensor elementwise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Relu,
    6,
    OpSchema()
        .SetDoc(Relu_ver6_doc)
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Relu_ver13_doc = R"DOC(
Relu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the rectified linear function, y = max(0, x), is applied to
the tensor elementwise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Relu,
    13,
    OpSchema()
        .SetDoc(Relu_ver13_doc)
        .Input(0, "X", "Input tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Output(0, "Y", "Output tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Exp_ver6_doc = R"DOC(
Calculates the exponential of the given input tensor, element-wise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Exp,
    6,
    OpSchema()
        .SetDoc(Exp_ver6_doc)
        .Input(0, "input", "Input tensor", "T")
        .Output(
            0,
            "output",
            "The exponential of the input tensor computed "
            "element-wise",
            "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Log_ver6_doc = R"DOC(
Calculates the natural log of the given input tensor, element-wise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Log,
    6,
    OpSchema()
        .SetDoc(Log_ver6_doc)
        .Input(0, "input", "Input tensor", "T")
        .Output(
            0,
            "output",
            "The natural log of the input tensor computed "
            "element-wise",
            "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Tanh_ver6_doc = R"DOC(
Calculates the hyperbolic tangent of the given input tensor element-wise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Tanh,
    6,
    OpSchema()
        .SetDoc(Tanh_ver6_doc)
        .Input(0, "input", "Input tensor", "T")
        .Output(
            0,
            "output",
            "The hyperbolic tangent values of the input tensor "
            "computed element-wise",
            "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Pow_ver13_doc = R"DOC(
Pow takes input data (Tensor<T>) and exponent Tensor, and
produces one output data (Tensor<T>) where the function `f(x) = x^exponent`,
is applied to the data tensor elementwise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Pow,
    13,
    OpSchema()
        .SetDoc(GET_OP_DOC_STR(std::string(Pow_ver13_doc) + GenerateBroadcastingDocMul()))
        .Input(0, "X", "First operand, base of the exponent.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Input(
            1,
            "Y",
            "Second operand, power of the exponent.",
            "T1",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .Output(0, "Z", "Output tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .TypeConstraint(
            "T",
            {"tensor(int32)",
             "tensor(int64)",
             "tensor(float16)",
             "tensor(float)",
             "tensor(double)",
             "tensor(bfloat16)"},
            "Constrain input X and output types to float/int tensors.")
        .TypeConstraint(
            "T1",
            {"tensor(uint8)",
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
            "Constrain input Y types to float/int tensors.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (hasNInputShapes(ctx, 2))
            bidirectionalBroadcastShapeInference(
                ctx.getInputType(0)->tensor_type().shape(),
                ctx.getInputType(1)->tensor_type().shape(),
                *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
        }));

static const char* Pow_ver12_doc = R"DOC(
Pow takes input data (Tensor<T>) and exponent Tensor, and
produces one output data (Tensor<T>) where the function `f(x) = x^exponent`,
is applied to the data tensor elementwise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Pow,
    12,
    OpSchema()
        .SetDoc(GET_OP_DOC_STR(std::string(Pow_ver12_doc) + GenerateBroadcastingDocMul()))
        .Input(0, "X", "First operand, base of the exponent.", "T")
        .Input(1, "Y", "Second operand, power of the exponent.", "T1")
        .Output(0, "Z", "Output tensor.", "T")
        .TypeConstraint(
            "T",
            {"tensor(int32)", "tensor(int64)", "tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input X and output types to float/int tensors.")
        .TypeConstraint(
            "T1",
            {"tensor(uint8)",
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
            "Constrain input Y types to float/int tensors.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (hasNInputShapes(ctx, 2))
            bidirectionalBroadcastShapeInference(
                ctx.getInputType(0)->tensor_type().shape(),
                ctx.getInputType(1)->tensor_type().shape(),
                *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
        }));

static const char* Sigmoid_ver6_doc = R"DOC(
Sigmoid takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the sigmoid function, y = 1 / (1 + exp(-x)), is applied to the
tensor elementwise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Sigmoid,
    6,
    OpSchema()
        .SetDoc(Sigmoid_ver6_doc)
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

// Generate opschema for element-wise ops. Leaves type constraint "T"
// unspecified.
std::function<void(OpSchema&)> ElementwiseMultiOpDocGenerator_opset8(const char* name) {
  return [=](OpSchema& schema) {
    std::string doc;
    POPULATE_OP_DOC_STR(doc = R"DOC(
Element-wise {name} of each of the input tensors (with Numpy-style broadcasting support).
All inputs and outputs must have the same data type.
{broadcast_doc}
)DOC";
                        ReplaceAll(doc, "{name}", name);
                        ReplaceAll(doc, "{broadcast_doc}", GenerateBroadcastingDocMul().c_str()););
    schema.SetDoc(doc);
    schema.Input(0, "data_0", "List of tensors for " + std::string(name) + ".", "T", OpSchema::Variadic);
    schema.Output(0, name, "Output tensor.", "T");
    schema.TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      int num_inputs = static_cast<int>(ctx.getNumInputs());
      std::vector<const TensorShapeProto*> shapes;
      for (int i = 0; i < num_inputs; ++i) {
        auto input_type = ctx.getInputType(i);
        if (nullptr == input_type || !input_type->has_tensor_type() || !input_type->tensor_type().has_shape()) {
          return;
        }
        shapes.push_back(&input_type->tensor_type().shape());
      }

      multidirectionalBroadcastShapeInference(shapes, *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
    });
  };
}

ONNX_OPERATOR_SET_SCHEMA(
    Max,
    12,
    OpSchema()
        .FillUsing(ElementwiseMultiOpDocGenerator_opset8("max"))
        .TypeConstraint("T", OpSchema::all_numeric_types(), "Constrain input and output types to numeric tensors."));

ONNX_OPERATOR_SET_SCHEMA(
    Min,
    12,
    OpSchema()
        .FillUsing(ElementwiseMultiOpDocGenerator_opset8("min"))
        .TypeConstraint("T", OpSchema::all_numeric_types(), "Constrain input and output types to numeric tensors."));

ONNX_OPERATOR_SET_SCHEMA(
    Sum,
    8,
    OpSchema()
        .FillUsing(ElementwiseMultiOpDocGenerator_opset8("sum"))
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors."));

ONNX_OPERATOR_SET_SCHEMA(
    Mean,
    8,
    OpSchema()
        .FillUsing(ElementwiseMultiOpDocGenerator_opset8("mean"))
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors."));

static const char* Clip_ver12_doc = R"DOC(
Clip operator limits the given input within an interval. The interval is
specified by the inputs 'min' and 'max'. They default to
numeric_limits::lowest() and numeric_limits::max(), respectively.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Clip,
    12,
    OpSchema()
        .SetDoc(Clip_ver12_doc)
        .Input(0, "input", "Input tensor whose elements to be clipped", "T")
        .Input(
            1,
            "min",
            "Minimum value, under which element is replaced by min. "
            "It must be a scalar(tensor of empty shape).",
            "T",
            OpSchema::Optional)
        .Input(
            2,
            "max",
            "Maximum value, above which element is replaced by max. "
            "It must be a scalar(tensor of empty shape).",
            "T",
            OpSchema::Optional)
        .Output(0, "output", "Output tensor with clipped input elements", "T")
        .TypeConstraint("T", OpSchema::all_numeric_types(), "Constrain input and output types to all numeric tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Gemm_ver11_doc = R"DOC(General Matrix multiplication:
https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3

A' = transpose(A) if transA else A

B' = transpose(B) if transB else B

Compute Y = alpha * A' * B' + beta * C, where input tensor A has shape (M, K) or (K, M),
input tensor B has shape (K, N) or (N, K), input tensor C is broadcastable to shape (M, N),
and output tensor Y has shape (M, N). A will be transposed before doing the
computation if attribute transA is non-zero, same for B and transB.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Gemm,
    11,
    OpSchema()
        .SetDoc(GET_OP_DOC_STR(
            std::string(Gemm_ver11_doc) + GenerateBroadcastingDocUni("tensor C", "tensor A * B") + "\n" +
            GenerateOptionalArgumentsDoc()))
        .Input(
            0,
            "A",
            "Input tensor A. "
            "The shape of A should be (M, K) if transA is 0, "
            "or (K, M) if transA is non-zero.",
            "T")
        .Input(
            1,
            "B",
            "Input tensor B. "
            "The shape of B should be (K, N) if transB is 0, "
            "or (N, K) if transB is non-zero.",
            "T")
        .Input(
            2,
            "C",
            "Optional input tensor C. "
            "If not specified, the computation is done as if C is a scalar 0. "
            "The shape of C should be unidirectional broadcastable to (M, N).",
            "T",
            OpSchema::Optional)
        .Output(0, "Y", "Output tensor of shape (M, N).", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)",
             "tensor(float)",
             "tensor(double)",
             "tensor(uint32)",
             "tensor(uint64)",
             "tensor(int32)",
             "tensor(int64)"},
            "Constrain input and output types to float/int tensors.")
        .Attr("transA", "Whether A should be transposed", AttributeProto::INT, static_cast<int64_t>(0))
        .Attr("transB", "Whether B should be transposed", AttributeProto::INT, static_cast<int64_t>(0))
        .Attr("alpha", "Scalar multiplier for the product of input tensors A * B.", AttributeProto::FLOAT, 1.0f)
        .Attr("beta", "Scalar multiplier for input tensor C.", AttributeProto::FLOAT, 1.0f)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (hasNInputShapes(ctx, 2)) {
            auto transAAttr = ctx.getAttribute("transA");
            bool transA = transAAttr ? static_cast<int>(transAAttr->i()) != 0 : false;
            auto transBAttr = ctx.getAttribute("transB");
            bool transB = transBAttr ? static_cast<int>(transBAttr->i()) != 0 : false;
            auto& first_input_shape = getInputShape(ctx, 0);
            auto& second_input_shape = getInputShape(ctx, 1);
            if (first_input_shape.dim_size() != 2) {
              fail_shape_inference("First input does not have rank 2");
            }
            if (second_input_shape.dim_size() != 2) {
              fail_shape_inference("Second input does not have rank 2");
            }
            updateOutputShape(ctx, 0, {first_input_shape.dim(transA ? 1 : 0), second_input_shape.dim(transB ? 0 : 1)});
          }
        }));

void matmulShapeInference_opset_9(ONNX_NAMESPACE::InferenceContext& ctx, int input1Idx, int input2Idx) {
  if (!hasInputShape(ctx, input1Idx) || !hasInputShape(ctx, input2Idx)) {
    return;
  }

  const auto shape0 = ctx.getInputType(input1Idx)->tensor_type().shape();
  const auto shape1 = ctx.getInputType(input2Idx)->tensor_type().shape();

  if (shape0.dim_size() == 0 || shape1.dim_size() == 0) {
    fail_shape_inference("Input tensors of wrong rank (0).");
  }

  ONNX_NAMESPACE::TensorShapeProto shapeL, shapeR;

  // First promote each shape to at least rank-2. This logic is
  // specific to matmul, not generic broadcasting.
  {
    if (shape0.dim_size() == 1) {
      shapeL.add_dim()->set_dim_value(1);
      *shapeL.add_dim() = shape0.dim(0);
    } else {
      *shapeL.mutable_dim() = shape0.dim();
    }
    if (shape1.dim_size() == 1) {
      *shapeR.add_dim() = shape1.dim(0);
      shapeR.add_dim()->set_dim_value(1);
    } else {
      *shapeR.mutable_dim() = shape1.dim();
    }
  }

  // Check for compatible matrix multiply dimensions
  {
    auto dimL = shapeL.dim(shapeL.dim_size() - 1);
    auto dimR = shapeR.dim(shapeR.dim_size() - 2);
    if (dimL.has_dim_value() && dimR.has_dim_value() && dimL.dim_value() != dimR.dim_value()) {
      fail_shape_inference("Incompatible dimensions for matrix multiplication");
    }
  }

  ONNX_NAMESPACE::TensorShapeProto resultShape;

  // Now call out to generic multidimensional broadcasting for
  // the broadcastable prefixes.
  {
    ONNX_NAMESPACE::TensorShapeProto prefixShapeL, prefixShapeR;
    for (int i = 0; i < shapeL.dim_size() - 2; ++i) {
      *prefixShapeL.add_dim() = shapeL.dim(i);
    }
    for (int i = 0; i < shapeR.dim_size() - 2; ++i) {
      *prefixShapeR.add_dim() = shapeR.dim(i);
    }
    bidirectionalBroadcastShapeInference(prefixShapeL, prefixShapeR, resultShape);
  }

  // Back to matmul-specific. Add the trailing dimensions back in.
  {
    if (shape0.dim_size() != 1) {
      *resultShape.add_dim() = shapeL.dim(shapeL.dim_size() - 2);
    }
    if (shape1.dim_size() != 1) {
      *resultShape.add_dim() = shapeR.dim(shapeR.dim_size() - 1);
    }
  }

  *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape() = resultShape;
}

static const char* MatMul_ver9_doc = R"DOC(
Matrix product that behaves like numpy.matmul: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    MatMul,
    9,
    OpSchema()
        .Input(0, "A", "N-dimensional matrix A", "T")
        .Input(1, "B", "N-dimensional matrix B", "T")
        .Output(0, "Y", "Matrix multiply results from A * B", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)",
             "tensor(float)",
             "tensor(double)",
             "tensor(uint32)",
             "tensor(uint64)",
             "tensor(int32)",
             "tensor(int64)"},
            "Constrain input and output types to float/int tensors.")
        .SetDoc(MatMul_ver9_doc)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          matmulShapeInference_opset_9(ctx, 0, 1);
        }));

static const char* Expand_ver8_doc = R"DOC(
Broadcast the input tensor following the given shape and the broadcast rule.
The broadcast rule is similar to numpy.array(input) * numpy.ones(shape):
Dimensions are right alignment;
Two corresponding dimensions must have the same value, or one of them is equal to 1.
Also, this operator is similar to numpy.broadcast_to(input, shape),
but the major difference is numpy.broadcast_to() does not allow shape to be smaller than input.size().
It is possible that the output.shape is not equal to shape, when some dimensions in shape is equal to 1,
or the shape.ndim < input.shape.ndim.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Expand,
    8,
    OpSchema()
        .SetDoc(Expand_ver8_doc)
        .Input(0, "input", "Input tensor", "T")
        .Input(
            1,
            "shape",
            "A 1-D tensor indicates the shape you want to expand to, following the broadcast rule",
            "tensor(int64)")
        .Output(0, "output", "Output tensor", "T")
        .TypeConstraint("T", OpSchema::all_tensor_types(), "Constrain input and output types to all tensors.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          // Type inference
          propagateElemTypeFromInputToOutput(ctx, 0, 0);

          // Shape inference
          // For shape inference, we need both input shape
          const auto* shape_initializer = ctx.getInputData(1);
          if (hasNInputShapes(ctx, 2)) {
            const auto& shape_input_shape = ctx.getInputType(1)->tensor_type().shape();
            if (shape_input_shape.dim_size() != 1) {
              fail_shape_inference("'shape' input must be 1D tensor");
            }

            const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
            TensorShapeProto second_shape;
            if (nullptr != shape_initializer) {
              const auto& shape_data = ParseData<int64_t>(shape_initializer);

              for (const auto& e : shape_data) {
                auto* dim = second_shape.add_dim();
                dim->set_dim_value(e);
              }
            } else if (shape_input_shape.dim(0).has_dim_value()) {
              // Attempt rank inference using shape of shape input
              int64_t dim_value = shape_input_shape.dim(0).dim_value();
              for (int64_t i = 0; i < dim_value; ++i) {
                second_shape.add_dim();
              }
            } else {
              return;
            }
            bidirectionalBroadcastShapeInference(input_shape, second_shape, *getOutputShape(ctx, 0));
          }
        }));

static const char* Sign_ver9_doc = R"DOC(
Calculate the sign of the given input tensor element-wise.
If input > 0, output 1. if input < 0, output -1. if input == 0, output 0.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Sign,
    9,
    OpSchema()
        .SetDoc(Sign_ver9_doc)
        .Input(0, "input", "Input tensor", "T")
        .Output(
            0,
            "output",
            "The sign of the input tensor "
            "computed element-wise. It has the same shape and type of the input.",
            "T")
        .TypeConstraint("T", OpSchema::all_numeric_types(), "Constrain input and output types to all numeric tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Erf_ver9_doc = R"DOC(
Computes the error function of the given input tensor element-wise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Erf,
    9,
    OpSchema()
        .SetDoc(Erf_ver9_doc)
        .Input(0, "input", "Input tensor", "T")
        .Output(
            0,
            "output",
            "The error function of the input tensor "
            "computed element-wise. It has the same shape and type of the input.",
            "T")
        .TypeConstraint("T", OpSchema::all_numeric_types(), "Constrain input and output types to all numeric tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* CumSum_ver11_doc = R"DOC(
Performs cumulative sum of the input elements along the given axis.
By default, it will do the sum inclusively meaning the first element is copied as is.
Through an `exclusive` attribute, this behavior can change to exclude the first element.
It can also perform summation in the opposite direction of the axis. For that, set `reverse` attribute to 1.

Example:
```
input_x = [1, 2, 3]
axis=0
output = [1, 3, 6]
exclusive=1
output = [0, 1, 3]
exclusive=0
reverse=1
output = [6, 5, 3]
exclusive=1
reverse=1
output = [5, 3, 0]
```
 )DOC";

ONNX_OPERATOR_SET_SCHEMA(
    CumSum,
    11,
    OpSchema()
        .SetDoc(CumSum_ver11_doc)
        .Attr(
            "exclusive",
            "If set to 1 will return exclusive sum in which the top element is not included."
            " In other terms, if set to 1, the j-th output element would be the sum of the first (j-1) elements."
            " Otherwise, it would be the sum of the first j elements.",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Attr(
            "reverse",
            "If set to 1 will perform the sums in reverse direction.",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Input(
            0,
            "x",
            "An input tensor that is to be processed.",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .Input(
            1,
            "axis",
            "A 0-D tensor. Must be in the range [-rank(x), rank(x)-1]. "
            "Negative value means counting dimensions from the back.",
            "T2",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Output(
            0,
            "y",
            "Output tensor of the same type as 'x' with cumulative sums of the x's elements",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .TypeConstraint(
            "T",
            {"tensor(uint32)", "tensor(uint64)", "tensor(int32)", "tensor(int64)", "tensor(float)", "tensor(double)"},
            "Input can be of any tensor type.")
        .TypeConstraint("T2", {"tensor(int32)", "tensor(int64)"}, "axis tensor can be int32 or int64 only")
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput));

static const char* NegativeLogLikelihoodLoss_ver12_doc = R"DOC(
A NegativeLogLikelihoodLoss operator computes (weighted) negative log likelihood loss.
Its "input" tensor has the shape of (N, C, d1, d2, ..., dk) where k >= 0.
The "input" tensor contains log-probabilities for input[n, :, d_1, d_2,..., d_k] being in a class of [0, C).
The operator's "target" input tensor has the shape of (N, d1, d2, ..., dk). It encodes class labels (one of C classes)
or it may contain a special value (indicated by an attribute ignore_index) for N x d1 x d2 x ... x dk samples.
The loss value for input[n, :, d_1, d_2,...d_k] being classified as class c = target[n][d_1][d_2]...[d_k] is computed as:
    loss[n][d_1][d_2]...[d_k] = -input[n][c][d_1][d_2]...[d_k].
When an optional "weight" is provided, the sample loss is calculated as:
    loss[n][d_1][d_2]...[d_k] = -input[n][c][d_1][d_2]...[d_k] * weight[c].
loss is zero for the case when target-value equals ignore_index.

    loss[n][d_1][d_2]...[d_k] = 0, when target[n][d_1][d_2]...[d_k] = ignore_index
If "reduction" attribute is set to "none", the operator's output will be the above loss with shape (N, d1, d2, ..., dk).
If "reduction" attribute is set to "mean" (the default attribute value), the output loss is (weight) averaged:
    mean(loss), if "weight" is not provided,
or if weight is provided,
    sum(loss) / sum(weight[target[n][d_1][d_2]...[d_k]]]), for all samples.
If "reduction" attribute is set to "sum", the output is a scalar:
    sum(loss).
See also https://pytorch.org/docs/stable/nn.html#torch.nn.NLLLoss.
Example 1:
    // negative log likelihood loss, "none" reduction
    N, C, d1 = 2, 3, 2
    input = [[[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]],
             [[0.0, 1.0], [2.0, 2.0], [1.0, 2]]]
    target = [[2, 1], [0, 2]]
    loss = np.zeros((N, d1))
    for n in range(N):
        for d_1 in range(d1):
            c = target[n][d_1]
            loss[n][d_1] = -input[n][c][d_1]
    // print(loss)
    // [[-3. -2.]
    //  [-0. -2.]]
Example 2:
    // weighted negative log likelihood loss, sum reduction
    N, C, d1 = 2, 3, 2
    input = [[[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]],
            [[0.0, 1.0], [2.0, 2.0], [1.0, 2]]]
    target = [[2, 1], [0, 2]]
    weight = [0.2, 0.3, 0.1]
    loss = np.zeros((N, d1))
    for n in range(N):
        for d_1 in range(d1):
            c = target[n][d_1]
            loss[n][d_1] = -input[n][c][d_1] * weight[c]
    loss = np.sum(loss)
    // print(loss)
    // -1.1
Example 3:
    // weighted negative log likelihood loss, mean reduction
    N, C, d1 = 2, 3, 2
    input = [[[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]],
            [[0.0, 1.0], [2.0, 2.0], [1.0, 2]]]
    target = [[2, 1], [0, 2]]
    weight = [0.2, 0.3, 0.1]
    loss = np.zeros((N, d1))
    weight_total = 0
    for n in range(N):
        for d_1 in range(d1):
            c = target[n][d_1]
            loss[n][d_1] = -input[n][c][d_1] * weight[c]
            weight_total = weight_total + weight[c]
    loss = np.sum(loss) / weight_total
    // print(loss)
    // -1.57
)DOC";

TensorProto ToDimensionOneFloatTensor_old(float value) {
  auto t = ToTensor(std::vector<float>({value}));
  t.add_dims(1);
  return t;
}

TensorProto ToDimensionOneTensor_old(int32_t value) {
  auto t = ToTensor(std::vector<int32_t>({value}));
  t.add_dims(1);
  return t;
}

TensorProto ToDimensionOneInt64Tensor_old(int64_t value) {
  auto t = ToTensor(std::vector<int64_t>({value}));
  t.add_dims(1);
  return t;
}

TensorProto ToDimensionOneInt64Tensor_old(std::vector<int64_t> value) {
  auto t = ToTensor(value);
  t.add_dims(value.size());
  return t;
}

bool BuildContextDependentFunctionBody_opset12(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  if (ctx.getInputType(0) == nullptr) {
    // we cannot create a correct function body without knowing the input type
    return false;
  }
  auto input_type = ctx.getInputType(0)->tensor_type().elem_type();
  bool float_input = input_type == TensorProto_DataType_FLOAT;
  auto reduction_attr_proto = ctx.getAttribute("reduction");
  std::string reduction_attr =
      reduction_attr_proto != nullptr && reduction_attr_proto->has_s() ? reduction_attr_proto->s() : "mean";
  std::vector<FunctionBodyHelper::NodeDef> body;
  body.push_back({{"const_zero"}, "Constant", {}, {MakeAttribute("value", ToDimensionOneTensor_old(0))}});

  body.push_back({{"const_one"}, "Constant", {}, {MakeAttribute("value", ToDimensionOneTensor_old(1))}});

  body.push_back({{"expanded_target"}, "Unsqueeze", {"target"}, {MakeAttribute("axes", std::vector<int64_t>({1}))}});

  if (ctx.getAttribute("ignore_index") == nullptr) {
    body.push_back(
        {{"input_gather_element"},
         "GatherElements",
         {"input", "expanded_target"},
         {MakeAttribute("axis", (int64_t)1)}});

    body.push_back({{"loss_NCdd"}, "Neg", {"input_gather_element"}});

    body.push_back({{"loss_N1dd"}, "Slice", {"loss_NCdd", "const_zero", "const_one", "const_one"}});

    if (!ctx.hasInput(2)) {
      if (reduction_attr == "none") {
        body.push_back({{"loss"}, "Squeeze", {"loss_N1dd"}, {MakeAttribute("axes", std::vector<int64_t>({1}))}});
      } else {
        body.push_back({{"loss_Ndd"}, "Squeeze", {"loss_N1dd"}, {MakeAttribute("axes", std::vector<int64_t>({1}))}});
        if (reduction_attr == "mean") {
          body.push_back({{"loss"}, "ReduceMean", {"loss_Ndd"}, {MakeAttribute("keepdims", (int64_t)0)}});
        } else {
          body.push_back({{"loss"}, "ReduceSum", {"loss_Ndd"}, {MakeAttribute("keepdims", (int64_t)0)}});
        }
      }
    } else {
      body.push_back({{"weight_gather"}, "Gather", {"weight", "target"}});
      body.push_back(
          {{"loss_unweighted"}, "Squeeze", {"loss_N1dd"}, {MakeAttribute("axes", std::vector<int64_t>({1}))}});
      if (reduction_attr == "none") {
        body.push_back({{"loss"}, "Mul", {"loss_unweighted", "weight_gather"}});
      } else {
        body.push_back({{"loss_Ndd"}, "Mul", {"loss_unweighted", "weight_gather"}});
        if (reduction_attr == "mean") {
          body.push_back({{"loss_sum"}, "ReduceSum", {"loss_Ndd"}, {MakeAttribute("keepdims", (int64_t)0)}});
          body.push_back(
              {{"weight_gather_sum"}, "ReduceSum", {"weight_gather"}, {MakeAttribute("keepdims", (int64_t)0)}});
          body.push_back({{"loss"}, "Div", {"loss_sum", "weight_gather_sum"}});
        } else {
          body.push_back({{"loss"}, "ReduceSum", {"loss_Ndd"}, {MakeAttribute("keepdims", (int64_t)0)}});
        }
      }
    }
  } else {
    body.push_back(
        {{"const_ignore_index"},
         "Constant",
         {},
         {MakeAttribute("value", ToDimensionOneInt64Tensor_old(ctx.getAttribute("ignore_index")->i()))}});

    body.push_back({{"const_zero_target_typed"}, "Sub", {"expanded_target", "expanded_target"}});
    body.push_back(
        {{"expanded_target_int64"},
         "Cast",
         {"expanded_target"},
         {MakeAttribute("to", (int64_t)TensorProto_DataType::TensorProto_DataType_INT64)}});

    body.push_back({{"mask"}, "Equal", {"expanded_target_int64", "const_ignore_index"}});
    body.push_back({{"transform_targets"}, "Where", {"mask", "const_zero_target_typed", "expanded_target"}});
    body.push_back(
        {{"input_gather_element"},
         "GatherElements",
         {"input", "transform_targets"},
         {MakeAttribute("axis", (int64_t)1)}});
    body.push_back(
        {{"const_zero_float"}, "Constant", {}, {MakeAttribute("value", ToDimensionOneFloatTensor_old(0.0f))}});
    if (!float_input) {
      body.push_back(
          {{"const_zero_casted"},
           "Cast",
           {"const_zero_float"},
           {MakeAttribute("to", static_cast<int64_t>(input_type))}});
    }
    body.push_back(
        {{"input_gather_element_transform"},
         "Where",
         {"mask", float_input ? "const_zero_float" : "const_zero_casted", "input_gather_element"}});
    body.push_back({{"loss_NCdd"}, "Neg", {"input_gather_element_transform"}});
    body.push_back({{"loss_N1dd"}, "Slice", {"loss_NCdd", "const_zero", "const_one", "const_one"}});

    if (!ctx.hasInput(2)) {
      body.push_back({{"squeeze_mask"}, "Squeeze", {"mask"}, {MakeAttribute("axes", std::vector<int64_t>({1}))}});

      body.push_back(
          {{"const_one_float"}, "Constant", {}, {MakeAttribute("value", ToDimensionOneFloatTensor_old(1.0f))}});
      if (!float_input) {
        body.push_back(
            {{"const_one_casted"},
             "Cast",
             {"const_one_float"},
             {MakeAttribute("to", static_cast<int64_t>(input_type))}});
      }
      body.push_back(
          {{"weight_gather"},
           "Where",
           {"squeeze_mask",
            float_input ? "const_zero_float" : "const_zero_casted",
            float_input ? "const_one_float" : "const_one_casted"}});

    } else {
      body.push_back({{"weight_gather_temp"}, "Gather", {"weight", "transform_targets"}});

      body.push_back(
          {{"weight_gather_temp_1"},
           "Where",
           {"mask", float_input ? "const_zero_float" : "const_zero_casted", "weight_gather_temp"}});

      body.push_back(
          {{"weight_gather"}, "Squeeze", {"weight_gather_temp_1"}, {MakeAttribute("axes", std::vector<int64_t>({1}))}});
    }

    body.push_back({{"loss_unweighted"}, "Squeeze", {"loss_N1dd"}, {MakeAttribute("axes", std::vector<int64_t>({1}))}});
    if (reduction_attr == "none") {
      body.push_back({{"loss"}, "Mul", {"loss_unweighted", "weight_gather"}});
    } else {
      body.push_back({{"loss_Ndd"}, "Mul", {"loss_unweighted", "weight_gather"}});
      if (reduction_attr == "mean") {
        body.push_back({{"loss_sum"}, "ReduceSum", {"loss_Ndd"}, {MakeAttribute("keepdims", (int64_t)0)}});
        body.push_back(
            {{"weight_gather_sum"}, "ReduceSum", {"weight_gather"}, {MakeAttribute("keepdims", (int64_t)0)}});
        body.push_back({{"loss"}, "Div", {"loss_sum", "weight_gather_sum"}});
      } else {
        body.push_back({{"loss"}, "ReduceSum", {"loss_Ndd"}, {MakeAttribute("keepdims", (int64_t)0)}});
      }
    }
  }

  auto func_nodes = FunctionBodyHelper::BuildNodes(body);
  for (const auto& node : func_nodes) {
    auto new_node = functionProto.add_node();
    new_node->CopyFrom(node);
  }

  schema.BuildFunction(functionProto);
  return true;
}

ONNX_OPERATOR_SET_SCHEMA(
    NegativeLogLikelihoodLoss,
    12,
    OpSchema()
        .SetDoc(NegativeLogLikelihoodLoss_ver12_doc)
        .Input(0, "input", "Input tensor of shape (N, C) or (N, C, d1, d2, ..., dk).", "T")
        .Input(
            1,
            "target",
            "Target tensor of shape (N) or (N, d1, d2, ..., dk). Target element value shall be in range of [0, C). "
            "If ignore_index is specified, it may have a value outside [0, C) and the target values should either be "
            "in the range [0, C) or have the value ignore_index.",
            "Tind")
        .Input(
            2,
            "weight",
            "Optional rescaling weight tensor. "
            "If given, it has to be a tensor of size C. Otherwise, it is treated as if having all ones.",
            "T",
            OpSchema::Optional)
        .Output(0, "loss", "The negative log likelihood loss", "T")
        .Attr(
            "reduction",
            "Type of reduction to apply to loss: none, sum, mean (default). "
            "'none': the output is the loss for each sample. "
            "'sum': the output will be summed. "
            "'mean': the sum of the output will be divided by the sum of applied weights.",
            AttributeProto::STRING,
            std::string("mean"))
        .Attr(
            "ignore_index",
            "Specifies a target value that is ignored and does not contribute to the input gradient. It's an optional value.",
            AttributeProto::INT,
            false)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input, weight, and output types to floating-point tensors.")
        .TypeConstraint("Tind", {"tensor(int32)", "tensor(int64)"}, "Constrain target to integer types")
        .SetContextDependentFunctionBodyBuilder(BuildContextDependentFunctionBody_opset12)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          // Type inference
          propagateElemTypeFromInputToOutput(ctx, 0, 0);

          // Shape inference
          if (hasNInputShapes(ctx, 2)) {
            const TensorShapeProto& input_shape = ctx.getInputType(0)->tensor_type().shape();
            const TensorShapeProto& target_shape = ctx.getInputType(1)->tensor_type().shape();

            const int input_rank = static_cast<int>(input_shape.dim_size());
            const int target_rank = static_cast<int>(target_shape.dim_size());

            if (input_rank < 2) {
              fail_shape_inference("Input rank must be >= 2.");
            }
            if (target_rank != input_rank - 1) {
              fail_shape_inference("Target rank must be 1 less than the input rank.");
            }

            // match input dimensions (N, C, d1, ..., dk) with target
            // dimensions of (C, d1, ..., dk)
            for (int dim = 0; dim < target_rank; dim++) {
              const auto input_dim = dim == 0 ? input_shape.dim(dim) : input_shape.dim(dim + 1);
              const auto target_dim = target_shape.dim(dim);
              if (input_dim.has_dim_value() && target_dim.has_dim_value() &&
                  input_dim.dim_value() != target_dim.dim_value())
                fail_shape_inference("Input and target dimension value mismatch.");
            }

            if (ctx.getNumInputs() == 3 && hasInputShape(ctx, 2)) {
              const TensorShapeProto& weight_shape = ctx.getInputType(2)->tensor_type().shape();
              if (weight_shape.dim_size() != 1) {
                fail_shape_inference("Weight rank must be 1.");
              }
            }

            TensorShapeProto* output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
            if (getAttribute(ctx, "reduction", "mean") == "none") {
              // output tensor is of shape (N, d1, d2, ..., dk) if
              // reduction attribute is "none".
              for (int i = 0; i < input_rank - 1; i++) {
                auto* dim = output_shape->add_dim();
                if (i == 0)
                  *dim = input_shape.dim(i);
                else
                  *dim = input_shape.dim(i + 1);
              }
            }
            // otherwise output is a scalar.
          }
        }));

const char* reduction_doc_sce_opset12 =
    "Type of reduction to apply to loss: none, sum, mean(default). "
    "'none': no reduction will be applied, "
    "'sum': the output will be summed. "
    "'mean': the sum of the output will be divided by the number of "
    "elements in the output.";

static const char* SoftmaxCrossEntropyLoss_ver12_doc =
    R"DOC(Loss function that measures the softmax cross entropy
between 'scores' and 'labels'.
This operator first computes a loss tensor whose shape is identical to the labels input.
If the input is 2-D with shape (N, C), the loss tensor may be a N-element vector L = (l_1, l_2, ..., l_N).
If the input is N-D tensor with shape (N, C, D1, D2, ..., Dk),
the loss tensor L may have (N, D1, D2, ..., Dk) as its shape and L[i,][j_1][j_2]...[j_k] denotes a scalar element in L.
After L is available, this operator can optionally do a reduction operator.

shape(scores): (N, C) where C is the number of classes, or (N, C, D1, D2,..., Dk),
        with K >= 1 in case of K-dimensional loss.
shape(labels): (N) where each value is 0 <= labels[i] <= C-1, or (N, D1, D2,..., Dk),
        with K >= 1 in case of K-dimensional loss.

The loss for one sample, l_i, can caculated as follows:
    l[i][d1][d2]...[dk] = -y[i][c][d1][d2]..[dk], where i is the index of classes.
or
    l[i][d1][d2]...[dk] = -y[i][c][d1][d2]..[dk] * weights[c], if 'weights' is provided.

loss is zero for the case when label-value equals ignore_index.
    l[i][d1][d2]...[dk]  = 0, when labels[n][d1][d2]...[dk] = ignore_index

where:
    p = Softmax(scores)
    y = Log(p)
    c = labels[i][d1][d2]...[dk]

Finally, L is optionally reduced:
If reduction = 'none', the output is L with shape (N, D1, D2, ..., Dk).
If reduction = 'sum', the output is scalar: Sum(L).
If reduction = 'mean', the output is scalar: ReduceMean(L), or if weight is provided: ReduceSum(L) / ReduceSum(W),
where tensor W is of shape (N, D1, D2, ..., Dk) and W[n][d1][d2]...[dk] = weights[labels[i][d1][d2]...[dk]].
)DOC";

bool BuildContextDependentFunctionBodySCE_opset12(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  std::vector<FunctionBodyHelper::NodeDef> body;

  // Using stable implementation of LogSoftmax
  body.push_back({{"Shape3D"}, "Constant", {}, {MakeAttribute("value", ToDimensionOneInt64Tensor_old({0, 0, -1}))}});
  body.push_back({{"X_NCD"}, "Reshape", {"scores", "Shape3D"}});
  body.push_back({{"X_NDC"}, "Transpose", {"X_NCD"}, {MakeAttribute("perm", std::vector<int64_t>({0, 2, 1}))}});
  body.push_back({{"X_LogSM"}, "LogSoftmax", {"X_NDC"}, {MakeAttribute("axis", (int64_t)2)}});
  body.push_back({{"X_LogSM_NCD"}, "Transpose", {"X_LogSM"}, {MakeAttribute("perm", std::vector<int64_t>({0, 2, 1}))}});
  body.push_back({{"X_shape"}, "Shape", {"scores"}});
  body.push_back({{"X_Log"}, "Reshape", {"X_LogSM_NCD", "X_shape"}});

  // Review(mzs): Ideally we want to reuse the output from Log for sub-graph
  // output as well but looking at the graph resolve code it does not include
  // graph outputs as intermediate outputs, hence if intermediate X_log is
  // renamed as log_prob then it will be treated as graph output and will not be
  // available to NegativeLogLikelihoodLoss. May be my understanding is
  // incorrect or there is a bug in function population code in ORTbut I will
  // dig further to be 100%. In the meantime we just replicate the log.
  if (ctx.hasOutput(1)) {
    body.push_back({{"log_prob"}, "Identity", {"X_Log"}});
  }

  std::vector<std::string> input_tensor_names{"X_Log", "labels"};
  std::vector<FunctionBodyHelper::AttributeProtoWrapper> attributes{
      MakeRefAttribute("reduction", AttributeProto::STRING)};
  // Add weights as input if needed.
  if (ctx.hasInput(2)) {
    input_tensor_names.push_back("weights");
  }

  // add ignore_index attributes if needed.
  if (ctx.getAttribute("ignore_index") != nullptr) {
    attributes.push_back(MakeRefAttribute("ignore_index", AttributeProto::INT));
  }

  body.push_back({{"output"}, "NegativeLogLikelihoodLoss", input_tensor_names, attributes});

  auto func_nodes = FunctionBodyHelper::BuildNodes(body);
  for (const auto& node : func_nodes) {
    auto new_node = functionProto.add_node();
    new_node->CopyFrom(node);
  }

  schema.BuildFunction(functionProto);
  return true;
}

ONNX_OPERATOR_SET_SCHEMA(
    SoftmaxCrossEntropyLoss,
    12,
    OpSchema()
        .SetDoc(SoftmaxCrossEntropyLoss_ver12_doc)
        .Attr("reduction", reduction_doc_sce_opset12, AttributeProto::STRING, std::string("mean"))
        .Attr(
            "ignore_index",
            "Specifies a target value that is ignored and does not contribute to the input gradient. It's an optional value.",
            AttributeProto::INT,
            false)
        .Input(
            0,
            "scores",
            "The predicted outputs with shape [batch_size, class_size], or "
            "[batch_size, class_size, D1, D2 , ..., Dk], where K is the number of dimensions.",
            "T")
        .Input(
            1,
            "labels",
            "The ground truth output tensor, with shape [batch_size], or "
            "[batch_size, D1, D2, ..., Dk], where K is the number of dimensions. "
            "Labels element value shall be in range of [0, C). "
            "If ignore_index is specified, it may have a value outside [0, C) and the label values should either be "
            "in the range [0, C) or have the value ignore_index.",
            "Tind")
        .Input(
            2,
            "weights",
            "A manual rescaling weight given to each class. If given, it has to "
            "be a 1D Tensor assigning weight to each of the classes. Otherwise, "
            "it is treated as if having all ones.",
            "T",
            OpSchema::Optional)
        .Output(
            0,
            "output",
            "Weighted loss float Tensor. If reduction is 'none', this has the "
            "shape of [batch_size], or [batch_size, D1, D2, ..., Dk] in case of "
            "K-dimensional loss. Otherwise, it is a scalar.",
            "T")
        .Output(
            1,
            "log_prob",
            "Log probability tensor. If the output of softmax is prob, its value is log(prob).",
            "T",
            OpSchema::Optional)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeConstraint("Tind", {"tensor(int32)", "tensor(int64)"}, "Constrain target to integer types")
        .SetContextDependentFunctionBodyBuilder(BuildContextDependentFunctionBodySCE_opset12)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          std::string reduction = getAttribute(ctx, "reduction", "mean");
          if (reduction.compare("none") == 0) {
            if (hasInputShape(ctx, 1)) {
              propagateShapeFromInputToOutput(ctx, 1, 0);
            }
          } else {
            updateOutputShape(ctx, 0, TensorShapeProto());
          }

          if (ctx.getNumOutputs() == 2) {
            propagateElemTypeFromInputToOutput(ctx, 0, 1);
            propagateShapeFromInputToOutput(ctx, 0, 1);
          }
        }));

std::function<void(OpSchema&)> SoftmaxFamilyDocGenerator_opset1(const char* name, const char* description) {
  return [=](OpSchema& schema) {
    std::string doc;
    POPULATE_OP_DOC_STR(doc = R"DOC(
The operator computes the {name} ({description}) values for each layer in the batch
 of the given input. The input is a 2-D tensor (Tensor<float>) of size
(batch_size x input_feature_dimensions). The output tensor has the same shape
and contains the {name} values of the corresponding input.

Input does not need to explicitly be a 2D vector; rather, it will be
coerced into one. For an arbitrary n-dimensional tensor
input \in [a_0, a_1, ..., a_{k-1}, a_k, ..., a_{n-1}] and k is
the axis provided, then input will be coerced into a 2-dimensional tensor with
dimensions [a_0 * ... * a_{k-1}, a_k * ... * a_{n-1}]. For the default
case where axis=1, this means the input tensor will be coerced into a 2D tensor
of dimensions [a_0, a_1 * ... * a_{n-1}], where a_0 is often the batch size.
In this situation, we must have a_0 = N and a_1 * ... * a_{n-1} = D.
Each of these dimensions must be matched correctly, or else the operator
will throw errors.
)DOC";
                        ReplaceAll(doc, "{name}", name);
                        ReplaceAll(doc, "{description}", description););
    schema.SetDoc(doc);
    schema.Attr(
        "axis",
        "Describes the axis of the inputs when coerced "
        "to 2D; defaults to one because the 0th axis most likely describes "
        "the batch_size",
        AttributeProto::INT,
        static_cast<int64_t>(1));
    schema.Input(
        0,
        "input",
        "The input tensor that's coerced into a 2D matrix of size (NxD) "
        "as described above.",
        "T");
    schema.Output(
        0,
        "output",
        "The output values with the same "
        "shape as input tensor (the original size without coercion).",
        "T");
    schema.TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");
    schema.TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput);
  };
}

ONNX_OPERATOR_SET_SCHEMA(
    Softmax,
    1,
    OpSchema().FillUsing(SoftmaxFamilyDocGenerator_opset1("softmax", "normalized exponential")));

ONNX_OPERATOR_SET_SCHEMA(
    LogSoftmax,
    1,
    OpSchema().FillUsing(SoftmaxFamilyDocGenerator_opset1("logsoftmax", "log of softmax")));

ONNX_OPERATOR_SET_SCHEMA(
    Hardmax,
    1,
    OpSchema().FillUsing(
        SoftmaxFamilyDocGenerator_opset1("hardmax", "1 for the first maximum value, and 0 for all others")));

const char* kBroadcastDoc_old = R"DOC(
If necessary the right-hand-side argument will be broadcasted to match the
shape of left-hand-side argument. When broadcasting is specified, the second
tensor can either be of element size 1 (including a scalar tensor and any
tensor with rank equal to or smaller than the first tensor), or having its
shape as a contiguous subset of the first tensor's shape. The starting of the
mutually equal shape is specified by the argument "axis", and if it is not set,
suffix matching is assumed. 1-dim expansion doesn't work yet.

For example, the following tensor shapes are supported (with broadcast=1):

  shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar tensor
  shape(A) = (2, 3, 4, 5), shape(B) = (1, 1), i.e. B is an 1-element tensor
  shape(A) = (2, 3, 4, 5), shape(B) = (5,)
  shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
  shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
  shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0

Attribute `broadcast=1` needs to be passed to enable broadcasting.
)DOC";

std::function<void(OpSchema&)> MathDocGenerator_old(const char* name) {
  return [=](OpSchema& schema) {
    std::string doc;
    POPULATE_OP_DOC_STR(doc = R"DOC(
Performs element-wise binary {name} (with limited broadcast support).
{broadcast_doc})DOC";
                        ReplaceAll(doc, "{name}", name);
                        ReplaceAll(doc, "{broadcast_doc}", kBroadcastDoc_old););
    schema.SetDoc(doc);
    schema.Attr("broadcast", "Pass 1 to enable broadcasting", AttributeProto::INT, static_cast<int64_t>(0));

    // This attribute was added via AllowConsumed API in OpSchema.
    // After removing the API, we're now using the Attr API to simulate the old
    // definition.
    schema.Attr("consumed_inputs", "legacy optimization attribute.", AttributeProto::INTS, OPTIONAL_VALUE);
    schema.Attr(
        "axis", "If set, defines the broadcast dimensions. See doc for details.", AttributeProto::INT, OPTIONAL_VALUE);
    schema.Input(0, "A", "First operand, should share the type with the second operand.", "T");
    schema.Input(
        1,
        "B",
        "Second operand. With broadcasting can be of smaller size than A. "
        "If broadcasting is disabled it should be of the same size.",
        "T");
    schema.Output(0, "C", "Result, has same dimensions and type as A", "T");
    schema.TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");
  };
}

std::function<void(OpSchema&)> MathDocGenerator_old_opset6(const char* name) {
  return [=](OpSchema& schema) {
    std::string doc;
    POPULATE_OP_DOC_STR(doc = R"DOC(
Performs element-wise binary {name} (with limited broadcast support).
{broadcast_doc})DOC";
                        ReplaceAll(doc, "{name}", name);
                        ReplaceAll(doc, "{broadcast_doc}", kBroadcastDoc_old););
    schema.SetDoc(doc);
    schema.Attr("broadcast", "Pass 1 to enable broadcasting", AttributeProto::INT, static_cast<int64_t>(0));
    schema.Attr(
        "axis", "If set, defines the broadcast dimensions. See doc for details.", AttributeProto::INT, OPTIONAL_VALUE);
    schema.Input(0, "A", "First operand, should share the type with the second operand.", "T");
    schema.Input(
        1,
        "B",
        "Second operand. With broadcasting can be of smaller size than A. "
        "If broadcasting is disabled it should be of the same size.",
        "T");
    schema.Output(0, "C", "Result, has same dimensions and type as A", "T");
    schema.TypeConstraint(
        "T",
        OpSchema::numeric_types_for_math_reduction(),
        "Constrain input and output types to high-precision numeric tensors.");
    schema.TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput);
  };
}

ONNX_OPERATOR_SET_SCHEMA(Add, 1, OpSchema().FillUsing(MathDocGenerator_old("addition")));

ONNX_OPERATOR_SET_SCHEMA(Sub, 1, OpSchema().FillUsing(MathDocGenerator_old("subtraction")));

ONNX_OPERATOR_SET_SCHEMA(Mul, 1, OpSchema().FillUsing(MathDocGenerator_old("multiplication")));

ONNX_OPERATOR_SET_SCHEMA(Div, 1, OpSchema().FillUsing(MathDocGenerator_old("division")));

ONNX_OPERATOR_SET_SCHEMA(Add, 6, OpSchema().FillUsing(MathDocGenerator_old_opset6("addition")));

ONNX_OPERATOR_SET_SCHEMA(Sub, 6, OpSchema().FillUsing(MathDocGenerator_old_opset6("subtraction")));

ONNX_OPERATOR_SET_SCHEMA(Mul, 6, OpSchema().FillUsing(MathDocGenerator_old_opset6("multiplication")));

ONNX_OPERATOR_SET_SCHEMA(Div, 6, OpSchema().FillUsing(MathDocGenerator_old_opset6("division")));

static const char* Pow_ver1_doc = R"DOC(
Pow takes input data (Tensor<T>) and exponent Tensor, and
produces one output data (Tensor<T>) where the function `f(x) = x^exponent`,
is applied to the data tensor elementwise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Pow,
    1,
    OpSchema()
        .SetDoc(Pow_ver1_doc + std::string(kBroadcastDoc_old))
        .Input(0, "X", "Input tensor of any shape, base of the exponent.", "T")
        .Input(
            1,
            "Y",
            "Input tensor of any shape broadcastable to X shape, "
            "the exponent component.",
            "T")
        .Attr("broadcast", "Pass 1 to enable broadcasting", AttributeProto::INT, static_cast<int64_t>(0))
        .Attr(
            "axis",
            "If set, defines the broadcast dimensions. See doc for details.",
            AttributeProto::INT,
            OPTIONAL_VALUE)
        .Output(0, "Z", "Output tensor (same size as X)", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Pow_ver7_doc = R"DOC(
Pow takes input data (Tensor<T>) and exponent Tensor, and
produces one output data (Tensor<T>) where the function `f(x) = x^exponent`,
is applied to the data tensor elementwise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Pow,
    7,
    OpSchema()
        .SetDoc(std::string(Pow_ver7_doc) + GenerateBroadcastingDocMul())
        .Input(0, "X", "First operand, base of the exponent.", "T")
        .Input(1, "Y", "Second operand, power of the exponent.", "T")
        .Output(0, "Z", "Output tensor.", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (hasNInputShapes(ctx, 2))
            bidirectionalBroadcastShapeInference(
                ctx.getInputType(0)->tensor_type().shape(),
                ctx.getInputType(1)->tensor_type().shape(),
                *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
        }));

static const char* Neg_ver1_doc = R"DOC(
Neg takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where each element flipped sign, y = -x, is applied to
the tensor elementwise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Neg,
    1,
    OpSchema()
        .SetDoc(Neg_ver1_doc)
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        // This attribute was added via AllowConsumed API in OpSchema.
        // After removing the API, we're now using the Attr API to simulate the
        // old definition.
        .Attr("consumed_inputs", "legacy optimization attribute.", AttributeProto::INTS, OPTIONAL_VALUE)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors."));

static const char* Abs_ver1_doc = R"DOC(
Absolute takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the absolute is, y = abs(x), is applied to
the tensor elementwise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Abs,
    1,
    OpSchema()
        .SetDoc(Abs_ver1_doc)
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        // This attribute was added via AllowConsumed API in OpSchema.
        // After removing the API, we're now using the Attr API to simulate the
        // old definition.
        .Attr("consumed_inputs", "legacy optimization attribute.", AttributeProto::INTS, OPTIONAL_VALUE)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors."));

static const char* Reciprocal_ver1_doc = R"DOC(
Reciprocal takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the reciprocal is, y = 1/x, is applied to
the tensor elementwise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Reciprocal,
    1,
    OpSchema()
        .SetDoc(Reciprocal_ver1_doc)
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        // This attribute was added via AllowConsumed API in OpSchema.
        // After removing the API, we're now using the Attr API to simulate the
        // old definition.
        .Attr("consumed_inputs", "legacy optimization attribute.", AttributeProto::INTS, OPTIONAL_VALUE)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors."));

static const char* Floor_ver1_doc = R"DOC(
Floor takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the floor is, y = floor(x), is applied to
the tensor elementwise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Floor,
    1,
    OpSchema()
        .SetDoc(Floor_ver1_doc)
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        // This attribute was added via AllowConsumed API in OpSchema.
        // After removing the API, we're now using the Attr API to simulate the
        // old definition.
        .Attr("consumed_inputs", "legacy optimization attribute.", AttributeProto::INTS, OPTIONAL_VALUE)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors."));

static const char* Ceil_ver1_doc = R"DOC(
Ceil takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the ceil is, y = ceil(x), is applied to
the tensor elementwise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Ceil,
    1,
    OpSchema()
        .SetDoc(Ceil_ver1_doc)
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        // This attribute was added via AllowConsumed API in OpSchema.
        // After removing the API, we're now using the Attr API to simulate the
        // old definition.
        .Attr("consumed_inputs", "legacy optimization attribute.", AttributeProto::INTS, OPTIONAL_VALUE)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors."));

static const char* Sqrt_ver1_doc = R"DOC(
Square root takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the square root is, y = x^0.5, is applied to
the tensor elementwise. If x is negative, then it will return NaN.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Sqrt,
    1,
    OpSchema()
        .SetDoc(Sqrt_ver1_doc)
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        // This attribute was added via AllowConsumed API in OpSchema.
        // After removing the API, we're now using the Attr API to simulate the
        // old definition.
        .Attr("consumed_inputs", "legacy optimization attribute.", AttributeProto::INTS, OPTIONAL_VALUE)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors."));

static const char* Relu_ver1_doc = R"DOC(
Relu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the rectified linear function, y = max(0, x), is applied to
the tensor elementwise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Relu,
    1,
    OpSchema()
        .SetDoc(Relu_ver1_doc)
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        // This attribute was added via AllowConsumed API in OpSchema.
        // After removing the API, we're now using the Attr API to simulate the
        // old definition.
        .Attr("consumed_inputs", "legacy optimization attribute.", AttributeProto::INTS, OPTIONAL_VALUE)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors."));

static const char* LeakyRelu_ver1_doc = R"DOC(
LeakyRelu takes input data (Tensor<T>) and an argument alpha, and produces one
output data (Tensor<T>) where the function `f(x) = alpha * x for x < 0`,
`f(x) = x for x >= 0`, is applied to the data tensor elementwise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    LeakyRelu,
    1,
    OpSchema()
        .Attr("alpha", "Coefficient of leakage default to 0.01.", AttributeProto::FLOAT, 0.01f)
        .SetDoc(LeakyRelu_ver1_doc)
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        // This attribute was added via AllowConsumed API in OpSchema.
        // After removing the API, we're now using the Attr API to simulate the
        // old definition.
        .Attr("consumed_inputs", "legacy optimization attribute.", AttributeProto::INTS, OPTIONAL_VALUE)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors."));

static const char* Selu_ver1_doc = R"DOC(
Selu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the scaled exponential linear unit function,
`y = gamma * (alpha * e^x - alpha) for x <= 0`, `y = gamma * x for x > 0`,
is applied to the tensor elementwise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Selu,
    1,
    OpSchema()
        .Attr("alpha", "Coefficient of SELU default to 1.6732.", AttributeProto::FLOAT, 1.6732f)
        .Attr("gamma", "Coefficient of SELU default to 1.0507.", AttributeProto::FLOAT, 1.0507f)
        // This attribute was added via AllowConsumed API in OpSchema.
        // After removing the API, we're now using the Attr API to simulate the
        // old definition.
        .Attr("consumed_inputs", "legacy optimization attribute.", AttributeProto::INTS, OPTIONAL_VALUE)
        .SetDoc(Selu_ver1_doc)
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors."));

static const char* Elu_ver1_doc = R"DOC(
Elu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the function `f(x) = alpha * (exp(x) - 1.) for x <
0`, `f(x) = x for x >= 0`., is applied to the tensor elementwise.

)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Elu,
    1,
    OpSchema()
        .Attr("alpha", "Coefficient of ELU default to 1.0.", AttributeProto::FLOAT, 1.0f)
        // This attribute was added via AllowConsumed API in OpSchema.
        // After removing the API, we're now using the Attr API to simulate the
        // old definition.
        .Attr("consumed_inputs", "legacy optimization attribute.", AttributeProto::INTS, OPTIONAL_VALUE)
        .SetDoc(Elu_ver1_doc)
        .Input(0, "X", "1D input tensor", "T")
        .Output(0, "Y", "1D input tensor", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors."));

static const char* Exp_ver1_doc = R"DOC(
Calculates the exponential of the given input tensor, element-wise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Exp,
    1,
    OpSchema()
        .SetDoc(Exp_ver1_doc)
        .Input(0, "input", "Input tensor", "T")
        .Output(
            0,
            "output",
            "The exponential of the input tensor computed "
            "element-wise",
            "T")
        // This attribute was added via AllowConsumed API in OpSchema.
        // After removing the API, we're now using the Attr API to simulate the
        // old definition.
        .Attr("consumed_inputs", "legacy optimization attribute.", AttributeProto::INTS, OPTIONAL_VALUE)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors."));

static const char* Log_ver1_doc = R"DOC(
Calculates the natural log of the given input tensor, element-wise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Log,
    1,
    OpSchema()
        .SetDoc(Log_ver1_doc)
        .Input(0, "input", "Input tensor", "T")
        .Output(
            0,
            "output",
            "The natural log of the input tensor computed "
            "element-wise",
            "T")
        // This attribute was added via AllowConsumed API in OpSchema.
        // After removing the API, we're now using the Attr API to simulate the
        // old definition.
        .Attr("consumed_inputs", "legacy optimization attribute.", AttributeProto::INTS, OPTIONAL_VALUE)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors."));

static const char* Tanh_ver1_doc = R"DOC(
Calculates the hyperbolic tangent of the given input tensor element-wise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Tanh,
    1,
    OpSchema()
        .SetDoc(Tanh_ver1_doc)
        .Input(0, "input", "1-D input tensor", "T")
        .Output(
            0,
            "output",
            "The hyperbolic tangent values of the input tensor "
            "computed element-wise",
            "T")
        // This attribute was added via AllowConsumed API in OpSchema.
        // After removing the API, we're now using the Attr API to simulate the
        // old definition.
        .Attr("consumed_inputs", "legacy optimization attribute.", AttributeProto::INTS, OPTIONAL_VALUE)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors."));

static const char* PRelu_ver1_doc = R"DOC(

PRelu takes input data (Tensor<T>) and slope tensor as input, and produces one
output data (Tensor<T>) where the function `f(x) = slope * x for x < 0`,
`f(x) = x for x >= 0`., is applied to the data tensor elementwise.

)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    PRelu,
    1,
    OpSchema()
        .SetDoc(PRelu_ver1_doc)
        .Input(0, "X", "Input tensor", "T")
        .Input(
            1,
            "slope",
            "Slope tensor. If `Slope` is of size 1, the value is shared"
            "across different channels",
            "T")
        .Output(0, "Y", "Output tensor", "T")
        // This attribute was added via AllowConsumed API in OpSchema.
        // After removing the API, we're now using the Attr API to simulate the
        // old definition.
        .Attr("consumed_inputs", "legacy optimization attribute.", AttributeProto::INTS, OPTIONAL_VALUE)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors."));

ONNX_OPERATOR_SET_SCHEMA(
    PRelu,
    6,
    OpSchema()
        .SetDoc(PRelu_ver1_doc)
        .Input(0, "X", "Input tensor", "T")
        .Input(
            1,
            "slope",
            "Slope tensor. If `Slope` is of size 1, the value is shared"
            "across different channels",
            "T")
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* PRelu_ver7_doc = R"DOC(
PRelu takes input data (Tensor<T>) and slope tensor as input, and produces one
output data (Tensor<T>) where the function `f(x) = slope * x for x < 0`,
`f(x) = x for x >= 0`., is applied to the data tensor elementwise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    PRelu,
    7,
    OpSchema()
        .SetDoc(
            GET_OP_DOC_STR(std::string(PRelu_ver7_doc) + GenerateBroadcastingDocUni("tensor slope", "input tensor X")))
        .Input(0, "X", "Input tensor", "T")
        .Input(
            1,
            "slope",
            "Slope tensor. The shape of slope can be smaller then first input X; "
            "if so, its shape must be unidirectional broadcastable to X",
            "T")
        .Output(0, "Y", "Output tensor (same size as X)", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Sigmoid_ver1_doc = R"DOC(
Sigmoid takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the sigmoid function, y = 1 / (1 + exp(-x)), is applied to the
tensor elementwise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Sigmoid,
    1,
    OpSchema()
        .SetDoc(Sigmoid_ver1_doc)
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        // This attribute was added via AllowConsumed API in OpSchema.
        // After removing the API, we're now using the Attr API to simulate the
        // old definition.
        .Attr("consumed_inputs", "legacy optimization attribute.", AttributeProto::INTS, OPTIONAL_VALUE)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors."));

static const char* HardSigmoid_ver1_doc = R"DOC(
HardSigmoid takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the HardSigmoid function, y = max(0, min(1, alpha * x + beta)),
is applied to the tensor elementwise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    HardSigmoid,
    1,
    OpSchema()
        .Attr("alpha", "Value of alpha default to 0.2", AttributeProto::FLOAT, 0.2f)
        .Attr("beta", "Value of beta default to 0.5", AttributeProto::FLOAT, 0.5f)
        // This attribute was added via AllowConsumed API in OpSchema.
        // After removing the API, we're now using the Attr API to simulate the
        // old definition.
        .Attr("consumed_inputs", "legacy optimization attribute.", AttributeProto::INTS, OPTIONAL_VALUE)
        .SetDoc(HardSigmoid_ver1_doc)
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors."));

static const char* Max_ver1_doc = R"DOC(
Element-wise max of each of the input tensors. All inputs and outputs must
have the same shape and data type.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Max,
    1,
    OpSchema()
        .SetDoc(Max_ver1_doc)
        .Input(0, "data_0", "List of tensors for Max.", "T", OpSchema::Variadic)
        .Output(0, "max", "Output tensor. Same dimension as inputs.", "T")
        // This attribute was added via AllowConsumed API in OpSchema.
        // After removing the API, we're now using the Attr API to simulate the
        // old definition.
        .Attr("consumed_inputs", "legacy optimization attribute.", AttributeProto::INTS, OPTIONAL_VALUE)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors."));

static const char* Min_ver1_doc = R"DOC(
Element-wise min of each of the input tensors. All inputs and outputs must
have the same shape and data type.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Min,
    1,
    OpSchema()
        .SetDoc(Min_ver1_doc)
        .Input(0, "data_0", "List of tensors for Min", "T", OpSchema::Variadic)
        .Output(0, "min", "Output tensor. Same dimension as inputs.", "T")
        // This attribute was added via AllowConsumed API in OpSchema.
        // After removing the API, we're now using the Attr API to simulate the
        // old definition.
        .Attr("consumed_inputs", "legacy optimization attribute.", AttributeProto::INTS, OPTIONAL_VALUE)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors."));

static const char* Sum_ver1_doc = R"DOC(
Element-wise sum of each of the input tensors. All inputs and outputs must
have the same shape and data type.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Sum,
    1,
    OpSchema()
        .SetDoc(Sum_ver1_doc)
        .Input(0, "data_0", "List of tensors for Sum.", "T", OpSchema::Variadic)
        .Output(0, "sum", "Output tensor. Same dimension as inputs.", "T")
        // This attribute was added via AllowConsumed API in OpSchema.
        // After removing the API, we're now using the Attr API to simulate the
        // old definition.
        .Attr("consumed_inputs", "legacy optimization attribute.", AttributeProto::INTS, OPTIONAL_VALUE)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors."));

static const char* Mean_ver1_doc = R"DOC(
Element-wise mean of each of the input tensors. All inputs and outputs must
have the same shape and data type.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Mean,
    1,
    OpSchema()
        .SetDoc(Mean_ver1_doc)
        .Input(0, "data_0", "List of tensors for Mean.", "T", OpSchema::Variadic)
        .Output(0, "mean", "Output tensor. Same dimension as inputs.", "T")
        // This attribute was added via AllowConsumed API in OpSchema.
        // After removing the API, we're now using the Attr API to simulate the
        // old definition.
        .Attr("consumed_inputs", "legacy optimization attribute.", AttributeProto::INTS, OPTIONAL_VALUE)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors."));

static const char* Clip_ver1_doc = R"DOC(
Clip operator limits the given input within an interval. The interval is
specified with arguments 'min' and 'max'. They default to
numeric_limits::lowest() and numeric_limits::max() respectively.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Clip,
    1,
    OpSchema()
        .SetDoc(Clip_ver1_doc)
        .Attr("min", "Minimum value, under which element is replaced by min", AttributeProto::FLOAT, OPTIONAL_VALUE)
        .Attr("max", "Maximum value, above which element is replaced by max", AttributeProto::FLOAT, OPTIONAL_VALUE)
        // This attribute was added via AllowConsumed API in OpSchema.
        // After removing the API, we're now using the Attr API to simulate the
        // old definition.
        .Attr("consumed_inputs", "legacy optimization attribute.", AttributeProto::INTS, OPTIONAL_VALUE)
        .Input(0, "input", "Input tensor whose elements to be clipped", "T")
        .Output(0, "output", "Output tensor with clipped input elements", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors."));

static const char* Gemm_ver1_doc = R"DOC(General Matrix multiplication:
https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3
Compute Y = alpha * A * B + beta * C, where input tensor A has
dimension (M X K), input tensor B has dimension (K X N), input tensor C and
output tensor Y have dimension (M X N).
If attribute broadcast is non-zero, input tensor C will be broadcasted to match
the dimension requirement. A will be transposed before doing the computation
if attribute transA is non-zero, same for B and transB.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Gemm,
    1,
    OpSchema()
        .SetDoc(Gemm_ver1_doc)
        .Input(0, "A", "Input tensor A", "T")
        .Input(1, "B", "Input tensor B", "T")
        .Input(2, "C", "Input tensor C, can be inplace.", "T")
        .Output(0, "Y", "Output tensor.", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        // This attribute was added via AllowConsumed API in OpSchema.
        // After removing the API, we're now using the Attr API to simulate the
        // old definition.
        .Attr("transA", "Whether A should be transposed", AttributeProto::INT, static_cast<int64_t>(0))
        .Attr("transB", "Whether B should be transposed", AttributeProto::INT, static_cast<int64_t>(0))
        .Attr("broadcast", "Whether C should be broadcasted", AttributeProto::INT, static_cast<int64_t>(0))
        .Attr(
            "alpha",
            "Scalar multiplier for the product of input tensors A * B, the default value is 1.0.",
            AttributeProto::FLOAT,
            1.0f)
        .Attr("beta", "Scalar multiplier for input tensor C, the default value is 1.0.", AttributeProto::FLOAT, 1.0f));

static const char* Gemm_ver6_doc = R"DOC(General Matrix multiplication:
https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3
Compute Y = alpha * A * B + beta * C, where input tensor A has
dimension (M X K), input tensor B has dimension (K X N), input tensor C and
output tensor Y have dimension (M X N).
If attribute broadcast is non-zero, input tensor C will be broadcasted to match
the dimension requirement. A will be transposed before doing the computation
if attribute transA is non-zero, same for B and transB.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Gemm,
    6,
    OpSchema()
        .SetDoc(Gemm_ver6_doc)
        .Input(0, "A", "Input tensor A", "T")
        .Input(1, "B", "Input tensor B", "T")
        .Input(2, "C", "Input tensor C", "T")
        .Output(0, "Y", "Output tensor.", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .Attr("transA", "Whether A should be transposed", AttributeProto::INT, static_cast<int64_t>(0))
        .Attr("transB", "Whether B should be transposed", AttributeProto::INT, static_cast<int64_t>(0))
        .Attr("broadcast", "Whether C should be broadcasted", AttributeProto::INT, static_cast<int64_t>(0))
        .Attr(
            "alpha",
            "Scalar multiplier for the product of input tensors A * B, the default value is 1.0.",
            AttributeProto::FLOAT,
            1.0f)
        .Attr("beta", "Scalar multiplier for input tensor C, the default value is 1.0.", AttributeProto::FLOAT, 1.0f)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (hasNInputShapes(ctx, 2)) {
            auto transAAttr = ctx.getAttribute("transA");
            bool transA = transAAttr ? static_cast<int>(transAAttr->i()) != 0 : false;
            auto transBAttr = ctx.getAttribute("transB");
            bool transB = transBAttr ? static_cast<int>(transBAttr->i()) != 0 : false;

            *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim() =
                ctx.getInputType(0)->tensor_type().shape().dim(transA ? 1 : 0);
            *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim() =
                ctx.getInputType(1)->tensor_type().shape().dim(transB ? 0 : 1);
          } else if (
              hasInputShape(ctx, 2) &&
              (!ctx.getAttribute("broadcast") || static_cast<int>(ctx.getAttribute("broadcast")->i()) == 0)) {
            *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape() = ctx.getInputType(2)->tensor_type().shape();
          }
        }));

static const char* Gemm_ver7_doc = R"DOC(General Matrix multiplication:
https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3

A' = transpose(A) if transA else A

B' = transpose(B) if transB else B

Compute Y = alpha * A' * B' + beta * C, where input tensor A has shape (M, K) or (K, M),
input tensor B has shape (K, N) or (N, K), input tensor C is broadcastable to shape (M, N),
and output tensor Y has shape (M, N). A will be transposed before doing the
computation if attribute transA is non-zero, same for B and transB.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Gemm,
    7,
    OpSchema()
        .SetDoc(GET_OP_DOC_STR(std::string(Gemm_ver7_doc) + GenerateBroadcastingDocUni("tensor C", "tensor A * B")))
        .Input(
            0,
            "A",
            "Input tensor A. "
            "The shape of A should be (M, K) if transA is 0, "
            "or (K, M) if transA is non-zero.",
            "T")
        .Input(
            1,
            "B",
            "Input tensor B. "
            "The shape of B should be (K, N) if transB is 0, "
            "or (N, K) if transB is non-zero.",
            "T")
        .Input(
            2,
            "C",
            "Input tensor C. "
            "The shape of C should be unidirectional broadcastable to (M, N).",
            "T")
        .Output(0, "Y", "Output tensor of shape (M, N).", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .Attr("transA", "Whether A should be transposed", AttributeProto::INT, static_cast<int64_t>(0))
        .Attr("transB", "Whether B should be transposed", AttributeProto::INT, static_cast<int64_t>(0))
        .Attr("alpha", "Scalar multiplier for the product of input tensors A * B.", AttributeProto::FLOAT, 1.0f)
        .Attr("beta", "Scalar multiplier for input tensor C.", AttributeProto::FLOAT, 1.0f)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (hasNInputShapes(ctx, 2)) {
            auto transAAttr = ctx.getAttribute("transA");
            bool transA = transAAttr ? static_cast<int>(transAAttr->i()) != 0 : false;
            auto transBAttr = ctx.getAttribute("transB");
            bool transB = transBAttr ? static_cast<int>(transBAttr->i()) != 0 : false;
            auto& first_input_shape = getInputShape(ctx, 0);
            auto& second_input_shape = getInputShape(ctx, 1);
            if (first_input_shape.dim_size() != 2) {
              fail_shape_inference("First input does not have rank 2");
            }
            if (second_input_shape.dim_size() != 2) {
              fail_shape_inference("Second input does not have rank 2");
            }
            updateOutputShape(ctx, 0, {first_input_shape.dim(transA ? 1 : 0), second_input_shape.dim(transB ? 0 : 1)});
          }
        }));

static const char* Gemm_ver9_doc = R"DOC(General Matrix multiplication:
https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3

A' = transpose(A) if transA else A

B' = transpose(B) if transB else B

Compute Y = alpha * A' * B' + beta * C, where input tensor A has shape (M, K) or (K, M),
input tensor B has shape (K, N) or (N, K), input tensor C is broadcastable to shape (M, N),
and output tensor Y has shape (M, N). A will be transposed before doing the
computation if attribute transA is non-zero, same for B and transB.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Gemm,
    9,
    OpSchema()
        .SetDoc(GET_OP_DOC_STR(std::string(Gemm_ver9_doc) + GenerateBroadcastingDocUni("tensor C", "tensor A * B")))
        .Input(
            0,
            "A",
            "Input tensor A. "
            "The shape of A should be (M, K) if transA is 0, "
            "or (K, M) if transA is non-zero.",
            "T")
        .Input(
            1,
            "B",
            "Input tensor B. "
            "The shape of B should be (K, N) if transB is 0, "
            "or (N, K) if transB is non-zero.",
            "T")
        .Input(
            2,
            "C",
            "Input tensor C. "
            "The shape of C should be unidirectional broadcastable to (M, N).",
            "T")
        .Output(0, "Y", "Output tensor of shape (M, N).", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)",
             "tensor(float)",
             "tensor(double)",
             "tensor(uint32)",
             "tensor(uint64)",
             "tensor(int32)",
             "tensor(int64)"},
            "Constrain input and output types to float/int tensors.")
        .Attr("transA", "Whether A should be transposed", AttributeProto::INT, static_cast<int64_t>(0))
        .Attr("transB", "Whether B should be transposed", AttributeProto::INT, static_cast<int64_t>(0))
        .Attr("alpha", "Scalar multiplier for the product of input tensors A * B.", AttributeProto::FLOAT, 1.0f)
        .Attr("beta", "Scalar multiplier for input tensor C.", AttributeProto::FLOAT, 1.0f)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (hasNInputShapes(ctx, 2)) {
            auto transAAttr = ctx.getAttribute("transA");
            bool transA = transAAttr ? static_cast<int>(transAAttr->i()) != 0 : false;
            auto transBAttr = ctx.getAttribute("transB");
            bool transB = transBAttr ? static_cast<int>(transBAttr->i()) != 0 : false;
            auto& first_input_shape = getInputShape(ctx, 0);
            auto& second_input_shape = getInputShape(ctx, 1);
            if (first_input_shape.dim_size() != 2) {
              fail_shape_inference("First input does not have rank 2");
            }
            if (second_input_shape.dim_size() != 2) {
              fail_shape_inference("Second input does not have rank 2");
            }
            updateOutputShape(ctx, 0, {first_input_shape.dim(transA ? 1 : 0), second_input_shape.dim(transB ? 0 : 1)});
          }
        }));

static const char* Max_ver6_doc = R"DOC(
Element-wise max of each of the input tensors. All inputs and outputs must
have the same shape and data type.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Max,
    6,
    OpSchema()
        .SetDoc(Max_ver6_doc)
        .Input(0, "data_0", "List of tensors for Max.", "T", OpSchema::Variadic)
        .Output(0, "max", "Output tensor. Same dimension as inputs.", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Min_ver6_doc = R"DOC(
Element-wise min of each of the input tensors. All inputs and outputs must
have the same shape and data type.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Min,
    6,
    OpSchema()
        .SetDoc(Min_ver6_doc)
        .Input(0, "data_0", "List of tensors for Min", "T", OpSchema::Variadic)
        .Output(0, "min", "Output tensor. Same dimension as inputs.", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Sum_ver6_doc = R"DOC(
Element-wise sum of each of the input tensors. All inputs and outputs must
have the same shape and data type.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Sum,
    6,
    OpSchema()
        .SetDoc(Sum_ver6_doc)
        .Input(0, "data_0", "List of tensors for Sum.", "T", OpSchema::Variadic)
        .Output(0, "sum", "Output tensor. Same dimension as inputs.", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Mean_ver6_doc = R"DOC(
Element-wise mean of each of the input tensors. All inputs and outputs must
have the same shape and data type.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Mean,
    6,
    OpSchema()
        .SetDoc(Mean_ver6_doc)
        .Input(0, "data_0", "List of tensors for Mean.", "T", OpSchema::Variadic)
        .Output(0, "mean", "Output tensor. Same dimension as inputs.", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* MatMul_ver1_doc = R"DOC(
Matrix product that behaves like numpy.matmul: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    MatMul,
    1,
    OpSchema()
        .Input(0, "A", "N-dimensional matrix A", "T")
        .Input(1, "B", "N-dimensional matrix B", "T")
        .Output(0, "Y", "Matrix multiply results from A * B", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .SetDoc(MatMul_ver1_doc)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (!hasNInputShapes(ctx, 2)) {
            return;
          }

          const auto shape0 = ctx.getInputType(0)->tensor_type().shape();
          const auto shape1 = ctx.getInputType(1)->tensor_type().shape();

          if (shape0.dim_size() == 0 || shape1.dim_size() == 0) {
            fail_shape_inference("Input tensors of wrong rank (0).");
          }

          TensorShapeProto shapeL, shapeR;

          // First promote each shape to at least rank-2. This logic is
          // specific to matmul, not generic broadcasting.
          {
            if (shape0.dim_size() == 1) {
              shapeL.add_dim()->set_dim_value(1);
              *shapeL.add_dim() = shape0.dim(0);
            } else {
              *shapeL.mutable_dim() = shape0.dim();
            }
            if (shape1.dim_size() == 1) {
              *shapeR.add_dim() = shape1.dim(0);
              shapeR.add_dim()->set_dim_value(1);
            } else {
              *shapeR.mutable_dim() = shape1.dim();
            }
          }

          // Check for compatible matrix multiply dimensions
          {
            auto dimL = shapeL.dim(shapeL.dim_size() - 1);
            auto dimR = shapeR.dim(shapeR.dim_size() - 2);
            if (dimL.has_dim_value() && dimR.has_dim_value() && dimL.dim_value() != dimR.dim_value()) {
              fail_shape_inference("Incompatible dimensions for matrix multiplication");
              ;
            }
          }

          TensorShapeProto resultShape;

          // Now call out to generic multidimensional broadcasting for
          // the broadcastable prefixes.
          {
            TensorShapeProto prefixShapeL, prefixShapeR;
            for (int i = 0; i < shapeL.dim_size() - 2; ++i) {
              *prefixShapeL.add_dim() = shapeL.dim(i);
            }
            for (int i = 0; i < shapeR.dim_size() - 2; ++i) {
              *prefixShapeR.add_dim() = shapeR.dim(i);
            }
            bidirectionalBroadcastShapeInference(prefixShapeL, prefixShapeR, resultShape);
          }

          // Back to matmul-specific. Add the trailing dimensions back in.
          {
            if (shape0.dim_size() != 1) {
              *resultShape.add_dim() = shapeL.dim(shapeL.dim_size() - 2);
            }
            if (shape1.dim_size() != 1) {
              *resultShape.add_dim() = shapeR.dim(shapeR.dim_size() - 1);
            }
          }

          *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape() = resultShape;
        }));

static const char* TopK_ver1_doc = R"DOC(
Retrieve the top-K elements along a specified axis. Given an input tensor of
shape [a_1, a_2, ..., a_n, r] and integer argument k, return two outputs:
  -Value tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n]
    which contains the values of the top k elements along the specified axis
  -Index tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n] which
   contains the indices of the top k elements (original indices from the input
   tensor).
Given two equivalent values, this operator uses the indices along the axis  as
 a tiebreaker. That is, the element with the lower index will appear first.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    TopK,
    1,
    OpSchema()
        .SetDoc(TopK_ver1_doc)
        .Input(0, "X", "Tensor of shape [a_1, a_2, ..., a_n, r]", "T")
        .Output(
            0,
            "Values",
            "Tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n] "
            "containing top K values from the input tensor",
            "T")
        .Output(
            1,
            "Indices",
            "Tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n] "
            "containing the corresponding input tensor indices for the top K "
            "values.",
            "I")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeConstraint("I", {"tensor(int64)"}, "Constrain index tensor to int64")
        .Attr("k", "Number of top elements to retrieve", AttributeProto::INT, true)
        .Attr("axis", "Dimension on which to do the sort.", AttributeProto::INT, static_cast<int64_t>(-1))
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          // Type inference:
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          updateOutputElemType(ctx, 1, TensorProto::INT64);

          // Shape inference:
          if (!hasInputShape(ctx, 0))
            return;
          auto& input_shape = getInputShape(ctx, 0);
          int64_t rank = input_shape.dim_size();
          int64_t axis = getAttribute(ctx, "axis", -1);
          if (axis < 0)
            axis += rank;
          if (axis < 0 || axis >= rank) {
            fail_shape_inference("Invalid value for attribute axis");
          }
          int64_t k = getAttribute(ctx, "k", -1);
          if (k <= 0) {
            fail_shape_inference("Invalid value for attribute k");
          }
          // TODO: unclear what results should be if axis has less than k
          // elements.
          TensorShapeProto result_shape = input_shape;
          result_shape.mutable_dim(static_cast<int>(axis))->set_dim_value(k);
          updateOutputShape(ctx, 0, result_shape);
          updateOutputShape(ctx, 1, result_shape);
        }));

static const char* TopK_ver10_doc = R"DOC(
Retrieve the top-K elements along a specified axis. Given an input tensor of
shape [a_1, a_2, ..., a_n, r] and integer argument k, return two outputs:
  -Value tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n]
    which contains the values of the top k elements along the specified axis
  -Index tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n] which
   contains the indices of the top k elements (original indices from the input
   tensor).

Given two equivalent values, this operator uses the indices along the axis  as
 a tiebreaker. That is, the element with the lower index will appear first.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    TopK,
    10,
    OpSchema()
        .SetDoc(TopK_ver10_doc)
        .Input(0, "X", "Tensor of shape [a_1, a_2, ..., a_n, r]", "T")
        .Input(
            1,
            "K",
            "A 1-D tensor containing a single positive value corresponding to the number of top elements to retrieve",
            "tensor(int64)")
        .Output(
            0,
            "Values",
            "Tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n] "
            "containing top K values from the input tensor",
            "T")
        .Output(
            1,
            "Indices",
            "Tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n] "
            "containing the corresponding input tensor indices for the top K "
            "values.",
            "I")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeConstraint("I", {"tensor(int64)"}, "Constrain index tensor to int64")
        .Attr("axis", "Dimension on which to do the sort.", AttributeProto::INT, static_cast<int64_t>(-1))
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          // Type inference:
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          updateOutputElemType(ctx, 1, TensorProto::INT64);
          // Shape inference:
          if (!hasInputShape(ctx, 0))
            return;
          auto& input_shape = getInputShape(ctx, 0);
          int64_t rank = input_shape.dim_size();
          int64_t axis = getAttribute(ctx, "axis", -1);
          if (axis < 0)
            axis += rank;
          if (axis < 0 || axis >= rank) {
            fail_shape_inference("Invalid value for attribute axis");
          }

          const auto& axis_dim = input_shape.dim(static_cast<int>(axis));
          const auto* k = ctx.getInputData(1);

          // Infer output shape if:
          // (1) 'K' is available
          // (2) axis_dim has dim value
          // Othewise cannot reliably compute output shape as axis dim value is
          // unknown and hence cannot determine if axis dim value >= k (which
          // should be enforced)
          if (nullptr != k && axis_dim.has_dim_value()) {
            int64_t k_value = 0;
            if (k->dims_size() != 1 || k->dims(0) != 1) {
              fail_shape_inference("K input must be a one-dimensional tensor of size 1.");
            }

            if (k->data_type() == TensorProto::INT64) {
              const auto& data = ParseData<int64_t>(k);
              k_value = data[0];
            } else {
              fail_shape_inference("K input must be of type int64.");
            }

            if (axis_dim.dim_value() < k_value) {
              fail_shape_inference("Axis has less than the requested k elements.");
            }

            TensorShapeProto result_shape = input_shape;
            result_shape.mutable_dim(static_cast<int>(axis))->set_dim_value(k_value);

            updateOutputShape(ctx, 0, result_shape);
            updateOutputShape(ctx, 1, result_shape);

            return;
          }

          // Infer output shapes' rank in any case
          auto* output_shape_0 = getOutputShape(ctx, 0);
          auto* output_shape_1 = getOutputShape(ctx, 1);
          for (int i = 0; i < input_shape.dim_size(); ++i) {
            output_shape_0->add_dim();
            output_shape_1->add_dim();
          }

          return;
        }));

static const char* Clip_ver6_doc = R"DOC(
Clip operator limits the given input within an interval. The interval is
specified with arguments 'min' and 'max'. They default to
numeric_limits::lowest() and numeric_limits::max() respectively.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Clip,
    6,
    OpSchema()
        .SetDoc(Clip_ver6_doc)
        .Attr(
            "min",
            "Minimum value, under which element is replaced by min",
            AttributeProto::FLOAT,
            std::numeric_limits<float>::lowest())
        .Attr(
            "max",
            "Maximum value, above which element is replaced by max",
            AttributeProto::FLOAT,
            std::numeric_limits<float>::max())
        .Input(0, "input", "Input tensor whose elements to be clipped", "T")
        .Output(0, "output", "Output tensor with clipped input elements", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Clip_ver11_doc = R"DOC(
Clip operator limits the given input within an interval. The interval is
specified by the inputs 'min' and 'max'. They default to
numeric_limits::lowest() and numeric_limits::max(), respectively.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Clip,
    11,
    OpSchema()
        .SetDoc(Clip_ver11_doc)
        .Input(0, "input", "Input tensor whose elements to be clipped", "T")
        .Input(
            1,
            "min",
            "Minimum value, under which element is replaced by min. "
            "It must be a scalar(tensor of empty shape).",
            "T",
            OpSchema::Optional)
        .Input(
            2,
            "max",
            "Maximum value, above which element is replaced by max. "
            "It must be a scalar(tensor of empty shape).",
            "T",
            OpSchema::Optional)
        .Output(0, "output", "Output tensor with clipped input elements", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

std::function<void(OpSchema&)> ElementwiseMultiOpDocGenerator_old(const char* name) {
  return [=](OpSchema& schema) {
    std::string doc;
    POPULATE_OP_DOC_STR(doc = R"DOC(
Element-wise {name} of each of the input tensors (with Numpy-style broadcasting support).
All inputs and outputs must have the same data type.
{broadcast_doc}
)DOC";
                        ReplaceAll(doc, "{name}", name);
                        ReplaceAll(doc, "{broadcast_doc}", GenerateBroadcastingDocMul().c_str()););
    schema.SetDoc(doc);
    schema.Input(0, "data_0", "List of tensors for " + std::string(name) + ".", "T", OpSchema::Variadic);
    schema.Output(0, name, "Output tensor.", "T");
    schema.TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");
    schema.TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      int num_inputs = static_cast<int>(ctx.getNumInputs());
      std::vector<const TensorShapeProto*> shapes;
      for (int i = 0; i < num_inputs; ++i) {
        auto input_type = ctx.getInputType(i);
        if (nullptr == input_type || !input_type->has_tensor_type() || !input_type->tensor_type().has_shape()) {
          return;
        }
        shapes.push_back(&input_type->tensor_type().shape());
      }

      multidirectionalBroadcastShapeInference(shapes, *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
    });
  };
}

ONNX_OPERATOR_SET_SCHEMA(Max, 8, OpSchema().FillUsing(ElementwiseMultiOpDocGenerator_old("max")));

ONNX_OPERATOR_SET_SCHEMA(Min, 8, OpSchema().FillUsing(ElementwiseMultiOpDocGenerator_old("min")));

static const char* LeakyRelu_ver6_doc = R"DOC(
LeakyRelu takes input data (Tensor<T>) and an argument alpha, and produces one
output data (Tensor<T>) where the function `f(x) = alpha * x for x < 0`,
`f(x) = x for x >= 0`, is applied to the data tensor elementwise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    LeakyRelu,
    6,
    OpSchema()
        .Attr("alpha", "Coefficient of leakage.", AttributeProto::FLOAT, 0.01f)
        .SetDoc(LeakyRelu_ver6_doc)
        .Input(0, "X", "Input tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Output(0, "Y", "Output tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* PRelu_ver9_doc = R"DOC(
PRelu takes input data (Tensor<T>) and slope tensor as input, and produces one
output data (Tensor<T>) where the function `f(x) = slope * x for x < 0`,
`f(x) = x for x >= 0`., is applied to the data tensor elementwise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    PRelu,
    9,
    OpSchema()
        .SetDoc(
            GET_OP_DOC_STR(std::string(PRelu_ver9_doc) + GenerateBroadcastingDocUni("tensor slope", "input tensor X")))
        .Input(0, "X", "Input tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Input(
            1,
            "slope",
            "Slope tensor. The shape of slope can be smaller then first input X; "
            "if so, its shape must be unidirectional broadcastable to X",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .Output(0, "Y", "Output tensor (same size as X)", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .TypeConstraint(
            "T",
            {"tensor(float16)",
             "tensor(float)",
             "tensor(double)",
             "tensor(uint32)",
             "tensor(uint64)",
             "tensor(int32)",
             "tensor(int64)"},
            "Constrain input and output types to float/int tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));
} // namespace ONNX_NAMESPACE
