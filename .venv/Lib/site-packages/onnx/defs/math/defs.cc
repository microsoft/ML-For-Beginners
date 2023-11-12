/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <functional>
#include "onnx/defs/data_type_utils.h"
#include "onnx/defs/function.h"
#include "onnx/defs/math/utils.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/tensor_proto_util.h"

namespace ONNX_NAMESPACE {

inline int MathOpTwoIntegers(std::string op_type, int a, int b) {
  if (op_type == "Add") {
    return a + b;
  } else if (op_type == "Sub") {
    return a - b;
  } else if (op_type == "Mul") {
    return a * b;
  }
  fail_shape_inference("Wrong op_type name for running propagation: ", op_type);
}

inline void MathOpDataPropagator(DataPropagationContext& ctx, std::string op_type) {
  const auto input_0 = ctx.getInputData(0);
  const auto input_1 = ctx.getInputData(1);
  if (input_0 == nullptr || input_1 == nullptr) {
    return;
  }
  int size_0 = input_0->dim_size();
  int size_1 = input_1->dim_size();
  // Fails to broadcast if the ranks are different and no any rank is 1
  if (size_0 != size_1 && size_0 != 1 && size_1 != 1) {
    fail_shape_inference("Invalid rank for ", op_type, " broadcasting: (", size_0, ") vs (", size_1, ").");
  }
  TensorShapeProto tsp;
  for (int i = 0; i < std::max(size_0, size_1); ++i) {
    auto& input_dim_0 = input_0->dim(size_0 == 1 ? 0 : i);
    auto& input_dim_1 = input_1->dim(size_1 == 1 ? 0 : i);
    if (input_dim_0.has_dim_value() && input_dim_1.has_dim_value()) {
      tsp.mutable_dim()->Add()->set_dim_value(
          MathOpTwoIntegers(op_type, input_dim_0.dim_value(), input_dim_1.dim_value()));
    } else {
      // Cannot compute the value; simply add an empty dim without value and param
      tsp.mutable_dim()->Add();
    }
  }
  ctx.addOutputData(0, std::move(tsp));
}

std::function<void(OpSchema&)> MathDocGenerator(const char* name) {
  return [=](OpSchema& schema) {
    std::string doc;
    POPULATE_OP_DOC_STR(doc = R"DOC(
Performs element-wise binary {name} (with Numpy-style broadcasting support).

{broadcast_doc}

(Opset 14 change): Extend supported types to include uint8, int8, uint16, and int16.
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
        "T", OpSchema::all_numeric_types_ir4(), "Constrain input and output types to all numeric tensors.");
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

ONNX_OPERATOR_SET_SCHEMA(
    Add,
    14,
    OpSchema().FillUsing(MathDocGenerator("addition")).PartialDataPropagationFunction([](DataPropagationContext& ctx) {
      MathOpDataPropagator(ctx, "Add");
    }));

ONNX_OPERATOR_SET_SCHEMA(
    Sub,
    14,
    OpSchema()
        .FillUsing(MathDocGenerator("subtraction"))
        .PartialDataPropagationFunction([](DataPropagationContext& ctx) { MathOpDataPropagator(ctx, "Sub"); }));

static const char* Mod_doc = R"DOC(
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
    13,
    OpSchema()
        .SetDoc(Mod_doc)
        .Attr(
            "fmod",
            "Whether the operator should behave like fmod (default=0 meaning it will do integer mods); Set this to 1 to force fmod treatment",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Input(0, "A", "Dividend tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Input(1, "B", "Divisor tensor", "T", OpSchema::Single, true, 1, OpSchema::NonDifferentiable)
        .Output(0, "C", "Remainder tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .TypeConstraint(
            "T",
            OpSchema::all_numeric_types_ir4(),
            "Constrain input and output types to high-precision numeric tensors.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (hasNInputShapes(ctx, 2))
            bidirectionalBroadcastShapeInference(
                ctx.getInputType(0)->tensor_type().shape(),
                ctx.getInputType(1)->tensor_type().shape(),
                *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
        }));

ONNX_OPERATOR_SET_SCHEMA(
    Mul,
    14,
    OpSchema()
        .FillUsing(MathDocGenerator("multiplication"))
        .PartialDataPropagationFunction([](DataPropagationContext& ctx) { MathOpDataPropagator(ctx, "Mul"); }));

ONNX_OPERATOR_SET_SCHEMA(Div, 14, OpSchema().FillUsing(MathDocGenerator("division")));

static const char* Neg_ver13_doc = R"DOC(
Neg takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where each element flipped sign, y = -x, is applied to
the tensor elementwise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Neg,
    13,
    OpSchema()
        .SetDoc(Neg_ver13_doc)
        .Input(0, "X", "Input tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Output(0, "Y", "Output tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .TypeConstraint(
            "T",
            {"tensor(float)",
             "tensor(int32)",
             "tensor(int8)",
             "tensor(int16)",
             "tensor(int64)",
             "tensor(float16)",
             "tensor(double)",
             "tensor(bfloat16)"},
            "Constrain input and output types to signed numeric tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Abs_ver13_doc = R"DOC(
Absolute takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the absolute is, y = abs(x), is applied to
the tensor elementwise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Abs,
    13,
    OpSchema()
        .SetDoc(Abs_ver13_doc)
        .Input(0, "X", "Input tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Output(0, "Y", "Output tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .TypeConstraint(
            "T",
            OpSchema::all_numeric_types_ir4(),
            "Constrain input and output types to all numeric tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Reciprocal_ver13_doc = R"DOC(
Reciprocal takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the reciprocal is, y = 1/x, is applied to
the tensor elementwise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Reciprocal,
    13,
    OpSchema()
        .SetDoc(Reciprocal_ver13_doc)
        .Input(0, "X", "Input tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Output(0, "Y", "Output tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Floor_ver13_doc = R"DOC(
Floor takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the floor is, y = floor(x), is applied to
the tensor elementwise. If x is integral, +0, -0, NaN,  or infinite, x itself is returned.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Floor,
    13,
    OpSchema()
        .SetDoc(Floor_ver13_doc)
        .Input(0, "X", "Input tensor", "T", OpSchema::Single, true, 1, OpSchema::NonDifferentiable)
        .Output(0, "Y", "Output tensor", "T", OpSchema::Single, true, 1, OpSchema::NonDifferentiable)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Ceil_ver13_doc = R"DOC(
Ceil takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the ceil is, y = ceil(x), is applied to
the tensor elementwise. If x is integral, +0, -0, NaN,  or infinite, x itself is returned.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Ceil,
    13,
    OpSchema()
        .SetDoc(Ceil_ver13_doc)
        .Input(0, "X", "Input tensor", "T", OpSchema::Single, true, 1, OpSchema::NonDifferentiable)
        .Output(0, "Y", "Output tensor", "T", OpSchema::Single, true, 1, OpSchema::NonDifferentiable)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Sqrt_ver13_doc = R"DOC(
Square root takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the square root is, y = x^0.5, is applied to
the tensor elementwise. If x is negative, then it will return NaN.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Sqrt,
    13,
    OpSchema()
        .SetDoc(Sqrt_ver13_doc)
        .Input(0, "X", "Input tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Output(0, "Y", "Output tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Relu_ver14_doc = R"DOC(
Relu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the rectified linear function, y = max(0, x), is applied to
the tensor elementwise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Relu,
    14,
    OpSchema()
        .SetDoc(Relu_ver14_doc)
        .Input(0, "X", "Input tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Output(0, "Y", "Output tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .TypeConstraint(
            "T",
            {"tensor(float)",
             "tensor(int32)",
             "tensor(int8)",
             "tensor(int16)",
             "tensor(int64)",
             "tensor(float16)",
             "tensor(double)",
             "tensor(bfloat16)"},
            "Constrain input and output types to signed numeric tensors.")
        .FunctionBody(
            R"ONNX(
          {
            Zero = Constant <value = float {0.0}>()
            ZeroCast = CastLike (Zero, X)
            Y = Max (X, ZeroCast)
          }
        )ONNX",
            18)
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* LeakyRelu_ver16_doc = R"DOC(
LeakyRelu takes input data (Tensor<T>) and an argument alpha, and produces one
output data (Tensor<T>) where the function `f(x) = alpha * x for x < 0`,
`f(x) = x for x >= 0`, is applied to the data tensor elementwise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    LeakyRelu,
    16,
    OpSchema()
        .Attr("alpha", "Coefficient of leakage.", AttributeProto::FLOAT, 0.01f)
        .SetDoc(LeakyRelu_ver16_doc)
        .Input(0, "X", "Input tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Output(0, "Y", "Output tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .TypeConstraint(
            "T",
            {"tensor(bfloat16)", "tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput)
        .FunctionBody(R"ONNX(
          {
            Alpha = Constant <value_float: float = @alpha>()
            AlphaCast = CastLike (Alpha, X)
            Zero = Constant <value = float {0.0}>()
            ZeroCast = CastLike(Zero, X)
            XLessThanZero = Less(X, ZeroCast)
            AlphaMulX = Mul (AlphaCast, X)
            Y = Where (XLessThanZero, AlphaMulX, X)
          }
        )ONNX"));

static const char* ThresholdedRelu_ver10_doc = R"DOC(
ThresholdedRelu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the rectified linear function, y = x for x > alpha, y = 0 otherwise,
is applied to the tensor elementwise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    ThresholdedRelu,
    10,
    OpSchema()
        .SetDoc(ThresholdedRelu_ver10_doc)
        .Attr("alpha", "Threshold value", AttributeProto::FLOAT, 1.0f)
        .Input(0, "X", "Input tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Output(0, "Y", "Output tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput)
        .FunctionBody(
            R"ONNX(
          {
            Alpha = Constant <value_float: float = @alpha>()
            AlphaCast = CastLike (Alpha, X)
            Zero = Constant <value = float {0.0}>()
            ZeroCast = CastLike (Zero, X)
            AlphaLessThanX = Less(AlphaCast, X)
            Y = Where(AlphaLessThanX, X, ZeroCast)
          }
        )ONNX",
            18));

static const char* Selu_ver6_doc = R"DOC(
Selu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the scaled exponential linear unit function,
`y = gamma * (alpha * e^x - alpha) for x <= 0`, `y = gamma * x for x > 0`,
is applied to the tensor elementwise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Selu,
    6,
    OpSchema()
        .Attr(
            "alpha",
            "Coefficient of SELU default to 1.67326319217681884765625 "
            "(i.e., float32 approximation of 1.6732632423543772848170429916717).",
            AttributeProto::FLOAT,
            1.67326319217681884765625f)
        .Attr(
            "gamma",
            "Coefficient of SELU default to 1.05070102214813232421875 "
            "(i.e., float32 approximation of 1.0507009873554804934193349852946).",
            AttributeProto::FLOAT,
            1.05070102214813232421875f)
        .SetDoc(Selu_ver6_doc)
        .Input(0, "X", "Input tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Output(0, "Y", "Output tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput)
        .FunctionBody(
            R"ONNX(
          {
            Alpha = Constant <value_float: float = @alpha>()
            AlphaCast = CastLike (Alpha, X)
            Gamma = Constant <value_float: float = @gamma>()
            GammaCast = CastLike (Gamma, X)
            Zero = Constant <value = float {0.0}>()
            ZeroCast = CastLike (Zero, X)
            ExpX = Exp (X)
            AlphaMulExpX = Mul(AlphaCast, ExpX)
            AlphaMulExpXSubAlpha = Sub (AlphaMulExpX, AlphaCast)
            Neg = Mul (GammaCast, AlphaMulExpXSubAlpha)
            Pos = Mul (GammaCast, X)
            XLessThanZero = Less (X, ZeroCast)
            Y = Where(XLessThanZero, Neg, Pos)
          }
        )ONNX",
            18));

static const char* Elu_ver6_doc = R"DOC(
Elu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the function `f(x) = alpha * (exp(x) - 1.) for x <
0`, `f(x) = x for x >= 0`., is applied to the tensor elementwise.

)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Elu,
    6,
    OpSchema()
        .Attr("alpha", "Coefficient of ELU.", AttributeProto::FLOAT, 1.0f)
        .SetDoc(Elu_ver6_doc)
        .Input(0, "X", "1D input tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Output(0, "Y", "1D output tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput)
        .FunctionBody(
            R"ONNX(
          {
            Alpha = Constant <value_float: float = @alpha>()
            AlphaCast = CastLike (Alpha, X)
            Zero = Constant <value = float {0.0}>()
            ZeroCast = CastLike (Zero, X)
            One = Constant <value = float {1.0}>()
            OneCast = CastLike (One, X)
            XLessThanZero = Less (X, ZeroCast)
            ExpX = Exp (X)
            ExpXSubOne = Sub (ExpX, OneCast)
            AlphaMulExpXSubOne = Mul (AlphaCast, ExpXSubOne)
            Y = Where(XLessThanZero, AlphaMulExpXSubOne, X)
          }
        )ONNX",
            18));

static const char* mish_ver18_doc = R"DOC(
Mish: A Self Regularized Non-Monotonic Neural Activation Function.

Perform the linear unit element-wise on the input tensor X using formula:

```
mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
```
)DOC";

static const char* celu_ver12_doc = R"DOC(
Continuously Differentiable Exponential Linear Units:
Perform the linear unit element-wise on the input tensor X
using formula:

```
max(0,x) + min(0,alpha*(exp(x/alpha)-1))
```
)DOC";

static float celu_default_alpha = 1.0;

TensorProto ToDimensionOneFloatTensor(float value) {
  auto t = ToTensor(std::vector<float>({value}));
  t.add_dims(1);
  return t;
}

TensorProto ToDimensionOneTensor(int32_t value) {
  auto t = ToTensor(std::vector<int32_t>({value}));
  t.add_dims(1);
  return t;
}

TensorProto ToDimensionOneInt64Tensor(int64_t value) {
  auto t = ToTensor(std::vector<int64_t>({value}));
  t.add_dims(1);
  return t;
}

TensorProto ToDimensionOneInt64Tensor(std::vector<int64_t> value) {
  auto t = ToTensor(value);
  t.add_dims(value.size());
  return t;
}

bool BuildContextDependentFunctionBodyCelu(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  float alpha = ctx.getAttribute("alpha") != nullptr ? ctx.getAttribute("alpha")->f() : celu_default_alpha;
  FunctionBuilder builder(functionProto);
  builder.Const("alpha", std::vector<float>{alpha}).Add(R"(
            X_alpha = Div (X, alpha)
            Elu_Result = Elu <alpha = 1.0>(X_alpha)
            Y = Mul (alpha, Elu_Result)
        )");
  schema.BuildFunction(functionProto);
  return true;
}

ONNX_OPERATOR_SET_SCHEMA(
    Celu,
    12,
    OpSchema()
        .SetDoc(celu_ver12_doc)
        .Input(0, "X", "Input tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Output(0, "Y", "Output tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Attr(
            "alpha",
            "The Alpha value in Celu formula which control the shape of "
            "the unit. The default value is 1.0.",
            AttributeProto::FLOAT,
            celu_default_alpha)
        .TypeConstraint("T", {"tensor(float)"}, "Constrain input and output types to float32 tensors.")
        .SetContextDependentFunctionBodyBuilder(BuildContextDependentFunctionBodyCelu)
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Mish,
    18,
    OpSchema()
        .SetDoc(mish_ver18_doc)
        .Input(0, "X", "Input tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Output(0, "Y", "Output tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input X and output types to float tensors.")
        .FunctionBody(R"ONNX(
          {
            Softplus_X = Softplus (X)
            TanHSoftplusX = Tanh (Softplus_X)
            Y = Mul (X, TanHSoftplusX)
           }
        )ONNX")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Exp_ver13_doc = R"DOC(
Calculates the exponential of the given input tensor, element-wise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Exp,
    13,
    OpSchema()
        .SetDoc(Exp_ver13_doc)
        .Input(0, "input", "Input tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Output(
            0,
            "output",
            "The exponential of the input tensor computed "
            "element-wise",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Log_ver13_doc = R"DOC(
Calculates the natural log of the given input tensor, element-wise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Log,
    13,
    OpSchema()
        .SetDoc(Log_ver13_doc)
        .Input(0, "input", "Input tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Output(
            0,
            "output",
            "The natural log of the input tensor computed "
            "element-wise",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Tanh_ver13_doc = R"DOC(
Calculates the hyperbolic tangent of the given input tensor element-wise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Tanh,
    13,
    OpSchema()
        .SetDoc(Tanh_ver13_doc)
        .Input(0, "input", "Input tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Output(
            0,
            "output",
            "The hyperbolic tangent values of the input tensor "
            "computed element-wise",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Pow_ver15_doc = R"DOC(
Pow takes input data (Tensor<T>) and exponent Tensor, and
produces one output data (Tensor<T>) where the function `f(x) = x^exponent`,
is applied to the data tensor elementwise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Pow,
    15,
    OpSchema()
        .SetDoc(GET_OP_DOC_STR(std::string(Pow_ver15_doc) + GenerateBroadcastingDocMul()))
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
             "tensor(double)",
             "tensor(bfloat16)"},
            "Constrain input Y types to float/int tensors.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (hasNInputShapes(ctx, 2))
            bidirectionalBroadcastShapeInference(
                ctx.getInputType(0)->tensor_type().shape(),
                ctx.getInputType(1)->tensor_type().shape(),
                *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
        }));

static const char* PRelu_ver16_doc = R"DOC(
PRelu takes input data (Tensor<T>) and slope tensor as input, and produces one
output data (Tensor<T>) where the function `f(x) = slope * x for x < 0`,
`f(x) = x for x >= 0`., is applied to the data tensor elementwise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    PRelu,
    16,
    OpSchema()
        .SetDoc(
            GET_OP_DOC_STR(std::string(PRelu_ver16_doc) + GenerateBroadcastingDocUni("tensor slope", "input tensor X")))
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
            {"tensor(bfloat16)",
             "tensor(float16)",
             "tensor(float)",
             "tensor(double)",
             "tensor(uint32)",
             "tensor(uint64)",
             "tensor(int32)",
             "tensor(int64)"},
            "Constrain input and output types to float/int tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput)
        .FunctionBody(R"ONNX(
        {
          Zero = Constant <value = float {0.0}>()
          ZeroCast = CastLike(Zero, X)
          XLessThanZero = Less (X, ZeroCast)
          SlopeMulX = Mul (slope, X)
          Y = Where(XLessThanZero, SlopeMulX, X)
        }
        )ONNX"));

static const char* Sigmoid_ver13_doc = R"DOC(
Sigmoid takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the sigmoid function, y = 1 / (1 + exp(-x)), is applied to the
tensor elementwise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Sigmoid,
    13,
    OpSchema()
        .SetDoc(Sigmoid_ver13_doc)
        .Input(0, "X", "Input tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Output(0, "Y", "Output tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* HardSigmoid_ver6_doc = R"DOC(
HardSigmoid takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the HardSigmoid function, y = max(0, min(1, alpha * x + beta)),
is applied to the tensor elementwise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    HardSigmoid,
    6,
    OpSchema()
        .Attr("alpha", "Value of alpha.", AttributeProto::FLOAT, 0.2f)
        .Attr("beta", "Value of beta.", AttributeProto::FLOAT, 0.5f)
        .SetDoc(HardSigmoid_ver6_doc)
        .Input(0, "X", "Input tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Output(0, "Y", "Output tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput)
        .FunctionBody(
            R"ONNX(
          {
            Alpha = Constant <value_float: float = @alpha>()
            AlphaCast = CastLike (Alpha, X)
            Beta = Constant <value_float: float = @beta>()
            BetaCast = CastLike (Beta, X)
            Zero = Constant <value = float {0.0}>()
            ZeroCast = CastLike (Zero, X)
            One = Constant <value = float {1.0}>()
            OneCast = CastLike (One, X)
            AlphaMulX = Mul (X, AlphaCast)
            AlphaMulXAddBeta = Add (AlphaMulX, BetaCast)
            MinOneOrAlphaMulXAddBeta = Min (AlphaMulXAddBeta, OneCast)
            Y = Max(MinOneOrAlphaMulXAddBeta, ZeroCast)
          }
        )ONNX",
            18));

static const char* HardSwish_ver14_doc = R"DOC(
HardSwish takes one input data (Tensor<T>) and produces one output data (Tensor<T>) where
the HardSwish function, y = x * max(0, min(1, alpha * x + beta)) = x * HardSigmoid<alpha, beta>(x),
where alpha = 1/6 and beta = 0.5, is applied to the tensor elementwise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    HardSwish,
    14,
    OpSchema()
        .SetDoc(HardSwish_ver14_doc)
        .Input(0, "X", "Input tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Output(0, "Y", "Output tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput)
        .FunctionBody(R"ONNX(
          {
            HS_X = HardSigmoid<alpha = 0.16666667163372, beta = 0.5>(X)
            Y = Mul (X, HS_X)
          }
        )ONNX"));

// Generate opschema for element-wise ops. Leaves type constraint "T"
// unspecified.
std::function<void(OpSchema&)> ElementwiseMultiOpDocGenerator(const char* name) {
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
    schema.Input(
        0,
        "data_0",
        "List of tensors for " + std::string(name) + ".",
        "T",
        OpSchema::Variadic,
        true,
        1,
        OpSchema::Differentiable);
    schema.Output(0, name, "Output tensor.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable);
    schema.TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      int num_inputs = static_cast<int>(ctx.getNumInputs());
      std::vector<const TensorShapeProto*> shapes;
      shapes.reserve(num_inputs);
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
    13,
    OpSchema()
        .FillUsing(ElementwiseMultiOpDocGenerator("max"))
        .TypeConstraint(
            "T",
            OpSchema::all_numeric_types_ir4(),
            "Constrain input and output types to numeric tensors."));

ONNX_OPERATOR_SET_SCHEMA(
    Min,
    13,
    OpSchema()
        .FillUsing(ElementwiseMultiOpDocGenerator("min"))
        .TypeConstraint(
            "T",
            OpSchema::all_numeric_types_ir4(),
            "Constrain input and output types to numeric tensors."));

ONNX_OPERATOR_SET_SCHEMA(
    Sum,
    13,
    OpSchema()
        .FillUsing(ElementwiseMultiOpDocGenerator("sum"))
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
            "Constrain input and output types to float tensors."));

ONNX_OPERATOR_SET_SCHEMA(
    Mean,
    13,
    OpSchema()
        .FillUsing(ElementwiseMultiOpDocGenerator("mean"))
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
            "Constrain input and output types to float tensors."));

static const char* Clip_ver13_doc = R"DOC(
Clip operator limits the given input within an interval. The interval is
specified by the inputs 'min' and 'max'. They default to
numeric_limits::lowest() and numeric_limits::max(), respectively.
)DOC";

bool BuildContextDependentFunctionBodyClip(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  bool has_min = ctx.hasInput(1);
  bool has_max = ctx.hasInput(2);

  FunctionBuilder builder(functionProto);
  if (!has_min && !has_max) {
    builder.Add("output = Identity (input)");
  } else if (has_min && !has_max) {
    builder.Add("input_less_than_min = Less (input, min)");
    builder.Add("output = Where (input_less_than_min, min, input)");
  } else if (!has_min && has_max) {
    builder.Add("input_large_than_max = Less (max, input)");
    builder.Add("output = Where (input_large_than_max, max, input)");
  } else {
    builder.Add("input_less_than_min = Less (input, min)");
    builder.Add("tmp = Where (input_less_than_min, min, input)");
    builder.Add("output_large_than_max = Less (max, tmp)");
    builder.Add("output = Where (output_large_than_max, max, tmp)");
  }

  schema.BuildFunction(functionProto);
  return true;
}

ONNX_OPERATOR_SET_SCHEMA(
    Clip,
    13,
    OpSchema()
        .SetDoc(Clip_ver13_doc)
        .Input(
            0,
            "input",
            "Input tensor whose elements to be clipped",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .Input(
            1,
            "min",
            "Minimum value, under which element is replaced by min. "
            "It must be a scalar(tensor of empty shape).",
            "T",
            OpSchema::Optional,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Input(
            2,
            "max",
            "Maximum value, above which element is replaced by max. "
            "It must be a scalar(tensor of empty shape).",
            "T",
            OpSchema::Optional,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Output(
            0,
            "output",
            "Output tensor with clipped input elements",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .TypeConstraint(
            "T",
            OpSchema::all_numeric_types_ir4(),
            "Constrain input and output types to all numeric tensors.")
        .SetContextDependentFunctionBodyBuilder(BuildContextDependentFunctionBodyClip)
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Softmax,
    13,
    OpSchema()
        .FillUsing(SoftmaxFamilyDocGenerator(
            "Softmax",
            "normalized exponential",
            "Softmax(input, axis) = Exp(input) / ReduceSum(Exp(input), axis=axis, keepdims=1) "))
        // function body builder for opset version 13 (the default opset version is the same
        // as the operator's since_version.
        .SetContextDependentFunctionBodyBuilder(
            [](const FunctionBodyBuildContext& ctx, const OpSchema& schema, FunctionProto& functionProto) -> bool {
              int64_t axis = ctx.getAttribute("axis") != nullptr ? ctx.getAttribute("axis")->i() : -1;
              FunctionBuilder builder(functionProto);
              builder.Const1D("axes", axis)
                  .Add("X_ReduceMax = ReduceMax <keepdims = 1> (input)", "axes", std::vector<int64_t>({axis}))
                  .Add(R"(
                    X_Sub = Sub (input, X_ReduceMax)
                    X_Exp = Exp (X_Sub)
                    X_ReduceSum = ReduceSum <keepdims = 1> (X_Exp, axes)
                    output = Div (X_Exp, X_ReduceSum)
                )");

              schema.BuildFunction(functionProto);
              return true;
            })
        // function body builder for opset version 18.
        // ReduceSum is updated in opset 18 to have axes as the second input.
        // Therefore function body for opset version 18
        // is different than the one defined using opset version 13.
        .SetContextDependentFunctionBodyBuilder(
            [](const FunctionBodyBuildContext& ctx, const OpSchema& schema, FunctionProto& functionProto) -> bool {
              int64_t axis = ctx.getAttribute("axis") != nullptr ? ctx.getAttribute("axis")->i() : -1;
              FunctionBuilder builder(functionProto);
              builder.Const1D("axes", axis).Add("X_ReduceMax = ReduceMax <keepdims = 1> (input, axes)").Add(R"(
                    X_Sub = Sub (input, X_ReduceMax)
                    X_Exp = Exp (X_Sub)
                    X_ReduceSum = ReduceSum <keepdims = 1> (X_Exp, axes)
                    output = Div (X_Exp, X_ReduceSum)
                )");

              schema.BuildFunction(functionProto);
              return true;
            },
            18));

ONNX_OPERATOR_SET_SCHEMA(
    LogSoftmax,
    13,
    OpSchema()
        .FillUsing(SoftmaxFamilyDocGenerator(
            "LogSoftmax",
            "log of softmax",
            "LogSoftmax(input, axis) = Log(Softmax(input, axis=axis))"))
        // Function for opset 13
        .SetContextDependentFunctionBodyBuilder(
            [](const FunctionBodyBuildContext& ctx, const OpSchema& schema, FunctionProto& functionProto) -> bool {
              const int64_t axis = ctx.getAttribute("axis") != nullptr ? ctx.getAttribute("axis")->i() : -1;
              FunctionBuilder builder(functionProto);
              builder.Const1D("axes", axis)
                  .Add("X_ReduceMax = ReduceMax <keepdims = 1> (input)", "axes", std::vector<int64_t>({axis}))
                  .Add(R"(
                    X_Sub = Sub (input, X_ReduceMax)
                    X_Exp = Exp (X_Sub)
                    X_ReduceSum = ReduceSum <keepdims = 1> (X_Exp, axes)
                    X_Log = Log (X_ReduceSum)
                    output = Sub (X_Sub, X_Log)
                )");

              schema.BuildFunction(functionProto);
              return true;
            },
            13)
        // Function for opset 18
        .SetContextDependentFunctionBodyBuilder(
            [](const FunctionBodyBuildContext& ctx, const OpSchema& schema, FunctionProto& functionProto) -> bool {
              const int64_t axis = ctx.getAttribute("axis") != nullptr ? ctx.getAttribute("axis")->i() : -1;
              FunctionBuilder builder(functionProto);
              builder.Const1D("axes", axis).Add("X_ReduceMax = ReduceMax <keepdims = 1> (input, axes)").Add(R"(
                    X_Sub = Sub (input, X_ReduceMax)
                    X_Exp = Exp (X_Sub)
                    X_ReduceSum = ReduceSum <keepdims = 1> (X_Exp, axes)
                    X_Log = Log (X_ReduceSum)
                    output = Sub (X_Sub, X_Log)
                )");

              schema.BuildFunction(functionProto);
              return true;
            },
            18));

ONNX_OPERATOR_SET_SCHEMA(
    Hardmax,
    13,
    OpSchema().FillUsing(SoftmaxFamilyDocGenerator(
        "Hardmax",
        "hardmax",
        "Hardmax(element in input, axis) = 1 if the element is the first maximum value along the specified axis, 0 otherwise")));

static const char* Softsign_ver1_doc = R"DOC(
Calculates the softsign (x/(1+|x|)) of the given input tensor element-wise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Softsign,
    1,
    OpSchema()
        .SetDoc(Softsign_ver1_doc)
        .Input(0, "input", "Input tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Output(
            0,
            "output",
            "The softsign (x/(1+|x|)) values of the input tensor computed element-wise",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput)
        .FunctionBody(
            R"ONNX(
          {
            One = Constant <value = float {1.0}>()
            OneCast = CastLike (One, input)
            AbsInput = Abs(input)
            OneAddAbsInput = Add (OneCast, AbsInput)
            output = Div(input, OneAddAbsInput)
          }
        )ONNX",
            18));

static const char* Softplus_ver1_doc = R"DOC(
Softplus takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the softplus function, y = ln(exp(x) + 1), is applied to
the tensor elementwise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Softplus,
    1,
    OpSchema()
        .SetDoc(Softplus_ver1_doc)
        .Input(0, "X", "1D input tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Output(0, "Y", "1D input tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput)
        .FunctionBody(
            R"ONNX(
            {
              exp_x = Exp (X)
              one = Constant <value = float {1.0}>()
              one_cast = CastLike (one, X)
              exp_x_add_one = Add (exp_x, one_cast)
              Y = Log (exp_x_add_one)
            }
            )ONNX",
            18));

static const char* Gemm_ver13_doc = R"DOC(General Matrix multiplication:
https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3

* A' = transpose(A) if transA else A
* B' = transpose(B) if transB else B

Compute Y = alpha * A' * B' + beta * C, where input tensor A has shape (M, K) or (K, M),
input tensor B has shape (K, N) or (N, K), input tensor C is broadcastable to shape (M, N),
and output tensor Y has shape (M, N). A will be transposed before doing the
computation if attribute transA is non-zero, same for B and transB.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Gemm,
    13,
    OpSchema()
        .SetDoc(GET_OP_DOC_STR(
            std::string(Gemm_ver13_doc) + GenerateBroadcastingDocUni("tensor C", "tensor A * B") + "\n" +
            GenerateOptionalArgumentsDoc()))
        .Input(
            0,
            "A",
            "Input tensor A. "
            "The shape of A should be (M, K) if transA is 0, "
            "or (K, M) if transA is non-zero.",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .Input(
            1,
            "B",
            "Input tensor B. "
            "The shape of B should be (K, N) if transB is 0, "
            "or (N, K) if transB is non-zero.",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .Input(
            2,
            "C",
            "Optional input tensor C. "
            "If not specified, the computation is done as if C is a scalar 0. "
            "The shape of C should be unidirectional broadcastable to (M, N).",
            "T",
            OpSchema::Optional,
            true,
            1,
            OpSchema::Differentiable)
        .Output(0, "Y", "Output tensor of shape (M, N).", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .TypeConstraint(
            "T",
            {"tensor(float16)",
             "tensor(float)",
             "tensor(double)",
             "tensor(uint32)",
             "tensor(uint64)",
             "tensor(int32)",
             "tensor(int64)",
             "tensor(bfloat16)"},
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

void matmulShapeInference(ONNX_NAMESPACE::InferenceContext& ctx, int input1Idx, int input2Idx) {
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

static const char* MatMul_ver13_doc = R"DOC(
Matrix product that behaves like numpy.matmul: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    MatMul,
    13,
    OpSchema()
        .Input(0, "A", "N-dimensional matrix A", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Input(1, "B", "N-dimensional matrix B", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Output(0, "Y", "Matrix multiply results from A * B", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .TypeConstraint(
            "T",
            {"tensor(float16)",
             "tensor(float)",
             "tensor(double)",
             "tensor(uint32)",
             "tensor(uint64)",
             "tensor(int32)",
             "tensor(int64)",
             "tensor(bfloat16)"},
            "Constrain input and output types to float/int tensors.")
        .SetDoc(MatMul_ver13_doc)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          matmulShapeInference(ctx, 0, 1);
        }));

static const char* TopK_ver11_doc = R"DOC(
Retrieve the top-K largest or smallest elements along a specified axis. Given an input tensor of
shape [a_1, a_2, ..., a_n, r] and integer argument k, return two outputs:

* Value tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n]
  which contains the values of the top k elements along the specified axis
* Index tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n] which
  contains the indices of the top k elements (original indices from the input
  tensor).

* If "largest" is 1 (the default value) then the k largest elements are returned.
* If "sorted" is 1 (the default value) then the resulting k elements will be sorted.
* If "sorted" is 0, order of returned 'Values' and 'Indices' are undefined.

Given two equivalent values, this operator uses the indices along the axis as
a tiebreaker. That is, the element with the lower index will appear first.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    TopK,
    11,
    OpSchema()
        .SetDoc(TopK_ver11_doc)
        .Input(
            0,
            "X",
            "Tensor of shape [a_1, a_2, ..., a_n, r]",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .Input(
            1,
            "K",
            "A 1-D tensor containing a single positive value corresponding to the number of top elements to retrieve",
            "tensor(int64)",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Output(
            0,
            "Values",
            "Tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n] "
            "containing top K values from the input tensor",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .Output(
            1,
            "Indices",
            "Tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n] "
            "containing the corresponding input tensor indices for the top K "
            "values.",
            "I",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .TypeConstraint("T", OpSchema::all_numeric_types(), "Constrain input and output types to numeric tensors.")
        .TypeConstraint("I", {"tensor(int64)"}, "Constrain index tensor to int64")
        .Attr(
            "axis",
            "Dimension on which to do the sort. Negative value means counting dimensions "
            "from the back. Accepted range is [-r, r-1] where r = rank(input).",
            AttributeProto::INT,
            static_cast<int64_t>(-1))
        .Attr(
            "largest",
            "Whether to return the top-K largest or smallest elements.",
            AttributeProto::INT,
            static_cast<int64_t>(1))
        .Attr("sorted", "Whether to return the elements in sorted order.", AttributeProto::INT, static_cast<int64_t>(1))
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

static const char* Sin_ver7_doc = R"DOC(
Calculates the sine of the given input tensor, element-wise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Sin,
    7,
    OpSchema()
        .SetDoc(Sin_ver7_doc)
        .Input(0, "input", "Input tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Output(
            0,
            "output",
            "The sine of the input tensor computed "
            "element-wise",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Cos_ver7_doc = R"DOC(
Calculates the cosine of the given input tensor, element-wise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Cos,
    7,
    OpSchema()
        .SetDoc(Cos_ver7_doc)
        .Input(0, "input", "Input tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Output(
            0,
            "output",
            "The cosine of the input tensor computed "
            "element-wise",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Tan_ver7_doc = R"DOC(
Calculates the tangent of the given input tensor, element-wise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Tan,
    7,
    OpSchema()
        .SetDoc(Tan_ver7_doc)
        .Input(0, "input", "Input tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Output(
            0,
            "output",
            "The tangent of the input tensor computed "
            "element-wise",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Asin_ver7_doc = R"DOC(
Calculates the arcsine (inverse of sine) of the given input tensor, element-wise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Asin,
    7,
    OpSchema()
        .SetDoc(Asin_ver7_doc)
        .Input(0, "input", "Input tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Output(
            0,
            "output",
            "The arcsine of the input tensor computed "
            "element-wise",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Acos_ver7_doc = R"DOC(
Calculates the arccosine (inverse of cosine) of the given input tensor, element-wise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Acos,
    7,
    OpSchema()
        .SetDoc(Acos_ver7_doc)
        .Input(0, "input", "Input tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Output(
            0,
            "output",
            "The arccosine of the input tensor computed "
            "element-wise",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Atan_ver7_doc = R"DOC(
Calculates the arctangent (inverse of tangent) of the given input tensor, element-wise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Atan,
    7,
    OpSchema()
        .SetDoc(Atan_ver7_doc)
        .Input(0, "input", "Input tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Output(
            0,
            "output",
            "The arctangent of the input tensor computed "
            "element-wise",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Expand_ver13_doc = R"DOC(
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
    13,
    OpSchema()
        .SetDoc(Expand_ver13_doc)
        .Input(0, "input", "Input tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Input(
            1,
            "shape",
            "A 1-D tensor indicates the shape you want to expand to, following the broadcast rule",
            "tensor(int64)",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Output(0, "output", "Output tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .TypeConstraint("T", OpSchema::all_tensor_types_ir4(), "Constrain input and output types to all tensors.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          // Type inference
          propagateElemTypeFromInputToOutput(ctx, 0, 0);

          // Shape inference
          // For shape inference, we need both input shape
          if (hasNInputShapes(ctx, 2)) {
            const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
            bool found = false;
            TensorShapeProto second_shape = getShapeInput(ctx, 1, found);
            if (found) {
              bidirectionalBroadcastShapeInference(input_shape, second_shape, *getOutputShape(ctx, 0));
            }
          }
        }));

static const char* Sinh_ver9_doc = R"DOC(
Calculates the hyperbolic sine of the given input tensor element-wise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Sinh,
    9,
    OpSchema()
        .SetDoc(Sinh_ver9_doc)
        .Input(0, "input", "Input tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Output(
            0,
            "output",
            "The hyperbolic sine values of the input tensor "
            "computed element-wise",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Cosh_ver9_doc = R"DOC(
Calculates the hyperbolic cosine of the given input tensor element-wise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Cosh,
    9,
    OpSchema()
        .SetDoc(Cosh_ver9_doc)
        .Input(0, "input", "Input tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Output(
            0,
            "output",
            "The hyperbolic cosine values of the input tensor "
            "computed element-wise",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Asinh_ver9_doc = R"DOC(
Calculates the hyperbolic arcsine of the given input tensor element-wise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Asinh,
    9,
    OpSchema()
        .SetDoc(Asinh_ver9_doc)
        .Input(0, "input", "Input tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Output(
            0,
            "output",
            "The hyperbolic arcsine values of the input tensor "
            "computed element-wise",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Acosh_ver9_doc = R"DOC(
Calculates the hyperbolic arccosine of the given input tensor element-wise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Acosh,
    9,
    OpSchema()
        .SetDoc(Acosh_ver9_doc)
        .Input(0, "input", "Input tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Output(
            0,
            "output",
            "The hyperbolic arccosine values of the input tensor "
            "computed element-wise",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Atanh_ver9_doc = R"DOC(
Calculates the hyperbolic arctangent of the given input tensor element-wise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Atanh,
    9,
    OpSchema()
        .SetDoc(Atanh_ver9_doc)
        .Input(0, "input", "Input tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Output(
            0,
            "output",
            "The hyperbolic arctangent values of the input tensor "
            "computed element-wise",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Sign_ver13_doc = R"DOC(
Calculate the sign of the given input tensor element-wise.
If input > 0, output 1. if input < 0, output -1. if input == 0, output 0.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Sign,
    13,
    OpSchema()
        .SetDoc(Sign_ver13_doc)
        .Input(0, "input", "Input tensor", "T", OpSchema::Single, true, 1, OpSchema::NonDifferentiable)
        .Output(
            0,
            "output",
            "The sign of the input tensor "
            "computed element-wise. It has the same shape and type of the input.",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .TypeConstraint(
            "T",
            OpSchema::all_numeric_types_ir4(),
            "Constrain input and output types to all numeric tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Erf_ver13_doc = R"DOC(
Computes the error function of the given input tensor element-wise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Erf,
    13,
    OpSchema()
        .SetDoc(Erf_ver13_doc)
        .Input(0, "input", "Input tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Output(
            0,
            "output",
            "The error function of the input tensor "
            "computed element-wise. It has the same shape and type of the input.",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .TypeConstraint(
            "T",
            OpSchema::all_numeric_types_ir4(),
            "Constrain input and output types to all numeric tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* QLinearMatMul_ver10_doc = R"DOC(
Matrix product that behaves like numpy.matmul: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html.
It consumes two quantized input tensors, their scales and zero points, scale and zero point of output,
and computes the quantized output. The quantization formula is y = saturate((x / y_scale) + y_zero_point).
For (x / y_scale), it is rounding to nearest ties to even. Refer to https://en.wikipedia.org/wiki/Rounding for details.
Scale and zero point must have same shape. They must be either scalar (per tensor) or N-D tensor
(per row for 'a' and per column for 'b'). Scalar refers to per tensor quantization whereas N-D refers to per row
or per column quantization. If the input is 2D of shape [M, K] then zero point and scale tensor may be
an M element vector [v_1, v_2, ..., v_M] for per row quantization and K element vector of shape [v_1, v_2, ..., v_K]
for per column quantization. If the input is N-D tensor with shape [D1, D2, M, K] then zero point and scale tensor may
have shape [D1, D2, M, 1] for per row quantization and shape [D1, D2, 1, K] for per column quantization.
Production must never overflow, and accumulation may overflow if and only if in 32 bits.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    QLinearMatMul,
    10,
    OpSchema()
        .SetDoc(QLinearMatMul_ver10_doc)
        .Input(0, "a", "N-dimensional quantized matrix a", "T1", OpSchema::Single, true, 1, OpSchema::NonDifferentiable)
        .Input(
            1,
            "a_scale",
            "scale of quantized input a",
            "tensor(float)",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Input(
            2,
            "a_zero_point",
            "zero point of quantized input a",
            "T1",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Input(3, "b", "N-dimensional quantized matrix b", "T2", OpSchema::Single, true, 1, OpSchema::NonDifferentiable)
        .Input(
            4,
            "b_scale",
            "scale of quantized input b",
            "tensor(float)",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Input(
            5,
            "b_zero_point",
            "zero point of quantized input b",
            "T2",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Input(
            6,
            "y_scale",
            "scale of quantized output y",
            "tensor(float)",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Input(
            7,
            "y_zero_point",
            "zero point of quantized output y",
            "T3",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Output(
            0,
            "y",
            "Quantized matrix multiply results from a * b",
            "T3",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .TypeConstraint(
            "T1",
            {"tensor(int8)", "tensor(uint8)"},
            "Constrain input a and its zero point data type to 8-bit integer tensor.")
        .TypeConstraint(
            "T2",
            {"tensor(int8)", "tensor(uint8)"},
            "Constrain input b and its zero point data type to 8-bit integer tensor.")
        .TypeConstraint(
            "T3",
            {"tensor(int8)", "tensor(uint8)"},
            "Constrain output y and its zero point data type to 8-bit integer tensor.")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          auto a_type = ctx.getInputType(0);
          auto b_type = ctx.getInputType(3);
          if (nullptr == a_type || nullptr == b_type ||
              a_type->value_case() != ONNX_NAMESPACE::TypeProto::kTensorType ||
              b_type->value_case() != ONNX_NAMESPACE::TypeProto::kTensorType) {
            fail_type_inference("inputs are expected to have tensor type.");
          }

          auto a_zero_point_type = ctx.getInputType(2);
          if (nullptr == a_zero_point_type ||
              a_zero_point_type->tensor_type().elem_type() != a_type->tensor_type().elem_type()) {
            fail_type_inference("input and zero_point pair is expected to have be same type.");
          }

          auto b_zero_point_type = ctx.getInputType(5);
          if (nullptr == b_zero_point_type ||
              b_zero_point_type->tensor_type().elem_type() != b_type->tensor_type().elem_type()) {
            fail_type_inference("input and zero_point pair is expected to have same type.");
          }

          propagateElemTypeFromInputToOutput(ctx, 7, 0);

          matmulShapeInference(ctx, 0, 3);
        }));

static const char* MatMulInteger_ver10_doc = R"DOC(
Matrix product that behaves like numpy.matmul: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html.
The production MUST never overflow. The accumulation may overflow if and only if in 32 bits.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    MatMulInteger,
    10,
    OpSchema()
        .SetDoc(MatMulInteger_ver10_doc)
        .Input(0, "A", "N-dimensional matrix A", "T1", OpSchema::Single, true, 1, OpSchema::NonDifferentiable)
        .Input(1, "B", "N-dimensional matrix B", "T2", OpSchema::Single, true, 1, OpSchema::NonDifferentiable)
        .Input(
            2,
            "a_zero_point",
            "Zero point tensor for input 'A'. It's optional and default value is 0. It could be a scalar or N-D tensor. "
            "Scalar refers to per tensor quantization whereas N-D refers to per row quantization. "
            "If the input is 2D of shape [M, K] then zero point tensor may be an M element vector [zp_1, zp_2, ..., zp_M]. "
            "If the input is N-D tensor with shape [D1, D2, M, K] then zero point tensor may have shape [D1, D2, M, 1]. ",
            "T1",
            OpSchema::Optional,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Input(
            3,
            "b_zero_point",
            "Zero point tensor for input 'B'. It's optional and default value is 0. It could be a scalar or a N-D tensor, "
            "Scalar refers to per tensor quantization whereas N-D refers to per col quantization. "
            "If the input is 2D of shape [K, N] then zero point tensor may be an N element vector [zp_1, zp_2, ..., zp_N]. "
            "If the input is N-D tensor with shape [D1, D2, K, N] then zero point tensor may have shape [D1, D2, 1, N]. ",
            "T2",
            OpSchema::Optional,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Output(
            0,
            "Y",
            "Matrix multiply results from A * B",
            "T3",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .TypeConstraint("T1", {"tensor(int8)", "tensor(uint8)"}, "Constrain input A data type to 8-bit integer tensor.")
        .TypeConstraint("T2", {"tensor(int8)", "tensor(uint8)"}, "Constrain input B data type to 8-bit integer tensor.")
        .TypeConstraint("T3", {"tensor(int32)"}, "Constrain output Y data type as 32-bit integer tensor.")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          auto a_type = ctx.getInputType(0);
          auto b_type = ctx.getInputType(1);
          auto y_type = ctx.getOutputType(0);
          if (nullptr == a_type || nullptr == b_type || nullptr == y_type ||
              a_type->value_case() != ONNX_NAMESPACE::TypeProto::kTensorType ||
              b_type->value_case() != ONNX_NAMESPACE::TypeProto::kTensorType) {
            fail_type_inference("inputs are expected to have tensor type and output type should not be null.");
          }

          // Right now we only support int32
          y_type->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto::INT32);

          matmulShapeInference(ctx, 0, 1);
        }));
static const char* CumSum_ver14_doc = R"DOC(
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
    14,
    OpSchema()
        .SetDoc(CumSum_ver14_doc)
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
            OpSchema::numeric_types_for_math_reduction_ir4(),
            "Constrain input and output types to high-precision numeric tensors.")
        .TypeConstraint("T2", {"tensor(int32)", "tensor(int64)"}, "axis tensor can be int32 or int64 only")
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput));

static const char* Round_ver11_doc = R"DOC(
Round takes one input Tensor and rounds the values, element-wise, meaning
it finds the nearest integer for each value.
In case of halfs, the rule is to round them to the nearest even integer.
If input x is integral, +0, -0, NaN,  or infinite, x itself is returned.
The output tensor has the same shape and type as the input.

Examples:
```
round([0.9]) = [1.0]
round([2.5]) = [2.0]
round([2.3]) = [2.0]
round([1.5]) = [2.0]
round([-4.5]) = [-4.0]
```
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Round,
    11,
    OpSchema()
        .SetDoc(Round_ver11_doc)
        .Input(0, "X", "Input tensor", "T", OpSchema::Single, true, 1, OpSchema::NonDifferentiable)
        .Output(0, "Y", "Output tensor", "T", OpSchema::Single, true, 1, OpSchema::NonDifferentiable)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Det_ver11_doc = R"DOC(
Det calculates determinant of a square matrix or batches of square matrices.
Det takes one input tensor of shape `[*, M, M]`, where `*` is zero or more batch dimensions,
and the inner-most 2 dimensions form square matrices.
The output is a tensor of shape `[*]`, containing the determinants of all input submatrices.
e.g., When the input is 2-D, the output is a scalar(shape is empty: `[]`).
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Det,
    11,
    OpSchema()
        .SetDoc(Det_ver11_doc)
        .Input(0, "X", "Input tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Output(0, "Y", "Output tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to floating-point tensors.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          // Type inference
          propagateElemTypeFromInputToOutput(ctx, 0, 0);

          // Shape inference
          if (hasInputShape(ctx, 0)) {
            const TensorShapeProto& input_shape = ctx.getInputType(0)->tensor_type().shape();
            TensorShapeProto* output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
            const int rank = static_cast<int>(input_shape.dim_size());

            if (rank < 2) {
              fail_shape_inference("Input rank must be >= 2.");
            }

            const auto mat_w = input_shape.dim(rank - 1);
            const auto mat_h = input_shape.dim(rank - 2);
            if (mat_w.has_dim_value() && mat_h.has_dim_value() && (mat_w.dim_value() != mat_h.dim_value())) {
              fail_shape_inference(
                  "The inner-most 2 dimensions must have the same size (mat_w:",
                  mat_w.dim_value(),
                  " != mat_h:",
                  mat_h.dim_value(),
                  ").");
            }

            for (int i = 0; i < rank - 2; ++i) {
              auto* dim = output_shape->add_dim();
              *dim = input_shape.dim(i);
            }
          }
        }));

static const char* NegativeLogLikelihoodLoss_ver13_doc = R"DOC(
A NegativeLogLikelihoodLoss operator computes (weighted) negative log likelihood loss.
Its "input" tensor has the shape of (N, C, d1, d2, ..., dk) where k >= 0.
The "input" tensor contains log-probabilities for input[n, :, d_1, d_2,..., d_k] being in a class of [0, C).
The operator's "target" input tensor has the shape of (N, d1, d2, ..., dk). It encodes class labels (one of C classes)
or it may contain a special value (indicated by an attribute ignore_index) for N x d1 x d2 x ... x dk samples.
The loss value for input[n, :, d_1, d_2,...d_k] being classified as class c = target[n][d_1][d_2]...[d_k] is computed as:

```
loss[n][d_1][d_2]...[d_k] = -input[n][c][d_1][d_2]...[d_k].
```

When an optional "weight" is provided, the sample loss is calculated as:

```
loss[n][d_1][d_2]...[d_k] = -input[n][c][d_1][d_2]...[d_k] * weight[c].
```

loss is zero for the case when target-value equals ignore_index.

```
loss[n][d_1][d_2]...[d_k] = 0, when target[n][d_1][d_2]...[d_k] = ignore_index
```

If "reduction" attribute is set to "none", the operator's output will be the above loss with shape (N, d1, d2, ..., dk).
If "reduction" attribute is set to "mean" (the default attribute value), the output loss is (weight) averaged:

```
mean(loss), if "weight" is not provided,
```

or if weight is provided,

```
sum(loss) / sum(weight[target[n][d_1][d_2]...[d_k]]]), for all samples.
```

If "reduction" attribute is set to "sum", the output is a scalar: `sum(loss)`.

See also https://pytorch.org/docs/stable/nn.html#torch.nn.NLLLoss.

Example 1:

```
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
```

Example 2:

```
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
```

Example 3:

```
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
```
)DOC";

bool BuildContextDependentFunctionBody(
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

  FunctionBuilder builder(functionProto);
  builder.Const1D("const_zero", int64_t(0))
      .Const1D("const_one", int64_t(1))
      .Const1D("axes", int64_t(1))
      .Add("expanded_target = Unsqueeze (target, axes)");

  if (ctx.getAttribute("ignore_index") == nullptr) {
    builder.Add(R"(
      input_gather_element = GatherElements <axis = 1> (input, expanded_target)
      loss_NCdd = Neg (input_gather_element)
      loss_N1dd = Slice (loss_NCdd, const_zero, const_one, const_one)
    )");

    if (!ctx.hasInput(2)) {
      if (reduction_attr == "none") {
        builder.Add("loss = Squeeze (loss_N1dd, axes)");
      } else {
        builder.Add("loss_Ndd = Squeeze (loss_N1dd, axes)");
        if (reduction_attr == "mean") {
          builder.Add("loss = ReduceMean <keepdims = 0> (loss_Ndd)");
        } else {
          builder.Add("loss = ReduceSum <keepdims = 0> (loss_Ndd)");
        }
      }
    } else {
      builder.Add("weight_gather = Gather (weight, target)");
      builder.Add("loss_unweighted = Squeeze (loss_N1dd, axes)");
      if (reduction_attr == "none") {
        builder.Add("loss = Mul (loss_unweighted, weight_gather)");
      } else {
        builder.Add("loss_Ndd = Mul (loss_unweighted, weight_gather)");
        if (reduction_attr == "mean") {
          builder.Add(R"(
            loss_sum = ReduceSum <keepdims = 0> (loss_Ndd)
            weight_gather_sum = ReduceSum <keepdims = 0> (weight_gather)
            loss = Div (loss_sum, weight_gather_sum)
          )");
        } else {
          builder.Add("loss = ReduceSum <keepdims = 0> (loss_Ndd)");
        }
      }
    }
  } else {
    builder.Const1D("const_ignore_index", ctx.getAttribute("ignore_index")->i());
    builder.Add(R"(
      const_zero_target_typed = Sub (expanded_target, expanded_target)
      expanded_target_int64 = Cast <to = 7> (expanded_target)
      mask = Equal (expanded_target_int64, const_ignore_index)
      transform_targets = Where (mask, const_zero_target_typed, expanded_target)
    )");
    builder.Add("input_gather_element = GatherElements <axis = 1> (input, transform_targets)");
    builder.Const1D("const_zero_float", 0.0f);
    if (!float_input) {
      builder.Add("const_zero_casted = Cast (const_zero_float)", "to", static_cast<int64_t>(input_type))
          .Add("input_gather_element_transform = Where (mask, const_zero_casted, input_gather_element)");
    } else
      builder.Add("input_gather_element_transform = Where (mask, const_zero_float, input_gather_element)");
    builder.Add("loss_NCdd = Neg (input_gather_element_transform)");
    builder.Add("loss_N1dd = Slice (loss_NCdd, const_zero, const_one, const_one)");

    if (!ctx.hasInput(2)) {
      builder.Add("squeeze_mask = Squeeze (mask, axes)");
      builder.Const1D("const_one_float", 1.0f);
      if (!float_input) {
        builder.Add("const_one_casted = Cast (const_one_float)", "to", static_cast<int64_t>(input_type))
            .Add("weight_gather = Where (squeeze_mask, const_zero_casted, const_one_casted)");
      } else
        builder.Add("weight_gather = Where (squeeze_mask, const_zero_float, const_one_float)");

    } else {
      builder.Add("weight_gather_temp = Gather (weight, transform_targets)");
      builder.Add(
          float_input ? "weight_gather_temp_1 = Where (mask, const_zero_float, weight_gather_temp)"
                      : "weight_gather_temp_1 = Where (mask, const_zero_casted, weight_gather_temp)");
      builder.Add("weight_gather = Squeeze (weight_gather_temp_1, axes)");
    }

    builder.Add("loss_unweighted = Squeeze (loss_N1dd, axes)");
    if (reduction_attr == "none") {
      builder.Add("loss = Mul (loss_unweighted, weight_gather)");
    } else {
      builder.Add("loss_Ndd = Mul (loss_unweighted, weight_gather)");
      if (reduction_attr == "mean") {
        builder.Add(R"(
            loss_sum = ReduceSum <keepdims = 0> (loss_Ndd)
            weight_gather_sum = ReduceSum <keepdims = 0> (weight_gather)
            loss = Div (loss_sum, weight_gather_sum)
        )");
      } else {
        builder.Add("loss = ReduceSum <keepdims = 0> (loss_Ndd)");
      }
    }
  }

  schema.BuildFunction(functionProto);
  return true;
}

ONNX_OPERATOR_SET_SCHEMA(
    NegativeLogLikelihoodLoss,
    13,
    OpSchema()
        .SetDoc(NegativeLogLikelihoodLoss_ver13_doc)
        .Input(
            0,
            "input",
            "Input tensor of shape (N, C) or (N, C, d1, d2, ..., dk).",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .Input(
            1,
            "target",
            "Target tensor of shape (N) or (N, d1, d2, ..., dk). Target element value shall be in range of [0, C). "
            "If ignore_index is specified, it may have a value outside [0, C) and the target values should either be "
            "in the range [0, C) or have the value ignore_index.",
            "Tind",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Input(
            2,
            "weight",
            "Optional rescaling weight tensor. "
            "If given, it has to be a tensor of size C. Otherwise, it is treated as if having all ones.",
            "T",
            OpSchema::Optional,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Output(0, "loss", "The negative log likelihood loss", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
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
        .SetContextDependentFunctionBodyBuilder(BuildContextDependentFunctionBody)
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
              fail_shape_inference("Input rank must be >= 2.")
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

void einsumRankInference(ONNX_NAMESPACE::InferenceContext& ctx, std::string equation) {
  const size_t numInputs = ctx.getNumInputs();
  if (numInputs < 1 || !hasNInputShapes(ctx, static_cast<int>(numInputs))) {
    return;
  }

  auto* output_shape = getOutputShape(ctx, 0);
  std::string left_equation;

  equation.erase(std::remove(equation.begin(), equation.end(), ' '),
                 equation.end()); // Remove space char
  auto mid_index = equation.find("->");
  if (mid_index != std::string::npos) {
    // Separate right and left hand sides of the equation
    left_equation = equation.substr(0, mid_index);
  } else {
    // No right hand side
    left_equation = equation;
  }

  std::string term;
  size_t num_operands = 0;
  size_t num_ellipsis = 0;
  size_t num_ellipsis_indices = 0;

  // Parse the left-hand side
  std::stringstream str(left_equation);
  while (!str.eof()) {
    std::getline(str, term, ',');
    auto ellipsis_index = term.find("...");
    if (numInputs <= num_operands) {
      fail_shape_inference("Number of input tensors does not match the operands in the equation.");
    }
    size_t rank = ctx.getInputType(num_operands)->tensor_type().shape().dim_size();
    if (ellipsis_index != std::string::npos) {
      // If there is an ellipsis, the number of dimensions it represents
      // must be total dim - letter dimensions
      if (num_ellipsis == 0) {
        if (rank + 3 < term.size()) {
          fail_shape_inference("Ellipsis represents incompatible dimensions.");
        }
        num_ellipsis_indices = rank - term.size() + 3;
      } else { // ellipsis has been seen before. Check that if dimensions
               // are compatible
        if (num_ellipsis_indices != rank - term.size() + 3) {
          fail_shape_inference("Ellipsis represents incompatible dimensions.");
        }
      }
      num_ellipsis++;
    } else {
      if (rank != term.size()) {
        fail_shape_inference("Rank of input ", num_operands, " does not match the equation indices.");
      }
    }
    num_operands++;
  }

  if (numInputs != num_operands) {
    fail_shape_inference("Number of input tensors does not match the operands in the equation.");
  }

  const size_t number_of_letters = 26;
  size_t num_letter_occurrences[number_of_letters] = {0};
  // Parse the provided right-hand side
  if (mid_index != std::string::npos) {
    std::string right_equation = equation.substr(mid_index + 2);
    auto right_ellipsis_index = right_equation.find("...");
    if (right_ellipsis_index != std::string::npos) { // Right-hand side contains ellipsis
      for (size_t i = 0; i < num_ellipsis_indices; ++i) {
        output_shape->add_dim();
      }
    }
    for (char c : right_equation) { // Add a dimension per each character
                                    // in right hand equation
      if (c != '.') {
        output_shape->add_dim();
      }
    }
  } else { // Infer the dimension for right-hand side
    // If there's an ellipsis, add it's corresponding dimensions
    for (size_t i = 0; i < num_ellipsis_indices; i++) {
      output_shape->add_dim();
    }
    for (size_t i = 0; i < left_equation.size(); i++) { // Count chars that appear exactly once on left hand side
      if ((left_equation.at(i) != ',') && (left_equation.at(i) != '.')) {
        num_letter_occurrences[left_equation.at(i) - 'a']++;
      }
    }
    for (size_t index = 0; index < number_of_letters; index++) {
      if (num_letter_occurrences[index] == 1) {
        output_shape->add_dim();
      }
    }
  }
}

static const char* Einsum_ver12_doc = R"DOC(
An einsum of the form `term1, term2 -> output-term` produces an output tensor using the following equation

```
output[output-term] = reduce-sum( input1[term1] * input2[term] )
```

where the reduce-sum performs a summation over all the indices occurring in the input terms (term1, term2)
that do not occur in the output-term.

The Einsum operator evaluates algebraic tensor operations on a sequence of tensors, using the Einstein summation
convention. The equation string contains a comma-separated sequence of lower case letters. Each term corresponds to
an operand tensor, and the characters within the terms correspond to operands dimensions.

This sequence may be followed by "->" to separate the left and right hand side of the equation.
If the equation contains "->" followed by the right-hand side, the explicit (not classical) form of the Einstein
summation is performed, and the right-hand side indices indicate output tensor dimensions. In other cases,
output indices are (implicitly) set to the alphabetically sorted sequence of indices appearing exactly once in the
equation.

When a dimension character is repeated in the left-hand side, it represents summation along the dimension.

The equation may contain ellipsis ("...") to enable broadcasting. Ellipsis must indicate a fixed number of dimensions.
Specifically, every occurrence of ellipsis in the equation must represent the same number of dimensions.
The right-hand side may contain exactly one ellipsis. In implicit mode, the ellipsis dimensions are set to the
beginning of the output. The equation string may contain space (U+0020) character.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Einsum,
    12,
    OpSchema()
        .SetDoc(Einsum_ver12_doc)
        .Attr("equation", "Einsum expression string.", AttributeProto::STRING)
        .Input(0, "Inputs", "Operands", "T", OpSchema::Variadic, true, 1, OpSchema::Differentiable)
        .Output(0, "Output", "Output tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .TypeConstraint(
            "T",
            OpSchema::all_numeric_types(),
            "Constrain input and output types to all numerical tensor types.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          // Type inference
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          std::string equation = getAttribute(ctx, "equation", "");
          if (equation.compare("") == 0) {
            return;
          }
          einsumRankInference(ctx, equation);
        }));

const char* reduction_doc_sce =
    "Type of reduction to apply to loss: none, sum, mean(default). "
    "'none': no reduction will be applied, "
    "'sum': the output will be summed. "
    "'mean': the sum of the output will be divided by the number of "
    "elements in the output.";

static const char* SoftmaxCrossEntropyLoss_ver13_doc =
    R"DOC(Loss function that measures the softmax cross entropy
between 'scores' and 'labels'.
This operator first computes a loss tensor whose shape is identical to the labels input.
If the input is 2-D with shape (N, C), the loss tensor may be a N-element vector L = (l_1, l_2, ..., l_N).
If the input is N-D tensor with shape (N, C, D1, D2, ..., Dk),
the loss tensor L may have (N, D1, D2, ..., Dk) as its shape and L[i,][j_1][j_2]...[j_k] denotes a scalar element in L.
After L is available, this operator can optionally do a reduction operator.

* shape(scores): (N, C) where C is the number of classes, or (N, C, D1, D2,..., Dk),
  with K >= 1 in case of K-dimensional loss.
* shape(labels): (N) where each value is 0 <= labels[i] <= C-1, or (N, D1, D2,..., Dk),
  with K >= 1 in case of K-dimensional loss.

The loss for one sample, l_i, can caculated as follows:
```
l[i][d1][d2]...[dk] = -y[i][c][d1][d2]..[dk], where i is the index of classes.
```
or
```
l[i][d1][d2]...[dk] = -y[i][c][d1][d2]..[dk] * weights[c], if 'weights' is provided.
```

loss is zero for the case when label-value equals ignore_index.
```
l[i][d1][d2]...[dk]  = 0, when labels[n][d1][d2]...[dk] = ignore_index
```

where:
```
p = Softmax(scores)
y = Log(p)
c = labels[i][d1][d2]...[dk]
```

Finally, L is optionally reduced:

* If reduction = 'none', the output is L with shape (N, D1, D2, ..., Dk).
* If reduction = 'sum', the output is scalar: Sum(L).
* If reduction = 'mean', the output is scalar: ReduceMean(L), or if weight is provided: `ReduceSum(L) / ReduceSum(W)`,
  where tensor W is of shape `(N, D1, D2, ..., Dk)` and `W[n][d1][d2]...[dk] = weights[labels[i][d1][d2]...[dk]]`.
)DOC";

bool BuildContextDependentFunctionBodySCE(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  FunctionBuilder builder(functionProto);
  // Using stable implementation of LogSoftmax
  builder //
      .Const("Shape3D", std::vector<int64_t>({0, 0, -1})) //
      .Add(R"(
        X_NCD = Reshape (scores, Shape3D)
        X_NDC = Transpose <perm = [0, 2, 1]> (X_NCD)
        X_LogSM = LogSoftmax <axis = 2> (X_NDC)
        X_LogSM_NCD = Transpose <perm = [0, 2, 1]> (X_LogSM)
        X_shape = Shape (scores)
        X_Log = Reshape (X_LogSM_NCD, X_shape)
      )");

  // Review(mzs): Ideally we want to reuse the output from Log for sub-graph
  // output as well but looking at the graph resolve code it does not include
  // graph outputs as intermediate outputs, hence if intermediate X_log is
  // renamed as log_prob then it will be treated as graph output and will not be
  // available to NegativeLogLikelihoodLoss. May be my understanding is
  // incorrect or there is a bug in function population code in ORTbut I will
  // dig further to be 100%. In the meantime we just replicate the log.
  if (ctx.hasOutput(1)) {
    builder.Add("log_prob = Identity (X_Log)");
  }

  builder.Add(
      ctx.hasInput(2)
          ? "output = NegativeLogLikelihoodLoss <reduction : string = @reduction, ignore_index : int = @ignore_index> (X_Log, labels, weights)"
          : "output = NegativeLogLikelihoodLoss <reduction : string = @reduction, ignore_index : int = @ignore_index> (X_Log, labels)");

  schema.BuildFunction(functionProto);
  return true;
}

ONNX_OPERATOR_SET_SCHEMA(
    SoftmaxCrossEntropyLoss,
    13,
    OpSchema()
        .SetDoc(SoftmaxCrossEntropyLoss_ver13_doc)
        .Attr("reduction", reduction_doc_sce, AttributeProto::STRING, std::string("mean"))
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
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .Input(
            1,
            "labels",
            "The ground truth output tensor, with shape [batch_size], or "
            "[batch_size, D1, D2, ..., Dk], where K is the number of dimensions. "
            "Labels element value shall be in range of [0, C). "
            "If ignore_index is specified, it may have a value outside [0, C) and the label values should either be "
            "in the range [0, C) or have the value ignore_index.",
            "Tind",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Input(
            2,
            "weights",
            "A manual rescaling weight given to each class. If given, it has to "
            "be a 1D Tensor assigning weight to each of the classes. Otherwise, "
            "it is treated as if having all ones.",
            "T",
            OpSchema::Optional,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Output(
            0,
            "output",
            "Weighted loss float Tensor. If reduction is 'none', this has the "
            "shape of [batch_size], or [batch_size, D1, D2, ..., Dk] in case of "
            "K-dimensional loss. Otherwise, it is a scalar.",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .Output(
            1,
            "log_prob",
            "Log probability tensor. If the output of softmax is prob, its value is log(prob).",
            "T",
            OpSchema::Optional,
            true,
            1,
            OpSchema::Differentiable)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
            "Constrain input and output types to float tensors.")
        .TypeConstraint("Tind", {"tensor(int32)", "tensor(int64)"}, "Constrain target to integer types")
        .SetContextDependentFunctionBodyBuilder(BuildContextDependentFunctionBodySCE)
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

template <typename T>
static T get_scalar_value_from_tensor(const ONNX_NAMESPACE::TensorProto* t) {
  if (t == nullptr) {
    return T{};
  }

  auto data_type = t->data_type();
  switch (data_type) {
    case ONNX_NAMESPACE::TensorProto::FLOAT:
      return static_cast<T>(ONNX_NAMESPACE::ParseData<float>(t).at(0));
    case ONNX_NAMESPACE::TensorProto::DOUBLE:
      return static_cast<T>(ONNX_NAMESPACE::ParseData<double>(t).at(0));
    case ONNX_NAMESPACE::TensorProto::INT32:
      return static_cast<T>(ONNX_NAMESPACE::ParseData<int32_t>(t).at(0));
    case ONNX_NAMESPACE::TensorProto::INT64:
      return static_cast<T>(ONNX_NAMESPACE::ParseData<int64_t>(t).at(0));
    default:
      fail_shape_inference("Unsupported input data type of ", data_type);
  }
}

static const char* DFT_ver17_doc = R"DOC(Computes the discrete Fourier transform of input.)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    DFT,
    17,
    OpSchema()
        .SetDoc(DFT_ver17_doc)
        .Attr(
            "onesided",
            "If onesided is 1, only values for w in [0, 1, 2, ..., floor(n_fft/2) + 1] are returned because "
            "the real-to-complex Fourier transform satisfies the conjugate symmetry, i.e., X[m, w] = X[m,w]=X[m,n_fft-w]*. "
            "Note if the input or window tensors are complex, then onesided output is not possible. "
            "Enabling onesided with real inputs performs a Real-valued fast Fourier transform (RFFT). "
            "When invoked with real or complex valued input, the default value is 0. "
            "Values can be 0 or 1.",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Attr(
            "axis",
            "The axis on which to perform the DFT. By default this value is set to 1, which corresponds to the first dimension after the batch index.",
            AttributeProto::INT,
            static_cast<int64_t>(1))
        .Attr(
            "inverse",
            "Whether to perform the inverse discrete fourier transform. By default this value is set to 0, which corresponds to false.",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Input(
            0,
            "input",
            "For real input, the following shape is expected: [batch_idx][signal_dim1][signal_dim2]...[signal_dimN][1]. "
            "For complex input, the following shape is expected: [batch_idx][signal_dim1][signal_dim2]...[signal_dimN][2]. "
            "The first dimension is the batch dimension. "
            "The following N dimentions correspond to the signal's dimensions. "
            "The final dimension represents the real and imaginary parts of the value in that order.",
            "T1",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Input(
            1,
            "dft_length",
            "The length of the signal."
            "If greater than the axis dimension, the signal will be zero-padded up to dft_length. "
            "If less than the axis dimension, only the first dft_length values will be used as the signal. "
            "It's an optional value. ",
            "T2",
            OpSchema::Optional,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Output(
            0,
            "output",
            "The Fourier Transform of the input vector."
            "If onesided is 0, the following shape is expected: [batch_idx][signal_dim1][signal_dim2]...[signal_dimN][2]. "
            "If axis=1 and onesided is 1, the following shape is expected: [batch_idx][floor(signal_dim1/2)+1][signal_dim2]...[signal_dimN][2]. "
            "If axis=2 and onesided is 1, the following shape is expected: [batch_idx][signal_dim1][floor(signal_dim2/2)+1]...[signal_dimN][2]. "
            "If axis=N and onesided is 1, the following shape is expected: [batch_idx][signal_dim1][signal_dim2]...[floor(signal_dimN/2)+1][2]. "
            "The signal_dim at the specified axis is equal to the dft_length.",
            "T1")
        .TypeConstraint(
            "T1",
            {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
            "Constrain input and output types to float tensors.")
        .TypeConstraint("T2", {"tensor(int32)", "tensor(int64)"}, "Constrain scalar length types to int64_t.")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          bool is_onesided = static_cast<bool>(getAttribute(ctx, "onesided", 0));
          bool inverse = static_cast<bool>(getAttribute(ctx, "inverse", 0));

          if (inverse && is_onesided) {
            fail_shape_inference("is_onesided and inverse attributes cannot be enabled at the same time");
          }

          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (!hasInputShape(ctx, 0)) {
            // If no shape is available for the input, skip shape inference...
            return;
          }

          // In general the output shape will match the input shape exactly
          // So initialize the output shape with the input shape
          auto& input_shape = getInputShape(ctx, 0);
          ONNX_NAMESPACE::TensorShapeProto result_shape_proto = input_shape;

          // Get the axis where the DFT will be performed.
          auto axis = static_cast<int>(getAttribute(ctx, "axis", 1));
          auto rank = input_shape.dim_size();

          if (!(-rank <= axis && axis < rank)) {
            fail_shape_inference("axis attribute value ", axis, " is invalid for a tensor of rank ", rank);
          }

          auto axis_idx = (axis >= 0 ? axis : axis + rank);

          // If dft_length is specified, then we should honor the shape.
          // Set the output dimension to match the dft_length on the axis.
          // If onesided this will be adjusted later on...
          const TensorProto* dft_length = nullptr;
          if (ctx.hasInput(1)) {
            dft_length = ctx.getInputData(1);
            if (dft_length == nullptr) {
              // If we cannot read the dft_length, we cannot infer shape
              // return...
              return;
            }
          }

          if (nullptr != dft_length) {
            if (dft_length->dims_size() != 0) {
              fail_shape_inference("dft_length input must be a scalar.");
            }
            auto dft_length_value = get_scalar_value_from_tensor<int64_t>(dft_length);
            result_shape_proto.mutable_dim(axis_idx)->set_dim_value(dft_length_value);
          }
          // When DFT is onesided, the output shape is half the size of the input shape
          // along the specified axis.
          if (is_onesided) {
            auto axis_dimension = result_shape_proto.dim(axis_idx);
            // We need to update the output shape dimension along the specified axis,
            // but sometimes the dimension will be a free dimension or be otherwise unset.
            // Only perform inference when a input dimension value exists.
            if (axis_dimension.has_dim_value()) {
              auto original_signal_size = axis_dimension.dim_value();
              auto half_signal_size = (original_signal_size >> 1) + 1;
              result_shape_proto.mutable_dim(axis_idx)->set_dim_value(half_signal_size);
            } else {
              // Clear the value and param (which would otherwie be inherited from the input).
              result_shape_proto.mutable_dim(axis_idx)->clear_dim_value();
              result_shape_proto.mutable_dim(axis_idx)->clear_dim_param();
            }
          }

          // Coerce the last dimension to 2.
          auto dim_size = static_cast<int64_t>(result_shape_proto.dim_size());
          result_shape_proto.mutable_dim(static_cast<int>(dim_size - 1))->set_dim_value(2);

          updateOutputShape(ctx, 0, result_shape_proto);
        }));

std::function<void(OpSchema&)> CosineSumWindowOpDocGenerator(const char* name) {
  return [=](OpSchema& schema) {
    std::string doc;
    POPULATE_OP_DOC_STR(doc = R"DOC(
Generates a {name} window as described in the paper https://ieeexplore.ieee.org/document/1455106.
)DOC";
                        ReplaceAll(doc, "{name}", name););

    schema.SetDoc(doc);
    schema.Attr(
        "output_datatype",
        "The data type of the output tensor. "
        "Strictly must be one of the values from DataType enum in TensorProto whose values correspond to T2. "
        "The default value is 1 = FLOAT. ",
        AttributeProto::INT,
        static_cast<int64_t>(TensorProto_DataType_FLOAT));
    schema.Attr(
        "periodic",
        "If 1, returns a window to be used as periodic function. If 0, return a symmetric window. "
        "When 'periodic' is specified, hann computes a window of length size + 1 and returns the first size points. "
        "The default value is 1. ",
        AttributeProto::INT,
        static_cast<int64_t>(1));
    schema.Input(
        0,
        "size",
        "A scalar value indicating the length of the window.",
        "T1",
        OpSchema::Single,
        true,
        1,
        OpSchema::NonDifferentiable);
    std::string output_doc("A {name} window with length: size. The output has the shape: [size].");
    ReplaceAll(output_doc, "{name}", name);
    schema.Output(0, "output", output_doc, "T2", OpSchema::Single, true, 1, OpSchema::NonDifferentiable);
    schema.TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
      // Update the output data type to the output_datatype
      auto output_datatype = getAttribute(ctx, "output_datatype", static_cast<int64_t>(TensorProto_DataType_FLOAT));
      updateOutputElemType(ctx, 0, output_datatype);

      if (!hasInputShape(ctx, 0)) {
        // If no shape is available for the input, skip shape inference.
        return;
      }

      const auto* size = ctx.getInputData(0);
      if (size == nullptr) {
        // Size is not available, so return early
        return;
      }

      if (size->dims_size() != 0) {
        fail_shape_inference("size input must be a scalar.");
      }

      auto size_value = get_scalar_value_from_tensor<int64_t>(size);
      if (size_value <= 0) {
        fail_shape_inference("size input must be greater than 0.");
      }

      ONNX_NAMESPACE::TensorShapeProto result_shape;
      result_shape.add_dim()->set_dim_value(size_value);
      updateOutputShape(ctx, 0, result_shape);
    });
  };
}

ONNX_OPERATOR_SET_SCHEMA(
    HannWindow,
    17,
    OpSchema()
        .FillUsing(CosineSumWindowOpDocGenerator("Hann"))
        .TypeConstraint("T1", {"tensor(int32)", "tensor(int64)"}, "Constrain the input size to int64_t.")
        .TypeConstraint("T2", OpSchema::all_numeric_types_ir4(), "Constrain output types to numeric tensors.")
        .FunctionBody(R"ONNX(
        {
          A0 = Constant <value = float {0.5}>()
          A1 = Constant <value = float {0.5}>()
          A2 = Constant <value = float {0.0}>()
          Zero = Constant <value = float {0.0}>()
          One = Constant <value = float {1.0}>()
          Two = Constant <value = float {2.0}>()
          Tau = Constant <value = float {6.2831853}>()
          Periodic_Size_FP = Cast <to = 1> (size)
          Symmetric_Size_FP = Sub(Periodic_Size_FP, One)
          IsPeriodic = Constant <value_int : int = @periodic>()
          IsPeriodic_FP = Cast <to = 1> (IsPeriodic)
          IsSymmetric_FP = Sub(One, IsPeriodic_FP)
          Periodic_Component = Mul(Periodic_Size_FP, IsPeriodic_FP)
          Symmetric_Component = Mul(Symmetric_Size_FP, IsSymmetric_FP)
          Size_FP = Add(Periodic_Component, Symmetric_Component)
          AngularIncrement = Div (Tau, Size_FP)
          Range = Range (Zero, Periodic_Size_FP, One)
          RangeAngular = Mul (Range, AngularIncrement)
          TwoRangeAngular = Mul (RangeAngular, Two)
          CosTwoRangeAngular = Cos (TwoRangeAngular)
          A2_Component = Mul (A2, CosTwoRangeAngular)
          CosRangeAngular = Cos (RangeAngular)
          A1_Component = Mul (A1, CosRangeAngular)
          Temp0 = Sub (A0, A1_Component)
          Temp1 = Add (Temp0, A2_Component)
          output = Cast <to : int = @output_datatype> (Temp1)
        }
        )ONNX"));

ONNX_OPERATOR_SET_SCHEMA(
    HammingWindow,
    17,
    OpSchema()
        .FillUsing(CosineSumWindowOpDocGenerator("Hamming"))
        .TypeConstraint("T1", {"tensor(int32)", "tensor(int64)"}, "Constrain the input size to int64_t.")
        .TypeConstraint("T2", OpSchema::all_numeric_types_ir4(), "Constrain output types to numeric tensors.")
        .FunctionBody(R"ONNX(
        {
          A0 = Constant <value = float {0.54347826087}>()
          A1 = Constant <value = float {0.45652173913}>()
          A2 = Constant <value = float {0.0}>()
          Zero = Constant <value = float {0.0}>()
          One = Constant <value = float {1.0}>()
          Two = Constant <value = float {2.0}>()
          Tau = Constant <value = float {6.2831853}>()
          Periodic_Size_FP = Cast <to = 1> (size)
          Symmetric_Size_FP = Sub(Periodic_Size_FP, One)
          IsPeriodic = Constant <value_int : int = @periodic>()
          IsPeriodic_FP = Cast <to = 1> (IsPeriodic)
          IsSymmetric_FP = Sub(One, IsPeriodic_FP)
          Periodic_Component = Mul(Periodic_Size_FP, IsPeriodic_FP)
          Symmetric_Component = Mul(Symmetric_Size_FP, IsSymmetric_FP)
          Size_FP = Add(Periodic_Component, Symmetric_Component)
          AngularIncrement = Div (Tau, Size_FP)
          Range = Range (Zero, Periodic_Size_FP, One)
          RangeAngular = Mul (Range, AngularIncrement)
          TwoRangeAngular = Mul (RangeAngular, Two)
          CosTwoRangeAngular = Cos (TwoRangeAngular)
          A2_Component = Mul (A2, CosTwoRangeAngular)
          CosRangeAngular = Cos (RangeAngular)
          A1_Component = Mul (A1, CosRangeAngular)
          Temp0 = Sub (A0, A1_Component)
          Temp1 = Add (Temp0, A2_Component)
          output = Cast <to : int = @output_datatype> (Temp1)
        }
        )ONNX"));

ONNX_OPERATOR_SET_SCHEMA(
    BlackmanWindow,
    17,
    OpSchema()
        .FillUsing(CosineSumWindowOpDocGenerator("Blackman"))
        .TypeConstraint("T1", {"tensor(int32)", "tensor(int64)"}, "Constrain the input size to int64_t.")
        .TypeConstraint("T2", OpSchema::all_numeric_types_ir4(), "Constrain output types to numeric tensors.")
        .FunctionBody(R"ONNX(
        {
          A0 = Constant <value = float {0.42}>()
          A1 = Constant <value = float {0.5}>()
          A2 = Constant <value = float {0.08}>()
          Zero = Constant <value = float {0.0}>()
          One = Constant <value = float {1.0}>()
          Two = Constant <value = float {2.0}>()
          Tau = Constant <value = float {6.2831853}>()
          Periodic_Size_FP = Cast <to = 1> (size)
          Symmetric_Size_FP = Sub(Periodic_Size_FP, One)
          IsPeriodic = Constant <value_int : int = @periodic>()
          IsPeriodic_FP = Cast <to = 1> (IsPeriodic)
          IsSymmetric_FP = Sub(One, IsPeriodic_FP)
          Periodic_Component = Mul(Periodic_Size_FP, IsPeriodic_FP)
          Symmetric_Component = Mul(Symmetric_Size_FP, IsSymmetric_FP)
          Size_FP = Add(Periodic_Component, Symmetric_Component)
          AngularIncrement = Div (Tau, Size_FP)
          Range = Range (Zero, Periodic_Size_FP, One)
          RangeAngular = Mul (Range, AngularIncrement)
          TwoRangeAngular = Mul (RangeAngular, Two)
          CosTwoRangeAngular = Cos (TwoRangeAngular)
          A2_Component = Mul (A2, CosTwoRangeAngular)
          CosRangeAngular = Cos (RangeAngular)
          A1_Component = Mul (A1, CosRangeAngular)
          Temp0 = Sub (A0, A1_Component)
          Temp1 = Add (Temp0, A2_Component)
          output = Cast <to : int = @output_datatype> (Temp1)
        }
        )ONNX"));

static const char* MelWeightMatrix_ver17_doc = R"DOC(
Generate a MelWeightMatrix that can be used to re-weight a Tensor containing a linearly sampled frequency spectra (from DFT or STFT) into num_mel_bins frequency information based on the [lower_edge_hertz, upper_edge_hertz] range on the mel scale.
This function defines the mel scale in terms of a frequency in hertz according to the following formula:

    mel(f) = 2595 * log10(1 + f/700)

In the returned matrix, all the triangles (filterbanks) have a peak value of 1.0.

The returned MelWeightMatrix can be used to right-multiply a spectrogram S of shape [frames, num_spectrogram_bins] of linear scale spectrum values (e.g. STFT magnitudes) to generate a "mel spectrogram" M of shape [frames, num_mel_bins].
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    MelWeightMatrix,
    17,
    OpSchema()
        .SetDoc(MelWeightMatrix_ver17_doc)
        .Attr(
            "output_datatype",
            "The data type of the output tensor. "
            "Strictly must be one of the values from DataType enum in TensorProto whose values correspond to T3. "
            "The default value is 1 = FLOAT. ",
            AttributeProto::INT,
            static_cast<int64_t>(TensorProto_DataType_FLOAT))
        .Input(
            0,
            "num_mel_bins",
            "The number of bands in the mel spectrum.",
            "T1",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Input(
            1,
            "dft_length",
            "The size of the original DFT. "
            "The size of the original DFT is used to infer the size of the onesided DFT, which is understood to be floor(dft_length/2) + 1, i.e. the spectrogram only contains the nonredundant DFT bins.",
            "T1",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Input(
            2,
            "sample_rate",
            "Samples per second of the input signal used to create the spectrogram. Used to figure out the frequencies corresponding to each spectrogram bin, which dictates how they are mapped into the mel scale.",
            "T1",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Input(
            3,
            "lower_edge_hertz",
            "Lower bound on the frequencies to be included in the mel spectrum. This corresponds to the lower edge of the lowest triangular band.",
            "T2",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Input(
            4,
            "upper_edge_hertz",
            "The desired top edge of the highest frequency band.",
            "T2",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Output(
            0,
            "output",
            "The Mel Weight Matrix. "
            "The output has the shape: [floor(dft_length/2) + 1][num_mel_bins].",
            "T3",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .TypeConstraint("T1", {"tensor(int32)", "tensor(int64)"}, "Constrain to integer tensors.")
        .TypeConstraint(
            "T2",
            {"tensor(float)", "tensor(float16)", "tensor(double)", "tensor(bfloat16)"},
            "Constrain to float tensors")
        .TypeConstraint("T3", OpSchema::all_numeric_types_ir4(), "Constrain to any numerical types.")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          auto output_datatype = getAttribute(ctx, "output_datatype", static_cast<int64_t>(TensorProto_DataType_FLOAT));
          updateOutputElemType(ctx, 0, output_datatype);

          if (!hasInputShape(ctx, 0) || !hasInputShape(ctx, 1)) {
            return;
          }

          const auto* num_mel_bins = ctx.getInputData(0);
          const auto* dft_length = ctx.getInputData(1);
          if (nullptr == num_mel_bins || nullptr == dft_length) {
            return;
          }

          int64_t num_mel_bins_value = -1;
          int64_t dft_length_value = -1;
          if (num_mel_bins->dims_size() != 0) {
            fail_shape_inference("num_mel_bins input must be scalar.");
          }
          num_mel_bins_value = get_scalar_value_from_tensor<int64_t>(num_mel_bins);

          if (dft_length->dims_size() != 0) {
            fail_shape_inference("dft_length input must be scalar.");
          }
          dft_length_value = get_scalar_value_from_tensor<int64_t>(dft_length);

          if (num_mel_bins_value > 0 && dft_length_value > 0) {
            ONNX_NAMESPACE::TensorShapeProto result_shape;
            result_shape.add_dim()->set_dim_value(static_cast<int64_t>((dft_length_value >> 1) + 1));
            result_shape.add_dim()->set_dim_value(num_mel_bins_value);
            updateOutputShape(ctx, 0, result_shape);
          }
        }));

static const char* STFT_ver17_doc = R"DOC(Computes the Short-time Fourier Transform of the signal.)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    STFT,
    17,
    OpSchema()
        .SetDoc(STFT_ver17_doc)
        .Attr(
            "onesided",
            "If onesided is 1, only values for w in [0, 1, 2, ..., floor(n_fft/2) + 1] are returned because "
            "the real-to-complex Fourier transform satisfies the conjugate symmetry, i.e., X[m, w] = X[m,w]=X[m,n_fft-w]*. "
            "Note if the input or window tensors are complex, then onesided output is not possible. "
            "Enabling onesided with real inputs performs a Real-valued fast Fourier transform (RFFT)."
            "When invoked with real or complex valued input, the default value is 1. "
            "Values can be 0 or 1.",
            AttributeProto::INT,
            static_cast<int64_t>(1))
        .Input(
            0,
            "signal",
            "Input tensor representing a real or complex valued signal. "
            "For real input, the following shape is expected: [batch_size][signal_length][1]. "
            "For complex input, the following shape is expected: [batch_size][signal_length][2], where "
            "[batch_size][signal_length][0] represents the real component and [batch_size][signal_length][1] represents the imaginary component of the signal.",
            "T1",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Input(
            1,
            "frame_step",
            "The number of samples to step between successive DFTs.",
            "T2",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Input(
            2,
            "window",
            "A tensor representing the window that will be slid over the signal."
            "The window must have rank 1 with shape: [window_shape]. "
            "It's an optional value. ",
            "T1",
            OpSchema::Optional,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Input(
            3,
            "frame_length",
            "A scalar representing the size of the DFT. "
            "It's an optional value.",
            "T2",
            OpSchema::Optional,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Output(
            0,
            "output",
            "The Short-time Fourier Transform of the signals."
            "If onesided is 1, the output has the shape: [batch_size][frames][dft_unique_bins][2], where dft_unique_bins is frame_length // 2 + 1 (the unique components of the DFT) "
            "If onesided is 0, the output has the shape: [batch_size][frames][frame_length][2], where frame_length is the length of the DFT.",
            "T1",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .TypeConstraint(
            "T1",
            {"tensor(float)", "tensor(float16)", "tensor(double)", "tensor(bfloat16)"},
            "Constrain signal and output to float tensors.")
        .TypeConstraint("T2", {"tensor(int32)", "tensor(int64)"}, "Constrain scalar length types to int64_t.")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);

          // Get signal size
          // The signal size is needed to perform inference because the size of the signal
          // is needed to compute the number of DFTs in the output.
          //
          // 1) Check if shape exists, return if not
          // 2) Get the shape
          // 3) Check if signal dim value exists, return if not
          if (!hasInputShape(ctx, 0)) {
            return;
          }

          auto& input_shape = getInputShape(ctx, 0);
          auto signal_dim = input_shape.dim(1);
          if (!signal_dim.has_dim_value()) {
            return;
          }
          auto signal_size = signal_dim.dim_value();

          // The frame step is a required input.
          // Its value is needed to compute the number output nDFTs, so return early is missing.
          const auto* frame_step = ctx.getInputData(1);
          if (nullptr == frame_step) {
            return;
          }
          auto frame_step_value = get_scalar_value_from_tensor<int64_t>(frame_step);

          // Determine the size of the DFT based on the 2 optional inputs window and frame_length.
          // One must be set.
          int64_t dft_size = -1;
          const TensorProto* frame_length = nullptr;
          if (ctx.hasInput(3)) {
            frame_length = ctx.getInputData(3);
            if (frame_length == nullptr) {
              // If we cannot read the frame_length, we cannot infer shape
              // return...
              return;
            }
          }

          const TensorShapeProto* window_shape = nullptr;
          if (ctx.getNumInputs() >= 3) {
            window_shape = getOptionalInputShape(ctx, 2);
          } else {
            window_shape = nullptr;
          }

          if (window_shape == nullptr && frame_length == nullptr) {
            // STFT expects to have at least one of these inputs set: [window, frame_length],
            // but they may not be available at shape inference time
            return;
          } else if (window_shape != nullptr && frame_length != nullptr) {
            if (frame_length->dims_size() != 0) {
              fail_shape_inference("frame_length input must be scalar.");
            }
            auto frame_length_value = get_scalar_value_from_tensor<int64_t>(frame_length);

            // Ensure that the window length and the dft_length match.
            if (window_shape->dim_size() != 1) {
              fail_shape_inference("window input must have rank = 1.");
            }
            if (window_shape->dim(0).has_dim_value()) {
              auto window_length = window_shape->dim(0).dim_value();
              if (window_length != frame_length_value) {
                fail_type_inference(
                    "If STFT has both a window input and frame_length specified, the dimension of the window must match the frame_length specified!");
              }
            }

            dft_size = frame_length_value;
          } else if (window_shape != nullptr) {
            // Ensure that the window length and the dft_length match.
            if (window_shape->dim_size() != 1) {
              fail_shape_inference("window input must have rank = 1.");
            }
            if (window_shape->dim(0).has_dim_value()) {
              dft_size = window_shape->dim(0).dim_value();
            } else {
              // Cannot determine the window size, and there is no frame_length,
              // So shape inference cannot proceed.
              return;
            }
          } else if (frame_length != nullptr) {
            if (frame_length->dims_size() != 0) {
              fail_shape_inference("frame_length input must be scalar.");
            }
            dft_size = get_scalar_value_from_tensor<int64_t>(frame_length);
          }

          bool is_onesided = static_cast<bool>(getAttribute(ctx, "onesided", 0));
          int64_t dft_unique_bins = is_onesided ? ((dft_size >> 1) + 1) : dft_size;

          auto n_dfts = static_cast<int64_t>((signal_size - dft_size) / static_cast<float>(frame_step_value)) + 1;

          // The output has the following shape: [batch_size][frames][dft_unique_bins][2]
          ONNX_NAMESPACE::TensorShapeProto result_shape_proto;
          auto batch_dim = result_shape_proto.add_dim();

          if (input_shape.dim(0).has_dim_value()) {
            batch_dim->set_dim_value(input_shape.dim(0).dim_value()); // batch size
          }

          result_shape_proto.add_dim()->set_dim_value(n_dfts);
          result_shape_proto.add_dim()->set_dim_value(dft_unique_bins);
          result_shape_proto.add_dim()->set_dim_value(2);
          updateOutputShape(ctx, 0, result_shape_proto);
        }));
} // namespace ONNX_NAMESPACE
