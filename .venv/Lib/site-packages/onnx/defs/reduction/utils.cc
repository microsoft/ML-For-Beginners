/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "onnx/defs/reduction/utils.h"
#include <algorithm>

namespace ONNX_NAMESPACE {
std::vector<std::string> GetSupportedDataTypesForReductionOps(bool supports8bit) {
  if (supports8bit) {
    auto data_types = OpSchema::numeric_types_for_math_reduction_ir4();
    data_types.push_back("tensor(uint8)");
    data_types.push_back("tensor(int8)");

    return data_types;
  }

  return OpSchema::numeric_types_for_math_reduction_ir4();
}

std::function<void(OpSchema&)> ReduceDocGenerator_opset13_18(
    const char* name,
    bool supports_8bit_datatypes,
    bool axes_input,
    const char* func_body,
    ContextDependentFunctionBodyBuilder function_builder) {
  return [=](OpSchema& schema) {
    std::string doc;
    POPULATE_OP_DOC_STR(doc = R"DOC(
Computes the {name} of the input tensor's elements along the provided axes. The resulting
tensor has the same rank as the input if keepdims equals 1. If keepdims equals 0, then
the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
valid.

The above behavior is similar to numpy, with the exception that numpy defaults keepdims to
False instead of True.)DOC";
                        ReplaceAll(doc, "{name}", name););
    schema.SetDoc(doc.c_str());
    schema.Attr(
        "keepdims",
        "Keep the reduced dimension or not, default 1 means keep reduced dimension.",
        AttributeProto::INT,
        static_cast<int64_t>(1));
    schema.Input(0, "data", "An input tensor.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable);
    if (axes_input) {
      schema.Attr(
          "noop_with_empty_axes",
          "Defines behavior if 'axes' is empty. Default behavior with 'false' is to reduce all axes. "
          "When axes is empty and this attribute is set to true, input tensor will not be reduced,"
          "and the output tensor would be equivalent to input tensor.",
          AttributeProto::INT,
          static_cast<int64_t>(0));
      schema.Input(
          1,
          "axes",
          "Optional input list of integers, along which to reduce. "
          "The default is to reduce over all the dimensions of the input tensor if 'noop_with_empty_axes' is false, "
          "else act as an Identity op when 'noop_with_empty_axes' is true. "
          "Accepted range is [-r, r-1] where r = rank(data).",
          "tensor(int64)",
          OpSchema::Optional,
          true,
          1,
          OpSchema::NonDifferentiable);
    } else {
      schema.Attr(
          "axes",
          "A list of integers, along which to reduce. The default is to reduce over "
          "all the dimensions of the input tensor. Accepted range is [-r, r-1] where r = rank(data).",
          AttributeProto::INTS,
          OPTIONAL_VALUE);
    }
    schema.Output(0, "reduced", "Reduced output tensor.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable);
    schema.TypeConstraint(
        "T",
        GetSupportedDataTypesForReductionOps(supports_8bit_datatypes),
        supports_8bit_datatypes ? "Constrain input and output types to high-precision and 8 bit numeric tensors."
                                : "Constrain input and output types to high-precision numeric tensors.");
    if (func_body) {
      schema.FunctionBody(func_body);
    } else if (function_builder) {
      schema.SetContextDependentFunctionBodyBuilder(function_builder);
    }
    schema.TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      if (!hasNInputShapes(ctx, 1)) {
        return;
      }

      int64_t keep_dims = 1, noop_with_empty_axes = 0;
      auto attr_proto = ctx.getAttribute("keepdims");
      if (attr_proto) {
        keep_dims = attr_proto->i();
      }
      auto noop_attr_proto = ctx.getAttribute("noop_with_empty_axes");
      if (noop_attr_proto) {
        noop_with_empty_axes = noop_attr_proto->i();
      }
      std::vector<int64_t> axes;
      if (ctx.hasInput(1)) { // axes is input
        if (ctx.getAttribute("axes")) {
          fail_shape_inference("axes as an input and attribute cannot be specified at the same time.");
        }

        const TensorProto* axesInitializer = ctx.getInputData(1);
        if (axesInitializer == nullptr) {
          // skip if axes is not an initializer
          return;
        }
        std::vector<int64_t> axes_values = ParseData<int64_t>(axesInitializer);
        axes.assign(axes_values.begin(), axes_values.end());
      } else { // axes is attribute
        auto axes_proto = ctx.getAttribute("axes");
        if (axes_proto)
          axes.assign(axes_proto->ints().begin(), axes_proto->ints().end());
      }
      auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
      if (noop_with_empty_axes && axes.empty()) {
        propagateShapeFromInputToOutput(ctx, 0, 0);
        return;
      }
      int64_t input_ndim = input_shape.dim_size();
      auto output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

      for (size_t i = 0; i < axes.size(); ++i) {
        if (axes[i] < -input_ndim || axes[i] >= input_ndim) {
          fail_shape_inference("axis must be in [-rank, rank-1]. input rank was ", input_ndim);
        }
        if (axes[i] < 0)
          axes[i] += input_ndim;
      }
      for (int i = 0; i < input_ndim; ++i) {
        // axes empty means reduce all dim
        if (!axes.empty() && std::find(axes.begin(), axes.end(), i) == axes.end()) {
          auto dim = output_shape->add_dim();
          dim->CopyFrom(input_shape.dim(i));
        } else {
          if (keep_dims == 1) {
            auto dim = output_shape->add_dim();
            dim->set_dim_value(1);
          }
        }
      }
    });
  };
}
} // namespace ONNX_NAMESPACE
