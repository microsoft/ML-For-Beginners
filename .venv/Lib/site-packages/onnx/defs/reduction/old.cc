/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <functional>

#include "onnx/defs/reduction/utils.h"
#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {

std::vector<std::string> GetSupportedDataTypesForReductionOps_opset12(bool supports8bit) {
  if (supports8bit) {
    auto data_types = OpSchema::numeric_types_for_math_reduction();
    data_types.push_back("tensor(uint8)");
    data_types.push_back("tensor(int8)");

    return data_types;
  }

  return OpSchema::numeric_types_for_math_reduction();
}

std::function<void(OpSchema&)> ReduceDocGenerator_opset12(const char* name, bool supports_8bit_datatypes = false) {
  return [=](OpSchema& schema) {
    std::string doc;
    POPULATE_OP_DOC_STR(doc = R"DOC(
Computes the {name} of the input tensor's element along the provided axes. The resulting
tensor has the same rank as the input if keepdims equals 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned.

The above behavior is similar to numpy, with the exception that numpy defaults keepdims to
False instead of True.)DOC";
                        ReplaceAll(doc, "{name}", name););
    schema.SetDoc(doc.c_str());
    schema.Attr(
        "axes",
        "A list of integers, along which to reduce. The default is to reduce over "
        "all the dimensions of the input tensor. Accepted range is [-r, r-1] where r = rank(data).",
        AttributeProto::INTS,
        OPTIONAL_VALUE);
    schema.Attr(
        "keepdims",
        "Keep the reduced dimension or not, default 1 means keep reduced dimension.",
        AttributeProto::INT,
        static_cast<int64_t>(1));
    schema.Input(0, "data", "An input tensor.", "T");
    schema.Output(0, "reduced", "Reduced output tensor.", "T");
    schema.TypeConstraint(
        "T",
        GetSupportedDataTypesForReductionOps_opset12(supports_8bit_datatypes),
        supports_8bit_datatypes ? "Constrain input and output types to high-precision and 8 bit numeric tensors."
                                : "Constrain input and output types to high-precision numeric tensors.");
    schema.TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      if (!hasNInputShapes(ctx, 1)) {
        return;
      }

      int64_t keep_dims = 1;
      auto attr_proto = ctx.getAttribute("keepdims");
      if (attr_proto) {
        keep_dims = attr_proto->i();
      }
      auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
      int64_t input_ndim = input_shape.dim_size();
      auto output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
      std::vector<int64_t> axes;
      auto axes_proto = ctx.getAttribute("axes");
      if (axes_proto)
        axes.assign(axes_proto->ints().begin(), axes_proto->ints().end());

      for (size_t i = 0; i < axes.size(); ++i) {
        if (axes[i] < -input_ndim || axes[i] >= input_ndim) {
          fail_shape_inference("axis must be in [-rank, rank-1]. input rank was ", input_ndim);
        }
        if (axes[i] < 0)
          axes[i] += input_ndim;
      }
      // do we need handle negative axis?
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

ONNX_OPERATOR_SET_SCHEMA(ReduceMax, 12, OpSchema().FillUsing(ReduceDocGenerator_opset12("max", true)));

ONNX_OPERATOR_SET_SCHEMA(ReduceMin, 12, OpSchema().FillUsing(ReduceDocGenerator_opset12("min", true)));

ONNX_OPERATOR_SET_SCHEMA(ReduceSum, 11, OpSchema().FillUsing(ReduceDocGenerator_opset12("sum")));

ONNX_OPERATOR_SET_SCHEMA(ReduceSumSquare, 11, OpSchema().FillUsing(ReduceDocGenerator_opset12("sum square")));

ONNX_OPERATOR_SET_SCHEMA(ReduceMean, 11, OpSchema().FillUsing(ReduceDocGenerator_opset12("mean")));

ONNX_OPERATOR_SET_SCHEMA(ReduceProd, 11, OpSchema().FillUsing(ReduceDocGenerator_opset12("product")));

ONNX_OPERATOR_SET_SCHEMA(ReduceLogSum, 11, OpSchema().FillUsing(ReduceDocGenerator_opset12("log sum")));

ONNX_OPERATOR_SET_SCHEMA(ReduceLogSumExp, 11, OpSchema().FillUsing(ReduceDocGenerator_opset12("log sum exponent")));

ONNX_OPERATOR_SET_SCHEMA(ReduceL1, 11, OpSchema().FillUsing(ReduceDocGenerator_opset12("L1 norm")));

ONNX_OPERATOR_SET_SCHEMA(ReduceL2, 11, OpSchema().FillUsing(ReduceDocGenerator_opset12("L2 norm")));

std::function<void(OpSchema&)> ArgReduceDocGenerator_opset12(const char* name) {
  return [=](OpSchema& schema) {
    std::string doc;
    POPULATE_OP_DOC_STR(doc = R"DOC(
Computes the indices of the {name} elements of the input tensor's element along the
provided axis. The resulting tensor has the same rank as the input if keepdims equals 1.
If keepdims equal 0, then the resulting tensor has the reduced dimension pruned.
If select_last_index is True (default False), the index of the last occurrence of the {name}
is selected if the {name} appears more than once in the input. Otherwise the index of the
first occurrence is selected.
The type of the output tensor is integer.)DOC";
                        ReplaceAll(doc, "{name}", name););
    schema.SetDoc(doc.c_str());
    schema.Attr(
        "axis",
        "The axis in which to compute the arg indices. Accepted range is [-r, r-1] where r = rank(data).",
        AttributeProto::INT,
        static_cast<int64_t>(0));
    schema.Attr(
        "keepdims",
        "Keep the reduced dimension or not, default 1 means keep reduced dimension.",
        AttributeProto::INT,
        static_cast<int64_t>(1));
    schema.Attr(
        "select_last_index",
        "Whether to select the last index or the first index if the {name} appears in multiple indices, default is False (first index).",
        AttributeProto::INT,
        static_cast<int64_t>(0));
    schema.Input(0, "data", "An input tensor.", "T");
    schema.Output(0, "reduced", "Reduced output tensor with integer data type.", "tensor(int64)");
    schema.TypeConstraint(
        "T", OpSchema::all_numeric_types(), "Constrain input and output types to all numeric tensors.");
    schema.TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      // set output element type to int64
      updateOutputElemType(ctx, 0, TensorProto_DataType_INT64);

      if (!hasNInputShapes(ctx, 1)) {
        return;
      }

      auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
      auto output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
      int64_t input_ndim = input_shape.dim_size();
      int64_t axis = 0; // default to 0
      auto axis_proto = ctx.getAttribute("axis");
      if (axis_proto) {
        axis = axis_proto->i();
        if (axis < -input_ndim || axis >= input_ndim) {
          fail_shape_inference("'axis' must be in [-rank(indices), rank(indices)-1]");
        }
        if (axis < 0)
          axis += input_ndim;
      }

      int64_t keep_dims = 1;
      auto attr_proto = ctx.getAttribute("keepdims");
      if (attr_proto) {
        keep_dims = attr_proto->i();
      }
      // do we need handle negative axis?
      for (int i = 0; i < input_ndim; ++i) {
        if (i != axis) {
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
} // namespace ONNX_NAMESPACE

ONNX_OPERATOR_SET_SCHEMA(ArgMax, 12, OpSchema().FillUsing(ArgReduceDocGenerator_opset12("max")));

ONNX_OPERATOR_SET_SCHEMA(ArgMin, 12, OpSchema().FillUsing(ArgReduceDocGenerator_opset12("min")));

std::function<void(OpSchema&)> ReduceDocGenerator_opset1(const char* name, const char* empty_value, int opset = 1) {
  return [=](OpSchema& schema) {
    std::string doc;
    POPULATE_OP_DOC_STR(doc = R"DOC(
Computes the {name} of the input tensor's element along the provided axes. The resulting
tensor has the same rank as the input if keepdims equals 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned. Input tensors of rank zero are
valid. Reduction over an empty set of values yields {empty_value}.

The above behavior is similar to numpy, with the exception that numpy defaults keepdims to
False instead of True.)DOC";
                        ReplaceAll(doc, "{name}", name););
    ReplaceAll(doc, "{empty_value}", empty_value);
    schema.SetDoc(doc.c_str());
    schema.Attr(
        "axes",
        opset >= 11 ? "A list of integers, along which to reduce. The default is to reduce over "
                      "all the dimensions of the input tensor. Accepted range is [-r, r-1] where r = rank(data)."
                    : "A list of integers, along which to reduce. The default is to reduce over "
                      "all the dimensions of the input tensor.",
        AttributeProto::INTS,
        OPTIONAL_VALUE);
    schema.Attr(
        "keepdims",
        "Keep the reduced dimension or not, default 1 means keep reduced dimension.",
        AttributeProto::INT,
        static_cast<int64_t>(1));
    schema.Input(0, "data", "An input tensor.", "T");
    schema.Output(0, "reduced", "Reduced output tensor.", "T");
    schema.TypeConstraint(
        "T",
        OpSchema::numeric_types_for_math_reduction(),
        "Constrain input and output types to high-precision numeric tensors.");
    schema.TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      if (!hasNInputShapes(ctx, 1)) {
        return;
      }

      int64_t keep_dims = 1;
      auto attr_proto = ctx.getAttribute("keepdims");
      if (attr_proto) {
        keep_dims = attr_proto->i();
      }
      auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
      int64_t input_ndim = input_shape.dim_size();
      auto output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
      std::vector<int64_t> axes;
      auto axes_proto = ctx.getAttribute("axes");
      if (axes_proto)
        axes.assign(axes_proto->ints().begin(), axes_proto->ints().end());

      for (size_t i = 0; i < axes.size(); ++i) {
        if (axes[i] < 0)
          axes[i] += input_ndim;
      }
      // do we need handle negative axis?
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

ONNX_OPERATOR_SET_SCHEMA(ReduceMax, 1, OpSchema().FillUsing(ReduceDocGenerator_opset1("max", EMPTY_MIN)));

ONNX_OPERATOR_SET_SCHEMA(ReduceMin, 1, OpSchema().FillUsing(ReduceDocGenerator_opset1("min", EMPTY_MAX)));

ONNX_OPERATOR_SET_SCHEMA(ReduceSum, 1, OpSchema().FillUsing(ReduceDocGenerator_opset1("sum", EMPTY_ZERO)));

ONNX_OPERATOR_SET_SCHEMA(ReduceSumSquare, 1, OpSchema().FillUsing(ReduceDocGenerator_opset1("sum square", EMPTY_ZERO)));

ONNX_OPERATOR_SET_SCHEMA(ReduceMean, 1, OpSchema().FillUsing(ReduceDocGenerator_opset1("mean", EMPTY_UNDEFINED)));

ONNX_OPERATOR_SET_SCHEMA(ReduceProd, 1, OpSchema().FillUsing(ReduceDocGenerator_opset1("product", EMPTY_ONE)));

ONNX_OPERATOR_SET_SCHEMA(ReduceLogSum, 1, OpSchema().FillUsing(ReduceDocGenerator_opset1("log sum", EMPTY_MINUS_INF)));

ONNX_OPERATOR_SET_SCHEMA(
    ReduceLogSumExp,
    1,
    OpSchema().FillUsing(ReduceDocGenerator_opset1("log sum exponent", EMPTY_MINUS_INF)));

ONNX_OPERATOR_SET_SCHEMA(ReduceL1, 1, OpSchema().FillUsing(ReduceDocGenerator_opset1("L1 norm", EMPTY_ZERO)));

ONNX_OPERATOR_SET_SCHEMA(ReduceL2, 1, OpSchema().FillUsing(ReduceDocGenerator_opset1("L2 norm", EMPTY_ZERO)));

ONNX_OPERATOR_SET_SCHEMA(ReduceMax, 11, OpSchema().FillUsing(ReduceDocGenerator_opset1("max", EMPTY_MIN, 11)));

ONNX_OPERATOR_SET_SCHEMA(ReduceMin, 11, OpSchema().FillUsing(ReduceDocGenerator_opset1("min", EMPTY_MAX, 11)));

std::function<void(OpSchema&)> ArgReduceDocGenerator_opset1(const char* name) {
  return [=](OpSchema& schema) {
    std::string doc;
    POPULATE_OP_DOC_STR(doc = R"DOC(
Computes the indices of the {name} elements of the input tensor's element along the
provided axis. The resulting tensor has the same rank as the input if keepdims equals 1.
If keepdims equal 0, then the resulted tensor have the reduced dimension pruned.
The type of the output tensor is integer.)DOC";
                        ReplaceAll(doc, "{name}", name););
    schema.SetDoc(doc.c_str());
    schema.Attr("axis", "The axis in which to compute the arg indices.", AttributeProto::INT, static_cast<int64_t>(0));
    schema.Attr(
        "keepdims",
        "Keep the reduced dimension or not, default 1 means keep reduced dimension.",
        AttributeProto::INT,
        static_cast<int64_t>(1));
    schema.Input(0, "data", "An input tensor.", "T");
    schema.Output(0, "reduced", "Reduced output tensor with integer data type.", "tensor(int64)");
    schema.TypeConstraint(
        "T", OpSchema::all_numeric_types(), "Constrain input and output types to all numeric tensors.");
    schema.TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      // set output element type to int64
      updateOutputElemType(ctx, 0, TensorProto_DataType_INT64);

      if (!hasNInputShapes(ctx, 1)) {
        return;
      }

      auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
      auto output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
      int64_t input_ndim = input_shape.dim_size();
      int64_t axis = 0; // default to 0
      auto axis_proto = ctx.getAttribute("axis");
      if (axis_proto) {
        axis = axis_proto->i();
        if (axis < 0)
          axis += input_ndim;
      }

      int64_t keep_dims = 1;
      auto attr_proto = ctx.getAttribute("keepdims");
      if (attr_proto) {
        keep_dims = attr_proto->i();
      }
      // do we need handle negative axis?
      for (int i = 0; i < input_ndim; ++i) {
        if (i != axis) {
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
} // namespace ONNX_NAMESPACE

ONNX_OPERATOR_SET_SCHEMA(ArgMax, 1, OpSchema().FillUsing(ArgReduceDocGenerator_opset1("max")));

ONNX_OPERATOR_SET_SCHEMA(ArgMin, 1, OpSchema().FillUsing(ArgReduceDocGenerator_opset1("min")));

std::function<void(OpSchema&)> ArgReduceDocGenerator_opset11(const char* name) {
  return [=](OpSchema& schema) {
    std::string doc = R"DOC(
Computes the indices of the {name} elements of the input tensor's element along the
provided axis. The resulting tensor has the same rank as the input if keepdims equals 1.
If keepdims equal 0, then the resulting tensor has the reduced dimension pruned.
The input tensor must not be empty.
The type of the output tensor is integer.)DOC";
    ReplaceAll(doc, "{name}", name);
    schema.SetDoc(doc.c_str());
    schema.Attr(
        "axis",
        "The axis in which to compute the arg indices. Accepted range is [-r, r-1] where r = rank(data).",
        AttributeProto::INT,
        static_cast<int64_t>(0));
    schema.Attr(
        "keepdims",
        "Keep the reduced dimension or not, default 1 means keep reduced dimension.",
        AttributeProto::INT,
        static_cast<int64_t>(1));
    schema.Input(0, "data", "An input tensor.", "T");
    schema.Output(0, "reduced", "Reduced output tensor with integer data type.", "tensor(int64)");
    schema.TypeConstraint(
        "T", OpSchema::all_numeric_types(), "Constrain input and output types to all numeric tensors.");
    schema.TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      // set output element type to int64
      updateOutputElemType(ctx, 0, TensorProto_DataType_INT64);

      if (!hasNInputShapes(ctx, 1)) {
        return;
      }

      auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
      auto output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
      int64_t input_ndim = input_shape.dim_size();
      int64_t axis = 0; // default to 0
      auto axis_proto = ctx.getAttribute("axis");
      if (axis_proto) {
        axis = axis_proto->i();
        if (axis < -input_ndim || axis >= input_ndim) {
          fail_shape_inference("'axis' must be in [-rank(indices), rank(indices)-1]");
        }
        if (axis < 0)
          axis += input_ndim;
      }

      int64_t keep_dims = 1;
      auto attr_proto = ctx.getAttribute("keepdims");
      if (attr_proto) {
        keep_dims = attr_proto->i();
      }
      // do we need handle negative axis?
      for (int i = 0; i < input_ndim; ++i) {
        if (i != axis) {
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
} // namespace ONNX_NAMESPACE

ONNX_OPERATOR_SET_SCHEMA(ArgMax, 11, OpSchema().FillUsing(ArgReduceDocGenerator_opset11("max")));
ONNX_OPERATOR_SET_SCHEMA(ArgMin, 11, OpSchema().FillUsing(ArgReduceDocGenerator_opset11("min")));

ONNX_OPERATOR_SET_SCHEMA(ReduceMax, 13, OpSchema().FillUsing(ReduceOpGenerator("max", EMPTY_MIN, true)));
ONNX_OPERATOR_SET_SCHEMA(ReduceMin, 13, OpSchema().FillUsing(ReduceOpGenerator("min", EMPTY_MAX, true)));
ONNX_OPERATOR_SET_SCHEMA(ReduceSumSquare, 13, OpSchema().FillUsing(ReduceOpGenerator("sum square", EMPTY_ZERO)));
ONNX_OPERATOR_SET_SCHEMA(ReduceMean, 13, OpSchema().FillUsing(ReduceOpGenerator("mean", EMPTY_UNDEFINED)));
ONNX_OPERATOR_SET_SCHEMA(ReduceProd, 13, OpSchema().FillUsing(ReduceOpGenerator("product", EMPTY_ONE)));
ONNX_OPERATOR_SET_SCHEMA(ReduceLogSum, 13, OpSchema().FillUsing(ReduceOpGenerator("log sum", EMPTY_MINUS_INF)));
ONNX_OPERATOR_SET_SCHEMA(
    ReduceLogSumExp,
    13,
    OpSchema().FillUsing(ReduceOpGenerator("log sum exponent", EMPTY_MINUS_INF)));
ONNX_OPERATOR_SET_SCHEMA(ReduceL1, 13, OpSchema().FillUsing(ReduceOpGenerator("L1 norm", EMPTY_ZERO)));
ONNX_OPERATOR_SET_SCHEMA(ReduceL2, 13, OpSchema().FillUsing(ReduceOpGenerator("L2 norm", EMPTY_ZERO)));

ONNX_OPERATOR_SET_SCHEMA(ReduceMax, 18, OpSchema().FillUsing(ReduceOpGenerator("max", EMPTY_MIN, true, true)));
ONNX_OPERATOR_SET_SCHEMA(ReduceMin, 18, OpSchema().FillUsing(ReduceOpGenerator("min", EMPTY_MAX, true, true)));
} // namespace ONNX_NAMESPACE
