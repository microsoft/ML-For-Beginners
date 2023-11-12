/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "onnx/defs/tensor/utils.h"
#include <numeric>

namespace ONNX_NAMESPACE {
void resizeShapeInferenceHelper(
    const TensorShapeProto& input_shape,
    const std::vector<int64_t>& sizes_data,
    TensorShapeProto* output_shape) {
  if (!sizes_data.empty()) {
    for (int i = 0; i < input_shape.dim_size(); ++i) {
      auto* dim = output_shape->mutable_dim(i);
      if (sizes_data[i] > 0) {
        dim->set_dim_value(sizes_data[i]);
      }
    }
    return;
  }
}

void KeepAspectRatioHelper(
    KeepAspectRatioPolicy policy,
    const TensorShapeProto& input_shape,
    const std::vector<int64_t>& axes,
    std::vector<int64_t>& sizes_data) {
  if (policy != KeepAspectRatioPolicy::NOT_LARGER && policy != KeepAspectRatioPolicy::NOT_SMALLER) {
    return;
  }
  float scale = policy == KeepAspectRatioPolicy::NOT_LARGER ? std::numeric_limits<float>::max()
                                                            : std::numeric_limits<float>::min();
  std::function<float(float, float)> reduce_f;
  if (policy == KeepAspectRatioPolicy::NOT_LARGER) {
    reduce_f = [](float a, float b) { return std::min(a, b); };
  } else {
    reduce_f = [](float a, float b) { return std::max(a, b); };
  }

  bool has_unknown_dim = false;
  for (size_t i = 0; i < sizes_data.size(); i++) {
    int d = axes.empty() ? i : axes[i];
    if (!input_shape.dim(d).has_dim_value()) {
      has_unknown_dim = true;
      break;
    }
    float s = sizes_data[i] / static_cast<float>(input_shape.dim(d).dim_value());
    scale = reduce_f(scale, s);
  }
  // If there's at least one unknown dim we can't infer the output shape, since it
  // will depend on the original aspect ratio of the input.
  for (size_t i = 0; i < sizes_data.size(); i++) {
    int d = axes.empty() ? i : axes[i];
    sizes_data[i] = has_unknown_dim ? -1 : std::roundf(scale * input_shape.dim(d).dim_value());
  }
}

void resizeShapeInferenceHelper(
    const TensorShapeProto& input_shape,
    const std::vector<float>& scales_data,
    TensorShapeProto* output_shape) {
  for (int i = 0; i < input_shape.dim_size(); ++i) {
    auto* dim = output_shape->mutable_dim(i);
    // If input_shape has dim_value, we calculate the scaled result
    // If input_shape doesn's have one, we leave it here
    if (input_shape.dim(i).has_dim_value()) {
      int64_t dim_value =
          static_cast<int64_t>(std::floor(static_cast<float>(input_shape.dim(i).dim_value()) * scales_data[i]));
      // If output_shape has dim_value, we validate the caculated result
      // If output_shape doesn's have one, we set it to the scaled result
      if (dim->has_dim_value()) {
        if (static_cast<int64_t>(dim->dim_value()) != dim_value) {
          fail_shape_inference(
              "Dimension value inferred (",
              dim_value,
              ") is not equal to the existing dim value (",
              dim->dim_value(),
              ").");
        }
      } else {
        dim->set_dim_value(static_cast<int64_t>(dim_value));
      } // dim->has_dim_value()
    } // input_shape.dim(i).has_dim_value()
  }
}

void resizeShapeInferenceVersioned(InferenceContext& ctx, int opset_version) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }
  const auto& input_shape = getInputShape(ctx, 0);
  auto* output_shape = getOutputShape(ctx, 0);

  bool hasScalesInput = ctx.hasInput(2);
  bool hasSizesInput = ctx.hasInput(3);

  const TensorProto* scales = 2 < ctx.getNumInputs() ? ctx.getInputData(2) : nullptr;
  std::vector<int64_t> sizes_data;
  if (3 < ctx.getNumInputs()) {
    bool found_sizes = false;
    const auto sizes_shape = getShapeInput(ctx, 3, found_sizes);
    // If sizes is an empty shape, assume it's not provided
    if (found_sizes) {
      if (sizes_shape.dim_size() == 0) {
        hasSizesInput = false;
      } else {
        for (int i = 0; i < sizes_shape.dim_size(); ++i) {
          sizes_data.push_back(sizes_shape.dim(i).dim_value());
        }
      }
    }
  }

  // If scales is an empty constant, assume it's not provided
  if (scales && ParseData<float>(scales).empty()) {
    hasScalesInput = false;
    scales = nullptr;
  }

  if (opset_version >= 13) {
    if (hasScalesInput + hasSizesInput != 1) {
      fail_shape_inference("Either `sizes` or `scales` must be provided, but not both of them");
    }
  }

  auto keep_aspect_ratio_policy_attr = ctx.getAttribute("keep_aspect_ratio_policy");
  KeepAspectRatioPolicy keep_aspect_ratio_policy = KeepAspectRatioPolicy::STRETCH;
  if (keep_aspect_ratio_policy_attr && keep_aspect_ratio_policy_attr->has_s()) {
    auto str = keep_aspect_ratio_policy_attr->s();
    if (str == "stretch") {
      keep_aspect_ratio_policy = KeepAspectRatioPolicy::STRETCH;
    } else if (str == "not_larger") {
      keep_aspect_ratio_policy = KeepAspectRatioPolicy::NOT_LARGER;
    } else if (str == "not_smaller") {
      keep_aspect_ratio_policy = KeepAspectRatioPolicy::NOT_SMALLER;
    } else {
      fail_shape_inference("Unknown value for `keep_aspect_ratio_policy`: ", str, ".");
    }
  }

  if (hasScalesInput && keep_aspect_ratio_policy != KeepAspectRatioPolicy::STRETCH) {
    fail_shape_inference(
        "Providing `scales` is incompatible with a `keep_aspect_ratio_policy` other than \"stretch\".");
  }

  if (output_shape->dim_size() > 0) {
    if (output_shape->dim_size() != input_shape.dim_size()) {
      fail_shape_inference(
          "Ranks inferred (",
          input_shape.dim_size(),
          ") is not equal to the existing rank value (",
          output_shape->dim_size(),
          ").");
    }
  } else { // Infer the rank of output anyway
    for (int i = 0; i < input_shape.dim_size(); ++i) {
      output_shape->add_dim();
    }
  }

  auto axes_attr = ctx.getAttribute("axes");
  size_t rank_x = input_shape.dim_size();
  std::vector<int64_t> axes;
  if (axes_attr) {
    axes = RetrieveValues<int64_t>(*axes_attr);
    checkAxesRange(axes, rank_x);
    adjustNegativeAxes(axes, rank_x);
    checkDuplicateAxes(axes, rank_x);
  }
  if (hasSizesInput) {
    if (!axes.empty()) {
      if (sizes_data.size() != axes.size()) {
        fail_shape_inference(
            "Number of elements of input 'sizes' (",
            sizes_data.size(),
            ") does not match the number of axes (",
            axes.size(),
            ").");
      }
    } else {
      // sizes_data contains scales for all axes
      if (sizes_data.size() != rank_x) {
        fail_shape_inference(
            "Number of elements of input 'sizes' (",
            sizes_data.size(),
            ") must be same as rank of input 'X' (",
            rank_x,
            ").");
      }
    }

    // Process sizes_data according to the selected policy
    KeepAspectRatioHelper(keep_aspect_ratio_policy, input_shape, axes, sizes_data);

    // If axes subset is provided, populate new sizes_data with all dims
    if (!axes.empty()) {
      std::vector<int64_t> tmp(rank_x);
      for (size_t i = 0; i < rank_x; i++) {
        tmp[i] = input_shape.dim(i).has_dim_value() ? input_shape.dim(i).dim_value() : -1;
      }
      for (size_t i = 0; i < axes.size(); i++) {
        int d = axes[i];
        tmp[d] = sizes_data[i];
      }
      std::swap(tmp, sizes_data);
    }

    resizeShapeInferenceHelper(input_shape, sizes_data, output_shape);
  } else if (nullptr != scales) {
    // Infer output shape's dimension value if 'scales' is known.
    if (scales->data_type() == TensorProto::FLOAT) {
      auto scales_data = ParseData<float>(scales);

      if (!axes.empty()) {
        // scales_data contains scales for a subset of axes. The rest should not be resized
        if (scales_data.size() != axes.size()) {
          fail_shape_inference(
              "Number of elements of input 'scales' (",
              scales_data.size(),
              ") does not match the number of axes (",
              axes.size(),
              ").");
        }

        std::vector<float> tmp(rank_x, 1.0f);
        for (size_t i = 0; i < axes.size(); i++) {
          int d = axes[i];
          tmp[d] = scales_data[i];
        }
        std::swap(tmp, scales_data);
      } else {
        // scales_data contains scales for all axes
        if (scales_data.size() != static_cast<size_t>(input_shape.dim_size())) {
          fail_shape_inference("Number of elements of input 'scales' must be same as rank of input 'X'");
        }
      }
      resizeShapeInferenceHelper(input_shape, scales_data, output_shape);
    } else {
      fail_shape_inference("Input 'scales' must have float element type.");
    }
  } // nullptr != scales
}

void resizeShapeInference_opset18_to_19(InferenceContext& ctx) {
  resizeShapeInferenceVersioned(ctx, 19);
}

void resizeShapeInference_opset13_to_18(InferenceContext& ctx) {
  resizeShapeInferenceVersioned(ctx, 13);
}

void resizeShapeInference_opset11_to_12(InferenceContext& ctx) {
  resizeShapeInferenceVersioned(ctx, 11);
}

void resizeShapeInferenceHelper_opset7_to_10(
    const TensorShapeProto& input_shape,
    const std::vector<float>& scales_data,
    TensorShapeProto* output_shape) {
  for (int i = 0; i < input_shape.dim_size(); ++i) {
    auto* dim = output_shape->mutable_dim(i);
    // If input_shape has dim_value, we calculate the scaled result
    // If input_shape doesn's have one, we leave it here
    if (input_shape.dim(i).has_dim_value()) {
      int64_t dim_value =
          static_cast<int64_t>(std::floor(static_cast<float>(input_shape.dim(i).dim_value()) * scales_data[i]));
      // If output_shape has dim_value, we validate the caculated result
      // If output_shape doesn's have one, we set it to the scaled result
      if (dim->has_dim_value()) {
        if (static_cast<int64_t>(dim->dim_value()) != dim_value) {
          fail_shape_inference(
              "Dimension value inferred (",
              dim_value,
              ") is not equal to the existing dim value (",
              dim->dim_value(),
              ").");
        }
      } else {
        dim->set_dim_value(static_cast<int64_t>(dim_value));
      } // dim->has_dim_value()
    } // input_shape.dim(i).has_dim_value()
  }
}

void resizeShapeInference_opset7_to_10(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }
  const auto& input_shape = getInputShape(ctx, 0);
  auto* output_shape = getOutputShape(ctx, 0);
  const auto scales = ctx.getInputData(1);

  if (output_shape->dim_size() > 0) {
    if (output_shape->dim_size() != input_shape.dim_size()) {
      fail_shape_inference(
          "Ranks inferred (",
          input_shape.dim_size(),
          ") is not equal to the existing rank value (",
          output_shape->dim_size(),
          ").");
    }
  } else { // Infer the rank of output anyway
    for (int i = 0; i < input_shape.dim_size(); ++i) {
      output_shape->add_dim();
    }
  }

  if (nullptr != scales) {
    // Infer output shape's dimension value if 'scales' is known.
    if (scales->data_type() == TensorProto::FLOAT) {
      const auto& scales_data = ParseData<float>(scales);
      if (scales_data.size() != static_cast<size_t>(input_shape.dim_size())) {
        fail_shape_inference("Number of elements of input 'scales' must be same as rank of input 'X'");
      }
      resizeShapeInferenceHelper_opset7_to_10(input_shape, scales_data, output_shape);
    } else {
      fail_shape_inference("Input 'scales' must have float element type.");
    } // nullptr != scales
  }
}

std::function<void(OpSchema&)> PadDocGenerator(const char* description, const char* mode_description) {
  return [=](OpSchema& schema) {
    schema.SetDoc(description);
    schema.Attr("mode", mode_description, AttributeProto::STRING, std::string("constant"));
    schema.Input(0, "data", "Input tensor.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable);
    schema.Input(
        1,
        "pads",
        "Tensor of integers indicating the number of padding elements to add or remove (if negative) "
        "at the beginning and end of each axis. For 2D input tensor, it is the number of pixels. "
        "`pads` should be a 1D tensor of shape [2 * num_axes] where `num_axes` refers to the number "
        "of elements in the `axes` input or the input rank if `axes` are not provided explicitly. "
        "`pads` format should be: [x1_begin, x2_begin, ..., x1_end, x2_end,...], "
        "where xi_begin is the number of pad values added at the beginning of axis `axes[i]` and "
        "xi_end, the number of pad values added at the end of axis `axes[i]`.",
        "tensor(int64)",
        OpSchema::Single,
        true,
        1,
        OpSchema::NonDifferentiable);
    schema.Input(
        2,
        "constant_value",
        "(Optional) A scalar value to be used if the mode chosen is `constant` (by default it is 0, "
        "empty string or False).",
        "T",
        OpSchema::Optional,
        true,
        1,
        OpSchema::NonDifferentiable);
    schema.Input(
        3,
        "axes",
        "1-D tensor of axes that `pads` apply to. Negative value means counting dimensions "
        "from the back. Accepted range is [-r, r-1] where r = rank(data). Behavior is undefined if an "
        "axis is repeated. If not provided, all axes are assumed (`[0, 1, ..., input_rank-1]`).",
        "Tind",
        OpSchema::Optional,
        true,
        1,
        OpSchema::NonDifferentiable);

    schema.Output(0, "output", "Tensor after padding.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable);
    schema.TypeConstraint(
        "T", OpSchema::all_tensor_types_ir4(), "Constrain input and output types to all tensor types.");
    schema.TypeConstraint("Tind", {"tensor(int32)", "tensor(int64)"}, "Constrain indices to integer types");
    schema.TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      // Type inference
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      // Shape inference needs the input data shape
      if (!hasNInputShapes(ctx, 1)) {
        return;
      }
      const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
      const auto input_rank = input_shape.dim_size();

      std::vector<int64_t> axes;
      if (hasInputShape(ctx, 3)) { //'axes' input
        auto axes_initializer = ctx.getInputData(3);
        if (axes_initializer == nullptr)
          return; // can't do shape inference then

        axes = ParseData<int64_t>(axes_initializer);
        checkAxesRange(axes, input_rank);
        adjustNegativeAxes(axes, input_rank);
        checkDuplicateAxes(axes, input_rank);
      } else {
        axes.resize(input_rank);
        std::iota(axes.begin(), axes.end(), 0);
      }

      int num_axes = axes.size();
      auto* output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

      // Populating default dims
      std::vector<TensorShapeProto_Dimension*> out_dims(input_rank);
      for (int i = 0; i < input_rank; ++i) {
        out_dims[i] = output_shape->add_dim();
      }

      // Shape Inference if
      //     1. 'pads' are available.
      // and 2. 'axes' are available, or default.
      const TensorProto* pads_initializer = ctx.getInputData(1);
      if (nullptr != pads_initializer && !axes.empty()) {
        if (pads_initializer->dims_size() != 1 || pads_initializer->data_type() != TensorProto::INT64) {
          fail_shape_inference("'pads' input must be a 1D (shape: [2 * num_axes]) tensor of type int64");
        }

        const auto& pads_data = ParseData<int64_t>(pads_initializer);
        if (pads_data.size() != static_cast<size_t>(2 * num_axes)) {
          fail_shape_inference(
              "Pads has incorrect number of values. Expected 2 * ",
              num_axes,
              " values. Got ",
              pads_data.size(),
              " values.");
        }

        // Set default dim values
        for (int i = 0; i < input_rank; ++i) {
          const auto& input_dim = input_shape.dim(i);
          if (input_dim.has_dim_value()) {
            out_dims[i]->set_dim_value(input_dim.dim_value());
          }
        }

        for (int i = 0; i < num_axes; ++i) {
          auto axis = axes[i];
          const auto& input_dim = input_shape.dim(axis);
          auto& out_dim = *out_dims[axis];
          auto total_pad = pads_data[i] + pads_data[num_axes + i];
          if (input_dim.has_dim_value()) {
            out_dim.set_dim_value(input_dim.dim_value() + total_pad);
          } else if (total_pad == 0) {
            out_dim = input_dim;
          }
        }
      }
    });
  };
};
} // namespace ONNX_NAMESPACE
