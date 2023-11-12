// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "onnx/defs/shape_inference.h"

namespace ONNX_NAMESPACE {

inline void appendDimToTensorShapeProto(TensorShapeProto& tsp, const TensorShapeProto* input_data, int index) {
  if (index >= input_data->dim_size() || index < -input_data->dim_size()) {
    fail_shape_inference("indices must be in [-rank, rank-1].");
  } else {
    *tsp.add_dim() = input_data->dim((index < 0) ? input_data->dim_size() + index : index);
  }
}

// Returns true if the given axis attribute is 0
inline bool axisIsZero(DataPropagationContext& ctx, bool defaultZero = false) {
  auto axisAttr = ctx.getAttribute("axis");
  // if axis is not defined
  if (!axisAttr) {
    if (defaultZero) {
      return true;
    } else {
      fail_shape_inference("Required attribute axis is missing");
      return false;
    }
  }
  int axis = static_cast<int>(axisAttr->i());
  auto input_data_0 = ctx.getInputData(0);
  if (input_data_0 == nullptr) {
    return false;
  }
  int rank = input_data_0->dim_size();
  if (axis < -rank || axis >= rank) {
    fail_shape_inference("axis must be in [-rank, rank-1].");
    return false;
  }
  if (axis < 0) {
    axis += rank;
  }
  // Only supports axis = 0 since the data comes from Shape
  return axis == 0;
}

inline void PropagateShapeDataFromInputToOutput(DataPropagationContext& ctx, int idx) {
  // propogate input data
  const auto input_data = ctx.getInputData(idx);
  if (input_data != nullptr) {
    TensorShapeProto tsp;
    tsp.CopyFrom(*input_data);
    ctx.addOutputData(0, std::move(tsp));
  }
}

inline void GatherOp13DataPropagator(DataPropagationContext& ctx) {
  if (!axisIsZero(ctx, true)) {
    return;
  }
  const auto input_data = ctx.getInputData(0);
  if (input_data == nullptr) {
    return;
  }
  const auto input_indices = ctx.getInputData(1);
  if (input_data == nullptr || input_indices == nullptr) {
    return;
  }
  TensorShapeProto tsp;
  for (int i = 0; i < input_indices->dim_size(); ++i) {
    if (input_indices->dim(i).has_dim_value()) {
      appendDimToTensorShapeProto(tsp, input_data, input_indices->dim(i).dim_value());
    } else {
      return;
    }
  }
  if (tsp.dim_size() > 0) {
    ctx.addOutputData(0, std::move(tsp));
  }
}

} // namespace ONNX_NAMESPACE
