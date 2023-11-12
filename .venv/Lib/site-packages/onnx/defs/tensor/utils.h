/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "onnx/defs/schema.h"
#include "onnx/defs/tensor_proto_util.h"

#include <cmath>

namespace ONNX_NAMESPACE {
// The below is called by ops after opset 11, inclusively.
void resizeShapeInference(InferenceContext& ctx);

void resizeShapeInferenceHelper(
    const TensorShapeProto& input_shape,
    const std::vector<float>& scales_data,
    TensorShapeProto* output_shape);

void resizeShapeInferenceHelper(
    const TensorShapeProto& input_shape,
    const std::vector<int64_t>& sizes_data,
    TensorShapeProto* output_shape);

// Belows are called by ops between opset versions in the name inclusively.
void resizeShapeInference_opset7_to_10(InferenceContext& ctx);
void resizeShapeInference_opset11_to_12(InferenceContext& ctx);
void resizeShapeInference_opset13_to_18(InferenceContext& ctx);
void resizeShapeInference_opset18_to_19(InferenceContext& ctx);

void resizeShapeInferenceHelper_opset7_to_10(
    const TensorShapeProto& input_shape,
    const std::vector<float>& scales_data,
    TensorShapeProto* output_shape);

enum class KeepAspectRatioPolicy {
  STRETCH,
  NOT_LARGER,
  NOT_SMALLER,
};

void KeepAspectRatioHelper(
    KeepAspectRatioPolicy policy,
    const TensorShapeProto& input_shape,
    const std::vector<int64_t>& axes,
    std::vector<int64_t>& sizes_data);

extern const char* NonZero_ver9_doc;

std::function<void(OpSchema&)> PadDocGenerator(const char* description, const char* mode_description);
} // namespace ONNX_NAMESPACE
