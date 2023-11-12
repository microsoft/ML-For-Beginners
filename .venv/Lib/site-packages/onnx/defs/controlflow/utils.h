/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {

void ClearShape(TypeProto& input_type);

int handle_negative_axis_validate(const std::string& attrib, int axis, int rank);

void IfInferenceFunction(InferenceContext& ctx);

void LoopInferenceFunction(InferenceContext& ctx);

void ScanInferenceFunction(InferenceContext& ctx);

} // namespace ONNX_NAMESPACE
