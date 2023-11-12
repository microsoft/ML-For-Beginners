/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "onnx/defs/schema.h"
#include "onnx/defs/tensor_proto_util.h"

#include <cmath>

namespace ONNX_NAMESPACE {
std::function<void(OpSchema&)> ReduceDocGenerator_opset13_18(
    const char* name,
    bool supports_8bit_datatypes = false,
    bool axes_input = false,
    const char* func_body = nullptr,
    ContextDependentFunctionBodyBuilder function_builder = nullptr);
} // namespace ONNX_NAMESPACE
