/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cmath>

#include "onnx/defs/schema.h"
#include "onnx/defs/tensor_proto_util.h"

namespace ONNX_NAMESPACE {

// Constants used to indicate value returned by reduction of an empty set of values.
constexpr const char* EMPTY_ZERO = "0";
constexpr const char* EMPTY_ONE = "1";
constexpr const char* EMPTY_UNDEFINED = "undefined";
constexpr const char* EMPTY_MIN =
    "minus infinity (if supported by the datatype) or the minimum value of the data type otherwise";
constexpr const char* EMPTY_MAX =
    "plus infinity (if supported by the datatype) or the maximum value of the data type otherwise";
constexpr const char* EMPTY_MINUS_INF = "minus infinity (if supported by the datatype) or undefined otherwise";

std::function<void(OpSchema&)> ReduceOpGenerator(
    const char* name,
    const char* empty_value,
    bool supports_8bit_datatypes = false,
    bool axes_input = false,
    const char* func_body = nullptr,
    ContextDependentFunctionBodyBuilder function_builder = nullptr,
    bool supports_boolean_datatype = false);

inline std::function<void(OpSchema&)> ReduceOpDynamicAxes(const char* name, const char* empty_value) {
  return ReduceOpGenerator(name, empty_value, false, true, nullptr, nullptr, false);
}

inline std::function<void(OpSchema&)>
ReduceFunctionOp(const char* name, const char* empty_value, const char* func_body) {
  return ReduceOpGenerator(name, empty_value, false, true, func_body);
}

} // namespace ONNX_NAMESPACE
