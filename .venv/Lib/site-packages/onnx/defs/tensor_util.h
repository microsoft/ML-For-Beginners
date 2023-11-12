/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "onnx/common/ir.h"

namespace ONNX_NAMESPACE {

template <typename T>
const std::vector<T> ParseData(const Tensor* tensor);

} // namespace ONNX_NAMESPACE
