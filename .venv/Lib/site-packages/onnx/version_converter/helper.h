// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Helper Methods for Adapters

#pragma once

#include <vector>

#include "onnx/common/ir.h"

namespace ONNX_NAMESPACE {
namespace version_conversion {
int check_numpy_unibroadcastable_and_require_broadcast(
    const std::vector<Dimension>& input1_sizes,
    const std::vector<Dimension>& input2_sizes);

void assert_numpy_multibroadcastable(
    const std::vector<Dimension>& input1_sizes,
    const std::vector<Dimension>& input2_sizes);

void assertNotParams(const std::vector<Dimension>& sizes);

void assertInputsAvailable(const ArrayRef<Value*>& inputs, const char* name, uint64_t num_inputs);
} // namespace version_conversion
} // namespace ONNX_NAMESPACE
