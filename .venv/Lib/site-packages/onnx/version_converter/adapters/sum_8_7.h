// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Adapter for Sum in default domain from version 8 to 7

#pragma once

#include <memory>
#include <vector>

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE {
namespace version_conversion {

class Sum_8_7 final : public Adapter {
 public:
  explicit Sum_8_7() : Adapter("Sum", OpSetID(8), OpSetID(7)) {}

  void adapt_sum_8_7(std::shared_ptr<Graph>, Node* node) const {
    // Throw an exception if any broadcasting occurs
    const ArrayRef<Value*>& inputs = node->inputs();
    // Determine if inputs are of different sizes
    for (int i = 1; i < (int)inputs.size(); i++) {
      std::vector<Dimension> A_sizes = inputs[i - 1]->sizes();
      std::vector<Dimension> B_sizes = inputs[i]->sizes();
      assert_numpy_multibroadcastable(A_sizes, B_sizes);
    }
  }

  Node* adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    adapt_sum_8_7(graph, node);
    return node;
  }
};

} // namespace version_conversion
} // namespace ONNX_NAMESPACE
