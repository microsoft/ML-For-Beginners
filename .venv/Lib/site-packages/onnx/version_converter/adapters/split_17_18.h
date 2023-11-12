// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Adapter for Split in default domain from version 17 to 18

#pragma once

#include "onnx/version_converter/adapters/adapter.h"
#include "onnx/version_converter/adapters/transformers.h"

namespace ONNX_NAMESPACE {
namespace version_conversion {

class Split_17_18 : public Adapter {
 public:
  explicit Split_17_18() : Adapter("Split", OpSetID(17), OpSetID(18)) {}

  void adapt_split_17_18(std::shared_ptr<Graph>, Node* node) const {
    const auto num_outputs = node->outputs().size();
    SetAttribute(knum_outputs, num_outputs);
  }

  Node* adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    // if node does not have neither 'num_outputs' attribute nor 'split' input
    if (!node->hasAttribute(knum_outputs) && node->inputs().size() != 2) {
      adapt_split_17_18(graph, node);
    }
    return node;
  }
};

} // namespace version_conversion
} // namespace ONNX_NAMESPACE
