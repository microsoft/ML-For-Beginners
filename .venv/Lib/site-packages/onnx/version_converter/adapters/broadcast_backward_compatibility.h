// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Adapter for broadcasting ops in default domain from version 7 to 6

#pragma once

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE {
namespace version_conversion {

class BroadcastBackwardCompatibility final : public Adapter {
 public:
  explicit BroadcastBackwardCompatibility(const std::string& op_name, const OpSetID& initial, const OpSetID& target)
      : Adapter(op_name, initial, target) {}

  void adapt_broadcast_backward_compatibility(std::shared_ptr<Graph>, Node* node) const {
    // Verify that broadcasts are allowed in limited spec of opset version 6
    // Multidirectional broadcasting, as defined in Broadcasting.md
    // MathDocGenerator provides differences
    // Main change: encode broadcasting commands as explicit attribute
    const ArrayRef<Value*>& inputs = node->inputs();
    assertInputsAvailable(inputs, name().c_str(), 2);
    const std::vector<Dimension>& A_sizes = inputs[0]->sizes();
    const std::vector<Dimension>& B_sizes = inputs[1]->sizes();
    // Ensure that first input is larger than or equal to the second
    // numpy_unibroadcastable here is considered to be equivalent to opset1_broadcastable
    // This is because backwards conversion does not allow for an axis that is not
    // suffix matching
    int req_broadcast = check_numpy_unibroadcastable_and_require_broadcast(A_sizes, B_sizes);
    ONNX_ASSERTM(
        req_broadcast != -1,
        "%s being converted from %d to %d does "
        "not have broadcastable inputs.",
        name().c_str(),
        initial_version().version(),
        target_version().version());
    if (req_broadcast == 1) {
      // If conditional is not fulfilled, we have a default broadcast
      // Add broadcast attribute
      node->i_(kbroadcast, 1);
    }
  }

  Node* adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    adapt_broadcast_backward_compatibility(graph, node);
    return node;
  }
};

} // namespace version_conversion
} // namespace ONNX_NAMESPACE
