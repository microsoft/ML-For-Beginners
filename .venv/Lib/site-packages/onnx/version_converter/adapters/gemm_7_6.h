// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Adapter for Gemm in default domain from version 7 to 6

#pragma once

#include <memory>
#include <vector>

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE {
namespace version_conversion {

class Gemm_7_6 final : public Adapter {
 public:
  explicit Gemm_7_6() : Adapter("Gemm", OpSetID(7), OpSetID(6)) {}

  void adapt_gemm_7_6(std::shared_ptr<Graph>, Node* node) const {
    const ArrayRef<Value*>& inputs = node->inputs();
    assertInputsAvailable(inputs, name().c_str(), 3);
    const auto& A_shape = inputs[0]->sizes();
    const auto& B_shape = inputs[1]->sizes();
    // Determine if C is broadcastable
    const auto& C_shape = inputs[2]->sizes();
    // Create (M, N) to input to numpy_unibroadcastable
    // TODO: Reconcile fact that shapes aren't determined for 1st 2 inputs
    std::vector<Dimension> MN;
    if (node->hasAttribute(ktransA) && node->i(ktransA) == 1) {
      MN.emplace_back(A_shape[1]);
    } else {
      MN.emplace_back(A_shape[0]);
    }
    if (node->hasAttribute(ktransB) && node->i(ktransB) == 1) {
      MN.emplace_back(B_shape[0]);
    } else {
      MN.emplace_back(B_shape[1]);
    }
    int req_broadcast = check_numpy_unibroadcastable_and_require_broadcast(MN, C_shape);
    ONNX_ASSERTM(
        req_broadcast != -1,
        "%s being converted from %d to %d does "
        "not have broadcastable inputs.",
        name().c_str(),
        initial_version().version(),
        target_version().version());
    if (req_broadcast == 1) {
      node->i_(kbroadcast, 1);
    }
  }

  Node* adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    adapt_gemm_7_6(graph, node);
    return node;
  }
};

} // namespace version_conversion
} // namespace ONNX_NAMESPACE
