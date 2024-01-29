// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Adapter for all ops that remove consumed_inputs

#pragma once

#include <memory>
#include <string>

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE {
namespace version_conversion {

class RemoveConsumedInputs : public Adapter {
 public:
  explicit RemoveConsumedInputs(const std::string& op_name, const OpSetID& initial, const OpSetID& target)
      : Adapter(op_name, initial, target) {}

  Node* adapt(std::shared_ptr<Graph>, Node* node) const override {
    if (node->hasAttribute(kconsumed_inputs))
      node->removeAttribute(kconsumed_inputs);
    return node;
  }
};

} // namespace version_conversion
} // namespace ONNX_NAMESPACE
