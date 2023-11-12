// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Adapter for BatchNormalization in default domain from version 13 to 14

#pragma once

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE {
namespace version_conversion {

class BatchNormalization_13_14 final : public Adapter {
 public:
  explicit BatchNormalization_13_14() : Adapter("BatchNormalization", OpSetID(13), OpSetID(14)) {}

  void adapt_batch_normalization_13_14(Node* node) const {
    ONNX_ASSERTM(
        node->outputs().size() < 4,
        "BatchNormalization outputs 4 and 5 are not "
        "supported in Opset 14.");
  }

  Node* adapt(std::shared_ptr<Graph>, Node* node) const override {
    adapt_batch_normalization_13_14(node);
    return node;
  }
};

} // namespace version_conversion
} // namespace ONNX_NAMESPACE
