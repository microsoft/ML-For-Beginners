// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Adapter for MaxPool in default domain from version 8 to 7

#pragma once

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE {
namespace version_conversion {

class MaxPool_8_7 final : public Adapter {
 public:
  explicit MaxPool_8_7() : Adapter("MaxPool", OpSetID(8), OpSetID(7)) {}

  void adapt_maxpool_8_7(std::shared_ptr<Graph>, Node* node) const {
    const ArrayRef<Value*>& outputs = node->outputs();
    ONNX_ASSERTM(outputs.size() != 2, "Opset version 7 of MaxPool cannot include Indices output");
    if (node->hasAttribute(kstorage_order))
      node->removeAttribute(kstorage_order);
  }

  Node* adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    adapt_maxpool_8_7(graph, node);
    return node;
  }
};

} // namespace version_conversion
} // namespace ONNX_NAMESPACE
