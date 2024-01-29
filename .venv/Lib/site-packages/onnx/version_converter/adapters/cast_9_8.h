// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Adapter for Cast in default domain from version 9 to 8

#pragma once

#include <memory>

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE {
namespace version_conversion {

class Cast_9_8 final : public Adapter {
 public:
  explicit Cast_9_8() : Adapter("Cast", OpSetID(9), OpSetID(8)) {}

  void adapt_cast_9_8(std::shared_ptr<Graph>, Node* node) const {
    if (node->inputs()[0]->elemType() == TensorProto_DataType_STRING || node->i(kto) == TensorProto_DataType_STRING)
      ONNX_ASSERTM(false, "Casting From/To STRING data type is not supported")
  }

  Node* adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    adapt_cast_9_8(graph, node);
    return node;
  }
};

} // namespace version_conversion
} // namespace ONNX_NAMESPACE
