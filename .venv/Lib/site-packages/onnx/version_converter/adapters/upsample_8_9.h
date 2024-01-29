// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Adapter for Upsample in default domain from version 8 to 9

#pragma once

#include <memory>
#include <vector>

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE {
namespace version_conversion {

struct Upsample_8_9 final : public Adapter {
  explicit Upsample_8_9() : Adapter("Upsample", OpSetID(8), OpSetID(9)) {}

  void adapt_upsample_8_9(std::shared_ptr<Graph> graph, Node* node) const {
    Symbol input_dirs = Symbol("scales");
    int dim = (int)(node->fs(kscales).size());
    Tensor t;
    t.elem_type() = TensorProto_DataType_FLOAT;
    t.sizes() = std::vector<int64_t>{dim};
    auto& data = t.floats();

    if (node->hasAttribute(input_dirs)) {
      for (double scale : node->fs(kscales)) {
        data.emplace_back((float)scale);
      }

      Node* constant = graph->create(kConstant);
      constant->insertBefore(node);
      constant->t_(kvalue, t);
      node->addInput(constant->output());
      node->removeAttribute(kscales);
    }
  }

  Node* adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    adapt_upsample_8_9(graph, node);
    return node;
  }
};

} // namespace version_conversion
} // namespace ONNX_NAMESPACE
