// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Adapter for Clip in default domain from version 10 to 11

#pragma once
#include <limits>
#include <memory>

namespace ONNX_NAMESPACE {
namespace version_conversion {

class Clip_10_11 final : public Adapter {
 public:
  explicit Clip_10_11() : Adapter("Clip", OpSetID(10), OpSetID(11)) {}

  void adapt_clip_10_11(std::shared_ptr<Graph> graph, Node* node) const {
    bool has_min = node->hasAttribute(kmin);
    bool has_max = node->hasAttribute(kmax);

    // Turn min/max attributes into tensor (if present) and add value as input
    if (has_min) {
      attrToInput(graph, node, node->f(kmin));
      node->removeAttribute(kmin);
    }
    if (has_max) {
      if (!has_min) {
        attrToInput(graph, node, std::numeric_limits<float>::lowest());
      }
      attrToInput(graph, node, node->f(kmax));
      node->removeAttribute(kmax);
    }
  }

  void attrToInput(std::shared_ptr<Graph> graph, Node* node, float val) const {
    Tensor t;
    t.elem_type() = TensorProto_DataType_FLOAT;
    auto& data = t.floats();
    data.emplace_back(val);
    Node* constant = graph->create(kConstant);
    constant->insertBefore(node);
    constant->t_(kvalue, t);
    node->addInput(constant->output());
  }

  Node* adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    adapt_clip_10_11(graph, node);
    return node;
  }
};

} // namespace version_conversion
} // namespace ONNX_NAMESPACE
