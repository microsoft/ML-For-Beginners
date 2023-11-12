// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Adapter for Dropout in default domain from version 11 to 12

#pragma once

namespace ONNX_NAMESPACE {
namespace version_conversion {

class Dropout_11_12 final : public Adapter {
 public:
  explicit Dropout_11_12() : Adapter("Dropout", OpSetID(11), OpSetID(12)) {}

  void adapt_dropout_11_12(std::shared_ptr<Graph> graph, Node* node) const {
    float ratio;
    if (node->hasAttribute(kratio)) {
      ratio = node->f(kratio);
      node->removeAttribute(kratio);
    } else {
      ratio = 0.5;
    }

    Tensor t_ratio;
    t_ratio.elem_type() = TensorProto_DataType_FLOAT;
    auto& data_ratio = t_ratio.floats();
    data_ratio.emplace_back(ratio);
    Node* constant = graph->create(kConstant);
    constant->insertBefore(node);
    constant->t_(kvalue, t_ratio);
    node->addInput(constant->output());
  }

  Node* adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    adapt_dropout_11_12(graph, node);
    return node;
  }
};

} // namespace version_conversion
} // namespace ONNX_NAMESPACE
