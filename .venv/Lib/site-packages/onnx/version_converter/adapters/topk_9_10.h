// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Adapter for TopK in default domain from version 9 to 10

#pragma once

namespace ONNX_NAMESPACE {
namespace version_conversion {

class TopK_9_10 final : public Adapter {
 public:
  explicit TopK_9_10() : Adapter("TopK", OpSetID(9), OpSetID(10)) {}

  void adapt_topk_9_10(std::shared_ptr<Graph> graph, Node* node) const {
    Tensor t;
    t.elem_type() = TensorProto_DataType_INT64;
    t.sizes() = std::vector<int64_t>{1};
    auto& data = t.int64s();
    data.emplace_back(node->i(kk));

    Node* constant = graph->create(kConstant);
    constant->insertBefore(node);
    constant->t_(kvalue, t);
    node->addInput(constant->output());

    node->removeAttribute(kk);
  }

  Node* adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    adapt_topk_9_10(graph, node);
    return node;
  }
};

} // namespace version_conversion
} // namespace ONNX_NAMESPACE
