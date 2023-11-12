// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Adapter for Resize in default domain from version 10 to 11

#pragma once

namespace ONNX_NAMESPACE {
namespace version_conversion {

class Resize_10_11 final : public Adapter {
 public:
  explicit Resize_10_11() : Adapter("Resize", OpSetID(10), OpSetID(11)) {}

  void adapt_resize_10_11(std::shared_ptr<Graph> graph, Node* node) const {
    int input_rank = node->inputs()[0]->sizes().size();

    Value* scales_input = node->inputs()[1];
    node->addInput(scales_input);

    Tensor t;
    t.sizes() = std::vector<int64_t>{2 * input_rank};
    t.elem_type() = TensorProto_DataType_FLOAT;
    auto& data = t.floats();

    for (int i = 0; i < input_rank; i++)
      data.emplace_back(0);
    for (int i = 0; i < input_rank; i++)
      data.emplace_back(1);

    Node* constant = graph->create(kConstant);
    constant->insertBefore(node);
    constant->t_(kvalue, t);
    node->replaceInput(1, constant->output());
  }

  Node* adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    adapt_resize_10_11(graph, node);
    return node;
  }
};

} // namespace version_conversion
} // namespace ONNX_NAMESPACE
