// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Adapter for Pad in default domain from version 10 to 11

#pragma once

#include <memory>
#include <vector>

namespace ONNX_NAMESPACE {
namespace version_conversion {

class Pad_10_11 final : public Adapter {
 public:
  explicit Pad_10_11() : Adapter("Pad", OpSetID(10), OpSetID(11)) {}

  void adapt_pad_10_11(std::shared_ptr<Graph> graph, Node* node) const {
    // Turn pads attribute into input
    Tensor t_pads;
    t_pads.elem_type() = TensorProto_DataType_INT64;
    auto& data_pads = t_pads.int64s();
    for (int64_t shape : node->is(kpads)) {
      data_pads.emplace_back(shape);
    }
    t_pads.sizes() = std::vector<int64_t>{(int64_t)data_pads.size()};
    Value* v_pads = graph->addInitializerAndCreateValue(t_pads);
    node->addInput(v_pads);
    node->removeAttribute(kpads);
    // Turn value attribute into input
    if (!node->hasAttribute(kmode) || node->s(kmode) == "constant") {
      if (!node->hasAttribute(kvalue))
        node->f_(kvalue, 0.);
      Tensor t_value;
      t_value.elem_type() = TensorProto_DataType_FLOAT;
      auto& data_value = t_value.floats();
      data_value.emplace_back(node->f(kvalue));
      Node* constant = graph->create(kConstant);
      constant->insertBefore(node);
      constant->t_(kvalue, t_value);
      node->addInput(constant->output());
      node->removeAttribute(kvalue);
    }
  }

  Node* adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    adapt_pad_10_11(graph, node);
    return node;
  }
};

} // namespace version_conversion
} // namespace ONNX_NAMESPACE
