// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Adapter for Reshape in default domain from version 4 to 5

#pragma once

#include "onnx/version_converter/adapters/remove_consumed_inputs.h"

namespace ONNX_NAMESPACE {
namespace version_conversion {

class Reshape_4_5 final : public RemoveConsumedInputs {
 public:
  explicit Reshape_4_5() : RemoveConsumedInputs("Reshape", OpSetID(4), OpSetID(5)) {}

  void adapt_reshape_4_5(std::shared_ptr<Graph> graph, Node* node) const {
    // Create Input from Attribute - add as Initializer
    // Create tensor for value attribute
    Tensor t;
    t.elem_type() = TensorProto_DataType_INT64;
    auto& data = t.int64s();
    // Turn shapes attribute into tensor
    for (int64_t shape : node->is(kshape)) {
      data.emplace_back(shape);
    }
    // Add value as input to node
    Node* constant = graph->create(kConstant);
    constant->insertBefore(node);
    constant->t_(kvalue, t);
    node->addInput(constant->output());
    // Remove kshape attribute
    node->removeAttribute(kshape);
  }

  Node* adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    RemoveConsumedInputs::adapt(graph, node);
    adapt_reshape_4_5(graph, node);
    return node;
  }
};

} // namespace version_conversion
} // namespace ONNX_NAMESPACE
