// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Adapter for Scatter in default domain from version 10 to 11

#pragma once

#include <memory>

namespace ONNX_NAMESPACE {
namespace version_conversion {

class Scatter_10_11 final : public Adapter {
 public:
  explicit Scatter_10_11() : Adapter("Scatter", OpSetID(10), OpSetID(11)) {}

  Node* adapt_scatter_10_11(std::shared_ptr<Graph> graph, Node* node) const {
    int axis = node->hasAttribute(kaxis) ? node->i(kaxis) : 0;

    // Replace the node with an equivalent ScatterElements node
    Node* scatter_elements = graph->create(kScatterElements);
    scatter_elements->i_(kaxis, axis);
    scatter_elements->addInput(node->inputs()[0]);
    scatter_elements->addInput(node->inputs()[1]);
    scatter_elements->addInput(node->inputs()[2]);
    node->replaceAllUsesWith(scatter_elements);

    scatter_elements->insertBefore(node);
    node->destroy();

    return scatter_elements;
  }

  Node* adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    return adapt_scatter_10_11(graph, node);
  }
};

} // namespace version_conversion
} // namespace ONNX_NAMESPACE
