// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Adapter for Scan in default domain from version 8 to 9

#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE {
namespace version_conversion {
struct Scan_8_9 final : public Adapter {
  explicit Scan_8_9() : Adapter("Scan", OpSetID(8), OpSetID(9)) {}

  void adapt_scan_8_9(std::shared_ptr<Graph>, Node* node) const {
    const std::vector<Value*> inputs(node->inputs().vec());
    const std::vector<Value*> outputs(node->outputs().vec());

    // Handling Attribute Changes

    Symbol dirs = Symbol("directions");
    if (node->hasAttribute(dirs)) {
      const std::vector<int64_t> directions(node->is(dirs));
      node->removeAttribute(dirs);
      node->is_(Symbol("scan_input_directions"), std::move(directions));
    }

    // Handling Input and Output Changes

    node->removeAllInputs();

    ONNX_ASSERTM(inputs[0]->uniqueName() == "", "Unsupported conversion to opset 9");

    for (Value* input : inputs) {
      if (!input->sizes().empty()) {
        std::vector<Dimension> new_sizes(input->sizes().begin() + 1, input->sizes().end());
        input->setSizes(new_sizes);
        node->addInput(input);
      }
    }

    for (Value* output : outputs) {
      if (!output->sizes().empty()) {
        std::vector<Dimension> new_sizes(output->sizes().begin() + 1, output->sizes().end());
        output->setSizes(new_sizes);
      }
    }
  }

  Node* adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    adapt_scan_8_9(graph, node);
    return node;
  }
};

} // namespace version_conversion
} // namespace ONNX_NAMESPACE
