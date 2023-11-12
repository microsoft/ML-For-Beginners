// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Adapter for Scan in default domain from version 9 to 8

#pragma once

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE {
namespace version_conversion {

struct Scan_9_8 final : public Adapter {
  explicit Scan_9_8() : Adapter("Scan", OpSetID(9), OpSetID(8)) {}

  void adapt_scan_9_8(std::shared_ptr<Graph>, Node* node) const {
    const std::vector<Value*> inputs(node->inputs().vec());
    const std::vector<Value*> outputs(node->outputs().vec());

    // Handling Attribute Changes

    Symbol input_dirs = Symbol("scan_input_directions");
    if (node->hasAttribute(input_dirs)) {
      const std::vector<int64_t> scan_input_directions(node->is(input_dirs));
      node->removeAttribute(input_dirs);
      node->is_(Symbol("directions"), std::move(scan_input_directions));
    }

    Symbol output_dirs = Symbol("scan_output_directions");
    if (node->hasAttribute(output_dirs)) {
      const std::vector<int64_t> scan_output_directions(node->is(output_dirs));
      for (int64_t x : scan_output_directions) {
        ONNX_ASSERTM(x == 0, "Unsupported output direction for Version 8");
      }
      node->removeAttribute(output_dirs);
    }

    Symbol input_axes = Symbol("scan_input_axes");
    if (node->hasAttribute(input_axes)) {
      const std::vector<int64_t> scan_input_axes(node->is(input_axes));
      for (int64_t x : scan_input_axes) {
        ONNX_ASSERTM(x == 0, "Unsupported input axes for Version 8");
      }
      node->removeAttribute(input_axes);
    }

    Symbol output_axes = Symbol("scan_output_axes");
    if (node->hasAttribute(output_axes)) {
      const std::vector<int64_t> scan_output_axes(node->is(output_axes));
      for (int64_t x : scan_output_axes) {
        ONNX_ASSERTM(x == 0, "Unsupported output axes for Version 8");
      }
      node->removeAttribute(output_axes);
    }

    // Handling Input and Output Changes

    node->removeAllInputs();

    Value* v = new Value(node, 0);
    v->setUniqueName("");
    v->setElemType(TensorProto_DataType::TensorProto_DataType_INT32);
    node->addInput(v);

    for (Value* input : inputs) {
      std::vector<Dimension> new_sizes{Dimension(1)};
      new_sizes.insert(new_sizes.end(), input->sizes().begin(), input->sizes().end());
      input->setSizes(new_sizes);
      node->addInput(input);
    }

    for (Value* output : outputs) {
      std::vector<Dimension> new_sizes{Dimension(1)};
      new_sizes.insert(new_sizes.end(), output->sizes().begin(), output->sizes().end());
      output->setSizes(new_sizes);
    }
  }

  Node* adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    adapt_scan_9_8(graph, node);
    return node;
  }
};

} // namespace version_conversion
} // namespace ONNX_NAMESPACE
