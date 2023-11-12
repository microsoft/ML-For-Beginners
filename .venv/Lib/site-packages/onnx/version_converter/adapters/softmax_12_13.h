// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Adapter for Softmax amd LogSoftmax in default domain from version 12 to 13

#pragma once

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE {
namespace version_conversion {

class Softmax_12_13 final : public Adapter {
 public:
  explicit Softmax_12_13(const std::string& op_name) : Adapter(op_name, OpSetID(12), OpSetID(13)) {}

  void adapt_softmax_12_13(std::shared_ptr<Graph> graph, Node* node) const {
    int old_axis = node->hasAttribute(kaxis) ? node->i(kaxis) : 1;
    int input_rank = node->inputs()[0]->sizes().size();

    if (old_axis < 0)
      old_axis = input_rank + old_axis;

    if (old_axis == input_rank - 1)
      node->i_(kaxis, -1);
    else {
      //    -- shape ------------------
      //   /                           |
      // ----- flatten -- softmax -- reshape

      // get original softmax's input shape
      Symbol kShape("Shape");
      Node* shape = graph->create(kShape);
      shape->addInput(node->inputs()[0]);
      shape->insertBefore(node);

      // Insert Flatten node before softmax
      Node* flatten = graph->create(kFlatten);
      flatten->addInput(node->inputs()[0]);
      flatten->insertBefore(node);
      flatten->i_(kaxis, old_axis);
      node->replaceInput(0, flatten->output());

      // Softmax along the last axis of the flattened 2D tensor
      node->i_(kaxis, -1);

      // Insert Reshape node after softmax
      const std::string original_output_name = node->output()->uniqueName();
      const use_list original_uses(node->output()->uses());
      node->output()->setUniqueName(original_output_name + "_intermediate");
      Node* reshape = graph->create(kReshape);
      reshape->addInput(node->outputs()[0]);
      reshape->addInput(shape->output());
      reshape->output()->setUniqueName(original_output_name);
      reshape->insertAfter(node);

      // Fix outputs & wiring
      if (node->output()->sizes().size() != 0) {
        reshape->output()->setSizes(node->output()->sizes());
      }
      reshape->output()->setElemType(node->output()->elemType());
      node->output()->wipeSizes();
      for (Use u : original_uses) {
        u.user->replaceInputWith(node->output(), reshape->output());
      }
      for (size_t i = 0; i < graph->outputs().size(); i++) {
        if (graph->outputs()[i]->uniqueName() == original_output_name) {
          graph->return_node()->replaceInput(i, reshape->output());
        }
      }
    }
  }

  Node* adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    adapt_softmax_12_13(graph, node);
    return node;
  }
};

} // namespace version_conversion
} // namespace ONNX_NAMESPACE
