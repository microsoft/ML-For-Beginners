// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Adapter for broadcasting ops in default domain from version 6 to 7

#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE {
namespace version_conversion {

class BroadcastForwardCompatibility final : public Adapter {
 public:
  explicit BroadcastForwardCompatibility(const std::string& op_name, const OpSetID& initial, const OpSetID& target)
      : Adapter(op_name, initial, target) {}

  void adapt_broadcast_forward_compatibility(std::shared_ptr<Graph> graph, Node* node) const {
    // Remove axis and broadcast attributes
    // Assess whether axis requires reshaping
    if (node->hasAttribute(kbroadcast)) {
      const ArrayRef<Value*>& inputs = node->inputs();
      assertInputsAvailable(inputs, name().c_str(), 2);
      const std::vector<Dimension>& A_sizes = inputs[0]->sizes();
      const std::vector<Dimension>& B_sizes = inputs[1]->sizes();
      // Also assert that broadcasting syntax are correct if axis is not present
      if (node->hasAttribute(kaxis)) {
        if (node->i(kaxis) != (int)(A_sizes.size() - B_sizes.size())) {
          // Add a Reshape node before input B
          Node* n = graph->create(kUnsqueeze);
          n->addInput(inputs[1]);
          std::vector<int64_t> axes;
          std::vector<Dimension> new_sizes = B_sizes;
          auto size = A_sizes.size() > B_sizes.size() ? A_sizes.size() - B_sizes.size() : 0;
          axes.reserve(size);
          new_sizes.reserve(new_sizes.size() + size);
          for (size_t i = 0; i < size; i++) {
            axes.emplace_back(B_sizes.size() + i);
            new_sizes.emplace_back(Dimension(1));
          }
          if (target_version().version() >= 13) { // Unsqueeze takes 'axes' input
            Tensor t;
            t.elem_type() = TensorProto_DataType_INT64;
            t.sizes() = std::vector<int64_t>{static_cast<int64_t>(axes.size())};
            auto& data = t.int64s();
            for (auto a : axes) {
              data.emplace_back(a);
            }
            Node* constant = graph->create(kConstant);
            constant->insertBefore(node);
            constant->t_(kvalue, t);
            node->addInput(constant->output());
          } else { // Unsqueeze takes 'axes' attribute
            n->is_(kaxes, std::forward<const std::vector<int64_t>>(axes));
          }
          // Move n before node
          n->insertBefore(node);
          // Set 2nd input to node to 1st of n and output of n to 2nd input to node
          n->output()->setSizes(new_sizes);
          node->replaceInput(1, n->output());
        }
      }
      node->removeAttribute(kbroadcast);
    }
    if (node->hasAttribute(kaxis))
      node->removeAttribute(kaxis);
    // Assert multi_broadcastable on inputs
    const ArrayRef<Value*>& inputs = node->inputs();
    assert_numpy_multibroadcastable(inputs[0]->sizes(), inputs[1]->sizes());
  }

  Node* adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    adapt_broadcast_forward_compatibility(graph, node);
    return node;
  }
};

} // namespace version_conversion
} // namespace ONNX_NAMESPACE
