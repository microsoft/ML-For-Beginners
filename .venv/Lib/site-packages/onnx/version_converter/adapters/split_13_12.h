// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Adapter for all ops that remove consumed_inputs

#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE {
namespace version_conversion {

class Split_13_12 : public Adapter {
 public:
  explicit Split_13_12() : Adapter("Split", OpSetID(13), OpSetID(12)) {}

  Node* adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    // Identify if 'split' is statically determined; if so, feed as attribute
    const ArrayRef<Value*>& inputs = node->inputs();
    // Get 'split' from initializer or constant operator
    // Identify whether we have a Constant Op or an Initializer
    Value* const_val = inputs[1];
    Node* node_ptr = const_val->node();
    if (node_ptr->kind() == kConstant) {
      // Get value attribute of kConstant
      const std::vector<int64_t>& int64s = node_ptr->t(kvalue).int64s();
      if (int64s.empty()) {
        // Also handle raw data
        std::string raw_data = node_ptr->t(kvalue).raw();
        ONNX_ASSERTM(
            raw_data.size() != 0 && raw_data.size() % 8 == 0,
            "Raw Data must be non-empty and size must be a multiple of 8");
        int64_t* raw = (int64_t*)const_cast<char*>(raw_data.c_str());
        node->is_(ksplit, std::vector<int64_t>(raw, raw + node_ptr->t(kvalue).size_from_dim(0)));
      } else {
        node->is_(ksplit, std::forward<const std::vector<int64_t>>(int64s));
      }
      // If Constant node isn't used anywhere else, remove it
      node->removeInput(1);
      if (const_val->uses().size() < 1) {
        node_ptr->destroy();
      }
    } else {
      // Get Value name, find Initializer with same name
      for (const auto& initializer : graph->initializers()) {
        if (initializer.name() == inputs[1]->uniqueName()) {
          node->is_(ksplit, std::forward<const std::vector<int64_t>>(initializer.int64s()));
          node->removeInput(1);
          // Remove initializer
          if (const_val->uses().size() < 1)
            graph->eraseInitializerAndInput(const_val);
          break;
        }
      }
    }
    ONNX_ASSERTM(node->hasAttribute(ksplit), "No initializer or constant input to node found");
    return node;
  }
};

} // namespace version_conversion
} // namespace ONNX_NAMESPACE
