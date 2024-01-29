// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Adapter for Add in default domain from version 6 to 5

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE {
namespace version_conversion {

class TypeRestriction : public Adapter {
 public:
  explicit TypeRestriction(
      const std::string& op_name,
      const OpSetID& initial,
      const OpSetID& target,
      const std::vector<TensorProto_DataType>& unallowed_types)
      : Adapter(op_name, initial, target), unallowed_types_(unallowed_types) {}

  void adapt_type_restriction(std::shared_ptr<Graph>, Node* node) const {
    // Since consumed_inputs is optional, no need to add it (as in batchnorm)
    // Iterate over all inputs and outputs
    for (Value* input : node->inputs()) {
      isUnallowed(input);
    }
    for (Value* output : node->outputs()) {
      isUnallowed(output);
    }
  }

  Node* adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    adapt_type_restriction(graph, node);
    return node;
  }

 private:
  std::vector<TensorProto_DataType> unallowed_types_;

  void isUnallowed(Value* val) const {
    ONNX_ASSERTM(
        std::find(std::begin(unallowed_types_), std::end(unallowed_types_), val->elemType()) ==
            std::end(unallowed_types_),
        "DataType (%d) of Input or Output"
        " of operator '%s' is unallowed for Opset Version %d.",
        val->elemType(),
        name().c_str(),
        target_version().version());
  }
};

} // namespace version_conversion
} // namespace ONNX_NAMESPACE
