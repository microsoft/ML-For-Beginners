// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Adapter indicating compatibility of op between opsets with separate
// definitions

#pragma once

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE {
namespace version_conversion {

struct ExtendSupportedTypes final : public Adapter {
  explicit ExtendSupportedTypes(const std::string& op_name, const OpSetID& initial, const OpSetID& target)
      : Adapter(op_name, initial, target) {}

  Node* create_cast_op(
      std::shared_ptr<Graph> graph,
      ArrayRef<Value*> inputs,
      const int to_type,
      const std::vector<Dimension>& output_shape,
      const std::string& name) const {
    Node* node = graph->create(kCast, inputs);
    node->i_(kto, to_type);
    node->output()->setUniqueName(name);
    node->output()->setSizes(output_shape);
    node->output()->setElemType(to_type);
    return node;
  }

  void adapt_type_extension(std::shared_ptr<Graph> graph, Node* node) const {
    const ArrayRef<Value*>& inputs = node->inputs();
    const ArrayRef<Value*>& outputs = node->outputs();
    const std::string original_output_name = node->output()->uniqueName();

    const int input_type = inputs.size() > 0 ? inputs[0]->elemType() : -1;
    const int output_type = outputs[0]->elemType();

    const std::unordered_set<int>& supported_version8_types = {
        TensorProto_DataType::TensorProto_DataType_FLOAT,
        TensorProto_DataType::TensorProto_DataType_FLOAT16,
        TensorProto_DataType::TensorProto_DataType_DOUBLE,
    };

    const std::unordered_set<int>& unsupported_version9_types = {
        TensorProto_DataType::TensorProto_DataType_COMPLEX128,
        TensorProto_DataType::TensorProto_DataType_COMPLEX64,
        TensorProto_DataType::TensorProto_DataType_STRING,
    };

    ONNX_ASSERTM(
        unsupported_version9_types.find(input_type) == unsupported_version9_types.end(), "Unsupported Input Type");
    ONNX_ASSERTM(
        unsupported_version9_types.find(output_type) == unsupported_version9_types.end(), "Unsupported Output Type");

    bool castInput = (node->kind() != kConstant);
    bool castOutput = (node->kind() != kGreater && node->kind() != kLess);
    if (castInput && supported_version8_types.find(input_type) == supported_version8_types.end()) {
      for (size_t i = 0; i < inputs.size(); i++) {
        Node* pre_cast = create_cast_op(
            graph,
            inputs[i],
            TensorProto_DataType::TensorProto_DataType_FLOAT,
            inputs[i]->sizes(),
            "pre_cast_" + ONNX_NAMESPACE::to_string(i));
        pre_cast->insertBefore(node);
        node->replaceInput(i, pre_cast->output());
      }
    }
    if (castOutput && supported_version8_types.find(output_type) == supported_version8_types.end()) {
      const use_list original_uses(node->output()->uses());
      node->output()->setElemType(TensorProto_DataType::TensorProto_DataType_FLOAT);
      node->output()->setUniqueName(original_output_name + "_intermediate_output");
      Node* post_cast = create_cast_op(graph, outputs[0], output_type, outputs[0]->sizes(), original_output_name);

      post_cast->insertAfter(node);

      for (Use u : original_uses) {
        u.user->replaceInputWith(node->output(), post_cast->output());
      }

      for (size_t i = 0; i < graph->outputs().size(); i++) {
        if (graph->outputs()[i]->uniqueName() == node->output()->uniqueName()) {
          graph->return_node()->replaceInput(i, post_cast->output());
        }
      }
    }
  }

  Node* adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    adapt_type_extension(graph, node);
    return node;
  }
};

} // namespace version_conversion
} // namespace ONNX_NAMESPACE
