// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx/common/model_helpers.h"

#include "onnx/checker.h"
#include "onnx/defs/schema.h"
#include "onnx/string_utils.h"

namespace ONNX_NAMESPACE {

Common::Status BuildNode(
    const std::string& name,
    const std::string& domain,
    const std::string& doc_string,
    const std::string& op_type,
    std::vector<std::string> const& inputs,
    std::vector<std::string> const& outputs,
    NodeProto* node) {
  if (node == NULL) {
    return Common::Status(Common::CHECKER, Common::INVALID_ARGUMENT, "node_proto should not be nullptr.");
  }
  node->set_name(name);
  node->set_domain(domain);
  node->set_doc_string(doc_string);
  node->set_op_type(op_type);
  for (auto& input : inputs) {
    node->add_input(input);
  }
  for (auto& output : outputs) {
    node->add_output(output);
  }

  return Common::Status::OK();
}
} // namespace ONNX_NAMESPACE
