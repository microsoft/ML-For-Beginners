/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "onnx/defs/function.h"
#include "onnx/defs/schema.h"
#include "onnx/string_utils.h"

namespace ONNX_NAMESPACE {
std::string InteralTensorNameGenerator(const std::string& node_name, const std::string& internal_name) {
  std::string new_name = "Func_" + node_name + internal_name;
  return new_name;
}

void FunctionExpandHelper(
    const NodeProto& node,
    const FunctionProto& func,
    GraphProto& g,
    const std::string& node_prefix) {
  // Create a temporary unique node prefix for tensor names
  std::string uniq_prefix = node_prefix;
  if (uniq_prefix.empty()) {
    const void* address = static_cast<const void*>(&node);
    std::stringstream ss;
    ss << address;
    uniq_prefix = ss.str();
  }
  std::string node_name = node.has_name() ? node.name() : func.name() + uniq_prefix;
  std::unordered_map<std::string, std::string> io_names_map;
  std::unordered_map<std::string, AttributeProto> attr_map;

  for (int idx = 0; idx < node.input_size(); ++idx) {
    if (idx >= func.input_size()) {
      ONNX_THROW("Input for function node " + node_name + " is out of bounds");
    }
    io_names_map[func.input().Get(idx)] = node.input().Get(idx);
  }
  for (int idx = 0; idx < node.output_size(); ++idx) {
    if (idx >= func.output_size()) {
      ONNX_THROW("Output for function node " + node_name + " is out of bounds");
    }
    // If the node output is missing, the corresponding function output should
    // be treated as an internal value (not as missing) because it could also be
    // an intermediate value.
    if (node.output().Get(idx) == "") {
      continue;
    }
    io_names_map[func.output().Get(idx)] = node.output().Get(idx);
  }

  for (auto& attr : node.attribute()) {
    attr_map[attr.name()] = attr;
  }

  // For undefined attributes of the function node
  // add default values obtained from the function schema.
  // get the domain version for function schema
  int domain_version = -1;
  for (const auto& opset_import : func.opset_import()) {
    if (opset_import.domain() == node.domain()) {
      domain_version = static_cast<int>(opset_import.version());
    }
  }
  if (domain_version == -1) {
    ONNX_THROW("No opset import registered for domain '" + node.domain() + "' in function proto");
  }

  const OpSchemaRegistry* schema_registry = OpSchemaRegistry::Instance();
  const auto schema = schema_registry->GetSchema(node.op_type(), domain_version, node.domain());
  std::map<std::string, OpSchema::Attribute> default_attrs = schema->attributes();

  for (const auto& pair : default_attrs) {
    const auto& attr_name = pair.first;
    const auto& attr = pair.second;
    if (!attr_map.count(attr_name)) {
      attr_map[attr_name] = attr.default_value;
    }
  }

  for (auto& function_node : func.node()) {
    NodeProto* new_node = g.add_node();
    new_node->CopyFrom(function_node);
    new_node->clear_input();
    new_node->clear_output();
    new_node->clear_attribute();
    for (auto& input : function_node.input()) {
      if (io_names_map.count(input)) {
        new_node->add_input(io_names_map[input]);
      } else {
        new_node->add_input(InteralTensorNameGenerator(node_name, input));
      }
    }
    for (auto& output : function_node.output()) {
      if (io_names_map.count(output)) {
        new_node->add_output(io_names_map[output]);
      } else {
        new_node->add_output(InteralTensorNameGenerator(node_name, output));
      }
    }
    for (auto& attr : function_node.attribute()) {
      if (attr.has_ref_attr_name()) {
        if (attr_map.count(attr.ref_attr_name())) {
          AttributeProto* new_attr = new_node->add_attribute();
          new_attr->CopyFrom(attr_map[attr.ref_attr_name()]);
          new_attr->set_name(attr.name());
        }
      } else {
        AttributeProto* new_attr = new_node->add_attribute();
        new_attr->CopyFrom(attr);
      }
    }
  }
}

std::vector<NodeProto> FunctionBodyHelper::BuildNodes(const std::vector<NodeDef>& node_defs) {
  std::vector<NodeProto> nodes(node_defs.size());

  for (size_t i = 0; i < node_defs.size(); i++) {
    const NodeDef& node = node_defs[i];
    NodeProto& n = nodes[i];

    n.set_op_type(node.op_type);
    n.set_domain(node.domain);
    for (const auto& i : node.inputs) {
      n.add_input(i);
    }
    for (const auto& o : node.outputs) {
      n.add_output(o);
    }
    for (const auto& attr : node.attributes) {
      *(n.add_attribute()) = attr.proto;
    }
  }

  return nodes;
}

void FunctionBodyHelper::BuildNodes(FunctionProto& functionProto, const std::vector<NodeDef>& node_defs) {
  for (size_t i = 0; i < node_defs.size(); i++) {
    const NodeDef& node = node_defs[i];
    auto* np = functionProto.add_node();

    np->set_op_type(node.op_type);
    np->set_domain(node.domain);
    for (const auto& inp : node.inputs) {
      np->add_input(inp);
    }
    for (const auto& o : node.outputs) {
      np->add_output(o);
    }
    for (const auto& attr : node.attributes) {
      *(np->add_attribute()) = attr.proto;
    }
  }
}

bool FunctionBodyHelper::BuildFunctionProto(
    FunctionProto& functionProto,
    const OpSchema& schema,
    const std::vector<NodeDef>& node_defs,
    const std::vector<OperatorSetIdProto>& relied_opsets) {
  BuildNodes(functionProto, node_defs);

  for (auto& relied_opset : relied_opsets) {
    *(functionProto.mutable_opset_import()->Add()) = relied_opset;
  }

  schema.BuildFunction(functionProto);
  return true;
}

} // namespace ONNX_NAMESPACE
