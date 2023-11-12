/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "attr_proto_util.h"
#include "onnx/common/constants.h"
#include "onnx/common/status.h"
#include "onnx/defs/parser.h"
#include "onnx/defs/schema.h"
#include "tensor_proto_util.h"

namespace ONNX_NAMESPACE {
// Helper function to expand a function node given the function proto
void FunctionExpandHelper(
    const NodeProto& node,
    const FunctionProto& func,
    GraphProto& g,
    const std::string& node_prefix = "");

class FunctionBodyHelper {
 public:
  struct AttributeProtoWrapper {
    AttributeProto proto;

    AttributeProtoWrapper() {}

    AttributeProtoWrapper(const AttributeProto& attr_prot) {
      proto = attr_prot;
    }

    template <typename T>
    AttributeProtoWrapper(const std::string& attr_name, const T& value) {
      proto = MakeAttribute(attr_name, value);
    }
  };

  struct NodeDef {
    NodeDef(
        std::vector<std::string> outputs,
        std::string op_type,
        std::vector<std::string> inputs,
        std::vector<AttributeProtoWrapper> attributes = {},
        std::string domain = "")
        : outputs(std::move(outputs)),
          op_type(std::move(op_type)),
          inputs(std::move(inputs)),
          attributes(std::move(attributes)),
          domain(std::move(domain)) {}

    std::vector<std::string> outputs;
    std::string op_type;
    std::vector<std::string> inputs;
    std::vector<AttributeProtoWrapper> attributes;
    std::string domain;
  };

  /*
  BuildNodes() is an utility function for easily define a Function Body.

  To build a simple node:
    {{"Z"}, "Add", {"X", "Y"}} represents Z = Add(X,Y)

  To build a node with attribute:
    {{"Y"}, "Concat", {"X1", "X2", "X3"}, {{"axis", 1}}}
      represents Y = Concat(X1,X2,X3) with axis = 1
    The attribute type are infered from the attribute value's c++ type
    Supported value types are
      int64_t -> int, vector<int64_t> -> ints
      float -> float, vector<float> -> floats
      string -> string, vector<string> ->strings
    For refering an attribute from parent, use:
      {MakeRefAttribute("axes", AttributeProto::INTS)}}

  To build a node which belongs to a domain other than onnx standard domain:
    {{"Z"}, "Foo", {"X", "Y"}, "customdomain"} represents Z = customdomain.Foo(X,Y)
    or
    {{"Y"}, "Bar", {"X1", "X2", "X3"}, {{"axis", 1}}, "customdomain"}
      represents Y = customdomain.Bar(X1,X2,X3) with axis = 1

  For more examples, please find the references of this function
  */
  static std::vector<NodeProto> BuildNodes(const std::vector<NodeDef>& node_defs);

  static void BuildNodes(FunctionProto& functionProto, const std::vector<NodeDef>& node_defs);

  static bool BuildFunctionProto(
      FunctionProto& functionProto,
      const OpSchema& schema,
      const std::vector<NodeDef>& node_defs,
      const std::vector<OperatorSetIdProto>& relied_opsets);

  template <typename T>
  static NodeDef Const(const std::string& name, const T& value) {
    return NodeDef{{name}, "Constant", {}, {{"value", ToTensor<T>(value)}}};
  }

  template <typename T>
  static NodeDef Const(const std::string& name, const std::vector<T>& values) {
    return NodeDef{{name}, "Constant", {}, {{"value", ToTensor<T>(values)}}};
  }
};

class FunctionBuilder {
 public:
  FunctionBuilder(FunctionProto& funProto_) : funProto(funProto_) {}

  FunctionBuilder& Add(const char* nodes_txt) {
    OnnxParser parser(nodes_txt);
    auto& nodes = *funProto.mutable_node();

    while (!parser.EndOfInput()) {
      auto status = parser.Parse(*nodes.Add());
      if (!status.IsOK())
        ONNX_THROW_EX(std::logic_error("Error parsing node:" + status.ErrorMessage()));
    }

    return *this;
  }

  FunctionBuilder& Add(const char* node_txt, const AttributeProto& attr) {
    OnnxParser parser(node_txt);
    auto& node = *funProto.add_node();
    auto status = parser.Parse(node);
    if (!status.IsOK()) {
      ONNX_THROW_EX(std::logic_error("Error parsing node:" + status.ErrorMessage()));
    }

    if (!parser.EndOfInput()) {
      ONNX_THROW_EX(std::logic_error("Error unexpected extra input in node:" + status.ErrorMessage()));
    }

    *node.add_attribute() = attr;

    return *this;
  }

  template <typename T>
  FunctionBuilder& Add(const char* node_txt, const std::string& attr_name, const T& attr_value) {
    return Add(node_txt, MakeAttribute(attr_name, attr_value));
  }

  FunctionBuilder& Const(const std::string& name, const TensorProto& tensor) {
    std::string constant_op(name);
    constant_op += " = Constant()";
    return Add(constant_op.c_str(), MakeAttribute("value", tensor));
  }

  // Creates a scalar constant (a tensor of rank zero).
  template <typename T>
  FunctionBuilder& Const(const std::string& name, T const_value) {
    std::string constant_op(name);
    constant_op += " = Constant()";
    return Add(constant_op.c_str(), MakeAttribute("value", ToTensor(const_value)));
  }

  // Creates a 1D tensor constant consisting of a single value.
  template <typename T>
  FunctionBuilder& Const1D(const std::string& name, T const_value) {
    std::string constant_op(name);
    constant_op += " = Constant()";
    auto tensor = ToTensor(const_value);
    tensor.add_dims(1);
    return Add(constant_op.c_str(), MakeAttribute("value", tensor));
  }

  // Creates a 1D tensor constant consisting of zero or more values.
  template <typename T>
  FunctionBuilder& Const(const std::string& name, const std::vector<T>& values) {
    std::string constant_op(name);
    constant_op += " = Constant()";
    auto tensor = ToTensor(values);
    tensor.add_dims(values.size()); // Treat as 1D tensor.

    return Add(constant_op.c_str(), MakeAttribute("value", tensor));
  }

  FunctionBuilder& AddOpset(const char* domain, int version) {
    auto* opset = funProto.add_opset_import();
    opset->set_domain(domain);
    opset->set_version(version);
    return *this;
  }

 private:
  FunctionProto& funProto;
};

} // namespace ONNX_NAMESPACE
