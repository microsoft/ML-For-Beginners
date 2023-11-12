// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include "onnx/defs/function.h"
#include "onnx/defs/schema.h"
#include "onnx/onnx-data_pb.h"
#include "onnx/onnx-operators_pb.h"
#include "onnx/onnx_pb.h"
#include "onnx/string_utils.h"

namespace ONNX_NAMESPACE {
namespace checker {
class ValidationError final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
  const char* what() const noexcept override {
    if (!expanded_message_.empty()) {
      return expanded_message_.c_str();
    }
    return std::runtime_error::what();
  }
  void AppendContext(const std::string& context) {
    expanded_message_ = ONNX_NAMESPACE::MakeString(std::runtime_error::what(), "\n\n==> Context: ", context);
  }

 private:
  std::string expanded_message_;
};

#define fail_check(...) \
  ONNX_THROW_EX(ONNX_NAMESPACE::checker::ValidationError(ONNX_NAMESPACE::MakeString(__VA_ARGS__)));

class CheckerContext final {
 public:
  int get_ir_version() const {
    return ir_version_;
  }
  void set_ir_version(int v) {
    ir_version_ = v;
  }
  const std::unordered_map<std::string, int>& get_opset_imports() const {
    return opset_imports_;
  }
  void set_opset_imports(std::unordered_map<std::string, int> imps) {
    opset_imports_ = std::move(imps);
  }
  bool is_main_graph() const {
    return is_main_graph_;
  }
  void set_is_main_graph(bool is_main_graph) {
    is_main_graph_ = is_main_graph;
  }

  void set_schema_registry(const ISchemaRegistry* schema_registry) {
    schema_registry_ = schema_registry;
  }

  const ISchemaRegistry* get_schema_registry() const {
    return schema_registry_;
  }

  void set_model_dir(const std::string& model_dir) {
    model_dir_ = model_dir;
  }

  std::string get_model_dir() const {
    return model_dir_;
  }

  explicit CheckerContext() : ir_version_(-1) {}

 private:
  int ir_version_;
  std::unordered_map<std::string, int> opset_imports_;
  bool is_main_graph_ = true;
  const ISchemaRegistry* schema_registry_ = OpSchemaRegistry::Instance();
  std::string model_dir_;
};

class LexicalScopeContext final {
 public:
  LexicalScopeContext() = default;

  // Construct an instance with the lexical scope from the parent graph to allow
  // lookup of names from that scope via this_or_ancestor_graph_has.
  // The caller must ensure parent_context remains valid for the entire lifetime
  // of the new instance. Alternatively, if that cannot be guaranteed, create an
  // instance with the default constructor and populate output_names with the
  // values from the parent scope so the values are copied instead.
  LexicalScopeContext(const LexicalScopeContext& parent_context) : parent_context_{&parent_context} {}
  LexicalScopeContext& operator=(const LexicalScopeContext& parent_context) {
    parent_context_ = &parent_context;
    return *this;
  }

  void add(const std::string& name) {
    output_names.insert(name);
  }

  bool this_graph_has(const std::string& name) const {
    return output_names.find(name) != output_names.cend();
  }

  bool this_or_ancestor_graph_has(const std::string& name) const {
    return this_graph_has(name) || (parent_context_ && parent_context_->this_or_ancestor_graph_has(name));
  }

  // public for backwards compatibility. please prefer the public interface of
  // this class over directly changing output_names
  std::unordered_set<std::string> output_names;

 private:
  const LexicalScopeContext* parent_context_{nullptr};
};

using IR_VERSION_TYPE = decltype(Version::IR_VERSION);
void check_value_info(const ValueInfoProto& value_info, const CheckerContext&);
void check_tensor(const TensorProto& tensor, const CheckerContext&);
void check_sparse_tensor(const SparseTensorProto& sparse_tensor, const CheckerContext&);
void check_sequence(const SequenceProto& sequence, const CheckerContext&);
void check_map(const MapProto& map, const CheckerContext&);
void check_optional(const OptionalProto& opt, const CheckerContext&);
void check_attribute(const AttributeProto& attr, const CheckerContext&, const LexicalScopeContext&);
void check_node(const NodeProto& node, const CheckerContext&, const LexicalScopeContext&);
void check_graph(const GraphProto& graph, const CheckerContext&, const LexicalScopeContext&);
void check_function(const FunctionProto& function, const CheckerContext&, const LexicalScopeContext&);

// Check schema compatibility for 2 opset versions for a given node.
// Checks whether the schema for 2 versions is same, this is true when the opschema
// does not change between versions.
void check_opset_compatibility(
    const NodeProto& node,
    const CheckerContext& ctx,
    const std::unordered_map<std::string, int>& func_opset_imports,
    const std::unordered_map<std::string, int>& model_opset_imports);

// Checks all model local functions present in ModelProto
void check_model_local_functions(
    const ModelProto& model,
    const CheckerContext& ctx,
    const LexicalScopeContext& parent_lex);

void check_model(const ModelProto& model, bool full_check = false);
void check_model(const std::string& model_path, bool full_check = false);

bool check_is_experimental_op(const NodeProto& node);

} // namespace checker
} // namespace ONNX_NAMESPACE
