/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "onnx/defs/schema.h"

#include <stdexcept>
#include <unordered_set>
#include <utility>

#include "onnx/checker.h"
#include "onnx/defs/operator_sets.h"
#include "onnx/defs/operator_sets_preview.h"
#include "onnx/defs/operator_sets_training.h"

#ifdef ONNX_ML
#include "onnx/defs/operator_sets_ml.h"
#endif

#include "onnx/common/assertions.h"
#include "onnx/defs/parser.h"

namespace ONNX_NAMESPACE {
// -1 means ONNX schema hasn't been loaded yet
// 0 means all versions of ONNX schema have been loaded
// Other positive integer means the ONNX schemas for the specified version have been loaded
int OpSchemaRegistry::loaded_schema_version = -1;

constexpr int OpSchema::kUninitializedSinceVersion;

// By default if opset_version_to_load=0, it registers all opset schema for all opset versions
// Otherwise, it only registers the latest schema according to opset_version_to_load
void RegisterSchema(OpSchema schema, int opset_version_to_load, bool fail_duplicate_schema) {
  OpSchemaRegistry::OpSchemaRegisterOnce ONNX_UNUSED registration(schema, opset_version_to_load, fail_duplicate_schema);
}

#ifndef NDEBUG
DbgOperatorSetTracker& DbgOperatorSetTracker::Instance() {
  static DbgOperatorSetTracker instance;
  return instance;
}
#endif

const std::string& OpSchema::FormalParameter::GetName() const {
  return name_;
}

const DataTypeSet& OpSchema::FormalParameter::GetTypes() const {
  return type_set_;
}

DataTypeSet& OpSchema::FormalParameter::MutableTypes() {
  return type_set_;
}

const std::string& OpSchema::FormalParameter::GetTypeStr() const {
  return type_str_;
}

const std::string& OpSchema::FormalParameter::GetDescription() const {
  return description_;
}

OpSchema::FormalParameterOption OpSchema::FormalParameter::GetOption() const {
  return param_option_;
}

bool OpSchema::FormalParameter::GetIsHomogeneous() const {
  return is_homogeneous_;
}

int OpSchema::FormalParameter::GetMinArity() const {
  return min_arity_;
}

OpSchema::DifferentiationCategory OpSchema::FormalParameter::GetDifferentiationCategory() const {
  return differentiation_category_;
}

OpSchemaRegistry* OpSchemaRegistry::Instance() {
  static OpSchemaRegistry instance;
  return &instance;
}

void OpSchema::CheckInputOutputType(struct InferenceContext& ctx) const {
  std::unordered_map<std::string, std::string> type_constraints;
  // check all input types
  for (size_t in_idx = 0; in_idx < ctx.getNumInputs(); ++in_idx) {
    // If the last input is Variadic by definition, checker still needs to check the rest of actual input's type
    const auto& param = (in_idx < inputs_.size()) ? inputs_[in_idx] : inputs_.back();
    const auto& type_str = param.GetTypeStr();
    const auto& param_type = ctx.getInputType(in_idx);
    const auto& all_types = param.GetTypes();
    if (nullptr == param_type || param_type->value_case() == TypeProto::VALUE_NOT_SET) {
      continue;
    } else if (!all_types.empty() && all_types.find(Utils::DataTypeUtils::ToType(*param_type)) == all_types.end()) {
      fail_check(
          param.GetName(),
          " typestr: ",
          type_str,
          ", has unsupported type: ",
          *Utils::DataTypeUtils::ToType(*param_type));
    }
    if (param.GetIsHomogeneous()) {
      const auto& type_proto = Utils::DataTypeUtils::ToType(*param_type);
      auto p = type_constraints.emplace(type_str, *type_proto);
      if (!p.second) {
        // failed to insert a new element due to a duplication, now check consistency
        if (p.first->second != *type_proto) {
          fail_check(param.GetName(), " has inconsistent type ", *Utils::DataTypeUtils::ToType(*param_type));
        }
      }
    }
  } // for inputs
  // check all output types
  for (size_t out_idx = 0; out_idx < ctx.getNumOutputs(); ++out_idx) {
    // If the last output is Variadic by definition, checker still needs to check the rest of actual output's type
    const auto& param = (out_idx < outputs_.size()) ? outputs_[out_idx] : outputs_.back();
    const auto& type_str = param.GetTypeStr();
    const auto& param_type = ctx.getOutputType(out_idx);
    const auto& all_types = param.GetTypes();
    bool output_type_found = true;
    // infer type if necessary
    if (param_type->value_case() == TypeProto::VALUE_NOT_SET) {
      if (all_types.size() == 1) {
        *param_type = Utils::DataTypeUtils::ToTypeProto(*all_types.begin());
      } else if (type_constraints.find(type_str) != type_constraints.end()) {
        auto data_type = Utils::DataTypeUtils::ToType(type_constraints[type_str]);
        *param_type = Utils::DataTypeUtils::ToTypeProto(data_type);
      } else {
        output_type_found = false;
      }
    }
    if (!output_type_found) {
      continue;
    }
    if (!all_types.empty() && all_types.find(Utils::DataTypeUtils::ToType(*param_type)) == all_types.end()) {
      fail_check(param.GetName(), " has unsupported type ", *Utils::DataTypeUtils::ToType(*param_type));
    }
    if (param.GetIsHomogeneous()) {
      const auto& type_proto = Utils::DataTypeUtils::ToType(*param_type);
      if (type_constraints.find(type_str) == type_constraints.end()) {
        type_constraints[type_str] = *type_proto;
      } else if (type_constraints[type_str] != *type_proto) {
        fail_check(param.GetName(), " has inconsistent type ", *Utils::DataTypeUtils::ToType(*param_type));
      }
    } // else
  } // for outputs
}

void OpSchema::Verify(const NodeProto& node) const {
  if (deprecated_) {
    fail_check("Operator '", name_, "' has been deprecated since version ", since_version_);
  }

  // Check the number of inputs.
  if (node.input_size() < min_input_ || node.input_size() > max_input_) {
    fail_check(
        "Node (",
        node.name(),
        ") has input size ",
        node.input_size(),
        " not in range [min=",
        min_input_,
        ", max=",
        max_input_,
        "].");
  }

  if (!num_inputs_allowed_(node.input_size())) {
    fail_check("Node (", node.name(), ") has input size ", node.input_size(), " not in allowed input sizes.");
  }

  // Check the number of outputs.
  if (node.output_size() < min_output_ || node.output_size() > max_output_) {
    fail_check(
        "Node (",
        node.name(),
        ") has output size ",
        node.output_size(),
        " not in range [min=",
        min_output_,
        ", max=",
        max_output_,
        "].");
  }

  if (!num_outputs_allowed_(node.output_size())) {
    fail_check("Node (", node.name(), "has output size ", node.output_size(), " not in allowed output sizes.");
  }

  // Check the values of inputs / outputs
  for (int in_idx = 0; in_idx < node.input_size(); ++in_idx) {
    if (in_idx >= static_cast<int>(inputs_.size())) {
      if (!inputs_.empty() && Variadic == inputs_.back().GetOption()) {
        // The last input formal parameter should be variadic.
        break;
      } else {
        fail_check(
            "Node (",
            node.name(),
            ") has more inputs (",
            node.input_size(),
            ") than declared (",
            inputs_.size(),
            ") in op definition.");
      }
    }
    if (node.input(in_idx).empty() && (Single == inputs_[in_idx].GetOption())) {
      fail_check("Node (", node.name(), ")'s input ", in_idx, " is marked single but has an empty string in the graph");
    }
  }

  for (int out_idx = 0; out_idx < node.output_size(); ++out_idx) {
    if (out_idx >= static_cast<int>(outputs_.size())) {
      if (!outputs_.empty() && Variadic == outputs_.back().GetOption()) {
        // The last output formal parameter should be variadic.
        break;
      } else {
        fail_check(
            "Node (",
            node.name(),
            ") has more outputs (",
            node.output_size(),
            ") than declared (",
            outputs_.size(),
            ") in op definition.");
      }
    }

    if (node.output(out_idx).empty() && (Single == outputs_[out_idx].GetOption())) {
      fail_check(
          "Node (", node.name(), ")'s output ", out_idx, " is marked single but has an empty string in the graph");
    }
  }

  // An internal symbol is defined as starting with two underscores. Attributes
  // with names meeting this condition are considered implementation details
  // and should be ignored for the purpose of schema checking.
  auto isInternalSymbol = [](const std::string& sym) -> bool {
    return sym.length() >= 2 && sym[0] == '_' && sym[1] == '_';
  };

  // Check attributes
  std::unordered_set<std::string> seen_attr_names{};
  for (const auto& attr_proto : node.attribute()) {
    const auto& name = attr_proto.name();

    if (!seen_attr_names.insert(name).second) {
      fail_check("Attribute '", name, "' appeared multiple times.");
    };

    const auto& search = attributes_.find(name);
    AttributeProto::AttributeType expected_type;
    if (search != attributes_.end()) {
      expected_type = search->second.type;
    } else if (allows_unchecked_attributes_ || isInternalSymbol(name)) {
      continue;
    } else {
      fail_check("Unrecognized attribute: ", name, " for operator ", node.op_type());
    }

    // Type would be UNDEFINED if not set
    if (attr_proto.type() != expected_type) {
      fail_check("Mismatched attribute type in '", node.name() + " : " + name, "'");
    }

    // ref_attr_name is only valid when non-empty
    // we simply read default value if not present
    if (!attr_proto.ref_attr_name().empty()) {
      continue;
    }

    switch (expected_type) {
      // if attr_proto().type() != UNDEFINED
      // we consider primitive types to be set even
      // if proto3 did not output default values into the stream
      // in which case we will read the default
      case AttributeProto::FLOAT:
      case AttributeProto::INT:
      case AttributeProto::STRING:
        break;
      case AttributeProto::TENSOR:
        if (!attr_proto.has_t()) {
          fail_check("Attribute '", name, "' is expected to have field 't'");
        }
        break;
      case AttributeProto::SPARSE_TENSOR:
        if (!attr_proto.has_sparse_tensor()) {
          fail_check("Attribute '", name, "' is expected to have field 'sparse_tensor'");
        }
        break;
      case AttributeProto::GRAPH:
        if (!attr_proto.has_g()) {
          fail_check("Attribute '", name, "' is expected to have field 'g'");
        }
        break;
      case AttributeProto::TYPE_PROTO:
        if (!attr_proto.has_tp()) {
          fail_check("Attribute '", name, "' is expected to have field 'type_proto'");
        }
        break;
      case AttributeProto::INTS:
      case AttributeProto::FLOATS:
      case AttributeProto::TENSORS:
      case AttributeProto::STRINGS:
      case AttributeProto::SPARSE_TENSORS:
      case AttributeProto::GRAPHS:
      case AttributeProto::TYPE_PROTOS:
        // No check ... whether an empty list is a valid value for the attribute
        // is op specific.
        break;
      default:
        fail_check("Attribute '", name, " has unknown expected type");
    }
  }
  for (const auto& pair : attributes_) {
    const auto& attr = pair.second;
    if (!attr.required) {
      continue;
    }
    if (!seen_attr_names.count(attr.name)) {
      fail_check("Required attribute '", attr.name, "' is missing.");
    }
  }

  // Phew. All verifications passed.
}

OpSchema& OpSchema::SinceVersion(OperatorSetVersion v) {
  since_version_ = v;

  // SinceVersion is called after FunctionBody and SetContextDependentFunctionBodyBuilder are called
  // when defining a op.
  // FunctionBody() and SetContextDependentFunctionBodyBuilder() use -1 as the default opset_version
  // default opset_version is for a FunctionProto of the same opset_version as the op's since_version_.
  // It is indexed with -1 so we need to reindex it with since_version_.
  //
  // FunctionProtos of non-default opset_versions are for models whose opset version is higher than the op's
  // opset version such that ops used in the default function_proto are no longer valid. For example:
  // A model of opset version 18 contains a LayerNormalization op.
  // LayerNormalization is function op whese function body uses ReduceMean op.
  // LayerNormalization's since_version is 17 thus it is good for the model of opset 18.
  // however, if a runtime needs to inline LayerNormalization, the inlined model has a ReduceMean op.
  // ReduceMean in opset 18 is different from opset 17.
  // This requires us to define more than one function body
  std::map<int, ContextDependentFunctionBodyBuilder>::const_iterator it =
      opset_version_to_function_builder_.find(OpSchema::kUninitializedSinceVersion);

  if (it != opset_version_to_function_builder_.cend()) {
    opset_version_to_function_builder_[since_version_] = it->second;
    opset_version_to_function_builder_.erase(it);
  }

  std::map<int, std::shared_ptr<FunctionProto>>::const_iterator it_function_body =
      opset_version_to_function_body_.find(OpSchema::kUninitializedSinceVersion);
  if (it_function_body != opset_version_to_function_body_.cend()) {
    opset_version_to_function_body_[since_version_] = it_function_body->second;
    UpdateFunctionProtoOpsetImportVersion(*opset_version_to_function_body_[since_version_], since_version_);
    opset_version_to_function_body_.erase(it_function_body);
  }

  return *this;
}

OpSchema& OpSchema::Deprecate() {
  deprecated_ = true;
  return *this;
}

OpSchema& OpSchema::NumInputs(std::set<int> allowed_input_nums) {
  num_inputs_allowed_ = [allowed_input_nums = std::move(allowed_input_nums)](int n) -> bool {
    return allowed_input_nums.count(n);
  };
  return *this;
}

OpSchema& OpSchema::NumOutputs(std::set<int> allowed_output_nums) {
  num_outputs_allowed_ = [allowed_output_nums = std::move(allowed_output_nums)](int n) -> bool {
    return allowed_output_nums.count(n) > 0;
  };
  return *this;
}

OpSchema& OpSchema::TypeAndShapeInferenceFunction(InferenceFunction inferenceFunction) {
  tensor_inference_function_ = std::move(inferenceFunction);
  return *this;
}

OpSchema& OpSchema::PartialDataPropagationFunction(DataPropagationFunction dataPropagationFunction) {
  data_propagation_function_ = std::move(dataPropagationFunction);
  return *this;
}

OpSchema& OpSchema::SetSupportLevel(SupportType support) {
  support_ = support;
  return *this;
}

// Functions to specify name for the operator schema.
OpSchema& OpSchema::SetName(std::string name) {
  name_ = std::move(name);
  return *this;
}

OpSchema& OpSchema::SetName(const char* name) {
  return SetName(std::string(name));
}

// Functions to specify code location for the operator schema.
OpSchema& OpSchema::SetLocation(std::string file, int line) {
  file_ = std::move(file);
  line_ = line;
  return *this;
}

OpSchema& OpSchema::SetLocation(const char* file, int line) {
  return SetLocation(std::string(file), line);
}

OpSchema& OpSchema::SetDomain(std::string domain) {
  domain_ = std::move(domain);
  return *this;
}

OpSchema& OpSchema::SetDomain(const char* domain) {
  return SetDomain(std::string(domain));
}

OpSchema& OpSchema::Attr(Attribute attr) {
  auto name = attr.name; // copy name so we can move attr in the next line
  attributes_.insert(std::make_pair(std::move(name), std::move(attr)));
  return *this;
}

OpSchema& OpSchema::Attr(std::string name, std::string description, AttributeProto::AttributeType type, bool required) {
  Attr(Attribute{std::move(name), std::move(description), type, required});
  return *this;
}

OpSchema& OpSchema::Attr(
    std::string name,
    std::string description,
    std::string conditionExplanation,
    AttributeProto::AttributeType attr_type) {
  AttributeProto a;
  a.set_name(name);
  a.set_type(attr_type);
  if (attr_type == AttributeProto_AttributeType_UNDEFINED) {
    a.mutable_t()->set_data_type(TensorProto_DataType_UNDEFINED);
  }
  a.mutable_doc_string()->assign(std::move(conditionExplanation));
  return Attr(Attribute{std::move(name), std::move(description), std::move(a)});
}

OpSchema& OpSchema::Attr(const char* name, const char* description, AttributeProto::AttributeType type, bool required) {
  return Attr(std::string(name), std::string(description), type, required);
}

#define ATTR_SETTER_WITH_SINGLE_VALUE(type, field, attrtype)                                                           \
  OpSchema& OpSchema::Attr(                                                                                            \
      std::string name, std::string description, AttributeProto::AttributeType attr_type, const type& default_value) { \
    if (attrtype != attr_type) {                                                                                       \
      fail_schema("Attribute specification type mismatch.");                                                           \
    }                                                                                                                  \
    AttributeProto a;                                                                                                  \
    a.set_name(name);                                                                                                  \
    a.set_##field(default_value);                                                                                      \
    a.set_type(attr_type);                                                                                             \
    Attr(Attribute(std::move(name), std::move(description), std::move(a)));                                            \
    return *this;                                                                                                      \
  }                                                                                                                    \
  OpSchema& OpSchema::Attr(                                                                                            \
      const char* name, const char* description, AttributeProto::AttributeType attr_type, const type& default_value) { \
    return Attr(std::string(name), std::string(description), attr_type, default_value);                                \
  }

#define ATTR_SETTER_WITH_LIST_VALUE(type, field, attrtype)                  \
  OpSchema& OpSchema::Attr(                                                 \
      std::string name,                                                     \
      std::string description,                                              \
      AttributeProto::AttributeType attr_type,                              \
      const std::vector<type>& default_value) {                             \
    if (attrtype != attr_type) {                                            \
      fail_schema("Attribute specification type mismatch.");                \
    }                                                                       \
    AttributeProto a;                                                       \
    a.set_name(name);                                                       \
    a.set_type(attr_type);                                                  \
    for (const auto& v : default_value) {                                   \
      a.add_##field(v);                                                     \
    }                                                                       \
    Attr(Attribute(std::move(name), std::move(description), std::move(a))); \
    return *this;                                                           \
  }

#define ATTR_SETTER_WITH_SINGLE_COMPLEXVALUE(type, field, attrtype)                                                    \
  OpSchema& OpSchema::Attr(                                                                                            \
      std::string name, std::string description, AttributeProto::AttributeType attr_type, const type& default_value) { \
    if (attrtype != attr_type) {                                                                                       \
      fail_schema("Attribute specification type mismatch.");                                                           \
    }                                                                                                                  \
    AttributeProto a;                                                                                                  \
    a.set_name(name);                                                                                                  \
    *(a.mutable_##field()) = default_value;                                                                            \
    a.set_type(attr_type);                                                                                             \
    Attr(Attribute(std::move(name), std::move(description), a));                                                       \
    return *this;                                                                                                      \
  }

#define ATTR_SETTER_WITH_LIST_COMPLEXVALUE(type, field, attrtype)           \
  OpSchema& OpSchema::Attr(                                                 \
      std::string name,                                                     \
      std::string description,                                              \
      AttributeProto::AttributeType attr_type,                              \
      const std::vector<type>& default_value) {                             \
    if (attrtype != attr_type) {                                            \
      fail_schema("Attribute specification type mismatch.");                \
    }                                                                       \
    AttributeProto a;                                                       \
    a.set_name(name);                                                       \
    a.set_type(attr_type);                                                  \
    for (const auto& v : default_value) {                                   \
      *(a.add_##field()) = v;                                               \
    }                                                                       \
    Attr(Attribute(std::move(name), std::move(description), std::move(a))); \
    return *this;                                                           \
  }

ATTR_SETTER_WITH_SINGLE_VALUE(int64_t, i, AttributeProto::INT)
ATTR_SETTER_WITH_SINGLE_VALUE(float, f, AttributeProto::FLOAT)
ATTR_SETTER_WITH_SINGLE_VALUE(std::string, s, AttributeProto::STRING)
ATTR_SETTER_WITH_SINGLE_COMPLEXVALUE(TensorProto, t, AttributeProto::TENSOR)
ATTR_SETTER_WITH_SINGLE_COMPLEXVALUE(GraphProto, g, AttributeProto::GRAPH)
ATTR_SETTER_WITH_SINGLE_COMPLEXVALUE(TypeProto, tp, AttributeProto::TYPE_PROTO)
ATTR_SETTER_WITH_LIST_VALUE(int64_t, ints, AttributeProto::INTS)
ATTR_SETTER_WITH_LIST_VALUE(float, floats, AttributeProto::FLOATS)
ATTR_SETTER_WITH_LIST_COMPLEXVALUE(std::string, strings, AttributeProto::STRINGS)
ATTR_SETTER_WITH_LIST_COMPLEXVALUE(TensorProto, tensors, AttributeProto::TENSORS)
ATTR_SETTER_WITH_LIST_COMPLEXVALUE(GraphProto, graphs, AttributeProto::GRAPHS)
ATTR_SETTER_WITH_LIST_COMPLEXVALUE(TypeProto, type_protos, AttributeProto::TYPE_PROTOS)

OpSchema& OpSchema::AllowUncheckedAttributes() {
  allows_unchecked_attributes_ = true;
  return *this;
}

OpSchema& OpSchema::Input(int n, FormalParameter formal_parameter) {
  if (inputs_.size() <= static_cast<size_t>(n)) {
    inputs_.resize(n + 1);
  }
  inputs_[n] = std::move(formal_parameter);
  return *this;
}

OpSchema& OpSchema::Input(
    int n,
    std::string name,
    const std::string& description,
    std::string type_str,
    OpSchema::FormalParameterOption param_option,
    bool is_homogeneous,
    int min_arity,
    DifferentiationCategory differentiation_category) {
  return Input(
      n,
      FormalParameter(
          std::move(name),
#ifndef __ONNX_NO_DOC_STRINGS
          description,
#else
          std::string(),
#endif
          std::move(type_str),
          param_option,
          is_homogeneous,
          min_arity,
          differentiation_category));
}

OpSchema& OpSchema::Input(
    int n,
    const char* name,
    const char* description,
    const char* type_str,
    FormalParameterOption param_option,
    bool is_homogeneous,
    int min_arity,
    DifferentiationCategory differentiation_category) {
  return Input(
      n,
      std::string(name),
#ifndef __ONNX_NO_DOC_STRINGS
      std::string(description),
#else
      std::string(),
#endif
      std::string(type_str),
      param_option,
      is_homogeneous,
      min_arity,
      differentiation_category);
}

OpSchema& OpSchema::Output(int n, FormalParameter formal_parameter) {
  if (outputs_.size() <= static_cast<size_t>(n)) {
    outputs_.resize(n + 1);
  }
  outputs_[n] = std::move(formal_parameter);
  return *this;
}

OpSchema& OpSchema::Output(
    int n,
    std::string name,
    const std::string& description,
    std::string type_str,
    OpSchema::FormalParameterOption param_option,
    bool is_homogeneous,
    int min_arity,
    DifferentiationCategory differentiation_category) {
  return Output(
      n,
      FormalParameter(
          std::move(name),
#ifndef __ONNX_NO_DOC_STRINGS
          description,
#else
          std::string(),
#endif
          std::move(type_str),
          param_option,
          is_homogeneous,
          min_arity,
          differentiation_category));
}

OpSchema& OpSchema::Output(
    int n,
    const char* name,
    const char* description,
    const char* type_str,
    FormalParameterOption param_option,
    bool is_homogeneous,
    int min_arity,
    DifferentiationCategory differentiation_category) {
  return Output(
      n,
      std::string(name),
#ifndef __ONNX_NO_DOC_STRINGS
      std::string(description),
#else
      std::string(),
#endif
      std::string(type_str),
      param_option,
      is_homogeneous,
      min_arity,
      differentiation_category);
}

OpSchema&
OpSchema::TypeConstraint(std::string type_str, std::vector<std::string> constraints, std::string description) {
  if (type_constraints_.end() != type_constraints_.find(type_str)) {
    fail_schema("Duplicate type constraint name");
  }

  DataTypeSet d;
  for (const auto& t : constraints) {
    d.insert(Utils::DataTypeUtils::ToType(t));
  }
  type_constraints_.insert(std::make_pair(type_str, std::make_pair(d, description)));
  type_constraint_params_.push_back(
      TypeConstraintParam(std::move(type_str), std::move(constraints), std::move(description)));
  return *this;
}

OpSchema& OpSchema::TypeConstraint(
    const char* type_str,
    std::initializer_list<const char*> constraints,
    const char* description) {
  std::vector<std::string> constraints_vector;
  constraints_vector.reserve(constraints.size());
  for (auto iter = constraints.begin(); iter != constraints.end(); ++iter) {
    constraints_vector.push_back(*iter);
  }

  return TypeConstraint(std::string(type_str), constraints_vector, std::string(description));
}

void OpSchema::ParseAndSetTypes(
    /*out*/ std::vector<OpSchema::FormalParameter>* formal_parameters) {
  for (auto& formal_parameter : *formal_parameters) {
    auto& type = formal_parameter.GetTypeStr();
    DataTypeSet allowed_types;
    auto it = type_constraints_.find(type);
    if (it != type_constraints_.end()) {
      allowed_types = it->second.first;
    } else {
      allowed_types.emplace(Utils::DataTypeUtils::ToType(type));
    }

    formal_parameter.MutableTypes() = allowed_types;
  }
}

OpSchema& OpSchema::SetContextDependentFunctionBodyBuilder(
    ContextDependentFunctionBodyBuilder functionBuilder,
    int opset_version) {
  if (opset_version == OpSchema::kUninitializedSinceVersion && since_version_ != OpSchema::kUninitializedSinceVersion) {
    opset_version_to_function_builder_[since_version_] = std::move(functionBuilder);
  } else {
    opset_version_to_function_builder_[opset_version] = std::move(functionBuilder);
  }
  return *this;
}

bool OpSchema::BuildContextDependentFunction(
    const FunctionBodyBuildContext& ctx,
    FunctionProto& function_proto,
    int requested_opset_version) const {
  if (requested_opset_version == OpSchema::kUninitializedSinceVersion)
    requested_opset_version = since_version_;

  std::map<int, ContextDependentFunctionBodyBuilder>::const_iterator it =
      opset_version_to_function_builder_.upper_bound(requested_opset_version);
  if (opset_version_to_function_builder_.empty() || it == opset_version_to_function_builder_.begin()) {
    ONNX_THROW_EX(std::out_of_range(
        std::string("Cannot find a function builder that satisfies the requested opset version: op_type = ") +
        this->name_ + ", opset_version = " + std::to_string(requested_opset_version) + "."));
  } else {
    --it;
    const ContextDependentFunctionBodyBuilder& body_builder = it->second;
    if (!body_builder(ctx, *this, function_proto)) {
      return false;
    }
    //// default opset import may have been added to function_proto by OpSchema::BuildFunction
    //// we need to update its version with the specified opset_version
    UpdateFunctionProtoOpsetImportVersion(function_proto, requested_opset_version);
    ValidateReferencedOpsInFuncton(&function_proto, requested_opset_version, it->first);
    return true;
  }
}

// A function of a schema (either stored in opset_version_to_function_body_ or built with one of function builder
// in opset_version_to_function_builder_) has predefined opset_imports. Before returning the function, we shall
// update the predefined opset_imports so that it is consistent with the requested version.
// Note that this call only update opset_import of the default domain.
// TODO: extend this call to work for no-default domains.
void OpSchema::UpdateFunctionProtoOpsetImportVersion(FunctionProto& function_proto, int requested_opset_version) const {
  bool opset_import_exist = false;
  for (int i = 0; i < function_proto.opset_import_size(); i++) {
    auto* schema_opset = function_proto.mutable_opset_import(i);
    if (schema_opset->domain() == domain_) {
      if (schema_opset->version() != requested_opset_version) {
        schema_opset->set_version(requested_opset_version);
      }
      opset_import_exist = true;
    }
  }

  if (!opset_import_exist) {
    auto* schema_opset = function_proto.mutable_opset_import()->Add();
    schema_opset->set_domain(domain_);
    schema_opset->set_version(requested_opset_version);
  }
}

OpSchema& OpSchema::FunctionBody(const char* func_body, int opset_version) {
  if (opset_version == OpSchema::kUninitializedSinceVersion && since_version_ != OpSchema::kUninitializedSinceVersion) {
    opset_version = since_version_;
  }
  std::shared_ptr<FunctionProto> function_proto(new FunctionProto());
  OnnxParser parser(func_body);
  auto status = parser.Parse(*function_proto->mutable_node());
  if (!status.IsOK())
    ONNX_THROW_EX(std::logic_error("Error parsing function body:" + status.ErrorMessage()));
  if (!parser.EndOfInput())
    ONNX_THROW_EX(std::logic_error("Extra unparsed input unexpected."));

  // opset import may have been set
  // we may need to update its version with the specified opset_version
  UpdateFunctionProtoOpsetImportVersion(*function_proto, opset_version);

  opset_version_to_function_body_.insert(std::make_pair(opset_version, function_proto));
  return *this;
}

OpSchema& OpSchema::FunctionBody(const std::vector<NodeProto>& func_nodes, int opset_version) {
  if (opset_version == OpSchema::kUninitializedSinceVersion && since_version_ != OpSchema::kUninitializedSinceVersion) {
    opset_version = since_version_;
  }
  std::shared_ptr<FunctionProto> function_proto(new FunctionProto());
  for (const auto& node : func_nodes) {
    auto new_node = function_proto->add_node();
    new_node->CopyFrom(node);
  }

  // opset import may have been set
  // we may need to update its version with the specified opset_version
  UpdateFunctionProtoOpsetImportVersion(*function_proto, opset_version);
  opset_version_to_function_body_.insert(std::make_pair(opset_version, function_proto));
  return *this;
}

OpSchema& OpSchema::FunctionBody(
    const std::vector<NodeProto>& func_nodes,
    const std::vector<OperatorSetIdProto>& relied_opsets,
    int opset_version) {
  if (opset_version == OpSchema::kUninitializedSinceVersion && since_version_ != OpSchema::kUninitializedSinceVersion) {
    opset_version = since_version_;
  }

  std::shared_ptr<FunctionProto> function_proto(new FunctionProto());
  for (auto& relied_opset : relied_opsets) {
    *(function_proto->mutable_opset_import()->Add()) = relied_opset;
  }

  for (const auto& node : func_nodes) {
    auto new_node = function_proto->add_node();
    new_node->CopyFrom(node);
  }
  // opset import may have been set
  // we may need to update its version with the specified opset_version
  UpdateFunctionProtoOpsetImportVersion(*function_proto, opset_version);
  opset_version_to_function_body_.insert(std::make_pair(opset_version, function_proto));
  return *this;
}

const FunctionProto* OpSchema::GetFunction(int requested_opset_version, bool validate) const {
  if (opset_version_to_function_body_.empty())
    return nullptr;
  // Return latest FunctionProto when opset version request is not set
  if (requested_opset_version == OpSchema::kUninitializedSinceVersion) {
    return opset_version_to_function_body_.rbegin()->second.get();
  }
  std::map<int, std::shared_ptr<FunctionProto>>::const_iterator it =
      opset_version_to_function_body_.upper_bound(requested_opset_version);
  if (it != opset_version_to_function_body_.begin()) {
    --it;
    int function_since_version = it->first;
    const FunctionProto* function = it->second.get();
    if (!validate || ValidateReferencedOpsInFuncton(function, requested_opset_version, function_since_version)) {
      return function;
    }
  }
  return nullptr;
}

// when requesting a function at loading time,
// requested_opset_version does not have to be the same as function_since_version.
// When they are not the same, it is necessary to verify that ops used to define the function
// are not updated between function_since_version and requested_opset_version (include requested_opset_version).
// this call only validate ops in the default domain.
// TODO: validate ops in other domains.
bool OpSchema::ValidateReferencedOpsInFuncton(
    const FunctionProto* function,
    int requested_opset_version,
    int function_since_version,
    std::set<std::string>* updated_ops) const {
  bool all_ops_are_invalid = true;
  if (requested_opset_version == function_since_version) {
    return all_ops_are_invalid;
  }
  for (auto& node : function->node()) {
    if (node.domain() == "" || node.domain() == "ai.onnx") {
      const OpSchema* op1 =
          OpSchemaRegistry::Instance()->GetSchema(node.op_type(), requested_opset_version, node.domain());
      const OpSchema* op2 =
          OpSchemaRegistry::Instance()->GetSchema(node.op_type(), function_since_version, node.domain());
      if (op1 != op2) {
        if (updated_ops) {
          updated_ops->insert(node.op_type());
        }
        all_ops_are_invalid = false;
      }
    }
  }

  return all_ops_are_invalid;
}

OpSchema& OpSchema::FillUsing(const std::function<void(OpSchema&)>& populator) {
  if (populator) {
    populator(*this);
  }
  return *this;
}

void OpSchema::BuildFunction(FunctionProto& function_body) const {
  function_body.set_name(this->name_);
  function_body.set_doc_string(this->doc_);
  function_body.set_domain(this->domain_);
  for (auto& i : inputs_) {
    function_body.add_input(i.GetName());
  }
  for (auto& o : outputs_) {
    function_body.add_output(o.GetName());
  }
  for (auto& a : attributes_) {
    function_body.add_attribute(a.first);
  }

  // In a typical onnx function where the function and all the
  // ops in function body belong to the same domain we implicitly add
  // {domain_, since_version_} to funciton opset imports if it is not already added.
  // This is simply for convienince. If any of the function body ops do not belong to same
  // domain as function itself, then the function author needs to explicitly add all the relevant
  // opset imports.
  if (function_body.opset_import().size() == 0) {
    auto* schema_opset = function_body.mutable_opset_import()->Add();
    schema_opset->set_domain(domain_);
    schema_opset->set_version(since_version_);
  }
}

void OpSchema::Finalize() {
#define ENFORCE(x)                                                                                      \
  do {                                                                                                  \
    if (!(x))                                                                                           \
      ONNX_THROW_EX(std::logic_error("ONNX Schema " + name_ + ": failed validating the check: " + #x)); \
  } while (0)

  // Calculate min/max number of inputs.
  // <Min number of inputs> = <number of "single" inputs> + <number of
  // "optional" but not trailing inputs>. <Max number of inputs> = <number of
  // all inputs or std::numeric_limits<int>::max() (if the last input is
  // variadic).

  // Flag indicates whether an optional input is trailing one (there's no single
  // or variadic input behind).
  for (size_t i = 0; i < inputs_.size(); ++i) {
    switch (inputs_[i].GetOption()) {
      case OpSchema::Single:
        ++max_input_;
        min_input_ = max_input_;
        break;
      case OpSchema::Optional:
        ++max_input_;
        break;
      case OpSchema::Variadic:
        // Only last input formal parameter could be variadic.
        ENFORCE((inputs_.size() - 1) == i);
        min_input_ = max_input_ + inputs_[i].GetMinArity();
        max_input_ = std::numeric_limits<int>::max();
        break;
    }
  }

  // Calculate min/max number of outputs.
  for (size_t i = 0; i < outputs_.size(); ++i) {
    switch (outputs_[i].GetOption()) {
      case OpSchema::Single:
        ++max_output_;
        min_output_ = max_output_;
        break;
      case OpSchema::Optional:
        ++max_output_;
        break;
      case OpSchema::Variadic:
        // Only last output formal parameter could be variadic.
        ENFORCE((outputs_.size() - 1) == i);
        min_output_ = max_output_ + outputs_[i].GetMinArity();
        max_output_ = std::numeric_limits<int>::max();
        break;
    }
  }

  // all inputs and outputs have names
  for (const auto& it : inputs_) {
    ENFORCE(!(it.GetName().empty()));
  }
  for (const auto& it : outputs_) {
    ENFORCE(!(it.GetName().empty()));
  }

  ParseAndSetTypes(&inputs_);
  ParseAndSetTypes(&outputs_);

  for (auto& func : opset_version_to_function_body_) {
    BuildFunction(*func.second);
  }
}

std::ostream& operator<<(std::ostream& out, const OpSchema& schema) {
  if (!schema.attributes_.empty()) {
    out << "Attributes:" << std::endl;
    for (const auto& pair : schema.attributes_) {
      out << "  " << pair.second.name << " : " << pair.second.description << std::endl;
    }
  }
  if (schema.max_input_ > 0) {
    out << "Inputs:" << std::endl;
    if (!schema.inputs_.empty()) {
      for (size_t i = 0; i < schema.inputs_.size(); ++i) {
        const auto& p = schema.inputs_[i];
        const auto& name = p.GetName();
        const auto& description = p.GetDescription();
        const auto& type_str = p.GetTypeStr();
        out << "  " << i << ", " << (!name.empty() ? name : "(unnamed)") << " : "
            << (!description.empty() ? description : "(no doc)") << " : "
            << (!type_str.empty() ? type_str : "(no type)") << std::endl;
      }
    } else {
      out << "  (no explicit description available)" << std::endl;
    }
  }
  if (schema.max_output_ > 0) {
    out << "Outputs:" << std::endl;
    if (!schema.outputs_.empty()) {
      for (size_t i = 0; i < schema.outputs_.size(); ++i) {
        const auto& p = schema.outputs_[i];
        const auto& name = p.GetName();
        const auto& description = p.GetDescription();
        const auto& type_str = p.GetTypeStr();
        out << "  " << i << ", " << (!name.empty() ? name : "(unnamed)") << " : "
            << (!description.empty() ? description : "(no doc)") << " : "
            << (!type_str.empty() ? type_str : "(no type)") << std::endl;
      }
    } else {
      out << "  (no explicit description available)" << std::endl;
    }
  }
  out << std::endl;
  if (schema.doc()) {
    out << schema.doc();
  } else {
    out << "(no documentation yet)" << std::endl;
  }
  out << std::endl;
  if (schema.line_) {
    out << "Defined at " << schema.file_ << ":" << schema.line_ << std::endl;
  }
  return out;
}

OpSchemaRegistry::DomainToVersionRange& OpSchemaRegistry::DomainToVersionRange::Instance() {
  static DomainToVersionRange domain_to_version_range;
  return domain_to_version_range;
};

// Private method used by OpSchemaRegisterOnce and OpSchemaRegistry::map()
OpName_Domain_Version_Schema_Map& OpSchemaRegistry::GetMapWithoutEnsuringRegistration() {
  static OpName_Domain_Version_Schema_Map map;
  return map;
}

OpName_Domain_Version_Schema_Map& OpSchemaRegistry::map() {
  auto& map = GetMapWithoutEnsuringRegistration();

  // The following class is used to register operators the
  // first time this method is called, in a thread-safe fashion.
  class SchemasRegisterer {
   public:
    SchemasRegisterer() {
      // In debug builds, the number of schema registered in this constructor
      // is compared against the number of calls to schema registration macros.
#ifndef NDEBUG
      size_t dbg_initial_schema_count = GetRegisteredSchemaCount();
#endif

      RegisterOnnxOperatorSetSchema();

#ifdef ONNX_ML
      RegisterOnnxMLOperatorSetSchema();
#endif

      // Invoke register of training operators.
      RegisterOnnxTrainingOperatorSetSchema();

      // Invoke register of experimental operators.
      RegisterOnnxPreviewOperatorSetSchema();

#ifndef NDEBUG
      size_t dbg_registered_schema_count = GetRegisteredSchemaCount() - dbg_initial_schema_count;
      // Check enabled only if schemas for all opset versions are loaded
      if (OpSchemaRegistry::Instance()->GetLoadedSchemaVersion() == 0) {
        ONNX_ASSERTM(
            dbg_registered_schema_count == ONNX_DBG_GET_COUNT_IN_OPSETS(),
            "%u schema were exposed from operator sets and automatically placed into the static registry.  "
            "%u were expected based on calls to registration macros. Operator set functions may need to be updated.",
            dbg_registered_schema_count,
            ONNX_DBG_GET_COUNT_IN_OPSETS());
      }
#endif
    }

   private:
    static size_t GetRegisteredSchemaCount() {
      size_t count = 0;
      for (auto& x : GetMapWithoutEnsuringRegistration()) {
        for (auto& y : x.second) {
          count += y.second.size();
        }
      }
      return count;
    }
  };

#ifndef __ONNX_DISABLE_STATIC_REGISTRATION
  static SchemasRegisterer schemasRegisterer;
#endif

  return map;
}

size_t ReplaceAll(std::string& s, const char* from, const char* to) {
  size_t numReplaced = 0;
  std::string::size_type lenFrom = std::strlen(from);
  std::string::size_type lenTo = std::strlen(to);
  for (std::string::size_type pos = s.find(from); pos != std::string::npos; pos = s.find(from, pos + lenTo)) {
    s.replace(pos, lenFrom, to);
    numReplaced++;
  }
  return numReplaced;
}

} // namespace ONNX_NAMESPACE
