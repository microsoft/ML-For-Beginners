// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <climits>
#include <limits>
#include <tuple>
#include <unordered_map>

#include "onnx/checker.h"
#include "onnx/defs/function.h"
#include "onnx/defs/parser.h"
#include "onnx/defs/printer.h"
#include "onnx/defs/schema.h"
#include "onnx/py_utils.h"
#include "onnx/shape_inference/implementation.h"
#include "onnx/version_converter/convert.h"

namespace ONNX_NAMESPACE {
namespace py = pybind11;
using namespace pybind11::literals;

template <typename ProtoType>
static std::tuple<bool, py::bytes, py::bytes> Parse(const char* cstr) {
  ProtoType proto{};
  OnnxParser parser(cstr);
  auto status = parser.Parse(proto);
  std::string out;
  proto.SerializeToString(&out);
  return std::make_tuple(status.IsOK(), py::bytes(status.ErrorMessage()), py::bytes(out));
}

template <typename ProtoType>
static std::string ProtoBytesToText(const py::bytes& bytes) {
  ProtoType proto{};
  ParseProtoFromPyBytes(&proto, bytes);
  return ProtoToString(proto);
}

template <typename T, typename Ts = typename std::remove_const<T>::type>
std::pair<std::unique_ptr<Ts[]>, std::unordered_map<std::string, T*>> ParseProtoFromBytesMap(
    std::unordered_map<std::string, py::bytes> bytesMap) {
  std::unique_ptr<Ts[]> values(new Ts[bytesMap.size()]);
  std::unordered_map<std::string, T*> result;
  size_t i = 0;
  for (auto kv : bytesMap) {
    ParseProtoFromPyBytes(&values[i], kv.second);
    result[kv.first] = &values[i];
    i++;
  }
  return std::make_pair(std::move(values), result);
}

std::unordered_map<std::string, py::bytes> CallNodeInferenceFunction(
    OpSchema* schema,
    const py::bytes& nodeBytes,
    std::unordered_map<std::string, py::bytes> valueTypesByNameBytes,
    std::unordered_map<std::string, py::bytes> inputDataByNameBytes,
    std::unordered_map<std::string, py::bytes> inputSparseDataByNameBytes,
    std::unordered_map<std::string, int> opsetImports,
    const int irVersion) {
  NodeProto node{};
  ParseProtoFromPyBytes(&node, nodeBytes);
  // Early fail if node is badly defined - may throw ValidationError
  schema->Verify(node);

  // Convert arguments to C++ types, allocating memory
  const auto& valueTypes = ParseProtoFromBytesMap<TypeProto>(valueTypesByNameBytes);
  const auto& inputData = ParseProtoFromBytesMap<const TensorProto>(inputDataByNameBytes);
  const auto& inputSparseData = ParseProtoFromBytesMap<const SparseTensorProto>(inputSparseDataByNameBytes);
  if (opsetImports.empty()) {
    opsetImports[schema->domain()] = schema->SinceVersion();
  }

  shape_inference::GraphInferenceContext graphInferenceContext(
      valueTypes.second, opsetImports, nullptr, {}, OpSchemaRegistry::Instance(), nullptr, irVersion);
  // Construct inference context and get results - may throw InferenceError
  shape_inference::InferenceContextImpl ctx(
      node, valueTypes.second, inputData.second, inputSparseData.second, nullptr, &graphInferenceContext);
  schema->GetTypeAndShapeInferenceFunction()(ctx);
  // Verify the inference succeeded - may also throw ValidationError
  // Note that input types were not validated until now (except that their count was correct)
  schema->CheckInputOutputType(ctx);

  // Convert back into bytes returned to Python
  std::unordered_map<std::string, py::bytes> typeProtoBytes;
  for (size_t i = 0; i < ctx.allOutputTypes_.size(); i++) {
    const auto& proto = ctx.allOutputTypes_[i];
    if (proto.IsInitialized()) {
      std::string s;
      proto.SerializeToString(&s);
      typeProtoBytes[node.output(i)] = py::bytes(s);
    }
  }

  return typeProtoBytes;
}

PYBIND11_MODULE(onnx_cpp2py_export, onnx_cpp2py_export) {
  onnx_cpp2py_export.doc() = "Python interface to ONNX";

  onnx_cpp2py_export.attr("ONNX_ML") = py::bool_(
#ifdef ONNX_ML
      true
#else // ONNX_ML
      false
#endif // ONNX_ML
  );

  // Submodule `schema`
  auto defs = onnx_cpp2py_export.def_submodule("defs");
  defs.doc() = "Schema submodule";
  py::register_exception<SchemaError>(defs, "SchemaError");

  py::class_<OpSchema> op_schema(defs, "OpSchema", "Schema of an operator.");

  // Define the class enums first because they are used as default values in function definitions
  py::enum_<OpSchema::FormalParameterOption>(op_schema, "FormalParameterOption")
      .value("Single", OpSchema::Single)
      .value("Optional", OpSchema::Optional)
      .value("Variadic", OpSchema::Variadic);

  py::enum_<OpSchema::DifferentiationCategory>(op_schema, "DifferentiationCategory")
      .value("Unknown", OpSchema::Unknown)
      .value("Differentiable", OpSchema::Differentiable)
      .value("NonDifferentiable", OpSchema::NonDifferentiable);

  py::enum_<AttributeProto::AttributeType>(op_schema, "AttrType")
      .value("FLOAT", AttributeProto::FLOAT)
      .value("INT", AttributeProto::INT)
      .value("STRING", AttributeProto::STRING)
      .value("TENSOR", AttributeProto::TENSOR)
      .value("GRAPH", AttributeProto::GRAPH)
      .value("FLOATS", AttributeProto::FLOATS)
      .value("INTS", AttributeProto::INTS)
      .value("STRINGS", AttributeProto::STRINGS)
      .value("TENSORS", AttributeProto::TENSORS)
      .value("GRAPHS", AttributeProto::GRAPHS)
      .value("SPARSE_TENSOR", AttributeProto::SPARSE_TENSOR)
      .value("SPARSE_TENSORS", AttributeProto::SPARSE_TENSORS)
      .value("TYPE_PROTO", AttributeProto::TYPE_PROTO)
      .value("TYPE_PROTOS", AttributeProto::TYPE_PROTOS);

  py::enum_<OpSchema::SupportType>(op_schema, "SupportType")
      .value("COMMON", OpSchema::SupportType::COMMON)
      .value("EXPERIMENTAL", OpSchema::SupportType::EXPERIMENTAL);

  py::class_<OpSchema::Attribute>(op_schema, "Attribute")
      .def(
          py::init([](std::string name, AttributeProto::AttributeType type, std::string description, bool required) {
            // Construct an attribute.
            // Use a lambda to swap the order of the arguments to match the Python API
            return OpSchema::Attribute(std::move(name), std::move(description), type, required);
          }),
          py::arg("name"),
          py::arg("type"),
          py::arg("description") = "",
          py::kw_only(),
          py::arg("required") = true)
      .def(
          py::init([](std::string name, const py::object& default_value, std::string description) {
            // Construct an attribute with a default value.
            // Attributes with default values are not required
            auto bytes = default_value.attr("SerializeToString")().cast<py::bytes>();
            AttributeProto proto{};
            ParseProtoFromPyBytes(&proto, bytes);
            return OpSchema::Attribute(std::move(name), std::move(description), std::move(proto));
          }),
          py::arg("name"),
          py::arg("default_value"), // type: onnx.AttributeProto
          py::arg("description") = "")
      .def_readonly("name", &OpSchema::Attribute::name)
      .def_readonly("description", &OpSchema::Attribute::description)
      .def_readonly("type", &OpSchema::Attribute::type)
      .def_property_readonly(
          "_default_value",
          [](OpSchema::Attribute* attr) -> py::bytes {
            std::string out;
            attr->default_value.SerializeToString(&out);
            return out;
          })
      .def_readonly("required", &OpSchema::Attribute::required);

  py::class_<OpSchema::TypeConstraintParam>(op_schema, "TypeConstraintParam")
      .def(
          py::init<std::string, std::vector<std::string>, std::string>(),
          py::arg("type_param_str"),
          py::arg("allowed_type_strs"),
          py::arg("description") = "")
      .def_readonly("type_param_str", &OpSchema::TypeConstraintParam::type_param_str)
      .def_readonly("allowed_type_strs", &OpSchema::TypeConstraintParam::allowed_type_strs)
      .def_readonly("description", &OpSchema::TypeConstraintParam::description);

  py::class_<OpSchema::FormalParameter>(op_schema, "FormalParameter")
      .def(
          py::init([](std::string name,
                      std::string type_str,
                      const std::string& description,
                      OpSchema::FormalParameterOption param_option,
                      bool is_homogeneous,
                      int min_arity,
                      OpSchema::DifferentiationCategory differentiation_category) {
            // Use a lambda to swap the order of the arguments to match the Python API
            return OpSchema::FormalParameter(
                std::move(name),
                description,
                std::move(type_str),
                param_option,
                is_homogeneous,
                min_arity,
                differentiation_category);
          }),
          py::arg("name"),
          py::arg("type_str"),
          py::arg("description") = "",
          py::kw_only(),
          py::arg("param_option") = OpSchema::Single,
          py::arg("is_homogeneous") = true,
          py::arg("min_arity") = 1,
          py::arg("differentiation_category") = OpSchema::DifferentiationCategory::Unknown)

      .def_property_readonly("name", &OpSchema::FormalParameter::GetName)
      .def_property_readonly("types", &OpSchema::FormalParameter::GetTypes)
      .def_property_readonly("type_str", &OpSchema::FormalParameter::GetTypeStr)
      .def_property_readonly("description", &OpSchema::FormalParameter::GetDescription)
      .def_property_readonly("option", &OpSchema::FormalParameter::GetOption)
      .def_property_readonly("is_homogeneous", &OpSchema::FormalParameter::GetIsHomogeneous)
      .def_property_readonly("min_arity", &OpSchema::FormalParameter::GetMinArity)
      .def_property_readonly("differentiation_category", &OpSchema::FormalParameter::GetDifferentiationCategory)
      // Legacy camel cased names. We retain them for backward compatibility.
      // TODO(#5074): Remove these before the 1.16 release.
      .def_property_readonly(
          "typeStr",
          [](const OpSchema::FormalParameter& self) {
            auto warnings = py::module::import("warnings");
            warnings.attr("warn")(
                "OpSchema.FormalParameter.typeStr is deprecated and will be removed in 1.16. "
                "Use OpSchema.FormalParameter.type_str instead.");
            return self.GetTypeStr();
          })
      .def_property_readonly(
          "isHomogeneous",
          [](const OpSchema::FormalParameter& self) {
            auto warnings = py::module::import("warnings");
            warnings.attr("warn")(
                "OpSchema.FormalParameter.isHomogeneous is deprecated and will be removed in 1.16. "
                "Use OpSchema.FormalParameter.is_homogeneous instead.");
            return self.GetIsHomogeneous();
          })
      .def_property_readonly("differentiationCategory", [](const OpSchema::FormalParameter& self) {
        auto warnings = py::module::import("warnings");
        warnings.attr("warn")(
            "OpSchema.FormalParameter.differentiationCategory is deprecated and will be removed in 1.16. "
            "Use OpSchema.FormalParameter.differentiation_category instead.");
        return self.GetDifferentiationCategory();
      });

  op_schema
      .def(
          py::init([](std::string name,
                      std::string domain,
                      int since_version,
                      std::string doc,
                      std::vector<OpSchema::FormalParameter> inputs,
                      std::vector<OpSchema::FormalParameter> outputs,
                      std::vector<std::tuple<std::string, std::vector<std::string>, std::string>> type_constraints,
                      std::vector<OpSchema::Attribute> attributes) {
            auto self = OpSchema();

            self.SetName(std::move(name)).SetDomain(std::move(domain)).SinceVersion(since_version).SetDoc(doc);
            // Add inputs and outputs
            for (auto i = 0; i < inputs.size(); ++i) {
              self.Input(i, std::move(inputs[i]));
            }
            for (auto i = 0; i < outputs.size(); ++i) {
              self.Output(i, std::move(outputs[i]));
            }
            // Add type constraints
            for (auto& type_constraint : type_constraints) {
              std::string type_str;
              std::vector<std::string> constraints;
              std::string description;
              tie(type_str, constraints, description) = std::move(type_constraint);
              self.TypeConstraint(std::move(type_str), std::move(constraints), std::move(description));
            }
            // Add attributes
            for (auto& attribute : attributes) {
              self.Attr(std::move(attribute));
            }

            self.Finalize();

            return self;
          }),
          py::arg("name"),
          py::arg("domain"),
          py::arg("since_version"),
          py::arg("doc") = "",
          py::kw_only(),
          py::arg("inputs") = std::vector<OpSchema::FormalParameter>{},
          py::arg("outputs") = std::vector<OpSchema::FormalParameter>{},
          py::arg("type_constraints") = std::vector<std::tuple<
              std::string /* type_str */,
              std::vector<std::string> /* constraints */,
              std::string /* description */>>{},
          py::arg("attributes") = std::vector<OpSchema::Attribute>{})
      .def_property("name", &OpSchema::Name, [](OpSchema& self, const std::string& name) { self.SetName(name); })
      .def_property(
          "domain", &OpSchema::domain, [](OpSchema& self, const std::string& domain) { self.SetDomain(domain); })
      .def_property("doc", &OpSchema::doc, [](OpSchema& self, const std::string& doc) { self.SetDoc(doc); })
      .def_property_readonly("file", &OpSchema::file)
      .def_property_readonly("line", &OpSchema::line)
      .def_property_readonly("support_level", &OpSchema::support_level)
      .def_property_readonly("since_version", &OpSchema::since_version)
      .def_property_readonly("deprecated", &OpSchema::deprecated)
      .def_property_readonly("function_opset_versions", &OpSchema::function_opset_versions)
      .def_property_readonly(
          "context_dependent_function_opset_versions", &OpSchema::context_dependent_function_opset_versions)
      .def_property_readonly(
          "all_function_opset_versions",
          [](OpSchema* op) -> std::vector<int> {
            std::vector<int> all_function_opset_versions = op->function_opset_versions();
            std::vector<int> context_dependent_function_opset_versions =
                op->context_dependent_function_opset_versions();
            all_function_opset_versions.insert(
                all_function_opset_versions.end(),
                context_dependent_function_opset_versions.begin(),
                context_dependent_function_opset_versions.end());
            std::sort(all_function_opset_versions.begin(), all_function_opset_versions.end());
            all_function_opset_versions.erase(
                std::unique(all_function_opset_versions.begin(), all_function_opset_versions.end()),
                all_function_opset_versions.end());
            return all_function_opset_versions;
          })
      .def_property_readonly("min_input", &OpSchema::min_input)
      .def_property_readonly("max_input", &OpSchema::max_input)
      .def_property_readonly("min_output", &OpSchema::min_output)
      .def_property_readonly("max_output", &OpSchema::max_output)
      .def_property_readonly("attributes", &OpSchema::attributes)
      .def_property_readonly("inputs", &OpSchema::inputs)
      .def_property_readonly("outputs", &OpSchema::outputs)
      .def_property_readonly("has_type_and_shape_inference_function", &OpSchema::has_type_and_shape_inference_function)
      .def_property_readonly("has_data_propagation_function", &OpSchema::has_data_propagation_function)
      .def_property_readonly("type_constraints", &OpSchema::typeConstraintParams)
      .def_static("is_infinite", [](int v) { return v == std::numeric_limits<int>::max(); })
      .def(
          "_infer_node_outputs",
          CallNodeInferenceFunction,
          py::arg("nodeBytes"),
          py::arg("valueTypesByNameBytes"),
          py::arg("inputDataByNameBytes") = std::unordered_map<std::string, py::bytes>{},
          py::arg("inputSparseDataByNameBytes") = std::unordered_map<std::string, py::bytes>{},
          py::arg("opsetImports") = std::unordered_map<std::string, int>{},
          py::arg("irVersion") = int(IR_VERSION))
      .def_property_readonly("has_function", &OpSchema::HasFunction)
      .def_property_readonly(
          "_function_body",
          [](OpSchema* op) -> py::bytes {
            std::string bytes = "";
            if (op->HasFunction())
              op->GetFunction()->SerializeToString(&bytes);
            return py::bytes(bytes);
          })
      .def(
          "get_function_with_opset_version",
          [](OpSchema* op, int opset_version) -> py::bytes {
            std::string bytes = "";
            const FunctionProto* function_proto = op->GetFunction(opset_version);
            if (function_proto) {
              function_proto->SerializeToString(&bytes);
            }
            return py::bytes(bytes);
          })
      .def_property_readonly("has_context_dependent_function", &OpSchema::HasContextDependentFunction)
      .def(
          "get_context_dependent_function",
          [](OpSchema* op, const py::bytes& bytes, const std::vector<py::bytes>& input_types_bytes) -> py::bytes {
            NodeProto proto{};
            ParseProtoFromPyBytes(&proto, bytes);
            std::string func_bytes = "";
            if (op->HasContextDependentFunction()) {
              std::vector<TypeProto> input_types;
              input_types.reserve(input_types_bytes.size());
              for (auto& type_bytes : input_types_bytes) {
                TypeProto type_proto{};
                ParseProtoFromPyBytes(&type_proto, type_bytes);
                input_types.push_back(type_proto);
              }
              FunctionBodyBuildContextImpl ctx(proto, input_types);
              FunctionProto func_proto;
              op->BuildContextDependentFunction(ctx, func_proto);
              func_proto.SerializeToString(&func_bytes);
            }
            return py::bytes(func_bytes);
          })
      .def(
          "get_context_dependent_function_with_opset_version",
          [](OpSchema* op, int opset_version, const py::bytes& bytes, const std::vector<py::bytes>& input_types_bytes)
              -> py::bytes {
            NodeProto proto{};
            ParseProtoFromPyBytes(&proto, bytes);
            std::string func_bytes = "";
            if (op->HasContextDependentFunctionWithOpsetVersion(opset_version)) {
              std::vector<TypeProto> input_types;
              input_types.reserve(input_types_bytes.size());
              for (auto& type_bytes : input_types_bytes) {
                TypeProto type_proto{};
                ParseProtoFromPyBytes(&type_proto, type_bytes);
                input_types.push_back(type_proto);
              }
              FunctionBodyBuildContextImpl ctx(proto, input_types);
              FunctionProto func_proto;
              op->BuildContextDependentFunction(ctx, func_proto, opset_version);
              func_proto.SerializeToString(&func_bytes);
            }
            return py::bytes(func_bytes);
          });

  defs.def(
          "has_schema",
          [](const std::string& op_type, const std::string& domain) -> bool {
            return OpSchemaRegistry::Schema(op_type, domain) != nullptr;
          },
          "op_type"_a,
          "domain"_a = ONNX_DOMAIN)
      .def(
          "schema_version_map",
          []() -> std::unordered_map<std::string, std::pair<int, int>> {
            return OpSchemaRegistry::DomainToVersionRange::Instance().Map();
          })
      .def(
          "get_schema",
          [](const std::string& op_type, const int max_inclusive_version, const std::string& domain) -> OpSchema {
            const auto* schema = OpSchemaRegistry::Schema(op_type, max_inclusive_version, domain);
            if (!schema) {
              fail_schema(
                  "No schema registered for '" + op_type + "' version '" + std::to_string(max_inclusive_version) +
                  "' and domain '" + domain + "'!");
            }
            return *schema;
          },
          "op_type"_a,
          "max_inclusive_version"_a,
          "domain"_a = ONNX_DOMAIN,
          "Return the schema of the operator *op_type* and for a specific version.")
      .def(
          "get_schema",
          [](const std::string& op_type, const std::string& domain) -> OpSchema {
            const auto* schema = OpSchemaRegistry::Schema(op_type, domain);
            if (!schema) {
              fail_schema("No schema registered for '" + op_type + "' and domain '" + domain + "'!");
            }
            return *schema;
          },
          "op_type"_a,
          "domain"_a = ONNX_DOMAIN,
          "Return the schema of the operator *op_type* and for a specific version.")
      .def(
          "get_all_schemas",
          []() -> const std::vector<OpSchema> { return OpSchemaRegistry::get_all_schemas(); },
          "Return the schema of all existing operators for the latest version.")
      .def(
          "get_all_schemas_with_history",
          []() -> const std::vector<OpSchema> { return OpSchemaRegistry::get_all_schemas_with_history(); },
          "Return the schema of all existing operators and all versions.");

  // Submodule `checker`
  auto checker = onnx_cpp2py_export.def_submodule("checker");
  checker.doc() = "Checker submodule";

  py::class_<checker::CheckerContext> checker_context(checker, "CheckerContext");
  checker_context.def(py::init<>())
      .def_property("ir_version", &checker::CheckerContext::get_ir_version, &checker::CheckerContext::set_ir_version)
      .def_property(
          "opset_imports", &checker::CheckerContext::get_opset_imports, &checker::CheckerContext::set_opset_imports);

  py::register_exception<checker::ValidationError>(checker, "ValidationError");

  checker.def("check_value_info", [](const py::bytes& bytes, const checker::CheckerContext& ctx) -> void {
    ValueInfoProto proto{};
    ParseProtoFromPyBytes(&proto, bytes);
    checker::check_value_info(proto, ctx);
  });

  checker.def("check_tensor", [](const py::bytes& bytes, const checker::CheckerContext& ctx) -> void {
    TensorProto proto{};
    ParseProtoFromPyBytes(&proto, bytes);
    checker::check_tensor(proto, ctx);
  });

  checker.def("check_sparse_tensor", [](const py::bytes& bytes, const checker::CheckerContext& ctx) -> void {
    SparseTensorProto proto{};
    ParseProtoFromPyBytes(&proto, bytes);
    checker::check_sparse_tensor(proto, ctx);
  });

  checker.def("check_attribute", [](const py::bytes& bytes, const checker::CheckerContext& ctx) -> void {
    AttributeProto proto{};
    ParseProtoFromPyBytes(&proto, bytes);
    checker::check_attribute(proto, ctx, checker::LexicalScopeContext());
  });

  checker.def("check_node", [](const py::bytes& bytes, const checker::CheckerContext& ctx) -> void {
    NodeProto proto{};
    ParseProtoFromPyBytes(&proto, bytes);
    checker::LexicalScopeContext lex_ctx;
    checker::check_node(proto, ctx, lex_ctx);
  });

  checker.def("check_function", [](const py::bytes& bytes, const checker::CheckerContext& ctx) -> void {
    FunctionProto proto{};
    ParseProtoFromPyBytes(&proto, bytes);
    checker::check_function(proto, ctx, checker::LexicalScopeContext());
  });

  checker.def("check_graph", [](const py::bytes& bytes, const checker::CheckerContext& ctx) -> void {
    GraphProto proto{};
    ParseProtoFromPyBytes(&proto, bytes);
    checker::LexicalScopeContext lex_ctx;
    checker::check_graph(proto, ctx, lex_ctx);
  });

  checker.def(
      "check_model",
      [](const py::bytes& bytes, bool full_check) -> void {
        ModelProto proto{};
        ParseProtoFromPyBytes(&proto, bytes);
        checker::check_model(proto, full_check);
      },
      "bytes"_a,
      "full_check"_a = false);

  checker.def(
      "check_model_path",
      (void (*)(const std::string& path, bool full_check)) & checker::check_model,
      "path"_a,
      "full_check"_a = false);

  // Submodule `version_converter`
  auto version_converter = onnx_cpp2py_export.def_submodule("version_converter");
  version_converter.doc() = "VersionConverter submodule";
  py::register_exception<ConvertError>(version_converter, "ConvertError");

  version_converter.def("convert_version", [](const py::bytes& bytes, py::int_ target) {
    ModelProto proto{};
    ParseProtoFromPyBytes(&proto, bytes);
    shape_inference::InferShapes(proto);
    auto result = version_conversion::ConvertVersion(proto, target);
    std::string out;
    result.SerializeToString(&out);
    return py::bytes(out);
  });

  // Submodule `shape_inference`
  auto shape_inference = onnx_cpp2py_export.def_submodule("shape_inference");
  shape_inference.doc() = "Shape Inference submodule";
  py::register_exception<InferenceError>(shape_inference, "InferenceError");

  shape_inference.def(
      "infer_shapes",
      [](const py::bytes& bytes, bool check_type, bool strict_mode, bool data_prop) {
        ModelProto proto{};
        ParseProtoFromPyBytes(&proto, bytes);
        ShapeInferenceOptions options{check_type, strict_mode == true ? 1 : 0, data_prop};
        shape_inference::InferShapes(proto, OpSchemaRegistry::Instance(), options);
        std::string out;
        proto.SerializeToString(&out);
        return py::bytes(out);
      },
      "bytes"_a,
      "check_type"_a = false,
      "strict_mode"_a = false,
      "data_prop"_a = false);

  shape_inference.def(
      "infer_shapes_path",
      [](const std::string& model_path,
         const std::string& output_path,
         bool check_type,
         bool strict_mode,
         bool data_prop) -> void {
        ShapeInferenceOptions options{check_type, strict_mode == true ? 1 : 0, data_prop};
        shape_inference::InferShapes(model_path, output_path, OpSchemaRegistry::Instance(), options);
      });

  shape_inference.def(
      "infer_function_output_types",
      [](const py::bytes& function_proto_bytes,
         const std::vector<py::bytes> input_types_bytes,
         const std::vector<py::bytes> attributes_bytes) -> std::vector<py::bytes> {
        FunctionProto proto{};
        ParseProtoFromPyBytes(&proto, function_proto_bytes);

        std::vector<TypeProto> input_types;
        input_types.reserve(input_types_bytes.size());
        for (const py::bytes& bytes : input_types_bytes) {
          TypeProto type;
          ParseProtoFromPyBytes(&type, bytes);
          input_types.push_back(type);
        }

        std::vector<AttributeProto> attributes;
        attributes.reserve(attributes_bytes.size());
        for (const py::bytes& bytes : attributes_bytes) {
          AttributeProto attr;
          ParseProtoFromPyBytes(&attr, bytes);
          attributes.push_back(attr);
        }

        std::vector<TypeProto> output_types = shape_inference::InferFunctionOutputTypes(proto, input_types, attributes);
        std::vector<py::bytes> result;
        result.reserve(output_types.size());
        for (auto& type_proto : output_types) {
          std::string out;
          type_proto.SerializeToString(&out);
          result.push_back(py::bytes(out));
        }
        return result;
      });

  // Submodule `parser`
  auto parser = onnx_cpp2py_export.def_submodule("parser");
  parser.doc() = "Parser submodule";

  parser.def("parse_model", Parse<ModelProto>);
  parser.def("parse_graph", Parse<GraphProto>);
  parser.def("parse_function", Parse<FunctionProto>);
  parser.def("parse_node", Parse<NodeProto>);

  // Submodule `printer`
  auto printer = onnx_cpp2py_export.def_submodule("printer");
  printer.doc() = "Printer submodule";

  printer.def("model_to_text", ProtoBytesToText<ModelProto>);
  printer.def("function_to_text", ProtoBytesToText<FunctionProto>);
  printer.def("graph_to_text", ProtoBytesToText<GraphProto>);
}

} // namespace ONNX_NAMESPACE
