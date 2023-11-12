/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <climits>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <ostream>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "onnx/common/common.h"
#include "onnx/common/constants.h"
#include "onnx/defs/shape_inference.h"

namespace ONNX_NAMESPACE {

struct FunctionBodyBuildContext {
  virtual const AttributeProto* getAttribute(const std::string& name) const = 0;
  virtual bool hasInput(int inputIndex) const = 0;
  virtual bool hasOutput(int inputIndex) const = 0;
  // getInputType(i) should return null for missing optional inputs, or if
  // type-inference could not infer the input-type (erroneous model).
  virtual const TypeProto* getInputType(int inputIndex) const = 0;
  virtual ~FunctionBodyBuildContext() {}
};

struct FunctionBodyBuildContextImpl : public FunctionBodyBuildContext {
  // Input_types: use a default TypeProto for missing types. We use a different convention
  // here (from FunctionBodyBuildContext) to simplify python interoperability.
  // The default value for input_types is included only for backward compatibility.
  // It can be used for functions that do not depend on the type-context, but
  // will not be sufficient for functions that do use the type-context.
  FunctionBodyBuildContextImpl(const NodeProto& node_proto, const std::vector<TypeProto>& input_types = {})
      : node_proto_(node_proto), input_types_(input_types) {
    for (auto& attr : node_proto.attribute()) {
      attributesByName_[attr.name()] = &attr;
    }
  }

  const AttributeProto* getAttribute(const std::string& name) const override {
    auto iter = attributesByName_.find(name);
    if (iter == attributesByName_.end()) {
      return nullptr;
    } else {
      return iter->second;
    }
  }

  bool hasInput(int inputIndex) const override {
    if (inputIndex >= node_proto_.input_size())
      return false;
    return node_proto_.input(inputIndex) != "";
  }

  bool hasOutput(int inputIndex) const override {
    if (inputIndex >= node_proto_.output_size())
      return false;
    return node_proto_.output(inputIndex) != "";
  }

  const TypeProto* getInputType(int inputIndex) const override {
    if (inputIndex < 0)
      return nullptr;
    size_t j = static_cast<size_t>(inputIndex);
    if (j >= input_types_.size())
      return nullptr;
    // Convert default value (no variant set) into null.
    if (input_types_[j].value_case() == TypeProto::ValueCase::VALUE_NOT_SET)
      return nullptr;
    return &input_types_[j];
  }

  std::unordered_map<std::string, const AttributeProto*> attributesByName_;

  NodeProto node_proto_;
  std::vector<TypeProto> input_types_;
};

using FunctionBodyQueryFunction = std::function<bool(FunctionBodyBuildContext&)>;

class OpSchema;
using ContextDependentFunctionBodyBuilder =
    std::function<bool(const FunctionBodyBuildContext&, const OpSchema&, FunctionProto&)>;

class SchemaError final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;

  SchemaError(const std::string& message) : std::runtime_error(message) {}

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

#define fail_schema(...) ONNX_THROW_EX(ONNX_NAMESPACE::SchemaError(ONNX_NAMESPACE::MakeString(__VA_ARGS__)));

using OperatorSetVersion = int;

using DataTypeSet = std::unordered_set<DataType>;

// Type constraint map. Key is type string. Value is data type set and
// description.
using TypeConstraintMap = std::unordered_map<std::string, std::pair<DataTypeSet, std::string>>;

/**
 * @brief A class to record the schema of an op.
 *
 * OpSchema records the common interface of an op specified by its name.
 *
 * To register an OpSchema, one can use the macro ONNX_OPERATOR_SCHEMA(name) and
 * then append the various functions in the class. For example, for an op
 * that takes in two inputs, one output, and the first input and output
 * could be in-place, can be written as
 *
 *     ONNX_OPERATOR_SCHEMA(name)
 *         .NumInputs(2).NumOutputs(1).AllowConsumed({{0, 0}});
 *
 * To manufacture methods that may be used to register an OpSchema
 * non-statically, the following may be used:
 *
 *     ONNX_OPERATOR_SET_SCHEMA(name, version, OpSchema()
 *         .NumInputs(2).NumOutputs(1).AllowConsumed({{0, 0}}));
 */
class OpSchema final {
 public:
  static constexpr int kUninitializedSinceVersion = -1;
  // Formal parameter options.
  enum FormalParameterOption : uint8_t {
    // The formal parameter is single and not optional.
    // Number of supplied actual parameters must be 1.
    Single = 0,
    // The formal parameter is single and optional.
    // Number of supplied actual parameters may be 0 or 1.
    Optional = 1,
    // The formal parameter is variadic.
    // Number of supplied actual parameters must be N or more, where
    // the minimum value N is indicated separately (default value 1).
    Variadic = 2,
  };
  enum DifferentiationCategory : uint8_t {
    // Whether this formal parameter is differentiable or not cannot
    // be statically determined. It also covers variadic formal
    // parameters which contain both of differentiable and
    // non-differentiable variables.
    Unknown = 0,
    // This formal parameter is differentiable. That is, this formal
    // parameter can be differentiable input of Gradient operator.
    Differentiable = 1,
    // This formal parameter is not differentiable. That is, this formal
    // parameter can not be differentiable input of Gradient operator.
    NonDifferentiable = 2
  };

  // Formal parameter represenation, including input/output name, typeStr,
  // description, and type constraints.
  class FormalParameter final {
   public:
    // Constructor.
    FormalParameter() = default;

    explicit FormalParameter(
        std::string name,
        DataTypeSet allowed_type_set,
        std::string type_str,
        const std::string& description,
        FormalParameterOption param_option = Single,
        bool is_homogeneous = true,
        int min_arity = 1,
        DifferentiationCategory differentiation_category = Unknown)
        : name_(std::move(name)),
          type_set_(std::move(allowed_type_set)),
          type_str_(std::move(type_str)),
#ifndef __ONNX_NO_DOC_STRINGS
          description_(description),
#endif
          param_option_(param_option),
          is_homogeneous_(is_homogeneous),
          min_arity_(min_arity),
          differentiation_category_(differentiation_category) {
#ifdef __ONNX_NO_DOC_STRINGS
      ONNX_UNUSED_PARAMETER(description);
#endif
    }

    explicit FormalParameter(
        std::string name,
        const std::string& description,
        std::string type_str,
        FormalParameterOption param_option = Single,
        bool is_homogeneous = true,
        int min_arity = 1,
        DifferentiationCategory differentiation_category = Unknown)
        : name_(std::move(name)),
          type_str_(std::move(type_str)),
#ifndef __ONNX_NO_DOC_STRINGS
          description_(description),
#endif
          param_option_(param_option),
          is_homogeneous_(is_homogeneous),
          min_arity_(min_arity),
          differentiation_category_(differentiation_category) {
#ifdef __ONNX_NO_DOC_STRINGS
      ONNX_UNUSED_PARAMETER(description);
#endif
    }

    // Get formal parameter name.
    const std::string& GetName() const;

    // Get allowed data types.
    const DataTypeSet& GetTypes() const;

    // Get formal parameter type string.
    const std::string& GetTypeStr() const;

    // Get formal parameter description.
    const std::string& GetDescription() const;

    // Get the parameter option, it could be Single, Optional or Variadic.
    FormalParameterOption GetOption() const;

    // Get whether a variadic parameter requires all to be of same type
    bool GetIsHomogeneous() const;

    // Get minimum arity. Applicable only in the Variadic case.
    int GetMinArity() const;

    // Get the differentiation property of this formal parameter.
    DifferentiationCategory GetDifferentiationCategory() const;

   private:
    friend class OpSchema;

    DataTypeSet& MutableTypes();

    // Formal parameter name.
    std::string name_;

    // A set of data types supported for <*this> formal parameter.
    // It should contain at least one element if this formal parameter is good.
    DataTypeSet type_set_;

    // The <parameter type> string specified when registring an op.
    // It could be a supported data type or a type constraint key, which
    // maps to a set of supported data types.
    std::string type_str_;

    // Formal parameter description.
    std::string description_;

    // Formal parameter option.
    FormalParameterOption param_option_;

    // For variadic parameters, a flag indicating if all parameters must be of
    // same type
    bool is_homogeneous_;

    // Minimum number of parameters expected. Applicable only for Variadic.
    int min_arity_;

    // True if this parameter can be an differentiable inputs of Gradient.
    // Otherwise, using this parameter as an differentiable inputs of Gradient
    // is prohibited.
    DifferentiationCategory differentiation_category_;
  };

  enum class SupportType : uint8_t {
    COMMON, // Supported by all frameworks that support this IR.
    EXPERIMENTAL, // This OP is experimental and can be changed or removed in
                  // the future.
  };

  OpSchema() : OpSchema("unknown", "unknown", 0) {}
  OpSchema(std::string name, std::string file, int line)
      : name_(std::move(name)), file_(std::move(file)), line_(line), support_(SupportType::COMMON) {}

  /**
   * @brief Returns the file that the op schema is registered from.
   */
  const std::string& file() const {
    return file_;
  }

  /**
   * @brief Returns the line in file that the op schema is registered from.
   */
  int line() const {
    return line_;
  }

  /**
   * @brief Returns the support level of the op schema.
   */
  SupportType support_level() const {
    return support_;
  }

  /**
   * @brief Returns the docstring of the op schema.
   */
  const char* doc() const {
    return doc_.empty() ? nullptr : doc_.c_str();
  }

  // Check if input and output types fall into valid set and match each other
  void CheckInputOutputType(struct InferenceContext&) const;

  /**
   * @brief Verifies if a NodeProto matches the pattern specified in
   * the schema.
   */
  void Verify(const NodeProto& node) const;

  // Functions to set the property of the operator schemas.
  // Sets the number of inputs, either a fixed number or a min and a max.

  /**
   * The earliest operator set version which this operator was
   * present in.  If an operator has had no BC-breaking changes,
   * this is simply the first operator set the operator was a member
   * of; if it has had BC-breaking changes, then for the semantics
   * /as described/ in the OpSchema entry, this version describes
   * the operator set which introduced the BC-breaking change.
   *
   * For example, suppose op Foo was added in v3, and had a BC-breaking
   * change in v6.  Then there will be an op schema entry for Foo with
   * SinceVersion(3), and another, updated op schema entry for Foo
   * with SinceVersion(6).
   */
  OpSchema& SinceVersion(OperatorSetVersion n); // aka int

  /**
   * Marks this op as deprecated as of it's since_version. This will cause the
   * Schema() lookup functions to return nullptr when the version is in the
   * deprecated range.
   */
  OpSchema& Deprecate();

  bool Deprecated() const {
    return deprecated_;
  }

  /**
   * @brief Input could be one of the values specified in allowed_input_nums.
   */
  OpSchema& NumInputs(std::set<int> allowed_input_nums);

  /**
   * @brief Output could be one of the values specified in allowed_output_nums.
   */
  OpSchema& NumOutputs(std::set<int> allowed_output_nums);

  // Shape Inference
  //
  // Note that signatures are defined to allow for forward-declaring
  // any structs used from ir.h
  OpSchema& TypeAndShapeInferenceFunction(InferenceFunction inferenceFunction);
  InferenceFunction GetTypeAndShapeInferenceFunction() const {
    return tensor_inference_function_ ? tensor_inference_function_ : dummyInferenceFunction;
  }

  OpSchema& PartialDataPropagationFunction(DataPropagationFunction dataProgationFunction);
  DataPropagationFunction GetDataPropagationFunction() const {
    return data_propagation_function_ ? data_propagation_function_ : dummyDataPropagationFunction;
  }

  // Set the support level for the op schema.
  OpSchema& SetSupportLevel(SupportType supportType);

  // Functions to do documentation for the operator schema.
  // This may be disabled to save memory.
  OpSchema& SetDoc(const char* doc) {
#ifndef __ONNX_NO_DOC_STRINGS
    SetDoc(std::string(doc));
#else
    ONNX_UNUSED_PARAMETER(doc);
#endif

    return *this;
  }

  OpSchema& SetDoc(const std::string& doc) {
#ifndef __ONNX_NO_DOC_STRINGS
    doc_ = doc;
#else
    ONNX_UNUSED_PARAMETER(doc);
#endif
    return *this;
  }

  // Functions to specify name for the operator schema.
  OpSchema& SetName(const char* name);
  OpSchema& SetName(std::string name);

  // Functions to specify code location for the operator schema.
  OpSchema& SetLocation(const char* file, int line);
  OpSchema& SetLocation(std::string file, int line);

  // Functions to specify domain for the operator schema.
  // Default domain value (ONNX_DOMAIN) means it's ONNX domain.
  OpSchema& SetDomain(const char* domain);
  OpSchema& SetDomain(std::string domain);

  struct Attribute final {
    Attribute(std::string name_, std::string description_, AttributeProto::AttributeType type_, bool required_)
        : name(std::move(name_)),
          description(std::move(description_)),
          type(type_),
          required(required_),
          default_value() {}

    Attribute(std::string name_, std::string description_, AttributeProto default_value_)
        : name(std::move(name_)),
          description(std::move(description_)),
          type(default_value_.type()),
          required(false),
          default_value(std::move(default_value_)) {}

    const std::string name;
    const std::string description;
    AttributeProto::AttributeType type;
    bool required;
    AttributeProto default_value;
  };

  OpSchema& Attr(Attribute attr);

// Register "optional" attribute with default value.
#define ATTR_SETTER_WITH_DEFAULT_VALUE(TypeName)                                                                    \
  OpSchema& Attr(                                                                                                   \
      std::string name, std::string description, AttributeProto::AttributeType type, const TypeName& defaultValue); \
  /* non-STL wrapper to reduce binary size */                                                                       \
  OpSchema& Attr(                                                                                                   \
      const char* name, const char* description, AttributeProto::AttributeType type, const TypeName& defaultValue); \
  OpSchema& Attr(                                                                                                   \
      std::string name,                                                                                             \
      std::string description,                                                                                      \
      AttributeProto::AttributeType type,                                                                           \
      const std::vector<TypeName>& defaultValue);

  ATTR_SETTER_WITH_DEFAULT_VALUE(int64_t)
  ATTR_SETTER_WITH_DEFAULT_VALUE(float)
  ATTR_SETTER_WITH_DEFAULT_VALUE(std::string)
  ATTR_SETTER_WITH_DEFAULT_VALUE(TensorProto)
  ATTR_SETTER_WITH_DEFAULT_VALUE(GraphProto)
  ATTR_SETTER_WITH_DEFAULT_VALUE(TypeProto)

  // Register "required" attribute without default value.
  OpSchema& Attr(std::string name, std::string description, AttributeProto::AttributeType type, bool required = true);

  // Non-STL wrapper to reduce binary size
  OpSchema& Attr(const char* name, const char* description, AttributeProto::AttributeType type, bool required = true);

  OpSchema& AllowUncheckedAttributes();

  // Type constraint.
  struct TypeConstraintParam final {
    TypeConstraintParam(
        std::string type_param_str_,
        std::vector<std::string> allowed_type_strs_,
        std::string description_)
        : type_param_str(std::move(type_param_str_)),
          allowed_type_strs(std::move(allowed_type_strs_)),
          description(std::move(description_)) {}

    // Type parameter string, for example, "T", "T1", etc.
    std::string type_param_str;
    // Allowed type strings for <*this> type parameter, for example,
    // "tensor(float)".
    std::vector<std::string> allowed_type_strs;
    // Type parameter description.
    std::string description;
  };

  // Grammar for type strings used in Input(), Output().
  // <type> ::= <data_type> |
  //            tensor(<data_type>) |
  //            seq(<type>) |
  //            map(<data_type>, <type>) |
  //            <type_parameter>
  // <data_type> :: = float | int32 | string | bool | uint8
  //                | int8 | uint16 | int16 | int64 | float16 | double
  // <type_parameter> ::= any type parameter string, say "T".
  //
  // NOTE: 1) <type_parameter> will always be together with a type constraints
  // specification.
  //       2) <type> ::= <data_type> means the data is scalar (zero dimension).
  //
  // Example:
  // ONNX_OPERATOR_SET_SCHEMA(Sum, 1, OpSchema()
  // .Input(0, "input_a", "the first input", "T")
  // .Input(1, "input_b", "the second input", "T")
  // .Output(0, "sum", "the sum of two numbers", "T")
  // .TypeConstraint("T", {"float", "double", "int32"}, "allowed data types for
  // sum."))
  //
  // Optional = true means that the input might have empty input value
  // (represented as "") in the graph even though the later inputs have values.
  // It's useful for complex situation when there are several independent
  // optional inputs.
  OpSchema& Input(int n, FormalParameter formal_parameter);

  OpSchema& Input(
      int n,
      std::string name,
      const std::string& description,
      std::string type_str,
      FormalParameterOption param_option = Single,
      bool is_homogeneous = true,
      int min_arity = 1,
      DifferentiationCategory differentiation_category = Unknown);

  // Non-STL wrapper to reduce binary size
  OpSchema& Input(
      int n,
      const char* name,
      const char* description,
      const char* type_str,
      FormalParameterOption param_option = Single,
      bool is_homogeneous = true,
      int min_arity = 1,
      DifferentiationCategory differentiation_category = Unknown);

  OpSchema& Output(int n, FormalParameter formal_parameter);

  OpSchema& Output(
      int n,
      std::string name,
      const std::string& description,
      std::string type_str,
      FormalParameterOption param_option = Single,
      bool is_homogeneous = true,
      int min_arity = 1,
      DifferentiationCategory differentiation_category = Unknown);

  // Non-STL wrapper to reduce binary size
  OpSchema& Output(
      int n,
      const char* name,
      const char* description,
      const char* type_str,
      FormalParameterOption param_option = Single,
      bool is_homogeneous = true,
      int min_arity = 1,
      DifferentiationCategory differentiation_category = Unknown);

  OpSchema& TypeConstraint(std::string type_str, std::vector<std::string> constraints, std::string description);

  // Non-STL wrapper to reduce binary size
  OpSchema&
  TypeConstraint(const char* type_str, std::initializer_list<const char*> constraints, const char* description);

  // Convenience members for types

  // All high-precision numeric types.
  static const std::vector<std::string>& numeric_types_for_math_reduction_ir9() {
    static const std::vector<std::string> numeric_types_for_math_reduction_ir9 = {
        "tensor(uint32)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(int64)",
        "tensor(float16)",
        "tensor(float)",
        "tensor(double)",
        "tensor(bfloat16)",
        "tensor(float8e4m3fn)",
        "tensor(float8e4m3fnuz)",
        "tensor(float8e5m2)",
        "tensor(float8e5m2fnuz)"};
    return numeric_types_for_math_reduction_ir9;
  }

  static const std::vector<std::string>& numeric_types_for_math_reduction_ir4() {
    static const std::vector<std::string> numeric_types_for_math_reduction_ir4 = {
        "tensor(uint32)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(int64)",
        "tensor(float16)",
        "tensor(float)",
        "tensor(double)",
        "tensor(bfloat16)"};
    return numeric_types_for_math_reduction_ir4;
  }

  // Deprecated function, use numeric_types_for_math_reduction_ir4 instead. It will be removed in onnx==1.15.0.
  static const std::vector<std::string>& numeric_types_for_math_reduction_with_bfloat() {
    return numeric_types_for_math_reduction_ir4();
  }

  static const std::vector<std::string>& numeric_types_for_math_reduction() {
    static const std::vector<std::string> numeric_types_for_math_reduction = {
        "tensor(uint32)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(int64)",
        "tensor(float16)",
        "tensor(float)",
        "tensor(double)"};
    return numeric_types_for_math_reduction;
  }

  static const std::vector<std::string>& all_numeric_types_ir9() {
    static const std::vector<std::string> all_numeric_types_ir9 = {
        "tensor(uint8)",
        "tensor(uint16)",
        "tensor(uint32)",
        "tensor(uint64)",
        "tensor(int8)",
        "tensor(int16)",
        "tensor(int32)",
        "tensor(int64)",
        "tensor(float16)",
        "tensor(float)",
        "tensor(double)",
        "tensor(bfloat16)",
        "tensor(float8e4m3fn)",
        "tensor(float8e4m3fnuz)",
        "tensor(float8e5m2)",
        "tensor(float8e5m2fnuz)"};
    return all_numeric_types_ir9;
  }

  static const std::vector<std::string>& all_numeric_types_ir4() {
    static const std::vector<std::string> all_numeric_types_ir4 = {
        "tensor(uint8)",
        "tensor(uint16)",
        "tensor(uint32)",
        "tensor(uint64)",
        "tensor(int8)",
        "tensor(int16)",
        "tensor(int32)",
        "tensor(int64)",
        "tensor(float16)",
        "tensor(float)",
        "tensor(double)",
        "tensor(bfloat16)"};
    return all_numeric_types_ir4;
  }

  // Deprecated function, use all_numeric_types_ir4 instead. It will be removed in onnx==1.15.0.
  static const std::vector<std::string>& all_numeric_types_with_bfloat() {
    return all_numeric_types_ir4();
  }

  static const std::vector<std::string>& all_numeric_types() {
    static const std::vector<std::string> all_numeric_types = {
        "tensor(uint8)",
        "tensor(uint16)",
        "tensor(uint32)",
        "tensor(uint64)",
        "tensor(int8)",
        "tensor(int16)",
        "tensor(int32)",
        "tensor(int64)",
        "tensor(float16)",
        "tensor(float)",
        "tensor(double)"};
    return all_numeric_types;
  }

  static const std::vector<std::string>& all_numeric_sequence_types() {
    static const std::vector<std::string> all_numeric_sequence_types = {
        "seq(tensor(uint8))",
        "seq(tensor(uint16))",
        "seq(tensor(uint32))",
        "seq(tensor(uint64))",
        "seq(tensor(int8))",
        "seq(tensor(int16))",
        "seq(tensor(int32))",
        "seq(tensor(int64))",
        "seq(tensor(float16))",
        "seq(tensor(float))",
        "seq(tensor(double))"};
    return all_numeric_sequence_types;
  }

  static const std::vector<std::string>& all_tensor_types() {
    static const std::vector<std::string> all_tensor_types = {
        "tensor(uint8)",
        "tensor(uint16)",
        "tensor(uint32)",
        "tensor(uint64)",
        "tensor(int8)",
        "tensor(int16)",
        "tensor(int32)",
        "tensor(int64)",
        "tensor(float16)",
        "tensor(float)",
        "tensor(double)",
        "tensor(string)",
        "tensor(bool)",
        "tensor(complex64)",
        "tensor(complex128)"};
    return all_tensor_types;
  }

  static const std::vector<std::string>& all_tensor_types_ir4() {
    static const std::vector<std::string> all_tensor_types_ir4 = {
        "tensor(uint8)",
        "tensor(uint16)",
        "tensor(uint32)",
        "tensor(uint64)",
        "tensor(int8)",
        "tensor(int16)",
        "tensor(int32)",
        "tensor(int64)",
        "tensor(bfloat16)",
        "tensor(float16)",
        "tensor(float)",
        "tensor(double)",
        "tensor(string)",
        "tensor(bool)",
        "tensor(complex64)",
        "tensor(complex128)"};
    return all_tensor_types_ir4;
  }

  // Deprecated function, use all_tensor_types_ir4 instead. It will be removed in onnx==1.15.0.
  static const std::vector<std::string>& all_tensor_types_with_bfloat() {
    return all_tensor_types_ir4();
  }

  static const std::vector<std::string>& all_tensor_types_ir9() {
    static const std::vector<std::string> all_tensor_types_ir9 = {
        "tensor(uint8)",        "tensor(uint16)",         "tensor(uint32)",     "tensor(uint64)",
        "tensor(int8)",         "tensor(int16)",          "tensor(int32)",      "tensor(int64)",
        "tensor(bfloat16)",     "tensor(float16)",        "tensor(float)",      "tensor(double)",
        "tensor(string)",       "tensor(bool)",           "tensor(complex64)",  "tensor(complex128)",
        "tensor(float8e4m3fn)", "tensor(float8e4m3fnuz)", "tensor(float8e5m2)", "tensor(float8e5m2fnuz)"};
    return all_tensor_types_ir9;
  }

  static const std::vector<std::string>& all_tensor_sequence_types() {
    static const std::vector<std::string> all_tensor_sequence_types = {
        "seq(tensor(uint8))",
        "seq(tensor(uint16))",
        "seq(tensor(uint32))",
        "seq(tensor(uint64))",
        "seq(tensor(int8))",
        "seq(tensor(int16))",
        "seq(tensor(int32))",
        "seq(tensor(int64))",
        "seq(tensor(float16))",
        "seq(tensor(float))",
        "seq(tensor(double))",
        "seq(tensor(string))",
        "seq(tensor(bool))",
        "seq(tensor(complex64))",
        "seq(tensor(complex128))"};
    return all_tensor_sequence_types;
  }

  static const std::vector<std::string>& all_tensor_sequence_types_ir4() {
    static const std::vector<std::string> all_tensor_sequence_types_ir4 = {
        "seq(tensor(uint8))",
        "seq(tensor(uint16))",
        "seq(tensor(uint32))",
        "seq(tensor(uint64))",
        "seq(tensor(int8))",
        "seq(tensor(int16))",
        "seq(tensor(int32))",
        "seq(tensor(int64))",
        "seq(tensor(bfloat16))",
        "seq(tensor(float16))",
        "seq(tensor(float))",
        "seq(tensor(double))",
        "seq(tensor(string))",
        "seq(tensor(bool))",
        "seq(tensor(complex64))",
        "seq(tensor(complex128))"};
    return all_tensor_sequence_types_ir4;
  }

  // Deprecated function, use all_tensor_sequence_types_ir4 instead. It will be removed in onnx==1.15.0.
  static const std::vector<std::string>& all_tensor_sequence_types_with_bfloat() {
    return all_tensor_sequence_types_ir4();
  }

  static const std::vector<std::string>& all_tensor_sequence_types_ir9() {
    static const std::vector<std::string> all_tensor_sequence_types_ir4 = {
        "seq(tensor(uint8))",      "seq(tensor(uint16))",        "seq(tensor(uint32))",
        "seq(tensor(uint64))",     "seq(tensor(int8))",          "seq(tensor(int16))",
        "seq(tensor(int32))",      "seq(tensor(int64))",         "seq(tensor(bfloat16))",
        "seq(tensor(float16))",    "seq(tensor(float))",         "seq(tensor(double))",
        "seq(tensor(string))",     "seq(tensor(bool))",          "seq(tensor(complex64))",
        "seq(tensor(complex128))", "seq(tensor(float8e4m3fn))",  "seq(tensor(float8e4m3fnuz))",
        "seq(tensor(float8e5m2))", "seq(tensor(float8e5m2fnuz))"};
    return all_tensor_sequence_types_ir4;
  }

  static const std::vector<std::string>& all_optional_types() {
    static const std::vector<std::string> all_optional_types = {
        "optional(seq(tensor(uint8)))",  "optional(seq(tensor(uint16)))",    "optional(seq(tensor(uint32)))",
        "optional(seq(tensor(uint64)))", "optional(seq(tensor(int8)))",      "optional(seq(tensor(int16)))",
        "optional(seq(tensor(int32)))",  "optional(seq(tensor(int64)))",     "optional(seq(tensor(float16)))",
        "optional(seq(tensor(float)))",  "optional(seq(tensor(double)))",    "optional(seq(tensor(string)))",
        "optional(seq(tensor(bool)))",   "optional(seq(tensor(complex64)))", "optional(seq(tensor(complex128)))",
        "optional(tensor(uint8))",       "optional(tensor(uint16))",         "optional(tensor(uint32))",
        "optional(tensor(uint64))",      "optional(tensor(int8))",           "optional(tensor(int16))",
        "optional(tensor(int32))",       "optional(tensor(int64))",          "optional(tensor(float16))",
        "optional(tensor(float))",       "optional(tensor(double))",         "optional(tensor(string))",
        "optional(tensor(bool))",        "optional(tensor(complex64))",      "optional(tensor(complex128))"};
    return all_optional_types;
  }

  static const std::vector<std::string>& all_optional_types_ir4() {
    static const std::vector<std::string> all_optional_types = {
        "optional(seq(tensor(uint8)))",      "optional(seq(tensor(uint16)))", "optional(seq(tensor(uint32)))",
        "optional(seq(tensor(uint64)))",     "optional(seq(tensor(int8)))",   "optional(seq(tensor(int16)))",
        "optional(seq(tensor(int32)))",      "optional(seq(tensor(int64)))",  "optional(seq(tensor(bfloat16)))",
        "optional(seq(tensor(float16)))",    "optional(seq(tensor(float)))",  "optional(seq(tensor(double)))",
        "optional(seq(tensor(string)))",     "optional(seq(tensor(bool)))",   "optional(seq(tensor(complex64)))",
        "optional(seq(tensor(complex128)))", "optional(tensor(uint8))",       "optional(tensor(uint16))",
        "optional(tensor(uint32))",          "optional(tensor(uint64))",      "optional(tensor(int8))",
        "optional(tensor(int16))",           "optional(tensor(int32))",       "optional(tensor(int64))",
        "optional(tensor(bfloat16))",        "optional(tensor(float16))",     "optional(tensor(float))",
        "optional(tensor(double))",          "optional(tensor(string))",      "optional(tensor(bool))",
        "optional(tensor(complex64))",       "optional(tensor(complex128))"};
    return all_optional_types;
  }

  // Deprecated function, use all_optional_types_ir4 instead. It will be removed in onnx==1.15.0.
  static const std::vector<std::string>& all_optional_types_with_bfloat() {
    return all_optional_types_ir4();
  }
  static const std::vector<std::string>& all_optional_types_ir9() {
    static const std::vector<std::string> all_optional_types = {
        "optional(seq(tensor(uint8)))",      "optional(seq(tensor(uint16)))", "optional(seq(tensor(uint32)))",
        "optional(seq(tensor(uint64)))",     "optional(seq(tensor(int8)))",   "optional(seq(tensor(int16)))",
        "optional(seq(tensor(int32)))",      "optional(seq(tensor(int64)))",  "optional(seq(tensor(bfloat16)))",
        "optional(seq(tensor(float16)))",    "optional(seq(tensor(float)))",  "optional(seq(tensor(double)))",
        "optional(seq(tensor(string)))",     "optional(seq(tensor(bool)))",   "optional(seq(tensor(complex64)))",
        "optional(seq(tensor(complex128)))", "optional(tensor(uint8))",       "optional(tensor(uint16))",
        "optional(tensor(uint32))",          "optional(tensor(uint64))",      "optional(tensor(int8))",
        "optional(tensor(int16))",           "optional(tensor(int32))",       "optional(tensor(int64))",
        "optional(tensor(bfloat16))",        "optional(tensor(float16))",     "optional(tensor(float))",
        "optional(tensor(double))",          "optional(tensor(string))",      "optional(tensor(bool))",
        "optional(tensor(complex64))",       "optional(tensor(complex128))",  "optional(tensor(float8e4m3fn))",
        "optional(tensor(float8e4m3fnuz))",  "optional(tensor(float8e5m2))",  "optional(tensor(float8e5m2fnuz))"};
    return all_optional_types;
  }

  // Calls the passed function with `this` as an argument. Useful for
  // adding docs for temlated/macro ops.
  OpSchema& FillUsing(const std::function<void(OpSchema&)>& populator);

  friend std::ostream& operator<<(std::ostream& out, const OpSchema& schema);

  const std::string& domain() const {
    return domain_;
  }

  const std::map<std::string, Attribute>& attributes() const {
    return attributes_;
  }

  // Get input formal parameters.
  const std::vector<FormalParameter>& inputs() const {
    return inputs_;
  }

  // Get output formal parameters.
  const std::vector<FormalParameter>& outputs() const {
    return outputs_;
  }

  const std::vector<TypeConstraintParam>& typeConstraintParams() const {
    return type_constraint_params_;
  }

  const TypeConstraintMap& typeConstraintMap() const {
    return type_constraints_;
  }

  const std::string& Name() const {
    return name_;
  }

  OperatorSetVersion SinceVersion() const {
    return since_version_;
  }

  int since_version() const {
    return since_version_;
  }

  bool deprecated() const {
    return deprecated_;
  }

  int min_input() const {
    return min_input_;
  }
  int max_input() const {
    return max_input_;
  }
  int min_output() const {
    return min_output_;
  }
  int max_output() const {
    return max_output_;
  }

  bool has_type_and_shape_inference_function() const {
    return tensor_inference_function_ ? true : false;
  }

  bool has_data_propagation_function() const {
    return data_propagation_function_ ? true : false;
  }

  std::vector<int> function_opset_versions() const {
    std::vector<int> opset_versions;
    std::map<int, std::shared_ptr<FunctionProto>>::const_iterator it = opset_version_to_function_body_.cbegin();
    for (; it != opset_version_to_function_body_.cend(); ++it) {
      opset_versions.push_back(it->first);
    }
    return opset_versions;
  }

  bool HasFunction() const {
    return !opset_version_to_function_body_.empty();
  }

  OpSchema& FunctionBody(const std::vector<NodeProto>& func_nodes, int opset_version = kUninitializedSinceVersion);

  OpSchema& FunctionBody(
      const std::vector<NodeProto>& func_nodes,
      const std::vector<OperatorSetIdProto>& opsets,
      int opset_version = kUninitializedSinceVersion);

  OpSchema& FunctionBody(const char* func_body, int opset_version = kUninitializedSinceVersion);

  // since_version_ of an OpSchema tells the last opset version when an op is defined.
  // When the op's definition is changed, a new OpSchema (of the same op_type) is created
  // with a newer since_version_, reflecting the opset version at the time of change.
  // For a function op, operators used to define its function body may change
  // while there is no change to the function op definition itself.
  // When this happens, mutiple function bodies are provided, each for a specific opset version.
  //
  // Take LogSoftmax for example. Its latest opset version is 13.
  // In LogSoftmax's function body, ReduceMax (with since_version_ 1, 11, 12, 18) is used.
  // When a model containing LogSoftmax with opset_import version within 13 to 17 is loaded, function body
  // with opset_version 13 is used for inlining.
  // When the same model but opset_import version 18 is loaded, function body
  // with opset_version 18 is used for inlining.
  // Clearly function body for opset_import version 13 will not work
  // in a model with opset_import version 18 because the function body make worng use of ReduceMax(18).
  // Inside GetFunction we ensure that ops being used to construct a function body do not endure such
  // issue.
  const FunctionProto* GetFunction(
      int requested_opset_version = OpSchema::kUninitializedSinceVersion,
      bool validate = false) const;

  std::vector<int> context_dependent_function_opset_versions() const {
    std::vector<int> opset_versions;
    std::map<int, ContextDependentFunctionBodyBuilder>::const_iterator it = opset_version_to_function_builder_.cbegin();
    for (; it != opset_version_to_function_builder_.cend(); ++it) {
      opset_versions.push_back(it->first);
    }
    return opset_versions;
  }

  bool HasContextDependentFunction() const {
    return !opset_version_to_function_builder_.empty();
  }

  bool HasContextDependentFunctionWithOpsetVersion(int opset_version) const {
    return opset_version_to_function_builder_.find(opset_version) != opset_version_to_function_builder_.end();
  }

  OpSchema& SetContextDependentFunctionBodyBuilder(
      ContextDependentFunctionBodyBuilder,
      int opset_version = kUninitializedSinceVersion);

  bool BuildContextDependentFunction(
      const FunctionBodyBuildContext& ctx,
      FunctionProto& function_proto,
      int requested_opset_version = OpSchema::kUninitializedSinceVersion) const;

  // Verifies that the schema is valid and all specifications are compatible.
  // It will also parse all type strings specified for inputs/outputs into valid
  // TypeProto and create global unique string pointer as the DataType for
  // efficiency.
  void Finalize();

  // Build function with information stored in opschema
  void BuildFunction(FunctionProto& function_body) const;

 private:
  void ParseAndSetTypes(
      /*out*/ std::vector<OpSchema::FormalParameter>* formalParameters);
  bool ValidateReferencedOpsInFuncton(
      const FunctionProto* function,
      int requested_opset_version,
      int function_since_version,
      std::set<std::string>* updated_ops = nullptr) const;
  void UpdateFunctionProtoOpsetImportVersion(FunctionProto& function_proto, int opset_version) const;

  std::string name_;
  std::string file_;
  std::string doc_;
  // Default domain value ("") means it's ONNX domain.
  std::string domain_ = ONNX_DOMAIN;
  std::map<std::string, Attribute> attributes_{};
  bool allows_unchecked_attributes_ = false;
  std::vector<FormalParameter> inputs_;
  std::vector<FormalParameter> outputs_;
  std::vector<TypeConstraintParam> type_constraint_params_;
  TypeConstraintMap type_constraints_;
  int line_ = 0;
  SupportType support_;
  int min_input_ = 0;
  int max_input_ = 0;
  int min_output_ = 0;
  int max_output_ = 0;
  // The default is a little goofy, since it is never what you want
  OperatorSetVersion since_version_ = kUninitializedSinceVersion;
  bool deprecated_{};
  std::function<bool(int)> num_inputs_allowed_ = [](int) { return true; };
  std::function<bool(int)> num_outputs_allowed_ = [](int) { return true; };
  InferenceFunction tensor_inference_function_;
  DataPropagationFunction data_propagation_function_;

  std::map<int, std::shared_ptr<FunctionProto>> opset_version_to_function_body_;
  std::map<int, ContextDependentFunctionBodyBuilder> opset_version_to_function_builder_;
};

// Map type to store operator schemas. The format is,
// <OpName, <Domain, <OperatorSetVersion, OpSchema>>>.
using OpName_Domain_Version_Schema_Map =
    std::unordered_map<std::string, std::unordered_map<std::string, std::map<OperatorSetVersion, OpSchema>>>;

class ISchemaRegistry {
 public:
  virtual ~ISchemaRegistry() = default;

  virtual const OpSchema*
  GetSchema(const std::string& key, const int maxInclusiveVersion, const std::string& domain = ONNX_DOMAIN) const = 0;
};

/**
 * @brief A registry to hold all the operator schemas.
 */
class OpSchemaRegistry final : public ISchemaRegistry {
 public:
  // A singleton class to store domain to min/max op_set version map, as well as
  // domain to last-release op_set version map.
  class DomainToVersionRange final {
   public:
    DomainToVersionRange() {
      // Increase the highest version when you make BC-breaking changes to the
      // operator schema on specific domain. Update the lowest version when it's
      // determined to remove too old version history.
      map_[ONNX_DOMAIN] = std::make_pair(1, 19);
      map_[AI_ONNX_ML_DOMAIN] = std::make_pair(1, 3);
      map_[AI_ONNX_TRAINING_DOMAIN] = std::make_pair(1, 1);
      // ONNX's preview domain contains operators subject to change, so
      // versining is not meaningful and that domain should have only one
      // version.
      map_[AI_ONNX_PREVIEW_TRAINING_DOMAIN] = std::make_pair(1, 1);
      // Version corresponding last release of ONNX. Update this to match with
      // the max version above in a *release* version of ONNX. But in other
      // versions, the max version may be ahead of the last-release-version.
      last_release_version_map_[ONNX_DOMAIN] = 19;
      last_release_version_map_[AI_ONNX_ML_DOMAIN] = 3;
      last_release_version_map_[AI_ONNX_TRAINING_DOMAIN] = 1;
      last_release_version_map_[AI_ONNX_PREVIEW_TRAINING_DOMAIN] = 1;
    }

    const std::unordered_map<std::string, std::pair<int, int>>& Map() const {
      return map_;
    }

    const std::unordered_map<std::string, int>& LastReleaseVersionMap() const {
      return last_release_version_map_;
    }

    // Add customized domain to min/max version.
    // Onnx partners are able to use onnx operator schema api to
    // register customized op in their own domain.
    // Can optionally specify last_release_version (to make it similar to
    // standard ONNX domains as above). Custom-domains are free to interpret
    // this as appropriate (that is, as relative to releases of custom-domain
    // as opposed to ONNX releases).
    void
    AddDomainToVersion(const std::string& domain, int min_version, int max_version, int last_release_version = -1) {
      std::lock_guard<std::mutex> lock(mutex_);
      assert(map_.end() == map_.find(domain));
      map_[domain] = std::make_pair(min_version, max_version);
      // If a last-release-version is not explicitly specified, use max as
      // last-release-version.
      if (last_release_version == -1)
        last_release_version = max_version;
      assert(last_release_version_map_.end() == last_release_version_map_.find(domain));
      last_release_version_map_[domain] = last_release_version;
    }

    static DomainToVersionRange& Instance();

   private:
    // Key: domain. Value: <lowest version, highest version> pair.
    std::unordered_map<std::string, std::pair<int, int>> map_;

    // Key: domain. Value: most recent release opset version. Note that
    // the highest opset version may be ahead of the most recent release's opset
    // version.
    std::unordered_map<std::string, int> last_release_version_map_;

    std::mutex mutex_;
  };

  class OpSchemaRegisterOnce final {
   public:
    OpSchemaRegisterOnce(OpSchema& op_schema, int opset_version_to_load = 0) {
      ONNX_TRY {
        op_schema.Finalize();
        auto& m = GetMapWithoutEnsuringRegistration();
        auto& op_name = op_schema.Name();
        auto& op_domain = op_schema.domain();
        auto ver = op_schema.SinceVersion();
        if (OpSchema::kUninitializedSinceVersion == ver) {
          op_schema.SinceVersion(1);
          ver = op_schema.SinceVersion();
        }
        // Stops because the opset_version is higher than opset_version_to_load
        if (opset_version_to_load != 0 && ver > opset_version_to_load) {
          return;
        }
        if (m[op_name][op_domain].count(ver)) {
          const auto& schema = m[op_name][op_domain][ver];
          std::stringstream err;
          err << "Trying to register schema with name " << op_name << " (domain: " << op_domain << " version: " << ver
              << ") from file " << op_schema.file() << " line " << op_schema.line()
              << ", but it is already registered from file " << schema.file() << " line " << schema.line() << std::endl;
          fail_schema(err.str());
        }
        // Return early if schema for the targeted opset version has already been loaded
        if (opset_version_to_load != 0 && !m[op_name][op_domain].empty()) {
          return;
        }
        auto ver_range_map = DomainToVersionRange::Instance().Map();
        auto ver_range_it = ver_range_map.find(op_domain);
        if (ver_range_it == ver_range_map.end()) {
          std::stringstream err;
          err << "Trying to register schema with name " << op_name << " (domain: " << op_domain << " version: " << ver
              << ") from file " << op_schema.file() << " line " << op_schema.line() << ", but its domain is not"
              << " known by the checker." << std::endl;

          fail_schema(err.str());
        }
        auto lower_bound_incl = ver_range_it->second.first;
        auto upper_bound_incl = ver_range_it->second.second;
        if (!(lower_bound_incl <= ver && upper_bound_incl >= ver)) {
          std::stringstream err;
          err << "Trying to register schema with name " << op_name << " (domain: " << op_domain << " version: " << ver
              << ") from file " << op_schema.file() << " line " << op_schema.line() << ", but its version is not "
              << "in the inclusive range [" << lower_bound_incl << ", " << upper_bound_incl
              << "] (usually, this means you "
              << "bumped the operator version but "
              << "forgot to update the version range in DomainToVersionRange "
              << "in onnx/defs/schema.h)." << std::endl;
          fail_schema(err.str());
        }

        m[op_name][op_domain].insert(std::pair<int, OpSchema&&>(ver, std::move(op_schema)));
      }
      ONNX_CATCH(const std::exception& e) {
        ONNX_HANDLE_EXCEPTION([&]() { std::cerr << "Schema error: " << e.what() << std::endl; });
      }
    }
  };

  // Return the latest schema for an operator in specified domain.
  // Domain with default value ONNX_DOMAIN means ONNX.
  static const OpSchema* Schema(const std::string& key, const std::string& domain = ONNX_DOMAIN) {
    auto& m = map();
    if (m.count(key) && m[key].count(domain)) {
      return &m[key][domain].rbegin()->second;
    } else {
      return nullptr;
    }
  }

  // Return the schema with biggest version, which is not greater than specified
  // <maxInclusiveVersion> in specified domain. Domain with default value
  // ONNX_DOMAIN means ONNX.
  static const OpSchema*
  Schema(const std::string& key, const int maxInclusiveVersion, const std::string& domain = ONNX_DOMAIN) {
    auto& m = map();
    if (m.count(key) && m[key].count(domain)) {
      auto pos = m[key][domain].lower_bound(maxInclusiveVersion);
      if (m[key][domain].begin() == pos && pos->first > maxInclusiveVersion) {
        // All versions are greater than specified version.
        return nullptr;
      }
      if (m[key][domain].end() == pos || pos->first > maxInclusiveVersion) {
        // All versions are less than specified version, or,
        // The <pos> version is greater than specified version.
        pos--;
      }

      // Schema with exact version as specified one exists.
      return &(pos->second);
    } else {
      return nullptr;
    }
  }

  static OpSchemaRegistry* Instance();

  const OpSchema* GetSchema(
      const std::string& key,
      const int maxInclusiveVersion,
      const std::string& domain = ONNX_DOMAIN) const override {
    return Schema(key, maxInclusiveVersion, domain);
  }
  static void SetLoadedSchemaVersion(int target_version) {
    loaded_schema_version = target_version;
  }
  static int GetLoadedSchemaVersion() {
    return loaded_schema_version;
  }

 private:
  // OpSchemaRegistry should not need to be instantiated except statically
  // within this class
  OpSchemaRegistry() = default;

  /**
   * @brief Returns the underlying string to OpSchema map.
   *
   * You should not manually manipulate the map object returned. Instead, use
   * the macros defined such as ONNX_OPERATOR_SET_SCHEMA to register your
   * operator schema.
   *
   * We wrap it inside a function to avoid the static initialization order
   * fiasco.
   */
  static OpName_Domain_Version_Schema_Map& GetMapWithoutEnsuringRegistration();
  static OpName_Domain_Version_Schema_Map& map();
  static int loaded_schema_version;

 public:
  static const std::vector<OpSchema> get_all_schemas_with_history() {
    std::vector<OpSchema> r;
    for (auto& x : map()) {
      for (auto& y : x.second) {
        for (auto& z : y.second) {
          r.emplace_back(z.second);
        }
      }
    }
    return r;
  }

  static const std::vector<OpSchema> get_all_schemas() {
    std::vector<OpSchema> r;
    for (auto& x : map()) {
      for (auto& y : x.second) {
        auto& version2schema = y.second;
        r.emplace_back(version2schema.rbegin()->second);
      }
    }
    return r;
  }
};

void RegisterSchema(OpSchema schema, int opset_version_to_load = 0);

// Registers the latest opset schema before opset_version_to_load
// By default opset_version_to_load=0 means it will register all versions
template <class T>
void RegisterOpSetSchema(int opset_version_to_load = 0) {
  T::ForEachSchema([opset_version_to_load](OpSchema&& schema) { RegisterSchema(schema, opset_version_to_load); });
};

// Forward declaration for the non-specialized GetOpSchema method.  This
// enforces a consistent signature on functions that query individual schema,
// which are defined as specializations of this function.
template <typename T>
OpSchema GetOpSchema();

#define ONNX_OPERATOR_SET_SCHEMA(name, ver, impl) ONNX_OPERATOR_SET_SCHEMA_EX(name, Onnx, ONNX_DOMAIN, ver, true, impl)

#define ONNX_ML_OPERATOR_SET_SCHEMA(name, ver, impl) \
  ONNX_OPERATOR_SET_SCHEMA_EX(name, OnnxML, AI_ONNX_ML_DOMAIN, ver, true, impl)

#define ONNX_TRAINING_OPERATOR_SET_SCHEMA(name, ver, impl) \
  ONNX_OPERATOR_SET_SCHEMA_EX(name, OnnxTraining, AI_ONNX_TRAINING_DOMAIN, ver, true, impl)

#define ONNX_PREVIEW_TRAINING_OPERATOR_SET_SCHEMA(name, ver, impl) \
  ONNX_OPERATOR_SET_SCHEMA_EX(name, OnnxPreview, AI_ONNX_PREVIEW_TRAINING_DOMAIN, ver, true, impl)

// Defines specialization of GetOpSchema for a class whose name is determined
// based on a convention using name, domain, and version.  Operator schema are
// normally included in operator sets and registered in OpSchemaRegistry::map().
// In this case, callers should set dbg_included_in_static_opset to true.  This
// assists with runtime validation in DEBUG builds ensuring the intended set
// of operator schema is registered.
#define ONNX_OPERATOR_SET_SCHEMA_EX(name, domain, domain_str, ver, dbg_included_in_static_opset, impl)  \
  class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(domain, ver, name);                                         \
  template <>                                                                                           \
  OpSchema GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(domain, ver, name)>() {                      \
    return impl.SetName(#name).SetDomain(domain_str).SinceVersion(ver).SetLocation(__FILE__, __LINE__); \
  }                                                                                                     \
  size_t dbg_count_check_##name##_##domain##_ver##ver =                                                 \
      (dbg_included_in_static_opset) ? ONNX_DBG_INCREMENT_COUNT_IN_OPSETS() : 0;
#ifdef NDEBUG
#define ONNX_DBG_INCREMENT_COUNT_IN_OPSETS() 0
#else
#define ONNX_DBG_INCREMENT_COUNT_IN_OPSETS() DbgOperatorSetTracker::Instance().IncrementCount()
#define ONNX_DBG_GET_COUNT_IN_OPSETS() DbgOperatorSetTracker::Instance().GetCount()

class DbgOperatorSetTracker {
 public:
  static DbgOperatorSetTracker& Instance();

  size_t IncrementCount() {
    return ++count_;
  }

  size_t GetCount() const {
    return count_;
  }

 private:
  size_t count_ = 0;
};
#endif

// Naming convention for operator schema classes
#define ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(domain, ver, name) name##_##domain##_ver##ver

// Naming convention for preview operator schema classes
#define ONNX_PREVIEW_OPERATOR_SET_SCHEMA_CLASS_NAME(ver, name) \
  ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(OnnxPreview, ver, name)

// Helper function
size_t ReplaceAll(std::string& s, const char* from, const char* to);

#ifdef __GNUC__
#define ONNX_UNUSED __attribute__((__unused__))
#else
#define ONNX_UNUSED
#endif

// Legacy macros to register schema at static initialization
#define ONNX_OPERATOR_SCHEMA(name) ONNX_OPERATOR_SCHEMA_UNIQ_HELPER(__COUNTER__, name)
#define ONNX_OPERATOR_SCHEMA_UNIQ_HELPER(Counter, name) ONNX_OPERATOR_SCHEMA_UNIQ(Counter, name)
#define ONNX_OPERATOR_SCHEMA_UNIQ(Counter, name)                                                                      \
  static ONNX_NAMESPACE::OpSchemaRegistry::OpSchemaRegisterOnce(op_schema_register_once##name##Counter) ONNX_UNUSED = \
      OpSchema(#name, __FILE__, __LINE__)

// Helper function
size_t ReplaceAll(std::string& s, const char* from, const char* to);

inline std::string GenerateOptionalArgumentsDoc() {
  return "This operator has **optional** inputs/outputs. "
         "See [the doc](IR.md) for more details about the representation of "
         "optional arguments. An empty string may be used in the place of "
         "an actual argument's name to indicate a missing argument. "
         "Trailing optional arguments (those not followed by an argument "
         "that is present) may also be simply omitted.\n";
}

inline std::string GenerateBroadcastingDocMul() {
  return "This operator supports **multidirectional (i.e., Numpy-style) broadcasting**;"
         " for more details please check [the doc](Broadcasting.md).";
}

inline std::string GenerateBroadcastingDocUni(const char* from, const char* to) {
  std::string ret = "This operator supports **unidirectional broadcasting** (";
  ret = ret + from + " should be unidirectional broadcastable to " + to +
      ");"
      " for more details please check [the doc](Broadcasting.md).";
  return ret;
}

/*
 * Macros for setting operator documentation
 * Use this macro for simple SetDoc() calls that generate documentation
 * directly. This is the macro to use in almost all cases.
 * Sample usage guidelines:
 * const char* doc_str = "foo";
 * SetDoc(GET_OP_DOC_STR(doc_str))
 *
 * SetDoc(GET_OP_DOC_STR(
            std::string(BitShift_ver11_doc) + GenerateBroadcastingDocMul()))
 */
#ifndef __ONNX_NO_DOC_STRINGS
#define GET_OP_DOC_STR(doc_str) (doc_str)
#else
#define GET_OP_DOC_STR(doc_str) ("")
#endif

/*
 * Use this macro when the documentation needs to be populated in some
 * complicated way like string substitutions, etc before calling SetDoc.
 * Sample usage guidelines:
    std::string doc;
    POPULATE_OP_DOC_STR(
        doc = R"DOC(
Returns the tensor resulted from performing the `{name}` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting
support).

{broadcast_doc}
)DOC";
        ReplaceAll(doc, "{name}", name);
        ReplaceAll(
            doc, "{broadcast_doc}", GenerateBroadcastingDocMul().c_str()););
    schema.SetDoc(doc);
 *
 */
#ifndef __ONNX_NO_DOC_STRINGS
#define POPULATE_OP_DOC_STR(DocPopulatorCode) \
  do {                                        \
    DocPopulatorCode                          \
  } while (0)
#else
#define POPULATE_OP_DOC_STR(DocPopulatorCode)
#endif

} // namespace ONNX_NAMESPACE
