// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "onnx/defs/function.h"
#include "onnx/defs/schema.h"
#include "onnx/proto_utils.h"
#include "onnx/string_utils.h"

namespace ONNX_NAMESPACE {
namespace shape_inference {

using ModelLocalFunctionsMap = std::unordered_map<std::string, const FunctionProto*>;

// We reuse TensorShapeProto to propagate statically known (partial) information about
// the values of tensors. It is intended for tensors used to store shape information
// (the return values of ops like Shape and input values of ops like Reshape/Expand).

// A DataValueMap is used to store the statically known (partial) values of variables.
using DataValueMap = std::unordered_map<std::string, TensorShapeProto>;

class SymbolTableImpl : public SymbolTable {
 public:
  SymbolTableImpl() : index_(0) {}

  void addFromGraph(const GraphProto& g) override {
    AddExistingSymbolicDims(g.input());
    AddExistingSymbolicDims(g.output());
    AddExistingSymbolicDims(g.value_info());
  }
  // Creates a new unique symbol with the given prefix and adds it to the SymbolTable
  // Returns the newly created symbol
  std::string createNew(const std::string& symbol_prefix) override {
    std::string newSymbol;
    do {
      newSymbol = symbol_prefix + std::to_string(index_++);
    } while (existing_symbols.count(newSymbol) > 0);
    existing_symbols.insert(newSymbol);
    return newSymbol;
  }

 private:
  unsigned int index_;
  std::unordered_set<std::string> existing_symbols;

  // TypeProto_Tensor or TypeProto_SparseTensor
  template <typename TensorTypeProto>
  void AddExistingSymbolicDims(const TensorTypeProto& tensorType) {
    if (tensorType.has_shape()) {
      for (int i = 0; i < tensorType.shape().dim_size(); ++i) {
        if (tensorType.shape().dim(i).has_dim_param()) {
          existing_symbols.insert(tensorType.shape().dim(i).dim_param());
        }
      }
    }
  }

  void AddExistingSymbolicDims(const TypeProto& typeProto) {
    const auto val_case = typeProto.value_case();
    switch (val_case) {
      case TypeProto::kTensorType:
        AddExistingSymbolicDims(typeProto.tensor_type());
        break;
      case TypeProto::kSparseTensorType:
        AddExistingSymbolicDims(typeProto.sparse_tensor_type());
        break;
      case TypeProto::kSequenceType:
        AddExistingSymbolicDims(typeProto.sequence_type().elem_type());
        break;
      case TypeProto::kOptionalType:
        AddExistingSymbolicDims(typeProto.optional_type().elem_type());
        break;
      case TypeProto::kMapType:
        AddExistingSymbolicDims(typeProto.map_type().value_type());
        break;
      default:
        break;
    }
  }

  void AddExistingSymbolicDims(const google::protobuf::RepeatedPtrField<ValueInfoProto>& protos) {
    for (const auto& proto : protos) {
      AddExistingSymbolicDims(proto.type());
    }
  }
};

struct GraphInferenceContext {
  GraphInferenceContext(
      const std::unordered_map<std::string, TypeProto*>& outer_scope_value_types_by_name_in,
      const std::unordered_map<std::string, int> opset_imports_in,
      SymbolTable* symbol_table_in = nullptr,
      const ModelLocalFunctionsMap& model_local_functions_in = {},
      const ISchemaRegistry* schema_registry_in = OpSchemaRegistry::Instance(),
      DataValueMap* generated_shape_data_by_name_in = nullptr,
      const int ir_version_in = IR_VERSION)
      : outer_scope_value_types_by_name{&outer_scope_value_types_by_name_in},
        opset_imports{opset_imports_in},
        symbol_table{symbol_table_in},
        model_local_functions{model_local_functions_in},
        schema_registry{schema_registry_in},
        generated_shape_data_by_name{generated_shape_data_by_name_in},
        ir_version{ir_version_in} {}

  const std::unordered_map<std::string, TypeProto*>* outer_scope_value_types_by_name;
  const std::unordered_map<std::string, int> opset_imports;
  SymbolTable* symbol_table;
  const ModelLocalFunctionsMap& model_local_functions;
  const ISchemaRegistry* schema_registry;
  DataValueMap* generated_shape_data_by_name;
  const int ir_version;
};

class GraphInferencerImpl : public GraphInferencer {
 public:
  GraphInferencerImpl(GraphProto& g, GraphInferenceContext& context) : g_{&g}, context_{&context}, options_() {}
  GraphInferencerImpl(GraphProto& g, GraphInferenceContext& context, const ShapeInferenceOptions& options)
      : g_{&g}, context_{&context}, options_(options) {}

  std::vector<const TypeProto*> doInferencing(
      const std::vector<const TypeProto*>& inputTypes,
      const std::vector<const TensorProto*>& inputData) override;

 private:
  GraphProto* g_;
  GraphInferenceContext* context_;
  ShapeInferenceOptions options_;
};

struct InferenceContextImpl : public InferenceContext {
  InferenceContextImpl(
      NodeProto& n,
      const std::unordered_map<std::string, TypeProto*>& valueTypesByName,
      const std::unordered_map<std::string, const TensorProto*>& inputDataByName,
      const std::unordered_map<std::string, const SparseTensorProto*>& inputSparseDataByName,
      const ShapeInferenceOptions& options,
      DataValueMap* generatedShapeData = nullptr,
      GraphInferenceContext* graphInferenceContext = nullptr)
      : graphInferenceContext_{graphInferenceContext}, options_(options) {
    for (auto& attr : *n.mutable_attribute()) {
      attributesByName_[attr.name()] = &attr;
      if (attr.has_g()) {
        // need a mutable GraphProto to run inferencing on this attribute
        graphProtoAttributesByName_[attr.name()] = attr.mutable_g();
      }
    }

    for (const auto& input : n.input()) {
      auto valueTypesIter = valueTypesByName.find(input);
      if (valueTypesIter != valueTypesByName.end()) {
        allInputTypes_.push_back(valueTypesIter->second);
      } else {
        allInputTypes_.push_back(nullptr);
      }

      // input data can be in 1 of the 3 containers
      // inputDataByName - this is when input is TensorProto
      // inputSparseDataByName - this is when input is SparseTensorProto
      // generatedShapeData - this is when input was generated as part of partial data propagation
      const auto inputDataIter = inputDataByName.find(input);
      if (inputDataIter != inputDataByName.cend()) {
        allInputData_.push_back(inputDataIter->second);
        allInputSparseData_.push_back(nullptr);
        allShapeInputData_.push_back(nullptr);
      } else {
        allInputData_.push_back(nullptr);
        const auto inputSparseDataIter = inputSparseDataByName.find(input);
        if (inputSparseDataIter != inputSparseDataByName.cend()) {
          allInputSparseData_.push_back(inputSparseDataIter->second);
          allShapeInputData_.push_back(nullptr);
        } else {
          allInputSparseData_.push_back(nullptr);
          if (generatedShapeData != nullptr) {
            const auto inputShapeDataIter = generatedShapeData->find(input);
            if (inputShapeDataIter != generatedShapeData->cend()) {
              allShapeInputData_.push_back(&inputShapeDataIter->second);
            } else {
              allShapeInputData_.push_back(nullptr);
            }
          } else {
            allShapeInputData_.push_back(nullptr);
          }
        }
      }
    }

    allOutputTypes_.resize(n.output_size());
  }

  const AttributeProto* getAttribute(const std::string& name) const override {
    auto iter = attributesByName_.find(name);
    if (iter == attributesByName_.end()) {
      return nullptr;
    } else {
      return iter->second;
    }
  }

  size_t getNumInputs() const override {
    return allInputTypes_.size();
  }

  const TypeProto* getInputType(size_t index) const override {
    if (index >= allInputTypes_.size()) {
      ONNX_THROW("Input " + ONNX_NAMESPACE::to_string(index) + " is out of bounds.");
    }
    return allInputTypes_[index];
  }

  const TensorProto* getInputData(size_t index) const override {
    if (index >= allInputData_.size()) {
      ONNX_THROW("Input " + ONNX_NAMESPACE::to_string(index) + " is out of bounds.");
    }
    return allInputData_[index];
  }

  const TensorShapeProto* getSymbolicInput(size_t index) const override {
    if (index >= allShapeInputData_.size()) {
      ONNX_THROW("Input " + ONNX_NAMESPACE::to_string(index) + " is out of bounds.");
    }

    return allShapeInputData_[index];
  }

  const SparseTensorProto* getInputSparseData(size_t index) const override {
    if (index >= allInputSparseData_.size()) {
      ONNX_THROW("Input " + ONNX_NAMESPACE::to_string(index) + " is out of bounds.");
    }
    return allInputSparseData_[index];
  }

  size_t getNumOutputs() const override {
    return allOutputTypes_.size();
  }

  TypeProto* getOutputType(size_t index) override {
    if (index >= allOutputTypes_.size()) {
      ONNX_THROW("Output " + ONNX_NAMESPACE::to_string(index) + " is out of bounds.");
    }
    return &allOutputTypes_[index];
  }

  GraphInferencer* getGraphAttributeInferencer(const std::string& attr_name) override {
    if (!graphInferenceContext_) {
      fail_type_inference("GraphProto attribute inferencing is not enabled in this InferenceContextImpl instance.");
    }

    GraphInferencer* inferencer = nullptr;

    auto entry = graphAttributeInferencers_.find(attr_name);
    if (entry == graphAttributeInferencers_.cend()) {
      // create GraphInferencer instance
      auto attrNameToGraphProto = graphProtoAttributesByName_.find(attr_name);
      if (attrNameToGraphProto == graphProtoAttributesByName_.cend()) {
        fail_type_inference("Attribute ", attr_name, " does not contain a graph.");
      }

      std::unique_ptr<GraphInferencer> new_inferencer{
          new GraphInferencerImpl(*attrNameToGraphProto->second, *graphInferenceContext_, options_)};

      inferencer = new_inferencer.get();
      graphAttributeInferencers_.emplace(attr_name, std::move(new_inferencer));
    } else {
      inferencer = entry->second.get();
    }

    return inferencer;
  }

  std::vector<const TensorProto*> allInputData_;
  std::vector<const SparseTensorProto*> allInputSparseData_;
  std::vector<const TensorShapeProto*> allShapeInputData_;
  std::unordered_map<std::string, const AttributeProto*> attributesByName_;
  std::unordered_map<std::string, GraphProto*> graphProtoAttributesByName_;
  std::vector<const TypeProto*> allInputTypes_;
  std::vector<TypeProto> allOutputTypes_;
  GraphInferenceContext* graphInferenceContext_;

  // mutable as internal cache of GraphInferencer instances
  mutable std::unordered_map<std::string, std::unique_ptr<GraphInferencer>> graphAttributeInferencers_;
  ShapeInferenceOptions options_;
};

struct DataPropagationContextImpl : public DataPropagationContext {
  DataPropagationContextImpl(
      NodeProto& n,
      const std::unordered_map<std::string, TypeProto*>& valueTypesByName,
      const std::unordered_map<std::string, const TensorProto*>& inputDataByName,
      DataValueMap& generatedShapeData)
      : generatedShapeData_(generatedShapeData) {
    size_t input_idx = 0;

    for (auto& attr : *n.mutable_attribute()) {
      attributesByName_[attr.name()] = &attr;
    }

    for (const auto& input : n.input()) {
      inputIndexToNameMap_.insert({input_idx++, input});

      auto valueTypesIter = valueTypesByName.find(input);
      if (valueTypesIter != valueTypesByName.end()) {
        allInputTypes_.push_back(valueTypesIter->second);
      } else {
        allInputTypes_.push_back(nullptr);
      }

      const auto inputDataIter = inputDataByName.find(input);
      if (inputDataIter != inputDataByName.cend()) {
        allInputData_.push_back(inputDataIter->second);
      } else {
        allInputData_.push_back(nullptr);
      }
    }

    size_t output_idx = 0;
    for (const auto& output : n.output()) {
      outputIndexToNameMap_.insert({output_idx++, output});
    }

    allOutputTypes_.resize(n.output_size());
  }

  const AttributeProto* getAttribute(const std::string& name) const override {
    auto iter = attributesByName_.find(name);
    if (iter == attributesByName_.end()) {
      return nullptr;
    } else {
      return iter->second;
    }
  }

  size_t getNumInputs() const override {
    return allInputTypes_.size();
  }

  const TypeProto* getInputType(size_t index) const override {
    if (index >= allInputTypes_.size()) {
      ONNX_THROW("Input " + ONNX_NAMESPACE::to_string(index) + " is out of bounds.");
    }
    return allInputTypes_[index];
  }

  size_t getNumOutputs() const override {
    return allOutputTypes_.size();
  }

  const TypeProto* getOutputType(size_t index) const override {
    if (index >= allOutputTypes_.size()) {
      ONNX_THROW("Output " + ONNX_NAMESPACE::to_string(index) + " is out of bounds.");
    }
    return &allOutputTypes_[index];
  }

  // Convert integer vector into TensorShapeProto
  template <typename INTEGER>
  void vectorToTensorShapeProto(const std::vector<INTEGER>& input_vals, TensorShapeProto& converted_tsp) const {
    for (unsigned int i = 0; i < input_vals.size(); ++i) {
      converted_tsp.mutable_dim()->Add()->set_dim_value(input_vals[i]);
    }
  }

  const TensorShapeProto* getInputData(size_t index) override {
    if (index >= allInputData_.size()) {
      ONNX_THROW("Input " + ONNX_NAMESPACE::to_string(index) + " is out of bounds.");
    }
    const std::string input_name = inputIndexToNameMap_.at(index);
    // Gets it from previous data propagation
    auto iter = generatedShapeData_.find(input_name);
    if (iter != generatedShapeData_.end()) {
      return &iter->second;
    }
    // Otherwise, gets it from initializer if it exists
    const auto* input_data = allInputData_[index];
    // Only scalar (0D tensor) or 1D tensor can be converted for now
    // TODO: It should support tensors with more dimension on demand
    if (input_data != nullptr && (input_data->dims_size() == 0 || input_data->dims_size() == 1)) {
      TensorShapeProto tsp;

      if (input_data->data_type() == TensorProto_DataType_INT64) {
        vectorToTensorShapeProto(ParseData<int64_t>(input_data), tsp);
      } else if (input_data->data_type() == TensorProto_DataType_INT32) {
        vectorToTensorShapeProto(ParseData<int32_t>(input_data), tsp);
      } else {
        // Only supports integer type to form a shape
        return nullptr;
      }

      // Adds this TensorShapeProto from initializer into generatedShapeData
      // for future use
      auto result = generatedShapeData_.insert({input_name, std::move(tsp)});
      if (result.second) {
        return &(result.first->second);
      }
    }
    return nullptr;
  }

  void addOutputData(size_t index, TensorShapeProto&& tsp) override {
    if (index >= outputIndexToNameMap_.size()) {
      ONNX_THROW("Input " + ONNX_NAMESPACE::to_string(index) + " is out of bounds.");
    }
    auto result = generatedShapeData_.insert({outputIndexToNameMap_.at(index), std::move(tsp)});
    if (!result.second) {
      fail_shape_inference("Data for input  " + ONNX_NAMESPACE::to_string(index) + " already exists.");
    }
  }

  std::vector<const TensorProto*> allInputData_;
  std::unordered_map<size_t, std::string> inputIndexToNameMap_;
  std::unordered_map<size_t, std::string> outputIndexToNameMap_;
  std::vector<const TypeProto*> allInputTypes_;
  std::vector<TypeProto> allOutputTypes_;
  DataValueMap& generatedShapeData_;
  std::unordered_map<std::string, const AttributeProto*> attributesByName_;
};

void checkShapesAndTypes(const TypeProto_Sequence& inferredType, const TypeProto_Sequence& existingType);

void checkShapesAndTypes(const TypeProto& inferredType, const TypeProto& existingType);

template <typename TensorTypeProto>
void GenerateSymbolicShape(TensorTypeProto* inferredType, SymbolTable& symbolTable);

void MaterializeSymbolicShape(TypeProto* inferredType, SymbolTable& symbolTable);

void mergeShapesAndTypes(const TypeProto_Tensor& inferredType, TypeProto_Tensor* existingType);

void mergeShapesAndTypes(const TypeProto_SparseTensor& inferredType, TypeProto_SparseTensor* existingType);

void mergeShapesAndTypes(const TypeProto_Sequence& inferredType, TypeProto_Tensor* existingType);

void mergeShapesAndTypes(const TypeProto& inferredType, TypeProto* existingType);

///
/// ModelLocalFunctionsMap is a map of function id -> model local function proto
/// All the ONNX helper utilities expect the function id == <function_proto.domain>:<function_proto.name>
///
void InferShapes(
    GraphProto* g,
    const std::unordered_map<std::string, int>& opset_imports,
    const ISchemaRegistry* schema_registry = OpSchemaRegistry::Instance(),
    const ShapeInferenceOptions& options = {},
    const ModelLocalFunctionsMap& in_model_functions = {});

void InferShapes(
    ModelProto& m,
    const ISchemaRegistry* schema_registry = OpSchemaRegistry::Instance(),
    const ShapeInferenceOptions& options = {},
    DataValueMap* generated_shape_data_by_name = nullptr);

void InferShapes(
    const std::string& model_path,
    const std::string& save_path = "",
    const ISchemaRegistry* schema_registry = OpSchemaRegistry::Instance(),
    const ShapeInferenceOptions& options = {},
    DataValueMap* generated_shape_data_by_name = nullptr);

///
/// ModelLocalFunctionsMap is a map of function id -> model local function proto
/// All the ONNX helper utilities expect the function id == <function_proto.domain>:<function_proto.name>
///
void InferShapeForFunctionNode(
    const FunctionProto& func,
    const ISchemaRegistry* schema_registry,
    InferenceContext& ctx,
    const ShapeInferenceOptions& options = {},
    const ModelLocalFunctionsMap& model_local_functions_map = {},
    SymbolTable* symbolTable = nullptr,
    DataValueMap* generated_shape_data_by_name = nullptr);

///
/// ModelLocalFunctionsMap is a map of function id -> model local function proto
/// All the ONNX helper utilities expect the function id == <function_proto.domain>:<function_proto.name>
///
void InferShapeForFunctionNode(
    const FunctionProto& func_proto,
    const std::unordered_map<std::string, int>& func_opset_imports,
    const ISchemaRegistry* schema_registry,
    InferenceContext& ctx,
    const ShapeInferenceOptions& options = {},
    const ModelLocalFunctionsMap& model_local_functions_map = {},
    SymbolTable* symbolTable = nullptr,
    DataValueMap* generated_shape_data_by_name = nullptr);

///
/// Apply type-and-shape-inference based checks to a Function body.
/// Returns the inferred types of the outputs of the function.
/// Inference depends on the types of the inputs of the function as well as
/// the attribute values supplied.
/// A TypeProto with value_case() == TypeProto::ValueCase::VALUE_NOT_SET is used
/// for missing optional parameters.
///
std::vector<TypeProto> InferFunctionOutputTypes(
    const FunctionProto& func_proto,
    const std::vector<TypeProto>& input_types,
    const std::vector<AttributeProto>& attributes);

std::string GetErrorWithNodeInfo(const NodeProto& n, const std::runtime_error& err);

void TraverseGraphsToAddExistingSymbols(const GraphProto& g, SymbolTable& symbolTable);

} // namespace shape_inference
} // namespace ONNX_NAMESPACE
