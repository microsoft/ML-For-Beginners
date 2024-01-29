// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx/shape_inference/implementation.h"

#include <algorithm>
#include <fstream>
#include <list>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "onnx/checker.h"
#include "onnx/common/common.h"
#include "onnx/common/file_utils.h"
#include "onnx/defs/data_type_utils.h"
#include "onnx/proto_utils.h"
#include "onnx/shape_inference/attribute_binder.h"
#include "onnx/string_utils.h"

namespace ONNX_NAMESPACE {
namespace shape_inference {
namespace {

std::string GetValueCaseString(const TypeProto& type) {
  switch (type.value_case()) {
    case TypeProto::ValueCase::kTensorType:
      return "tensor_type";
    case TypeProto::ValueCase::kSequenceType:
      return "sequence_type";
    case TypeProto::ValueCase::kMapType:
      return "map_type";
    case TypeProto::ValueCase::kOptionalType:
      return "optional_type";
#ifdef ONNX_ML
    case TypeProto::ValueCase::kOpaqueType:
      return "opaque_type";
#endif
    case TypeProto::ValueCase::kSparseTensorType:
      return "sparse_tensor_type";
    case TypeProto::ValueCase::VALUE_NOT_SET:
      return "NOT_SET";
  }
  return ONNX_NAMESPACE::to_string(type.value_case());
}

std::string GetElemTypeString(const TypeProto_Tensor& type) {
#ifndef ONNX_USE_LITE_PROTO
  std::string type_str = TensorProto::DataType_Name(static_cast<TensorProto_DataType>(type.elem_type()));
  if (!type_str.empty()) {
    return type_str;
  }
#endif
  return ONNX_NAMESPACE::to_string(type.elem_type());
}

std::string GetElemTypeString(const TypeProto_SparseTensor& type) {
#ifndef ONNX_USE_LITE_PROTO
  std::string type_str = TensorProto::DataType_Name(static_cast<TensorProto_DataType>(type.elem_type()));
  if (!type_str.empty()) {
    return type_str;
  }
#endif
  return ONNX_NAMESPACE::to_string(type.elem_type());
}

inline bool IsOnnxDomainOp(const NodeProto& node, const std::string& op_type) {
  return (IsOnnxDomain(node.domain()) && (node.op_type() == op_type));
}
} // namespace

template <class T>
void CheckTensorShapesAndTypes(const T& inferred_type, const T& existing_type) {
  if (inferred_type.elem_type() != TensorProto::UNDEFINED && existing_type.elem_type() != TensorProto::UNDEFINED &&
      existing_type.elem_type() != inferred_type.elem_type()) {
    std::stringstream ss;
    ss << "Inferred elem type differs from existing elem type: (" << GetElemTypeString(inferred_type) << ") vs ("
       << GetElemTypeString(existing_type) << ")";
    fail_type_inference(ss.str());
  }

  if (!inferred_type.has_shape() || !existing_type.has_shape()) {
    return;
  }

  if (inferred_type.shape().dim_size() != existing_type.shape().dim_size()) {
    std::stringstream ss;
    ss << "Inferred shape and existing shape differ in rank: (" << inferred_type.shape().dim_size() << ") vs ("
       << existing_type.shape().dim_size() << ")";
    fail_shape_inference(ss.str());
  }

  for (int i = 0; i < inferred_type.shape().dim_size(); ++i) {
    const auto& inferred_dim = inferred_type.shape().dim(i);
    const auto& existing_dim = existing_type.shape().dim(i);
    if (inferred_dim.has_dim_value() && existing_dim.has_dim_value() &&
        inferred_dim.dim_value() != existing_dim.dim_value()) {
      std::stringstream ss;
      ss << "Inferred shape and existing shape differ in dimension " << i << ": (" << inferred_dim.dim_value()
         << ") vs (" << existing_dim.dim_value() << ")";
      fail_shape_inference(ss.str());
    }
  }
}

void checkShapesAndTypes(const TypeProto& inferred_type, const TypeProto& existing_type) {
  const auto inferred_value_case = inferred_type.value_case();
  const auto existing_value_case = existing_type.value_case();
  if (inferred_value_case == TypeProto::ValueCase::VALUE_NOT_SET ||
      existing_value_case == TypeProto::ValueCase::VALUE_NOT_SET) {
    // nothing to check; will assign inferredType to undefined existingType
    return;
  }
  if (inferred_value_case != existing_value_case) {
    fail_type_inference(
        "type case mismatch. existing=",
        GetValueCaseString(existing_type),
        " inferred=",
        GetValueCaseString(inferred_type));
  }

  if (inferred_value_case == TypeProto::kTensorType && existing_value_case == TypeProto::kTensorType) {
    CheckTensorShapesAndTypes(inferred_type.tensor_type(), existing_type.tensor_type());
  } else if (
      inferred_value_case == TypeProto::kSparseTensorType && existing_value_case == TypeProto::kSparseTensorType) {
    CheckTensorShapesAndTypes(inferred_type.sparse_tensor_type(), existing_type.sparse_tensor_type());
  } else if (inferred_value_case == TypeProto::kSequenceType && existing_value_case == TypeProto::kSequenceType) {
    checkShapesAndTypes(inferred_type.sequence_type().elem_type(), existing_type.sequence_type().elem_type());
  } else if (inferred_value_case == TypeProto::kOptionalType && existing_value_case == TypeProto::kOptionalType) {
    checkShapesAndTypes(inferred_type.optional_type().elem_type(), existing_type.optional_type().elem_type());
  } else if (
      inferred_value_case == TypeProto::TypeProto::kMapType && existing_value_case == TypeProto::TypeProto::kMapType) {
    if (inferred_type.map_type().key_type() != existing_type.map_type().key_type()) {
      fail_type_inference(
          "key type mismatch from MapProto. existing=",
          Utils::DataTypeUtils::ToDataTypeString(existing_type.map_type().key_type()),
          " inferred=",
          Utils::DataTypeUtils::ToDataTypeString(inferred_type.map_type().key_type()));
    }
    checkShapesAndTypes(inferred_type.map_type().value_type(), existing_type.map_type().value_type());
  } else {
    fail_type_inference("type case unsupported. existing=", existing_value_case, " inferred=", inferred_value_case);
  }
}

void mergeShapesAndTypes(const TypeProto_Tensor& inferred_type, TypeProto_Tensor* existing_type) {
  if (existing_type->elem_type() == TensorProto::UNDEFINED) {
    existing_type->set_elem_type(inferred_type.elem_type());
  }

  if (!inferred_type.has_shape()) {
    return;
  }

  if (!existing_type->has_shape()) {
    *existing_type->mutable_shape() = inferred_type.shape();
    return;
  }

  for (int i = 0; i < inferred_type.shape().dim_size(); ++i) {
    const auto& inferred_dim = inferred_type.shape().dim(i);
    auto* existing_dim = existing_type->mutable_shape()->mutable_dim(i);
    if ((!existing_dim->has_dim_value() && !existing_dim->has_dim_param()) || inferred_dim.has_dim_value()) {
      *existing_dim = inferred_dim;
    }
  }
}

void mergeShapesAndTypes(const TypeProto_SparseTensor& inferred_type, TypeProto_SparseTensor* existing_type) {
  if (existing_type->elem_type() == TensorProto::UNDEFINED) {
    existing_type->set_elem_type(inferred_type.elem_type());
  }

  if (!inferred_type.has_shape()) {
    return;
  }

  if (!existing_type->has_shape()) {
    *existing_type->mutable_shape() = inferred_type.shape();
    return;
  }

  for (int i = 0; i < inferred_type.shape().dim_size(); ++i) {
    const auto& inferred_dim = inferred_type.shape().dim(i);
    auto* existing_dim = existing_type->mutable_shape()->mutable_dim(i);
    if ((!existing_dim->has_dim_value() && !existing_dim->has_dim_param()) || inferred_dim.has_dim_value()) {
      *existing_dim = inferred_dim;
    }
  }
}

void mergeShapesAndTypes(const TypeProto& inferred_type, TypeProto* existing_type) {
  // Check before merge
  checkShapesAndTypes(inferred_type, *existing_type);
  const auto inferred_val_case = inferred_type.value_case();
  if (inferred_val_case == TypeProto::kTensorType) {
    mergeShapesAndTypes(inferred_type.tensor_type(), existing_type->mutable_tensor_type());
  } else if (inferred_val_case == TypeProto::kSparseTensorType) {
    mergeShapesAndTypes(inferred_type.sparse_tensor_type(), existing_type->mutable_sparse_tensor_type());
  } else if (inferred_val_case == TypeProto::kSequenceType) {
    mergeShapesAndTypes(
        inferred_type.sequence_type().elem_type(), existing_type->mutable_sequence_type()->mutable_elem_type());
  } else if (inferred_val_case == TypeProto::kOptionalType) {
    mergeShapesAndTypes(
        inferred_type.optional_type().elem_type(), existing_type->mutable_optional_type()->mutable_elem_type());
  } else if (inferred_val_case == TypeProto::kMapType) {
    if (existing_type->map_type().key_type() == TensorProto::UNDEFINED) {
      existing_type->mutable_map_type()->set_key_type(inferred_type.map_type().key_type());
    }
    mergeShapesAndTypes(inferred_type.map_type().value_type(), existing_type->mutable_map_type()->mutable_value_type());
  }
}

// TypeProto_Tensor or TypeProto_SparseTensor
template <typename TensorTypeProto>
void GenerateSymbolicShape(TensorTypeProto* inferred_type, SymbolTable& symbol_table) {
  if (!inferred_type->has_shape()) {
    return;
  }
  for (int i = 0; i < inferred_type->shape().dim_size(); ++i) {
    // set a symbol if it doesn't have dim_value and dim_param
    auto* dim = inferred_type->mutable_shape()->mutable_dim(i);
    if (!dim->has_dim_value() && !dim->has_dim_param()) {
      dim->set_dim_param(symbol_table.createNew());
    }
  }
}

void MaterializeSymbolicShape(TypeProto* inferred_type, SymbolTable& symbol_table) {
  const auto inferred_val_case = inferred_type->value_case();
  if (inferred_val_case == TypeProto::ValueCase::VALUE_NOT_SET) {
    return;
  }

  if (inferred_val_case == TypeProto::kTensorType) {
    GenerateSymbolicShape(inferred_type->mutable_tensor_type(), symbol_table);
  } else if (inferred_val_case == TypeProto::kSparseTensorType) {
    GenerateSymbolicShape(inferred_type->mutable_sparse_tensor_type(), symbol_table);
  } else if (inferred_val_case == TypeProto::kSequenceType) {
    MaterializeSymbolicShape(inferred_type->mutable_sequence_type()->mutable_elem_type(), symbol_table);
  } else if (inferred_val_case == TypeProto::kOptionalType) {
    MaterializeSymbolicShape(inferred_type->mutable_optional_type()->mutable_elem_type(), symbol_table);
  } else if (inferred_val_case == TypeProto::kMapType) {
    MaterializeSymbolicShape(inferred_type->mutable_map_type()->mutable_value_type(), symbol_table);
  } else {
    fail_shape_inference("type case unsupported for symbolic shape inference. inferred=", inferred_val_case);
  }
}

std::string GetModelLocalFunctionsMapIdentifier(const std::string& domain, const std::string& func_name) {
  return domain + ":" + func_name;
}

class ShapeInferenceImplBase {
 public:
  void updateType(const std::string& name, TypeProto* inferred_type) {
    if (inferred_type->value_case() == TypeProto::ValueCase::VALUE_NOT_SET) {
      return;
    }

    if (symbol_table) {
      MaterializeSymbolicShape(inferred_type, *symbol_table);
    }

    // Find any pre-existing type and shape info. If there is such,
    // then check for compatibility with the inferred
    // information. Otherwise, initialize it in an empty state.
    auto iter = value_types_by_name.find(name);
    TypeProto* existing_type = nullptr;
    if (iter != value_types_by_name.end()) {
      existing_type = iter->second;
    } else {
      // Create a new value_info if defined type does not exist
      auto vi = g.add_value_info(); // TODO: clean this up
      vi->set_name(name);
      existing_type = vi->mutable_type();
      // For undefined output type, update both value_info and output for now
      // Update existing output with undefined type: assign inferred type to it
      iter = undefined_value_types_by_name.find(name);
      if (iter != undefined_value_types_by_name.end()) {
        *iter->second = *inferred_type;
      }
    }

    // TODO: cleanup this by merging with previous if-else
    // Now we can merge pre-existing and inferred info
    mergeShapesAndTypes(*inferred_type, existing_type);

    // Make merged info available to further inference.
    value_types_by_name[name] = existing_type;
  }

  void updateType(ValueInfoProto& valueInfo) {
    if (valueInfo.has_type()) {
      value_types_by_name[valueInfo.name()] = valueInfo.mutable_type();
    } else {
      undefined_value_types_by_name[valueInfo.name()] = valueInfo.mutable_type();
    }
  }

  template <typename T>
  void addTemporaryConstant(const std::string& name, const T& vector) {
    input_data_by_name_holder[name] = ToTensor(vector);
    input_data_by_name[name] = &input_data_by_name_holder[name];
  }

  void preprocess(const NodeProto& n) {
    if (checker::check_is_experimental_op(n)) {
      has_experimental_op = true;
    } else if (IsOnnxDomainOp(n, "Constant") && n.output().size() == 1) {
      const std::string& output_name = n.output(0);
      for (const auto& attr : n.attribute()) {
        if (attr.name() == "value") {
          if (attr.type() == AttributeProto::TENSOR && attr.has_t()) {
            if (reuse_constant_tensors) {
              input_data_by_name[output_name] = &attr.t();
            } else {
              input_data_by_name_holder[output_name] = attr.t();
              input_data_by_name[output_name] = &input_data_by_name_holder[output_name];
            }
          } else if (attr.type() == AttributeProto::SPARSE_TENSOR && attr.has_sparse_tensor()) {
            if (reuse_constant_tensors) {
              input_sparse_data_by_name[output_name] = &attr.sparse_tensor();
            }
          }
        } else {
          switch (attr.type()) {
            case AttributeProto::INTS: {
              std::vector<int64_t> ints{attr.ints().begin(), attr.ints().end()};
              addTemporaryConstant(output_name, ints);
              break;
            }
            case AttributeProto::INT: {
              std::vector<int64_t> ints({attr.i()});
              addTemporaryConstant(output_name, ints);
              break;
            }
            case AttributeProto::FLOATS: {
              std::vector<float> floats{attr.floats().begin(), attr.floats().end()};
              addTemporaryConstant(output_name, floats);
              break;
            }
            case AttributeProto::FLOAT: {
              std::vector<float> floats({attr.f()});
              addTemporaryConstant(output_name, floats);
              break;
            }
            default:
              break;
          }
        }
      }
    }
  }

  // Initialize a DataValueMap for a called function from the DataValueMap of the caller
  void bindValuesOnCall(
      const DataValueMap& caller_map,
      const NodeProto& caller,
      DataValueMap& callee_map,
      const FunctionProto& callee) {
    auto num_inputs = (std::min)(caller.input_size(), callee.input_size());
    for (int i = 0; i < num_inputs; ++i) {
      const std::string& actual = caller.input(i);
      const std::string& formal = callee.input(i);
      if (!actual.empty()) {
        auto it = caller_map.find(actual);
        if (it != caller_map.end()) {
          callee_map[formal] = it->second;
        }
      }
    }
  }

  // Update a DataValueMap for a calling function from the DataValueMap of the callee
  void bindValuesOnReturn(
      const DataValueMap& callee_map,
      const FunctionProto& callee,
      DataValueMap& caller_map,
      const NodeProto& caller) {
    auto num_outputs = (std::min)(caller.output_size(), callee.output_size());
    for (int i = 0; i < num_outputs; ++i) {
      const std::string& actual = caller.output(i);
      const std::string& formal = callee.output(i);
      if (!actual.empty()) {
        auto it = callee_map.find(formal);
        if (it != callee_map.end()) {
          caller_map[actual] = it->second;
        }
      }
    }
  }

  void processCall(const NodeProto& caller, const FunctionProto& callee, InferenceContext& ctx) {
    DataValueMap callee_value_map;
    if (generated_shape_data_by_name != nullptr) {
      bindValuesOnCall(*generated_shape_data_by_name, caller, callee_value_map, callee);
    }
    InferShapeForFunctionNode(
        callee, schema_registry, ctx, options, model_local_functions_map, symbol_table, &callee_value_map);
    if (generated_shape_data_by_name != nullptr) {
      bindValuesOnReturn(callee_value_map, callee, *generated_shape_data_by_name, caller);
    }
  }

  void process(NodeProto& n) {
    // Resolve domain for node
    auto dit = opset_imports.find(n.domain());
    if (dit == opset_imports.end()) {
      // Both "" (ONNX_DOMAIN) and "ai.onnx" (AI_ONNX_DOMAIN) refer to the default ONNX domain
      if (n.domain() == ONNX_DOMAIN) {
        dit = opset_imports.find(AI_ONNX_DOMAIN);
      }
      if (dit == opset_imports.end()) {
        fail_type_inference(
            "Cannot infer type and shape for node name ",
            n.name(),
            ". No opset import for domain ",
            n.domain(),
            " optype ",
            n.op_type());
      }
    }
    auto domain_version = dit->second;
    const auto schema = schema_registry->GetSchema(n.op_type(), domain_version, n.domain());
    InferenceContextImpl ctx(
        n,
        value_types_by_name,
        input_data_by_name,
        input_sparse_data_by_name,
        options,
        generated_shape_data_by_name,
        &graph_inference_context);

    ONNX_TRY {
      if (schema) {
        if (schema->has_type_and_shape_inference_function()) {
          schema->GetTypeAndShapeInferenceFunction()(ctx);
        } else if (schema->HasFunction()) {
          processCall(n, *(schema->GetFunction()), ctx);
        } else {
          // Continue with inference for remaining nodes
          return;
        }
      } else if (model_local_functions_map.size() > 0) {
        auto iter = model_local_functions_map.find(GetModelLocalFunctionsMapIdentifier(n.domain(), n.op_type()));
        if (iter != model_local_functions_map.end()) {
          processCall(n, *(iter->second), ctx);
        } else {
          has_unsupported_op = true;
          return;
        }
      } else {
        has_unsupported_op = true;
        return;
      }
    }
    ONNX_CATCH(const ONNX_NAMESPACE::InferenceError& ex) {
      ONNX_HANDLE_EXCEPTION([&]() {
        // onnx does not support unsupported/experimental operators
        // so it won't consider it as an error
        if (!has_unsupported_op && !has_experimental_op) {
          inference_errors.push_back(GetErrorWithNodeInfo(n, ex));
        }
      });
      // Continue with inference for remaining nodes
      return;
    }

    ONNX_TRY {
      // check the type-equality for input and output
      if (options.check_type && schema) {
        schema->CheckInputOutputType(ctx);
      }

      for (int i = 0; i < n.output_size(); ++i) {
        // skip type and shape propagation for missing optional outputs.
        if (!n.output(i).empty())
          updateType(n.output(i), ctx.getOutputType(i));
      }

      preprocess(n);

      // If data propagation is enabled, propagate shape data if it exists.
      if (options.enable_data_propagation && schema && schema->has_data_propagation_function()) {
        if (generated_shape_data_by_name == nullptr) {
          fail_shape_inference(
              "Container for generated shape data cannot be nullptr when enable_data_propagation option is set.");
        }
        DataPropagationContextImpl data_propagation_ctx(
            n, value_types_by_name, input_data_by_name, *generated_shape_data_by_name);
        schema->GetDataPropagationFunction()(data_propagation_ctx);
      }
    }
    ONNX_CATCH(const std::runtime_error& err) {
      ONNX_HANDLE_EXCEPTION([&]() { fail_shape_inference(GetErrorWithNodeInfo(n, err)); });
    }
  }

  // TypeProto_Tensor or TypeProto_SparseTensor
  template <typename T>
  void processInitializer(
      const std::string& name,
      const T& tensorValue,
      TypeProto& initializer_type,
      std::unordered_map<std::string, const T*>& map) {
    map[name] = &tensorValue;
    auto iter = value_types_by_name.find(name);
    // If it already exists in input, check input and initializer is sync
    // use shape info from input (input has priority over initializer)
    if (iter != value_types_by_name.end()) {
      checkShapesAndTypes(initializer_type, *iter->second);
      // CheckTensorShapesAndTypes(*initializer_tensor_type, *iter->second->mutable_tensor_type());
    }
    // Support IR>=4: some tensors can only exist in initializer and not in input
    // So shape_inference should make use of initializer shapes
    // Store initializer shape info in value_info as well
    else if (ir_version >= 4) {
      initializer_type_list.push_back(std::move(initializer_type));
      value_types_by_name[name] = &initializer_type_list.back();
    }
  }

  void process(GraphProto& graph) {
    if (symbol_table) {
      TraverseGraphsToAddExistingSymbols(graph, *symbol_table);
    }
    for (auto& vi : *graph.mutable_value_info()) {
      updateType(vi);
    }
    for (auto& vi : *graph.mutable_input()) {
      updateType(vi);
    }
    for (auto& vi : *graph.mutable_output()) {
      updateType(vi);
    }
    for (const auto& tp : graph.initializer()) {
      TypeProto initializer_type;
      TypeProto_Tensor* initializer_tensor_type = initializer_type.mutable_tensor_type();
      initializer_tensor_type->set_elem_type(tp.data_type());
      // set the shape according to the initializer shape info
      auto* shape = initializer_tensor_type->mutable_shape();
      for (int i = 0; i < tp.dims_size(); ++i) {
        shape->add_dim()->set_dim_value(tp.dims(i));
      }
      processInitializer(tp.name(), tp, initializer_type, input_data_by_name);
    }
    for (const auto& tp : graph.sparse_initializer()) {
      TypeProto initializer_type;
      auto* initializer_sparse_tensor_type = initializer_type.mutable_sparse_tensor_type();
      initializer_sparse_tensor_type->set_elem_type(tp.values().data_type());
      // set the shape according to the initializer shape info
      auto* shape = initializer_sparse_tensor_type->mutable_shape();
      for (int i = 0; i < tp.dims_size(); ++i) {
        shape->add_dim()->set_dim_value(tp.dims(i));
      }
      processInitializer(tp.values().name(), tp, initializer_type, input_sparse_data_by_name);
    }
    for (auto& n : *graph.mutable_node()) {
      process(n);
    }
  }

  void process(const NodeProto& n, internal::AttributeBinder& attribute_binder) {
    NodeProto copy_n(n);
    attribute_binder.VisitNode(&copy_n);
    process(copy_n);
  }

  void process(const FunctionProto& func_proto, InferenceContext& ctx) {
    // Ensure Constant node tensor-attributes are copied
    bool old_reuse_constant_tensors = reuse_constant_tensors;
    reuse_constant_tensors = false;

    // Get a temporary tensor-shape map
    const int num_actual_inputs = static_cast<int>(ctx.getNumInputs());
    const auto num_func_inputs = func_proto.input_size();
    std::vector<TypeProto> types_cache(num_func_inputs);
    for (int i = 0; i < num_func_inputs; ++i) {
      auto& parameter_name = func_proto.input().Get(i);
      auto* type_ptr = (i < num_actual_inputs) ? ctx.getInputType(i) : nullptr;
      // nullptr is valid, and indicates a missing optional input
      if (type_ptr != nullptr) {
        // Use a temporary copy of original type.
        // TODO: investigate whether we can eliminate use of temporary copy
        types_cache[i] = *type_ptr;
        value_types_by_name[parameter_name] = &types_cache[i];
      } else
        value_types_by_name[parameter_name] = nullptr;
    }

    // Create a temporary initializer value map
    for (int i = 0; i < num_actual_inputs && i < num_func_inputs; ++i) {
      const TypeProto* type = ctx.getInputType(i);
      if (type != nullptr) {
        if (type->value_case() == TypeProto::kTensorType && ctx.getInputData(i) != nullptr) {
          input_data_by_name[func_proto.input().Get(i)] = ctx.getInputData(i);
        } else if (type->value_case() == TypeProto::kSparseTensorType && ctx.getInputSparseData(i) != nullptr) {
          input_sparse_data_by_name[func_proto.input().Get(i)] = ctx.getInputSparseData(i);
        }
      }
    }

    std::unordered_map<std::string, const AttributeProto*> attr_map;
    for (auto& attr : func_proto.attribute()) {
      if (ctx.getAttribute(attr) != nullptr) {
        attr_map[attr] = ctx.getAttribute(attr);
      }
    }

    for (auto& default_value : func_proto.attribute_proto()) {
      const std::string& name = default_value.name();
      const AttributeProto* value = ctx.getAttribute(name);
      attr_map[name] = (value != nullptr) ? value : &default_value;
    }

    internal::AttributeBinder attribute_binder(attr_map);
    for (auto& n : func_proto.node()) {
      process(n, attribute_binder);
    }

    for (int i = 0; i < func_proto.output_size(); ++i) {
      const std::string& output_name = func_proto.output().Get(i);
      // Skip if no type inferred for the tensor
      auto iter = value_types_by_name.find(output_name);
      if (iter != value_types_by_name.cend()) {
        // Copy the type info to ctx
        // to pass back to main graph
        auto type_proto = ctx.getOutputType(i);
        type_proto->CopyFrom(*(iter->second));
      }
    }

    reuse_constant_tensors = old_reuse_constant_tensors;
  }

 public:
  ShapeInferenceImplBase(
      GraphProto* g_in,
      const std::unordered_map<std::string, TypeProto*>& outer_scope_value_types_by_name_in,
      const std::unordered_map<std::string, int>& opset_imports_in,
      const ShapeInferenceOptions& options_in,
      SymbolTable* symbol_table_in,
      const ModelLocalFunctionsMap& model_local_functions_map_in,
      const ISchemaRegistry* schema_registry_in = OpSchemaRegistry::Instance(),
      DataValueMap* generated_shape_data_by_name_in = nullptr,
      const int ir_version_in = IR_VERSION // default the latest one
      )
      : g(*g_in),
        value_types_by_name(outer_scope_value_types_by_name_in),
        opset_imports(opset_imports_in),
        options(options_in),
        symbol_table(symbol_table_in),
        model_local_functions_map(model_local_functions_map_in),
        schema_registry(schema_registry_in),
        generated_shape_data_by_name(generated_shape_data_by_name_in),
        ir_version(ir_version_in),
        graph_inference_context{
            value_types_by_name,
            opset_imports,
            symbol_table,
            model_local_functions_map,
            schema_registry,
            generated_shape_data_by_name,
            ir_version} {
    if (options.enable_data_propagation && generated_shape_data_by_name == nullptr) {
      fail_shape_inference(
          "Container for generated shape data cannot be nullptr when enable_data_propagation option is set.");
    }
  }

  void finalizeShapeInference() {
    auto& errors = getErrors();
    // Throw shape inference error if any. Error mode right now only supports 0 and 1.
    // When set to 0, any node level shape inference errors are not thrown. This is to support backward compatiblity
    // with 1.7 and earlier releases. When set to 1 it will throw all exceptions.
    // TODO: Add a more granular way for exception handling.
    if (!errors.empty() && options.error_mode > 0) {
      std::string full_errors = "Inference error(s): ";
      for (const std::string& error : inference_errors) {
        full_errors += error + "\n";
      }
      fail_shape_inference(full_errors);
    }
  }

  const std::vector<std::string>& getErrors() const {
    return inference_errors;
  }

 private:
  GraphProto& g;
  std::unordered_map<std::string, TypeProto*> value_types_by_name;
  const std::unordered_map<std::string, int>& opset_imports;

  const ShapeInferenceOptions& options;
  SymbolTable* symbol_table;
  const ModelLocalFunctionsMap& model_local_functions_map;
  const ISchemaRegistry* schema_registry;
  DataValueMap* generated_shape_data_by_name;
  int ir_version;
  GraphInferenceContext graph_inference_context;

  std::unordered_map<std::string, TypeProto*> undefined_value_types_by_name;
  std::unordered_map<std::string, const TensorProto*> input_data_by_name;
  std::unordered_map<std::string, TensorProto> input_data_by_name_holder;
  std::unordered_map<std::string, const SparseTensorProto*> input_sparse_data_by_name;

  bool has_experimental_op = false;
  bool has_unsupported_op = false;

  std::vector<std::string> inference_errors;

  std::list<TypeProto> initializer_type_list;

  // reuse_constant_tensors: controls whether we need to copy tensors occurring as attributes
  // in Constant nodes. We avoid it for inference for graphs, but must make a copy for functions.
  bool reuse_constant_tensors = true;
};

static void InferShapesImpl(
    GraphProto* g,
    const std::unordered_map<std::string, TypeProto*>& outer_scope_value_types_by_name,
    const std::unordered_map<std::string, int>& opset_imports,
    const ShapeInferenceOptions& options,
    SymbolTable* symbol_table,
    const ModelLocalFunctionsMap& model_local_functions_map,
    const ISchemaRegistry* schema_registry = OpSchemaRegistry::Instance(),
    DataValueMap* generated_shape_data_by_name = nullptr,
    const int ir_version = IR_VERSION // default the latest one
) {
  DataValueMap empty;
  if (generated_shape_data_by_name == nullptr) {
    generated_shape_data_by_name = &empty;
  }
  ShapeInferenceImplBase base(
      g,
      outer_scope_value_types_by_name,
      opset_imports,
      options,
      symbol_table,
      model_local_functions_map,
      schema_registry,
      generated_shape_data_by_name,
      ir_version);
  base.process(*g);
  base.finalizeShapeInference();
}

// Either ModelProto or FunctionProto
template <class T>
std::unordered_map<std::string, int> GetOpsetImportsFromProto(const T& proto) {
  std::unordered_map<std::string, int> opset_imports;
  for (const auto& opset_import : proto.opset_import()) {
    opset_imports[opset_import.domain()] = static_cast<int>(opset_import.version());
  }
  return opset_imports;
}

void InferShapes(
    GraphProto* g,
    const std::unordered_map<std::string, int>& opset_imports,
    const ISchemaRegistry* schema_registry,
    const ShapeInferenceOptions& options,
    const std::unordered_map<std::string, const FunctionProto*>& model_local_functions) {
  SymbolTableImpl symbol_table;
  InferShapesImpl(
      g,
      std::unordered_map<std::string, TypeProto*>(0),
      opset_imports,
      options,
      &symbol_table,
      model_local_functions,
      schema_registry);
}

void InferShapes(
    ModelProto& m,
    const ISchemaRegistry* schema_registry,
    const ShapeInferenceOptions& options,
    DataValueMap* generated_shape_data_by_name) {
  auto opset_imports = GetOpsetImportsFromProto(m);
  SymbolTableImpl symbol_table;
  ModelLocalFunctionsMap model_local_functions_by_id;
  for (const auto& function_proto : m.functions()) {
    model_local_functions_by_id.insert(
        {GetModelLocalFunctionsMapIdentifier(function_proto.domain(), function_proto.name()), &function_proto});
  }
  InferShapesImpl(
      m.mutable_graph(),
      std::unordered_map<std::string, TypeProto*>(0),
      opset_imports,
      options,
      &symbol_table,
      model_local_functions_by_id,
      schema_registry,
      generated_shape_data_by_name,
      m.ir_version());
}

void InferShapes(
    const std::string& model_path,
    const std::string& save_path,
    const ISchemaRegistry* schema_registry,
    const ShapeInferenceOptions& options,
    DataValueMap* generated_shape_data_by_name) {
  ModelProto model;
  LoadProtoFromPath(model_path, model);
  InferShapes(model, schema_registry, options, generated_shape_data_by_name);
  // Save the inferred model to the original model path
  // Use SerializeToString instead of SerializeToOstream due to LITE_PROTO
  std::fstream output(save_path, std::ios::out | std::ios::trunc | std::ios::binary);
  std::string model_string;
  ONNX_TRY {
    model.SerializeToString(&model_string);
    output << model_string;
  }
  ONNX_CATCH(...) {
    fail_check("Unable to save inferred model to the target path:", save_path);
  }
}

// Infer shape for functions
void InferShapeForFunctionNode(
    const FunctionProto& func_proto,
    const std::unordered_map<std::string, int>& func_opset_imports,
    const ISchemaRegistry* schema_registry,
    InferenceContext& ctx,
    const ShapeInferenceOptions& options,
    const std::unordered_map<std::string, const FunctionProto*>& model_local_functions_map,
    SymbolTable* symbol_table,
    DataValueMap* generated_shape_data_by_name) {
  GraphProto g;
  ShapeInferenceImplBase base(
      &g,
      {}, // outer_scope_value_types_by_name
      func_opset_imports,
      options,
      symbol_table,
      model_local_functions_map,
      schema_registry,
      generated_shape_data_by_name);
  base.process(func_proto, ctx);
  base.finalizeShapeInference();
}

void InferShapeForFunctionNode(
    const FunctionProto& function_proto,
    const ISchemaRegistry* schema_registry,
    InferenceContext& ctx,
    const ShapeInferenceOptions& options,
    const std::unordered_map<std::string, const FunctionProto*>& model_local_functions_map,
    SymbolTable* symbol_table,
    DataValueMap* generated_shape_data_by_name) {
  auto opset_imports = GetOpsetImportsFromProto(function_proto);
  InferShapeForFunctionNode(
      function_proto,
      opset_imports,
      schema_registry,
      ctx,
      options,
      model_local_functions_map,
      symbol_table,
      generated_shape_data_by_name);
}

struct FunctionInferenceContext : public InferenceContext {
  FunctionInferenceContext(
      const FunctionProto& func_proto,
      const std::vector<TypeProto>& input_types,
      const std::vector<AttributeProto>& attributes,
      const ShapeInferenceOptions& options)
      : input_types_(input_types), options_(options) {
    for (const auto& attr : attributes) {
      attributesByName_[attr.name()] = &attr;
    }
    auto num_outputs = func_proto.output_size();
    for (int i = 0; i < num_outputs; i++) {
      output_types_.push_back(TypeProto());
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
  size_t getNumInputs() const override {
    return input_types_.size();
  }

  size_t getNumOutputs() const override {
    return output_types_.size();
  }

  const TypeProto* getInputType(size_t index) const override {
    // We should return nullptr for missing optional parameters.
    // An uninitialized TypeProto() is used for missing optional parameters, and
    // is mapped to a nullptr here.
    if (index >= input_types_.size())
      return nullptr;
    if (input_types_[index].value_case() == TypeProto::ValueCase::VALUE_NOT_SET)
      return nullptr;
    return &input_types_[index];
  }

  TypeProto* getOutputType(size_t index) override {
    return (index < output_types_.size()) ? &output_types_[index] : nullptr;
  }

  GraphInferencer* getGraphAttributeInferencer(const std::string& attribute_name) override {
    ONNX_UNUSED_PARAMETER(attribute_name); // This method is unused for function-type-inference.
    return nullptr;
  }

  const TensorProto* getInputData(size_t index) const override {
    ONNX_UNUSED_PARAMETER(index); // This inference doesn't take advantage of statically known input values.
    return nullptr;
  }

  const SparseTensorProto* getInputSparseData(size_t index) const override {
    ONNX_UNUSED_PARAMETER(index); // This inference doesn't take advantage of statically known input values.
    return nullptr;
  }

  const TensorShapeProto* getSymbolicInput(size_t index) const override {
    ONNX_UNUSED_PARAMETER(index); // This inference doesn't take advantage of data-propagation.
    return nullptr;
  }

  std::vector<TypeProto> popOutputTypes() {
    return std::move(output_types_);
  }

 private:
  const std::vector<TypeProto>& input_types_;
  std::vector<TypeProto> output_types_;
  std::unordered_map<std::string, const AttributeProto*> attributesByName_;
  ShapeInferenceOptions options_;
};

std::vector<TypeProto> InferFunctionOutputTypes(
    const FunctionProto& function_proto,
    const std::vector<TypeProto>& input_types,
    const std::vector<AttributeProto>& attributes) {
  // TODO: if it is desirable for infer_function_output_types to provide check_type, strict_mode, data_prop,
  // we can add them to the Python API. For now we just assume the default options.
  ShapeInferenceOptions options{true, 1, false};
  FunctionInferenceContext ctx(function_proto, input_types, attributes, options);
  auto opset_imports = GetOpsetImportsFromProto(function_proto);
  GraphProto g;
  ShapeInferenceImplBase base(
      &g,
      {}, // outer_scope_value_types_by_name
      opset_imports,
      options,
      /*symbol_table*/ nullptr,
      /*model_local_functions_map*/ {},
      /*schema_registry*/ OpSchemaRegistry::Instance(),
      /*generated_shape_data_by_name*/ nullptr);
  base.process(function_proto, ctx);
  base.finalizeShapeInference();
  return ctx.popOutputTypes();
}

std::vector<const TypeProto*> GraphInferencerImpl::doInferencing(
    const std::vector<const TypeProto*>& input_types,
    const std::vector<const TensorProto*>& input_data) {
  SymbolTable* symbol_table = context_->symbol_table;
  int num_inputs = int(input_types.size());
  std::unordered_set<std::string> initializer_name_set;
  for (const auto& tp : g_->initializer()) {
    initializer_name_set.insert(tp.name());
  }

  if (context_->ir_version >= 4) {
    if (g_->input_size() != num_inputs) {
      fail_shape_inference("Graph has ", g_->input_size(), " inputs but ", num_inputs, " were provided");
    }
    for (int i = 0; i < g_->input_size(); ++i) {
      if (initializer_name_set.count(g_->input(i).name()) > 0) {
        fail_shape_inference(
            "Cannot use the same name as both a subgraph initializer and subgraph input: ", g_->input(i).name());
      }
    }
  } else {
    // IR < 4 requires all initializers to be optional inputs
    // So the number of graph input can be larger than the number of node input
    if (num_inputs > g_->input_size()) {
      fail_shape_inference(
          "Graph has ",
          g_->input_size(),
          " inputs but ",
          num_inputs,
          " were provided.",
          "The number of graph input cannot be smaller than the number of node input");
    } else if (num_inputs < g_->input_size()) {
      for (int i = 0; i < g_->input_size(); ++i) {
        if (i < num_inputs && initializer_name_set.count(g_->input(i).name()) > 0) {
          fail_shape_inference("Graph initializer names must appear after the actual inputs: ", g_->input(i).name());
        } else if (i >= num_inputs && initializer_name_set.count(g_->input(i).name()) == 0) {
          // Further check whether the additional input is in initializers
          fail_shape_inference("Cannot find missing input: ", g_->input(i).name(), "in initializers. ");
        }
      }
    }
  }

  for (int i = 0, end = num_inputs; i < end; ++i) {
    const TypeProto* inferred_input = input_types[i];

    if (!inferred_input)
      continue;

    TypeProto* graph_input = g_->mutable_input(i)->mutable_type();
    // Even if graphInput doesn't have defined type, it will assign inferredType to it
    mergeShapesAndTypes(*inferred_input, graph_input);

    if (symbol_table) {
      MaterializeSymbolicShape(graph_input, *symbol_table);
    }
  }

  // future: pass inputData into InferShapes either directly, or indirectly by
  // updating initializers that match subgraph inputs.
  (void)input_data;
  InferShapesImpl(
      g_,
      *context_->outer_scope_value_types_by_name, // never null
      context_->opset_imports,
      options_,
      symbol_table,
      context_->model_local_functions,
      context_->schema_registry,
      context_->generated_shape_data_by_name);

  std::vector<const TypeProto*> graph_output_types;
  graph_output_types.reserve(g_->output().size());
  for (const ValueInfoProto& output : g_->output()) {
    graph_output_types.push_back(&output.type());
  }

  return graph_output_types;
}

std::string GetErrorWithNodeInfo(const NodeProto& n, const std::runtime_error& err) {
  std::string op_name = n.has_name() ? (", node name: " + n.name()) : "";
  return "(op_type:" + n.op_type() + op_name + "): " + err.what();
}

void TraverseGraphsToAddExistingSymbols(const GraphProto& g, SymbolTable& symbol_table) {
  symbol_table.addFromGraph(g);
  for (const auto& n : g.node()) {
    for (auto& attr : n.attribute()) {
      if (attr.has_g()) {
        TraverseGraphsToAddExistingSymbols(attr.g(), symbol_table);
      }
    }
  }
}

} // namespace shape_inference
} // namespace ONNX_NAMESPACE
