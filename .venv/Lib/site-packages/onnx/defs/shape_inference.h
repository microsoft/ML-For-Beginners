/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <functional>
#include "onnx/defs/data_type_utils.h"
#include "onnx/proto_utils.h"
#include "onnx/string_utils.h"

namespace ONNX_NAMESPACE {

using Dim = TensorShapeProto_Dimension;

struct ShapeInferenceOptions {
  // Checks the type-equality for input and output
  bool check_type;
  // 1: Will throw any node level shape infer errors
  // 0: Won't throw node-level shape infer errors, but other errors
  // like merging existing shape with inferred etc are thrown
  int error_mode;
  // Enables data propagation for limited operators
  // to perform shape computation
  bool enable_data_propagation;
  ShapeInferenceOptions(bool check_type_val = false, int strict_mode_val = 0, bool data_prop_val = false)
      : check_type(check_type_val), error_mode(strict_mode_val), enable_data_propagation(data_prop_val){};
};

// Maintains a SymbolTable for symbolic shape inference
class SymbolTable {
 public:
  // Adds existing symbols from a main graph or subgraph
  virtual void addFromGraph(const GraphProto& g) = 0;
  // Creates a new symbol which is not duplicate as any existing one
  virtual std::string createNew(const std::string& symbol_prefix) = 0;
  virtual ~SymbolTable() = default;
};

class GraphInferencer {
 public:
  // Perform inferencing on the graph contained in GraphInferencer.
  // Returns the graph output types post-inferencing.
  virtual std::vector<const TypeProto*> doInferencing(
      const std::vector<const TypeProto*>& inputTypes,
      const std::vector<const TensorProto*>& inputData) = 0;
  virtual ~GraphInferencer() = default;
};

// Exception class used for handling errors in type and shape inference

class InferenceError final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;

  InferenceError(const std::string& message) : std::runtime_error(message) {}

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

#define fail_type_inference(...) \
  ONNX_THROW_EX(ONNX_NAMESPACE::InferenceError(ONNX_NAMESPACE::MakeString("[TypeInferenceError] ", __VA_ARGS__)));

#define fail_shape_inference(...) \
  ONNX_THROW_EX(ONNX_NAMESPACE::InferenceError(ONNX_NAMESPACE::MakeString("[ShapeInferenceError] ", __VA_ARGS__)));

struct InferenceContext {
  virtual const AttributeProto* getAttribute(const std::string& name) const = 0;
  virtual size_t getNumInputs() const = 0;
  virtual const TypeProto* getInputType(size_t index) const = 0;
  virtual bool hasInput(size_t index) const {
    // The default implementation below is used for backward-compatibility
    // for implementations of InferenceContext that don't provide an explicit
    // implementation. This works for normal usage, but may be imprecise in
    // the edge-case where an input is supplied but has no known type.
    // However, inference-methods work only under the assumption that the
    // input-types of all inputs are known.
    return ((index < getNumInputs()) && (getInputType(index) != nullptr));
  }
  virtual const TensorProto* getInputData(size_t index) const = 0;
  virtual size_t getNumOutputs() const = 0;
  virtual TypeProto* getOutputType(size_t index) = 0;
  virtual GraphInferencer* getGraphAttributeInferencer(const std::string& attribute_name) = 0;
  virtual ~InferenceContext() {}
  virtual const SparseTensorProto* getInputSparseData(size_t index) const = 0;
  // Gets the shape inputs computed by partial data propagation.
  virtual const TensorShapeProto* getSymbolicInput(size_t index) const = 0;
};

// We use data propagation to perform partial evaluation of the model, to compute statically
// known information about tensor values. It is intended to improve the precision of shape
// inference. We reuse TensorShapeProto to represent the statically known values. One
// limitation of this is that TensorShapeProto can represent only integer values.
// As an example, data-propagation is intended to handle code-fragments like below:
//   shape = Shape(X)
//   batchsize = Slice(shape, [0], [1])
//   newshape = Concat (batchsize, [1024, 1024])
//   Z = Reshape(Y, newshape)
// If the shape of X is statically known, then data-propagation should be able to determine
// the value of newshape, as well as the shape of Z.
struct DataPropagationContext {
  virtual const AttributeProto* getAttribute(const std::string& name) const = 0;
  virtual size_t getNumInputs() const = 0;
  virtual const TypeProto* getInputType(size_t index) const = 0;
  virtual size_t getNumOutputs() const = 0;
  virtual const TypeProto* getOutputType(size_t index) const = 0;
  virtual ~DataPropagationContext() {}
  virtual const TensorShapeProto* getInputData(size_t index) = 0;
  virtual void addOutputData(size_t index, TensorShapeProto&& tp) = 0;
};

using InferenceFunction = std::function<void(InferenceContext&)>;
using DataPropagationFunction = std::function<void(DataPropagationContext&)>;

// This no-op inference function is used for operators without an
// inference implementation.
inline void dummyInferenceFunction(InferenceContext&){};

// This no-op data propagation function is used for operators without a defined data propagator
inline void dummyDataPropagationFunction(DataPropagationContext&){};

template <typename T>
inline bool getRepeatedAttribute(InferenceContext& ctx, std::string attr_name, std::vector<T>& values) {
  const auto* attr = ctx.getAttribute(attr_name);
  if (attr) {
    values = RetrieveValues<T>(*attr);
    return true;
  } else {
    return false;
  }
}

inline int64_t getAttribute(InferenceContext& ctx, const std::string& attributeName, int64_t defaultValue) {
  auto attr_proto = ctx.getAttribute(attributeName);
  if ((nullptr != attr_proto) && attr_proto->has_i())
    return attr_proto->i();
  return defaultValue;
}

inline int64_t getAttribute(DataPropagationContext& ctx, const std::string& attributeName, int64_t defaultValue) {
  auto attr_proto = ctx.getAttribute(attributeName);
  if ((nullptr != attr_proto) && attr_proto->has_i())
    return attr_proto->i();
  return defaultValue;
}

inline std::string
getAttribute(InferenceContext& ctx, const std::string& attributeName, const std::string& defaultValue) {
  auto attr_proto = ctx.getAttribute(attributeName);
  if ((nullptr != attr_proto) && attr_proto->has_s())
    return attr_proto->s();
  return defaultValue;
}

inline TensorShapeProto::Dimension operator*(TensorShapeProto::Dimension dim1, TensorShapeProto::Dimension dim2) {
  TensorShapeProto::Dimension result;
  if (dim1.has_dim_value() && dim2.has_dim_value()) {
    result.set_dim_value(dim1.dim_value() * dim2.dim_value());
  } else if (dim1.has_dim_value() && (dim1.dim_value() == 1)) {
    return dim2;
  } else if (dim2.has_dim_value() && (dim2.dim_value() == 1)) {
    return dim1;
  }
  return result;
}

inline TensorShapeProto::Dimension operator*(TensorShapeProto::Dimension dim1, int64_t dim2) {
  TensorShapeProto::Dimension result;
  if (dim1.has_dim_value()) {
    result.set_dim_value(dim1.dim_value() * dim2);
  } else if (dim2 == 1) {
    return dim1;
  }
  return result;
}

inline TensorShapeProto::Dimension operator/(TensorShapeProto::Dimension dim1, int64_t dim2) {
  TensorShapeProto::Dimension result;
  if (dim1.has_dim_value()) {
    result.set_dim_value(dim1.dim_value() / dim2);
  } else if (dim2 == 1) {
    return dim1;
  }
  return result;
}

// if from >= upto_exclusive, return 1.
// Caller must make sure upto_exclusive is less than or equal to shape.size()
// Caller must make sure from>=0
inline TensorShapeProto::Dimension multiplyDims(const TensorShapeProto& shape, int from, int upto_exclusive) {
  TensorShapeProto::Dimension dim;
  dim.set_dim_value(1);
  for (int i = from; i < upto_exclusive; ++i) {
    dim = dim * shape.dim(i);
  }
  return dim;
}

inline int32_t getTensorElementType(const TypeProto& type) {
  int32_t result = TensorProto::UNDEFINED;
  const auto value_case = type.value_case();
  if (value_case == TypeProto::kTensorType) {
    result = type.tensor_type().elem_type();
  } else if (value_case == TypeProto::kSparseTensorType) {
    result = type.sparse_tensor_type().elem_type();
  }
  return result;
}

inline void setTensorElementType(int32_t elem_type, TypeProto::ValueCase value_case, TypeProto& type) {
  if (value_case == TypeProto::kTensorType) {
    type.mutable_tensor_type()->set_elem_type(elem_type);
  } else if (value_case == TypeProto::kSparseTensorType) {
    type.mutable_sparse_tensor_type()->set_elem_type(elem_type);
  }
}

void propagateElemTypeWithValidation(const TypeProto* input_type, TypeProto* output_type);

void propagateElemTypeFromInputToOutput(InferenceContext& ctx, size_t inputIndex, size_t outputIndex);

void propagateElemTypeFromTensorInputToOutput(InferenceContext& ctx, size_t inputIndex, size_t outputIndex);

inline void propagateElemTypeFromDtypeToOutput(
    InferenceContext& ctx,
    const int data_type,
    size_t outputIndex,
    TypeProto::ValueCase expected_value_case) {
  const auto attribute_tensor_datatype = data_type;
  auto output_type = ctx.getOutputType(outputIndex);
  const auto output_value_case = output_type->value_case();
  if (output_value_case == TypeProto::VALUE_NOT_SET || output_value_case == expected_value_case) {
    setTensorElementType(attribute_tensor_datatype, expected_value_case, *output_type);
  } else {
    // This is not expected to happen
    fail_type_inference(
        "Output ", outputIndex, " expected to have: ", expected_value_case, " or UNDEFINED. Got: ", output_value_case);
  }
}

inline void propagateElemTypeFromDtypeToOutput(InferenceContext& ctx, const int data_type, size_t outputIndex) {
  propagateElemTypeFromDtypeToOutput(ctx, data_type, outputIndex, TypeProto::kTensorType);
}

inline void propagateElemTypeFromDtypeToOutput(InferenceContext& ctx, const AttributeProto* attr, size_t outputIndex) {
  int32_t data_type = TensorProto::UNDEFINED;
  TypeProto::ValueCase expected_value_case = TypeProto::VALUE_NOT_SET;
  const auto attr_type = attr->type();
  if (attr_type == AttributeProto::TENSOR) {
    if (attr->t().dims().size() != 1) {
      fail_type_inference("Attribute expected to have a one-dim tensor");
    }
    data_type = attr->t().data_type();
    expected_value_case = TypeProto::kTensorType;
  } else if (attr_type == AttributeProto::SPARSE_TENSOR) {
    if (attr->sparse_tensor().dims().size() != 1) {
      fail_type_inference("Attribute expected to have a one-dim sparse tensor");
    }
    data_type = attr->sparse_tensor().values().data_type();
    expected_value_case = TypeProto::kSparseTensorType;
  } else {
    fail_type_inference("Attribute expected to have tensor or sparse tensor type");
  }

  propagateElemTypeFromDtypeToOutput(ctx, data_type, outputIndex, expected_value_case);
}

inline bool hasShape(const TypeProto& type) {
  if (type.has_tensor_type()) {
    return type.tensor_type().has_shape();
  } else if (type.has_sparse_tensor_type()) {
    return type.sparse_tensor_type().has_shape();
  } else if (type.has_sequence_type() && type.sequence_type().has_elem_type()) {
    return hasShape(type.sequence_type().elem_type());
  } else if (type.has_optional_type() && type.optional_type().has_elem_type()) {
    return hasShape(type.optional_type().elem_type());
  }
  return false;
}

template <typename Context>
inline bool hasInputShape(Context& ctx, size_t n) {
  return ctx.getNumInputs() > static_cast<size_t>(n) && ctx.getInputType(n) && hasShape(*ctx.getInputType(n));
}

template <typename Context>
inline bool hasNInputShapes(Context& ctx, size_t n) {
  for (size_t i = 0; i < n; i++) {
    if (!hasInputShape(ctx, i)) {
      return false;
    }
  }
  return true;
}

inline const TensorShapeProto& getInputShape(InferenceContext& ctx, size_t n) {
  const auto* input_type = ctx.getInputType(n);
  const auto value_case = input_type->value_case();
  if (value_case != TypeProto::kTensorType && value_case != TypeProto::kSparseTensorType) {
    fail_type_inference("Attribute expected to have tensor or sparse tensor type");
  }
  if (value_case == TypeProto::kTensorType) {
    return input_type->tensor_type().shape();
  } else {
    return input_type->sparse_tensor_type().shape();
  }
}

inline const TensorShapeProto* getOptionalInputShape(InferenceContext& ctx, size_t n) {
  const auto* input_type = ctx.getInputType(n);

  if (input_type == nullptr) {
    return nullptr;
  }

  const auto value_case = input_type->value_case();
  if (value_case != TypeProto::kTensorType && value_case != TypeProto::kSparseTensorType) {
    fail_type_inference("Attribute expected to have tensor or sparse tensor type");
  }
  if (value_case == TypeProto::kTensorType) {
    return &input_type->tensor_type().shape();
  } else {
    return &input_type->sparse_tensor_type().shape();
  }
}

// Caller must make sure fromDimIndex is strictly less than shape.dim_size()
inline void appendSingleDimCopiedFromInputTypeToOutputType(
    InferenceContext& ctx,
    size_t inputIndex,
    size_t outputIndex,
    size_t fromDimIndex) {
  auto output_type = ctx.getOutputType(outputIndex);
  const auto output_value_case = output_type->value_case();
  auto input_type = ctx.getInputType(inputIndex);
  const auto input_value_case = input_type->value_case();
  if (output_value_case != input_value_case) {
    fail_type_inference(
        "Input: ",
        inputIndex,
        " type: ",
        input_value_case,
        " does not match type of output: ",
        outputIndex,
        "type: ",
        output_value_case);
  }
  if (TypeProto::kTensorType == input_value_case) {
    auto* dim = output_type->mutable_tensor_type()->mutable_shape()->add_dim();
    *dim = input_type->tensor_type().shape().dim(static_cast<int>(fromDimIndex));
  } else if (TypeProto::kSparseTensorType == input_value_case) {
    auto* dim = output_type->mutable_sparse_tensor_type()->mutable_shape()->add_dim();
    *dim = input_type->sparse_tensor_type().shape().dim(static_cast<int>(fromDimIndex));
  } else {
    fail_type_inference(
        "Input ", inputIndex, " and Output ", outputIndex, " expected to have tensor or sparse tensor type");
  }
}

inline void propagateShape(const TypeProto* from_type, TypeProto* to_type) {
  const auto from_type_case = from_type->value_case();
  const auto to_type_case = to_type->value_case();
  if (from_type_case != to_type_case) {
    fail_shape_inference("Mismatch between source and target type. Source=", from_type_case, " Target=", to_type_case);
  }

  if (TypeProto::kTensorType == from_type_case || TypeProto::kSparseTensorType == from_type_case) {
    // If input shape is "unknown", the corresponding should be "unknown" too.
    // The way to make output shape unknown is not to assign it any value.
    if (hasShape(*from_type)) {
      if (TypeProto::kTensorType == from_type_case) {
        *to_type->mutable_tensor_type()->mutable_shape() = from_type->tensor_type().shape();
      } else {
        *to_type->mutable_sparse_tensor_type()->mutable_shape() = from_type->sparse_tensor_type().shape();
      }
    }
  } else if (TypeProto::kSequenceType == from_type_case) {
    propagateShape(&from_type->sequence_type().elem_type(), to_type->mutable_sequence_type()->mutable_elem_type());
  } else if (TypeProto::kOptionalType == from_type_case) {
    propagateShape(&from_type->optional_type().elem_type(), to_type->mutable_optional_type()->mutable_elem_type());
  } else if (TypeProto::kMapType == from_type_case) {
    propagateShape(&from_type->map_type().value_type(), to_type->mutable_map_type()->mutable_value_type());
  } else {
    fail_shape_inference("Unsupported Source/Target type=", from_type_case);
  }
}

inline void propagateShapeFromInputToOutput(InferenceContext& ctx, size_t inputIndex, size_t outputIndex) {
  auto output_type = ctx.getOutputType(outputIndex);
  auto input_type = ctx.getInputType(inputIndex);

  propagateShape(input_type, output_type);
}

inline void propagateShapeAndTypeFromFirstInput(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }
  propagateShapeFromInputToOutput(ctx, 0, 0);
}

inline void
updateOutputElemType(InferenceContext& ctx, size_t outputIndex, int32_t elemType, TypeProto::ValueCase expected_type) {
  auto output_type = ctx.getOutputType(outputIndex);
  if (output_type == nullptr) {
    fail_type_inference("Output ", outputIndex, " is null");
  }
  if (output_type->value_case() == expected_type || output_type->value_case() == TypeProto::VALUE_NOT_SET) {
    setTensorElementType(elemType, expected_type, *output_type);
  } else {
    // This is not expected to happen
    fail_type_inference("Output ", outputIndex, " expected to have tensor or sparse tensor type: ", expected_type);
  }
}

inline void updateOutputElemType(InferenceContext& ctx, size_t outputIndex, int32_t elemType) {
  updateOutputElemType(ctx, outputIndex, elemType, TypeProto::kTensorType);
}

// Infer type of an output from the value of a specified attribute, which is
// expected to have a valid value representing a TensorProto_DataType.
inline void propagateElemTypeFromAttributeToOutput(
    InferenceContext& ctx,
    const std::string& attributeName,
    size_t outputIndex,
    TypeProto::ValueCase expected_type,
    TensorProto_DataType default_value = TensorProto::UNDEFINED) {
  auto attr_proto = ctx.getAttribute(attributeName);
  if (nullptr == attr_proto) { // attribute not present
    if (default_value != TensorProto::UNDEFINED) {
      updateOutputElemType(ctx, outputIndex, default_value, expected_type);
      return;
    } else {
      fail_type_inference("Value of attribute ", attributeName, " not specified");
    }
  }
  if (!attr_proto->has_i()) {
    fail_type_inference("Attribute ", attributeName, " should be of integer type and specify a type.");
  }
  auto attr_value = attr_proto->i();
  auto elem_type = static_cast<TensorProto_DataType>(attr_value);
  if (!TensorProto_DataType_IsValid(elem_type)) {
    fail_type_inference("Attribute ", attributeName, " does not specify a valid type.");
  }
  updateOutputElemType(ctx, outputIndex, elem_type, expected_type);
}

inline void propagateElemTypeFromAttributeToOutput(
    InferenceContext& ctx,
    const std::string& attributeName,
    size_t outputIndex,
    TensorProto_DataType default_value = TensorProto::UNDEFINED) {
  propagateElemTypeFromAttributeToOutput(ctx, attributeName, outputIndex, TypeProto::kTensorType, default_value);
}

inline TensorShapeProto* getTensorMutableShape(TypeProto::ValueCase value_case, TypeProto& type) {
  if (value_case == TypeProto::kTensorType) {
    return type.mutable_tensor_type()->mutable_shape();
  } else if (value_case == TypeProto::kSparseTensorType) {
    return type.mutable_tensor_type()->mutable_shape();
  }
  return nullptr;
}

inline TensorShapeProto*
getOutputShape(InferenceContext& ctx, size_t n, TypeProto::ValueCase default_type = TypeProto::kTensorType) {
  auto output_type = ctx.getOutputType(n);
  if (output_type == nullptr) {
    fail_type_inference("Output ", n, " expected to have tensor or sparse type");
  }
  const auto output_value_case = output_type->value_case();
  if (output_value_case == TypeProto::kTensorType || output_value_case == TypeProto::kSparseTensorType) {
    return getTensorMutableShape(output_value_case, *output_type);
  } else if (output_value_case == TypeProto::VALUE_NOT_SET) {
    return getTensorMutableShape(default_type, *output_type);
  } else {
    fail_type_inference("Output ", n, " expected to have tensor type");
  }
}

inline void appendDim(TensorShapeProto* shape, int64_t dim_value) {
  shape->add_dim()->set_dim_value(dim_value);
}

inline void updateOutputShape(
    InferenceContext& ctx,
    size_t outputIndex,
    const TensorShapeProto& shape,
    TypeProto::ValueCase default_type = TypeProto::kTensorType) {
  auto* output_shape = getOutputShape(ctx, outputIndex, default_type);
  *output_shape = shape;
}

inline void updateOutputShape(
    InferenceContext& ctx,
    size_t outputIndex,
    const TensorProto& tensorProto,
    TypeProto::ValueCase default_type = TypeProto::kTensorType) {
  auto* output_shape = getOutputShape(ctx, outputIndex, default_type);
  for (auto d : tensorProto.dims()) {
    auto* dim = output_shape->add_dim();
    dim->set_dim_value(d);
  }
}

inline void updateOutputShape(
    InferenceContext& ctx,
    size_t outputIndex,
    std::initializer_list<TensorShapeProto::Dimension> dims,
    TypeProto::ValueCase default_type = TypeProto::kTensorType) {
  auto* output_shape = getOutputShape(ctx, outputIndex, default_type);
  for (auto& d : dims) {
    auto* dim = output_shape->add_dim();
    *dim = d;
  }
}

// Get shape input by first checking initializer and then propagated symbolic data.
// If neither is available, try rank inference.
// When one of above succeeds, `true` is stored in `found`.
// Otherwise, `false` is stored, which means that returned TensorShapeProto does not make sense.
TensorShapeProto getShapeInput(InferenceContext& ctx, size_t input_index, bool& found);

// Infer shape of an output from the value of a specified attribute, which is
// expected to be a list of integers specifying a valid shape.
inline void propagateShapeFromAttributeToOutput(
    InferenceContext& ctx,
    const std::string& attributeName,
    size_t outputIndex,
    TypeProto::ValueCase default_type = TypeProto::kTensorType) {
  auto attr_proto = ctx.getAttribute(attributeName);
  if ((nullptr == attr_proto) || (!attr_proto->has_type()) ||
      (attr_proto->type() != AttributeProto_AttributeType_INTS)) {
    fail_shape_inference("Attribute ", attributeName, " should specify a shape");
  }
  auto& int_list = attr_proto->ints();
  TensorShapeProto shape;
  for (auto dim_size : int_list) {
    if (dim_size < 0) {
      fail_shape_inference("Negative values are not allowed in a shape specification");
    }
    shape.add_dim()->set_dim_value(dim_size);
  }

  updateOutputShape(ctx, outputIndex, shape, default_type);
}

inline void multidirectionalBroadcastShapeInference(
    const std::vector<const TensorShapeProto*>& shapes,
    TensorShapeProto& resultShape) {
  int result_shape_size = 0;
  // Get the result shape size.
  for (size_t i = 0; i < shapes.size(); ++i) {
    if (shapes[i]->dim_size() > result_shape_size) {
      result_shape_size = shapes[i]->dim_size();
    }
  }

  for (int i = 0; i < result_shape_size; ++i) {
    int64_t dim_value = 1;
    TensorShapeProto_Dimension symbolic_dim;
    int num_symbolic_dims = 0;
    for (size_t j = 0; j < shapes.size(); ++j) {
      if (i < result_shape_size - shapes[j]->dim_size()) {
        // Shape j will be filled with 1 at dimension i;
        continue;
      }

      auto dim_i_j = shapes[j]->dim(i - result_shape_size + shapes[j]->dim_size());
      if (dim_i_j.has_dim_value()) {
        if (dim_i_j.dim_value() != 1) {
          if (dim_value != dim_i_j.dim_value() && dim_value != 1) {
            fail_shape_inference("Incompatible dimensions");
          } else {
            dim_value = dim_i_j.dim_value();
          }
        }
      } else {
        if (num_symbolic_dims == 0) {
          symbolic_dim = dim_i_j;
          ++num_symbolic_dims;
        } else if (dim_i_j.dim_param() != symbolic_dim.dim_param()) {
          ++num_symbolic_dims;
        }
      }
    }

    if (dim_value != 1 || num_symbolic_dims == 0) {
      resultShape.add_dim()->set_dim_value(dim_value);
    } else if (num_symbolic_dims == 1) {
      *resultShape.add_dim() = symbolic_dim;
    } else {
      resultShape.add_dim();
    }
  }
}

inline void bidirectionalBroadcastShapeInference(
    const TensorShapeProto& shapeL,
    const TensorShapeProto& shapeR,
    TensorShapeProto& resultShape) {
  std::vector<const TensorShapeProto*> shapes;
  shapes.push_back(&shapeL);
  shapes.push_back(&shapeR);
  multidirectionalBroadcastShapeInference(shapes, resultShape);
}

/*
Merge the dimension information from two TensorShapeProto_Dimension instances.
Values are merged into target from source.
If target has no dimension information, copy from source.
If source has no dimension information, ignore source.
If both have dimension information:
 - Prefer values over params. If both have values, values must match.
 - Prefer target param over source param if mismatched.
Fail if there are mismatches in dimension values.
Currently, there is no way to refine/update dimension information for the
source from information available in the target.
*/
inline void mergeInDimensionInfo(
    const TensorShapeProto_Dimension& source_dim,
    TensorShapeProto_Dimension& target_dim,
    int dim_index) {
  // if source has value, merge into target
  // else if target has value, preserve it
  // else merge params
  if (source_dim.has_dim_value()) {
    auto source_value = source_dim.dim_value();
    if (target_dim.has_dim_value()) {
      auto target_value = target_dim.dim_value();
      if (target_value != source_value) {
        fail_shape_inference(
            "Can't merge shape info. "
            "Both source and target dimension have values but they differ. Source=",
            source_value,
            " Target=",
            target_value,
            " Dimension=",
            dim_index);
      }
    } else {
      target_dim.set_dim_value(source_value);
    }
  } else if (target_dim.has_dim_value()) {
    // if target has a value we preserve it so do nothing
  } else if (target_dim.has_dim_param()) {
    // prefer target param over source
  } else if (source_dim.has_dim_param()) {
    target_dim.set_dim_param(source_dim.dim_param());
  }
}

void mergeInShapeInfo(const TensorShapeProto& source_shape, TypeProto_Tensor& target_type);

void mergeInShapeInfo(const TensorShapeProto& source_shape, TypeProto_SparseTensor& target_type);

/*
Merge the shape information from two TypeProto_Tensor instances.
Values are merged into target from source.
If target has no shape information, copy from source.
If source has no shape information, ignore source.
If both have shape information:
- merge each TensorShapeProto_Dimension separately.
- Prefer values over params. If both have values, values must match.
- Prefer target param over source param if mismatched.
Fail if there are mismatches in number of dimensions or dimension values.
*/
void mergeInShapeInfo(const TypeProto_Tensor& source, TypeProto_Tensor& target);

void mergeInShapeInfo(const TypeProto_SparseTensor& source, TypeProto_SparseTensor& target);

// Return a copy of a type, with a specified dimension removed from its shape.
inline TypeProto RemoveIthDimensionFromShape(const TypeProto& proto, int removed_dim) {
  TypeProto t(proto);
  auto mutable_shape = t.mutable_tensor_type()->mutable_shape();
  mutable_shape->clear_dim();

  const auto& dims = proto.tensor_type().shape().dim();

  for (int j = 0, end = dims.size(); j < end; ++j) {
    if (j != removed_dim)
      (*mutable_shape->add_dim()) = dims.Get(j);
  }

  return t;
}

// Return a copy of a type, with specified number of dimensions removed from the
// beginning.
inline TypeProto RemoveDimensionsFromShape(const TypeProto& proto, int num_dimensions) {
  TypeProto t(proto);
  auto mutable_shape = t.mutable_tensor_type()->mutable_shape();
  mutable_shape->clear_dim();

  const auto& dims = proto.tensor_type().shape().dim();

  // skip first num_dimensions
  for (int j = num_dimensions, end = dims.size(); j < end; ++j) {
    (*mutable_shape->add_dim()) = dims.Get(j);
  }

  return t;
}

// copied from GSL:
// https://github.com/Microsoft/GSL/blob/main/include/gsl/gsl_util
template <class T, class U>
static constexpr T narrow_cast(U&& u) noexcept {
  return static_cast<T>(std::forward<U>(u));
}

inline void checkInputRank(InferenceContext& ctx, size_t input_index, int expected_rank) {
  // We check the rank only if a rank is known for the input:
  if (hasInputShape(ctx, input_index)) {
    auto rank = getInputShape(ctx, input_index).dim_size();
    if (rank != expected_rank) {
      fail_shape_inference("Input ", input_index, " expected to have rank ", expected_rank, " but has rank ", rank);
    }
  }
}

// Unification (between dimensions and/or shapes) is at the heart of
// shape-inference. The current inference algorithm can check input
// shapes/dimensions of a node and update the output shapes/dimensions. It
// cannot currently update input shapes and dimensions (even though in some
// contexts this inference is possible). Hence, we have the variants below to
// support "const" and "mutable" dimensions/shapes in unification.

inline void checkDimEquality(int64_t value1, int64_t value2) {
  if (value1 != value2) {
    fail_shape_inference("Dimension mismatch in unification between ", value1, " and ", value2);
  }
}

inline void unifyDim(const Dim& dim1, const Dim& dim2) {
  if (dim1.has_dim_value() && dim2.has_dim_value())
    checkDimEquality(dim1.dim_value(), dim2.dim_value());
}

// TODO: The functionality of unifyDim is similar to that of
// mergeInDimensionInfo. However, the error messages are different. Leaving this
// duplication in-place to preserve error message content.
inline void unifyDim(const Dim& source_dim, Dim& target_dim) {
  if (source_dim.has_dim_value()) {
    auto source_value = source_dim.dim_value();
    if (target_dim.has_dim_value()) {
      auto target_value = target_dim.dim_value();
      checkDimEquality(source_value, target_value);
    } else {
      target_dim.set_dim_value(source_value);
    }
  } else if (target_dim.has_dim_value()) {
    // if target has a value we preserve it.
    // we cannot set source dim value.
  } else if (target_dim.has_dim_param()) {
    // prefer target param over source
    // we cannot currently unify the dim_params
  } else if (source_dim.has_dim_param()) {
    target_dim.set_dim_param(source_dim.dim_param());
  }
}

inline void unifyInputDim(InferenceContext& ctx, size_t input_index, int dim_index, Dim& dim) {
  // We unify the dimensions only if it is available for specified input:
  if (hasInputShape(ctx, input_index)) {
    auto& input_shape = getInputShape(ctx, input_index);
    // This shape is expected to have rank > dim_index:
    if (input_shape.dim_size() <= dim_index) {
      fail_shape_inference(
          "Input ", input_index, " expected to have rank >", dim_index, " but has rank ", input_shape.dim_size());
    }
    const Dim& input_dim = input_shape.dim(dim_index);
    // Now, unify dim and input_dim:
    unifyDim(input_dim, dim);
  }
}

// unifyDim: unifies a dimension with a constant value. If the dimension
// already has a value, we check for equality of new value with old value.
inline void unifyDim(Dim& dim, int64_t value) {
  if (dim.has_dim_value()) {
    checkDimEquality(dim.dim_value(), value);
  } else
    dim.set_dim_value(value);
}

// target-shape = Union (target-shape, source_shape)
// Example 1: same rank, different dimensions
//    input1 shape: (2, 3, 4, 'x')
//    input2 shape: (2, 'y', 5, 'x')
//    output shape: (2, None, None, 'x')
// Example 2: different rank
//    input1 shape: (2, 3, 4, 'x')
//    input2 shape: (2, 3, 4)
//    output shape: None
void UnionShapeInfo(const TensorShapeProto& source_shape, TypeProto_Tensor& target_type);

void UnionShapeInfo(const TensorShapeProto& source_shape, TypeProto_SparseTensor& target_type);

// target-type = Union (target-type, source-type)
// target and source are required to have the same type.
// Example 1: same tensor type, different shape
//    source: tensor elem_type: int64, shape: (2, 3, 4, 'x')
//    target: tensor elem_type: int64, shape: (2, 'y', 5, 'x')
//    output: tensor elem_type: int64, shape: (2, None, None, 'x')
// Example 2: same sequence type, different shape
//    source: sequence of tensor, elem_type: float, shape: (2, 3, 4)
//    target: sequence of tensor, elem_type: float, shape: None
//    output: sequence of tensor, elem_type: float, shape: None
void UnionTypeInfo(const TypeProto& source_type, TypeProto& target_type);

// adjustNegativeAxes: Negative axes values are translated to the right axis in the positive range
template <typename Axes>
void adjustNegativeAxes(Axes& axes, int rank) {
  std::transform(
      axes.begin(), axes.end(), axes.begin(), [&](int64_t axis) -> int64_t { return axis < 0 ? axis + rank : axis; });
}

// checkAxesRange: Checks that values are within the range [-rank, rank)
template <typename Axes>
void checkAxesRange(Axes& axes, int rank) {
  for (auto axis : axes) {
    if (axis < -rank || axis > (rank - 1))
      fail_shape_inference("Unexpected axis value: ", axis, ". Expected range [", -rank, ", ", rank, ")");
  }
}

// checkDuplicateAxes: Check that there are no duplicated axes
template <typename Axes>
void checkDuplicateAxes(Axes& axes, int rank) {
  std::vector<bool> tmp(rank, false);
  for (auto axis : axes) {
    int actual_axis = axis < 0 ? axis + rank : axis;
    if (tmp[actual_axis])
      fail_shape_inference("Axis ", axis, " is referred to more than once.");
    tmp[actual_axis] = true;
  }
}

} // namespace ONNX_NAMESPACE
