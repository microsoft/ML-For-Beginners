/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "shape_inference.h"
#include "onnx/defs/tensor_proto_util.h"

namespace ONNX_NAMESPACE {

// Note: for all methods below for propagating type or shape, callers are
// responsible to handle optional inputs/outputs and ensure that the specified
// index value is less than NumInputs/NumOutputs.
// Supports mixed tensor and sparse tensor
void propagateElemTypeFromTensorInputToOutput(InferenceContext& ctx, size_t inputIndex, size_t outputIndex) {
  auto input_type = ctx.getInputType(inputIndex);
  if (nullptr == input_type) {
    fail_type_inference("Input type was null");
  }

  const auto input_value_case = input_type->value_case();
  if (input_value_case != TypeProto::kTensorType && input_value_case != TypeProto::kSparseTensorType) {
    fail_type_inference(
        "Input ", inputIndex, " expected to have tensor or sparse tensor type. Got: ", input_value_case);
  }

  const auto input_elem_type = getTensorElementType(*input_type);
  if (input_elem_type == TensorProto::UNDEFINED) {
    fail_type_inference("Element type of input ", inputIndex, " unknown");
  }
  auto output_type = ctx.getOutputType(outputIndex);
  const auto output_value_case = output_type->value_case();
  if (output_value_case == TypeProto::kTensorType || output_value_case == TypeProto::kSparseTensorType) {
    setTensorElementType(input_elem_type, output_value_case, *output_type);
  } else if (output_value_case == TypeProto::VALUE_NOT_SET) {
    // Assume output will have the same type
    setTensorElementType(input_elem_type, input_value_case, *output_type);
  } else {
    // This is not expected to happen
    fail_type_inference(
        "Output ", outputIndex, " expected to have tensor or sparse tensor type. Got: ", output_value_case);
  }
}

void propagateElemTypeFromSequenceInputToOutput(InferenceContext& ctx, size_t inputIndex, size_t outputIndex) {
  auto input_type = ctx.getInputType(inputIndex);
  if (nullptr == input_type || input_type->value_case() != TypeProto::kSequenceType) {
    fail_type_inference("Input ", inputIndex, " expected to have sequence type");
  }
  auto input_seq_type = input_type->sequence_type();
  if (!input_seq_type.has_elem_type()) {
    fail_type_inference("Element type of sequence input ", inputIndex, " unknown");
  }

  auto output_type = ctx.getOutputType(outputIndex);
  output_type->mutable_sequence_type()->mutable_elem_type()->CopyFrom(input_seq_type.elem_type());
}

void propagateElemTypeFromOptionalInputToOutput(InferenceContext& ctx, size_t inputIndex, size_t outputIndex) {
  auto input_type = ctx.getInputType(inputIndex);
  if (nullptr == input_type || input_type->value_case() != TypeProto::kOptionalType) {
    fail_type_inference("Input ", inputIndex, " expected to have optional type");
  }
  auto input_opt_type = input_type->optional_type();
  if (!input_opt_type.has_elem_type()) {
    fail_type_inference("Element type of optional input ", inputIndex, " unknown");
  }

  auto output_type = ctx.getOutputType(outputIndex);
  output_type->mutable_optional_type()->mutable_elem_type()->CopyFrom(input_opt_type.elem_type());
}

void propagateElemTypeFromMapInputToOutput(InferenceContext& ctx, size_t inputIndex, size_t outputIndex) {
  auto input_type = ctx.getInputType(inputIndex);
  if (nullptr == input_type || input_type->value_case() != TypeProto::kMapType) {
    fail_type_inference("Input ", inputIndex, " expected to have map type");
  }
  auto input_map_type = input_type->map_type();
  if (!input_map_type.has_key_type()) {
    fail_type_inference("Key type of map input ", inputIndex, " unknown");
  }
  if (!input_map_type.has_value_type()) {
    fail_type_inference("Value type of map input ", inputIndex, " unknown");
  }

  auto output_type = ctx.getOutputType(outputIndex);
  output_type->mutable_map_type()->set_key_type(input_map_type.key_type());
  output_type->mutable_map_type()->mutable_value_type()->CopyFrom(input_map_type.value_type());
}

void propagateElemTypeFromInputToOutput(InferenceContext& ctx, size_t inputIndex, size_t outputIndex) {
  auto input_type = ctx.getInputType(inputIndex);
  if (nullptr == input_type) {
    fail_type_inference("Input ", inputIndex, " expected to have type but instead is null");
  }
  const auto input_value_case = input_type->value_case();
  if (input_value_case == TypeProto::kTensorType || input_value_case == TypeProto::kSparseTensorType) {
    propagateElemTypeFromTensorInputToOutput(ctx, inputIndex, outputIndex);
  } else if (input_value_case == TypeProto::kSequenceType) {
    propagateElemTypeFromSequenceInputToOutput(ctx, inputIndex, outputIndex);
  } else if (input_value_case == TypeProto::kOptionalType) {
    propagateElemTypeFromOptionalInputToOutput(ctx, inputIndex, outputIndex);
  } else if (input_value_case == TypeProto::kMapType) {
    propagateElemTypeFromMapInputToOutput(ctx, inputIndex, outputIndex);
  }
}

/*
Merge shape information from a source shape into a target shape.
* merges each TensorShapeProto_Dimension separately.
* prefer values over params.
* If both have values, values must match.
* prefer target param over source param if mismatched.
* Fail if there are mismatches in number of dimensions or dimension values.
*/
void mergeInShapeInfo(const TensorShapeProto& source, TensorShapeProto& target) {
  auto num_source_dims = source.dim_size();
  auto num_target_dims = target.dim_size();
  if (num_source_dims != num_target_dims) {
    fail_shape_inference(
        "Mismatch between number of source and target dimensions. Source=",
        num_source_dims,
        " Target=",
        num_target_dims);
  }

  auto& source_dims = source.dim();
  auto* target_dims = target.mutable_dim();

  for (int i = 0, end = source_dims.size(); i < end; ++i) {
    auto& source_dim = source_dims.Get(i);
    auto& target_dim = *target_dims->Mutable(i);
    mergeInDimensionInfo(source_dim, target_dim, i);
  }
}

void mergeInShapeInfo(const TensorShapeProto& source_shape, TypeProto_Tensor& target_type) {
  if (target_type.has_shape()) {
    // merge with existing info.
    mergeInShapeInfo(source_shape, *target_type.mutable_shape());
  } else {
    // copy to target
    (*target_type.mutable_shape()) = source_shape;
  }
}

void mergeInShapeInfo(const TensorShapeProto& source_shape, TypeProto_SparseTensor& target_type) {
  if (target_type.has_shape()) {
    // merge with existing info.
    mergeInShapeInfo(source_shape, *target_type.mutable_shape());
  } else {
    // copy to target
    (*target_type.mutable_shape()) = source_shape;
  }
}

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
void mergeInShapeInfo(const TypeProto_Tensor& source, TypeProto_Tensor& target) {
  if (source.has_shape())
    mergeInShapeInfo(source.shape(), target);
}

void mergeInShapeInfo(const TypeProto_SparseTensor& source, TypeProto_SparseTensor& target) {
  if (source.has_shape())
    mergeInShapeInfo(source.shape(), target);
}

/// <summary>
/// Utility function for UnionShapeInfoForTensor.
/// Both shapes must be of the same rank
/// </summary>
/// <param name="source_shape"></param>
/// <param name="target_shape">destination shape</param>
void UnionShapeInfo(const TensorShapeProto& source_shape, TensorShapeProto& target_shape) {
  auto source_rank = source_shape.dim_size();
  for (int i = 0; i < source_rank; ++i) {
    const auto source_dim = source_shape.dim(i);
    const auto target_dim = target_shape.dim(i);
    bool is_dims_conflict = [&]() {
      if (source_dim.has_dim_value()) {
        if (target_dim.has_dim_value() && target_dim.dim_value() == source_dim.dim_value()) {
          return false;
        }
        return true;
      }

      if (source_dim.has_dim_param()) {
        if (target_dim.has_dim_param() && target_dim.dim_param() == source_dim.dim_param()) {
          return false;
        }
        return true;
      }

      return (target_dim.has_dim_value() || target_dim.has_dim_param());
    }();
    if (is_dims_conflict && (target_dim.has_dim_value() || target_dim.has_dim_param())) {
      auto dim = target_shape.mutable_dim(i);
      dim->clear_dim_value();
      dim->clear_dim_param();
    }
  }
}

template <typename TENSOR_TYPE>
void UnionShapeInfoForTensor(const TensorShapeProto& source_shape, TENSOR_TYPE& target_type) {
  if (target_type.has_shape()) {
    TensorShapeProto* target_shape = target_type.mutable_shape();

    auto source_rank = source_shape.dim_size();
    auto target_rank = target_shape->dim_size();
    if (source_rank != target_rank) {
      target_type.clear_shape();
      return;
    }

    UnionShapeInfo(source_shape, *target_shape);
  }
}

void UnionShapeInfo(const TensorShapeProto& source_shape, TypeProto_Tensor& target_type) {
  UnionShapeInfoForTensor(source_shape, target_type);
}

void UnionShapeInfo(const TypeProto_Tensor& source_type, TypeProto_Tensor& target_type) {
  // The union of a tensor of unknown rank and a tensor of known rank is a tensor of unknown rank.
  // Hence, if the source_type had unknown rank, we clear the shape of the target_type.
  // Otherwise, UnionShapeInfoForTensor handles the rest.
  if (source_type.has_shape()) {
    UnionShapeInfoForTensor(source_type.shape(), target_type);
  } else {
    target_type.clear_shape();
  }
}

void UnionShapeInfo(const TypeProto_SparseTensor& source_type, TypeProto_SparseTensor& target_type) {
  // The union of a tensor of unknown rank and a tensor of known rank is a tensor of unknown rank.
  // Hence, if the source_type had unknown rank, we clear the shape of the target_type.
  // Otherwise, UnionShapeInfoForTensor handles the rest.
  if (source_type.has_shape()) {
    UnionShapeInfoForTensor(source_type.shape(), target_type);
  } else {
    target_type.clear_shape();
  }
}

void UnionShapeInfo(const TensorShapeProto& source_shape, TypeProto_SparseTensor& target_type) {
  UnionShapeInfoForTensor(source_shape, target_type);
}

void UnionTypeInfo(const TypeProto& source_type, TypeProto& target_type) {
  if (source_type.value_case() != target_type.value_case()) {
    fail_type_inference("Mismatched type:", " source=", source_type.value_case(), " target=", target_type.value_case());
  }

  const auto target_case = target_type.value_case();
  if (target_case == TypeProto::ValueCase::kTensorType) {
    auto source_elem_type = source_type.tensor_type().elem_type();
    auto target_elem_type = target_type.tensor_type().elem_type();

    if (source_elem_type != target_elem_type) {
      fail_type_inference(
          "Mismatched tensor element type:", " source=", source_elem_type, " target=", target_elem_type);
    }

    UnionShapeInfo(source_type.tensor_type(), *target_type.mutable_tensor_type());
  } else if (target_case == TypeProto::ValueCase::kSparseTensorType) {
    auto source_elem_type = source_type.sparse_tensor_type().elem_type();
    auto target_elem_type = target_type.sparse_tensor_type().elem_type();
    if (source_elem_type != target_elem_type) {
      fail_type_inference(
          "Mismatched sparse tensor element type:", " source=", source_elem_type, " target=", target_elem_type);
    }
    UnionShapeInfo(source_type.sparse_tensor_type(), *target_type.mutable_sparse_tensor_type());
  } else if (target_case == TypeProto::ValueCase::kSequenceType) {
    if (!source_type.sequence_type().has_elem_type()) {
      fail_type_inference("source sequence type missing element type.");
    }
    if (!target_type.sequence_type().has_elem_type()) {
      fail_type_inference("target sequence type missing element type.");
    }
    UnionTypeInfo(source_type.sequence_type().elem_type(), *target_type.mutable_sequence_type()->mutable_elem_type());
  } else if (target_case == TypeProto::ValueCase::kOptionalType) {
    if (!source_type.optional_type().has_elem_type()) {
      fail_type_inference("source optional type missing element type.");
    }
    if (!target_type.optional_type().has_elem_type()) {
      fail_type_inference("target optional type missing element type.");
    }
    UnionTypeInfo(source_type.optional_type().elem_type(), *target_type.mutable_optional_type()->mutable_elem_type());
  } else if (target_case == TypeProto::ValueCase::kMapType) {
    if (!source_type.map_type().has_key_type()) {
      fail_type_inference("source map type missing key type.");
    }
    if (!target_type.map_type().has_key_type()) {
      fail_type_inference("target map type missing key type.");
    }
    auto source_key_type = source_type.map_type().key_type();
    auto target_key_type = target_type.map_type().key_type();
    if (source_key_type != target_key_type) {
      fail_type_inference(
          "Mismatched map tensor key type:",
          " source=",
          Utils::DataTypeUtils::ToDataTypeString(source_key_type),
          " target=",
          Utils::DataTypeUtils::ToDataTypeString(target_key_type));
    }

    if (!source_type.map_type().has_value_type()) {
      fail_type_inference("source map type missing value type.");
    }
    if (!target_type.map_type().has_value_type()) {
      fail_type_inference("target map type missing value type.");
    }
    UnionTypeInfo(source_type.map_type().value_type(), *target_type.mutable_map_type()->mutable_value_type());
  }
}

// Supports both Tensor and SparseTensor
// This does not fail if input_type is Tensor and output type is SparseTensor
// or the other way around. This is to support mixed cases when an op receives
// sparse input and outputs dense or vice-versa.
// If the output value_case is not set, then
// the input value_case is propagated.
void propagateTensorElemTypeWithValidation(const TypeProto* input_type, TypeProto* output_type) {
  if (nullptr == input_type) {
    fail_type_inference("Input type was null");
  }

  int32_t input_elem_type = TensorProto::UNDEFINED;
  const auto input_value_case = input_type->value_case();
  if (input_value_case == TypeProto::kTensorType || input_value_case == TypeProto::kSparseTensorType) {
    input_elem_type = getTensorElementType(*input_type);
    if (input_elem_type == TensorProto::UNDEFINED) {
      fail_type_inference("Element type of tensor or sparse tensor input was unknown");
    }
  } else {
    fail_type_inference("Input was expected to have tensor or sparse tensor type. Got ", input_value_case);
  }

  const auto output_value_case = output_type->value_case();
  if (output_value_case == TypeProto::VALUE_NOT_SET) {
    setTensorElementType(input_elem_type, input_value_case, *output_type);
  } else if (output_value_case == TypeProto::kTensorType || output_value_case == TypeProto::kSparseTensorType) {
    const auto output_elem_type = getTensorElementType(*output_type);
    if (output_elem_type != TensorProto::UNDEFINED) {
      if (input_elem_type != output_elem_type) {
        fail_type_inference(
            "Input element type of ", input_elem_type, " does not match existing output type of ", output_elem_type);
      }
    } else {
      setTensorElementType(input_elem_type, output_value_case, *output_type);
    }
  } else {
    // This is not expected to happen
    fail_type_inference("Output was expected to have tensor type. Got ", output_value_case);
  }
}

void propagateSequenceElemTypeWithValidation(const TypeProto* input_type, TypeProto* output_type) {
  if (nullptr == input_type) {
    fail_type_inference("Input type was null");
  }

  if (input_type->value_case() != TypeProto::kSequenceType) {
    fail_type_inference("Input was expected to have sequence type. Got ", input_type->value_case());
  }

  auto input_seq_type = input_type->sequence_type();

  if (input_seq_type.has_elem_type()) {
    propagateElemTypeWithValidation(
        &input_seq_type.elem_type(), output_type->mutable_sequence_type()->mutable_elem_type());
  } else {
    fail_type_inference("Element type of sequence input was unknown");
  }
}

void propagateOptionalElemTypeWithValidation(const TypeProto* input_type, TypeProto* output_type) {
  if (nullptr == input_type) {
    fail_type_inference("Input type was null");
  }

  if (input_type->value_case() != TypeProto::kOptionalType) {
    fail_type_inference("Input was expected to have optional type. Got ", input_type->value_case());
  }

  auto input_opt_type = input_type->optional_type();

  if (input_opt_type.has_elem_type()) {
    propagateElemTypeWithValidation(
        &input_opt_type.elem_type(), output_type->mutable_optional_type()->mutable_elem_type());
  } else {
    fail_type_inference("Element type of optional input was unknown");
  }
}

void propagateMapElemTypeWithValidation(const TypeProto* input_type, TypeProto* output_type) {
  if (nullptr == input_type) {
    fail_type_inference("Input type was null");
  }

  if (input_type->value_case() != TypeProto::kMapType) {
    fail_type_inference("Input was expected to have map type. Got ", input_type->value_case());
  }

  auto input_map_type = input_type->map_type();

  if (!input_map_type.has_key_type()) {
    fail_type_inference("Key type of map input was unknown");
  }
  if (!input_map_type.has_value_type()) {
    fail_type_inference("Value type of map input was unknown");
  }
  output_type->mutable_map_type()->set_key_type(input_map_type.key_type());
  propagateElemTypeWithValidation(&input_map_type.value_type(), output_type->mutable_map_type()->mutable_value_type());
}

// propagate the element type from an input type to an output type.
// if an existing output element type exists, validate it matches.
void propagateElemTypeWithValidation(const TypeProto* input_type, TypeProto* output_type) {
  if (nullptr == input_type) {
    fail_type_inference("Input type was null");
  }

  const auto input_value_case = input_type->value_case();
  if (input_value_case == TypeProto::kTensorType || input_value_case == TypeProto::kSparseTensorType) {
    propagateTensorElemTypeWithValidation(input_type, output_type);
  } else if (input_value_case == TypeProto::kSequenceType) {
    propagateSequenceElemTypeWithValidation(input_type, output_type);
  } else if (input_value_case == TypeProto::kOptionalType) {
    propagateOptionalElemTypeWithValidation(input_type, output_type);
  } else if (input_value_case == TypeProto::kMapType) {
    propagateMapElemTypeWithValidation(input_type, output_type);
  } else {
    fail_type_inference(
        "Input was expected to have either tensor, sequence, optional or map type. Got ", input_value_case);
  }
}

TensorShapeProto getShapeInput(InferenceContext& ctx, size_t input_index, bool& found) {
  TensorShapeProto shape_input;

  // First, check initializer.
  const TensorProto* shape_initializer = ctx.getInputData(input_index);
  if (shape_initializer) {
    const std::vector<int64_t>& shape_data = ParseData<int64_t>(shape_initializer);
    for (const int64_t& e : shape_data) {
      shape_input.add_dim()->set_dim_value(e);
    }
    found = true;
    return shape_input;
  }

  // Then, check symbolic input.
  const TensorShapeProto* symbolic_input = ctx.getSymbolicInput(input_index);
  if (symbolic_input) {
    shape_input.CopyFrom(*symbolic_input);
    found = true;
    return shape_input;
  }

  // Try rank inference.
  if (hasInputShape(ctx, input_index)) {
    const TensorShapeProto& shape_input_shape = getInputShape(ctx, input_index);
    if (shape_input_shape.dim_size() != 1) {
      fail_shape_inference("shape input must be 1D tensor");
    }
    if (shape_input_shape.dim(0).has_dim_value()) {
      // Attempt rank inference using shape of shape input
      int64_t dim_value = shape_input_shape.dim(0).dim_value();
      for (int64_t i = 0; i < dim_value; ++i) {
        shape_input.add_dim();
      }
      found = true;
      return shape_input;
    }
  }

  // Shape input was not found.
  found = false;
  return shape_input;
}

} // namespace ONNX_NAMESPACE
