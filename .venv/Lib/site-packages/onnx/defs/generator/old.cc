/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <functional>
#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {
static const char* Constant_ver13_doc = R"DOC(
This operator produces a constant tensor. Exactly one of the provided attributes, either value, sparse_value,
or value_* must be specified.
)DOC";

void ConstantInference(InferenceContext& ctx) {
  auto* value = ctx.getAttribute("value");
  auto* sparse_value = ctx.getAttribute("sparse_value");
  auto* value_int = ctx.getAttribute("value_int");
  auto* value_ints = ctx.getAttribute("value_ints");
  auto* value_float = ctx.getAttribute("value_float");
  auto* value_floats = ctx.getAttribute("value_floats");
  auto* value_string = ctx.getAttribute("value_string");
  auto* value_strings = ctx.getAttribute("value_strings");

  std::vector<bool> non_null_attr = {
      (nullptr != value),
      (nullptr != sparse_value),
      (nullptr != value_int),
      (nullptr != value_ints),
      (nullptr != value_float),
      (nullptr != value_floats),
      (nullptr != value_string),
      (nullptr != value_strings)};
  if (std::count(non_null_attr.begin(), non_null_attr.end(), true) != 1) {
    fail_shape_inference(
        "One and only one of the attributes 'value', 'value_*' or 'sparse_value' must be specified for a Constant node.");
  }

  if (nullptr != value) {
    // OpSchema::Verify check ensures that the attribute value has_t():
    const TensorProto& tensor_proto = value->t();
    updateOutputElemType(ctx, 0, tensor_proto.data_type());
    updateOutputShape(ctx, 0, tensor_proto);
    return;
  }

  if (nullptr != value_int) {
    // OpSchema::Verify check ensures that the attribute value has_i():
    if (!value_int->has_i()) {
      fail_shape_inference("Attribute 'value_int' expect an integer.")
    }
    updateOutputElemType(ctx, 0, TensorProto::INT64);
    updateOutputShape(ctx, 0, TensorShapeProto());
    return;
  }

  if (nullptr != value_ints) {
    // OpSchema::Verify check ensures that the attribute value has ints.
    if (value_ints->ints_size() < 1) {
      fail_shape_inference("Attribute 'value_ints' expect a list of integers.");
    }
    updateOutputElemType(ctx, 0, TensorProto::INT64);
    appendDim(getOutputShape(ctx, 0), value_ints->ints_size());
    return;
  }

  if (nullptr != value_float) {
    // OpSchema::Verify check ensures that the attribute value has_i():
    if (!value_float->has_f()) {
      fail_shape_inference("Attribute 'value_float' expect a float.");
    }
    updateOutputElemType(ctx, 0, TensorProto::FLOAT);
    updateOutputShape(ctx, 0, TensorShapeProto());
    return;
  }

  if (nullptr != value_floats) {
    // OpSchema::Verify check ensures that the attribute value has ints.
    if (value_floats->floats_size() < 1) {
      fail_shape_inference("Attribute 'value_floats' expect a list of floats.");
    }
    updateOutputElemType(ctx, 0, TensorProto::FLOAT);
    appendDim(getOutputShape(ctx, 0), value_floats->floats_size());
    return;
  }

  if (nullptr != value_string) {
    // OpSchema::Verify check ensures that the attribute value has_i():
    if (!value_string->has_s()) {
      fail_shape_inference("Attribute 'value_string' expect a string.");
    }
    updateOutputElemType(ctx, 0, TensorProto::STRING);
    updateOutputShape(ctx, 0, TensorShapeProto());
    return;
  }

  if (nullptr != value_strings) {
    // OpSchema::Verify check ensures that the attribute value has ints.
    if (value_strings->strings_size() < 1) {
      fail_shape_inference("Attribute 'value_strings' expect a list of strings.");
    }
    updateOutputElemType(ctx, 0, TensorProto::STRING);
    appendDim(getOutputShape(ctx, 0), value_strings->strings_size());
    return;
  }

  if (nullptr != sparse_value) {
    // OpSchema::Verify check ensures that the attribute value
    // has_sparse_tensor():
    const SparseTensorProto& sparse = sparse_value->sparse_tensor();
    // checker.cc::check_sparse_tensor checks that the sparse-value is
    // well-formed
    updateOutputElemType(ctx, 0, sparse.values().data_type());
    auto* output_shape = getOutputShape(ctx, 0);
    for (int i = 0; i < sparse.dims_size(); ++i)
      appendDim(output_shape, sparse.dims(i));
    return;
  }

  fail_shape_inference(
      "TypeAndShapeInferenceFunction implementation incomplete: "
      "this line should never be reached.");
}

ONNX_OPERATOR_SET_SCHEMA(
    Constant,
    13,
    OpSchema()
        .SetDoc(Constant_ver13_doc)
        .Attr("value", "The value for the elements of the output tensor.", AttributeProto::TENSOR, false)
        .Attr(
            "sparse_value",
            "The value for the elements of the output tensor in sparse format.",
            AttributeProto::SPARSE_TENSOR,
            false)
        .Attr(
            "value_int",
            "The value for the sole element for the scalar, int64, output tensor.",
            AttributeProto::INT,
            false)
        .Attr(
            "value_ints",
            "The values for the elements for the 1D, int64, output tensor.",
            AttributeProto::INTS,
            false)
        .Attr(
            "value_float",
            "The value for the sole element for the scalar, float32, output tensor.",
            AttributeProto::FLOAT,
            false)
        .Attr(
            "value_floats",
            "The values for the elements for the 1D, float32, output tensor.",
            AttributeProto::FLOATS,
            false)
        .Attr(
            "value_string",
            "The value for the sole element for the scalar, UTF-8 string, output tensor.",
            AttributeProto::STRING,
            false)
        .Attr(
            "value_strings",
            "The values for the elements for the 1D, UTF-8 string, output tensor.",
            AttributeProto::STRINGS,
            false)
        .Output(0, "output", "Output tensor containing the same value of the provided tensor.", "T")
        .TypeConstraint("T", OpSchema::all_tensor_types_ir4(), "Constrain input and output types to all tensor types.")
        .TypeAndShapeInferenceFunction(ConstantInference));

static const char* Constant_ver12_doc = R"DOC(
This operator produces a constant tensor. Exactly one of the provided attributes, either value, sparse_value,
or value_* must be specified.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Constant,
    12,
    OpSchema()
        .SetDoc(Constant_ver12_doc)
        .Attr("value", "The value for the elements of the output tensor.", AttributeProto::TENSOR, false)
        .Attr(
            "sparse_value",
            "The value for the elements of the output tensor in sparse format.",
            AttributeProto::SPARSE_TENSOR,
            false)
        .Attr(
            "value_int",
            "The value for the sole element for the scalar, int64, output tensor.",
            AttributeProto::INT,
            false)
        .Attr(
            "value_ints",
            "The values for the elements for the 1D, int64, output tensor.",
            AttributeProto::INTS,
            false)
        .Attr(
            "value_float",
            "The value for the sole element for the scalar, float32, output tensor.",
            AttributeProto::FLOAT,
            false)
        .Attr(
            "value_floats",
            "The values for the elements for the 1D, float32, output tensor.",
            AttributeProto::FLOATS,
            false)
        .Attr(
            "value_string",
            "The value for the sole element for the scalar, UTF-8 string, output tensor.",
            AttributeProto::STRING,
            false)
        .Attr(
            "value_strings",
            "The values for the elements for the 1D, UTF-8 string, output tensor.",
            AttributeProto::STRINGS,
            false)
        .Output(0, "output", "Output tensor containing the same value of the provided tensor.", "T")
        .TypeConstraint("T", OpSchema::all_tensor_types(), "Constrain input and output types to all tensor types.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          auto* value = ctx.getAttribute("value");
          auto* sparse_value = ctx.getAttribute("sparse_value");
          auto* value_int = ctx.getAttribute("value_int");
          auto* value_ints = ctx.getAttribute("value_ints");
          auto* value_float = ctx.getAttribute("value_float");
          auto* value_floats = ctx.getAttribute("value_floats");
          auto* value_string = ctx.getAttribute("value_string");
          auto* value_strings = ctx.getAttribute("value_strings");

          std::vector<bool> non_null_attr = {
              (nullptr != value),
              (nullptr != sparse_value),
              (nullptr != value_int),
              (nullptr != value_ints),
              (nullptr != value_float),
              (nullptr != value_floats),
              (nullptr != value_string),
              (nullptr != value_strings)};
          if (std::count(non_null_attr.begin(), non_null_attr.end(), true) != 1) {
            fail_shape_inference(
                "One and only one of the attributes 'value', 'value_*' or 'sparse_value' must be specified for a Constant node.");
          }

          if (nullptr != value) {
            // OpSchema::Verify check ensures that the attribute value has_t():
            const TensorProto& tensor_proto = value->t();
            updateOutputElemType(ctx, 0, tensor_proto.data_type());
            updateOutputShape(ctx, 0, tensor_proto);
            return;
          }

          if (nullptr != value_int) {
            // OpSchema::Verify check ensures that the attribute value has_i():
            if (!value_int->has_i()) {
              fail_shape_inference("Attribute 'value_int' expect an integer.");
            }
            updateOutputElemType(ctx, 0, TensorProto::INT64);
            updateOutputShape(ctx, 0, TensorShapeProto());
            return;
          }

          if (nullptr != value_ints) {
            // OpSchema::Verify check ensures that the attribute value has ints.
            if (value_ints->ints_size() < 1) {
              fail_shape_inference("Attribute 'value_ints' expect a list of integers.");
            }
            updateOutputElemType(ctx, 0, TensorProto::INT64);
            appendDim(getOutputShape(ctx, 0), value_ints->ints_size());
            return;
          }

          if (nullptr != value_float) {
            // OpSchema::Verify check ensures that the attribute value has_i():
            if (!value_float->has_f()) {
              fail_shape_inference("Attribute 'value_float' expect a float.");
            }
            updateOutputElemType(ctx, 0, TensorProto::FLOAT);
            updateOutputShape(ctx, 0, TensorShapeProto());
            return;
          }

          if (nullptr != value_floats) {
            // OpSchema::Verify check ensures that the attribute value has ints.
            if (value_floats->floats_size() < 1) {
              fail_shape_inference("Attribute 'value_floats' expect a list of floats.");
            }
            updateOutputElemType(ctx, 0, TensorProto::FLOAT);
            appendDim(getOutputShape(ctx, 0), value_floats->floats_size());
            return;
          }

          if (nullptr != value_string) {
            // OpSchema::Verify check ensures that the attribute value has_i():
            if (!value_string->has_s()) {
              fail_shape_inference("Attribute 'value_string' expect a string.");
            }
            updateOutputElemType(ctx, 0, TensorProto::STRING);
            updateOutputShape(ctx, 0, TensorShapeProto());
            return;
          }

          if (nullptr != value_strings) {
            // OpSchema::Verify check ensures that the attribute value has ints.
            if (value_strings->strings_size() < 1) {
              fail_shape_inference("Attribute 'value_strings' expect a list of strings.");
            }
            updateOutputElemType(ctx, 0, TensorProto::STRING);
            appendDim(getOutputShape(ctx, 0), value_strings->strings_size());
            return;
          }

          if (nullptr != sparse_value) {
            // OpSchema::Verify check ensures that the attribute value
            // has_sparse_tensor():
            const SparseTensorProto& sparse = sparse_value->sparse_tensor();
            // checker.cc::check_sparse_tensor checks that the sparse-value is
            // well-formed
            updateOutputElemType(ctx, 0, sparse.values().data_type());
            auto* output_shape = getOutputShape(ctx, 0);
            for (int i = 0; i < sparse.dims_size(); ++i)
              appendDim(output_shape, sparse.dims(i));
            return;
          }

          fail_shape_inference(
              "TypeAndShapeInferenceFunction implementation incomplete: "
              "this line should never be reached.");
        }));

static const char* Constant_ver1_doc = R"DOC(A constant tensor.)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Constant,
    1,
    OpSchema()
        .SetDoc(Constant_ver1_doc)
        .Attr("value", "The value for the elements of the output tensor.", AttributeProto::TENSOR)
        .Output(0, "output", "Output tensor containing the same value of the provided tensor.", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          auto attr_proto = ctx.getAttribute("value");
          if (nullptr == attr_proto)
            return; // attribute not present
          if (!attr_proto->has_t())
            return; // attribute has no tensor value
          const TensorProto& tensor_proto = attr_proto->t();
          updateOutputElemType(ctx, 0, tensor_proto.data_type());
          updateOutputShape(ctx, 0, tensor_proto);
        }));

static const char* Constant_ver9_doc = R"DOC(A constant tensor.)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Constant,
    9,
    OpSchema()
        .SetDoc(Constant_ver9_doc)
        .Attr("value", "The value for the elements of the output tensor.", AttributeProto::TENSOR)
        .Output(0, "output", "Output tensor containing the same value of the provided tensor.", "T")
        .TypeConstraint("T", OpSchema::all_tensor_types(), "Constrain input and output types to all tensor types.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          auto attr_proto = ctx.getAttribute("value");
          if (nullptr == attr_proto || !attr_proto->has_t())
            fail_shape_inference("Attribute 'value' of Constant node must exist with 'Tensor' data.");
          const TensorProto& tensor_proto = attr_proto->t();
          updateOutputElemType(ctx, 0, tensor_proto.data_type());
          updateOutputShape(ctx, 0, tensor_proto);
        }));

static const char* Constant_ver11_doc = R"DOC(
A constant tensor. Exactly one of the two attributes, either value or sparse_value,
must be specified.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Constant,
    11,
    OpSchema()
        .SetDoc(Constant_ver11_doc)
        .Attr("value", "The value for the elements of the output tensor.", AttributeProto::TENSOR, false)
        .Attr(
            "sparse_value",
            "The value for the elements of the output tensor in sparse format.",
            AttributeProto::SPARSE_TENSOR,
            false)
        .Output(0, "output", "Output tensor containing the same value of the provided tensor.", "T")
        .TypeConstraint("T", OpSchema::all_tensor_types(), "Constrain input and output types to all tensor types.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          auto* value = ctx.getAttribute("value");
          auto* sparse_value = ctx.getAttribute("sparse_value");

          if ((nullptr != value) && (nullptr != sparse_value))
            fail_shape_inference(
                "Only one of the attributes 'value' or 'sparse_value' must be specified for a Constant node.");

          if (nullptr != value) {
            // OpSchema::Verify check ensures that the attribute value has_t():
            const TensorProto& tensor_proto = value->t();
            updateOutputElemType(ctx, 0, tensor_proto.data_type());
            updateOutputShape(ctx, 0, tensor_proto);
            return;
          }

          if (nullptr != sparse_value) {
            // OpSchema::Verify check ensures that the attribute value
            // has_sparse_tensor():
            const SparseTensorProto& sparse = sparse_value->sparse_tensor();
            // checker.cc::check_sparse_tensor checks that the sparse-value is
            // well-formed
            updateOutputElemType(ctx, 0, sparse.values().data_type());
            auto* output_shape = getOutputShape(ctx, 0);
            for (int i = 0; i < sparse.dims_size(); ++i)
              appendDim(output_shape, sparse.dims(i));
            return;
          }
          fail_shape_inference(
              "One of the attributes 'value' or 'sparse_value' must be specified for a Constant node.");
        }));

} // namespace ONNX_NAMESPACE
