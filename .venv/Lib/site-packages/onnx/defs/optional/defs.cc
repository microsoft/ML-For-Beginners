/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <numeric>

#include "onnx/defs/function.h"
#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {
static std::vector<std::string> optional_and_tensor_types() {
  auto optional_types = OpSchema::all_optional_types();
  auto tensor_types = OpSchema::all_tensor_types();
  auto sequence_types = OpSchema::all_tensor_sequence_types();
  optional_types.insert(optional_types.end(), tensor_types.begin(), tensor_types.end());
  optional_types.insert(optional_types.end(), sequence_types.begin(), sequence_types.end());
  return optional_types;
}

static const char* Optional_ver15_doc = R"DOC(
Constructs an optional-type value containing either an empty optional of a certain type specified by the attribute,
or a non-empty value containing the input element.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Optional,
    15,
    OpSchema()
        .SetDoc(Optional_ver15_doc)
        .Input(0, "input", "The input element.", "V", OpSchema::Optional)
        .Attr("type", "Type of the element in the optional output", AttributeProto::TYPE_PROTO, OPTIONAL_VALUE)
        .Output(0, "output", "The optional output enclosing the input element.", "O")
        .TypeConstraint(
            "V",
            []() {
              auto t = OpSchema::all_tensor_types();
              auto s = OpSchema::all_tensor_sequence_types();
              t.insert(t.end(), s.begin(), s.end());
              return t;
            }(),
            "Constrain input type to all tensor and sequence types.")
        .TypeConstraint(
            "O",
            OpSchema::all_optional_types(),
            "Constrain output type to all optional tensor or optional sequence types.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          const size_t numOutputs = ctx.getNumOutputs();
          if (numOutputs != 1) {
            fail_type_inference("Optional is expected to have an output.");
          }

          const size_t numInputs = ctx.getNumInputs();
          const auto* attr_proto = ctx.getAttribute("type");

          if ((numInputs == 0) && (attr_proto != nullptr)) {
            if (!attr_proto->has_tp())
              fail_type_inference("Attribute 'type' should be a TypeProto and it should specify a type.");
            auto attr_tp = attr_proto->tp();

            ctx.getOutputType(0)->mutable_optional_type()->mutable_elem_type()->CopyFrom(attr_tp);
          } else if (numInputs == 1) {
            auto input_type = ctx.getInputType(0);
            if (input_type == nullptr) {
              fail_type_inference("Input type is null. Type information is expected for the input.");
            }
            ctx.getOutputType(0)->mutable_optional_type()->mutable_elem_type()->CopyFrom(*input_type);
          } else {
            fail_type_inference("Optional is expected to have either an input or the type attribute set.");
          }
        }));

static const char* OptionalHasElement_ver18_doc = R"DOC(
Returns true if (1) the input is an optional-type and contains an element,
or, (2) the input is a tensor or sequence type.
If the input is not provided or is an empty optional-type, this op returns false.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    OptionalHasElement,
    18,
    OpSchema()
        .SetDoc(OptionalHasElement_ver18_doc)
        .Input(0, "input", "The optional input.", "O", OpSchema::Optional)
        .Output(
            0,
            "output",
            "A scalar boolean tensor. If true, it indicates that optional-type input contains an element. Otherwise, it is empty.",
            "B")
        .TypeConstraint(
            "O",
            optional_and_tensor_types(),
            "Constrain input type to optional tensor and optional sequence types.")
        .TypeConstraint("B", {"tensor(bool)"}, "Constrain output to a boolean tensor.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          const size_t numInputs = ctx.getNumInputs();
          if (numInputs != 0 && numInputs != 1) {
            fail_type_inference("OptionalHasElement is expected to have 0 or 1 input.");
          }
          const size_t numOutputs = ctx.getNumOutputs();
          if (numOutputs != 1) {
            fail_type_inference("OptionalHasElement is expected to have 1 output.");
          }
          auto* output_tensor_type = ctx.getOutputType(0)->mutable_tensor_type();
          output_tensor_type->set_elem_type(TensorProto::BOOL);
          output_tensor_type->mutable_shape()->Clear();
        }));

static const char* OptionalGetElement_ver18_doc = R"DOC(
If the input is a tensor or sequence type, it returns the input.
If the input is an optional type, it outputs the element in the input.
It is an error if the input is an empty optional-type (i.e. does not have an element) and the behavior is undefined in this case.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    OptionalGetElement,
    18,
    OpSchema()
        .SetDoc(OptionalGetElement_ver18_doc)
        .Input(0, "input", "The optional input.", "O")
        .Output(0, "output", "Output element in the optional input.", "V")
        .TypeConstraint(
            "O",
            optional_and_tensor_types(),
            "Constrain input type to optional tensor and optional sequence types.")
        .TypeConstraint(
            "V",
            []() {
              auto t = OpSchema::all_tensor_types();
              auto s = OpSchema::all_tensor_sequence_types();
              t.insert(t.end(), s.begin(), s.end());
              return t;
            }(),
            "Constrain output type to all tensor or sequence types.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          const size_t numInputs = ctx.getNumInputs();
          if (numInputs != 1) {
            fail_type_inference("OptionalGetElement must have an input element.");
          }
          auto input_type = ctx.getInputType(0);
          if (input_type == nullptr) {
            fail_type_inference("Input type is null. Input must have Type information.");
          }
          if (input_type->has_optional_type()) {
            if (!input_type->optional_type().has_elem_type()) {
              fail_type_inference("Optional-type input must contain an element with type information.");
            }
            ctx.getOutputType(0)->CopyFrom(input_type->optional_type().elem_type());
          } else {
            propagateShapeAndTypeFromFirstInput(ctx);
          }
        }));

} // namespace ONNX_NAMESPACE
