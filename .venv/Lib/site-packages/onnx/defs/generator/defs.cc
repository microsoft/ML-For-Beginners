/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cmath>
#include "onnx/defs/function.h"
#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {
static const char* Constant_ver19_doc = R"DOC(
This operator produces a constant tensor. Exactly one of the provided attributes, either value, sparse_value,
or value_* must be specified.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Constant,
    19,
    OpSchema()
        .SetDoc(Constant_ver19_doc)
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
        .TypeConstraint("T", OpSchema::all_tensor_types_ir9(), "Constrain input and output types to all tensor types.")
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
        }));

static const char* ConstantOfShape_ver9_doc = R"DOC(
Generate a tensor with given value and shape.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    ConstantOfShape,
    9,
    OpSchema()
        .SetDoc(ConstantOfShape_ver9_doc)
        .Attr(
            "value",
            "(Optional) The value of the output elements."
            "Should be a one-element tensor. If not specified, it defaults to a tensor of value 0 and datatype float32",
            AttributeProto::TENSOR,
            OPTIONAL_VALUE)
        .Input(
            0,
            "input",
            "1D tensor. The shape of the expected output tensor. If empty tensor is given, the output would be a scalar."
            " All values must be >= 0.",
            "T1")
        .Output(
            0,
            "output",
            "Output tensor of shape specified by 'input'."
            "If attribute 'value' is specified, the value and datatype of the output tensor is taken from 'value'."
            "If attribute 'value' is not specified, the value in the output defaults to 0, and the datatype "
            "defaults to float32.",
            "T2")
        .TypeConstraint("T1", {"tensor(int64)"}, "Constrain input types.")
        .TypeConstraint(
            "T2",
            {"tensor(float16)",
             "tensor(float)",
             "tensor(double)",
             "tensor(int8)",
             "tensor(int16)",
             "tensor(int32)",
             "tensor(int64)",
             "tensor(uint8)",
             "tensor(uint16)",
             "tensor(uint32)",
             "tensor(uint64)",
             "tensor(bool)"},
            "Constrain output types to be numerics.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          if (ctx.getAttribute("value") != nullptr) {
            propagateElemTypeFromDtypeToOutput(ctx, ctx.getAttribute("value"), 0);
          } else {
            propagateElemTypeFromDtypeToOutput(ctx, TensorProto::FLOAT, 0);
          }

          bool found = false;
          TensorShapeProto output_shape = getShapeInput(ctx, 0, found);
          if (found) {
            *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape() = output_shape;
          }
        }));

static const char* EyeLike_ver9_doc = R"DOC(
Generate a 2D tensor (matrix) with ones on the diagonal and zeros everywhere else. Only 2D
tensors are supported, i.e. input T1 must be of rank 2. The shape of the output tensor is the
same as the input tensor. The data type can be specified by the 'dtype' argument. If
'dtype' is not specified, then the type of input tensor is used. By default, the main diagonal
is populated with ones, but attribute 'k' can be used to populate upper or lower diagonals.
The 'dtype' argument must be one of the data types specified in the 'DataType' enum field in the
TensorProto message and be valid as an output type.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    EyeLike,
    9,
    OpSchema()
        .SetDoc(EyeLike_ver9_doc)
        .Attr(
            "k",
            "(Optional) Index of the diagonal to be populated with ones. Default is 0."
            " If T2 is the output, this op sets T2[i, i+k] = 1. k = 0 populates the main diagonal, "
            "k > 0 populates an upper diagonal,  and k < 0 populates a lower diagonal.",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Attr(
            "dtype",
            "(Optional) The data type for the elements of the output tensor. If not specified,"
            "the data type of the input tensor T1 is used. If input tensor T1 is also not"
            "specified, then type defaults to 'float'.",
            AttributeProto::INT,
            OPTIONAL_VALUE)
        .Input(0, "input", "2D input tensor to copy shape, and optionally, type information from.", "T1")
        .Output(0, "output", "Output tensor, same shape as input tensor T1.", "T2")
        .TypeConstraint(
            "T1",
            {"tensor(float16)",
             "tensor(float)",
             "tensor(double)",
             "tensor(int8)",
             "tensor(int16)",
             "tensor(int32)",
             "tensor(int64)",
             "tensor(uint8)",
             "tensor(uint16)",
             "tensor(uint32)",
             "tensor(uint64)",
             "tensor(bool)"},
            "Constrain input types. Strings and complex are not supported.")
        .TypeConstraint(
            "T2",
            {"tensor(float16)",
             "tensor(float)",
             "tensor(double)",
             "tensor(int8)",
             "tensor(int16)",
             "tensor(int32)",
             "tensor(int64)",
             "tensor(uint8)",
             "tensor(uint16)",
             "tensor(uint32)",
             "tensor(uint64)",
             "tensor(bool)"},
            "Constrain output types. Strings and complex are not supported.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          if (ctx.getAttribute("dtype") != nullptr) {
            propagateElemTypeFromAttributeToOutput(ctx, "dtype", 0);
          } else {
            propagateElemTypeFromInputToOutput(ctx, 0, 0);
          }
          if (hasInputShape(ctx, 0)) {
            auto& input_shape = getInputShape(ctx, 0);
            if (input_shape.dim_size() != 2) {
              fail_shape_inference("Input tensor must be 2-dimensional");
            }
          }
          propagateShapeFromInputToOutput(ctx, 0, 0);
        }));

static const char* RandomUniform_ver1_doc = R"DOC(
Generate a tensor with random values drawn from a uniform distribution. The shape
of the tensor is specified by the `shape` argument and the range by `low` and `high`.

The data type is specified by the 'dtype' argument. The 'dtype' argument must
be one of the data types specified in the 'DataType' enum field in the
TensorProto message.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    RandomUniform,
    1,
    OpSchema()
        .SetDoc(RandomUniform_ver1_doc)
        .Attr("low", "Lower boundary of the output values.", AttributeProto::FLOAT, 0.0f)
        .Attr("high", "Upper boundary of the output values.", AttributeProto::FLOAT, 1.0f)
        .Attr(
            "seed",
            "(Optional) Seed to the random generator, if not specified we will auto generate one.",
            AttributeProto::FLOAT,
            OPTIONAL_VALUE)
        .Attr(
            "dtype",
            "The data type for the elements of the output tensor. If not specified, default is TensorProto::FLOAT.",
            AttributeProto::INT,
            static_cast<int64_t>(TensorProto::FLOAT))
        .Attr("shape", "The shape of the output tensor.", AttributeProto::INTS)
        .Output(0, "output", "Output tensor of random values drawn from uniform distribution", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain output types to float tensors.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromAttributeToOutput(ctx, "dtype", 0, TensorProto::FLOAT);
          propagateShapeFromAttributeToOutput(ctx, "shape", 0);
        }));

static const char* RandomNormal_ver1_doc = R"DOC(
Generate a tensor with random values drawn from a normal distribution. The shape
of the tensor is specified by the `shape` argument and the parameter of the normal distribution
specified by `mean` and `scale`.

The data type is specified by the 'dtype' argument. The 'dtype' argument must
be one of the data types specified in the 'DataType' enum field in the
TensorProto message.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    RandomNormal,
    1,
    OpSchema()
        .SetDoc(RandomNormal_ver1_doc)
        .Attr("mean", "The mean of the normal distribution.", AttributeProto::FLOAT, 0.0f)
        .Attr("scale", "The standard deviation of the normal distribution.", AttributeProto::FLOAT, 1.0f)
        .Attr(
            "seed",
            "(Optional) Seed to the random generator, if not specified we will auto generate one.",
            AttributeProto::FLOAT,
            OPTIONAL_VALUE)
        .Attr(
            "dtype",
            "The data type for the elements of the output tensor. Default is TensorProto::FLOAT.",
            AttributeProto::INT,
            static_cast<int64_t>(TensorProto::FLOAT))
        .Attr("shape", "The shape of the output tensor.", AttributeProto::INTS)
        .Output(0, "output", "Output tensor of random values drawn from normal distribution", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain output types to float tensors.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromAttributeToOutput(ctx, "dtype", 0, TensorProto::FLOAT);
          propagateShapeFromAttributeToOutput(ctx, "shape", 0);
        }));

static const char* RandomUniformLike_ver1_doc = R"DOC(
Generate a tensor with random values drawn from a uniform distribution.
The shape of the output tensor is copied from the shape of the input tensor,
and the parameters of the uniform distribution are specified by `low` and `high`.

The data type is specified by the 'dtype' argument, or copied from the input tensor if not provided.
The 'dtype' argument must be one of the data types specified in the 'DataType' enum field in the
TensorProto message and be valid as an output type.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    RandomUniformLike,
    1,
    OpSchema()
        .SetDoc(RandomUniformLike_ver1_doc)
        .Attr("low", "Lower boundary of the output values.", AttributeProto::FLOAT, 0.0f)
        .Attr("high", "Upper boundary of the output values.", AttributeProto::FLOAT, 1.0f)
        .Attr(
            "seed",
            "(Optional) Seed to the random generator, if not specified we will auto generate one.",
            AttributeProto::FLOAT,
            OPTIONAL_VALUE)
        .Attr(
            "dtype",
            "(Optional) The data type for the elements of the output tensor, if not specified, we will use "
            "the data type of the input tensor.",
            AttributeProto::INT,
            OPTIONAL_VALUE)
        .Input(0, "input", "Input tensor to copy shape and optionally type information from.", "T1")
        .Output(0, "output", "Output tensor of random values drawn from uniform distribution", "T2")
        .TypeConstraint(
            "T1",
            OpSchema::all_tensor_types(),
            "Constrain to any tensor type. If the dtype attribute is not provided this must be a valid output type.")
        .TypeConstraint(
            "T2",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain output types to float tensors.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          if (ctx.getAttribute("dtype") != nullptr)
            propagateElemTypeFromAttributeToOutput(ctx, "dtype", 0);
          else
            propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (!hasNInputShapes(ctx, 1)) {
            return;
          }
          propagateShapeFromInputToOutput(ctx, 0, 0);
        }));

static const char* RandomNormalLike_ver1_doc = R"DOC(
Generate a tensor with random values drawn from a normal distribution.
The shape of the output tensor is copied from the shape of the input tensor,
and the parameters of the normal distribution are specified by `mean` and `scale`.

The data type is specified by the 'dtype' argument, or copied from the input tensor if not provided.
The 'dtype' argument must be one of the data types specified in the 'DataType' enum field in the
TensorProto message, and be valid as an output type.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    RandomNormalLike,
    1,
    OpSchema()
        .SetDoc(RandomNormalLike_ver1_doc)
        .Attr("mean", "The mean of the normal distribution.", AttributeProto::FLOAT, 0.0f)
        .Attr("scale", "The standard deviation of the normal distribution.", AttributeProto::FLOAT, 1.0f)
        .Attr(
            "seed",
            "(Optional) Seed to the random generator, if not specified we will auto generate one.",
            AttributeProto::FLOAT,
            OPTIONAL_VALUE)
        .Attr(
            "dtype",
            "(Optional) The data type for the elements of the output tensor, if not specified, we will use "
            "the data type of the input tensor.",
            AttributeProto::INT,
            OPTIONAL_VALUE)
        .Input(0, "input", "Input tensor to copy shape and optionally type information from.", "T1")
        .Output(0, "output", "Output tensor of random values drawn from normal distribution", "T2")
        .TypeConstraint(
            "T1",
            OpSchema::all_tensor_types(),
            "Constrain to any tensor type. If the dtype attribute is not provided this must be a valid output type.")
        .TypeConstraint(
            "T2",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain output types to float tensors.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          if (ctx.getAttribute("dtype") != nullptr)
            propagateElemTypeFromAttributeToOutput(ctx, "dtype", 0);
          else
            propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (!hasNInputShapes(ctx, 1)) {
            return;
          }
          propagateShapeFromInputToOutput(ctx, 0, 0);
        }));

static const char* Multinomial_ver7_doc = R"DOC(
Generate a tensor of samples from a multinomial distribution according to the probabilities
of each of the possible outcomes.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Multinomial,
    7,
    OpSchema()
        .SetDoc(Multinomial_ver7_doc)
        .Attr("sample_size", "Number of times to sample.", AttributeProto::INT, static_cast<int64_t>(1))
        .Attr(
            "seed",
            "(Optional) Seed to the random generator, if not specified we will auto generate one.",
            AttributeProto::FLOAT,
            OPTIONAL_VALUE)
        .Attr(
            "dtype",
            "(Optional) The data type for the elements of the output tensor, if not specified, we will use int32.",
            AttributeProto::INT,
            static_cast<int64_t>(TensorProto::INT32))
        .Input(
            0,
            "input",
            "Input tensor with shape [batch_size, class_size], where class_size is the number of all possible outcomes. Each value along the axis zero represents the unnormalized log-probability of each corresponding outcome in a batch.",
            "T1")
        .Output(
            0,
            "output",
            "Output tensor with shape [batch_size, sample_size], where sample_size is the number of times to sample. Each value along the axis zero represents the outcome of the corresponding sample in a batch.",
            "T2")
        .TypeConstraint(
            "T1",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input types to float tensors.")
        .TypeConstraint("T2", {"tensor(int32)", "tensor(int64)"}, "Constrain output types to integral tensors.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          auto dtype = ctx.getAttribute("dtype");
          auto dataType = TensorProto_DataType::TensorProto_DataType_INT32;
          if (dtype != nullptr) {
            dataType = static_cast<TensorProto_DataType>(dtype->i());
            if (dataType != TensorProto_DataType::TensorProto_DataType_INT32 &&
                dataType != TensorProto_DataType::TensorProto_DataType_INT64) {
              fail_type_inference("Output type must be int32 or int64");
            }
          }
          updateOutputElemType(ctx, 0, dataType);

          TensorShapeProto::Dimension batch_size, sample_size;
          if (hasInputShape(ctx, 0)) {
            auto& input_shape = getInputShape(ctx, 0);
            if (input_shape.dim_size() != 2) {
              fail_shape_inference("Input tensor must have rank 2");
            }
            batch_size = input_shape.dim(0);
          } // else statically-unknown batch-size
          sample_size.set_dim_value(getAttribute(ctx, "sample_size", 1));
          updateOutputShape(ctx, 0, {batch_size, sample_size});
        }));

static const char* Range_ver11_doc = R"DOC(
Generate a tensor containing a sequence of numbers that begin at `start` and extends by increments of `delta`
up to `limit` (exclusive).

The number of elements in the output of range is computed as below:

```
number_of_elements = max( ceil( (limit - start) / delta ) , 0 )
```

The pseudocode determining the contents of the output is shown below:

```
for(int i=0; i<number_of_elements; ++i) {
  output[i] =  start + (i * delta);
}
```

Example 1

```
Inputs: start = 3, limit = 9, delta = 3
Output: [3, 6]
```

Example 2

```
Inputs: start = 10, limit = 4, delta = -2
Output: [10, 8, 6]
```
)DOC";

template <typename T>
inline int64_t
compute_output_dim_for_range(const TensorProto* start, const TensorProto* limit, const TensorProto* delta) {
  if (start->dims().size() != 0 || limit->dims().size() != 0 || delta->dims().size() != 0) {
    fail_shape_inference("Input to 'Range' op should be scalars (Tensor with only one element and shape empty)");
  }

  const auto& start_data = ParseData<T>(start);
  const auto& limit_data = ParseData<T>(limit);
  const auto& delta_data = ParseData<T>(delta);

  int64_t n = static_cast<int64_t>(ceil((1.0 * (limit_data[0] - start_data[0])) / delta_data[0]));

  if (n < 0)
    n = 0;

  return n;
}

ONNX_OPERATOR_SET_SCHEMA(
    Range,
    11,
    OpSchema()
        .SetDoc(Range_ver11_doc)
        .Input(0, "start", "Scalar. First entry for the range of output values.", "T")
        .Input(1, "limit", "Scalar. Exclusive upper limit for the range of output values.", "T")
        .Input(2, "delta", "Scalar. Value to step by.", "T")
        .Output(0, "output", "A 1-D tensor with same type as the inputs containing generated range of values.", "T")
        .TypeConstraint(
            "T",
            {"tensor(float)", "tensor(double)", "tensor(int16)", "tensor(int32)", "tensor(int64)"},
            "Constrain input types to common numeric type tensors.")
        .FunctionBody(R"ONNX(
          {
            sub_result = Sub (limit, start)
            sub_result_casted = Cast <to = 1> (sub_result)
            delta_casted = Cast <to = 1> (delta)
            div_result = Div (sub_result_casted, delta_casted)
            ceil_result = Ceil (div_result)
            ceil_result_relu = Relu (ceil_result)
            ceil_result_relu_int = Cast <to = 7> (ceil_result_relu)
            ceil_result_relu_bool = Cast <to = 9> (ceil_result_relu)
            variadic_output, output = Loop (ceil_result_relu_int, ceil_result_relu_bool, start)
              <body = loop_body_attribute (int64 i, bool cond, prev) => (cond_out, current, range) {
                cond_out = Identity (cond)
                current = Add (prev, delta)
                range = Identity (prev)
              }>
          }
        )ONNX")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          // Type inference
          propagateElemTypeFromInputToOutput(ctx, 0, 0);

          // Shape inference
          const auto* start_initializer = ctx.getInputData(0);
          const auto* limit_initializer = ctx.getInputData(1);
          const auto* delta_initializer = ctx.getInputData(2);

          // Output is always 1-D
          auto* output_dim = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim();

          // If any of Range's inputs are not initializers, the output dimension
          // value would remain unknown.
          if (start_initializer != nullptr && limit_initializer != nullptr && delta_initializer != nullptr) {
            // Make sure the input types are homogeneous
            if ((start_initializer->data_type() != limit_initializer->data_type()) ||
                (start_initializer->data_type() != delta_initializer->data_type())) {
              fail_shape_inference("All inputs to 'Range' op must be of the same type");
            }

            // Explicitly compute the output dimension if Range's inputs are
            // stored in initializer list.
            if (start_initializer->data_type() == TensorProto::FLOAT) {
              output_dim->set_dim_value(
                  compute_output_dim_for_range<float>(start_initializer, limit_initializer, delta_initializer));
            } else if (start_initializer->data_type() == TensorProto::INT32) {
              output_dim->set_dim_value(
                  compute_output_dim_for_range<int32_t>(start_initializer, limit_initializer, delta_initializer));
            } else if (start_initializer->data_type() == TensorProto::INT64) {
              output_dim->set_dim_value(
                  compute_output_dim_for_range<int64_t>(start_initializer, limit_initializer, delta_initializer));
            } else if (start_initializer->data_type() == TensorProto::DOUBLE) {
              output_dim->set_dim_value(
                  compute_output_dim_for_range<double>(start_initializer, limit_initializer, delta_initializer));
            } else {
              // 'float16' has no native CPU type -
              // stop with rank inference, no action here
            }

            return;
          }
        }));

static const char* Bernoulli_ver15_doc = R"DOC(
Draws binary random numbers (0 or 1) from a Bernoulli distribution. The input tensor should be a tensor
containing probabilities p (a value in the range [0,1]) to be used for drawing the binary random number,
where an output of 1 is produced with probability p and an output of 0 is produced with probability (1-p).

This operator is non-deterministic and may not produce the same values in different
implementations (even if a seed is specified).
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Bernoulli,
    15,
    OpSchema()
        .SetDoc(Bernoulli_ver15_doc)
        .Attr(
            "seed",
            "(Optional) Seed to the random generator, if not specified we will auto generate one.",
            AttributeProto::FLOAT,
            OPTIONAL_VALUE)
        .Attr(
            "dtype",
            "The data type for the elements of the output tensor. if not specified, we will use "
            "the data type of the input tensor.",
            AttributeProto::INT,
            OPTIONAL_VALUE)
        .Input(0, "input", "All values in input have to be in the range:[0, 1].", "T1")
        .Output(0, "output", "The returned output tensor only has values 0 or 1, same shape as input tensor.", "T2")
        .TypeConstraint(
            "T1",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input types to float tensors.")
        .TypeConstraint(
            "T2",
            {"tensor(float16)",
             "tensor(float)",
             "tensor(double)",
             "tensor(bfloat16)",
             "tensor(uint8)",
             "tensor(uint16)",
             "tensor(uint32)",
             "tensor(uint64)",
             "tensor(int8)",
             "tensor(int16)",
             "tensor(int32)",
             "tensor(int64)",
             "tensor(bool)"},
            "Constrain output types to all numeric tensors and bool tensors.")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          if (ctx.getAttribute("dtype") != nullptr)
            propagateElemTypeFromAttributeToOutput(ctx, "dtype", 0);
          else
            propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (!hasNInputShapes(ctx, 1)) {
            return;
          }
          propagateShapeFromInputToOutput(ctx, 0, 0);
        })
        .SetContextDependentFunctionBodyBuilder(
            [](const FunctionBodyBuildContext& ctx, const OpSchema& schema, FunctionProto& functionProto) -> bool {
              if (ctx.getInputType(0) == nullptr) {
                // we cannot create a correct function body without knowing the input type
                return false;
              }
              auto input_type = ctx.getInputType(0)->tensor_type().elem_type();
              auto dtype = ctx.getAttribute("dtype") != nullptr
                  ? static_cast<TensorProto_DataType>(ctx.getAttribute("dtype")->i())
                  : input_type;
              FunctionBuilder builder(functionProto);
              builder
                  .Add(
                      "X_random = RandomUniformLike <low = 0.0, high = 1.0, seed = @seed> (input)",
                      "dtype",
                      int64_t(input_type))
                  .Add("X_greater = Greater (X_random, input)")
                  .Add("output = Cast (X_greater)", "to", int64_t(dtype));
              schema.BuildFunction(functionProto);
              return true;
            }));
} // namespace ONNX_NAMESPACE
