/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cmath>
#include "onnx/common/assertions.h"
#include "onnx/defs/function.h"
#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {
const char* pads_doc =
    "Padding for the beginning and ending along each spatial axis, it can take any value greater "
    "than or equal to 0. The value represent the number of pixels added to the beginning "
    "and end part of the corresponding axis. `pads` format should be as follow "
    "[x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels "
    "added at the beginning of axis `i` and xi_end, the number of pixels added at "
    "the end of axis `i`. This attribute cannot be used simultaneously with "
    "auto_pad attribute. If not present, the padding defaults to 0 along start and end of each spatial axis.";
const char* conv_auto_pad_doc =
    "auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. Where "
    "default value is NOTSET, which means explicit padding is used. "
    "SAME_UPPER or SAME_LOWER mean pad the input so that "
    "`output_shape[i] = ceil(input_shape[i] / strides[i])` for each axis `i`. "
    "The padding is split between the two sides equally or almost equally (depending "
    "on whether it is even or odd). In case the padding is an odd number, the extra "
    "padding is added at the end for SAME_UPPER and at the beginning for SAME_LOWER.";
const char* conv_transpose_auto_pad_doc =
    "auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. Where "
    "default value is NOTSET, which means explicit padding is used. "
    "SAME_UPPER or SAME_LOWER mean pad the input so that "
    "`output_shape[i] = input_shape[i] * strides[i]` for each axis `i`. "
    "The padding is split between the two sides equally or almost equally (depending "
    "on whether it is even or odd). In case the padding is an odd number, the extra "
    "padding is added at the end for SAME_UPPER and at the beginning for SAME_LOWER.";

void convPoolShapeInference(
    InferenceContext& ctx,
    bool use_dilation,
    bool require_kernel_shape,
    int input1Idx,
    int input2Idx) {
  // we need the first input shape for this inference.
  if (!hasInputShape(ctx, input1Idx)) {
    return;
  }

  // if kernel shape is an input (and not attribute)
  // we need the shape of the second input.
  if (!require_kernel_shape && !hasInputShape(ctx, input2Idx)) {
    return;
  }

  auto input_shape = ctx.getInputType(input1Idx)->tensor_type().shape();
  if (input_shape.dim_size() < 2) {
    fail_shape_inference("Input tensor must have at least 2 dimensions");
  }

  // first dim is the batch axis and the next is the number of channels.
  size_t n_input_dims = static_cast<size_t>(input_shape.dim_size() - 2);

  // Only MaxPool and Conv support dilation. For
  // simplicity of the code, we just treat the rest of them as having all-1s
  // dilation.
  std::vector<int64_t> dilations;
  if (use_dilation && getRepeatedAttribute(ctx, "dilations", dilations)) {
    if (dilations.size() != n_input_dims) {
      fail_shape_inference("Attribute dilations has incorrect size");
    }
  } else {
    dilations.assign(n_input_dims, 1);
  }

  std::vector<int64_t> strides;
  if (getRepeatedAttribute(ctx, "strides", strides)) {
    if (strides.size() != n_input_dims) {
      fail_shape_inference("Attribute strides has incorrect size");
    }
  } else {
    strides.assign(n_input_dims, 1);
  }

  std::vector<int64_t> kernel_shape;
  if (getRepeatedAttribute(ctx, "kernel_shape", kernel_shape)) {
    if (kernel_shape.size() != n_input_dims) {
      fail_shape_inference("Attribute kernel_shape has incorrect size");
    }
  } else if (require_kernel_shape) {
    fail_shape_inference("Attribute kernel_shape must be specified");
  } else {
    auto second_input_shape = ctx.getInputType(input2Idx)->tensor_type().shape();
    for (int i = 2; i < second_input_shape.dim_size(); ++i) {
      if (!second_input_shape.dim(i).has_dim_value()) {
        return;
      }
      kernel_shape.push_back(second_input_shape.dim(i).dim_value());
    }
  }

  std::vector<int64_t> effective_kernel_shape = kernel_shape;
  for (int i = 0; i < static_cast<int>(kernel_shape.size()); i++) {
    // accounting for dilation, how big is the kernel in this dimension
    effective_kernel_shape[i] = (effective_kernel_shape[i] - 1) * dilations[i] + 1;
  }

  std::vector<int64_t> pads;
  if (getRepeatedAttribute(ctx, "pads", pads)) {
    if (pads.size() != n_input_dims * 2) {
      fail_shape_inference("Attribute pads has incorrect size");
    }
  } else {
    pads.assign(n_input_dims * 2, 0);
    const auto* auto_pad_attr = ctx.getAttribute("auto_pad");
    if ((nullptr != auto_pad_attr) && (auto_pad_attr->s() != "VALID")) {
      int input_dims_size = static_cast<int>(n_input_dims);
      for (int i = 0; i < input_dims_size; ++i) {
        int64_t residual = 0;
        int64_t stride = strides[i];
        if (stride > 1) {
          if (!input_shape.dim(2 + i).has_dim_value()) {
            continue;
          }
          residual = input_shape.dim(2 + i).dim_value();
          while (residual >= stride) {
            residual -= stride;
          }
        }
        int64_t total_pad = residual == 0 ? effective_kernel_shape[i] - stride : effective_kernel_shape[i] - residual;
        if (total_pad < 0)
          total_pad = 0;
        int64_t half_pad_small = total_pad >> 1;
        int64_t half_pad_big = total_pad - half_pad_small;
        if (auto_pad_attr->s() == "SAME_UPPER") {
          pads[i] = half_pad_small;
          pads[i + input_dims_size] = half_pad_big;
        } else if (auto_pad_attr->s() == "SAME_LOWER") {
          pads[i] = half_pad_big;
          pads[i + input_dims_size] = half_pad_small;
        }
      }
    }
  }

  auto output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

  if (require_kernel_shape) {
    // add the first two dimensions from the input.
    *output_shape->add_dim() = input_shape.dim(0);
    *output_shape->add_dim() = input_shape.dim(1);
  } else {
    *output_shape->add_dim() = input_shape.dim(0);
    auto& second_input_shape = getInputShape(ctx, input2Idx);
    if (second_input_shape.dim_size() < 1) {
      fail_shape_inference("Second input tensor has wrong dimension");
    }
    *output_shape->add_dim() = second_input_shape.dim(0);
  }

  int kernel_shape_size = static_cast<int>(kernel_shape.size());
  for (int i = 0; i < kernel_shape_size; ++i) {
    auto newdim = output_shape->add_dim();
    if (!input_shape.dim(2 + i).has_dim_value()) {
      continue;
    }
    // how big is the input, including padding
    int64_t effective_input_size = input_shape.dim(2 + i).dim_value();
    effective_input_size += pads[i];
    effective_input_size += pads[i + kernel_shape_size];

    // default is floor mode .i.e. ceil_mode is set to 0
    auto ceil_mode = getAttribute(ctx, "ceil_mode", 0);

    // how many times we can move the kernel from it's initial position, based
    // on the stride
    int64_t strided_kernel_positions;

    if (ceil_mode == 1)
      strided_kernel_positions =
          (int64_t)(std::ceil((effective_input_size - effective_kernel_shape[i]) / float(strides[i])));
    else
      strided_kernel_positions = (effective_input_size - effective_kernel_shape[i]) / strides[i];

    // add in the initial position
    newdim->set_dim_value(1 + strided_kernel_positions);
  }

  if (ctx.getNumOutputs() > 1) {
    // MaxPool with two outputs case.
    auto second_output_shape = ctx.getOutputType(1)->mutable_tensor_type()->mutable_shape();
    second_output_shape->CopyFrom(*output_shape);
  }
}

std::vector<std::string> GetSupportedDataTypesForPoolingOps(bool supports8bit) {
  if (supports8bit) {
    return {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(int8)", "tensor(uint8)"};
  }
  return {"tensor(float16)", "tensor(float)", "tensor(double)"};
}

std::function<void(OpSchema&)> PoolOpSchemaGenerator(
    const char* name,
    const char* opName,
    const char* additionalDescription,
    bool use_dilation,
    bool supports8bit = false) {
  return [=](OpSchema& schema) {
    std::string doc;
    POPULATE_OP_DOC_STR(
        doc = R"DOC(
 {name} consumes an input tensor X and applies {opName} pooling across
 the tensor according to kernel sizes, stride sizes, and pad lengths.
 {opName} pooling consisting of computing the {opName} on all values of a
 subset of the input tensor according to the kernel size and downsampling the
 data into the output tensor Y for further processing. The output spatial shape will be following:
 ```
 output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - {kernelSpatialShape}) / strides_spatial_shape[i] + 1)
 ```
 or
 ```
 output_spatial_shape[i] = ceil((input_spatial_shape[i] + pad_shape[i] - {kernelSpatialShape}) / strides_spatial_shape[i] + 1)
 ```
 if ceil_mode is enabled `pad_shape[i]` is the sum of pads along axis `i`.

 `auto_pad` is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following:
 ```
 VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - {kernelSpatialShape} + 1) / strides_spatial_shape[i])
 SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
 ```
 And pad shape will be following if `SAME_UPPER` or `SAME_LOWER`:
 ```
 pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + {kernelSpatialShape} - input_spatial_shape[i]
 ```
 {additionalDescription}
 )DOC";
        ReplaceAll(doc, "{name}", name);
        ReplaceAll(doc, "{opName}", opName);
        ReplaceAll(doc, "{additionalDescription}", additionalDescription);
        ReplaceAll(
            doc,
            "{kernelSpatialShape}",
            use_dilation ? "((kernel_spatial_shape[i] - 1) * dilations[i] + 1)" : "kernel_spatial_shape[i]"););
    schema.SetDoc(doc);
    schema.Attr("kernel_shape", "The size of the kernel along each axis.", AttributeProto::INTS);
    schema.Attr(
        "strides",
        "Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis.",
        AttributeProto::INTS,
        OPTIONAL_VALUE);
    schema.Attr("auto_pad", conv_auto_pad_doc, AttributeProto::STRING, std::string("NOTSET"));
    schema.Attr("pads", pads_doc, AttributeProto::INTS, OPTIONAL_VALUE);
    schema.Attr(
        "ceil_mode",
        "Whether to use ceil or floor (default) to compute the output shape.",
        AttributeProto::INT,
        static_cast<int64_t>(0));
    schema.Input(
        0,
        "X",
        "Input data tensor from the previous operator; "
        "dimensions for image case are (N x C x H x W), "
        "where N is the batch size, C is the number of "
        "channels, and H and W are the height and the "
        "width of the data. For non image case, the "
        "dimensions are in the form of "
        "(N x C x D1 x D2 ... Dn), where N is the batch "
        "size. Optionally, if dimension denotation is "
        "in effect, the operation expects the input "
        "data tensor to arrive with the dimension denotation "
        "of [DATA_BATCH, DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].",
        "T",
        OpSchema::Single,
        true,
        1,
        OpSchema::Differentiable);
    schema.Output(
        0,
        "Y",
        "Output data tensor from average or max pooling across "
        "the input tensor. Dimensions will vary based "
        "on various kernel, stride, and pad sizes. Floor value of "
        "the dimension is used",
        "T",
        OpSchema::Single,
        true,
        1,
        OpSchema::Differentiable);
    schema.TypeConstraint(
        "T",
        GetSupportedDataTypesForPoolingOps(supports8bit),
        supports8bit ? "Constrain input and output types to float and 8 bit tensors."
                     : "Constrain input and output types to float tensors.");
    schema.TypeAndShapeInferenceFunction([use_dilation](InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      if (ctx.getNumOutputs() > 1) {
        // MaxPool with two outputs case.
        auto output_type = ctx.getOutputType(1);
        if (output_type->value_case() == TypeProto::kTensorType ||
            output_type->value_case() == TypeProto::VALUE_NOT_SET) {
          output_type->mutable_tensor_type()->set_elem_type(TensorProto::INT64);
        }
      }
      convPoolShapeInference(ctx, use_dilation, true, 0, 1);
    });
  };
}

ONNX_OPERATOR_SET_SCHEMA(
    AveragePool,
    19,
    OpSchema()
        .FillUsing(PoolOpSchemaGenerator(
            "AveragePool",
            "average",
            "The output of each pooling window is divided by the number of elements (exclude pad when attribute count_include_pad is zero).",
            true, /* use_dilation: dilations attribute has been added in opset 19. */
            false /* supports8bit: does not support 8bit. */))
        .Attr(
            "dilations",
            "Dilation value along each spatial axis of filter. If not present, the dilation defaults to 1 along each spatial axis.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr(
            "count_include_pad",
            "Whether include pad pixels when calculating values for the edges. Default is 0, doesn't count include pad.",
            AttributeProto::INT,
            static_cast<int64_t>(0)));

ONNX_OPERATOR_SET_SCHEMA(
    MaxPool,
    12,
    OpSchema()
        .FillUsing(PoolOpSchemaGenerator(
            "MaxPool",
            "max",
            "The output of each pooling window is maximum number of elements exclude pad. ",
            true,
            true))
        .Attr(
            "storage_order",
            "The storage order of the tensor. 0 is row major, and 1 is column major. "
            "This attribute is used only to convert an n-tuple index value into "
            "a single integer value for producing the second output. ",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Attr(
            "dilations",
            "Dilation value along each spatial axis of filter. If not present, the dilation defaults to 1 along each spatial axis.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Output(
            1,
            "Indices",
            "Indices tensor from max pooling across the input tensor. "
            "The dimensions of indices are the same as output tensor. "
            "The values in indices of are the indices of the selected values during pooling. "
            "The indices are computed as flatten 1-D tensor, "
            "and the indices do not consider padding. "
            "So the values in indices are in [0, N x C x D1 x ... x Dn).",
            "I",
            OpSchema::Optional,
            true,
            1,
            OpSchema::NonDifferentiable)
        .TypeConstraint("I", {"tensor(int64)"}, "Constrain index tensor to int64"));

void maxUnpoolShapeInference(InferenceContext& ctx) {
  // we need at least two inputs to have a shape for this inference.
  if (ctx.getNumInputs() != 2 && ctx.getNumInputs() != 3) {
    fail_type_inference("MaxUnpool op must have either two or three inputs.");
  }
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasInputShape(ctx, 0)) {
    return; // If first input does not have shape, we cannot infer much.
  }
  auto input_shape = ctx.getInputType(0)->tensor_type().shape();
  if (input_shape.dim_size() < 2) {
    fail_shape_inference("Input tensor X must have at least 2 dimensions.");
  }

  // first dim is the batch axis and the next is the number of channels.
  size_t n_input_dims = static_cast<size_t>(input_shape.dim_size() - 2);

  std::vector<int64_t> pads;
  if (getRepeatedAttribute(ctx, "pads", pads)) {
    if (pads.size() != n_input_dims * 2) {
      fail_shape_inference("Attribute pads has incorrect size.");
    }
  } else {
    pads.assign(n_input_dims * 2, 0);
  }

  std::vector<int64_t> strides;
  if (getRepeatedAttribute(ctx, "strides", strides)) {
    if (strides.size() != n_input_dims) {
      fail_shape_inference("Attribute strides has incorrect size.");
    }
  } else {
    strides.assign(n_input_dims, 1);
  }

  std::vector<int64_t> kernel_shape;
  if (getRepeatedAttribute(ctx, "kernel_shape", kernel_shape)) {
    if (kernel_shape.size() != n_input_dims) {
      fail_shape_inference("Attribute kernel_shape has incorrect size.");
    }
  } else {
    fail_shape_inference("Attribute kernel_shape must be specified.");
  }

  if (ctx.getNumInputs() == 3) {
    // If the third input, output_size, is specified, then use that instead
    // of inferring shape from inputs.
    if (hasInputShape(ctx, 2)) {
      auto& output_shape = getInputShape(ctx, 2);
      if (output_shape.dim_size() != 1) {
        fail_type_inference("'output_shape' must be rank 1 tensor.");
      }
      if (output_shape.dim((int)0).has_dim_value() &&
          static_cast<int>(output_shape.dim((int)0).dim_value()) != input_shape.dim_size()) {
        fail_shape_inference("'output_shape' must have same number of elements as the shape of input tensor X.");
      }
    }
    return; // 'output_shape' is specified as input. Actual shape will be
            // determined at runtime.
  }

  auto final_output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

  *final_output_shape->add_dim() = input_shape.dim(0);
  *final_output_shape->add_dim() =
      ctx.getInputType(1)->tensor_type().shape().dim(1); // channels should be the second dim of second input.

  int kernel_shape_size = static_cast<int>(kernel_shape.size());
  for (int i = 0; i < kernel_shape_size; ++i) {
    auto newdim = final_output_shape->add_dim();
    if (!input_shape.dim(2 + i).has_dim_value()) {
      continue;
    }

    int64_t newdim_value = strides[i] * (input_shape.dim(2 + i).dim_value() - 1);
    newdim_value += kernel_shape[i];
    newdim_value -= pads[i];
    newdim_value -= pads[i + kernel_shape_size];

    // add in the initial position
    newdim->set_dim_value(newdim_value);
  }
}

static const char* MaxUnpool_ver9_doc = R"DOC(
MaxUnpool essentially computes the partial inverse of the MaxPool op.
 The input information to this op is typically the output information from a MaxPool op. The first
 input tensor X is the tensor that needs to be unpooled, which is typically the pooled tensor (first output)
 from MaxPool. The second input tensor, I, contains the indices to the (locally maximal) elements corrsponding
 to the elements in the first input tensor X. Input tensor I is typically the second output of the MaxPool op.
 The third (optional) input is a tensor that specifies the output size of the unpooling operation.

MaxUnpool is intended to do 'partial' inverse of the MaxPool op. 'Partial' because all the non-maximal
 values from the original input to MaxPool are set to zero in the output of the MaxUnpool op. Pooling
 the result of an unpooling operation should give back the original input to the unpooling op.

MaxUnpool can produce the same output size for several input sizes, which makes unpooling op ambiguous.
 The third input argument, output_size, is meant to disambiguate the op and produce output tensor of
 known/predictable size.

In addition to the inputs, MaxUnpool takes three attributes, namely kernel_shape, strides, and pads,
 which define the exact unpooling op. The attributes typically have the same values as the corrsponding
 pooling op that the unpooling op is trying to invert.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    MaxUnpool,
    11,
    OpSchema()
        .SetDoc(MaxUnpool_ver9_doc)
        .Attr("kernel_shape", "The size of the kernel along each axis.", AttributeProto::INTS)
        .Attr(
            "strides",
            "Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr("pads", pads_doc, AttributeProto::INTS, OPTIONAL_VALUE)
        .Input(
            0,
            "X",
            "Input data tensor that has to be unpooled. "
            "This tensor is typically the first output of the MaxPool op."
            "Dimensions for image case are (N x C x H x W), "
            "where N is the batch size, C is the number of "
            "channels, and H and W are the height and the "
            "width of the data. For non-image case, the "
            "dimensions are in the form of "
            "(N x C x D1 x D2 ... Dn), where N is the batch "
            "size. Optionally, if dimension denotation is "
            "in effect, the operation expects the input "
            "data tensor to arrive with the dimension denotation "
            "of [DATA_BATCH, DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].",
            "T1",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .Input(
            1,
            "I",
            "Input data tensor containing the indices corresponding to "
            "elements in the first input tensor X."
            "This tensor is typically the second output of the MaxPool op."
            "Dimensions must be the same as input tensor X. "
            "The indices are linear, i.e. computed considering the tensor as flattened 1-D tensor, "
            "assuming row-major storage. Also, the linear indices should not consider padding. "
            "So the values in indices are in the range [0, N x C x D1 x ... x Dn).",
            "T2",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Input(
            2,
            "output_shape",
            "The shape of the output can be explicitly set which will cause pads values to be auto generated. If 'output_shape' is specified, "
            "'pads' values are ignored.",
            "T2",
            OpSchema::Optional,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Output(
            0,
            "output",
            "Output data tensor that contains the result of the unpooling.",
            "T1",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .TypeConstraint(
            "T1",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeConstraint("T2", {"tensor(int64)"}, "Constrain index tensor to int64")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) { maxUnpoolShapeInference(ctx); }));

std::function<void(OpSchema&)> LpPoolOpSchemaGenerator(const char* name) {
  return [=](OpSchema& schema) {
    std::string doc;
    POPULATE_OP_DOC_STR(doc = R"DOC(
 {name} consumes an input tensor X and applies Lp pooling across
 the tensor according to kernel sizes, stride sizes, and pad lengths.
 Lp pooling consisting of computing the Lp norm on all values of a subset
 of the input tensor according to the kernel size and downsampling the
 data into the output tensor Y for further processing. The output spatial shape will be following:
 ```
 output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - {kernelSpatialShape}) / strides_spatial_shape[i] + 1)
 ```
 or
 ```
 output_spatial_shape[i] = ceil((input_spatial_shape[i] + pad_shape[i] - {kernelSpatialShape}) / strides_spatial_shape[i] + 1)
 ```
 if ceil_mode is enabled `pad_shape[i]` is the sum of pads along axis `i`.

 `auto_pad` is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following:
 ```
 VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - {kernelSpatialShape} + 1) / strides_spatial_shape[i])
 SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
 ```
 And pad shape will be following if `SAME_UPPER` or `SAME_LOWER`:
 ```
 pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + {kernelSpatialShape} - input_spatial_shape[i]
 ```)DOC";
                        ReplaceAll(doc, "{name}", name););
    schema.SetDoc(doc);
    schema.Attr("kernel_shape", "The size of the kernel along each axis.", AttributeProto::INTS);
    schema.Attr(
        "strides",
        "Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis.",
        AttributeProto::INTS,
        OPTIONAL_VALUE);
    schema.Attr(
        "dilations",
        "dilation value along each spatial axis of the filter. If not present, the dilation defaults is 1 along each spatial axis.",
        AttributeProto::INTS,
        OPTIONAL_VALUE);
    schema.Attr("auto_pad", conv_auto_pad_doc, AttributeProto::STRING, std::string("NOTSET"));
    schema.Attr("pads", pads_doc, AttributeProto::INTS, OPTIONAL_VALUE);
    schema.Attr(
        "p", "p value of the Lp norm used to pool over the input data.", AttributeProto::INT, static_cast<int64_t>(2));
    schema.Attr(
        "ceil_mode",
        "Whether to use ceil or floor (default) to compute the output shape.",
        AttributeProto::INT,
        static_cast<int64_t>(0));
    schema.Input(
        0,
        "X",
        "Input data tensor from the previous operator; "
        "dimensions for image case are (N x C x H x W), "
        "where N is the batch size, C is the number of "
        "channels, and H and W are the height and the "
        "width of the data. For non image case, the "
        "dimensions are in the form of "
        "(N x C x D1 x D2 ... Dn), where N is the "
        "batch size.",
        "T",
        OpSchema::Single,
        true,
        1,
        OpSchema::Differentiable);
    schema.Output(
        0,
        "Y",
        "Output data tensor from Lp pooling across the input "
        "tensor. Dimensions will vary based on various kernel, stride, and pad "
        "sizes.",
        "T",
        OpSchema::Single,
        true,
        1,
        OpSchema::Differentiable);
    schema.TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");
    schema.TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      convPoolShapeInference(ctx, true, true, 0, 1);
    });
  };
}

ONNX_OPERATOR_SET_SCHEMA(LpPool, 18, OpSchema().FillUsing(LpPoolOpSchemaGenerator("LpPool")));

// For ROI pool operations.
void roiPoolTypeShapeInference(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);

  // rois is the second input.
  if (!hasNInputShapes(ctx, 2)) {
    return;
  }

  auto input_shape = ctx.getInputType(0)->tensor_type().shape();
  auto rios_shape = ctx.getInputType(1)->tensor_type().shape();

  if (input_shape.dim_size() < 2) {
    fail_shape_inference("Input tensor must have at least 2 dimensions");
  }
  if (rios_shape.dim_size() != 2) {
    fail_shape_inference("RoIs tensor must have 2 dimensions");
  }

  // first dim is the batch axis and the next is the number of channels.
  size_t n_input_dims = static_cast<size_t>(input_shape.dim_size() - 2);

  std::vector<int64_t> pooled_shape;
  if (getRepeatedAttribute(ctx, "pooled_shape", pooled_shape)) {
    if (pooled_shape.size() != n_input_dims) {
      fail_shape_inference("Attribute pooled_shape has incorrect length");
    }
  } else {
    fail_shape_inference("Attribute pooled_shape must be specified");
  }

  // (num_rois, channels, pooled_shape[0], pooled_shape[1])
  auto output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

  *output_shape->add_dim() = rios_shape.dim(0);
  *output_shape->add_dim() = input_shape.dim(1);
  output_shape->add_dim()->set_dim_value(pooled_shape[0]);
  output_shape->add_dim()->set_dim_value(pooled_shape[1]);
}

std::function<void(OpSchema&)> RoiPoolOpSchemaGenerator(const char* name) {
  return [=](OpSchema& schema) {
    std::string doc;
    POPULATE_OP_DOC_STR(doc = R"DOC(
 ROI {name} pool consumes an input tensor X and region of interests (RoIs) to
 apply {name} pooling across each RoI, to produce output 4-D tensor of shape
 (num_rois, channels, pooled_shape[0], pooled_shape[1]).)DOC";
                        ReplaceAll(doc, "{name}", name););
    schema.SetDoc(doc);
    schema.Attr("pooled_shape", "ROI pool output shape (height, width).", AttributeProto::INTS);
    schema.Attr(
        "spatial_scale",
        "Multiplicative spatial scale factor to translate ROI coordinates from their input scale to the scale used when pooling.",
        AttributeProto::FLOAT,
        1.f);
    schema.Input(
        0,
        "X",
        "Input data tensor from the previous operator; "
        "dimensions for image case are (N x C x H x W), "
        "where N is the batch size, C is the number of "
        "channels, and H and W are the height and the "
        "width of the data.",
        "T",
        OpSchema::Single,
        true,
        1,
        OpSchema::Differentiable);
    schema.Input(
        1,
        "rois",
        "RoIs (Regions of Interest) to pool over. Should "
        "be a 2-D tensor of shape (num_rois, 5) given as "
        "[[batch_id, x1, y1, x2, y2], ...].",
        "T",
        OpSchema::Single,
        true,
        1,
        OpSchema::NonDifferentiable);
    schema.Output(
        0,
        "Y",
        "RoI pooled output 4-D tensor of shape (num_rois, channels, pooled_shape[0], pooled_shape[1]).",
        "T",
        OpSchema::Single,
        true,
        1,
        OpSchema::Differentiable);
    schema.TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");
    schema.TypeAndShapeInferenceFunction([](InferenceContext& ctx) { roiPoolTypeShapeInference(ctx); });
  };
}

ONNX_OPERATOR_SET_SCHEMA(MaxRoiPool, 1, OpSchema().FillUsing(RoiPoolOpSchemaGenerator("max")));

std::function<void(OpSchema&)> ConvOpSchemaGenerator(const char* filter_desc) {
  return [=](OpSchema& schema) {
    std::string doc;
    POPULATE_OP_DOC_STR(doc = R"DOC(
The convolution operator consumes an input tensor and {filter_desc}, and
computes the output.)DOC";
                        ReplaceAll(doc, "{filter_desc}", filter_desc););
    schema.SetDoc(doc);
    schema.Input(
        0,
        "X",
        "Input data tensor from previous layer; "
        "has size (N x C x H x W), where N is the batch size, "
        "C is the number of channels, and H and W are the "
        "height and width. Note that this is for the 2D image. "
        "Otherwise the size is (N x C x D1 x D2 ... x Dn). "
        "Optionally, if dimension denotation is "
        "in effect, the operation expects input data tensor "
        "to arrive with the dimension denotation of [DATA_BATCH, "
        "DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].",
        "T",
        OpSchema::Single,
        true,
        1,
        OpSchema::Differentiable);
    schema.Input(
        1,
        "W",
        "The weight tensor that will be used in the "
        "convolutions; has size (M x C/group x kH x kW), where C "
        "is the number of channels, and kH and kW are the "
        "height and width of the kernel, and M is the number "
        "of feature maps. For more than 2 dimensions, the "
        "kernel shape will be (M x C/group x k1 x k2 x ... x kn), "
        "where (k1 x k2 x ... kn) is the dimension of the kernel. "
        "Optionally, if dimension denotation is in effect, "
        "the operation expects the weight tensor to arrive "
        "with the dimension denotation of [FILTER_OUT_CHANNEL, "
        "FILTER_IN_CHANNEL, FILTER_SPATIAL, FILTER_SPATIAL ...]. "
        "Assuming zero based indices for the shape array, "
        "X.shape[1] == (W.shape[1] * group) == C and "
        "W.shape[0] mod G == 0. Or in other words "
        "FILTER_IN_CHANNEL multiplied by the number of groups "
        "should be equal to DATA_CHANNEL and the number of "
        "feature maps M should be a multiple of the number of "
        "groups G.",
        "T",
        OpSchema::Single,
        true,
        1,
        OpSchema::Differentiable);
    schema.Input(
        2,
        "B",
        "Optional 1D bias to be added to the convolution, has size of M.",
        "T",
        OpSchema::Optional,
        true,
        1,
        OpSchema::Differentiable);
    schema.Output(
        0,
        "Y",
        "Output data tensor that contains the result of the "
        "convolution. The output dimensions are functions "
        "of the kernel size, stride size, and pad lengths.",
        "T",
        OpSchema::Single,
        true,
        1,
        OpSchema::Differentiable);
    schema.TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");
    schema.Attr(
        "kernel_shape",
        "The shape of the convolution kernel. If not present, should be inferred from input W.",
        AttributeProto::INTS,
        OPTIONAL_VALUE);
    schema.Attr(
        "dilations",
        "dilation value along each spatial axis of the filter. If not present, the dilation defaults is 1 along each spatial axis.",
        AttributeProto::INTS,
        OPTIONAL_VALUE);
    schema.Attr(
        "strides",
        "Stride along each spatial axis. If not present, the stride defaults is 1 along each spatial axis.",
        AttributeProto::INTS,
        OPTIONAL_VALUE);
    schema.Attr("auto_pad", conv_auto_pad_doc, AttributeProto::STRING, std::string("NOTSET"));
    schema.Attr("pads", pads_doc, AttributeProto::INTS, OPTIONAL_VALUE);
    schema.Attr(
        "group",
        "number of groups input channels and output channels are divided into.",
        AttributeProto::INT,
        static_cast<int64_t>(1));
    schema.TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      convPoolShapeInference(ctx, true, false, 0, 1);
    });
  };
}

ONNX_OPERATOR_SET_SCHEMA(Conv, 11, OpSchema().FillUsing(ConvOpSchemaGenerator("a filter")));

static const char* QLinearConv_ver10_doc = R"DOC(
The convolution operator consumes a quantized input tensor, its scale and zero point,
a quantized filter, its scale and zero point, and output's scale and zero point,
and computes the quantized output. Each scale and zero-point pair must have same shape.
It means they must be either scalars (per tensor) or 1-D tensors (per output channel).
Each input or output and its related zero point must have same type.
When bias is present it must be quantized using scale = input scale * weight scale and
zero point as 0.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    QLinearConv,
    10,
    OpSchema()
        .SetDoc(QLinearConv_ver10_doc)
        .Input(
            0,
            "x",
            "Input data tensor from previous layer; "
            "has size (N x C x H x W), where N is the batch size, "
            "C is the number of channels, and H and W are the "
            "height and width. Note that this is for the 2D image. "
            "Otherwise the size is (N x C x D1 x D2 ... x Dn). "
            "Optionally, if dimension denotation is "
            "in effect, the operation expects input data tensor "
            "to arrive with the dimension denotation of [DATA_BATCH, "
            "DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].",
            "T1")
        .Input(
            1,
            "x_scale",
            "Scale tensor for input 'x'. It's a scalar, which means a per-tensor/layer quantization.",
            "tensor(float)")
        .Input(
            2,
            "x_zero_point",
            "Zero point tensor for input 'x'. It's a scalar, which means a per-tensor/layer quantization.",
            "T1")
        .Input(
            3,
            "w",
            "The weight tensor that will be used in the "
            "convolutions; has size (M x C/group x kH x kW), where C "
            "is the number of channels, and kH and kW are the "
            "height and width of the kernel, and M is the number "
            "of feature maps. For more than 2 dimensions, the "
            "kernel shape will be (M x C/group x k1 x k2 x ... x kn), "
            "where (k1 x k2 x ... kn) is the dimension of the kernel. "
            "Optionally, if dimension denotation is in effect, "
            "the operation expects the weight tensor to arrive "
            "with the dimension denotation of [FILTER_OUT_CHANNEL, "
            "FILTER_IN_CHANNEL, FILTER_SPATIAL, FILTER_SPATIAL ...]. "
            "X.shape[1] == (W.shape[1] * group) == C "
            "(assuming zero based indices for the shape array). "
            "Or in other words FILTER_IN_CHANNEL should be equal to DATA_CHANNEL. ",
            "T2")
        .Input(
            4,
            "w_scale",
            "Scale tensor for input 'w'. It could be a scalar or a 1-D tensor, which means a per-tensor/layer or per output channel quantization. If it's a 1-D tensor, its number of elements should be equal to the number of output channels (M).",
            "tensor(float)")
        .Input(
            5,
            "w_zero_point",
            "Zero point tensor for input 'w'. It could be a scalar or a 1-D tensor, which means a per-tensor/layer or per output channel quantization. If it's a 1-D tensor, its number of elements should be equal to the number of output channels (M).",
            "T2")
        .Input(
            6,
            "y_scale",
            "Scale tensor for output 'y'. It's a scalar, which means a per-tensor/layer quantization.",
            "tensor(float)")
        .Input(
            7,
            "y_zero_point",
            "Zero point tensor for output 'y'. It's a scalar, which means a per-tensor/layer quantization.",
            "T3")
        .Input(
            8,
            "B",
            "Optional 1D bias to be added to the convolution, has size of M. "
            "Bias must be quantized using scale = x_scale * w_scale and zero_point = 0",
            "T4",
            OpSchema::Optional)
        .Output(
            0,
            "y",
            "Output data tensor that contains the result of the "
            "convolution. The output dimensions are functions "
            "of the kernel size, stride size, and pad lengths.",
            "T3")
        .TypeConstraint("T1", {"tensor(int8)", "tensor(uint8)"}, "Constrain input type to 8-bit integer tensor.")
        .TypeConstraint("T2", {"tensor(int8)", "tensor(uint8)"}, "Constrain filter type to 8-bit integer tensor.")
        .TypeConstraint("T3", {"tensor(int8)", "tensor(uint8)"}, "Constrain output type to 8-bit integer tensor.")
        .TypeConstraint("T4", {"tensor(int32)"}, "Constrain bias type to 32-bit integer tensor.")
        .Attr("auto_pad", conv_auto_pad_doc, AttributeProto::STRING, std::string("NOTSET"))
        .Attr(
            "kernel_shape",
            "The shape of the convolution kernel. If not present, should be inferred from input 'w'.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr(
            "dilations",
            "dilation value along each spatial axis of the filter. If not present, the dilation defaults to 1 along each spatial axis.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr(
            "strides",
            "Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr(
            "pads",
            "Padding for the beginning and ending along each spatial axis, it can take any value greater than or equal to 0."
            "The value represent the number of pixels added to the beginning and end part of the corresponding axis."
            "`pads` format should be as follow [x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin the number of"
            "pixels added at the beginning of axis `i` and xi_end, the number of pixels added at the end of axis `i`."
            "This attribute cannot be used simultaneously with auto_pad attribute. If not present, the padding defaults"
            "to 0 along start and end of each spatial axis.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr(
            "group",
            "number of groups input channels and output channels are divided into. default is 1.",
            AttributeProto::INT,
            static_cast<int64_t>(1))
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          auto x_type = ctx.getInputType(0);
          auto w_type = ctx.getInputType(3);
          if (nullptr == x_type || nullptr == w_type || x_type->value_case() != TypeProto::kTensorType ||
              w_type->value_case() != TypeProto::kTensorType) {
            fail_type_inference("inputs are expected to have tensor type.");
          }

          auto x_zero_point_type = ctx.getInputType(2);
          if (nullptr == x_zero_point_type ||
              x_zero_point_type->tensor_type().elem_type() != x_type->tensor_type().elem_type()) {
            fail_type_inference("input and zero_point pair is expected to have be same type.");
          }

          auto w_zero_point_type = ctx.getInputType(5);
          if (nullptr == w_zero_point_type ||
              w_zero_point_type->tensor_type().elem_type() != w_type->tensor_type().elem_type()) {
            fail_type_inference("weight and zero_point pair is expected to have same type.");
          }

          propagateElemTypeFromInputToOutput(ctx, 7, 0);

          convPoolShapeInference(ctx, true, false, 0, 3);
        }));

static const char* ConvInteger_ver10_doc = R"DOC(
The integer convolution operator consumes an input tensor, its zero-point, a filter, and its zero-point,
and computes the output. The production MUST never overflow. The accumulation may overflow if and only if in 32 bits.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    ConvInteger,
    10,
    OpSchema()
        .SetDoc(ConvInteger_ver10_doc)
        .Input(
            0,
            "x",
            "Input data tensor from previous layer; "
            "has size (N x C x H x W), where N is the batch size, "
            "C is the number of channels, and H and W are the "
            "height and width. Note that this is for the 2D image. "
            "Otherwise the size is (N x C x D1 x D2 ... x Dn). "
            "Optionally, if dimension denotation is "
            "in effect, the operation expects input data tensor "
            "to arrive with the dimension denotation of [DATA_BATCH, "
            "DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].",
            "T1")
        .Input(
            1,
            "w",
            "The weight tensor that will be used in the "
            "convolutions; has size (M x C/group x kH x kW), where C "
            "is the number of channels, and kH and kW are the "
            "height and width of the kernel, and M is the number "
            "of feature maps. For more than 2 dimensions, the "
            "kernel shape will be (M x C/group x k1 x k2 x ... x kn), "
            "where (k1 x k2 x ... kn) is the dimension of the kernel. "
            "Optionally, if dimension denotation is in effect, "
            "the operation expects the weight tensor to arrive "
            "with the dimension denotation of [FILTER_OUT_CHANNEL, "
            "FILTER_IN_CHANNEL, FILTER_SPATIAL, FILTER_SPATIAL ...]. "
            "X.shape[1] == (W.shape[1] * group) == C "
            "(assuming zero based indices for the shape array). "
            "Or in other words FILTER_IN_CHANNEL should be equal to DATA_CHANNEL. ",
            "T2")
        .Input(
            2,
            "x_zero_point",
            "Zero point tensor for input 'x'. It's optional and default value is 0. It's a scalar, which means a per-tensor/layer quantization.",
            "T1",
            OpSchema::Optional)
        .Input(
            3,
            "w_zero_point",
            "Zero point tensor for input 'w'. It's optional and default value is 0.  It could be a scalar or a 1-D tensor, "
            "which means a per-tensor/layer or per output channel quantization. If it's a 1-D tensor, its number "
            "of elements should be equal to the number of output channels (M)",
            "T2",
            OpSchema::Optional)
        .Output(
            0,
            "y",
            "Output data tensor that contains the result of the "
            "convolution. The output dimensions are functions "
            "of the kernel size, stride size, and pad lengths.",
            "T3")
        .TypeConstraint(
            "T1",
            {"tensor(int8)", "tensor(uint8)"},
            "Constrain input x and its zero point data type to 8-bit integer tensor.")
        .TypeConstraint(
            "T2",
            {"tensor(int8)", "tensor(uint8)"},
            "Constrain input w and its zero point data type to 8-bit integer tensor.")
        .TypeConstraint("T3", {"tensor(int32)"}, "Constrain output y data type to 32-bit integer tensor.")
        .Attr("auto_pad", conv_auto_pad_doc, AttributeProto::STRING, std::string("NOTSET"))
        .Attr(
            "kernel_shape",
            "The shape of the convolution kernel. If not present, should be inferred from input 'w'.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr(
            "dilations",
            "dilation value along each spatial axis of the filter. If not present, the dilation defaults to 1 along each axis.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr(
            "strides",
            "Stride along each spatial axis. If not present, the stride defaults to 1 along each axis.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr(
            "pads",
            "Padding for the beginning and ending along each spatial axis, it can take any value greater than or equal to 0."
            "The value represent the number of pixels added to the beginning and end part of the corresponding axis."
            "`pads` format should be as follow [x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin the number of"
            "pixels added at the beginning of axis `i` and xi_end, the number of pixels added at the end of axis `i`."
            "This attribute cannot be used simultaneously with auto_pad attribute. If not present, the padding defaults"
            "to 0 along start and end of each spatial axis.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr(
            "group",
            "number of groups input channels and output channels are divided into. default is 1.",
            AttributeProto::INT,
            static_cast<int64_t>(1))
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          auto x_type = ctx.getInputType(0);
          auto w_type = ctx.getInputType(1);
          auto y_type = ctx.getOutputType(0);
          if (nullptr == x_type || nullptr == w_type || nullptr == y_type ||
              x_type->value_case() != TypeProto::kTensorType || w_type->value_case() != TypeProto::kTensorType) {
            fail_type_inference("inputs are expected to have tensor type and output type should not be null.");
          }

          // Right now we only support int32
          y_type->mutable_tensor_type()->set_elem_type(TensorProto::INT32);

          convPoolShapeInference(ctx, true, false, 0, 1);
        }));

void convTransposeShapeInference(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);

  // we need at least two inputs to have a shape for this inference.
  if (!hasNInputShapes(ctx, 2)) {
    return;
  }

  int64_t group = getAttribute(ctx, "group", 1);

  auto input_shape = ctx.getInputType(0)->tensor_type().shape();
  if (input_shape.dim_size() < 2) {
    return; // Input tensor should have at least two dimensions.
  }

  // first dim is the batch axis and the next is the number of channels.
  size_t n_input_dims = static_cast<size_t>(input_shape.dim_size() - 2);

  std::vector<int64_t> dilations;
  if (getRepeatedAttribute(ctx, "dilations", dilations)) {
    if (dilations.size() != n_input_dims) {
      return;
    }
  } else {
    dilations.assign(n_input_dims, 1);
  }

  std::vector<int64_t> strides;
  if (getRepeatedAttribute(ctx, "strides", strides)) {
    if (strides.size() != n_input_dims) {
      return;
    }
  } else {
    strides.assign(n_input_dims, 1);
  }

  std::vector<int64_t> kernel_shape;
  if (getRepeatedAttribute(ctx, "kernel_shape", kernel_shape)) {
    if (kernel_shape.size() != n_input_dims) {
      return;
    }
  } else {
    auto second_input_shape = ctx.getInputType(1)->tensor_type().shape();
    for (int i = 2; i < second_input_shape.dim_size(); ++i) {
      if (!second_input_shape.dim(i).has_dim_value()) {
        return;
      }
      kernel_shape.push_back(second_input_shape.dim(i).dim_value());
    }
  }

  std::vector<int64_t> effective_kernel_shape = kernel_shape;
  for (int i = 0; i < static_cast<int>(kernel_shape.size()); i++) {
    // accounting for dilation, how big is the kernel in this dimension
    effective_kernel_shape[i] = (effective_kernel_shape[i] - 1) * dilations[i] + 1;
  }

  std::vector<int64_t> pads;
  if (getRepeatedAttribute(ctx, "pads", pads)) {
    if (pads.size() != n_input_dims * 2) {
      fail_shape_inference("Attribute pads has incorrect size");
    }
    const auto* auto_pad_attr = ctx.getAttribute("auto_pad");
    if (nullptr != auto_pad_attr && auto_pad_attr->s() != "NOTSET") {
      fail_shape_inference("The pads attribute cannot be used simultaneously with auto_pad attribute");
    }
  } else {
    pads.assign(n_input_dims * 2, 0);
    const auto* auto_pad_attr = ctx.getAttribute("auto_pad");
    if ((nullptr != auto_pad_attr) && (auto_pad_attr->s() != "VALID")) {
      int input_dims_size = static_cast<int>(n_input_dims);
      for (int i = 0; i < input_dims_size; ++i) {
        int64_t total_pad = effective_kernel_shape[i] - strides[i];
        if (total_pad < 0)
          total_pad = 0;
        int64_t half_pad_small = total_pad >> 1;
        int64_t half_pad_big = total_pad - half_pad_small;
        if (auto_pad_attr->s() == "SAME_UPPER") {
          pads[i] = half_pad_small;
          pads[i + input_dims_size] = half_pad_big;
        } else if (auto_pad_attr->s() == "SAME_LOWER") {
          pads[i] = half_pad_big;
          pads[i + input_dims_size] = half_pad_small;
        }
      }
    }
  }

  std::vector<int64_t> output_shape;
  bool output_shape_presented = true;
  if (getRepeatedAttribute(ctx, "output_shape", output_shape)) {
    if (output_shape.size() != n_input_dims) {
      return;
    }
  } else {
    output_shape_presented = false;
  }

  std::vector<int64_t> output_padding;
  if (getRepeatedAttribute(ctx, "output_padding", output_padding)) {
    if (output_padding.size() != n_input_dims) { // Added only to one side.
      return;
    }
  } else {
    output_padding.assign(n_input_dims, 0);
  }

  auto final_output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

  *final_output_shape->add_dim() = input_shape.dim(0);
  *final_output_shape->add_dim() =
      ctx.getInputType(1)->tensor_type().shape().dim(1) * group; // channels should be the second dim of second input
                                                                 // multiply group.

  int size_of_output;
  if (output_shape_presented) {
    size_of_output = static_cast<int>(output_shape.size());
    for (int i = 0; i < size_of_output; ++i) {
      if (input_shape.dim(i + 2).has_dim_value()) {
        if (output_shape[i] < input_shape.dim(i + 2).dim_value()) {
          // TODO: throw exception?
          return; // output shape value cannot be smaller than the input shape
                  // value
        }
      }
      final_output_shape->add_dim()->set_dim_value(output_shape[i]);
    }
    return;
  } else {
    size_of_output = input_shape.dim_size() - 2;
    for (int i = 0; i < size_of_output; ++i) {
      if (input_shape.dim(i + 2).has_dim_value()) {
        int64_t output_shape_dim = strides[i] * (input_shape.dim(i + 2).dim_value() - 1) + output_padding[i] +
            effective_kernel_shape[i] - pads[i] - pads[i + n_input_dims];
        final_output_shape->add_dim()->set_dim_value(output_shape_dim);
      } else {
        final_output_shape->add_dim();
      }
    }
    return;
  }
}

std::function<void(OpSchema&)> ConvTransposeOpSchemaGenerator(const char* filter_desc) {
  return [=](OpSchema& schema) {
    std::string doc;
    POPULATE_OP_DOC_STR(doc = R"DOC(
The convolution transpose operator consumes an input tensor and {filter_desc},
and computes the output.

If the pads parameter is provided the shape of the output is calculated via the following equation:

  output_shape[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - pads[start_i] - pads[end_i]

output_shape can also be explicitly specified in which case pads values are auto generated using these equations:

  total_padding[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - output_shape[i]
  If (auto_pads == SAME_UPPER): pads[start_i] = total_padding[i]/2; pads[end_i] = total_padding[i] - (total_padding[i]/2)
  Else: pads[start_i] = total_padding[i] - (total_padding[i]/2); pads[end_i] = (total_padding[i]/2).

    )DOC";
                        ReplaceAll(doc, "{filter_desc}", filter_desc););
    schema.SetDoc(doc);
    schema.Input(
        0,
        "X",
        "Input data tensor from previous layer; has size (N x C x H x W)"
        ", where N is the batch size, C is the number of channels, and"
        " H and W are the height and width. Note that this is for the 2D image. "
        "Otherwise the size is (N x C x D1 x D2 ... x Dn)",
        "T",
        OpSchema::Single,
        true,
        1,
        OpSchema::Differentiable);
    schema.Input(
        1,
        "W",
        "The weight tensor that will be used in the "
        "convolutions; has size (C x M/group x kH x kW), where C "
        "is the number of channels, and kH and kW are the "
        "height and width of the kernel, and M is the number "
        "of feature maps. For more than 2 dimensions, the "
        "weight shape will be (C x M/group x k1 x k2 x ... x kn), "
        "where (k1 x k2 x ... x kn) is the dimension of the kernel. "
        "The number of channels in the output should be equal to W.shape[1] * group "
        "(assuming zero based indices of the shape array)",
        "T",
        OpSchema::Single,
        true,
        1,
        OpSchema::Differentiable);
    schema.Input(
        2,
        "B",
        "Optional 1D bias to be added to the convolution, has size of M.",
        "T",
        OpSchema::Optional,
        true,
        1,
        OpSchema::Differentiable);
    schema.Output(
        0,
        "Y",
        "Output data tensor that contains the result of the convolution. The "
        "output dimensions are functions of the kernel size, stride size, "
        "pad lengths and group count. "
        "The number of channels in the output should be equal to W.shape[1] * group "
        "(assuming zero based indices of the shape array)",
        "T",
        OpSchema::Single,
        true,
        1,
        OpSchema::Differentiable);
    schema.TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");
    schema.Attr(
        "kernel_shape",
        "The shape of the convolution kernel. If not present, should be inferred from input W.",
        AttributeProto::INTS,
        OPTIONAL_VALUE);
    schema.Attr(
        "output_shape",
        "The shape of the output can be explicitly set which will cause pads values to be auto generated. If output_shape is specified "
        "pads values are ignored. See doc for details for equations to generate pads",
        AttributeProto::INTS,
        OPTIONAL_VALUE);
    schema.Attr(
        "output_padding",
        "Additional elements added to the side with higher coordinate indices in the output. "
        "Each padding value in \"output_padding\" must be less than the corresponding stride/dilation dimension. "
        "By default, this attribute is a zero vector. "
        "Note that this attribute doesn't directly affect the computed output values. "
        "It only controls the selection of the computed values, "
        "so changing this attribute only adds or removes output elements. "
        "If \"output_shape\" is explicitly provided, "
        "\"output_padding\" does not contribute additional size to \"output_shape\" but "
        "participates in the computation of the needed padding amount. "
        "This is also called adjs or adjustment in some frameworks.",
        AttributeProto::INTS,
        OPTIONAL_VALUE);
    schema.Attr(
        "dilations",
        "dilation value along each spatial axis of the filter. If not present, the dilation defaults to 1 along each spatial axis.",
        AttributeProto::INTS,
        OPTIONAL_VALUE);
    schema.Attr(
        "strides",
        "Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis.",
        AttributeProto::INTS,
        OPTIONAL_VALUE);
    schema.Attr("auto_pad", conv_transpose_auto_pad_doc, AttributeProto::STRING, std::string("NOTSET"));
    schema.Attr("pads", pads_doc, AttributeProto::INTS, OPTIONAL_VALUE);
    schema.Attr(
        "group",
        "number of groups input channels and output channels are divided into.",
        AttributeProto::INT,
        static_cast<int64_t>(1));
    schema.TypeAndShapeInferenceFunction([](InferenceContext& ctx) { convTransposeShapeInference(ctx); });
  };
}

ONNX_OPERATOR_SET_SCHEMA(ConvTranspose, 11, OpSchema().FillUsing(ConvTransposeOpSchemaGenerator("a filter")));

static const char* DeformConv_ver19_doc = R"DOC(
Performs deformable convolution as described in https://arxiv.org/abs/1703.06211 and https://arxiv.org/abs/1811.11168.
This operator specification supports the general N-D case. Note that most common use cases have 2D or 3D data.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    DeformConv,
    19,
    OpSchema()
        .SetDoc(DeformConv_ver19_doc)
        .Input(
            0,
            "X",
            "Input data tensor. For 2D image data, it has shape (N, C, H, W) where N is the batch size, "
            "C is the number of input channels, and H and W are the height and width. "
            "In general, the shape is (N, C, D1, D2, ... , Dn) for n-dimensional data, where "
            "D1 to Dn are the spatial dimension sizes. Most common use cases have n = 2 or 3.",
            "T")
        .Input(
            1,
            "W",
            "Weight tensor that will be used in the convolutions. It has shape (oC, C/group, kH, kW), "
            "where oC is the number of output channels and kH and kW are the kernel height and width. "
            "For more than 2 dimensions, it has shape (oC, C/group, k1, k2, ... , kn).",
            "T")
        .Input(
            2,
            "offset",
            "Offset tensor denoting the offset for the sampling locations in the convolution kernel. "
            "It has shape (N, offset_group * kH * kW * 2, oH, oW) for 2D data or "
            "(N, offset_group * k1 * k2 * ... * kn * n, o1, o2, ... , on) for nD data. Use linear interpolation"
            "for fractional offset values. Sampling locations outside of the padded input tensor gives zero.",
            "T")
        .Input(
            3,
            "B",
            "Optional 1D bias of length oC to be added to the convolution. Default is a tensor of zeros.",
            "T",
            OpSchema::Optional)
        .Input(
            4,
            "mask",
            "The mask tensor to be applied to each position in the convolution kernel. "
            "It has shape (N, offset_group * kH * kW, oH, oW) for 2D data or "
            "(N, offset_group * k1 * k2 * ... * kn * n, o1, o2, ... , on) for nD data. Default is a "
            "tensor of ones.",
            "T",
            OpSchema::Optional)
        .Output(
            0,
            "Y",
            "Output data tensor that contains the result of convolution. It has shape (N, oC, oH, oW) "
            "for 2D data or (N, oC, o1, o2, ..., on) for nD data",
            "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .Attr(
            "dilations",
            "Dilation value along each spatial axis of the kernel. Default is 1 along each axis.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr(
            "group",
            "Number of groups the input and output channels, C and oC, are divided into. C and oC must both "
            "be divisible by group. Default is 1.",
            AttributeProto::INT,
            static_cast<int64_t>(1))
        .Attr(
            "kernel_shape",
            "Shape of the convolution kernel. If not present, it is inferred from the shape of input W.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr(
            "offset_group",
            "Number of groups of offset. C must be divisible by offset_group. Default is 1.",
            AttributeProto::INT,
            static_cast<int64_t>(1))
        .Attr(
            "pads",
            "Padding for the beginning and end along each spatial axis. The values represent the number of pixels "
            "added to the beginning and end of the corresponding axis and can take any nonnegative value. "
            "The format should be as follows: [x1_begin, x2_begin, ..., x1_end, x2_end, ...], where xi_begin "
            "is the number of pixels added at the beginning of axis `i` and xi_end is the number of pixels "
            "added at the end of axis `i`. Default is 0 along each axis.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr(
            "strides",
            "Stride along each spatial axis. Default is 1 along each axis.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          convPoolShapeInference(ctx, true, false, 0, 1);
        }));

// For GlobalPool operations.
void globalPoolTypeShapeInference(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);

  // needs at least one input with shape.
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }

  auto input_shape = ctx.getInputType(0)->tensor_type().shape();
  if (input_shape.dim_size() < 2) {
    return;
  }

  // first dim is the batch axis and the next is the number of channels.
  size_t n_input_dims = static_cast<size_t>(input_shape.dim_size() - 2);

  // (N, C, 1, 1, ..., 1)
  auto output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  *output_shape->add_dim() = input_shape.dim(0);
  *output_shape->add_dim() = input_shape.dim(1);

  for (size_t i = 0; i < n_input_dims; ++i) {
    output_shape->add_dim()->set_dim_value(1);
  }
}

std::function<void(OpSchema&)> GlobalPoolingOpSchemaGenerator(const char* op_type, const char* op) {
  return [=](OpSchema& schema) {
    std::string doc;
    POPULATE_OP_DOC_STR(doc = R"DOC(
 Global{op_type} consumes an input tensor X and applies {op} pooling across
 the values in the same channel. This is equivalent to {op_type} with kernel size
 equal to the spatial dimension of input tensor.)DOC";
                        ReplaceAll(doc, "{op_type}", op_type);
                        ReplaceAll(doc, "{op}", op););
    schema.SetDoc(doc);
    schema.Input(
        0,
        "X",
        "Input data tensor from the previous operator; "
        "dimensions for image case are (N x C x H x W), "
        "where N is the batch size, C is the number of "
        "channels, and H and W are the height and the width "
        "of the data. For non image case, the dimensions are "
        "in the form of (N x C x D1 x D2 ... Dn), "
        "where N is the batch size.",
        "T",
        OpSchema::Single,
        true,
        1,
        OpSchema::Differentiable);
    schema.Output(
        0,
        "Y",
        "Output data tensor from pooling across the input "
        "tensor. The output tensor has the same rank as the input. "
        "The first two dimensions of output shape are the same as "
        "the input (N x C), while the other dimensions are all 1.",
        "T",
        OpSchema::Single,
        true,
        1,
        OpSchema::Differentiable);
    schema.TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");
    schema.TypeAndShapeInferenceFunction([](InferenceContext& ctx) { globalPoolTypeShapeInference(ctx); });
  };
}
ONNX_OPERATOR_SET_SCHEMA(
    GlobalAveragePool,
    1,
    OpSchema().FillUsing(GlobalPoolingOpSchemaGenerator("AveragePool", "average")));
ONNX_OPERATOR_SET_SCHEMA(GlobalMaxPool, 1, OpSchema().FillUsing(GlobalPoolingOpSchemaGenerator("MaxPool", "max")));

std::function<void(OpSchema&)> GlobalLpPoolingOpSchemaGenerator(const char* op_type, const char* op) {
  return [=](OpSchema& schema) {
    std::string doc;
    POPULATE_OP_DOC_STR(doc = R"DOC(
 Global{op_type} consumes an input tensor X and applies {op} pooling across
 the values in the same channel. This is equivalent to {op_type} with kernel size
 equal to the spatial dimension of input tensor.)DOC";
                        ReplaceAll(doc, "{op_type}", op_type);
                        ReplaceAll(doc, "{op}", op););
    schema.SetDoc(doc);
    schema.Attr(
        "p", "p value of the Lp norm used to pool over the input data.", AttributeProto::INT, static_cast<int64_t>(2));
    schema.Input(
        0,
        "X",
        "Input data tensor from the previous operator; "
        "dimensions for image case are (N x C x H x W), "
        "where N is the batch size, C is the number of "
        "channels, and H and W are the height and the width "
        "of the data. For non image case, the dimensions are "
        "in the form of (N x C x D1 x D2 ... Dn), "
        "where N is the batch size.",
        "T",
        OpSchema::Single,
        true,
        1,
        OpSchema::Differentiable);
    schema.Output(
        0,
        "Y",
        "Output data tensor from pooling across the input "
        "tensor. The output tensor has the same rank as the input. "
        "The first two dimensions of output shape are the same as "
        "the input (N x C), while the other dimensions are all 1.",
        "T",
        OpSchema::Single,
        true,
        1,
        OpSchema::Differentiable);
    schema.TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");
    schema.TypeAndShapeInferenceFunction([](InferenceContext& ctx) { globalPoolTypeShapeInference(ctx); });
  };
}

ONNX_OPERATOR_SET_SCHEMA(GlobalLpPool, 2, OpSchema().FillUsing(GlobalLpPoolingOpSchemaGenerator("LpPool", "lp pool")));

static const char* BatchNormalization_ver15_doc = R"DOC(
Carries out batch normalization as described in the paper
https://arxiv.org/abs/1502.03167. Depending on the mode it is being run,
There are five required inputs 'X', 'scale', 'B', 'input_mean' and
'input_var'.
Note that 'input_mean' and 'input_var' are expected to be the estimated
statistics in inference mode (training_mode=False, default),
and the running statistics in training mode (training_mode=True).
There are multiple cases for the number of outputs, which we list below:

* Output case #1: Y, running_mean, running_var (training_mode=True)
* Output case #2: Y (training_mode=False)

When training_mode=False, extra outputs are invalid.
The outputs are updated as follows when training_mode=True:
```
running_mean = input_mean * momentum + current_mean * (1 - momentum)
running_var = input_var * momentum + current_var * (1 - momentum)

Y = (X - current_mean) / sqrt(current_var + epsilon) * scale + B
```
where:
```
current_mean = ReduceMean(X, axis=all_except_channel_index)
current_var =  ReduceVar(X, axis=all_except_channel_index)
```
Notice that `ReduceVar` refers to the population variance, and it equals to
`sum(sqrd(x_i - x_avg)) / N`
where `N` is the population size (this formula does not use sample size `N - 1`).

The computation of ReduceMean and ReduceVar uses float to avoid overflow for float16 inputs.

When training_mode=False:
```
Y = (X - input_mean) / sqrt(input_var + epsilon) * scale + B
```

For previous (depreciated) non-spatial cases, implementors are suggested
to flatten the input shape to (N x C * D1 * D2 * ... * Dn) before a BatchNormalization Op.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    BatchNormalization,
    15,
    OpSchema()
        .NumOutputs({1, 3})
        .SetDoc(BatchNormalization_ver15_doc + GenerateOptionalArgumentsDoc())
        .Attr("epsilon", "The epsilon value to use to avoid division by zero.", AttributeProto::FLOAT, 1e-5f)
        .Attr(
            "momentum",
            "Factor used in computing the running mean and variance."
            "e.g., running_mean = running_mean * momentum + mean * (1 - momentum).",
            AttributeProto::FLOAT,
            0.9f)
        .Attr(
            "training_mode",
            "If set to true, it indicates BatchNormalization is being used for training, and outputs 1, "
            "2, 3, and 4 would be populated.",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Input(
            0,
            "X",
            "Input data tensor from the previous operator; "
            "dimensions are in the form of (N x C x D1 x D2 ... Dn), "
            "where N is the batch size, C is the number of channels. "
            "Statistics are computed for every channel of C over N and D1 to Dn dimensions. "
            "For image data, input dimensions become (N x C x H x W). "
            "The op also accepts single dimension input of size N in which case C is assumed to be 1",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .Input(1, "scale", "Scale tensor of shape (C).", "T1", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Input(2, "B", "Bias tensor of shape (C).", "T1", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Input(
            3,
            "input_mean",
            "running (training) or estimated (testing) mean tensor of shape (C).",
            "T2",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .Input(
            4,
            "input_var",
            "running (training) or estimated (testing) variance tensor of shape (C).",
            "T2",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .Output(
            0,
            "Y",
            "The output tensor of the same shape as X",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .Output(
            1,
            "running_mean",
            "The running mean after the BatchNormalization operator.",
            "T2",
            OpSchema::Optional,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Output(
            2,
            "running_var",
            "The running variance after the BatchNormalization operator. This op uses the population size (N) for "
            "calculating variance, and not the sample size N-1.",
            "T2",
            OpSchema::Optional,
            true,
            1,
            OpSchema::NonDifferentiable)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
            "Constrain input and output types to float tensors.")
        .TypeConstraint(
            "T1",
            {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
            "Constrain scale and bias types to float tensors.")
        .TypeConstraint(
            "T2",
            {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
            "Constrain mean and variance types to float tensors.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateShapeAndTypeFromFirstInput(ctx);
          propagateShapeFromInputToOutput(ctx, 0, 0);

          // Inputs 1 to 4 must be of rank 1.
          checkInputRank(ctx, 1, 1);
          checkInputRank(ctx, 2, 1);
          checkInputRank(ctx, 3, 1);
          checkInputRank(ctx, 4, 1);

          Dim num_channels;

          if (hasInputShape(ctx, 0)) {
            if (getInputShape(ctx, 0).dim_size() > 1)
              unifyInputDim(ctx, 0, 1, num_channels);
            else
              unifyDim(num_channels, 1);
          }

          unifyInputDim(ctx, 1, 0, num_channels);
          unifyInputDim(ctx, 2, 0, num_channels);
          unifyInputDim(ctx, 3, 0, num_channels);
          unifyInputDim(ctx, 4, 0, num_channels);

          if (ctx.getAttribute("training_mode") && static_cast<int>(ctx.getAttribute("training_mode")->i()) != 0) {
            if (ctx.getNumOutputs() != 3)
              fail_shape_inference("This number of op outputs should be 3 when Training_mode = True, but it is not.");
          } else {
            if (ctx.getNumOutputs() != 1)
              fail_shape_inference("This number of op outputs should be 1 when Training_mode = False, but it is not.");
          }

          if (ctx.getNumOutputs() > 1) {
            TensorShapeProto outputs_shape;
            *outputs_shape.add_dim() = num_channels; // channel

            propagateElemTypeFromInputToOutput(ctx, 3, 1);
            updateOutputShape(ctx, 1, outputs_shape);

            if (ctx.getNumOutputs() > 2) {
              propagateElemTypeFromInputToOutput(ctx, 4, 2);
              updateOutputShape(ctx, 2, outputs_shape);
            }
          }
        }));

static const char* InstanceNormalization_ver6_doc = R"DOC(
Carries out instance normalization as described in the paper
https://arxiv.org/abs/1607.08022.

y = scale * (x - mean) / sqrt(variance + epsilon) + B,
where mean and variance are computed per instance per channel.

)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    InstanceNormalization,
    6,
    OpSchema()
        .SetDoc(InstanceNormalization_ver6_doc)
        .Attr("epsilon", "The epsilon value to use to avoid division by zero.", AttributeProto::FLOAT, 1e-5f)
        .Input(
            0,
            "input",
            "Input data tensor from the previous operator; "
            "dimensions for image case are (N x C x H x W), "
            "where N is the batch size, C is the number of "
            "channels, and H and W are the height and the "
            "width of the data. For non image case, the "
            "dimensions are in the form of "
            "(N x C x D1 x D2 ... Dn), where N is the batch "
            "size.",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .Input(
            1,
            "scale",
            "The input 1-dimensional scale tensor of size C.",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .Input(
            2,
            "B",
            "The input 1-dimensional bias tensor of size C.",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .Output(
            0,
            "output",
            "The output tensor of the same shape as input.",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) { propagateShapeAndTypeFromFirstInput(ctx); }));

static const char* LpNormalization_ver1_doc = R"DOC(
Given a matrix, apply Lp-normalization along the provided axis.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    LpNormalization,
    1,
    OpSchema()
        .Input(0, "input", "Input matrix", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Output(0, "output", "Matrix after normalization", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .SetDoc(LpNormalization_ver1_doc)
        .Attr(
            "axis",
            "The axis on which to apply normalization, -1 mean last axis.",
            AttributeProto::INT,
            static_cast<int64_t>(-1))
        .Attr(
            "p",
            "The order of the normalization, only 1 or 2 are supported.",
            AttributeProto::INT,
            static_cast<int64_t>(2))
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) { propagateShapeAndTypeFromFirstInput(ctx); }));

static const char* Dropout_ver13_doc = R"DOC(
Dropout takes an input floating-point tensor, an optional input ratio (floating-point scalar) and an optional input training_mode (boolean scalar). It produces two tensor outputs,
output (floating-point tensor) and mask (optional `Tensor<bool>`). If `training_mode` is true then the output Y will be a random dropout;
Note that this Dropout scales the masked input data by the following equation, so to convert the trained model into inference mode,
the user can simply not pass `training_mode` input or set it to false.
```
output = scale * data * mask,
```
where
```
scale = 1. / (1. - ratio).
```
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Dropout,
    13,
    OpSchema()
        .SetDoc(GET_OP_DOC_STR(std::string(Dropout_ver13_doc) + GenerateOptionalArgumentsDoc()))
        .Attr(
            "seed",
            "(Optional) Seed to the random generator, if not specified we will auto generate one.",
            AttributeProto::INT,
            OPTIONAL_VALUE)
        .Input(0, "data", "The input data as Tensor.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Input(
            1,
            "ratio",
            "The ratio of random dropout, with value in [0, 1). If this input was not set, "
            "or if it was set to 0, the output would be a simple copy of the input. "
            "If it's non-zero, output will be a random dropout of the scaled input, which is typically "
            "the case during training. It is an optional value, if not specified it will default to 0.5.",
            "T1",
            OpSchema::Optional,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Input(
            2,
            "training_mode",
            "If set to true then it indicates dropout is being used for training. It is an optional value hence unless "
            "specified explicitly, it is false. If it is false, ratio is ignored and the operation mimics inference mode where "
            "nothing will be dropped from the input data and if mask is requested as output it will contain all ones.",
            "T2",
            OpSchema::Optional,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Output(0, "output", "The output.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Output(1, "mask", "The output mask.", "T2", OpSchema::Optional, true, 1, OpSchema::NonDifferentiable)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
            "Constrain input and output types to float tensors.")
        .TypeConstraint(
            "T1",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input 'ratio' types to float tensors.")
        .TypeConstraint("T2", {"tensor(bool)"}, "Constrain output 'mask' types to boolean tensors.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (hasInputShape(ctx, 0)) {
            propagateShapeFromInputToOutput(ctx, 0, 0);
          }

          if (ctx.getNumInputs() > 1 && hasInputShape(ctx, 1)) {
            auto& ratio_input_shape = getInputShape(ctx, 1);
            if (static_cast<int>(ratio_input_shape.dim_size()) != 0) {
              fail_shape_inference("Ratio of Dropout must be a scalar.");
            }
          }

          if (ctx.getNumInputs() > 2 && hasInputShape(ctx, 2)) {
            auto& training_mode_input_shape = getInputShape(ctx, 2);
            if (static_cast<int>(training_mode_input_shape.dim_size()) != 0) {
              fail_shape_inference("training_mode of Dropout must be a scalar.");
            }
          }

          if (ctx.getNumOutputs() == 2) {
            updateOutputElemType(ctx, 1, TensorProto::BOOL);
            if (hasNInputShapes(ctx, 1)) {
              propagateShapeFromInputToOutput(ctx, 0, 1);
            }
          }
        }));

static const char* Shrink_ver9_doc = R"DOC(
Shrink takes one input data (Tensor<numeric>) and produces one Tensor output,
having same datatype and shape with input. It has two attributes, lambd and
bias. The formula of this operator is: If x < -lambd, y = x + bias;
If x > lambd, y = x - bias; Otherwise, y = 0.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Shrink,
    9,
    OpSchema()
        .SetDoc(Shrink_ver9_doc)
        .Attr("lambd", "The lambd value for the Shrink formulation. Default is 0.5.", AttributeProto::FLOAT, 0.5f)
        .Attr("bias", "The bias value added to output. Default is 0.", AttributeProto::FLOAT, 0.0f)
        .Input(0, "input", "The input data as Tensor.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Output(0, "output", "The output.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .TypeConstraint("T", OpSchema::all_numeric_types(), "Constrain input to only numeric types.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput)
        .FunctionBody(
            R"ONNX(
          {
            Lambd = Constant <value_float: float = @lambd>()
            LambdCast = CastLike (Lambd, input)
            Bias = Constant <value_float: float = @bias>()
            BiasCast = CastLike (Bias, input)
            Zero = Constant <value = float {0.0}>()
            ZeroCast = CastLike (Zero, input)
            NegLmbda = Neg (LambdCast)
            InputLessThanNegLambda = Less (input, NegLmbda)
            InputAddBias = Add (input, BiasCast)
            InputSubBias = Sub (input, BiasCast)
            LambdaLessThanInput = Less (LambdCast, input)
            InputSubBiasOrZero = Where (LambdaLessThanInput, InputSubBias, ZeroCast)
            output = Where(InputLessThanNegLambda, InputAddBias, InputSubBiasOrZero)
		      }
        )ONNX",
            18));

static const char* Flatten_ver13_doc = R"DOC(
Flattens the input tensor into a 2D matrix. If input tensor has shape
(d_0, d_1, ... d_n) then the output will have shape
(d_0 X d_1 ... d_(axis-1), d_axis X d_(axis+1) ... X dn).
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Flatten,
    13,
    OpSchema()
        .SetDoc(Flatten_ver13_doc)
        .Input(0, "input", "A tensor of rank >= axis.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Output(
            0,
            "output",
            "A 2D tensor with the contents of the input tensor, "
            "with input dimensions up to axis flattened to the outer dimension "
            "of the output and remaining input dimensions flattened into the inner "
            "dimension of the output.",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .TypeConstraint("T", OpSchema::all_tensor_types_ir4(), "Constrain input and output to all tensor types.")
        .Attr(
            "axis",
            "Indicate up to which input dimensions "
            "(exclusive) should be flattened to the outer dimension of the output. "
            "The value for axis must be in the range [-r, r], where r is the rank of the input tensor. "
            "Negative value means counting dimensions from the back. "
            "When axis = 0, the shape of the output tensor is (1, (d_0 X d_1 ... d_n), "
            "where the shape of the input tensor is (d_0, d_1, ... d_n). ",
            AttributeProto::INT,
            static_cast<int64_t>(1))
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (!hasInputShape(ctx, 0))
            return;
          auto& input_shape = getInputShape(ctx, 0);
          int rank = static_cast<int>(input_shape.dim_size());
          int axis = static_cast<int>(getAttribute(ctx, "axis", 1));
          if (axis < 0) {
            axis += rank;
          }
          if (axis > rank || axis < 0) {
            fail_shape_inference("Invalid value(", axis, ") for attribute 'axis'");
          }
          // TODO: is the operation defined for input-rank < 2?
          updateOutputShape(ctx, 0, {multiplyDims(input_shape, 0, axis), multiplyDims(input_shape, axis, rank)});
        }));

static const char* LRN_ver13_doc = R"DOC(
Local Response Normalization proposed in the [AlexNet paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf).
It normalizes over local input regions.
The local region is defined across the channels. For an element `X[n, c, d1, ..., dk]` in a tensor
of shape `(N x C x D1 x D2, ..., Dk)`, its region is
`{X[n, i, d1, ..., dk] | max(0, c - floor((size - 1) / 2)) <= i <= min(C - 1, c + ceil((size - 1) / 2))}`.

`square_sum[n, c, d1, ..., dk] = sum(X[n, i, d1, ..., dk] ^ 2)`,
where `max(0, c - floor((size - 1) / 2)) <= i <= min(C - 1, c + ceil((size - 1) / 2))`.

`Y[n, c, d1, ..., dk] = X[n, c, d1, ..., dk] / (bias + alpha / size * square_sum[n, c, d1, ..., dk] ) ^ beta`
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    LRN,
    13,
    OpSchema()
        .Attr("size", "The number of channels to sum over", AttributeProto::INT)
        .Attr("alpha", "Scaling parameter.", AttributeProto::FLOAT, 0.0001f)
        .Attr("beta", "The exponent.", AttributeProto::FLOAT, 0.75f)
        .Attr("bias", "", AttributeProto::FLOAT, 1.0f)
        .Input(
            0,
            "X",
            "Input data tensor from the previous operator; "
            "dimensions for image case are (N x C x H x W), "
            "where N is the batch size, C is the number of "
            "channels, and H and W are the height and the "
            "width of the data. For non image case, the "
            "dimensions are in the form of "
            "(N x C x D1 x D2 ... Dn), where N is the batch "
            "size. Optionally, if dimension denotation is "
            "in effect, the operation expects the input "
            "data tensor to arrive with the dimension denotation "
            "of [DATA_BATCH, DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .Output(
            0,
            "Y",
            "Output tensor, which has the shape and type as input tensor",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
            "Constrain input and output "
            " types to float tensors.")
        .SetDoc(LRN_ver13_doc)
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* TfIdfVectorizer_ver9_doc = R"DOC(
This transform extracts n-grams from the input sequence and save them as a vector. Input can
be either a 1-D or 2-D tensor. For 1-D input, output is the n-gram representation of that input.
For 2-D input, the output is also a  2-D tensor whose i-th row is the n-gram representation of the i-th input row.
More specifically, if input shape is [C], the corresponding output shape would be [max(ngram_indexes) + 1].
If input shape is [N, C], this operator produces a [N, max(ngram_indexes) + 1]-tensor.

In contrast to standard n-gram extraction, here, the indexes of extracting an n-gram from the original
sequence are not necessarily consecutive numbers. The discontinuity between indexes are controlled by the number of skips.
If the number of skips is 2, we should skip two tokens when scanning through the original sequence.
Let's consider an example. Assume that input sequence is [94, 17, 36, 12, 28] and the number of skips is 2.
The associated 2-grams are [94, 12] and [17, 28] respectively indexed by [0, 3] and [1, 4].
If the number of skips becomes 0, the 2-grams generated are [94, 17], [17, 36], [36, 12], [12, 28]
indexed by [0, 1], [1, 2], [2, 3], [3, 4], respectively.

The output vector (denoted by Y) stores the count of each n-gram;
Y[ngram_indexes[i]] indicates the times that the i-th n-gram is found. The attribute ngram_indexes is used to determine the mapping
between index i and the corresponding n-gram's output coordinate. If pool_int64s is [94, 17, 17, 36], ngram_indexes is [1, 0],
ngram_counts=[0, 0], then the Y[0] (first element in Y) and Y[1] (second element in Y) are the counts of [17, 36] and [94, 17],
respectively. An n-gram which cannot be found in pool_strings/pool_int64s should be ignored and has no effect on the output.
Note that we may consider all skips up to S when generating the n-grams.

The examples used above are true if mode is "TF". If mode is "IDF", all the counts larger than 1 would be truncated to 1 and
the i-th element in weights would be used to scale (by multiplication) the count of the i-th n-gram in pool. If mode is "TFIDF",
this operator first computes the counts of all n-grams and then scale them by the associated values in the weights attribute.

Only one of pool_strings and pool_int64s can be set. If pool_int64s is set, the input should be an integer tensor.
If pool_strings is set, the input must be a string tensor.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    TfIdfVectorizer,
    9,
    OpSchema()
        .Input(0, "X", "Input for n-gram extraction", "T", OpSchema::Single, true, 1, OpSchema::NonDifferentiable)
        .Output(0, "Y", "Ngram results", "T1", OpSchema::Single, true, 1, OpSchema::NonDifferentiable)
        .TypeConstraint(
            "T",
            {"tensor(string)", "tensor(int32)", "tensor(int64)"},
            "Input is ether string UTF-8 or int32/int64")
        .TypeConstraint("T1", {"tensor(float)"}, "1-D tensor of floats")
        .Attr(
            "max_gram_length",
            "Maximum n-gram length. If this value is 3, 3-grams will be used to generate the output.",
            AttributeProto::INT)
        .Attr(
            "min_gram_length",
            "Minimum n-gram length. If this value is 2 and max_gram_length is 3, output may contain counts of 2-grams and 3-grams.",
            AttributeProto::INT)
        .Attr(
            "max_skip_count",
            "Maximum number of items (integers/strings) to be skipped when constructing an n-gram from X. "
            "If max_skip_count=1, min_gram_length=2, max_gram_length=3, this operator may generate 2-grams "
            "with skip_count=0 and skip_count=1, and 3-grams with skip_count=0 and skip_count=1",
            AttributeProto::INT)
        .Attr(
            "pool_strings",
            "List of strings n-grams learned from the training set. Either this or pool_int64s attributes must be present but not both. "
            "It's an 1-D tensor starting with the collections of all 1-grams and ending with the collections of n-grams. "
            "The i-th element in pool stores the n-gram that should be mapped to coordinate ngram_indexes[i] in the output vector.",
            AttributeProto::STRINGS,
            OPTIONAL_VALUE)
        .Attr(
            "pool_int64s",
            "List of int64 n-grams learned from the training set. Either this or pool_strings attributes must be present but not both. "
            "It's an 1-D tensor starting with the collections of all 1-grams and ending with the collections of n-grams. "
            "The i-th element in pool stores the n-gram that should be mapped to coordinate ngram_indexes[i] in the output vector.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr(
            "ngram_counts",
            "The starting indexes of 1-grams, 2-grams, and so on in pool. "
            "It is useful when determining the boundary between two consecutive collections of n-grams. "
            "For example, if ngram_counts is [0, 17, 36], the first index (zero-based) of 1-gram/2-gram/3-gram "
            "in pool are 0/17/36. This format is essentially identical to CSR (or CSC) sparse matrix format, "
            "and we choose to use this due to its popularity.",
            AttributeProto::INTS)
        .Attr(
            "ngram_indexes",
            "list of int64s (type: AttributeProto::INTS). This list is parallel to the specified 'pool_*' attribute. "
            "The i-th element in ngram_indexes indicate the coordinate of the i-th n-gram in the output tensor.",
            AttributeProto::INTS)
        .Attr(
            "weights",
            "list of floats. This attribute stores the weight of each n-gram in pool. The i-th element in weights "
            "is the weight of the i-th n-gram in pool. Its length equals to the size of ngram_indexes. "
            "By default, weights is an all-one tensor.This attribute is used when mode is \"IDF\" or \"TFIDF\" "
            "to scale the associated word counts.",
            AttributeProto::FLOATS,
            OPTIONAL_VALUE)
        .Attr(
            "mode",
            "The weighting criteria. It can be one of \"TF\" (term frequency), "
            "\"IDF\" (inverse document frequency), and \"TFIDF\" (the combination of TF and IDF)",
            AttributeProto::STRING)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          auto output_elem_type = ctx.getOutputType(0)->mutable_tensor_type();
          output_elem_type->set_elem_type(TensorProto::FLOAT);

          if (hasInputShape(ctx, 0)) {
            std::vector<int64_t> ngram_indexes;
            getRepeatedAttribute(ctx, "ngram_indexes", ngram_indexes);
            if (ngram_indexes.empty() ||
                !std::all_of(ngram_indexes.cbegin(), ngram_indexes.cend(), [](int64_t i) { return i >= 0; })) {
              fail_shape_inference("ngram_indexes must be non-empty with no negative values");
            }

            auto greatest_hit = std::max_element(ngram_indexes.cbegin(), ngram_indexes.cend());
            auto max_last_axis = *greatest_hit + 1;

            TensorShapeProto output_shape;
            auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
            auto dim_size = input_shape.dim_size();
            if (dim_size == 1) {
              output_shape.add_dim()->set_dim_value(max_last_axis);
            } else if (dim_size == 2) {
              *output_shape.add_dim() = input_shape.dim(0);
              output_shape.add_dim()->set_dim_value(max_last_axis);
            } else {
              fail_shape_inference("Input tensor must have rank 1 or 2");
            }
            updateOutputShape(ctx, 0, output_shape);
          }
        })
        .SetDoc(TfIdfVectorizer_ver9_doc));

static const char* StringNormalizer_ver10_doc = R"DOC(
StringNormalization performs string operations for basic cleaning.
This operator has only one input (denoted by X) and only one output
(denoted by Y). This operator first examines the elements in the X,
and removes elements specified in "stopwords" attribute.
After removing stop words, the intermediate result can be further lowercased,
uppercased, or just returned depending the "case_change_action" attribute.
This operator only accepts [C]- and [1, C]-tensor.
If all elements in X are dropped, the output will be the empty value of string tensor with shape [1]
if input shape is [C] and shape [1, 1] if input shape is [1, C].
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    StringNormalizer,
    10,
    OpSchema()
        .Input(0, "X", "UTF-8 strings to normalize", "tensor(string)")
        .Output(0, "Y", "UTF-8 Normalized strings", "tensor(string)")
        .Attr(
            std::string("case_change_action"),
            std::string("string enum that cases output to be lowercased/uppercases/unchanged."
                        " Valid values are \"LOWER\", \"UPPER\", \"NONE\". Default is \"NONE\""),
            AttributeProto::STRING,
            std::string("NONE"))
        .Attr(
            std::string("is_case_sensitive"),
            std::string("Boolean. Whether the identification of stop words in X is case-sensitive. Default is false"),
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Attr(
            "stopwords",
            "List of stop words. If not set, no word would be removed from X.",
            AttributeProto::STRINGS,
            OPTIONAL_VALUE)
        .Attr(
            "locale",
            "Environment dependent string that denotes the locale according to which output strings needs to be upper/lowercased."
            "Default en_US or platform specific equivalent as decided by the implementation.",
            AttributeProto::STRING,
            OPTIONAL_VALUE)
        .SetDoc(StringNormalizer_ver10_doc)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          auto output_elem_type = ctx.getOutputType(0)->mutable_tensor_type();
          output_elem_type->set_elem_type(TensorProto::STRING);
          if (!hasInputShape(ctx, 0)) {
            return;
          }
          TensorShapeProto output_shape;
          auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
          auto dim_size = input_shape.dim_size();
          // Last axis dimension is unknown if we have stop-words since we do
          // not know how many stop-words are dropped
          if (dim_size == 1) {
            // Unknown output dimension
            output_shape.add_dim();
          } else if (dim_size == 2) {
            // Copy B-dim
            auto& b_dim = input_shape.dim(0);
            if (!b_dim.has_dim_value() || b_dim.dim_value() != 1) {
              fail_shape_inference("Input shape must have either [C] or [1,C] dimensions where C > 0");
            }
            *output_shape.add_dim() = b_dim;
            output_shape.add_dim();
          } else {
            fail_shape_inference("Input shape must have either [C] or [1,C] dimensions where C > 0");
          }
          updateOutputShape(ctx, 0, output_shape);
        }));

static const char* mvn_ver13_doc = R"DOC(
      A MeanVarianceNormalization Function: Perform mean variance normalization
      on the input tensor X using formula: `(X-EX)/sqrt(E(X-EX)^2)`
)DOC";

static const std::vector<int64_t> mvn_default_axes = {0, 2, 3};

ONNX_OPERATOR_SET_SCHEMA(
    MeanVarianceNormalization,
    13,
    OpSchema()
        .SetDoc(mvn_ver13_doc)
        .Input(0, "X", "Input tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Output(0, "Y", "Output tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Attr(
            "axes",
            "A list of integers, along which to reduce. The default is to "
            "caculate along axes [0,2,3] for calculating mean and variance "
            "along each channel. Two variables with the same C-coordinate "
            "are associated with the same mean and variance.",
            AttributeProto::INTS,
            mvn_default_axes)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
            "Constrain input and output types to all numeric tensors.")
        .FunctionBody(R"ONNX(
        {
          Exponent = Constant <value = float {2.0}>()
          Epsilon = Constant <value = float {1e-9}>()
          X_RM = ReduceMean <axes : ints = @axes> (X)
          EX_squared = Pow (X_RM, Exponent)
          X_squared = Pow (X, Exponent)
          E_Xsquared = ReduceMean <axes : ints = @axes> (X_squared)
          Variance = Sub (E_Xsquared, EX_squared)
          STD = Sqrt (Variance)
          X_variance = Sub (X, X_RM)
          Processed_STD = Add (STD, Epsilon)
          Y = Div (X_variance, Processed_STD)
        }
        )ONNX")
        .FunctionBody(
            R"ONNX(
        {
          Exponent = Constant <value = float {2.0}>()
          Epsilon = Constant <value = float {1e-9}>()
          axes = Constant <value_ints: ints = @axes>()
          X_RM = ReduceMean (X, axes)
          EX_squared = Pow (X_RM, Exponent)
          X_squared = Pow (X, Exponent)
          E_Xsquared = ReduceMean (X_squared, axes)
          Variance = Sub (E_Xsquared, EX_squared)
          STD = Sqrt (Variance)
          X_variance = Sub (X, X_RM)
          Processed_STD = Add (STD, Epsilon)
          Y = Div (X_variance, Processed_STD)
        }
        )ONNX",
            18));

void col2imShapeInference(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);

  // All inputs shapes are required
  if (!hasNInputShapes(ctx, 3)) {
    return;
  }

  // We assume image_shape has correct spatial dimensions for next validations
  // An alternative is get the the number of spatial dimensions as an input argument
  Dim n_input_dims;
  unifyInputDim(ctx, 1, 0, n_input_dims);

  unifyInputDim(ctx, 2, 0, n_input_dims);
  checkInputRank(ctx, 1, 1);
  checkInputRank(ctx, 2, 1);
  std::vector<int64_t> image_shape = {};
  const TensorProto* image_shape_data = ctx.getInputData(1);
  if (image_shape_data) {
    image_shape = ParseData<int64_t>(image_shape_data);
    unifyDim(n_input_dims, image_shape.size());
  }

  std::vector<int64_t> pads = {};
  if (getRepeatedAttribute(ctx, "pads", pads)) {
    if (pads.size() % 2) {
      fail_shape_inference("Attribute pads must have an even size");
    }
    unifyDim(n_input_dims, pads.size() / 2);
  }

  std::vector<int64_t> dilations = {};
  if (getRepeatedAttribute(ctx, "dilations", dilations)) {
    unifyDim(n_input_dims, dilations.size());
  }

  std::vector<int64_t> strides = {};
  if (getRepeatedAttribute(ctx, "strides", strides)) {
    unifyDim(n_input_dims, strides.size());
  }

  auto input_shape = ctx.getInputType(0)->tensor_type().shape();
  if (input_shape.dim_size() != 3) {
    fail_shape_inference("input must have rank 3.");
  }

  std::vector<int64_t> block_shape = {};
  const TensorProto* block_shape_data = ctx.getInputData(2);
  if (block_shape_data) {
    block_shape = ParseData<int64_t>(block_shape_data);
    unifyDim(n_input_dims, block_shape.size());
  }
  unifyInputDim(ctx, 2, 0, n_input_dims);

  int block_shape_size = 0;
  if (static_cast<int>(block_shape.size()) > 0) {
    block_shape_size = 1;
    for (const auto& dim : block_shape) {
      block_shape_size *= dim;
    }
  }
  // If we haven't inferred the number of image dimensions, we can't set inferred shape.
  if (!n_input_dims.has_dim_value()) {
    return;
  }

  // Final shape will be (N, C, dim_1, ..., dim_N)
  auto final_image_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

  // Dimensions N and C are always present
  Dim N, C;
  if (ctx.getInputType(0)->tensor_type().shape().dim(0).has_dim_value()) {
    N = input_shape.dim(0); // Otherwise, N is unknown.
  }
  *final_image_shape->add_dim() = N;

  if (block_shape_size > 0) {
    C = input_shape.dim(1) / block_shape_size; // Otherwise, C is unknown.
  }
  *final_image_shape->add_dim() = C;

  // Image dimensions are dynamic
  for (auto i = 0; i < n_input_dims.dim_value(); ++i) {
    Dim image_dim_i;
    if (image_shape.size() > 0) {
      image_dim_i.set_dim_value(image_shape[i]); // Otherwise, spatial dimensions are unknown
    }
    *final_image_shape->add_dim() = image_dim_i;
  }
  return;
}

static const char* Col2Im_ver18_doc = R"DOC(
The operator rearranges column blocks back into a multidimensional image

Col2Im behaves similarly to PyTorch's fold https://pytorch.org/docs/stable/generated/torch.nn.Fold.html,
but it only supports *batched* multi-dimensional image tensors.
Another implementation in Python with N-dimension support can be found at https://github.com/f-dangel/unfoldNd/.

NOTE:
  Although specifying image_shape looks redundant because it could be calculated from
  convolution formulas, it is required as input for more advanced scenarios as explained
  at PyTorch's implementation (https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/Col2Im.cpp#L10)
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Col2Im,
    18,
    OpSchema()
        .Attr(
            "dilations",
            "1-dimensional tensor with dilation value along each spatial axis of the image. "
            "If not present, the dilation defaults to 1 along each spatial axis of the image.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr(
            "pads",
            "1-dimensional tensor with padding value for the beginning and ending along each spatial axis, "
            "it can take any value greater than or equal to 0. "
            "The value represent the number of pixels added to the beginning "
            "and end part of the corresponding axis. `pads` format should be as follow "
            "[x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin is the number of pixels "
            "added at the beginning of axis `i` and xi_end is the number of pixels added at the end of axis `i`. "
            "If not present, the padding defaults to 0 along start and end of each spatial axis.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr(
            "strides",
            "1-dimensional tensor with stride value along each spatial axis. "
            "If not present, the stride defaults to 1 along each spatial axis.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .SetDoc(Col2Im_ver18_doc)
        .Input(
            0,
            "input",
            "Input data tensor to be rearranged from column blocks back into an image."
            " This is a 3-dimensional tensor containing [N, C * n-ary-product(block_shape), L],"
            " where N is batch dimension, C is image channel dimension and L is number of blocks."
            "The blocks are enumerated in increasing lexicographic-order of their indices."
            "For example, with an image-size 10*20 and block-size 9*18, there would be 2*3 blocks,"
            " enumerated in the order block(0, 0), block(0, 1), block(0, 2), block(1, 0), block(1, 1), block(1, 2).",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .Input(
            1,
            "image_shape",
            "The shape of the spatial dimensions of the image after rearranging the column blocks."
            "This is a 1-dimensional tensor with size of at least 2, containing the value [H_img, W_img] "
            " for a 2-D image or [dim_i1, dim_i2, ..., dim_iN] for a N-D image.",
            "tensor(int64)",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Input(
            2,
            "block_shape",
            "The shape of the block to apply on the input."
            "This is a 1-dimensional tensor of size of at least 2, containing the value [H_block, W_block] "
            " for a 2-D image or [dim_b1, dim_b2, ..., dim_bN] for a N-D block."
            "This is the block-shape before dilation is applied to it.",
            "tensor(int64)",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Output(
            0,
            "output",
            "Output tensor produced by rearranging blocks into an image.",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .TypeConstraint(
            "T",
            OpSchema::all_tensor_types_ir4(),
            "Constrain input and output types to all numeric tensor types.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) { col2imShapeInference(ctx); }));

static const char* LayerNormalization_ver17_doc = R"DOC(
      This is layer normalization defined in ONNX as function.
      The overall computation can be split into two stages.
      The first stage is standardization, which makes the
      normalized elements have zero mean and unit variances.
      The computation required by standardization can be
      described by the following equations.
      ```
      Mean = ReduceMean<axes=normalized_axes>(X)
      D = Sub(X, Mean)
      DD = Mul(D, D)
      Var = ReduceMean<axes=normalized_axes>(DD)
      VarEps = Add(Var, epsilon)
      StdDev = Sqrt(VarEps)
      InvStdDev = Reciprocal(StdDev)
      Normalized = Mul(D, InvStdDev)
      ```
      where `normalized_axes` is `[axis, ..., rank of X - 1]`.
      The variables `Var` and `StdDev` stand for variance and
      standard deviation, respectively. The second output is
      `Mean` and the last one is `InvStdDev`.
      Depending on `stash_type` attribute, the actual computation
      must happen in different floating-point precision.
      For example, if `stash_type` is 1, this operator casts
      all input variables to 32-bit float, perform the computation, and
      finally cast `Normalized` back to the original type of `X`.
      The second stage then scales and shifts the outcome of the
      first stage using
      ```
      NormalizedScaled = Mul(Normalized, Scale)
      Y = Add(NormalizedScaled, B)
      ```
      The second stage doesn't depends on `stash_type`.
      All equations are in [this syntax](https://github.com/onnx/onnx/blob/main/docs/Syntax.md).
      The same variable (i.e., input, output, and attribute) uses
      the same name in the equations above and this operator's definition.
      Let `d[i]` indicate the i-th dimension of `X`.
      If `X`'s shape is `[d[0], ..., d[axis-1], d[axis], ..., d[rank-1]]`,
      the shape of `Mean` and `InvStdDev` is `[d[0], ..., d[axis-1], 1, ..., 1]`.
      `Y` and `X` have the same shape.
)DOC";

bool BuildContextDependentFunctionBodyLayerNormalization(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto,
    int sinceVersion) {
  ONNX_ASSERT(sinceVersion == 17 || sinceVersion == 18);
  // LayerNormalization <axis, epsilon, stash_type> (X, Scale, B) => (Y, Mean?, InvStdDev?)
  auto* tp = ctx.getInputType(0);
  if ((tp == nullptr) || (!tp->has_tensor_type()))
    return false;
  int64_t T = tp->tensor_type().elem_type();

  auto type_attr = ctx.getAttribute("stash_type");
  int64_t U =
      (type_attr != nullptr) ? type_attr->i() : static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  if ((U != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) && (U != ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16))
    return false; // Error

  auto* axis_attr = ctx.getAttribute("axis");
  int64_t axis = (axis_attr != nullptr) ? axis_attr->i() : -1;
  auto* epsilon_attr = ctx.getAttribute("epsilon");
  float epsilon = (epsilon_attr != nullptr) ? epsilon_attr->f() : 1e-5f;

  auto mktensor = [](int64_t val) -> ONNX_NAMESPACE::TensorProto {
    auto tp = ONNX_NAMESPACE::ToTensor(std::vector<int64_t>{val});
    tp.add_dims(1);
    return tp;
  };
  // The treatment of "axis" is different in "LayerNormalization" and in Reduction operations.
  // This complicates the function definition, requiring reshaping inputs/outputs.
  // Input X shape: [d[0], ..., d[axis-1], d[axis], ..., d[rank-1]]
  // This is treated as a 2D shape [d[0] * ... * d[axis-1], d[axis] * ... * d[rank-1]]
  // Normalization is applied to the second dimension.
  // Output Y has same shape as X
  // Outputs Mean and InvStdDev have shape: [d[0], ..., d[axis-1], 1, ..., 1]
  FunctionBuilder builder(functionProto);
  builder.Const("FloatEpsilon", ToTensor<float>(epsilon))
      .Add("Epsilon = Cast (FloatEpsilon)", "to", U)
      .Add("XShape = Shape (X)") // shape of input tensor: 1D tensor
      .Add("Rank = Size (XShape)") // rank of input tensor: scalar
      .Add("Zero1D = Constant()", "value", mktensor(0)) // [0] : 1D tensor
      .Add("Axis1D = Constant()", "value", mktensor(axis)) // [axis] : 1D tensor
      .Add("PrefixShape = Slice (XShape, Zero1D, Axis1D)") // [d[0], ..., d[axis-1]]
      .Add(
          axis >= 0 // number of axes that are reduced =
              ? "NumReducedAxes = Sub (Rank, Axis1D)" // [rank - axis]: 1D tensor
              : "NumReducedAxes = Neg (Axis1D)") // [-axis] : 1D tensor
      .Add(
          "SuffixShape = ConstantOfShape (NumReducedAxes)",
          "value",
          mktensor(1)) // [1, ..., 1] for reduced axes
      .Add("ReducedShape = Concat <axis = 0> (PrefixShape, SuffixShape)") // [d[0], ..., d[axis-1], 1, ..., 1]
      .Add("X2D = Flatten (X)", "axis", axis)
      .Add("XU = Cast (X2D)", "to", U);
  if (sinceVersion == 17) {
    builder.Add("Mean2D = ReduceMean <axes = [1]> (XU)")
        .Add("Square = Mul (XU, XU)")
        .Add("MeanOfSquare = ReduceMean <axes = [1]> (Square)");
  } else if (sinceVersion == 18) {
    builder.Add("Axes_1 = Constant()", "value", mktensor(1))
        .Add("Mean2D = ReduceMean (XU, Axes_1)")
        .Add("Square = Mul (XU, XU)")
        .Add("MeanOfSquare = ReduceMean (Square, Axes_1)");
  }
  builder.Add("SquareOfMean = Mul (Mean2D, Mean2D)")
      .Add("Var = Sub (MeanOfSquare, SquareOfMean)")
      .Add("VarPlusEpsilon = Add (Var, Epsilon)")
      .Add("StdDev = Sqrt (VarPlusEpsilon)")
      .Add("Deviation = Sub (XU, Mean2D)")
      .Add("Normalized = Div (Deviation, StdDev)")
      .Add("NormalizedT = Cast (Normalized)", "to", T)
      .Add("Scale2D = Flatten <axis = 0> (Scale)")
      .Add("Scaled = Mul (NormalizedT, Scale2D)");
  if (ctx.hasInput(2)) {
    builder.Add("B2D = Flatten <axis=0> (B)");
    builder.Add("Biased = Add (Scaled, B2D)");
  } else {
    builder.Add("Biased = Identity (Scaled)");
  }
  builder.Add("Y = Reshape (Biased, XShape)");
  builder.Add("InvStdDev2D = Reciprocal (StdDev)");
  if (ctx.hasOutput(1))
    builder.Add("Mean = Reshape (Mean2D, ReducedShape)");
  if (ctx.hasOutput(2))
    builder.Add("InvStdDev = Reshape (InvStdDev2D, ReducedShape)");

  schema.BuildFunction(functionProto);
  return true;
}

bool BuildContextDependentFunctionBodyLayerNormalizationVer17(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  return BuildContextDependentFunctionBodyLayerNormalization(ctx, schema, functionProto, 17);
}

bool BuildContextDependentFunctionBodyLayerNormalizationVer18(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  return BuildContextDependentFunctionBodyLayerNormalization(ctx, schema, functionProto, 18);
}

ONNX_OPERATOR_SET_SCHEMA(
    LayerNormalization,
    17,
    OpSchema()
        .SetDoc(LayerNormalization_ver17_doc)
        .Attr(
            "axis",
            "The first normalization dimension. If rank(X) is r, axis' allowed range is [-r, r]. "
            "Negative value means counting dimensions from the back.",
            AttributeProto::INT,
            static_cast<int64_t>(-1))
        .Attr("epsilon", "The epsilon value to use to avoid division by zero.", AttributeProto::FLOAT, 1e-5f)
        .Attr(
            "stash_type",
            "Type of Mean and InvStdDev. This also specifies stage one's computation precision.",
            AttributeProto::INT,
            static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT))
        .AllowUncheckedAttributes()
        .Input(0, "X", "Tensor to be normalized.", "T")
        .Input(1, "Scale", "Scale tensor.", "T")
        .Input(2, "B", "Bias tensor.", "T", OpSchema::Optional)
        .Output(0, "Y", "Normalized tensor.", "T")
        .Output(1, "Mean", "Saved mean used during training to speed up gradient computation", "U", OpSchema::Optional)
        .Output(
            2,
            "InvStdDev",
            "Saved inverse standard deviation used during training to speed up gradient computation.",
            "U",
            OpSchema::Optional)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
            "Constrain input types and output Y type to float tensors.")
        .TypeConstraint("U", {"tensor(float)", "tensor(bfloat16)"}, "Type of Mean and InvStdDev tensors.")
        .SetContextDependentFunctionBodyBuilder(BuildContextDependentFunctionBodyLayerNormalizationVer17, 17)
        .SetContextDependentFunctionBodyBuilder(BuildContextDependentFunctionBodyLayerNormalizationVer18, 18)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateShapeAndTypeFromFirstInput(ctx);
          auto stash_type = static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
          auto stash_type_proto = ctx.getAttribute("stash_type");
          if (stash_type_proto) {
            stash_type = stash_type_proto->i();
          }
          if (ctx.getNumOutputs() > 1) {
            auto output_type = ctx.getOutputType(1);
            output_type->mutable_tensor_type()->set_elem_type(static_cast<int32_t>(stash_type));
          }
          if (ctx.getNumOutputs() > 2) {
            auto output_type = ctx.getOutputType(2);
            output_type->mutable_tensor_type()->set_elem_type(static_cast<int32_t>(stash_type));
          }
          if (!hasNInputShapes(ctx, 1)) {
            return;
          }
          auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
          int64_t input_ndim = input_shape.dim_size();
          int64_t axis = -1;
          auto axis_proto = ctx.getAttribute("axis");
          if (axis_proto) {
            axis = axis_proto->i();
          }
          if (axis < 0) {
            // Convert negative axis value to equivalent
            // positive value.
            axis += input_ndim;
          }

          if (ctx.getNumOutputs() > 1) {
            auto mean_shape = ctx.getOutputType(1)->mutable_tensor_type()->mutable_shape();
            mean_shape->CopyFrom(input_shape);
            for (int d = static_cast<int>(axis); d < input_ndim; ++d)
              mean_shape->mutable_dim(d)->set_dim_value(1);
          }

          if (ctx.getNumOutputs() > 2) {
            auto inv_std_dev_shape = ctx.getOutputType(2)->mutable_tensor_type()->mutable_shape();
            inv_std_dev_shape->CopyFrom(input_shape);
            for (int d = static_cast<int>(axis); d < input_ndim; ++d)
              inv_std_dev_shape->mutable_dim(d)->set_dim_value(1);
          }
        }));

static const char* GroupNormalization_ver18_doc = R"DOC(
A GroupNormalization function. Carries out group normalization as described in
the paper https://arxiv.org/abs/1803.08494

This operator transforms input according to
```
y = scale * (x - mean) / sqrt(variance + epsilon) + bias,
```
where the mean and variance are computed per instance per group of channels, and
`scale` and `bias` should be specified for each group of channels. The number of
groups `num_groups` should be divisible by the number of channels so that there are
an equal number of channels per group.

When the number of groups is the same as the number of channels, this operator is
equivalent to InstanceNormalization. When there is only one group, this operator
is equivalent to LayerNormalization.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    GroupNormalization,
    18,
    OpSchema()
        .SetDoc(GroupNormalization_ver18_doc)
        .Attr("epsilon", "The epsilon value to use to avoid division by zero.", AttributeProto::FLOAT, 1e-5f)
        .Attr(
            "num_groups",
            "The number of groups of channels. It should be a divisor of the number of channels `C`.",
            AttributeProto::INT,
            true)
        .Input(
            0,
            "X",
            "Input data tensor. Dimensions for image cases are `(N x C x H x W)`, where `N` is the batch size, "
            "`C` is the number of channels, and `H` and `W` are the height and width of the data. Statistics are "
            "computed for every group of channels over `C`, `H`, and `W`. For non-image cases, the dimensions are "
            "in the form of `(N x C x D1 x D2 ... Dn)`.",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .Input(
            1,
            "scale",
            "Scale tensor of shape `(num_groups)`.",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .Input(
            2,
            "bias",
            "Bias tensor of shape `(num_groups)`.",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .Output(
            0,
            "Y",
            "The output tensor of the same shape as `X`.",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
            "Constrain input and output types to float tensors.")
        .SetContextDependentFunctionBodyBuilder(
            [](const FunctionBodyBuildContext& ctx, const OpSchema& schema, FunctionProto& functionProto) {
              // GroupNormalization <epsilon, num_groups> (X, scale, bias) => (Y)
              auto* tp = ctx.getInputType(0);
              if ((tp == nullptr) || (!tp->has_tensor_type()))
                return false;
              int64_t T = tp->tensor_type().elem_type();

              auto* epsilon_attr = ctx.getAttribute("epsilon");
              float epsilon = (epsilon_attr != nullptr) ? epsilon_attr->f() : 1e-5f;
              auto* num_groups_attr = ctx.getAttribute("num_groups");
              if (num_groups_attr == nullptr)
                return false;
              int64_t num_groups = num_groups_attr->i();

              FunctionBuilder builder(functionProto);
              builder.Const1D("FloatEpsilon", epsilon)
                  .Add("Epsilon = Cast (FloatEpsilon)", "to", T)
                  .Add("XShape = Shape (X)") // shape of input tensor: 1D tensor
                  .Add("C = Shape <start = 1, end = 2> (X)")
                  .Const1D("NumGroups", num_groups)
                  .Add("GroupSize = Div (C, NumGroups)")
                  .Add("N = Shape <start = 0, end = 1> (X)") // batch size
                  .Add("InstanceShape = Shape <start = 2> (X)") // data instance shape

                  // NewShape = [N, num_groups, group_size, H, W, (...)]
                  .Add("NewShape = Concat <axis = 0> (N, NumGroups, GroupSize, InstanceShape)")
                  .Add("XReshaped = Reshape (X, NewShape)")

                  // Flatten into 3D tensor: [N, num_groups, group_size x H x W (x ...)]
                  .Add("Shape3D = Constant <value_ints = [0, 0, -1]> ()")
                  .Add("X3D = Reshape(XReshaped, Shape3D)")

                  // Calculate statistics
                  .Const1D("Axes2", (int64_t)2)
                  .Add("Mean = ReduceMean (X3D, Axes2)")
                  .Add("Square = Mul (X3D, X3D)")
                  .Add("MeanOfSquare = ReduceMean (Square, Axes2)")
                  .Add("SquareOfMean = Mul (Mean, Mean)")
                  .Add("Var = Sub (MeanOfSquare, SquareOfMean)")
                  .Add("VarPlusEpsilon = Add (Var, Epsilon)")
                  .Add("StdDev = Sqrt (VarPlusEpsilon)")
                  .Add("Deviation = Sub (X3D, Mean)")
                  .Add("Normalized = Div (Deviation, StdDev)")

                  // Reshape scale and bias for broadcasting
                  .Add("ScaleShape = Constant <value_ints = [1, -1, 1]> ()")
                  .Add("ScaleT = Cast (scale)", "to", T)
                  .Add("BiasT = Cast (bias)", "to", T)
                  .Add("ScaleReshaped = Reshape (ScaleT, ScaleShape)")
                  .Add("BiasReshaped = Reshape (BiasT, ScaleShape)")

                  // Calculate scaled and biased output
                  .Add("Scaled = Mul (ScaleReshaped, Normalized)")
                  .Add("Biased = Add (Scaled, BiasReshaped)")
                  .Add("Y = Reshape (Biased, XShape)");

              schema.BuildFunction(functionProto);
              return true;
            }));
} // namespace ONNX_NAMESPACE
