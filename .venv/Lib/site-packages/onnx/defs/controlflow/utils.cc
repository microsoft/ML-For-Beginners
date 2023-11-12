/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "onnx/defs/math/utils.h"
#include "onnx/defs/controlflow/utils.h"

namespace ONNX_NAMESPACE {

void ClearShape(TypeProto& input_type) {
  if (input_type.has_tensor_type()) {
    input_type.mutable_tensor_type()->clear_shape();
  } else if (input_type.has_sequence_type()) {
    auto& seq_type = *input_type.mutable_sequence_type();
    if (seq_type.has_elem_type()) {
      ClearShape(*(seq_type.mutable_elem_type()));
    }
  } else if (input_type.has_optional_type()) {
    auto& opt_type = *input_type.mutable_optional_type();
    if (opt_type.has_elem_type()) {
      ClearShape(*(opt_type.mutable_elem_type()));
    }
  }
}

void IfInferenceFunction(InferenceContext& ctx) {
  // there are no inputs so we just need to run the subgraph inferencing for
  // then/else subgraphs and apply those to the outputs.
  std::vector<const TypeProto*> subgraph_input_types; // none
  std::vector<const TensorProto*> input_data; // none

  std::vector<const TypeProto*> then_output_types;
  std::vector<const TypeProto*> else_output_types;

  // Run inferencing on the subgraph
  GraphInferencer* graphInferencer = ctx.getGraphAttributeInferencer("then_branch");
  if (graphInferencer) {
    then_output_types = graphInferencer->doInferencing(subgraph_input_types, input_data);
  }

  graphInferencer = ctx.getGraphAttributeInferencer("else_branch");
  if (graphInferencer) {
    else_output_types = graphInferencer->doInferencing(subgraph_input_types, input_data);
  }

  auto num_outputs = ctx.getNumOutputs();
  auto num_then_outputs = then_output_types.size();
  auto num_else_outputs = else_output_types.size();

  // the output types for then and else should be the same
  if (num_then_outputs != num_else_outputs) {
    fail_type_inference(
        "then_branch and else_branch produce different number of outputs. ",
        num_then_outputs,
        " != ",
        num_else_outputs);
  }

  if (num_then_outputs != num_outputs) {
    fail_type_inference("If node has ", num_outputs, " but subgraphs produce ", num_then_outputs);
  }

  for (size_t i = 0, end = then_output_types.size(); i < end; ++i) {
    auto then_output = then_output_types[i];
    auto else_output = else_output_types[i];

    auto* if_output = ctx.getOutputType(i);
    *if_output = *then_output;

    UnionTypeInfo(*else_output, *if_output);
  }
}

void LoopInferenceFunction(InferenceContext& ctx) {
  auto num_inputs = ctx.getNumInputs();
  assert(num_inputs >= 2);
  auto num_loop_state_vars = num_inputs - 2; // skip 'M' and 'cond'

  std::vector<const TypeProto*> subgraph_input_types;
  subgraph_input_types.reserve(num_inputs);

  std::vector<TypeProto> temporary_type_protos;
  temporary_type_protos.reserve(num_inputs - 2);

  // create TypeProto to validate iteration number type is the same as the
  // optional 'M' input for max iterations.
  TypeProto iter_num_type;
  iter_num_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT64);
  subgraph_input_types.push_back(&iter_num_type);

  // 'cond'
  subgraph_input_types.push_back(ctx.getInputType(1));

  // loop state value types get propagated to outputs, but shape may change
  // across iterations so don't propagate it to the outputs and don't pass it
  // into the subgraph inferencing
  for (size_t i = 2; i < num_inputs; ++i) {
    propagateElemTypeFromInputToOutput(ctx, i, i - 2);

    // copy so we can remove the shape before passing to the subgraph
    // inferencing
    temporary_type_protos.push_back(*ctx.getInputType(i));
    auto& input_type = temporary_type_protos.back();

    ClearShape(input_type);
    subgraph_input_types.push_back(&input_type);
  }

  // Run inferencing on the subgraph
  std::vector<const TypeProto*> subgraph_output_types;

  GraphInferencer* graphInferencer = ctx.getGraphAttributeInferencer("body");
  if (graphInferencer) {
    std::vector<const TensorProto*> input_data;
    input_data.push_back(nullptr); // iteration number
    for (size_t i = 1; i < num_inputs; ++i) {
      input_data.push_back(ctx.getInputData(i));
    }

    subgraph_output_types = graphInferencer->doInferencing(subgraph_input_types, input_data);
  }

  // if empty(), assume inferencing was skipped
  if (!subgraph_output_types.empty()) {
    auto num_outputs = ctx.getNumOutputs();

    // subgraph outputs the condition value first but that is only used
    // internally and not returned by Loop.
    if (subgraph_output_types.size() != num_outputs + 1) {
      fail_type_inference(
          "Graph attribute inferencing returned type information for ",
          subgraph_output_types.size(),
          " outputs. Expected ",
          num_outputs + 1);
    }

    // check loop state values match. we should already have type/shape info
    for (size_t i = 0; i < num_outputs; ++i) {
      auto* subgraph_output_type = subgraph_output_types[i + 1]; // skip 'cond'
      auto* loop_output_type = ctx.getOutputType(i);

      const bool is_loop_state_var = i < num_loop_state_vars;

      if (!subgraph_output_type->has_tensor_type() && !subgraph_output_type->has_sequence_type() &&
          !subgraph_output_type->has_optional_type()) {
        fail_type_inference(
            "Loop 'body' subgraph outputs should all be tensors or sequences or optionals, but output ",
            i,
            " was ",
            subgraph_output_type->value_case());
      }

      if (!is_loop_state_var && !subgraph_output_type->has_tensor_type()) {
        fail_type_inference(
            "Loop 'body' subgraph scan outputs should all be tensors but output ",
            i,
            " was ",
            subgraph_output_type->value_case());
      }

      // if there's an existing type check it matches. otherwise propagate
      propagateElemTypeWithValidation(subgraph_output_type, loop_output_type);

      if (is_loop_state_var) {
        // shape may change across iterations so ignore.
      } else {
        // propogate shape
        if (subgraph_output_type->tensor_type().has_shape()) {
          // per iteration output. first dimension will be number of iterations
          // but we don't know that value yet
          TypeProto inferred_type(*subgraph_output_type);
          auto* mutable_inferred_tensor_type = inferred_type.mutable_tensor_type();
          auto* mutable_inferred_shape = mutable_inferred_tensor_type->mutable_shape();

          mutable_inferred_shape->clear_dim();

          // add empty dimension for number of iterations
          mutable_inferred_shape->add_dim();

          // add dimensions from subgraph output shape
          for (const auto& dim : subgraph_output_type->tensor_type().shape().dim()) {
            (*mutable_inferred_shape->add_dim()) = dim;
          }

          mergeInShapeInfo(*mutable_inferred_tensor_type, *loop_output_type->mutable_tensor_type());
        }
      }
    }
  }
}

int handle_negative_axis_validate(const std::string& attrib, int axis, int rank) {
  if (!(-rank <= axis && axis < rank)) {
    fail_shape_inference(attrib, " axis value ", axis, " is invalid for a tensor of rank ", rank);
  }
  return (axis >= 0 ? axis : axis + rank);
}

void ScanInferenceFunction(InferenceContext& ctx) {
  auto num_inputs = ctx.getNumInputs();
  auto num_scan_inputs = narrow_cast<size_t>(ctx.getAttribute("num_scan_inputs")->i());
  auto num_loop_state_vars = num_inputs - num_scan_inputs;
  auto num_outputs = ctx.getNumOutputs();
  auto num_scan_outputs = num_outputs - num_loop_state_vars;

  std::vector<int64_t> axes, output_axes;
  if (getRepeatedAttribute(ctx, "scan_input_axes", axes)) {
    if (axes.size() != num_scan_inputs) {
      fail_shape_inference(
          "Number of scan input axes specified (",
          axes.size(),
          ") is not equal to number of scan inputs (",
          num_scan_inputs,
          ").");
    }
  } else {
    axes.insert(axes.end(), num_scan_inputs, 0);
  }

  if (getRepeatedAttribute(ctx, "scan_output_axes", output_axes)) {
    if (output_axes.size() != num_scan_outputs) {
      fail_shape_inference(
          "Number of scan output axes specified (",
          output_axes.size(),
          ") is not equal to number of scan outputs (",
          num_scan_outputs,
          ").");
    }
  } else {
    output_axes.insert(output_axes.end(), num_scan_outputs, 0);
  }

  std::vector<TypeProto> temporary_type_protos;
  temporary_type_protos.reserve(num_inputs);

  std::vector<const TypeProto*> subgraph_input_types;
  subgraph_input_types.reserve(num_inputs);

  TensorShapeProto_Dimension sequence_len_dim;

  for (size_t i = 0; i < num_inputs; ++i) {
    bool is_loop_state_var = i < num_loop_state_vars;
    bool has_shape = hasInputShape(ctx, i);
    const auto* input_type = ctx.getInputType(i);

    // Enforce type constraint for inputs
    if (!input_type || !input_type->has_tensor_type()) {
      fail_type_inference("Scan input ", i, " was not a tensor.");
    }

    if (is_loop_state_var) {
      // If it's a loop state variable we can propagate type and shape 1:1 to
      // the matching Scan output.
      // We can also pass through the type and shape to the subgraph but need to
      // remove the batch size dimension from the shape.
      propagateElemTypeFromInputToOutput(ctx, i, i);
      if (has_shape)
        propagateShapeFromInputToOutput(ctx, i, i);

      subgraph_input_types.push_back(input_type);
    } else {
      // For other inputs there is no fixed relationships to the Scan outputs,
      // so we don't propagate type/shape information.
      // We can pass through the type and shape to the subgraph inputs but
      // need to remove the sequence length dimensions from the shape.
      if (has_shape) {
        const auto& shape = input_type->tensor_type().shape();

        // remove sequence length dimensions and add to subgraph_input_types
        int axis = static_cast<int>(axes[i - num_loop_state_vars]);
        axis = handle_negative_axis_validate("scan_input_axes", axis, shape.dim_size());

        // update sequence_len if a value is available

        const auto& dims = shape.dim();
        mergeInDimensionInfo(dims.Get(axis), sequence_len_dim, 1);

        temporary_type_protos.push_back(RemoveIthDimensionFromShape(*input_type, axis));
        subgraph_input_types.push_back(&temporary_type_protos.back());

      } else {
        subgraph_input_types.push_back(input_type);
      }
    }
  }

  // Run inferencing on the subgraph
  std::vector<const TypeProto*> output_types;

  GraphInferencer* graphInferencer = ctx.getGraphAttributeInferencer("body");
  if (graphInferencer) {
    std::vector<const TensorProto*> input_data;
    input_data.reserve(num_inputs);
    for (size_t i = 0; i < num_inputs; ++i) {
      // ctx.getInputData(i), the input to scan, does not represent the input to
      // scan body. So, we pass in null, to represent an unknown value.
      input_data.push_back(nullptr);
    }

    output_types = graphInferencer->doInferencing(subgraph_input_types, input_data);
  }

  // if empty(), assume inferencing was skipped
  if (!output_types.empty()) {
    if (output_types.size() != num_outputs) {
      fail_type_inference(
          "Graph attribute inferencing returned type information for ",
          output_types.size(),
          " outputs. Expected ",
          num_outputs);
    }

    // propagate type/shape information for loop state variables and outputs
    for (size_t i = 0; i < num_outputs; ++i) {
      const bool is_loop_state_var = i < num_loop_state_vars;
      auto* subgraph_output_type = output_types[i];
      auto* scan_output_type = ctx.getOutputType(i);
      auto* mutable_scan_output_tensor_type = scan_output_type->mutable_tensor_type();

      if (!subgraph_output_type->has_tensor_type()) {
        fail_type_inference("Scan 'body' subgraph outputs should all be tensors but output ", i, " was not");
      }
      auto& subgraph_output_tensor_type = subgraph_output_type->tensor_type();

      if (is_loop_state_var) {
        // merge shape; type already propagated
        mergeInShapeInfo(subgraph_output_tensor_type, *mutable_scan_output_tensor_type);
      } else {
        scan_output_type->mutable_tensor_type()->set_elem_type(subgraph_output_tensor_type.elem_type());

        // propagate shape
        if (subgraph_output_tensor_type.has_shape()) {
          // infer shape of scan-output from the shape of scan-output-element
          // by adding sequence-length at the correct axis position
          const TensorShapeProto& subgraph_output_shape = subgraph_output_tensor_type.shape();
          TensorShapeProto inferred_shape;

          auto subgraph_output_rank = subgraph_output_shape.dim_size();
          auto output_rank = subgraph_output_rank + 1;
          int output_axis = static_cast<int>(output_axes[i - num_loop_state_vars]);
          output_axis = handle_negative_axis_validate("scan_output_axes", output_axis, output_rank);

          for (int j = 0; j < output_axis; ++j)
            *(inferred_shape.add_dim()) = subgraph_output_shape.dim(j);
          *(inferred_shape.add_dim()) = sequence_len_dim;
          for (int j = output_axis; j < subgraph_output_rank; ++j)
            *(inferred_shape.add_dim()) = subgraph_output_shape.dim(j);

          // Merge inferred shape with existing shape information
          mergeInShapeInfo(inferred_shape, *mutable_scan_output_tensor_type);
        }
      }
    }
  }
}

} // namespace ONNX_NAMESPACE
