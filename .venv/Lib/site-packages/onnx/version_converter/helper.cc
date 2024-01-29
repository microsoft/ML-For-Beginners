// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Helper Methods for Adapters

#include "onnx/version_converter/helper.h"

#include <vector>

namespace ONNX_NAMESPACE {
namespace version_conversion {
int check_numpy_unibroadcastable_and_require_broadcast(
    const std::vector<Dimension>& input1_sizes,
    const std::vector<Dimension>& input2_sizes) {
  // Check that input1 is larger
  if (input1_sizes.size() < input2_sizes.size())
    return -1;
  // Check that axis is input1_sizes.size()-input2_sizes.size()
  bool broadcast = false;
  int axis = (int)(input1_sizes.size() - input2_sizes.size());
  for (int i = 0; i < (int)input2_sizes.size(); i++) {
    if (input2_sizes[i].dim != input1_sizes[axis + i].dim && input2_sizes[i].dim != 1)
      return -1;
    if (input2_sizes[i].dim != input1_sizes[axis + i].dim)
      broadcast = true;
  }
  // Return true if broadcasting is required
  if (input1_sizes.size() > input2_sizes.size() || broadcast)
    return 1;
  else
    return 0;
}

void assert_numpy_multibroadcastable(
    const std::vector<Dimension>& input1_sizes,
    const std::vector<Dimension>& input2_sizes) {
  // Generalize above for multibroadcastable case
  const std::vector<Dimension>* A_ptr;
  const std::vector<Dimension>* B_ptr;
  int A;
  int B;
  if (input1_sizes.size() < input2_sizes.size()) {
    A_ptr = &input2_sizes;
    B_ptr = &input1_sizes;
    A = 2;
    B = 1;
  } else {
    A_ptr = &input1_sizes;
    B_ptr = &input2_sizes;
    A = 1;
    B = 2;
  }
  const std::vector<Dimension>& A_sizes = *A_ptr;
  const std::vector<Dimension>& B_sizes = *B_ptr;
  int axis = (int)(A_sizes.size() - B_sizes.size());
  for (int i = 0; i < (int)B_sizes.size(); i++) {
    ONNX_ASSERTM(
        B_sizes[i].dim == A_sizes[axis + i].dim || B_sizes[i].dim == 1 || A_sizes[axis + i].dim == 1,
        "Dimension %d of input %d does not match "
        "dimension %d of input %d, and neither's value is 1",
        i,
        B,
        axis + i,
        A);
  }
}

void assertNotParams(const std::vector<Dimension>& sizes) {
  for (const Dimension& dim : sizes) {
    ONNX_ASSERTM(dim.is_int, "%s Dimension is a param instead of an int.", dim.param.c_str());
  }
}

void assertInputsAvailable(const ArrayRef<Value*>& inputs, const char* name, uint64_t num_inputs) {
  ONNX_ASSERTM(
      inputs.size() == num_inputs,
      "%s in opset version 6 can only broadcast"
      " between %d inputs",
      name,
      num_inputs);
  for (int i = 0; i < (int)num_inputs; i++) {
    ONNX_ASSERTM(inputs[i]->has_sizes(), "Shape of input %d is not available.", num_inputs);
    assertNotParams(inputs[i]->sizes());
  }
}
} // namespace version_conversion
} // namespace ONNX_NAMESPACE
