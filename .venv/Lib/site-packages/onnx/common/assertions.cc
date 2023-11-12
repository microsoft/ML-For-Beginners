// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#include "onnx/common/assertions.h"
#include <cstdarg>
#include <cstdio>
#include "onnx/common/common.h"

namespace ONNX_NAMESPACE {

std::string barf(const char* fmt, ...) {
  char msg[2048];
  va_list args;

  va_start(args, fmt);
  // Although vsnprintf might have vulnerability issue while using format string with overflowed length,
  // it should be safe here to use fixed length for buffer "msg". No further checking is needed.
  vsnprintf(msg, 2048, fmt, args);
  va_end(args);

  return std::string(msg);
}

void throw_assert_error(std::string& msg) {
  ONNX_THROW_EX(assert_error(msg));
}

void throw_tensor_error(std::string& msg) {
  ONNX_THROW_EX(tensor_error(msg));
}

} // namespace ONNX_NAMESPACE
