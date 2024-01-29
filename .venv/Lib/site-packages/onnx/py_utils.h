// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>

#include "onnx/proto_utils.h"

namespace ONNX_NAMESPACE {
namespace py = pybind11;

template <typename Proto>
bool ParseProtoFromPyBytes(Proto* proto, const py::bytes& bytes) {
  // Get the buffer from Python bytes object
  char* buffer = nullptr;
  Py_ssize_t length;
  PyBytes_AsStringAndSize(bytes.ptr(), &buffer, &length);

  return ParseProtoFromBytes(proto, buffer, length);
}
} // namespace ONNX_NAMESPACE
