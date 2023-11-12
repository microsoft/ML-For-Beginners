// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "onnx/checker.h"
#include "onnx/common/path.h"

#include <fstream>

namespace ONNX_NAMESPACE {

template <typename T>
void LoadProtoFromPath(const std::string proto_path, T& proto) {
  std::fstream proto_stream(proto_path, std::ios::in | std::ios::binary);
  if (!proto_stream.good()) {
    fail_check("Unable to open proto file: ", proto_path, ". Please check if it is a valid proto. ");
  }
  std::string data{std::istreambuf_iterator<char>{proto_stream}, std::istreambuf_iterator<char>{}};
  if (!ParseProtoFromBytes(&proto, data.c_str(), data.size())) {
    fail_check(
        "Unable to parse proto from file: ", proto_path, ". Please check if it is a valid protobuf file of proto. ");
  }
}
} // namespace ONNX_NAMESPACE
