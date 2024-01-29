// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>

#include "onnx/common/status.h"
#include "onnx/onnx-operators_pb.h"

namespace ONNX_NAMESPACE {

// Helper function for register nodes in
// a FunctionProto. Attributes need to be
// registered separately.
Common::Status BuildNode(
    const std::string& name,
    const std::string& domain,
    const std::string& doc_string,
    const std::string& op_type,
    std::vector<std::string> const& inputs,
    std::vector<std::string> const& outputs,
    /*OUT*/ NodeProto* node);
} // namespace ONNX_NAMESPACE
