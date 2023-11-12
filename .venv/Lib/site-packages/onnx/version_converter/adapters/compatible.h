// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Adapter indicating compatibility of op between opsets with separate
// definitions

#pragma once

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE {
namespace version_conversion {

struct CompatibleAdapter final : public Adapter {
  explicit CompatibleAdapter(const std::string& op_name, const OpSetID& initial, const OpSetID& target)
      : Adapter(op_name, initial, target) {}

  Node* adapt(std::shared_ptr<Graph>, Node* node) const override {
    return node;
  }
};

} // namespace version_conversion
} // namespace ONNX_NAMESPACE
