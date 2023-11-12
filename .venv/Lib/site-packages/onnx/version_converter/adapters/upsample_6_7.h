// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Adapter for Upsample in default domain from version 6 to 7

#pragma once

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE {
namespace version_conversion {

struct Upsample_6_7 final : public Adapter {
  explicit Upsample_6_7() : Adapter("Upsample", OpSetID(6), OpSetID(7)) {}

  void adapt_upsample_6_7(std::shared_ptr<Graph>, Node* node) const {
    Symbol width_scale_symbol = Symbol("width_scale");
    Symbol height_scale_symbol = Symbol("height_scale");
    ONNX_ASSERTM(
        node->hasAttribute(width_scale_symbol) && node->hasAttribute(height_scale_symbol),
        "Upsample in opset 1 needs to have width_scale and height_scale attributes");

    auto width_scale = node->f(width_scale_symbol);
    auto height_scale = node->f(height_scale_symbol);

    auto input_shape = node->inputs()[0]->sizes();
    ONNX_ASSERTM(input_shape.size() == 4, "Upsample in opset 1 supports only 4D input tensor");
    std::vector<double> scales = {1.0, 1.0, height_scale, width_scale};

    node->fs_(kscales, std::move(scales));
    node->removeAttribute(width_scale_symbol);
    node->removeAttribute(height_scale_symbol);
  }

  Node* adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    adapt_upsample_6_7(graph, node);
    return node;
  }
};

} // namespace version_conversion
} // namespace ONNX_NAMESPACE
