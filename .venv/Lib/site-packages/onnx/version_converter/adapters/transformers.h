// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cinttypes>
#include <string>
#include <utility>
#include <vector>

// Node transformers commonly used in version-adapters:

// Capture context by copying values; the graph is unused by these transformers.

#define NODE_TRANSFORMER(node) [=](std::shared_ptr<Graph>, Node * node)

namespace ONNX_NAMESPACE {
namespace version_conversion {

inline NodeTransformerFunction RemoveAttribute(Symbol attr) {
  return NODE_TRANSFORMER(node) {
    if (node->hasAttribute(attr)) {
      node->removeAttribute(attr);
    }
    return node;
  };
}

inline NodeTransformerFunction RemoveAttribute(Symbol attr, int64_t value) {
  return NODE_TRANSFORMER(node) {
    if (node->hasAttribute(attr)) {
      ONNX_ASSERTM(node->i(attr) == value, "Attribute %s must have value %" PRId64, attr.toString(), value);
      node->removeAttribute(attr);
    }
    return node;
  };
}

inline NodeTransformerFunction RemoveAttributeNotEq(Symbol attr, int64_t value) {
  return NODE_TRANSFORMER(node) {
    if (node->hasAttribute(attr)) {
      ONNX_ASSERTM(node->i(attr) != value, "Attribute %s must not have value %" PRId64, attr.toString(), value);
      node->removeAttribute(attr);
    }
    return node;
  };
}

inline NodeTransformerFunction SetAttribute(Symbol attr, int64_t value) {
  return NODE_TRANSFORMER(node) {
    node->i_(attr, value);
    return node;
  };
}

inline NodeTransformerFunction SetAttribute(Symbol attr, const std::string& value) {
  return NODE_TRANSFORMER(node) {
    node->s_(attr, value);
    return node;
  };
}

inline NodeTransformerFunction SetAttribute(Symbol attr, std::vector<int64_t> value) {
  return NODE_TRANSFORMER(node) {
    std::vector<int64_t> local(value);
    node->is_(attr, std::move(local));
    return node;
  };
}

inline NodeTransformerFunction SetAttributeIfAbsent(Symbol attr, int64_t value) {
  return NODE_TRANSFORMER(node) {
    if (!node->hasAttribute(attr)) {
      node->i_(attr, value);
    }
    return node;
  };
}

} // namespace version_conversion
} // namespace ONNX_NAMESPACE
