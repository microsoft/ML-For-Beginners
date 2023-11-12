// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Version converter interface for ONNX models between different opset versions.

#pragma once

#include <stdlib.h>
#include <iostream>
#include <utility>
#include "onnx/common/ir.h"
#include "onnx/common/ir_pb_converter.h"
#include "onnx/common/stl_backports.h"
#include "onnx/defs/schema.h"
#include "onnx/proto_utils.h"
#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE {
namespace version_conversion {

// TODO: Consider creating interface for this class.
class BaseVersionConverter {
  // Schema for adapters: {<op_name>:{<from_domain>$<from_version>:{<to_domain>
  // <to_version>: adapter}}}
 protected:
  std::unordered_map<
      std::string,
      std::unordered_map<std::string, std::unordered_map<std::string, std::unique_ptr<Adapter>>>>
      adapters;

  // Map of All Versions of format {op_name: {domain: {version: schema}}}
  std::unordered_map<std::string, std::unordered_map<std::string, std::map<int64_t, const OpSchema*>>> all_schemas;

 public:
  BaseVersionConverter() = default;

  virtual ~BaseVersionConverter() = default;

  // adapter_lookup should be called in convert_version when the user would
  // like to identify the proper registered adapter in the adapters map for
  // a given Node from a certain version to another. It should only be called
  // when the user knows that an adapter should exist for the given context.
  const Adapter& adapter_lookup(const Node* op, const OpSetID& initial_version, const OpSetID& target_version) const {
    const std::string op_name = op->kind().toString();
    const std::string initial = initial_version.toString();
    const std::string target = target_version.toString();
    // Find appropriate adapter in adapters map for provided initial and target versions
    // TODO: Consider abstracting elements of this that are specific to
    // DefaultConverter to separate methods here and maintain the procedure in Base Converter
    const auto op_adapters = adapters.find(op_name);
    if (op_adapters != adapters.end()) {
      // If we're adapting downwards, we just want to find the one downwards
      // adapter implemented for initial_version. If we're adapting upwards, we
      // want to actually use the SinceVersion value for the given op.
      const auto target_map = op_adapters->second.find(initial);
      if (target_map != op_adapters->second.end()) {
        // Either adapt from SinceVersion or Incompatible Breaking Change
        const auto adapter_ptr = target_map->second.find(target);
        if (adapter_ptr != target_map->second.end()) {
          return *(adapter_ptr->second);
        } else {
          ONNX_ASSERTM(false, "No Adapter To Version %s for %s", target.c_str(), op_name.c_str());
        }
      } else {
        ONNX_ASSERTM(false, "No Adapter From Version %s for %s", initial.c_str(), op_name.c_str());
      }
    } else {
      // No adapters exist for the given op
      ONNX_ASSERTM(false, "No Adapter For %s", op_name.c_str());
    }
  }

  virtual ModelProto
  convert_version(const ModelProto& mp_in, const OpSetID& initial_version, const OpSetID& target_version) const = 0;

  void registerAdapter(std::unique_ptr<Adapter> a_ptr) {
    const OpSetID& iv = a_ptr->initial_version();
    const OpSetID& tv = a_ptr->target_version();
    adapters[a_ptr->name()][iv.toString()][tv.toString()] = std::move(a_ptr);
  }

  void registerAdapter(const char* op, int64_t from, int64_t to, NodeTransformerFunction transformer) {
    registerAdapter(make_unique<GenericAdapter>(op, from, to, transformer));
  }
};

} // namespace version_conversion
} // namespace ONNX_NAMESPACE
