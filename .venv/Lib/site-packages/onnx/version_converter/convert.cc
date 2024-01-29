// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "onnx/version_converter/convert.h"

#include <memory>
#include <string>

namespace ONNX_NAMESPACE {
namespace version_conversion {

ModelProto ConvertVersion(const ModelProto& mp_in, int target_version) {
  // Get initial_opsetid from mp_in
  OpSetID initial_struct(0);
  for (auto it = mp_in.opset_import().begin(); it != mp_in.opset_import().end(); ++it) {
    if (it->domain() == "" || it->domain() == "ai.onnx") {
      initial_struct.setVersion(it->version());
      break;
    }
  }
  OpSetID target_struct = OpSetID(target_version);
  DefaultVersionConverter v;
  return v.convert_version(mp_in, initial_struct, target_struct);
}

void DefaultVersionConverter::convert_graph(
    std::shared_ptr<Graph> g,
    const OpSetID& initial_version,
    const OpSetID& target_version) const {
  assertNonNull(g);

  // TODO: Move to Inter-Domain Converter
  // Get initial model versions
  // std::vector<OpSetID> initial_versions = g->opset_versions_mutable();

  // No conversion necessary if Model has single, equivalent opset version
  // if (initial_versions.size() == 1 && initial_versions[0].version ==
  //    target_version.version && initial_versions[0].domain ==
  //    target_version.domain) {
  //  return mp_in;
  // }

  // Check if versions are valid
  assertInVersionRange(initial_version.version());
  assertInVersionRange(target_version.version());

  // Iterate over all versions to target_version for specified
  int64_t curr_version = initial_version.version();
  int64_t step;
  if (target_version.version() > initial_version.version()) {
    step = 1;
  } else {
    step = -1;
  }
  // Identify index of this domain in g.opset_versions
  unsigned int domain_index = 0;
  for (unsigned int i = 0; i < g->opset_versions_mutable().size(); i++) {
    if (g->opset_versions_mutable()[i].domain() == "") {
      domain_index = i;
    }
  }
  while (curr_version != target_version.version()) {
    debug(
        "curr_version: " + ONNX_NAMESPACE::to_string(curr_version) +
        ", next_version: " + ONNX_NAMESPACE::to_string(curr_version + step));
    Node* cur_op;
    graph_node_list_iterator it = g->begin();
    // Iterate through and call adapter returned by adapter_lookup for ops from
    // current_version opset. We have to manipulate the iterator explicitly because cur_op
    // might change when applying the adapter (e.g. for deprecated ops)
    while (it != g->end()) {
      cur_op = *it;
      debug(std::string("Finding schema for ") + std::string(cur_op->kind().toString()));
      const std::string op_name = cur_op->kind().toString();
      if (op_name == "ConstantFill") {
        if (DEBUG) {
          std::cerr
              << "Warning: skipping schema search for experimental op 'ConstantFill' and keeping the op as is. "
                 "Please be advised the converted model may not be working properly if target runtime does not support this "
                 "experimental op."
              << std::endl;
        }
      } else if (cur_op->domain() != "" && cur_op->domain() != "ai.onnx") {
        if (DEBUG) {
          std::cerr << "Warning: opset domain '" << cur_op->domain() << "' is not supported." << std::endl;
        }
      } else if (op_name != "Undefined" && op_name != "Captured") {
        auto& op_domain_map = all_schemas.at(op_name);
        OpSetID curr_id(curr_version);
        OpSetID next_id(curr_version + step);
        if (searchOpDomainMap(op_domain_map, curr_version, step)) {
          // Op is specifically defined for this domain and version
          auto& op_adapter = adapter_lookup(cur_op, curr_id, next_id);
          // If adapter_lookup returns null, no adapter is present.
          // Error thrown by adapter_lookup
          if (DEBUG) {
            std::cerr << "Applying adapter" << std::endl;
          }
          // adapt should handle replacing node in graph
          cur_op = op_adapter.adapt(g, cur_op);
          it = graph_node_list_iterator(cur_op, kNextDirection);
        }
        // Recursively convert any subgraph attributes
        for (const auto& attr : cur_op->attributeNames()) {
          if (cur_op->kindOf(attr) == AttributeKind::g) {
            convert_graph(cur_op->g(attr), curr_id, next_id);
          }
        }
      }
      it++;
    }
    // Update model version
    curr_version += step;
    g->opset_versions_mutable()[domain_index].incrementVersion(step);
  }
}

ModelProto DefaultVersionConverter::convert_version(
    const ModelProto& mp_in,
    const OpSetID& initial_version,
    const OpSetID& target_version) const {
  const std::string& initial_domain = initial_version.domain();
  const std::string& target_domain = target_version.domain();
  assertDefaultDomain(initial_domain, target_domain);

  for (auto it = mp_in.opset_import().begin(); it != mp_in.opset_import().end(); ++it) {
    if (it->domain() == initial_version.domain()) {
      ONNX_ASSERTM(
          initial_version.version() == it->version(), "initial_version does not reflect current state of model");
    }
  }

  std::shared_ptr<Graph> g(ImportModelProto(mp_in));

  convert_graph(g, initial_version, target_version);

  // Export g as ModelProto
  debug("Finished conversion; returning model");
  ModelProto mp_out = PrepareOutput(mp_in);
  ExportModelProto(&mp_out, g);
  return mp_out;
}

} // namespace version_conversion
} // namespace ONNX_NAMESPACE
