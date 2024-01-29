/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <iostream>
#include <string>

#include "onnx/defs/parser.h"
#include "onnx/onnx_pb.h"

namespace ONNX_NAMESPACE {

std::ostream& operator<<(std::ostream& os, const TensorShapeProto_Dimension& dim);

std::ostream& operator<<(std::ostream& os, const TensorShapeProto& shape);

std::ostream& operator<<(std::ostream& os, const TypeProto_Tensor& tensortype);

std::ostream& operator<<(std::ostream& os, const TypeProto& type);

std::ostream& operator<<(std::ostream& os, const TensorProto& tensor);

std::ostream& operator<<(std::ostream& os, const ValueInfoProto& value_info);

std::ostream& operator<<(std::ostream& os, const ValueInfoList& vilist);

std::ostream& operator<<(std::ostream& os, const AttributeProto& attr);

std::ostream& operator<<(std::ostream& os, const AttrList& attrlist);

std::ostream& operator<<(std::ostream& os, const NodeProto& node);

std::ostream& operator<<(std::ostream& os, const NodeList& nodelist);

std::ostream& operator<<(std::ostream& os, const GraphProto& graph);

std::ostream& operator<<(std::ostream& os, const FunctionProto& fn);

std::ostream& operator<<(std::ostream& os, const ModelProto& model);

template <typename ProtoType>
std::string ProtoToString(const ProtoType& proto) {
  std::stringstream ss;
  ss << proto;
  return ss.str();
}

} // namespace ONNX_NAMESPACE
