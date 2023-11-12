/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "onnx/defs/printer.h"
#include <iomanip>
#include "onnx/defs/tensor_proto_util.h"

namespace ONNX_NAMESPACE {

using MetaDataProp = StringStringEntryProto;
using MetaDataProps = google::protobuf::RepeatedPtrField<StringStringEntryProto>;

class ProtoPrinter {
 public:
  ProtoPrinter(std::ostream& os) : output_(os) {}

  void print(const TensorShapeProto_Dimension& dim);

  void print(const TensorShapeProto& shape);

  void print(const TypeProto_Tensor& tensortype);

  void print(const TypeProto& type);

  void print(const TypeProto_Sequence& seqType);

  void print(const TypeProto_Map& mapType);

  void print(const TypeProto_Optional& optType);

  void print(const TypeProto_SparseTensor& sparseType);

  void print(const TensorProto& tensor);

  void print(const ValueInfoProto& value_info);

  void print(const ValueInfoList& vilist);

  void print(const AttributeProto& attr);

  void print(const AttrList& attrlist);

  void print(const NodeProto& node);

  void print(const NodeList& nodelist);

  void print(const GraphProto& graph);

  void print(const FunctionProto& fn);

  void print(const ModelProto& model);

  void print(const OperatorSetIdProto& opset);

  void print(const OpsetIdList& opsets);

  void print(const MetaDataProps& metadataprops) {
    printSet("[", ", ", "]", metadataprops);
  }

  void print(const MetaDataProp& metadata) {
    printQuoted(metadata.key());
    output_ << ": ";
    printQuoted(metadata.value());
  }

 private:
  template <typename T>
  inline void print(T prim) {
    output_ << prim;
  }

  void printQuoted(const std::string& str) {
    output_ << "\"";
    for (const char* p = str.c_str(); *p; ++p) {
      if ((*p == '\\') || (*p == '"'))
        output_ << '\\';
      output_ << *p;
    }
    output_ << "\"";
  }

  template <typename T>
  inline void printKeyValuePair(KeyWordMap::KeyWord key, const T& val, bool addsep = true) {
    if (addsep)
      output_ << "," << std::endl;
    output_ << std::setw(indent_level) << ' ' << KeyWordMap::ToString(key) << ": ";
    print(val);
  }

  inline void printKeyValuePair(KeyWordMap::KeyWord key, const std::string& val) {
    output_ << "," << std::endl;
    output_ << std::setw(indent_level) << ' ' << KeyWordMap::ToString(key) << ": ";
    printQuoted(val);
  }

  template <typename Collection>
  inline void printSet(const char* open, const char* separator, const char* close, Collection coll) {
    const char* sep = "";
    output_ << open;
    for (auto& elt : coll) {
      output_ << sep;
      print(elt);
      sep = separator;
    }
    output_ << close;
  }

  std::ostream& output_;
  int indent_level = 3;

  void indent() {
    indent_level += 3;
  }

  void outdent() {
    indent_level -= 3;
  }
};

void ProtoPrinter::print(const TensorShapeProto_Dimension& dim) {
  if (dim.has_dim_value())
    output_ << dim.dim_value();
  else if (dim.has_dim_param())
    output_ << dim.dim_param();
  else
    output_ << "?";
}

void ProtoPrinter::print(const TensorShapeProto& shape) {
  printSet("[", ",", "]", shape.dim());
}

void ProtoPrinter::print(const TypeProto_Tensor& tensortype) {
  output_ << PrimitiveTypeNameMap::ToString(tensortype.elem_type());
  if (tensortype.has_shape()) {
    if (tensortype.shape().dim_size() > 0)
      print(tensortype.shape());
  } else
    output_ << "[]";
}

void ProtoPrinter::print(const TypeProto_Sequence& seqType) {
  output_ << "seq(";
  print(seqType.elem_type());
  output_ << ")";
}

void ProtoPrinter::print(const TypeProto_Map& mapType) {
  output_ << "map(" << PrimitiveTypeNameMap::ToString(mapType.key_type()) << ", ";
  print(mapType.value_type());
  output_ << ")";
}

void ProtoPrinter::print(const TypeProto_Optional& optType) {
  output_ << "optional(";
  print(optType.elem_type());
  output_ << ")";
}

void ProtoPrinter::print(const TypeProto_SparseTensor& sparseType) {
  output_ << "sparse_tensor(" << PrimitiveTypeNameMap::ToString(sparseType.elem_type());
  if (sparseType.has_shape()) {
    if (sparseType.shape().dim_size() > 0)
      print(sparseType.shape());
  } else
    output_ << "[]";

  output_ << ")";
}

void ProtoPrinter::print(const TypeProto& type) {
  if (type.has_tensor_type())
    print(type.tensor_type());
  else if (type.has_sequence_type())
    print(type.sequence_type());
  else if (type.has_map_type())
    print(type.map_type());
  else if (type.has_optional_type())
    print(type.optional_type());
  else if (type.has_sparse_tensor_type())
    print(type.sparse_tensor_type());
}

void ProtoPrinter::print(const TensorProto& tensor) {
  output_ << PrimitiveTypeNameMap::ToString(tensor.data_type());
  if (tensor.dims_size() > 0)
    printSet("[", ",", "]", tensor.dims());

  if (!tensor.name().empty()) {
    output_ << " " << tensor.name();
  }
  // TODO: does not yet handle all types or externally stored data.
  if (tensor.has_raw_data()) {
    switch (static_cast<TensorProto::DataType>(tensor.data_type())) {
      case TensorProto::DataType::TensorProto_DataType_INT32:
        printSet(" {", ",", "}", ParseData<int32_t>(&tensor));
        break;
      case TensorProto::DataType::TensorProto_DataType_INT64:
        printSet(" {", ",", "}", ParseData<int64_t>(&tensor));
        break;
      case TensorProto::DataType::TensorProto_DataType_FLOAT:
        printSet(" {", ",", "}", ParseData<float>(&tensor));
        break;
      case TensorProto::DataType::TensorProto_DataType_DOUBLE:
        printSet(" {", ",", "}", ParseData<double>(&tensor));
        break;
      default:
        output_ << "..."; // ParseData not instantiated for other types.
        break;
    }
  } else {
    switch (static_cast<TensorProto::DataType>(tensor.data_type())) {
      case TensorProto::DataType::TensorProto_DataType_INT8:
      case TensorProto::DataType::TensorProto_DataType_INT16:
      case TensorProto::DataType::TensorProto_DataType_INT32:
      case TensorProto::DataType::TensorProto_DataType_UINT8:
      case TensorProto::DataType::TensorProto_DataType_UINT16:
      case TensorProto::DataType::TensorProto_DataType_BOOL:
        printSet(" {", ",", "}", tensor.int32_data());
        break;
      case TensorProto::DataType::TensorProto_DataType_INT64:
        printSet(" {", ",", "}", tensor.int64_data());
        break;
      case TensorProto::DataType::TensorProto_DataType_UINT32:
      case TensorProto::DataType::TensorProto_DataType_UINT64:
        printSet(" {", ",", "}", tensor.uint64_data());
        break;
      case TensorProto::DataType::TensorProto_DataType_FLOAT:
        printSet(" {", ",", "}", tensor.float_data());
        break;
      case TensorProto::DataType::TensorProto_DataType_DOUBLE:
        printSet(" {", ",", "}", tensor.double_data());
        break;
      case TensorProto::DataType::TensorProto_DataType_STRING: {
        const char* sep = "{";
        for (auto& elt : tensor.string_data()) {
          output_ << sep;
          printQuoted(elt);
          sep = ", ";
        }
        output_ << "}";
        break;
      }
      default:
        break;
    }
  }
}

void ProtoPrinter::print(const ValueInfoProto& value_info) {
  print(value_info.type());
  output_ << " " << value_info.name();
}

void ProtoPrinter::print(const ValueInfoList& vilist) {
  printSet("(", ", ", ")", vilist);
}

void ProtoPrinter::print(const AttributeProto& attr) {
  // Special case of attr-ref:
  if (attr.has_ref_attr_name()) {
    output_ << attr.name() << ": " << AttributeTypeNameMap::ToString(attr.type()) << " = @" << attr.ref_attr_name();
    return;
  }
  // General case:
  output_ << attr.name() << " = ";
  switch (attr.type()) {
    case AttributeProto_AttributeType_INT:
      output_ << attr.i();
      break;
    case AttributeProto_AttributeType_INTS:
      printSet("[", ", ", "]", attr.ints());
      break;
    case AttributeProto_AttributeType_FLOAT:
      output_ << attr.f();
      break;
    case AttributeProto_AttributeType_FLOATS:
      printSet("[", ", ", "]", attr.floats());
      break;
    case AttributeProto_AttributeType_STRING:
      output_ << "\"" << attr.s() << "\"";
      break;
    case AttributeProto_AttributeType_STRINGS: {
      const char* sep = "[";
      for (auto& elt : attr.strings()) {
        output_ << sep << "\"" << elt << "\"";
        sep = ", ";
      }
      output_ << "]";
      break;
    }
    case AttributeProto_AttributeType_GRAPH:
      indent();
      print(attr.g());
      outdent();
      break;
    case AttributeProto_AttributeType_GRAPHS:
      indent();
      printSet("[", ", ", "]", attr.graphs());
      outdent();
      break;
    case AttributeProto_AttributeType_TENSOR:
      print(attr.t());
      break;
    case AttributeProto_AttributeType_TENSORS:
      printSet("[", ", ", "]", attr.tensors());
      break;
    case AttributeProto_AttributeType_TYPE_PROTO:
      print(attr.tp());
      break;
    case AttributeProto_AttributeType_TYPE_PROTOS:
      printSet("[", ", ", "]", attr.type_protos());
      break;
    default:
      break;
  }
}

void ProtoPrinter::print(const AttrList& attrlist) {
  printSet(" <", ", ", ">", attrlist);
}

void ProtoPrinter::print(const NodeProto& node) {
  output_ << std::setw(indent_level) << ' ';
  printSet("", ", ", "", node.output());
  output_ << " = ";
  if (node.domain() != "")
    output_ << node.domain() << ".";
  output_ << node.op_type();
  bool has_subgraph = false;
  for (auto attr : node.attribute())
    if (attr.has_g() || (attr.graphs_size() > 0))
      has_subgraph = true;
  if ((!has_subgraph) && (node.attribute_size() > 0))
    print(node.attribute());
  printSet(" (", ", ", ")", node.input());
  if ((has_subgraph) && (node.attribute_size() > 0))
    print(node.attribute());
  output_ << "\n";
}

void ProtoPrinter::print(const NodeList& nodelist) {
  output_ << "{\n";
  for (auto& node : nodelist) {
    print(node);
  }
  if (indent_level > 3)
    output_ << std::setw(indent_level - 3) << "   ";
  output_ << "}";
}

void ProtoPrinter::print(const GraphProto& graph) {
  output_ << graph.name() << " " << graph.input() << " => " << graph.output() << " ";
  print(graph.node());
}

void ProtoPrinter::print(const ModelProto& model) {
  output_ << "<\n";
  printKeyValuePair(KeyWordMap::KeyWord::IR_VERSION, model.ir_version(), false);
  printKeyValuePair(KeyWordMap::KeyWord::OPSET_IMPORT, model.opset_import());
  if (model.has_producer_name())
    printKeyValuePair(KeyWordMap::KeyWord::PRODUCER_NAME, model.producer_name());
  if (model.has_producer_version())
    printKeyValuePair(KeyWordMap::KeyWord::PRODUCER_VERSION, model.producer_version());
  if (model.has_domain())
    printKeyValuePair(KeyWordMap::KeyWord::DOMAIN_KW, model.domain());
  if (model.has_model_version())
    printKeyValuePair(KeyWordMap::KeyWord::MODEL_VERSION, model.model_version());
  if (model.has_doc_string())
    printKeyValuePair(KeyWordMap::KeyWord::DOC_STRING, model.doc_string());
  if (model.metadata_props_size() > 0)
    printKeyValuePair(KeyWordMap::KeyWord::METADATA_PROPS, model.metadata_props());
  output_ << std::endl << ">" << std::endl;

  print(model.graph());
  for (const auto& fn : model.functions()) {
    output_ << std::endl;
    print(fn);
  }
}

void ProtoPrinter::print(const OperatorSetIdProto& opset) {
  output_ << "\"" << opset.domain() << "\" : " << opset.version();
}

void ProtoPrinter::print(const OpsetIdList& opsets) {
  printSet("[", ", ", "]", opsets);
}

void ProtoPrinter::print(const FunctionProto& fn) {
  output_ << "<\n";
  output_ << "  "
          << "domain: \"" << fn.domain() << "\",\n";
  output_ << "  "
          << "opset_import: ";
  printSet("[", ",", "]", fn.opset_import());
  output_ << "\n>\n";
  output_ << fn.name() << " ";
  if (fn.attribute_size() > 0)
    printSet("<", ",", ">", fn.attribute());
  printSet("(", ", ", ")", fn.input());
  output_ << " => ";
  printSet("(", ", ", ")", fn.output());
  output_ << "\n";
  print(fn.node());
}

#define DEF_OP(T)                                              \
  std::ostream& operator<<(std::ostream& os, const T& proto) { \
    ProtoPrinter printer(os);                                  \
    printer.print(proto);                                      \
    return os;                                                 \
  };

DEF_OP(TensorShapeProto_Dimension)

DEF_OP(TensorShapeProto)

DEF_OP(TypeProto_Tensor)

DEF_OP(TypeProto)

DEF_OP(TensorProto)

DEF_OP(ValueInfoProto)

DEF_OP(ValueInfoList)

DEF_OP(AttributeProto)

DEF_OP(AttrList)

DEF_OP(NodeProto)

DEF_OP(NodeList)

DEF_OP(GraphProto)

DEF_OP(FunctionProto)

DEF_OP(ModelProto)

} // namespace ONNX_NAMESPACE
