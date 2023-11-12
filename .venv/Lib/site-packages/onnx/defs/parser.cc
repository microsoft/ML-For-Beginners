/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Experimental language syntax and parser for ONNX. Please note that the syntax as formalized
// by this parser is preliminary and may change.

#include <stdexcept>
#include <string>
#include <unordered_map>

#include "onnx/onnx_pb.h"
#include "onnx/string_utils.h"

#include "onnx/defs/parser.h"

#define PARSE_TOKEN(x) CHECK_PARSER_STATUS(ParserBase::Parse(x))
#define PARSE(...) CHECK_PARSER_STATUS(Parse(__VA_ARGS__))
#define MATCH(...) CHECK_PARSER_STATUS(Match(__VA_ARGS__))

namespace ONNX_NAMESPACE {

Status ParserBase::Parse(Literal& result) {
  bool decimal_point = false;
  auto nextch = NextChar();
  auto from = next_;
  if (nextch == '"') {
    ++next_;
    bool has_escape = false;
    while ((next_ < end_) && (*next_ != '"')) {
      if (*next_ == '\\') {
        has_escape = true;
        ++next_;
        if (next_ >= end_)
          return ParseError("Incomplete string literal.");
      }
      ++next_;
    }
    if (next_ >= end_)
      return ParseError("Incomplete string literal.");
    ++next_;
    result.type = LiteralType::STRING_LITERAL;
    if (has_escape) {
      std::string& target = result.value;
      target.clear();
      target.reserve(next_ - from - 2); // upper bound
      // *from is the starting quote. *(next_-1) is the ending quote.
      // Copy what is in-between, except for the escape character
      while (++from < next_ - 1) {
        // Copy current char, if not escape, or next char otherwise.
        target.push_back(*from != '\\' ? (*from) : *(++from));
      }
    } else
      result.value = std::string(from + 1, next_ - from - 2); // skip enclosing quotes
  } else if ((isdigit(nextch) || (nextch == '-'))) {
    ++next_;

    while ((next_ < end_) && (isdigit(*next_) || (*next_ == '.'))) {
      if (*next_ == '.') {
        if (decimal_point)
          break; // Only one decimal point allowed in numeric literal
        decimal_point = true;
      }
      ++next_;
    }

    if (next_ == from)
      return ParseError("Value expected but not found.");

    // Optional exponent syntax: (e|E)(+|-)?[0-9]+
    if ((next_ < end_) && ((*next_ == 'e') || (*next_ == 'E'))) {
      decimal_point = true; // treat as float-literal
      ++next_;
      if ((next_ < end_) && ((*next_ == '+') || (*next_ == '-')))
        ++next_;
      while ((next_ < end_) && (isdigit(*next_)))
        ++next_;
    }

    result.value = std::string(from, next_ - from);
    result.type = decimal_point ? LiteralType::FLOAT_LITERAL : LiteralType::INT_LITERAL;
  }
  return Status::OK();
}

Status OnnxParser::Parse(IdList& idlist) {
  idlist.Clear();
  std::string id;
  ParseOptionalIdentifier(id);
  if (id.empty())
    return Status::OK(); // Treat as empty list of identifiers
  *idlist.Add() = id;
  while (Matches(',')) {
    ParseOptionalIdentifier(id);
    *idlist.Add() = id;
  }
  return Status::OK();
}

Status OnnxParser::Parse(char open, IdList& idlist, char close) {
  idlist.Clear();
  if (Matches(open)) {
    PARSE(idlist);
    MATCH(close);
  }
  return Status::OK();
}

Status OnnxParser::Parse(IdList& idlist, AttrList& attrlist) {
  idlist.Clear();
  attrlist.Clear();
  do {
    std::string id;
    ParseIdentifier(id);
    auto next = NextChar();
    if (next == ':' || next == '=')
      Parse(*attrlist.Add(), id);
    else
      *idlist.Add() = id;
  } while (Matches(','));
  return Status::OK();
}

Status OnnxParser::Parse(char open, IdList& idlist, AttrList& attrlist, char close) {
  if (Matches(open)) {
    PARSE(idlist, attrlist);
    MATCH(close);
  } else {
    idlist.Clear();
    attrlist.Clear();
  }
  return Status::OK();
}

Status OnnxParser::Parse(TensorShapeProto& shape) {
  shape.clear_dim();
  do {
    if (Matches('?')) {
      shape.add_dim();
    } else {
      // Check for a symbolic identifier ...
      std::string id;
      CHECK_PARSER_STATUS(ParseOptionalIdentifier(id));
      if (!id.empty()) {
        shape.add_dim()->set_dim_param(id);
      } else {
        // ...or a integer value
        int64_t dimval = 0;
        PARSE_TOKEN(dimval);
        shape.add_dim()->set_dim_value(dimval);
      }
    }
  } while (Matches(','));
  return Status::OK();
}

Status OnnxParser::Parse(TypeProto& typeProto) {
  std::string id;
  CHECK_PARSER_STATUS(ParseIdentifier(id));
  int dtype = PrimitiveTypeNameMap::Lookup(id);
  if (dtype != 0) {
    auto* tensortype = typeProto.mutable_tensor_type();
    tensortype->set_elem_type(dtype);
    tensortype->clear_shape();
    // Grammar:
    // float indicates scalar (rank 0)
    // float [] indicates unknown rank tensor (not a zero rank tensor)
    // float [one-or-more-dimensions] indicates tensor of known rank > 0.
    if (Matches('[')) {
      if (!Matches(']')) {
        PARSE(*tensortype->mutable_shape());
        MATCH(']');
      }
    } else {
      // Create shape with zero dimensions for scalar
      (void)(tensortype->mutable_shape());
    }
  } else {
    switch (KeyWordMap::Lookup(id)) {
      case KeyWordMap::KeyWord::SEQ_TYPE: {
        // Grammar: seq ( type )
        MATCH('(');
        auto* seqtype = typeProto.mutable_sequence_type();
        PARSE(*seqtype->mutable_elem_type());
        MATCH(')');
        break;
      }
      case KeyWordMap::KeyWord::MAP_TYPE: {
        // Grammar: map ( prim-type , type )
        MATCH('(');
        auto* maptype = typeProto.mutable_map_type();
        CHECK_PARSER_STATUS(ParseIdentifier(id));
        dtype = PrimitiveTypeNameMap::Lookup(id);
        if (dtype == 0) {
          return ParseError("Expecting primitive type as map key type.");
        }
        maptype->set_key_type(dtype);
        MATCH(',');
        PARSE(*maptype->mutable_value_type());
        MATCH(')');
        break;
      }
      case KeyWordMap::KeyWord::OPTIONAL_TYPE: {
        // Grammar: optional ( type )
        MATCH('(');
        auto* opttype = typeProto.mutable_optional_type();
        PARSE(*opttype->mutable_elem_type());
        MATCH(')');
        break;
      }
      case KeyWordMap::KeyWord::SPARSE_TENSOR_TYPE: {
        // Grammar: sparse_tensor ( tensor-type )
        MATCH('(');
        CHECK_PARSER_STATUS(ParseIdentifier(id));
        dtype = PrimitiveTypeNameMap::Lookup(id);
        if (dtype != 0) {
          auto* sparsetype = typeProto.mutable_sparse_tensor_type();
          sparsetype->set_elem_type(dtype);
          sparsetype->clear_shape();
          // Grammar:
          // float indicates scalar (rank 0)
          // float [] indicates unknown rank tensor (not a zero rank tensor)
          // float [one-or-more-dimensions] indicates tensor of known rank > 0.
          if (Matches('[')) {
            if (!Matches(']')) {
              PARSE(*sparsetype->mutable_shape());
              MATCH(']');
            }
          } else {
            // Create shape with zero dimensions for scalar
            (void)(sparsetype->mutable_shape());
          }
        } else {
          return ParseError("Unexpected type in sparse-tensor element type.");
        }
        MATCH(')');
        break;
      }
      default:
        return ParseError("Unexpected type.");
    }
  }
  return Status::OK();
}

Status OnnxParser::Parse(ValueInfoProto& valueinfo) {
  if (NextIsType())
    PARSE(*valueinfo.mutable_type());
  std::string name;
  CHECK_PARSER_STATUS(ParseIdentifier(name));
  valueinfo.set_name(name);
  return Status::OK();
}

Status OnnxParser::Parse(ValueInfoList& vilist) {
  vilist.Clear();
  MATCH('(');
  if (!Matches(')')) {
    do {
      PARSE(*vilist.Add());
    } while (Matches(','));
    MATCH(')');
  }
  return Status::OK();
}

// Each input element is a value-info with an optional initializer of the form "= initial-value".
// The value-info is added to the "inputs", while the initializer is added to initializers.
Status OnnxParser::ParseInput(ValueInfoList& inputs, TensorList& initializers) {
  inputs.Clear();
  if (Matches('(')) {
    if (!Matches(')')) {
      do {
        ValueInfoProto vi;
        PARSE(vi);
        *inputs.Add() = vi;
        if (Matches('=')) {
          // default value for input
          TensorProto& tp = *initializers.Add();
          tp.set_name(vi.name());
          CHECK_PARSER_STATUS(Parse(tp, vi.type()));
        }
      } while (Matches(','));
      MATCH(')');
    }
  }
  return Status::OK();
}

// This is handled slightly different from the inputs.
// Each element is either a value-info or an initializer.
// A value-info is added to the "value_infos", while an initializer is added to initializers.
Status OnnxParser::ParseValueInfo(ValueInfoList& value_infos, TensorList& initializers) {
  value_infos.Clear();
  if (Matches('<')) {
    if (!Matches('>')) {
      do {
        ValueInfoProto vi;
        PARSE(vi);
        if (Matches('=')) {
          // initializer
          TensorProto& tp = *initializers.Add();
          tp.set_name(vi.name());
          CHECK_PARSER_STATUS(Parse(tp, vi.type()));
        } else {
          // valueinfo
          *value_infos.Add() = vi;
        }
      } while (Matches(','));
      MATCH('>');
    }
  }
  return Status::OK();
}

Status OnnxParser::Parse(TensorProto& tensorProto) {
  tensorProto = TensorProto();
  // Parse the concrete tensor-type with numeric dimensions:
  TypeProto typeProto;
  PARSE(typeProto);
  ParseOptionalIdentifier(*tensorProto.mutable_name());
  (void)Matches('='); // Optional, to unify handling of initializers as well as tensor-protos in other contexts
  return Parse(tensorProto, typeProto);
}

// Parse TensorProto data given its type:
Status OnnxParser::Parse(TensorProto& tensorProto, const TypeProto& tensorTypeProto) {
  if (!tensorTypeProto.has_tensor_type())
    return ParseError("Error parsing TensorProto (expected a tensor type).");
  auto elem_type = tensorTypeProto.tensor_type().elem_type();
  tensorProto.set_data_type(elem_type);
  if (!tensorTypeProto.tensor_type().has_shape())
    return ParseError("Error parsing TensorProto (expected a tensor shape).");
  uint64_t n = 1;
  for (auto& dim : tensorTypeProto.tensor_type().shape().dim()) {
    if (!dim.has_dim_value())
      return ParseError("Error parsing TensorProto shape (expected numeric dimension).");
    auto dimval = dim.dim_value();
    tensorProto.add_dims(dimval);
    n *= dimval;
  }

  // tensorProto.mutable_int64_data()->Reserve(n);
  // Parse the actual values:

  int64_t intval;
  uint64_t uintval;
  float floatval;
  double dblval;
  std::string strval;
  MATCH('{');
  if (!Matches('}')) {
    do {
      switch (static_cast<TensorProto::DataType>(elem_type)) {
        case TensorProto::DataType::TensorProto_DataType_INT8:
        case TensorProto::DataType::TensorProto_DataType_INT16:
        case TensorProto::DataType::TensorProto_DataType_INT32:
        case TensorProto::DataType::TensorProto_DataType_UINT8:
        case TensorProto::DataType::TensorProto_DataType_UINT16:
        case TensorProto::DataType::TensorProto_DataType_BOOL:
          PARSE_TOKEN(intval);
          // TODO: check values are in the correct range.
          tensorProto.add_int32_data(intval);
          break;
        case TensorProto::DataType::TensorProto_DataType_INT64:
          PARSE_TOKEN(intval);
          tensorProto.add_int64_data(intval);
          break;
        case TensorProto::DataType::TensorProto_DataType_UINT32:
        case TensorProto::DataType::TensorProto_DataType_UINT64:
          PARSE_TOKEN(uintval);
          tensorProto.add_uint64_data(uintval);
          break;
        case TensorProto::DataType::TensorProto_DataType_FLOAT:
          PARSE_TOKEN(floatval);
          tensorProto.add_float_data(floatval);
          break;
        case TensorProto::DataType::TensorProto_DataType_DOUBLE:
          PARSE_TOKEN(dblval);
          tensorProto.add_double_data(dblval);
          break;
        case TensorProto::DataType::TensorProto_DataType_STRING:
          PARSE_TOKEN(strval);
          tensorProto.add_string_data(strval);
          break;
        default:
          return ParseError("Unhandled type: %d", elem_type);
      }
    } while (Matches(','));
    MATCH('}');
  }
  return Status::OK();
}

bool OnnxParser::NextIsIdentifier() {
  std::string id("");
  (void)PeekIdentifier(id);
  return !(id.empty());
}

bool OnnxParser::NextIsType() {
  std::string id("");
  (void)PeekIdentifier(id);
  if (PrimitiveTypeNameMap::IsTypeName(id))
    return true;
  switch (KeyWordMap::Lookup(id)) {
    case KeyWordMap::KeyWord::SEQ_TYPE:
    case KeyWordMap::KeyWord::MAP_TYPE:
    case KeyWordMap::KeyWord::OPTIONAL_TYPE:
    case KeyWordMap::KeyWord::SPARSE_TENSOR_TYPE:
      return true;
    default:
      return false;
  }
}

Status OnnxParser::ParseSingleAttributeValue(AttributeProto& attr) {
  // Parse a single-value
  auto next = NextChar();
  if (isalpha(next) || next == '_') {
    if (NextIsType()) {
      TypeProto typeProto;
      Parse(typeProto);
      next = NextChar();
      if ((next == '{') || (next == '=') || (NextIsIdentifier())) {
        attr.set_type(AttributeProto_AttributeType_TENSOR);
        auto& tensorProto = *attr.mutable_t();
        ParseOptionalIdentifier(*tensorProto.mutable_name());
        (void)Matches('='); // Optional, to unify handling of initializers
        Parse(tensorProto, typeProto);
      } else {
        attr.set_type(AttributeProto_AttributeType_TYPE_PROTO);
        attr.mutable_tp()->CopyFrom(typeProto);
      }
    } else {
      attr.set_type(AttributeProto_AttributeType_GRAPH);
      Parse(*attr.mutable_g());
    }
  } else if (Matches('@')) {
    std::string name;
    CHECK_PARSER_STATUS(ParseIdentifier(name));
    attr.set_ref_attr_name(name);
  } else {
    Literal literal;
    PARSE_TOKEN(literal);
    switch (literal.type) {
      case LiteralType::INT_LITERAL:
        attr.set_type(AttributeProto_AttributeType_INT);
        attr.set_i(std::stol(literal.value));
        break;
      case LiteralType::FLOAT_LITERAL:
        attr.set_type(AttributeProto_AttributeType_FLOAT);
        attr.set_f(static_cast<float>(std::stof(literal.value)));
        break;
      case LiteralType::STRING_LITERAL:
        attr.set_type(AttributeProto_AttributeType_STRING);
        attr.set_s(literal.value);
        break;
      default:
        return ParseError("Unexpected literal type.");
    }
  }
  return Status::OK();
}

Status OnnxParser::Parse(AttributeProto& attr) {
  attr.Clear();
  std::string name;
  CHECK_PARSER_STATUS(ParseIdentifier(name));
  return Parse(attr, name);
}

Status OnnxParser::Parse(AttributeProto& attr, std::string& name) {
  attr.set_name(name);
  if (Matches(':')) {
    CHECK_PARSER_STATUS(ParseIdentifier(name));
    int attrtype = AttributeTypeNameMap::Lookup(name);
    if (attrtype != 0) {
      attr.set_type(static_cast<AttributeProto_AttributeType>(attrtype));
    } else {
      return ParseError("Unexpected attribute type.");
    }
  }
  MATCH('=');
  if (NextChar() == '[') {
    // Parse a list of values. For now, empty list is not allowed, as we need to
    // figure out a type for the attribute.
    std::vector<Literal> vals;
    MATCH('[');
    do {
      AttributeProto nextval;
      CHECK_PARSER_STATUS(ParseSingleAttributeValue(nextval));
      switch (nextval.type()) {
        case AttributeProto_AttributeType_INT:
          attr.set_type(AttributeProto_AttributeType_INTS);
          attr.add_ints(nextval.i());
          break;
        case AttributeProto_AttributeType_FLOAT:
          attr.set_type(AttributeProto_AttributeType_FLOATS);
          attr.add_floats(nextval.f());
          break;
        case AttributeProto_AttributeType_STRING:
          attr.add_strings(nextval.s());
          attr.set_type(AttributeProto_AttributeType_STRINGS);
          break;
        default:
          break;
      }
    } while (Matches(','));
    MATCH(']');
  } else {
    CHECK_PARSER_STATUS(ParseSingleAttributeValue(attr));
  }
  return Status::OK();
}

Status OnnxParser::Parse(AttrList& attrlist) {
  attrlist.Clear();
  if (Matches('<')) {
    do {
      PARSE(*attrlist.Add());
    } while (Matches(','));
    MATCH('>');
  }
  return Status::OK();
}

Status OnnxParser::Parse(NodeProto& node) {
  PARSE(*node.mutable_output());
  MATCH('=');
  std::string domain("");
  std::string id;
  ParseIdentifier(id);
  while (Matches('.')) {
    if (!domain.empty())
      domain += ".";
    domain += id;
    ParseIdentifier(id);
  }
  node.set_domain(domain);
  node.set_op_type(id);
  PARSE(*node.mutable_attribute());
  MATCH('(');
  PARSE(*node.mutable_input());
  MATCH(')');
  if (node.attribute_size() == 0) {
    // Permit attributes to be specified before or after parameters.
    PARSE(*node.mutable_attribute());
  }
  return Status::OK();
}

Status OnnxParser::Parse(NodeList& nodelist) {
  nodelist.Clear();
  MATCH('{');
  while (!Matches('}')) {
    PARSE(*nodelist.Add());
  }
  return Status::OK();
}

Status OnnxParser::Parse(GraphProto& graph) {
  std::string id;
  ParseIdentifier(id);
  return Parse(id, graph);
}

Status OnnxParser::Parse(std::string name, GraphProto& graph) {
  graph.set_name(name);
  graph.mutable_initializer()->Clear();
  CHECK_PARSER_STATUS(ParseInput(*graph.mutable_input(), *graph.mutable_initializer()));
  MATCH('=');
  MATCH('>', false);
  PARSE(*graph.mutable_output());
  CHECK_PARSER_STATUS(ParseValueInfo(*graph.mutable_value_info(), *graph.mutable_initializer()));
  return Parse(*graph.mutable_node());
}

Status OnnxParser::Parse(FunctionProto& fn) {
  fn.Clear();
  std::string strval;
  if (Matches('<')) {
    do {
      KeyWordMap::KeyWord keyword = KeyWordMap::KeyWord::NONE;
      PARSE_TOKEN(keyword);
      MATCH(':');
      switch (keyword) {
        case KeyWordMap::KeyWord::OPSET_IMPORT:
          PARSE(*fn.mutable_opset_import());
          break;
        case KeyWordMap::KeyWord::DOC_STRING:
          PARSE_TOKEN(strval);
          fn.set_doc_string(strval);
          break;
        case KeyWordMap::KeyWord::DOMAIN_KW:
          PARSE_TOKEN(strval);
          fn.set_domain(strval);
          break;
        default:
          return ParseError("Unhandled keyword.");
      }
    } while (Matches(','));
    MATCH('>');
  }
  std::string id;
  ParseIdentifier(id);
  fn.set_name(id);

  PARSE('<', *fn.mutable_attribute(), *fn.mutable_attribute_proto(), '>');
  PARSE('(', *fn.mutable_input(), ')');
  MATCH('=');
  MATCH('>', false);
  PARSE('(', *fn.mutable_output(), ')');
  return Parse(*fn.mutable_node());
}

Status OnnxParser::Parse(OpsetIdList& opsets) {
  std::string strval;
  int64_t intval = 0;
  MATCH('[');
  if (!Matches(']')) {
    do {
      auto* import = opsets.Add();
      PARSE_TOKEN(strval);
      import->set_domain(strval);
      MATCH(':');
      PARSE_TOKEN(intval);
      import->set_version(intval);
    } while (Matches(','));
    MATCH(']');
  }
  return Status::OK();
}

Status OnnxParser::Parse(ModelProto& model) {
  model.Clear();
  std::string strval;
  int64_t intval;
  if (Matches('<')) {
    do {
      KeyWordMap::KeyWord keyword = KeyWordMap::KeyWord::NONE;
      PARSE_TOKEN(keyword);
      MATCH(':');
      switch (keyword) {
        case KeyWordMap::KeyWord::IR_VERSION:
          PARSE_TOKEN(intval);
          model.set_ir_version(intval);
          break;
        case KeyWordMap::KeyWord::OPSET_IMPORT:
          PARSE(*model.mutable_opset_import());
          break;
        case KeyWordMap::KeyWord::PRODUCER_NAME:
          PARSE_TOKEN(strval);
          model.set_producer_name(strval);
          break;
        case KeyWordMap::KeyWord::PRODUCER_VERSION:
          PARSE_TOKEN(strval);
          model.set_producer_version(strval);
          break;
        case KeyWordMap::KeyWord::DOMAIN_KW:
          PARSE_TOKEN(strval);
          model.set_domain(strval);
          break;
        case KeyWordMap::KeyWord::MODEL_VERSION:
          PARSE_TOKEN(intval);
          model.set_model_version(intval);
          break;
        case KeyWordMap::KeyWord::DOC_STRING:
          PARSE_TOKEN(strval);
          model.set_doc_string(strval);
          break;
        case KeyWordMap::KeyWord::METADATA_PROPS: {
          auto& metadata_props = *model.mutable_metadata_props();
          MATCH('[');
          if (!Matches(']')) {
            do {
              auto* metadata = metadata_props.Add();
              PARSE_TOKEN(strval);
              metadata->set_key(strval);
              MATCH(':');
              PARSE_TOKEN(strval);
              metadata->set_value(strval);
            } while (Matches(','));
            MATCH(']');
          }
          break;
        }
        default:
          return ParseError("Unhandled keyword.");
      }
    } while (Matches(','));
    MATCH('>');
  }
  PARSE(*model.mutable_graph());

  auto* functions = model.mutable_functions();
  while (!EndOfInput()) {
    PARSE(*functions->Add());
  }
  return Status::OK();
}

} // namespace ONNX_NAMESPACE
