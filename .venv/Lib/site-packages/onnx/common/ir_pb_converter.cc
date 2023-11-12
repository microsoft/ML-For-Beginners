// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#include "onnx/common/ir_pb_converter.h"
#include <sstream>

namespace ONNX_NAMESPACE {

// Part 1: convert ONNX Protobuf to IR
std::unique_ptr<Graph> graphProtoToGraph(const GraphProto& gp, bool nested, const int ir_version = IR_VERSION);

Tensor tensorProtoToTensor(const ONNX_NAMESPACE::TensorProto& tp) {
  Tensor ret;

  ret.sizes().reserve(tp.dims_size());
  for (int i = 0; i < tp.dims_size(); i++) {
    ret.sizes().push_back(tp.dims(i));
  }

  ret.elem_type() = tp.data_type();
  switch (tp.data_type()) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
    case ONNX_NAMESPACE::TensorProto_DataType_COMPLEX64: {
      ret.floats().reserve(tp.float_data_size());
      for (int i = 0; i < tp.float_data_size(); i++) {
        ret.floats().push_back(tp.float_data(i));
      }
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
    case ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16:
    case ONNX_NAMESPACE::TensorProto_DataType_BOOL:
    case ONNX_NAMESPACE::TensorProto_DataType_INT8:
    case ONNX_NAMESPACE::TensorProto_DataType_INT16:
    case ONNX_NAMESPACE::TensorProto_DataType_INT32:
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
    case ONNX_NAMESPACE::TensorProto_DataType_UINT16:
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E4M3FN:
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E4M3FNUZ:
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E5M2:
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E5M2FNUZ: {
      ret.int32s().reserve(tp.int32_data_size());
      for (int i = 0; i < tp.int32_data_size(); i++) {
        ret.int32s().push_back(tp.int32_data(i));
      }
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_INT64: {
      ret.int64s().reserve(tp.int64_data_size());
      for (int i = 0; i < tp.int64_data_size(); i++) {
        ret.int64s().push_back(tp.int64_data(i));
      }
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_UINT32:
    case ONNX_NAMESPACE::TensorProto_DataType_UINT64: {
      ret.uint64s().reserve(tp.uint64_data_size());
      for (int i = 0; i < tp.uint64_data_size(); i++) {
        ret.uint64s().push_back(tp.uint64_data(i));
      }
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
    case ONNX_NAMESPACE::TensorProto_DataType_COMPLEX128: {
      ret.doubles().reserve(tp.double_data_size());
      for (int i = 0; i < tp.double_data_size(); i++) {
        ret.doubles().push_back(tp.double_data(i));
      }
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_STRING: {
      ret.strings().reserve(tp.string_data_size());
      for (int i = 0; i < tp.string_data_size(); i++) {
        ret.strings().push_back(tp.string_data(i));
      }
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED:
      fail_convert("Unknown tensor data type");
  }

  // The only way to know if we should be using raw_data or
  // <type>_data is to look at which of them is size zero.
  if (tp.has_raw_data()) {
    ret.set_raw_data(tp.raw_data());
  }

  if (tp.has_name()) {
    ret.setName(tp.name());
  }
  if (tp.has_segment()) {
    ret.set_segment_begin_and_end(tp.segment().begin(), tp.segment().end());
  }
  return ret;
}

void convertAttribute(const ONNX_NAMESPACE::AttributeProto& ap, Node* n, const int ir_version = IR_VERSION) {
  Symbol sym = Symbol(ap.name());
  switch (ap.type()) {
    case ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT:
      n->f_(sym, ap.f());
      break;
    case ONNX_NAMESPACE::AttributeProto_AttributeType_FLOATS: {
      std::vector<double> floats;
      floats.reserve(ap.floats_size());
      for (int i = 0; i < ap.floats_size(); i++) {
        floats.push_back(ap.floats(i));
      }
      n->fs_(sym, std::move(floats));
      break;
    }
    case ONNX_NAMESPACE::AttributeProto_AttributeType_INT:
      n->i_(sym, ap.i());
      break;
    case ONNX_NAMESPACE::AttributeProto_AttributeType_INTS: {
      std::vector<int64_t> ints;
      ints.reserve(ap.ints_size());
      for (int i = 0; i < ap.ints_size(); i++) {
        ints.push_back(ap.ints(i));
      }
      n->is_(sym, std::move(ints));
      break;
    }
    case ONNX_NAMESPACE::AttributeProto_AttributeType_STRING:
      n->s_(sym, ap.s());
      break;
    case ONNX_NAMESPACE::AttributeProto_AttributeType_STRINGS: {
      std::vector<std::string> strings;
      strings.reserve(ap.strings_size());
      for (int i = 0; i < ap.strings_size(); i++) {
        strings.push_back(ap.strings(i));
      }
      n->ss_(sym, std::move(strings));
      break;
    }
    case ONNX_NAMESPACE::AttributeProto_AttributeType_TENSOR:
      n->t_(sym, tensorProtoToTensor(ap.t()));
      break;
    case ONNX_NAMESPACE::AttributeProto_AttributeType_TENSORS: {
      std::vector<Tensor> tensors;
      tensors.reserve(ap.tensors_size());
      for (int i = 0; i < ap.tensors_size(); i++) {
        tensors.push_back(tensorProtoToTensor(ap.tensors(i)));
      }
      n->ts_(sym, std::move(tensors));
      break;
    }
    case ONNX_NAMESPACE::AttributeProto_AttributeType_TYPE_PROTO:
      n->tp_(sym, ap.tp());
      break;
    case ONNX_NAMESPACE::AttributeProto_AttributeType_TYPE_PROTOS: {
      std::vector<TypeProto> types;
      types.reserve(ap.type_protos_size());
      for (int i = 0; i < ap.type_protos_size(); i++) {
        types.push_back(ap.type_protos(i));
      }
      n->tps_(sym, std::move(types));
      break;
    }
    case ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH:
      n->g_(sym, graphProtoToGraph(ap.g(), true, ir_version));
      break;
    case ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPHS: {
      std::vector<std::shared_ptr<Graph>> graphs;
      graphs.reserve(ap.graphs_size());
      for (int i = 0; i < ap.graphs_size(); i++) {
        graphs.push_back(graphProtoToGraph(ap.graphs(i), true, ir_version));
      }
      n->gs_(sym, std::move(graphs));
      break;
    }
    case ONNX_NAMESPACE::AttributeProto_AttributeType_SPARSE_TENSOR:
    case ONNX_NAMESPACE::AttributeProto_AttributeType_SPARSE_TENSORS:
      fail_convert("Sparse tensors not supported.");
      break;
    case ONNX_NAMESPACE::AttributeProto_AttributeType_UNDEFINED:
      fail_convert("Unknown tensor data type");
      break;
  }
}

void convertAttributes(ONNX_NAMESPACE::NodeProto& np, Node* n, const int ir_version = IR_VERSION) {
  for (int i = 0; i < np.attribute_size(); i++) {
    convertAttribute(np.attribute(i), n, ir_version);
  }
}

std::vector<Dimension> tensorShapeProtoToDimensions(const ONNX_NAMESPACE::TensorShapeProto& tsp) {
  std::vector<Dimension> dims;
  dims.reserve(tsp.dim_size());
  for (int i = 0; i < tsp.dim_size(); i++) {
    if (tsp.dim(i).has_dim_value()) {
      dims.emplace_back(tsp.dim(i).dim_value());
    } else if (tsp.dim(i).has_dim_param()) {
      dims.emplace_back(tsp.dim(i).dim_param());
    } else {
      // a dimension that has neither dim_value nor dim_param set
      // represents an unknown dimension unrelated to other unknown dimensions.
      dims.emplace_back();
    }
  }
  return dims;
}

void createDummyValue(
    std::unique_ptr<Graph>& g,
    const std::string& name,
    std::unordered_map<std::string, Value*>& value_by_name_of) {
  auto* undef = g->create(kCaptured, 1);
  g->appendNode(undef);
  undef->outputs()[0]->setUniqueName(name);
  value_by_name_of[name] = undef->outputs()[0];
}

std::unique_ptr<Graph> graphProtoToGraph(const ONNX_NAMESPACE::GraphProto& gp, bool nested, const int ir_version) {
  std::unique_ptr<Graph> g(new Graph());

  if (gp.has_name()) {
    g->setName(gp.name());
  }
  if (gp.has_doc_string()) {
    g->setDocString(gp.doc_string());
  }

  // Values are created (as in `new Value(..)`) by the Node that
  // outputs them. Therefore we initialize the Nodes and Values in
  // several stages.
  //
  // 1) add all input (to the graph) Values, owned by the sentinel Param node
  // 2) add all Nodes and their output Values, but don't intialize inputs
  // 3) initialize inputs of all Nodes
  // 4) initialize inputs of the Return sentinel node
  // 5) fill in type info for graph outputs, and register them as outputs
  // 6) fill in type info for Values from the value_info list in the graph

  // In ONNX proto land, Values are just strings. We are going to make
  // objects out of them, and equal strings must be mapped to the same
  // Value object.
  std::unordered_map<std::string, Value*> value_by_name_of;

  // We initialize Node inputs in a separate pass from the Nodes
  // themselves. To do so, we need to have access to the names of the
  // inputs.
  std::unordered_map<Node*, std::vector<std::string>> inputs_by_node;

  {
    // ONNX represents optional arguments in two ways
    //  - they are simply not provided
    //  - OR the empty string is passed as the input name
    // This is to handle that second case, which needs a dummy node to
    // be representable in the graph IR.
    auto* n = g->create(kUndefined, 1);
    g->appendNode(n);
    n->outputs()[0]->setUniqueName("");
    value_by_name_of[""] = n->outputs()[0];
  }

  for (int i = 0; i < gp.input_size(); i++) {
    const auto& vip = gp.input(i);
    auto v = g->addInput();
    const auto& tensor_type = vip.type().tensor_type();
    if (tensor_type.has_elem_type()) {
      v->setElemType(tensor_type.elem_type());
    }
    if (tensor_type.has_shape()) {
      v->setSizes(tensorShapeProtoToDimensions(tensor_type.shape()));
    }
    v->setUniqueName(vip.name());
    value_by_name_of[vip.name()] = v;
  }

  // initializers should be added before all nodes,
  // otherwise getNextUnique() may conflicts with an existing initializer name.
  for (int i = 0; i < gp.initializer_size(); ++i) {
    auto init = tensorProtoToTensor(gp.initializer(i));
    // If ir_version >= 4, initializer does not have to be included in input
    // Create a Value from initializer by addInitializerNode if name does not exist in input
    // and save it into value_by_name_of for later use (node input)
    if (ir_version >= 4 && value_by_name_of.count(init.name()) == 0) {
      value_by_name_of[init.name()] = g->addInitializerAndCreateValue(init);
    } else {
      // If ir_version < 4 or the initializer exists in input
      // Simply add initializer without creating new value
      // which means it will prioritize input value over initializer value if both exist
      g->addInitializer(init);
    }
  }

  for (int i = 0; i < gp.node_size(); i++) {
    auto np = gp.node(i);
    auto* n = g->create(Symbol(np.op_type()), /* num_outputs = */ np.output_size());
    g->appendNode(n);
    for (int j = 0; j < np.output_size(); j++) {
      auto out = n->outputs()[j];
      // we don't know the real type here, so that's done in a later pass
      out->setElemType(ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED);
      out->setUniqueName(np.output(j));
      value_by_name_of[np.output(j)] = out;
    }
    convertAttributes(np, n, ir_version);
    std::vector<std::string> inputs;
    inputs.reserve(np.input_size());
    for (int j = 0; j < np.input_size(); j++) {
      inputs.push_back(np.input(j));
    }
    inputs_by_node[n] = inputs;
    if (np.has_doc_string()) {
      n->setDocString(np.doc_string());
    }
    if (np.has_name()) {
      n->setName(np.name());
    }
    if (np.has_domain()) {
      n->setDomain(np.domain());
    }
  }

  for (auto n : g->nodes()) {
    auto search = inputs_by_node.find(n);
    if (search == inputs_by_node.end()) {
      continue;
    }
    for (const auto& input : search->second) {
      if (!value_by_name_of.count(input) && nested) {
        // Undefined reference to an input in a nested block. This may be a
        // captured value. Create a dummy node that we ignore later.
        createDummyValue(g, input, value_by_name_of);
      }

      if (!value_by_name_of.count(input)) {
        std::ostringstream msg;
        msg << "Input " << input << " is undefined!";
        ONNX_THROW_EX(std::out_of_range(msg.str()));
      }
      n->addInput(value_by_name_of.at(input));
    }
  }

  for (int i = 0; i < gp.output_size(); i++) {
    if (!value_by_name_of.count(gp.output(i).name()) && nested) {
      // Same captured value logic as above. We can consider outputs of a
      // graph to be "inputs" of a dummy "output" node. The same lexical
      // scoping rules are valid here, thus we need to add a dummy node
      // in the case of an undefined reference
      createDummyValue(g, gp.output(i).name(), value_by_name_of);
    }
    const auto& output_tensor_type = gp.output(i).type().tensor_type();
    if (output_tensor_type.has_elem_type()) {
      value_by_name_of[gp.output(i).name()]->setElemType(output_tensor_type.elem_type());
    }
    if (output_tensor_type.has_shape()) {
      value_by_name_of[gp.output(i).name()]->setSizes(tensorShapeProtoToDimensions(output_tensor_type.shape()));
    }
    g->registerOutput(value_by_name_of[gp.output(i).name()]);
  }

  for (int i = 0; i < gp.value_info_size(); i++) {
    const auto& tensor_type = gp.value_info(i).type().tensor_type();
    if (!value_by_name_of.count(gp.value_info(i).name())) {
      // Ideally the model should not have a value_info whose name does not exist in the graph (unused); simply skip it
      continue;
    }
    if (tensor_type.has_elem_type()) {
      value_by_name_of[gp.value_info(i).name()]->setElemType(tensor_type.elem_type());
    }
    if (tensor_type.has_shape()) {
      value_by_name_of[gp.value_info(i).name()]->setSizes(tensorShapeProtoToDimensions(tensor_type.shape()));
    }
  }

  return g;
}

std::unique_ptr<Graph> ImportModelProto(const ModelProto& mp) {
  if (!mp.has_ir_version()) {
    return nullptr;
  } else if (mp.ir_version() <= 1) {
    // ir_version=1 is not supported and ir_version=0 is illegal
    return nullptr;
  }

  std::unique_ptr<Graph> g(graphProtoToGraph(mp.graph(), false, mp.ir_version()));
  for (int i = 0; i < mp.opset_import_size(); i++) {
    OpSetID new_opset_version(mp.opset_import(i).domain(), mp.opset_import(i).version());
    g->forSelfAndEachSubGraph(
        [&new_opset_version](Graph* graph) { graph->opset_versions_mutable().emplace_back(new_opset_version); });
  }
  return g;
}

// Part 2: convert IR to ONNX Protobuf
std::string value_name(Value* n) {
  return n->uniqueName();
}

void encodeGraph(GraphProto* p_g, const std::shared_ptr<Graph>& g);

void encodeTensor(ONNX_NAMESPACE::TensorProto* p, const Tensor& tensor) {
  if (tensor.hasName()) {
    p->set_name(tensor.name());
  }
  if (tensor.is_segment()) {
    ONNX_NAMESPACE::TensorProto_Segment segment;
    segment.set_begin(tensor.segment_begin());
    segment.set_end(tensor.segment_end());
    p->mutable_segment()->CopyFrom(segment);
  }
  for (auto d : tensor.sizes()) {
    p->add_dims(d);
  }
  p->set_data_type(tensor.elem_type());
  switch (tensor.elem_type()) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
    case ONNX_NAMESPACE::TensorProto_DataType_COMPLEX64: {
      for (float x : tensor.floats()) {
        p->add_float_data(x);
      }
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
    case ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16:
    case ONNX_NAMESPACE::TensorProto_DataType_BOOL:
    case ONNX_NAMESPACE::TensorProto_DataType_INT8:
    case ONNX_NAMESPACE::TensorProto_DataType_INT16:
    case ONNX_NAMESPACE::TensorProto_DataType_INT32:
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
    case ONNX_NAMESPACE::TensorProto_DataType_UINT16: {
      for (int32_t x : tensor.int32s()) {
        p->add_int32_data(x);
      }
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_INT64: {
      for (int64_t x : tensor.int64s()) {
        p->add_int64_data(x);
      }
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_UINT32:
    case ONNX_NAMESPACE::TensorProto_DataType_UINT64: {
      for (uint64_t x : tensor.uint64s()) {
        p->add_uint64_data(x);
      }
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
    case ONNX_NAMESPACE::TensorProto_DataType_COMPLEX128: {
      for (double x : tensor.doubles()) {
        p->add_double_data(x);
      }
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_STRING: {
      for (const std::string& x : tensor.strings()) {
        p->add_string_data(x);
      }
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED:
      fail_convert("Unknown tensor data type");
  }
  if (tensor.is_raw_data()) {
    p->set_raw_data(tensor.raw());
  }
}

void addAttribute(ONNX_NAMESPACE::NodeProto* n_p, Node* n, Symbol name) {
  auto attr = n_p->add_attribute();
  attr->set_name(name.toString());
  switch (n->kindOf(name)) {
    case AttributeKind::f: {
      attr->set_f(static_cast<float>(n->f(name)));
      attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT);
    } break;
    case AttributeKind::fs: {
      attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_FLOATS);
      for (auto& v : n->fs(name))
        attr->add_floats(static_cast<float>(v));
    } break;
    case AttributeKind::i: {
      attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
      attr->set_i(n->i(name));
    } break;
    case AttributeKind::is: {
      attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INTS);
      for (auto& v : n->is(name))
        attr->add_ints(v);
    } break;
    case AttributeKind::s: {
      attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_STRING);
      attr->set_s(n->s(name));
    } break;
    case AttributeKind::ss: {
      attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_STRINGS);
      for (auto& v : n->ss(name))
        attr->add_strings(v);
    } break;
    case AttributeKind::t: {
      attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_TENSOR);
      auto t = attr->mutable_t();
      encodeTensor(t, n->t(name));
    } break;
    case AttributeKind::ts: {
      attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_TENSORS);
      for (auto& v : n->ts(name)) {
        auto t = attr->add_tensors();
        encodeTensor(t, v);
      }
    } break;
    case AttributeKind::g: {
      attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH);
      auto g = attr->mutable_g();
      encodeGraph(g, n->g(name));
    } break;
    case AttributeKind::gs: {
      attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPHS);
      for (auto& v : n->gs(name)) {
        auto g = attr->add_graphs();
        encodeGraph(g, v);
      }
    } break;
    case AttributeKind::tp: {
      attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_TYPE_PROTO);
      auto tp = attr->mutable_tp();
      tp->CopyFrom(n->tp(name));
    } break;
    case AttributeKind::tps: {
      attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_TYPE_PROTOS);
      for (auto& v : n->tps(name)) {
        auto tp = attr->add_type_protos();
        tp->CopyFrom(v);
      }
    } break;
  }
}

void encodeTypeProtoTensorType(ONNX_NAMESPACE::TypeProto_Tensor* tensor_type, Value* n) {
  if (n->elemType() != 0) {
    tensor_type->set_elem_type(n->elemType());
  }
  if (n->has_sizes()) {
    ONNX_NAMESPACE::TensorShapeProto* shape = tensor_type->mutable_shape();
    for (const Dimension& d : n->sizes()) {
      auto dim = shape->add_dim();
      if (!d.is_unknown) {
        if (d.is_int) {
          dim->set_dim_value(d.dim);
        } else {
          dim->set_dim_param(d.param);
        }
      }
    }
  }
}

void encodeValueInfo(ONNX_NAMESPACE::ValueInfoProto* v, Value* n) {
  v->set_name(value_name(n));
  if (n->elemType() != 0 || n->has_sizes()) {
    ONNX_NAMESPACE::TypeProto* t = v->mutable_type();
    ONNX_NAMESPACE::TypeProto_Tensor* tensor_type = t->mutable_tensor_type();
    encodeTypeProtoTensorType(tensor_type, n);
  }
}

void encodeGraph(GraphProto* p_g, const std::shared_ptr<Graph>& g) {
  ONNX_ASSERT(p_g != nullptr);

  if (g->has_name()) {
    p_g->set_name(g->name());
  }

  if (g->has_doc_string()) {
    p_g->set_doc_string(g->docString());
  }

  for (auto input : g->inputs()) {
    ONNX_NAMESPACE::ValueInfoProto* v = p_g->add_input();
    encodeValueInfo(v, input);
  }
  for (auto output : g->outputs()) {
    ONNX_NAMESPACE::ValueInfoProto* v = p_g->add_output();
    encodeValueInfo(v, output);
  }

  std::unordered_set<Value*> graph_outputs(g->outputs().begin(), g->outputs().end());

  for (auto node : g->nodes()) {
    if (node->kind() == kUndefined || node->kind() == kCaptured) {
      // Undefined nodes are used to represent optional inputs that are not
      // provided.
      continue;
    }
    auto p_n = p_g->add_node();
    for (auto input : node->inputs()) {
      if (input->node()->kind() == kUndefined) {
        p_n->add_input("");
      } else {
        p_n->add_input(value_name(input));
      }
    }
    for (auto output : node->outputs()) {
      p_n->add_output(value_name(output));
      // only save it if
      //  - it has actual information worth saving
      //  - it's not already saved in the graph outputs value info
      if (graph_outputs.find(output) != graph_outputs.end()) {
        continue;
      }
      if (output->elemType() == TensorProto_DataType_UNDEFINED && output->sizes().empty()) {
        continue;
      }
      ValueInfoProto* v = p_g->add_value_info();
      encodeValueInfo(v, output);
    }
    p_n->set_op_type(node->kind().toString());
    for (auto attr_name : node->attributeNames()) {
      addAttribute(p_n, node, attr_name);
    }
    if (node->has_doc_string()) {
      p_n->set_doc_string(node->docString());
    }
    if (node->has_name()) {
      p_n->set_name(node->name());
    }
    if (node->has_domain()) {
      p_n->set_domain(node->domain());
    }
  }

  auto num_initializers = g->initializers().size();
  for (unsigned int i = 0; i < num_initializers; i++) {
    auto p = p_g->add_initializer();
    p->set_name(g->initializer_names()[i]);
    encodeTensor(p, g->initializers()[i]);
  }
}

void ExportModelProto(ModelProto* p_m, const std::shared_ptr<Graph>& g) {
  GraphProto* p_g = p_m->mutable_graph();
  encodeGraph(p_g, g);
  // Add new opset_versions
  p_m->clear_opset_import();
  for (const OpSetID& opset : g->opset_versions_mutable()) {
    OperatorSetIdProto* opset_version_output = p_m->add_opset_import();
    opset_version_output->set_domain(opset.domain());
    opset_version_output->set_version(opset.version());
  }
}

ModelProto PrepareOutput(const ModelProto& mp_in) {
  ModelProto mp_out{};

  if (mp_in.has_ir_version()) {
    mp_out.set_ir_version(mp_in.ir_version());
  }
  if (mp_in.has_producer_name()) {
    mp_out.set_producer_name(mp_in.producer_name());
  }
  if (mp_in.has_producer_version()) {
    mp_out.set_producer_version(mp_in.producer_version());
  }
  if (mp_in.has_domain()) {
    mp_out.set_domain(mp_in.domain());
  }
  if (mp_in.has_model_version()) {
    mp_out.set_model_version(mp_in.model_version());
  }
  if (mp_in.has_doc_string()) {
    mp_out.set_doc_string(mp_in.doc_string());
  }
  for (int i = 0; i < mp_in.opset_import_size(); i++) {
    auto& oi_in = mp_in.opset_import(i);
    auto* oi_out = mp_out.add_opset_import();
    if (oi_in.has_domain()) {
      oi_out->set_domain(oi_in.domain());
    }
    if (oi_in.has_version()) {
      oi_out->set_version(oi_in.version());
    }
  }
  for (int i = 0; i < mp_in.metadata_props_size(); i++) {
    auto& pp_in = mp_in.metadata_props(i);
    auto* pp_out = mp_out.add_metadata_props();
    if (pp_in.has_key()) {
      pp_out->set_key(pp_in.key());
    }
    if (pp_in.has_value()) {
      pp_out->set_value(pp_in.value());
    }
  }

  return mp_out;
}

void assertNonNull(const std::shared_ptr<Graph>& g) {
  ONNX_ASSERTM(
      g.get() != nullptr,
      "Warning: onnx version converter is unable to parse input model. "
      "(The IR version of the ONNX model may be too old.)");
}

} // namespace ONNX_NAMESPACE
