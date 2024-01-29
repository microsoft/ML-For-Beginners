// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iostream>

#include "gtest/gtest.h"
#include "onnx/defs/parser.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/shape_inference.h"
#include "onnx/onnx_pb.h"
#include "onnx/shape_inference/implementation.h"

using namespace ONNX_NAMESPACE::shape_inference;

namespace ONNX_NAMESPACE {
// onnx/defs/controlflow/old.cc
void ScanInferenceFunctionOpset8(InferenceContext& ctx);
// onnx/defs/controlflow/defs.cc
void ScanInferenceFunction(InferenceContext& ctx);

namespace Test {

template <class Type>
void CreateDims(Type& proto, int num_dims) {
  auto mutable_shape = proto.mutable_shape();
  mutable_shape->clear_dim();

  for (int i = 0; i < num_dims; ++i)
    mutable_shape->add_dim();
}

template <class Type>
void SetDimValues(Type& proto, const std::vector<int>& values) {
  auto* mutable_shape = proto.mutable_shape();
  EXPECT_TRUE(mutable_shape->dim_size() == values.size());

  int idx = 0;
  for (auto value : values) {
    auto mutable_dim = mutable_shape->mutable_dim(idx++);
    if (value != -1)
      mutable_dim->set_dim_value(value);
  }
}

template <class Type>
void SetDimParams(Type& proto, const std::vector<const std::string*>& values) {
  auto mutable_shape = proto.mutable_shape();
  EXPECT_TRUE(mutable_shape->dim_size() == values.size());

  int idx = 0;
  for (auto value : values) {
    auto mutable_dim = mutable_shape->mutable_dim(idx++);
    if (value)
      mutable_dim->set_dim_param(*value);
  }
}

template <class Type>
void Dump(const Type& t) {
  auto& s_shape = t.shape();
  auto num_dims = s_shape.dim_size();
  std::cout << num_dims << " dims. ";
  for (int i = 0; i < num_dims; ++i) {
    auto x = s_shape.dim(0);
    auto y = x.has_dim_value();
    auto z = x.has_dim_param();

    std::cout << "Dim " << i << " Value:" << (y ? ONNX_NAMESPACE::to_string(x.dim_value()) : "<unset>")
              << ", Param:" << (z ? x.dim_param() : "<unset>") << "\n";
  }
};

TEST(ShapeInferenceTest, mergeShapeInfo_HasShape) {
  // source has shape, target doesn't
  {
    TypeProto_Tensor source;
    TypeProto_Tensor target;

    CreateDims(source, 1);
    SetDimValues(source, {1});
    mergeInShapeInfo(source, target);

    Dump(target);
    auto& shape = target.shape();
    EXPECT_TRUE(shape.dim_size() == 1 && shape.dim(0).dim_value() == 1);
  }

  // source has no shape, target does
  {
    TypeProto_Tensor source;
    TypeProto_Tensor target;

    CreateDims(target, 1);
    SetDimValues(target, {1});
    mergeInShapeInfo(source, target);

    Dump(target);
    auto& shape = target.shape();
    EXPECT_TRUE(shape.dim_size() == 1 && shape.dim(0).dim_value() == 1);
  }
  // source has shape, target doesn't
  {
    TypeProto_SparseTensor source;
    TypeProto_SparseTensor target;

    CreateDims(source, 1);
    SetDimValues(source, {1});
    mergeInShapeInfo(source, target);

    Dump(target);
    auto& shape = target.shape();
    EXPECT_TRUE(shape.dim_size() == 1 && shape.dim(0).dim_value() == 1);
  }

  // source has no shape, target does
  {
    TypeProto_SparseTensor source;
    TypeProto_SparseTensor target;

    CreateDims(target, 1);
    SetDimValues(target, {1});
    mergeInShapeInfo(source, target);

    Dump(target);
    auto& shape = target.shape();
    EXPECT_TRUE(shape.dim_size() == 1 && shape.dim(0).dim_value() == 1);
  }
}
TEST(ShapeInferenceTest, mergeShapeInfo_PreferValueOverParam) {
  std::string param = "A";

  // source has value, target has param. prefer value
  {
    TypeProto_Tensor source;
    TypeProto_Tensor target;

    CreateDims(source, 1);
    SetDimValues(source, {1});

    CreateDims(target, 1);
    SetDimParams(target, {&param});

    mergeInShapeInfo(source, target);

    Dump(target);
    auto& shape = target.shape();
    EXPECT_TRUE(shape.dim_size() == 1 && shape.dim(0).dim_value() == 1);
  }

  // source has param, target has value.
  {
    TypeProto_Tensor source;
    TypeProto_Tensor target;

    CreateDims(source, 1);
    SetDimParams(source, {&param});

    CreateDims(target, 1);
    SetDimValues(target, {1});

    mergeInShapeInfo(source, target);

    Dump(target);
    auto& shape = target.shape();
    EXPECT_TRUE(shape.dim_size() == 1 && shape.dim(0).dim_value() == 1);
  }
}

TEST(ShapeInferenceTest, mergeShapeInfo_CombineShapes) {
  // merge from both sides, preferring real value over -1
  {
    TypeProto_Tensor source;
    TypeProto_Tensor target;

    CreateDims(source, 2);
    SetDimValues(source, {-1, 2});

    CreateDims(target, 2);
    SetDimValues(target, {1, -1});

    mergeInShapeInfo(source, target);

    Dump(target);
    auto& shape = target.shape();
    EXPECT_TRUE(shape.dim(0).dim_value() == 1 && shape.dim(1).dim_value() == 2);
  }

  {
    TypeProto_SparseTensor source;
    TypeProto_SparseTensor target;

    CreateDims(source, 2);
    SetDimValues(source, {-1, 2});

    CreateDims(target, 2);
    SetDimValues(target, {1, -1});

    mergeInShapeInfo(source, target);

    Dump(target);
    auto& shape = target.shape();
    EXPECT_TRUE(shape.dim(0).dim_value() == 1 && shape.dim(1).dim_value() == 2);
  }

  // prefer value over param,
  {
    TypeProto_Tensor source;
    TypeProto_Tensor target;

    CreateDims(source, 2);
    SetDimValues(source, {-1, 2});

    CreateDims(target, 2);
    SetDimValues(target, {1, 0});
    // replace second dim with a param. the value from the source should be
    // preferred
    const std::string param = "A";
    target.mutable_shape()->mutable_dim(1)->set_dim_param(param);

    mergeInShapeInfo(source, target);

    Dump(target);
    auto& shape = target.shape();
    EXPECT_TRUE(shape.dim(0).dim_value() == 1 && shape.dim(1).dim_value() == 2);
  }
  {
    TypeProto_SparseTensor source;
    TypeProto_SparseTensor target;

    CreateDims(source, 2);
    SetDimValues(source, {-1, 2});

    CreateDims(target, 2);
    SetDimValues(target, {1, 0});
    // replace second dim with a param. the value from the source should be
    // preferred
    const std::string param = "A";
    target.mutable_shape()->mutable_dim(1)->set_dim_param(param);

    mergeInShapeInfo(source, target);

    Dump(target);
    auto& shape = target.shape();
    EXPECT_TRUE(shape.dim(0).dim_value() == 1 && shape.dim(1).dim_value() == 2);
  }
}

TEST(ShapeInferenceTest, mergeShapeInfo_Mismatches) {
#ifndef ONNX_NO_EXCEPTIONS
  // mismatched num dims
  {
    TypeProto_Tensor source;
    TypeProto_Tensor target;

    CreateDims(source, 2);
    SetDimValues(source, {-1, 2});

    CreateDims(target, 3);
    SetDimValues(target, {1, -1, 1});

    EXPECT_THROW(mergeInShapeInfo(source, target), ONNX_NAMESPACE::InferenceError);
  }

  {
    TypeProto_SparseTensor source;
    TypeProto_SparseTensor target;

    CreateDims(source, 2);
    SetDimValues(source, {-1, 2});

    CreateDims(target, 3);
    SetDimValues(target, {1, -1, 1});

    EXPECT_THROW(mergeInShapeInfo(source, target), ONNX_NAMESPACE::InferenceError);
  }

  // mismatched dim values
  {
    TypeProto_Tensor source;
    TypeProto_Tensor target;

    CreateDims(source, 2);
    SetDimValues(source, {2, 2});

    CreateDims(target, 2);
    SetDimValues(target, {2, 1});

    EXPECT_THROW(mergeInShapeInfo(source, target), ONNX_NAMESPACE::InferenceError);
  }

  {
    TypeProto_SparseTensor source;
    TypeProto_SparseTensor target;

    CreateDims(source, 2);
    SetDimValues(source, {2, 2});

    CreateDims(target, 2);
    SetDimValues(target, {2, 1});

    EXPECT_THROW(mergeInShapeInfo(source, target), ONNX_NAMESPACE::InferenceError);
  }
#endif
  // mismatched param value. prefer target
  {
    TypeProto_Tensor source;
    TypeProto_Tensor target;
    const std::string param_a = "A";
    const std::string param_b = "B";

    CreateDims(source, 1);
    SetDimParams(source, {&param_a});

    CreateDims(target, 1);
    SetDimParams(target, {&param_b});

    mergeInShapeInfo(source, target);

    auto& shape = target.shape();
    EXPECT_TRUE(shape.dim(0).dim_param() == "B");
  }
  {
    TypeProto_SparseTensor source;
    TypeProto_SparseTensor target;
    const std::string param_a = "A";
    const std::string param_b = "B";

    CreateDims(source, 1);
    SetDimParams(source, {&param_a});

    CreateDims(target, 1);
    SetDimParams(target, {&param_b});

    mergeInShapeInfo(source, target);

    auto& shape = target.shape();
    EXPECT_TRUE(shape.dim(0).dim_param() == "B");
  }
}

// Check subgraph inferencing via GraphInferencer using a Scan
static void doInferencingTest(bool use_scan_opset8) {
  auto* schemaRegistry = OpSchemaRegistry::Instance();
  GraphProto subgraph;

  // simple tensor without shape info
  TypeProto simple_tensor_no_shape;
  auto* tensor_type = simple_tensor_no_shape.mutable_tensor_type();
  tensor_type->set_elem_type(TensorProto_DataType_FLOAT);

  // simple tensor with shape info
  TypeProto simple_tensor = simple_tensor_no_shape;
  simple_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(2);

  // setup simple graph that can be used with Scan containing two Identity
  // nodes. one for the loop state variable. one for the scan output.
  {
    NodeProto loop_state_identity;
    loop_state_identity.set_name("loop_state_identity");
    loop_state_identity.set_domain(ONNX_DOMAIN);
    loop_state_identity.set_op_type("Identity");
    loop_state_identity.set_doc_string("loop state identity");
    loop_state_identity.add_input("loop_state_in");
    loop_state_identity.add_output("loop_state_out");

    *subgraph.add_node() = loop_state_identity;

    NodeProto scan_in_out_identity;
    scan_in_out_identity.set_name("scan_in_out_identity");
    scan_in_out_identity.set_domain(ONNX_DOMAIN);
    scan_in_out_identity.set_op_type("Identity");
    scan_in_out_identity.set_doc_string("scan identity");
    scan_in_out_identity.add_input("scan_in");
    scan_in_out_identity.add_output("scan_out");
    *subgraph.add_node() = scan_in_out_identity;

    ValueInfoProto loop_state_in;
    loop_state_in.set_name("loop_state_in");
    *loop_state_in.mutable_type() = simple_tensor;
    *subgraph.add_input() = loop_state_in;

    ValueInfoProto scan_in;
    scan_in.set_name("scan_in");
    *scan_in.mutable_type() = simple_tensor;
    *subgraph.add_input() = scan_in;

    ValueInfoProto loop_state_out = loop_state_in;
    loop_state_out.set_name("loop_state_out");
    *loop_state_out.mutable_type() = simple_tensor_no_shape;
    *subgraph.add_output() = loop_state_out;

    ValueInfoProto scan_state_out = scan_in;
    scan_state_out.set_name("scan_out");
    *scan_state_out.mutable_type() = simple_tensor_no_shape;
    *subgraph.add_output() = scan_state_out;
  }

  std::unordered_map<std::string, int> opset_imports;
  opset_imports[ONNX_DOMAIN] = 8; // Scan is v8

  const std::unordered_map<std::string, TypeProto*> outer_scope_value_types;
  SymbolTableImpl symbolTable;
  symbolTable.addFromGraph(subgraph);
  GraphInferenceContext graphInfCtx(outer_scope_value_types, opset_imports, &symbolTable);
  GraphInferencerImpl graphInferencer(subgraph, graphInfCtx);

  // loop_state_in and scan_in are the two inputs.
  // order in subgraphInputTypes matches their order as graph inputs.
  std::vector<const TypeProto*> subgraphInputTypes = {&simple_tensor, &simple_tensor};

  std::vector<const TensorProto*> subgraphInputData = {};
  ShapeInferenceOptions options{false, 0, false};
  auto output = graphInferencer.doInferencing(subgraphInputTypes, subgraphInputData);

  // check the subgraph outputs had their shape inferred when we called
  // doInferencing directly
  EXPECT_TRUE(output.size() == 2);

  auto checkType = [](const TypeProto& type, const TypeProto_Tensor& expect) {
    auto checkDims = [](const TensorShapeProto& l, const TensorShapeProto& r) {
      EXPECT_TRUE(l.dim_size() == r.dim_size());

      for (int i = 0, end = l.dim_size(); i < end; ++i) {
        // if (l.dim().Get(i).dim_value() != r.dim().Get(i).dim_value())
        //  break;
        EXPECT_TRUE(l.dim().Get(i).dim_value() == r.dim().Get(i).dim_value());
      }
    };

    EXPECT_TRUE(type.has_tensor_type());
    EXPECT_TRUE(type.tensor_type().elem_type() == expect.elem_type());
    checkDims(type.tensor_type().shape(), expect.shape());
  };

  checkType(*output[0], simple_tensor.tensor_type());
  checkType(*output[1], simple_tensor.tensor_type());

  // setup Scan node to test subgraph inferencing works as expected when called
  // from the operators type/shape inferencing function
  NodeProto scan;
  {
    AttributeProto num_scan_inputs;
    num_scan_inputs.set_name("num_scan_inputs");
    num_scan_inputs.set_i(1);

    AttributeProto body;
    body.set_name("body");
    *body.mutable_g() = subgraph;

    *scan.add_attribute() = num_scan_inputs;
    *scan.add_attribute() = body;

    scan.set_name("Scan");
    scan.set_domain(ONNX_DOMAIN);
    scan.set_doc_string("Scan node");
    scan.set_op_type("Scan");
    if (use_scan_opset8)
      scan.add_input(""); // optional sequence lens
    scan.add_input("loop_state_start");
    scan.add_input("scan_op_in");
    scan.add_output("loop_state_final");
    scan.add_output("scan_op_out");
  }

  TypeProto loop_state_in_tensor = simple_tensor_no_shape;
  auto* shape = loop_state_in_tensor.mutable_tensor_type()->mutable_shape();
  if (use_scan_opset8)
    shape->add_dim()->set_dim_value(1); // batch size
  shape->add_dim()->set_dim_value(2); // input size. must match subgraph

  TypeProto loop_state_out_tensor = loop_state_in_tensor; // should be unchanged

  TypeProto scan_in_tensor = simple_tensor_no_shape;
  shape = scan_in_tensor.mutable_tensor_type()->mutable_shape();
  if (use_scan_opset8)
    shape->add_dim()->set_dim_value(1); // batch size
  shape->add_dim()->set_dim_value(1); // sequence length
  shape->add_dim()->set_dim_value(2); // input size. must match subgraph

  TypeProto scan_out_tensor = scan_in_tensor; // should be unchanged

  std::unordered_map<std::string, TypeProto*> valueTypesByName;
  valueTypesByName["loop_state_start"] = &loop_state_in_tensor;
  valueTypesByName["scan_op_in"] = &scan_in_tensor;

  InferenceContextImpl ctx(scan, valueTypesByName, {}, {}, options, {}, &graphInfCtx);
  if (use_scan_opset8)
    ScanInferenceFunctionOpset8(ctx);
  else
    ScanInferenceFunction(ctx);

  EXPECT_TRUE(ctx.getNumOutputs() == 2);
  checkType(*ctx.getOutputType(0), loop_state_out_tensor.tensor_type());
  checkType(*ctx.getOutputType(1), scan_out_tensor.tensor_type());
}

// Check subgraph inferencing via GraphInferencer using a Scan (from opset 8)
TEST(GraphInferencerImplTest, Scan8_BasicTest) {
  doInferencingTest(true);
}

// Check subgraph inferencing via GraphInferencer using a Scan (from opset 9)
TEST(GraphInferencerImplTest, Scan9_BasicTest) {
  doInferencingTest(false);
}

void RunReshapeShapeInfTest(const char* modelStr, TensorShapeProto& expectedShape) {
  ModelProto model;
  OnnxParser parser(modelStr);
  auto status = parser.Parse(model);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  EXPECT_TRUE(parser.EndOfInput()) << "Extra unparsed input unexpected.";

  ShapeInferenceOptions options{true, 1, true};
  ONNX_NAMESPACE::shape_inference::InferShapes(model, ONNX_NAMESPACE::OpSchemaRegistry::Instance(), options);

  const auto inferredShape = model.graph().output(0).type().tensor_type().shape();
  EXPECT_TRUE(inferredShape.dim_size() == expectedShape.dim_size());

  for (int i = 0; i < inferredShape.dim_size(); i++) {
    EXPECT_TRUE(
        (inferredShape.dim(i).has_dim_value() && expectedShape.dim(i).has_dim_value()) ||
        (inferredShape.dim(i).has_dim_param() && expectedShape.dim(i).has_dim_param()));

    EXPECT_TRUE(
        inferredShape.dim(i).has_dim_value() ? inferredShape.dim(i).dim_value() == expectedShape.dim(i).dim_value()
                                             : inferredShape.dim(i).dim_param() == expectedShape.dim(i).dim_param());
  }
}
TEST(ShapeInferenceTest, ReshapeTestWithShapeAsSymInput) {
  const char* modelStr = R"ONNX(
<
  ir_version: 8,
  opset_import: [ "" : 15],
  producer_name: "DataPropagationTest",
  producer_version: "1.0",
  model_version: 1,
  doc_string: "A test model for data propagation."
>
agraph (float[batch_size, 256, 768, 3] x, float[batch_size, 196608] m) => (float[?, ?, ?] z)
{
    y = Shape<start = 0, end = 3>(x)
    z = Reshape(m, y)
}
)ONNX";

  TensorShapeProto expectedShape;
  expectedShape.mutable_dim()->Add()->set_dim_param("batch_size");
  expectedShape.mutable_dim()->Add()->set_dim_value(256);
  expectedShape.mutable_dim()->Add()->set_dim_value(768);

  RunReshapeShapeInfTest(modelStr, expectedShape);
}

TEST(ShapeInferenceTest, ReshapeTestWithShapeAsInitializer) {
  const char* modelStr = R"ONNX(
<
  ir_version: 8,
  opset_import: [ "" : 15],
  producer_name: "DataPropagationTest",
  producer_version: "1.0",
  model_version: 1,
  doc_string: "A test model for data propagation."
>
agraph (float[1, 196608] m) => (float[?, ?, ?] z)
<int64[3] shape = {1, 768, 256}>
{
    z = Reshape(m, shape)
}
)ONNX";

  TensorShapeProto expectedShape;
  expectedShape.mutable_dim()->Add()->set_dim_value(1);
  expectedShape.mutable_dim()->Add()->set_dim_value(768);
  expectedShape.mutable_dim()->Add()->set_dim_value(256);

  RunReshapeShapeInfTest(modelStr, expectedShape);
}

TEST(ShapeInferenceTest, ReshapeTestWithShapeAsInitializer1) {
  const char* modelStr = R"ONNX(
<
  ir_version: 8,
  opset_import: [ "" : 15],
  producer_name: "DataPropagationTest",
  producer_version: "1.0",
  model_version: 1,
  doc_string: "A test model for data propagation."
>
agraph (float[1, 196608] m) => (float[?, ?, ?] z)
<int64[3] shape = {1, -1, 256}>
{
    z = Reshape(m, shape)
}
)ONNX";

  TensorShapeProto expectedShape;
  expectedShape.mutable_dim()->Add()->set_dim_value(1);
  expectedShape.mutable_dim()->Add()->set_dim_value(768);
  expectedShape.mutable_dim()->Add()->set_dim_value(256);

  RunReshapeShapeInfTest(modelStr, expectedShape);
}

TEST(ShapeInferenceTest, CheckShapesAndTypesTest) {
#ifndef ONNX_NO_EXCEPTIONS
  // Tensor element types mis-match should cause an exception.
  TypeProto tensor_infer;
  auto* tensor_infer_type = tensor_infer.mutable_tensor_type();
  tensor_infer_type->set_elem_type(TensorProto_DataType_FLOAT);

  TypeProto tensor_exist;
  auto* tensor_exist_type = tensor_exist.mutable_tensor_type();
  tensor_exist_type->set_elem_type(TensorProto_DataType_UINT8);

  EXPECT_THROW(checkShapesAndTypes(tensor_infer, tensor_exist), ONNX_NAMESPACE::InferenceError);
#endif
}

} // namespace Test
} // namespace ONNX_NAMESPACE
