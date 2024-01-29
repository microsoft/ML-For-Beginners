// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gtest/gtest.h"
#include "onnx/checker.h"
#include "onnx/defs/parser.h"
#include "onnx/defs/printer.h"

using namespace ONNX_NAMESPACE;

namespace ONNX_NAMESPACE {
namespace Test {

template <typename T>
static void Parse(T& parsedData, const char* input) {
  OnnxParser parser(input);
  auto status = parser.Parse(parsedData);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  EXPECT_TRUE(parser.EndOfInput()) << "Extra unparsed input unexpected.";
  // Extra checks for printer:
  // Check we can convert data back to text form.
  std::string text1 = ProtoToString(parsedData);
  // Check that we can round-trip between the two representations.
  // We cannot expect equality between text1 and input due to white-space and syntactic sugar,
  // so, we convert it once more, and check for equality.
  T temp;
  status = OnnxParser::Parse(temp, text1.c_str());
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  std::string text2 = ProtoToString(temp);
  EXPECT_EQ(text1, text2);
}

template <typename T>
static void ExpectParseFailure(T& parsedData, const char* input) {
  auto status = OnnxParser::Parse(parsedData, input);
  EXPECT_FALSE(status.IsOK());
}

static void CheckModel(const char* code) {
  ModelProto model;
  Parse(model, code);

  checker::check_model(model);
}

TEST(ParserTest, EscapeStringLiteral) {
  OnnxParser parser(R"(
    "123\"56\\89"
  )");

  std::string s;
  auto status = parser.ParserBase::Parse(s);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  EXPECT_TRUE(parser.EndOfInput()) << "Extra unparsed input unexpected.";
  EXPECT_EQ(s, std::string("123\"56\\89"));
}

TEST(ParserTest, TypeTest) {
  TypeProto type;

  // 1-dimensional tensor type with symbolic dimension:
  Parse(type, "float[N]");
  EXPECT_TRUE(type.has_tensor_type());
  int float_type = static_cast<int>(TensorProto_DataType::TensorProto_DataType_FLOAT);
  int int32_type = static_cast<int>(TensorProto_DataType::TensorProto_DataType_INT32);
  EXPECT_EQ(type.tensor_type().elem_type(), float_type);
  EXPECT_TRUE(type.tensor_type().has_shape());
  EXPECT_EQ(type.tensor_type().shape().dim_size(), 1);
  EXPECT_EQ(type.tensor_type().shape().dim(0).dim_param(), "N");

  // scalar type:
  Parse(type, "float");
  EXPECT_TRUE(type.has_tensor_type());
  EXPECT_EQ(type.tensor_type().elem_type(), float_type);
  EXPECT_TRUE(type.tensor_type().has_shape());
  EXPECT_EQ(type.tensor_type().shape().dim_size(), 0);

  // tensor type with unknown rank:
  Parse(type, "float[]");
  EXPECT_TRUE(type.has_tensor_type());
  EXPECT_EQ(type.tensor_type().elem_type(), float_type);
  EXPECT_FALSE(type.tensor_type().has_shape());

  // 3-dimensional tensor
  Parse(type, "float[N,M,K]");
  EXPECT_EQ(type.tensor_type().shape().dim_size(), 3);

  // Unspecified dimension (neither symbolic nor constant)
  Parse(type, "float[N,?,K]");
  EXPECT_FALSE(type.tensor_type().shape().dim(1).has_dim_param());
  EXPECT_FALSE(type.tensor_type().shape().dim(1).has_dim_value());

  // sequence type:
  Parse(type, "seq(float[])");
  EXPECT_TRUE(type.has_sequence_type());
  auto& elttype = type.sequence_type().elem_type();
  EXPECT_TRUE(elttype.has_tensor_type());
  EXPECT_EQ(elttype.tensor_type().elem_type(), float_type);
  EXPECT_FALSE(elttype.tensor_type().has_shape());

  // optional type:
  Parse(type, "optional(float)");
  EXPECT_TRUE(type.has_optional_type());
  auto& optelttype = type.optional_type().elem_type();
  EXPECT_TRUE(optelttype.has_tensor_type());
  EXPECT_EQ(optelttype.tensor_type().elem_type(), float_type);
  EXPECT_TRUE(optelttype.tensor_type().has_shape());

  // optional type:
  Parse(type, "sparse_tensor(float[1000])");
  EXPECT_TRUE(type.has_sparse_tensor_type());
  EXPECT_EQ(type.sparse_tensor_type().elem_type(), float_type);
  EXPECT_EQ(type.sparse_tensor_type().shape().dim_size(), 1);

  // map type:
  Parse(type, "map(int32, float[N])");
  EXPECT_TRUE(type.has_map_type());
  EXPECT_EQ(type.map_type().key_type(), int32_type);
  auto& valtype = type.map_type().value_type();
  EXPECT_TRUE(valtype.has_tensor_type());
  EXPECT_EQ(valtype.tensor_type().elem_type(), float_type);
  EXPECT_EQ(valtype.tensor_type().shape().dim_size(), 1);
}

TEST(ParserTest, TensorProtoTest) {
  TensorProto tensorProto;

  // Concrete tensor-type with numeric dimensions expected:
  ExpectParseFailure(tensorProto, "int32[] {1, 2, 3, 4, 5}");

  // Symbolic dimensions are not allowed.
  ExpectParseFailure(tensorProto, "int32[N] {1, 2, 3, 4, 5}");

  Parse(tensorProto, "int32[5] {1, 2, 3, 4, 5}");

  Parse(tensorProto, "int32[5] T {1, 2, 3, 4, 5}");
  EXPECT_EQ(tensorProto.name(), "T");

  Parse(tensorProto, "float[5] {1, 2.0, 3.1, 4, 5.5}");

  Parse(tensorProto, "float[5] {1e1, 2.0e-1, 3.1E-1, 4E+1, 5.5e-10}");

  Parse(tensorProto, "string[2] { \"Hello\", \"World\" }");

  // String literals with escape character
  Parse(tensorProto, R"(
    string[2] { "Use a \"quoted\" word", "Use a backslash \\ like this." }
  )");
}

TEST(ParserTest, AttributeTest) {
  AttributeProto attr;

  Parse(attr, "x = 2");
  EXPECT_EQ(attr.name(), "x");
  EXPECT_EQ(attr.type(), AttributeProto_AttributeType::AttributeProto_AttributeType_INT);
  EXPECT_EQ(attr.i(), 2);

  Parse(attr, "x = 0.625");
  EXPECT_EQ(attr.type(), AttributeProto_AttributeType::AttributeProto_AttributeType_FLOAT);
  EXPECT_FLOAT_EQ(attr.f(), 0.625);

  Parse(attr, "x = [2, 4, 6]");
  EXPECT_EQ(attr.type(), AttributeProto_AttributeType::AttributeProto_AttributeType_INTS);
  EXPECT_EQ(attr.ints_size(), 3);

  Parse(attr, "x = [0.125, 0.625]");
  EXPECT_EQ(attr.type(), AttributeProto_AttributeType::AttributeProto_AttributeType_FLOATS);
  EXPECT_EQ(attr.floats_size(), 2);

  Parse(attr, "x = float[3] {2.1, 4.1, 6.1}");
  EXPECT_EQ(attr.type(), AttributeProto_AttributeType::AttributeProto_AttributeType_TENSOR);

  Parse(attr, "x = \"astring\"");
  EXPECT_EQ(attr.name(), "x");
  EXPECT_EQ(attr.type(), AttributeProto_AttributeType::AttributeProto_AttributeType_STRING);
  EXPECT_EQ(attr.s(), "astring");

  Parse(attr, "x = [\"abc\", \"def\"]");
  EXPECT_EQ(attr.type(), AttributeProto_AttributeType::AttributeProto_AttributeType_STRINGS);

  Parse(attr, "x : ints = @xyz");
  EXPECT_EQ(attr.ref_attr_name(), "xyz");
  EXPECT_EQ(attr.type(), AttributeProto_AttributeType::AttributeProto_AttributeType_INTS);

  Parse(attr, "x : ints = []");
  EXPECT_EQ(attr.type(), AttributeProto_AttributeType::AttributeProto_AttributeType_INTS);
  EXPECT_EQ(attr.ints_size(), 0);

  Parse(attr, R"ONNX(
    body = somegraph (float[N] y, float[N] z) => (float[N] w)
      {
        x = foo(y, z)
        w = bar(x, y)
      }
)ONNX");
  EXPECT_EQ(attr.type(), AttributeProto_AttributeType::AttributeProto_AttributeType_GRAPH);
  EXPECT_EQ(attr.g().node_size(), 2);

  Parse(attr, "type = float[3]");
  EXPECT_EQ(attr.type(), AttributeProto_AttributeType::AttributeProto_AttributeType_TYPE_PROTO);
  EXPECT_TRUE(attr.tp().has_tensor_type());
  int float_type = static_cast<int>(TensorProto_DataType::TensorProto_DataType_FLOAT);
  EXPECT_EQ(attr.tp().tensor_type().elem_type(), float_type);
}

TEST(ParserTest, AttrListTest) {
  const char* code = R"ONNX(
<
    x = 2,
    w = 3
>
)ONNX";

  AttrList attributes;
  Parse(attributes, code);
  EXPECT_EQ(attributes.size(), 2);
  EXPECT_EQ(attributes.Get(0).name(), "x");
  EXPECT_EQ(attributes.Get(1).name(), "w");
}

TEST(ParserTest, DomainOpCallTest) {
  const char* code = "x = somedomain.foo(y, z)";
  NodeProto n;
  Parse(n, code);
}

TEST(ParserTest, NodeTest) {
  const char* code = "x = foo(y, z)";
  NodeProto n;
  Parse(n, code);

  EXPECT_EQ(n.input_size(), 2);
  EXPECT_EQ(n.input(0), "y");
  EXPECT_EQ(n.input(1), "z");
  EXPECT_EQ(n.output_size(), 1);
  EXPECT_EQ(n.output(0), "x");
  EXPECT_EQ(n.op_type(), "foo");

  NodeList nl;
  Parse(nl, R"ONNX(
      {
       sub_result = Sub(limit, start)
       sub_result_casted = Cast<to = 1>(sub_result)
       delta_casted = Cast<to = 1>(delta)
       div_result = Div(sub_result_casted, delta_casted)
       ceil_result = Ceil(div_result)
       ceil_result_relu = Relu(ceil_result)
       ceil_result_relu_int = Cast<to = 7>(ceil_result_relu)
       ceil_result_relu_bool = Cast<to = 9>(ceil_result_relu)
       variadic_output, output = Loop (ceil_result_relu_int, ceil_result_relu_bool, start)
       }
       )ONNX");
}

TEST(ParserTest, QualifiedOpNameTest) {
  const char* code = "x = com.example.foo(y, z)";
  NodeProto n;
  Parse(n, code);

  EXPECT_EQ(n.domain(), "com.example");
  EXPECT_EQ(n.op_type(), "foo");
}

TEST(ParserTest, NodeListTest) {
  const char* code = R"ONNX(
{
    x = foo(y, z)
    w = bar(x, y)
}
)ONNX";

  GraphProto graph;
  Parse(*graph.mutable_node(), code);

  EXPECT_EQ(graph.node_size(), 2);
  EXPECT_EQ(graph.node(0).op_type(), "foo");
  EXPECT_EQ(graph.node(1).op_type(), "bar");
}

TEST(ParserTest, NodeAttrTest1) {
  const char* code = "x = foo <a = 100, b = 200.5, c = \"astring\"> (y, z)";
  NodeProto n;
  Parse(n, code);

  EXPECT_EQ(n.attribute_size(), 3);
  EXPECT_EQ(n.attribute(0).name(), "a");
  EXPECT_EQ(n.attribute(1).name(), "b");
  EXPECT_EQ(n.attribute(2).name(), "c");
}

TEST(ParserTest, NodeAttrTest2) {
  const char* code = "x = foo <d = [5, 10], e = [0.55, 0.66], f = [\"str1\", \"str2\"]> (y, z)";
  NodeProto n;
  Parse(n, code);
  EXPECT_EQ(n.attribute_size(), 3);
}

TEST(ParserTest, GraphTest) {
  const char* code = R"ONNX(
agraph (float[N] y, float[N] z) => (float[N] w)
<float[2] w1 = {1.0, 2.0}, float[3] w2 = {4.0, 5.0, 6.0}, float[N] x>
{
    # This is a comment.
    x = foo(y, z, w1) # More comments.
    w = bar(x, y, w2)
}
)ONNX";

  GraphProto graph;
  Parse(graph, code);

  EXPECT_EQ(graph.name(), "agraph");
  EXPECT_EQ(graph.input_size(), 2);
  EXPECT_EQ(graph.output_size(), 1);
  EXPECT_EQ(graph.node_size(), 2);
  EXPECT_EQ(graph.initializer_size(), 2);
  EXPECT_EQ(graph.value_info_size(), 1);
}

TEST(ParserTest, GraphPartialTypeTest) {
  const char* code = R"ONNX(
agraph (float[N] y, z) => (float[N] w)
{
    x = foo(y, z)
    w = bar(x, y)
}
)ONNX";

  GraphProto graph;
  Parse(graph, code);

  EXPECT_EQ(graph.name(), "agraph");
  EXPECT_EQ(graph.input_size(), 2);
  EXPECT_EQ(graph.output_size(), 1);
}

TEST(ParserTest, FunctionTest) {
  const char* code = R"ONNX(
<
  opset_import: [ "" : 10 ],
  domain: "ai.onnx.ml",
  doc_string: "A function test case."
>
f (y, z) => (w)
{
    x = Add(y, z)
    w = Mul(x, y)
}
)ONNX";

  FunctionProto fp;
  Parse(fp, code);

  EXPECT_EQ(fp.name(), "f");
  EXPECT_EQ(fp.input_size(), 2);
  EXPECT_EQ(fp.output_size(), 1);
  EXPECT_EQ(fp.node_size(), 2);
  EXPECT_EQ(fp.attribute_size(), 0);
  EXPECT_EQ(fp.opset_import_size(), 1);
}

TEST(ParserTest, InitializerTest) {
  const char* code = R"ONNX(
agraph (float y = {1.0}, float[N] z) => (float[N] w)
<float[2] w1 = {1.0, 2.0}, float[3] w2 = {4.0, 5.0, 6.0}, float[N] x>
{
    x = foo(y, z, w1)
    w = bar(x, y, w2)
}
)ONNX";

  GraphProto graph;
  Parse(graph, code);

  EXPECT_EQ(graph.input_size(), 2);
  EXPECT_EQ(graph.output_size(), 1);
  EXPECT_EQ(graph.initializer_size(), 3); // y, w1, w2
  EXPECT_EQ(graph.value_info_size(), 1); // x
}

TEST(ParserTest, IfNodeTest) {
  const char* code = R"ONNX(
z = If (b) <
    then_branch = g1 () => (float[N] z_then)
      {
        z_then = foo(y)
      },
    else_branch = g2 () => (float[N] z_else)
      {
        z_else = bar(x)
      }
    >
)ONNX";

  NodeProto node;
  Parse(node, code);
  EXPECT_EQ(node.input_size(), 1);
  EXPECT_EQ(node.output_size(), 1);
  EXPECT_EQ(node.attribute_size(), 2);
}

TEST(ParserTest, ModelTest) {
  const char* code = R"ONNX(
<
  ir_version: 7,
  opset_import: [ "ai.onnx.ml" : 10 ],
  producer_name: "ParserTest",
  producer_version: "1.0",
  domain: "ai.onnx.ml",
  model_version: 1,
  doc_string: "A parser test case model.",
  metadata_props: [ "somekey" : "somevalue", "key2" : "value2" ]
>
agraph (float[N] y, float[N] z) => (float[N] w)
{
    x = foo(y, z)
    w = bar(x, y)
}
)ONNX";

  ModelProto model;
  Parse(model, code);

  EXPECT_EQ(model.graph().input_size(), 2);
  EXPECT_EQ(model.graph().output_size(), 1);
  EXPECT_EQ(model.graph().node_size(), 2);
}

TEST(ParserTest, ModelCheckTest) {
  const char* code = R"ONNX(
<
  ir_version: 7,
  opset_import: [ "" : 10 ]
>
agraph (float[N, 128] X, float[128,10] W, float[10] B) => (float[N] C)
{
    T = MatMul(X, W)
    S = Add(T, B)
    C = Softmax(S)
}
)ONNX";

  CheckModel(code);
}

TEST(ParserTest, IfModelTest) {
  const char* code = R"ONNX(
<
  ir_version: 7,
  opset_import: [ "" : 13 ]
>
iftest (bool b, float[128] X, float[128] Y) => (float[128] Z)
{
  Z = If (b) <
      then_branch = g1 () => (float[128] z_then) { z_then = Identity(X) },
      else_branch = g2 () => (float[128] z_else) { z_else = Identity(Y) }
      >
}
)ONNX";

  CheckModel(code);
}

TEST(ParserTest, FunModelTest) {
  const char* code = R"ONNX(
<
  ir_version: 8,
  opset_import: [ "" : 10, "local" : 1 ]
>
agraph (float[N, 128] X, float[128,10] W, float[10] B) => (float[N] C)
{
  T = local.foo (X, W, B)
  C = local.square(T)
}

<
  opset_import: [ "" : 10 ],
  domain: "local",
  doc_string: "Function foo."
>
foo (x, w, b) => (c) {
  T = MatMul(x, w)
  S = Add(T, b)
  c = Softmax(S)
}

<
  opset_import: [ "" : 10 ],
  domain: "local",
  doc_string: "Function square."
>
square (x) => (y) {
  y = Mul (x, x)
}
)ONNX";

  CheckModel(code);

  const char* code_function_with_attributes = R"ONNX(
<
  ir_version: 9,
  opset_import: [ "" : 15, "custom_domain" : 1]
>
agraph (float[N] x) => (float[N] out)
{
  out = custom_domain.foo<alpha=2.0, gamma=3.0>(x)
}

<
domain: "custom_domain",
opset_import: [ "" : 15],
doc_string: "function foo"
>
  foo
  <alpha: float=4.0, gamma>
  (X) => (C)
  {
      constant_alpha = Constant<value_float: float=@alpha>()
      constant_gamma = Constant<value_float: float=@gamma>()
      constant_alpha_x = Mul(constant_alpha, X)
      C = Add(constant_alpha_x, constant_gamma)
  }
)ONNX";

  CheckModel(code_function_with_attributes);
}

TEST(ParserTest, TypesModelTest1) {
  const char* code = R"ONNX(
    <
    ir_version: 8,
    opset_import: [ "" : 18 ]
    >
    agraph (seq(float[N]) seqX) => (float[M, N] X)
    {
        X = ConcatFromSequence < axis = 0, new_axis = 1 >(seqX)
    }
)ONNX";
  CheckModel(code);
}

TEST(ParserTest, TypesModelTest2) {
  const char* code = R"ONNX(
    <
    ir_version: 8,
    opset_import: [ "" : 18 ]
    >
    agraph (float[N] tensorX, seq(float[N]) seqX, map(int32, float[N]) mapX, optional(float[N]) optionalX, sparse_tensor(float[N]) sparseX) => (float[N] X)
    {
        X = Identity (tensorX)
    }
)ONNX";
  CheckModel(code);
}

} // namespace Test
} // namespace ONNX_NAMESPACE
