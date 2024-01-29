// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iostream>

#include "gtest/gtest.h"
#include "onnx/checker.h"
#include "onnx/defs/parser.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/shape_inference.h"
#include "onnx/onnx_pb.h"
#include "onnx/shape_inference/implementation.h"

using namespace ONNX_NAMESPACE::shape_inference;

namespace ONNX_NAMESPACE {

namespace Test {

inline bool CompareShape(
    const TensorShapeProto& inferredShape,
    const TensorShapeProto& expectedShape,
    bool checkSameParam = false) {
  EXPECT_TRUE(inferredShape.dim_size() == expectedShape.dim_size())
      << "Dim size for inferred and expected shape is different.";

  for (int i = 0; i < inferredShape.dim_size(); i++) {
    EXPECT_TRUE(
        (inferredShape.dim(i).has_dim_value() == expectedShape.dim(i).has_dim_value()) &&
        (inferredShape.dim(i).has_dim_param() == expectedShape.dim(i).has_dim_param()))
        << "Inferred and expected dim values are different.";

    EXPECT_TRUE(
        inferredShape.dim(i).has_dim_value() ? inferredShape.dim(i).dim_value() == expectedShape.dim(i).dim_value()
            : checkSameParam                 ? inferredShape.dim(i).dim_param() == expectedShape.dim(i).dim_param()
                                             : true)
        << "Inferred and expected dims are different.";
  }

  return true;
}

TensorShapeProto RunDataPropagation(const char* graphCode, int domainVersion = 15) {
  // Parses the graph from graphCode
  GraphProto graph;
  OnnxParser parser(graphCode);
  auto status = parser.Parse(graph);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  EXPECT_TRUE(parser.EndOfInput()) << "Extra unparsed input unexpected.";

  // Constructs name to TypeProto map from value_info, input, output
  std::unordered_map<std::string, TypeProto*> valueTypesByName;
  for (auto& vi : *graph.mutable_value_info()) {
    if (vi.has_type()) {
      valueTypesByName[vi.name()] = vi.mutable_type();
    }
  }
  for (auto& vi : *graph.mutable_input()) {
    if (vi.has_type()) {
      valueTypesByName[vi.name()] = vi.mutable_type();
    }
  }
  for (auto& vi : *graph.mutable_output()) {
    if (vi.has_type()) {
      valueTypesByName[vi.name()] = vi.mutable_type();
    }
  }

  // Constructs name to TensorProto map from initializer
  std::unordered_map<std::string, const TensorProto*> inputDataByName;
  for (const auto& tp : graph.initializer()) {
    inputDataByName[tp.name()] = &tp;
  }
  // Collects data from constant nodes
  for (const auto& n : graph.node()) {
    if (n.op_type() != "Constant" || n.output().size() != 1) {
      continue;
    }
    for (const auto& attr : n.attribute()) {
      if (attr.name() == "value") {
        if (attr.type() == AttributeProto::TENSOR && attr.has_t()) {
          inputDataByName[n.output(0)] = &attr.t();
        }
      }
    }
  }

  // Runs data propagation on each node
  std::unordered_map<std::string, TensorShapeProto> generatedShapeDataByName;
  auto* schemaRegistry = OpSchemaRegistry::Instance();
  TensorShapeProto inferredShape;
  for (auto n : graph.node()) {
    // No need to run data propagation on Constant
    if (n.op_type() == "Constant") {
      continue;
    }
    DataPropagationContextImpl dataPropagationCtx(n, valueTypesByName, inputDataByName, generatedShapeDataByName);
    const auto schema = schemaRegistry->GetSchema(n.op_type(), domainVersion, n.domain());
    EXPECT_TRUE(schema->has_data_propagation_function());
    schema->GetDataPropagationFunction()(dataPropagationCtx);
  }

  // Assuming the graph being tested only has 1 output.
  // If this ever changes then fixes are required here.
  const auto inputShapeDataIter = generatedShapeDataByName.find(graph.output(0).name());
  EXPECT_TRUE(inputShapeDataIter != generatedShapeDataByName.cend());

  inferredShape.CopyFrom(inputShapeDataIter->second);

  // Returns the partial shape data for output
  return inferredShape;
}

TEST(DataPropagationImplTest, ShapeTest) {
  const char* code = R"ONNX(
agraph (int32[7,4,1] x) => (int32[3] y)
{
    xs = Shape(x)
    y = Cast<to = 7>(xs)
}
)ONNX";
  TensorShapeProto expected_tsp;
  expected_tsp.mutable_dim()->Add()->set_dim_value(7);
  expected_tsp.mutable_dim()->Add()->set_dim_value(4);
  expected_tsp.mutable_dim()->Add()->set_dim_value(1);
  const auto propagated_tsp = RunDataPropagation(code);
  EXPECT_TRUE(CompareShape(propagated_tsp, expected_tsp));
}

TEST(DataPropagationImplTest, SymbolicShapeTest) {
  const char* code = R"ONNX(
agraph (int32[N,3,256,256] x) => (int32[4] y)
{
    xs = Shape(x)
    y = Cast<to = 7>(xs)
}
)ONNX";
  TensorShapeProto expected_tsp;
  expected_tsp.mutable_dim()->Add()->set_dim_param("N");
  expected_tsp.mutable_dim()->Add()->set_dim_value(3);
  expected_tsp.mutable_dim()->Add()->set_dim_value(256);
  expected_tsp.mutable_dim()->Add()->set_dim_value(256);
  const auto propagated_tsp = RunDataPropagation(code);
  EXPECT_TRUE(CompareShape(propagated_tsp, expected_tsp, true));
}

TEST(DataPropagationImplTest, CastTest) {
  const char* code = R"ONNX(
agraph (int32[2,5] x) => (int32[2] y)
{
    xs = Shape(x)
    y = Cast<to = 7>(xs)
}
)ONNX";
  TensorShapeProto expected_tsp;
  expected_tsp.mutable_dim()->Add()->set_dim_value(2);
  expected_tsp.mutable_dim()->Add()->set_dim_value(5);
  const auto propagated_tsp = RunDataPropagation(code);
  EXPECT_TRUE(CompareShape(propagated_tsp, expected_tsp));
}

TEST(DataPropagationImplTest, SqueezeTest) {
  const char* code = R"ONNX(
agraph (int32[2,5] x) => (int32[2] z)
{
    xs = Shape(x)
    y = Squeeze(xs)
    z = Cast<to = 7>(y)
}
)ONNX";
  TensorShapeProto expected_tsp;
  expected_tsp.mutable_dim()->Add()->set_dim_value(2);
  expected_tsp.mutable_dim()->Add()->set_dim_value(5);
  const auto propagated_tsp = RunDataPropagation(code);
  EXPECT_TRUE(CompareShape(propagated_tsp, expected_tsp));
}

TEST(DataPropagationImplTest, UnsqueezeTest) {
  const char* code = R"ONNX(
agraph (int32[2,5] x) => (int32[1,2] w)
{
    xs = Shape(x)
    axis = Constant<value = int64[1] {1}>()
    z = Unsqueeze(xs, axis)
    w = Cast<to = 7>(z)
}
)ONNX";
  TensorShapeProto expected_tsp;
  expected_tsp.mutable_dim()->Add()->set_dim_value(2);
  expected_tsp.mutable_dim()->Add()->set_dim_value(5);
  const auto propagated_tsp = RunDataPropagation(code);
  EXPECT_TRUE(CompareShape(propagated_tsp, expected_tsp));
}

TEST(DataPropagationImplTest, SizeTest) {
  const char* code = R"ONNX(
agraph (int64[1] x) => (int32[1] w)
<int64[3] init = {2,3,5}>
{
    z = Size(init)
    w = Cast<to = 7>(z)
}
)ONNX";
  TensorShapeProto expected_tsp;
  expected_tsp.mutable_dim()->Add()->set_dim_value(3);
  const auto propagated_tsp = RunDataPropagation(code);
  EXPECT_TRUE(CompareShape(propagated_tsp, expected_tsp));
}

TEST(DataPropagationImplTest, AddTest) {
  const char* code = R"ONNX(
agraph (int32[2,4,5] x, int32[2,4,5] y) => (int32[3] w)
{
    xs = Shape(x)
    ys = Shape(y)
    z = Add(xs, ys)
    w = Cast<to = 7>(z)
}
)ONNX";
  TensorShapeProto expected_tsp;
  expected_tsp.mutable_dim()->Add()->set_dim_value(4);
  expected_tsp.mutable_dim()->Add()->set_dim_value(8);
  expected_tsp.mutable_dim()->Add()->set_dim_value(10);
  const auto propagated_tsp = RunDataPropagation(code);
  EXPECT_TRUE(CompareShape(propagated_tsp, expected_tsp));
}

TEST(DataPropagationImplTest, AddSymbolicShapeTest) {
  const char* code = R"ONNX(
agraph (int32[2,4,5] x, int32[2,4,M] y) => (int32[3] w)
{
    xs = Shape(x)
    ys = Shape(y)
    z = Add(xs, ys)
    w = Cast<to = 7>(z)
}
)ONNX";
  // Add({2,4,5}, {2,4,M}) = {4,8,?}
  TensorShapeProto expected_tsp;
  expected_tsp.mutable_dim()->Add()->set_dim_value(4);
  expected_tsp.mutable_dim()->Add()->set_dim_value(8);
  // Not computable so do not set value or param
  expected_tsp.mutable_dim()->Add();
  const auto propagated_tsp = RunDataPropagation(code);
  EXPECT_TRUE(CompareShape(propagated_tsp, expected_tsp));
}

TEST(DataPropagationImplTest, SubTest) {
  const char* code = R"ONNX(
agraph (int32[10,11,6] x, int32[5] y) => (int32[3] w)
{
    xs = Shape(x)
    ys = Shape(y)
    z = Sub(xs, ys)
    w = Cast<to = 7>(z)
}
)ONNX";
  TensorShapeProto expected_tsp;
  expected_tsp.mutable_dim()->Add()->set_dim_value(5);
  expected_tsp.mutable_dim()->Add()->set_dim_value(6);
  expected_tsp.mutable_dim()->Add()->set_dim_value(1);
  const auto propagated_tsp = RunDataPropagation(code);
  EXPECT_TRUE(CompareShape(propagated_tsp, expected_tsp));
}

TEST(DataPropagationImplTest, MulTest) {
  const char* code = R"ONNX(
agraph (int32[2] x, int32[5,1,7] y) => (int32[3] w)
{
    xs = Shape(x)
    ys = Shape(y)
    z = Mul(xs, ys)
    w = Cast<to = 7>(z)
}
)ONNX";
  TensorShapeProto expected_tsp;
  expected_tsp.mutable_dim()->Add()->set_dim_value(10);
  expected_tsp.mutable_dim()->Add()->set_dim_value(2);
  expected_tsp.mutable_dim()->Add()->set_dim_value(14);
  const auto propagated_tsp = RunDataPropagation(code);
  EXPECT_TRUE(CompareShape(propagated_tsp, expected_tsp));
}

TEST(DataPropagationImplTest, ConcatTest) {
  const char* code = R"ONNX(
agraph (int32[1,2] x, int32[3,4] y) => (int32[4] w)
{
    xs = Shape(x)
    ys = Shape(y)
    z = Concat<axis = 0>(xs, ys)
    w = Cast<to = 7>(z)
}
)ONNX";
  TensorShapeProto expected_tsp;
  expected_tsp.mutable_dim()->Add()->set_dim_value(1);
  expected_tsp.mutable_dim()->Add()->set_dim_value(2);
  expected_tsp.mutable_dim()->Add()->set_dim_value(3);
  expected_tsp.mutable_dim()->Add()->set_dim_value(4);
  const auto propagated_tsp = RunDataPropagation(code);
  EXPECT_TRUE(CompareShape(propagated_tsp, expected_tsp));
}

TEST(DataPropagationImplTest, GatherTest) {
  const char* code = R"ONNX(
agraph (int32[1,2,3,4,5,6] x) => (int32[3] w)
{
    xs = Shape(x)
    indices = Constant<value = int64[3] {0,3,5}>()
    z = Gather<axis = 0>(xs, indices)
    w = Cast<to = 7>(z)
}
)ONNX";
  TensorShapeProto expected_tsp;
  expected_tsp.mutable_dim()->Add()->set_dim_value(1);
  expected_tsp.mutable_dim()->Add()->set_dim_value(4);
  expected_tsp.mutable_dim()->Add()->set_dim_value(6);
  const auto propagated_tsp = RunDataPropagation(code);
  EXPECT_TRUE(CompareShape(propagated_tsp, expected_tsp));
}

TEST(DataPropagationImplTest, GatherNegativeIndicesTest) {
  const char* code = R"ONNX(
agraph (int32[1,2,3,4,5,6] x) => (int32[2] w)
{
    xs = Shape(x)
    indices = Constant<value = int64[2] {-2,-1}>()
    z = Gather<axis = 0>(xs, indices)
    w = Cast<to = 7>(z)
}
)ONNX";
  TensorShapeProto expected_tsp;
  expected_tsp.mutable_dim()->Add()->set_dim_value(5);
  expected_tsp.mutable_dim()->Add()->set_dim_value(6);
  const auto propagated_tsp = RunDataPropagation(code);
  EXPECT_TRUE(CompareShape(propagated_tsp, expected_tsp));
}

TEST(DataPropagationImplTest, SliceTest) {
  const char* code = R"ONNX(
agraph (int32[1,2,3,4,5,6,7,8] x) => (int32[2] w)
{
    xs = Shape(x)
    starts = Constant<value = int64[1] {1}>()
    ends = Constant<value = int64[1] {7}>()
    axes = Constant<value = int64[1] {0}>()
    steps = Constant<value = int64[1] {3}>()
    z = Slice(xs, starts, ends, axes, steps)
    w = Cast<to = 7>(z)
}
)ONNX";
  TensorShapeProto expected_tsp;
  expected_tsp.mutable_dim()->Add()->set_dim_value(2);
  expected_tsp.mutable_dim()->Add()->set_dim_value(5);
  const auto propagated_tsp = RunDataPropagation(code);
  EXPECT_TRUE(CompareShape(propagated_tsp, expected_tsp));
}

TEST(DataPropagationImplTest, SliceDefaultAxesAndStepTest) {
  const char* code = R"ONNX(
agraph (int32[1,2,3,4,5,6,7,8] x) => (int32[3] w)
{
    xs = Shape(x)
    starts = Constant<value = int64[1] {2}>()
    ends = Constant<value = int64[1] {5}>()
    z = Slice(xs, starts, ends)
    w = Cast<to = 7>(z)
}
)ONNX";
  TensorShapeProto expected_tsp;
  expected_tsp.mutable_dim()->Add()->set_dim_value(3);
  expected_tsp.mutable_dim()->Add()->set_dim_value(4);
  expected_tsp.mutable_dim()->Add()->set_dim_value(5);
  const auto propagated_tsp = RunDataPropagation(code);
  EXPECT_TRUE(CompareShape(propagated_tsp, expected_tsp));
}

TEST(DataPropagationImplTest, SliceNegativeStartEndStepTest) {
  const char* code = R"ONNX(
agraph (int32[1,2,3,4,5,6,7,8] x) => (int32[3] w)
{
    xs = Shape(x)
    starts = Constant<value = int64[1] {-3}>()
    ends = Constant<value = int64[1] {-7}>()
    axes = Constant<value = int64[1] {0}>()
    steps = Constant<value = int64[1] {-2}>()
    z = Slice(xs, starts, ends, axes, steps)
    w = Cast<to = 7>(z)
}
)ONNX";
  TensorShapeProto expected_tsp;
  expected_tsp.mutable_dim()->Add()->set_dim_value(6);
  expected_tsp.mutable_dim()->Add()->set_dim_value(4);
  const auto propagated_tsp = RunDataPropagation(code);
  EXPECT_TRUE(CompareShape(propagated_tsp, expected_tsp));
}

} // namespace Test
} // namespace ONNX_NAMESPACE
