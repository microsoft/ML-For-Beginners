// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iostream>

#include "gtest/gtest.h"
#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {
namespace Test {
TEST(OpRegistrationTest, GemmOp) {
  auto opSchema = OpSchemaRegistry::Schema("Gemm");
  EXPECT_TRUE(nullptr != opSchema);
  size_t input_size = opSchema->inputs().size();
  EXPECT_EQ(input_size, 3);
  EXPECT_EQ(opSchema->inputs()[0].GetTypes(), opSchema->outputs()[0].GetTypes());
  size_t attr_size = opSchema->attributes().size();
  EXPECT_EQ(attr_size, 4);
  EXPECT_NE(opSchema->attributes().count("alpha"), 0);
  EXPECT_EQ(opSchema->attributes().at("alpha").type, AttributeProto_AttributeType_FLOAT);
  EXPECT_NE(opSchema->attributes().count("beta"), 0);
  EXPECT_EQ(opSchema->attributes().at("beta").type, AttributeProto_AttributeType_FLOAT);
}
} // namespace Test
} // namespace ONNX_NAMESPACE
