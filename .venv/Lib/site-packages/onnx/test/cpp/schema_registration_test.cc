// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iostream>
#include "gtest/gtest.h"
#include "onnx/defs/operator_sets.h"
#include "onnx/defs/schema.h"

using namespace ONNX_NAMESPACE;

namespace ONNX_NAMESPACE {
namespace Test {

// By default ONNX registers all opset versions and selective schema loading cannot be tested
// So this test is run only when static registration is disabled
TEST(SchemaRegistrationTest, RegisterSpecifiedOpsetSchemaVersion) {
#ifdef __ONNX_DISABLE_STATIC_REGISTRATION
  EXPECT_TRUE(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion() == -1);
  RegisterOnnxOperatorSetSchema(13);
  EXPECT_TRUE(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion() == 13);

  auto opSchema = OpSchemaRegistry::Schema("Add");
  EXPECT_NE(nullptr, opSchema);
  EXPECT_EQ(opSchema->SinceVersion(), 13);

  // Should not find opset 12
  opSchema = OpSchemaRegistry::Schema("Add", 12);
  EXPECT_EQ(nullptr, opSchema);

  // Should not find opset 14
  opSchema = OpSchemaRegistry::Schema("Trilu");
  EXPECT_EQ(nullptr, opSchema);

  // Acos-7 is the latest Acos before specified 13
  opSchema = OpSchemaRegistry::Schema("Acos");
  EXPECT_NE(nullptr, opSchema);
  EXPECT_EQ(opSchema->SinceVersion(), 7);
#endif
}

} // namespace Test
} // namespace ONNX_NAMESPACE
