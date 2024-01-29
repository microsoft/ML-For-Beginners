// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <list>
#include <utility>

#include "gtest/gtest.h"
#include "onnx/common/path.h"

#ifdef _WIN32
// Only test clean_relative_path and normalize_separator on non-Windows
// because Windows has its own implementation for them from std::filesystem::path.
#else
using namespace ONNX_NAMESPACE;
namespace ONNX_NAMESPACE {
namespace Test {

TEST(PathTest, CleanRelativePathTest) {
  // Already normal.
  EXPECT_EQ(clean_relative_path("abc"), "abc");
  EXPECT_EQ(clean_relative_path("abc/def"), "abc/def");
  EXPECT_EQ(clean_relative_path("a/b/c"), "a/b/c");
  EXPECT_EQ(clean_relative_path("."), ".");
  EXPECT_EQ(clean_relative_path(".."), "..");
  EXPECT_EQ(clean_relative_path("../.."), "../..");
  EXPECT_EQ(clean_relative_path("../../abc"), "../../abc");
  // Remove trailing slash
  EXPECT_EQ(clean_relative_path("abc/"), "abc");
  EXPECT_EQ(clean_relative_path("abc/def/"), "abc/def");
  EXPECT_EQ(clean_relative_path("a/b/c/"), "a/b/c");
  EXPECT_EQ(clean_relative_path("./"), ".");
  EXPECT_EQ(clean_relative_path("../"), "..");
  EXPECT_EQ(clean_relative_path("../../"), "../..");
  // Remove doubled slash
  EXPECT_EQ(clean_relative_path("abc//def//ghi"), "abc/def/ghi");
  EXPECT_EQ(clean_relative_path("abc///"), "abc");
  EXPECT_EQ(clean_relative_path("abc//"), "abc");
  // Remove . elements
  EXPECT_EQ(clean_relative_path("abc/./def"), "abc/def");
  EXPECT_EQ(clean_relative_path("./abc/def"), "abc/def");
  EXPECT_EQ(clean_relative_path("abc/."), "abc");
  // Remove .. elements
  EXPECT_EQ(clean_relative_path("abc/def/ghi/../jkl"), "abc/def/jkl");
  EXPECT_EQ(clean_relative_path("abc/def/../ghi/../jkl"), "abc/jkl");
  EXPECT_EQ(clean_relative_path("abc/def/.."), "abc");
  EXPECT_EQ(clean_relative_path("abc/def/../.."), ".");
  EXPECT_EQ(clean_relative_path("abc/def/../../.."), "..");
  EXPECT_EQ(clean_relative_path("abc/def/../../../ghi/jkl/../../../mno"), "../../mno");
  EXPECT_EQ(clean_relative_path("../abc"), "../abc");
  // Combinations
  EXPECT_EQ(clean_relative_path("abc/./../def"), "def");
  EXPECT_EQ(clean_relative_path("abc//./../def"), "def");
  EXPECT_EQ(clean_relative_path("abc/../../././../def"), "../../def");
}
} // namespace Test
} // namespace ONNX_NAMESPACE
#endif
