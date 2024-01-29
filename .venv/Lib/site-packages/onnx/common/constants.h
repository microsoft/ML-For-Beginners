// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

namespace ONNX_NAMESPACE {
// For ONNX op/function registration.

// ONNX domains.
constexpr const char* AI_ONNX_ML_DOMAIN = "ai.onnx.ml";
constexpr const char* AI_ONNX_TRAINING_DOMAIN = "ai.onnx.training";
constexpr const char* AI_ONNX_PREVIEW_TRAINING_DOMAIN = "ai.onnx.preview.training";
// The following two are equivalent in an onnx proto representation.
constexpr const char* ONNX_DOMAIN = "";
constexpr const char* AI_ONNX_DOMAIN = "ai.onnx";

inline std::string NormalizeDomain(const std::string& domain) {
  return (domain == AI_ONNX_DOMAIN) ? ONNX_DOMAIN : domain;
}

inline bool IsOnnxDomain(const std::string& domain) {
  return (domain == AI_ONNX_DOMAIN) || ((domain == ONNX_DOMAIN));
}

constexpr bool OPTIONAL_VALUE = false;

// For dimension denotation.
constexpr const char* DATA_BATCH = "DATA_BATCH";
constexpr const char* DATA_CHANNEL = "DATA_CHANNEL";
constexpr const char* DATA_TIME = "DATA_TIME";
constexpr const char* DATA_FEATURE = "DATA_FEATURE";
constexpr const char* FILTER_IN_CHANNEL = "FILTER_IN_CHANNEL";
constexpr const char* FILTER_OUT_CHANNEL = "FILTER_OUT_CHANNEL";
constexpr const char* FILTER_SPATIAL = "FILTER_SPATIAL";

// For type denotation.
constexpr const char* TENSOR = "TENSOR";
constexpr const char* IMAGE = "IMAGE";
constexpr const char* AUDIO = "AUDIO";
constexpr const char* TEXT = "TEXT";

} // namespace ONNX_NAMESPACE
