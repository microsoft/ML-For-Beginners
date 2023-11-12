// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <sstream>
#include <string>

namespace ONNX_NAMESPACE {

#if defined(__ANDROID__)
template <typename T>
std::string to_string(T value) {
  std::ostringstream os;
  os << value;
  return os.str();
}

inline int stoi(const std::string& str) {
  std::stringstream ss;
  int n = 0;
  ss << str;
  ss >> n;
  return n;
}

#else
using std::stoi;
using std::to_string;
#endif // defined(__ANDROID__)

inline void MakeStringInternal(std::stringstream& /*ss*/) {}

template <typename T>
inline void MakeStringInternal(std::stringstream& ss, const T& t) {
  ss << t;
}

template <typename T, typename... Args>
inline void MakeStringInternal(std::stringstream& ss, const T& t, const Args&... args) {
  MakeStringInternal(ss, t);
  MakeStringInternal(ss, args...);
}

template <typename... Args>
std::string MakeString(const Args&... args) {
  std::stringstream ss;
  MakeStringInternal(ss, args...);
  return std::string(ss.str());
}

// Specializations for already-a-string types.
template <>
inline std::string MakeString(const std::string& str) {
  return str;
}
inline std::string MakeString(const char* c_str) {
  return std::string(c_str);
}
} // namespace ONNX_NAMESPACE
