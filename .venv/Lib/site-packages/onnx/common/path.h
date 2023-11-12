/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright (c) ONNX Project Contributors.

#pragma once

#include <string>
#ifdef _WIN32
#include <windows.h>
#include <filesystem>
#include "onnx/checker.h"
#endif

namespace ONNX_NAMESPACE {

#ifdef _WIN32
constexpr const char k_preferred_path_separator = '\\';
#else // POSIX
constexpr const char k_preferred_path_separator = '/';
#endif

#ifdef _WIN32
inline std::wstring path_join(const std::wstring& origin, const std::wstring& append) {
  return (std::filesystem::path(origin) / std::filesystem::path(append)).wstring();
}
inline std::wstring utf8str_to_wstring(const std::string& utf8str) {
  if (utf8str.size() > INT_MAX) {
    fail_check("utf8str_to_wstring: string is too long for converting to wstring.");
  }
  int size_required = MultiByteToWideChar(CP_UTF8, 0, utf8str.c_str(), (int)utf8str.size(), NULL, 0);
  std::wstring ws_str(size_required, 0);
  MultiByteToWideChar(CP_UTF8, 0, utf8str.c_str(), (int)utf8str.size(), &ws_str[0], size_required);
  return ws_str;
}

#else
std::string path_join(const std::string& origin, const std::string& append);
// TODO: also use std::filesystem::path for clean_relative_path after ONNX has supported C++17 for POSIX
// Clean up relative path when there is ".." in the path, e.g.: a/b/../c -> a/c
// It cannot work with absolute path
std::string clean_relative_path(const std::string& path);
#endif

} // namespace ONNX_NAMESPACE
