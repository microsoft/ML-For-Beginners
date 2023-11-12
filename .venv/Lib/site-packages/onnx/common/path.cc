/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright (c) ONNX Project Contributors.

#include "onnx/common/path.h"

namespace ONNX_NAMESPACE {
#ifdef _WIN32
#else

std::string path_join(const std::string& origin, const std::string& append) {
  if (origin.find_last_of(k_preferred_path_separator) != origin.length() - 1) {
    return origin + k_preferred_path_separator + append;
  }
  return origin + append;
}

std::string clean_relative_path(const std::string& path) {
  if (path.empty()) {
    return ".";
  }

  std::string out;

  size_t n = path.size();

  size_t r = 0;
  size_t dotdot = 0;

  while (r < n) {
    if (path[r] == k_preferred_path_separator) {
      r++;
      continue;
    }

    if (path[r] == '.' && (r + 1 == n || path[r + 1] == k_preferred_path_separator)) {
      r++;
      continue;
    }

    if (path[r] == '.' && path[r + 1] == '.' && (r + 2 == n || path[r + 2] == k_preferred_path_separator)) {
      r += 2;

      if (out.size() > dotdot) {
        while (out.size() > dotdot && out.back() != k_preferred_path_separator) {
          out.pop_back();
        }
        if (!out.empty())
          out.pop_back();
      } else {
        if (!out.empty()) {
          out.push_back(k_preferred_path_separator);
        }

        out.push_back('.');
        out.push_back('.');
        dotdot = out.size();
      }

      continue;
    }

    if (!out.empty() && out.back() != k_preferred_path_separator) {
      out.push_back(k_preferred_path_separator);
    }

    for (; r < n && path[r] != k_preferred_path_separator; r++) {
      out.push_back(path[r]);
    }
  }

  if (out.empty()) {
    out.push_back('.');
  }

  return out;
}
#endif

} // namespace ONNX_NAMESPACE
