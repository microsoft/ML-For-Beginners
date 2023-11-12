/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tensor_util.h"
#include <vector>
#include "onnx/common/platform_helpers.h"

namespace ONNX_NAMESPACE {

#define DEFINE_PARSE_DATA(type, typed_data_fetch)                          \
  template <>                                                              \
  const std::vector<type> ParseData(const Tensor* tensor) {                \
    std::vector<type> res;                                                 \
    if (!tensor->is_raw_data()) {                                          \
      const auto& data = tensor->typed_data_fetch();                       \
      res.insert(res.end(), data.begin(), data.end());                     \
      return res;                                                          \
    }                                                                      \
    /* make copy as we may have to reverse bytes */                        \
    std::string raw_data = tensor->raw();                                  \
    /* okay to remove const qualifier as we have already made a copy */    \
    char* bytes = const_cast<char*>(raw_data.c_str());                     \
    /*onnx is little endian serialized always-tweak byte order if needed*/ \
    if (!is_processor_little_endian()) {                                   \
      const size_t element_size = sizeof(type);                            \
      const size_t num_elements = raw_data.size() / element_size;          \
      for (size_t i = 0; i < num_elements; ++i) {                          \
        char* start_byte = bytes + i * element_size;                       \
        char* end_byte = start_byte + element_size - 1;                    \
        /* keep swapping */                                                \
        for (size_t count = 0; count < element_size / 2; ++count) {        \
          char temp = *start_byte;                                         \
          *start_byte = *end_byte;                                         \
          *end_byte = temp;                                                \
          ++start_byte;                                                    \
          --end_byte;                                                      \
        }                                                                  \
      }                                                                    \
    }                                                                      \
    /* raw_data.c_str()/bytes is a byte array and may not be properly  */  \
    /* aligned for the underlying type */                                  \
    /* We need to copy the raw_data.c_str()/bytes as byte instead of  */   \
    /* copying as the underlying type, otherwise we may hit memory   */    \
    /* misalignment issues on certain platforms, such as arm32-v7a */      \
    const size_t raw_data_size = raw_data.size();                          \
    res.resize(raw_data_size / sizeof(type));                              \
    memcpy(reinterpret_cast<char*>(res.data()), bytes, raw_data_size);     \
    return res;                                                            \
  }

DEFINE_PARSE_DATA(int32_t, int32s)
DEFINE_PARSE_DATA(int64_t, int64s)
DEFINE_PARSE_DATA(float, floats)
DEFINE_PARSE_DATA(double, doubles)
DEFINE_PARSE_DATA(uint64_t, uint64s)

} // namespace ONNX_NAMESPACE
