/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tensor_proto_util.h"

#include <string>
#include <vector>

#include "onnx/common/platform_helpers.h"
#include "onnx/defs/data_type_utils.h"
#include "onnx/defs/shape_inference.h"

namespace ONNX_NAMESPACE {

#define DEFINE_TO_TENSOR_ONE(type, enumType, field) \
  template <>                                       \
  TensorProto ToTensor<type>(const type& value) {   \
    TensorProto t;                                  \
    t.set_data_type(enumType);                      \
    t.add_##field##_data(value);                    \
    return t;                                       \
  }

#define DEFINE_TO_TENSOR_LIST(type, enumType, field)            \
  template <>                                                   \
  TensorProto ToTensor<type>(const std::vector<type>& values) { \
    TensorProto t;                                              \
    t.clear_##field##_data();                                   \
    t.set_data_type(enumType);                                  \
    for (const type& val : values) {                            \
      t.add_##field##_data(val);                                \
    }                                                           \
    return t;                                                   \
  }

#define DEFINE_PARSE_DATA(type, typed_data_fetch, tensorproto_datatype)                                            \
  template <>                                                                                                      \
  const std::vector<type> ParseData(const TensorProto* tensor_proto) {                                             \
    if (!tensor_proto->has_data_type() || tensor_proto->data_type() == TensorProto_DataType_UNDEFINED) {           \
      fail_shape_inference("The type of tensor: ", tensor_proto->name(), " is undefined so it cannot be parsed."); \
    } else if (tensor_proto->data_type() != tensorproto_datatype) {                                                \
      fail_shape_inference(                                                                                        \
          "ParseData type mismatch for tensor: ",                                                                  \
          tensor_proto->name(),                                                                                    \
          ". Expected:",                                                                                           \
          Utils::DataTypeUtils::ToDataTypeString(tensorproto_datatype),                                            \
          " Actual:",                                                                                              \
          Utils::DataTypeUtils::ToDataTypeString(tensor_proto->data_type()));                                      \
    }                                                                                                              \
    std::vector<type> res;                                                                                         \
    if (tensor_proto->has_data_location() && tensor_proto->data_location() == TensorProto_DataLocation_EXTERNAL) { \
      fail_shape_inference(                                                                                        \
          "Cannot parse data from external tensors. Please ",                                                      \
          "load external data into raw data for tensor: ",                                                         \
          tensor_proto->name());                                                                                   \
    } else if (!tensor_proto->has_raw_data()) {                                                                    \
      const auto& data = tensor_proto->typed_data_fetch();                                                         \
      int expected_size = 1;                                                                                       \
      for (int i = 0; i < tensor_proto->dims_size(); ++i) {                                                        \
        expected_size *= tensor_proto->dims(i);                                                                    \
      }                                                                                                            \
      if (tensor_proto->dims_size() != 0 && data.size() != expected_size) {                                        \
        fail_shape_inference(                                                                                      \
            "Data size mismatch. Tensor: ",                                                                        \
            tensor_proto->name(),                                                                                  \
            " expected size ",                                                                                     \
            expected_size,                                                                                         \
            " does not match the actual size",                                                                     \
            data.size());                                                                                          \
      }                                                                                                            \
      res.insert(res.end(), data.begin(), data.end());                                                             \
      return res;                                                                                                  \
    }                                                                                                              \
    if (tensor_proto->data_type() == TensorProto_DataType_STRING) {                                                \
      fail_shape_inference(                                                                                        \
          tensor_proto->name(),                                                                                    \
          " data type is string. string",                                                                          \
          " content is required to be stored in repeated bytes string_data field.",                                \
          " raw_data type cannot be string.");                                                                     \
    }                                                                                                              \
    /* The given tensor does have raw_data itself so parse it by given type */                                     \
    /* make copy as we may have to reverse bytes */                                                                \
    std::string raw_data = tensor_proto->raw_data();                                                               \
    if (raw_data.empty()) {                                                                                        \
      return res;                                                                                                  \
    }                                                                                                              \
    /* okay to remove const qualifier as we have already made a copy */                                            \
    char* bytes = const_cast<char*>(raw_data.c_str());                                                             \
    /* onnx is little endian serialized always-tweak byte order if needed */                                       \
    if (!is_processor_little_endian()) {                                                                           \
      const size_t element_size = sizeof(type);                                                                    \
      const size_t num_elements = raw_data.size() / element_size;                                                  \
      for (size_t i = 0; i < num_elements; ++i) {                                                                  \
        char* start_byte = bytes + i * element_size;                                                               \
        char* end_byte = start_byte + element_size - 1;                                                            \
        /* keep swapping */                                                                                        \
        for (size_t count = 0; count < element_size / 2; ++count) {                                                \
          char temp = *start_byte;                                                                                 \
          *start_byte = *end_byte;                                                                                 \
          *end_byte = temp;                                                                                        \
          ++start_byte;                                                                                            \
          --end_byte;                                                                                              \
        }                                                                                                          \
      }                                                                                                            \
    }                                                                                                              \
    /* raw_data.c_str()/bytes is a byte array and may not be properly  */                                          \
    /* aligned for the underlying type */                                                                          \
    /* We need to copy the raw_data.c_str()/bytes as byte instead of  */                                           \
    /* copying as the underlying type, otherwise we may hit memory   */                                            \
    /* misalignment issues on certain platforms, such as arm32-v7a */                                              \
    const size_t raw_data_size = raw_data.size();                                                                  \
    res.resize(raw_data_size / sizeof(type));                                                                      \
    memcpy(reinterpret_cast<char*>(res.data()), bytes, raw_data_size);                                             \
    return res;                                                                                                    \
  }

DEFINE_TO_TENSOR_ONE(float, TensorProto_DataType_FLOAT, float)
DEFINE_TO_TENSOR_ONE(bool, TensorProto_DataType_BOOL, int32)
DEFINE_TO_TENSOR_ONE(int32_t, TensorProto_DataType_INT32, int32)
DEFINE_TO_TENSOR_ONE(int64_t, TensorProto_DataType_INT64, int64)
DEFINE_TO_TENSOR_ONE(uint64_t, TensorProto_DataType_UINT64, uint64)
DEFINE_TO_TENSOR_ONE(double, TensorProto_DataType_DOUBLE, double)
DEFINE_TO_TENSOR_ONE(std::string, TensorProto_DataType_STRING, string)

DEFINE_TO_TENSOR_LIST(float, TensorProto_DataType_FLOAT, float)
DEFINE_TO_TENSOR_LIST(bool, TensorProto_DataType_BOOL, int32)
DEFINE_TO_TENSOR_LIST(int32_t, TensorProto_DataType_INT32, int32)
DEFINE_TO_TENSOR_LIST(int64_t, TensorProto_DataType_INT64, int64)
DEFINE_TO_TENSOR_LIST(uint64_t, TensorProto_DataType_UINT64, uint64)
DEFINE_TO_TENSOR_LIST(double, TensorProto_DataType_DOUBLE, double)
DEFINE_TO_TENSOR_LIST(std::string, TensorProto_DataType_STRING, string)

DEFINE_PARSE_DATA(int32_t, int32_data, TensorProto_DataType_INT32)
DEFINE_PARSE_DATA(int64_t, int64_data, TensorProto_DataType_INT64)
DEFINE_PARSE_DATA(float, float_data, TensorProto_DataType_FLOAT)
DEFINE_PARSE_DATA(double, double_data, TensorProto_DataType_DOUBLE)

} // namespace ONNX_NAMESPACE
