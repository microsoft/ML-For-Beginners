// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once
#include <stdint.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace ONNX_NAMESPACE {

#define FORALL_BUILTIN_SYMBOLS(_)   \
  _(spatial)                        \
  _(select_last_index)              \
  _(coordinate_transformation_mode) \
  _(PythonOp)                       \
  _(CppOp)                          \
  _(Param)                          \
  _(Select)                         \
  _(Return)                         \
  _(Eval)                           \
  _(add)                            \
  _(Add)                            \
  _(Div)                            \
  _(Mul)                            \
  _(Neg)                            \
  _(Sub)                            \
  _(Pow)                            \
  _(Sigmoid)                        \
  _(ArgMax)                         \
  _(Concat)                         \
  _(Softmax)                        \
  _(LogSoftmax)                     \
  _(Dropout)                        \
  _(Tanh)                           \
  _(mul)                            \
  _(neg)                            \
  _(sigmoid)                        \
  _(tanh)                           \
  _(Constant)                       \
  _(cat)                            \
  _(Slice)                          \
  _(Squeeze)                        \
  _(Undefined)                      \
  _(FusionGroup)                    \
  _(MatMul)                         \
  _(Gemm)                           \
  _(Tile)                           \
  _(SubConstant)                    \
  _(Scale)                          \
  _(Transpose)                      \
  _(Pad)                            \
  _(Reshape)                        \
  _(split)                          \
  _(chunk)                          \
  _(Offset)                         \
  _(value)                          \
  _(Subgraph)                       \
  _(BatchNormalization)             \
  _(Conv)                           \
  _(ConvTranspose)                  \
  _(is_test)                        \
  _(epsilon)                        \
  _(expand)                         \
  _(Expand)                         \
  _(order)                          \
  _(momentum)                       \
  _(consumed_inputs)                \
  _(kernels)                        \
  _(kernel_shape)                   \
  _(kernel)                         \
  _(scale)                          \
  _(strides)                        \
  _(stride)                         \
  _(pads)                           \
  _(pad)                            \
  _(beta)                           \
  _(alpha)                          \
  _(dilations)                      \
  _(dilation)                       \
  _(broadcast)                      \
  _(axis)                           \
  _(ratio)                          \
  _(size)                           \
  _(dim)                            \
  _(keepdims)                       \
  _(perm)                           \
  _(shape)                          \
  _(axes)                           \
  _(group)                          \
  _(inplace)                        \
  _(transA)                         \
  _(transB)                         \
  _(other)                          \
  _(__and__)                        \
  _(__lshift__)                     \
  _(__or__)                         \
  _(__rshift__)                     \
  _(__xor__)                        \
  _(abs)                            \
  _(acos)                           \
  _(asin)                           \
  _(atan)                           \
  _(atan2)                          \
  _(ceil)                           \
  _(clamp)                          \
  _(cos)                            \
  _(cosh)                           \
  _(div)                            \
  _(eq)                             \
  _(equal)                          \
  _(Exp)                            \
  _(ends)                           \
  _(expm1)                          \
  _(floor)                          \
  _(fmod)                           \
  _(frac)                           \
  _(ge)                             \
  _(gt)                             \
  _(le)                             \
  _(lerp)                           \
  _(lgamma)                         \
  _(Log)                            \
  _(log1p)                          \
  _(lt)                             \
  _(max)                            \
  _(min)                            \
  _(ne)                             \
  _(ones)                           \
  _(pow)                            \
  _(reciprocal)                     \
  _(remainder)                      \
  _(round)                          \
  _(rsqrt)                          \
  _(sin)                            \
  _(sinh)                           \
  _(Sqrt)                           \
  _(sub)                            \
  _(starts)                         \
  _(tan)                            \
  _(trunc)                          \
  _(zeros)                          \
  _(exponent)                       \
  _(device)                         \
  _(mode)                           \
  _(Identity)                       \
  _(Loop)                           \
  _(If)                             \
  _(body)                           \
  _(then_branch)                    \
  _(else_branch)                    \
  _(Captured)                       \
  _(__control_inputs)               \
  _(count_include_pad)              \
  _(storage_order)                  \
  _(Unsqueeze)                      \
  _(ReduceL1)                       \
  _(ReduceL2)                       \
  _(ReduceLogSum)                   \
  _(ReduceLogSumExp)                \
  _(ReduceMax)                      \
  _(ReduceMean)                     \
  _(ReduceMin)                      \
  _(ReduceProd)                     \
  _(ReduceSum)                      \
  _(ReduceSumSquare)                \
  _(Cast)                           \
  _(to)                             \
  _(PRelu)                          \
  _(Greater)                        \
  _(Less)                           \
  _(scales)                         \
  _(Upsample)                       \
  _(RNN)                            \
  _(layout)                         \
  _(k)                              \
  _(Flatten)                        \
  _(ScatterElements)                \
  _(Resize)                         \
  _(ceil_mode)                      \
  _(num_outputs)

enum BuiltinSymbol {
#define DEFINE_SYMBOL(s) k##s,
  FORALL_BUILTIN_SYMBOLS(DEFINE_SYMBOL)
#undef DEFINE_SYMBOL
      kLastSymbol, // where we start counting for new symbols
};

struct Symbol {
  Symbol() {}
  /*implicit*/ Symbol(BuiltinSymbol value) : value(value) {}
  explicit Symbol(const std::string& s);
  explicit Symbol(uint32_t value) : value(value) {}

  operator uint32_t() const {
    return value;
  }
  const char* toString() const;

 private:
  uint32_t value;
};

static inline bool operator==(Symbol lhs, Symbol rhs) {
  return static_cast<uint32_t>(lhs) == static_cast<uint32_t>(rhs);
}
// necessary to prevent ambiguous overload resolutions
static inline bool operator==(BuiltinSymbol lhs, Symbol rhs) {
  return static_cast<uint32_t>(lhs) == static_cast<uint32_t>(rhs);
}
static inline bool operator==(Symbol lhs, BuiltinSymbol rhs) {
  return static_cast<uint32_t>(lhs) == static_cast<uint32_t>(rhs);
}

inline Symbol operator"" _sym(const char* s, size_t) {
  return Symbol(s);
}

} // namespace ONNX_NAMESPACE

// make symbol behave like an integer in hash tables
namespace std {
template <>
struct hash<ONNX_NAMESPACE::Symbol> {
  std::size_t operator()(ONNX_NAMESPACE::Symbol s) const {
    return std::hash<uint32_t>()(static_cast<uint32_t>(s));
  }
};

} // namespace std
