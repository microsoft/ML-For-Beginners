/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {

// Forward declarations for ai.onnx version 1
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Abs);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Add);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, And);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ArgMax);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ArgMin);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, AveragePool);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, BatchNormalization);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Cast);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Ceil);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Clip);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Concat);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Constant);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Conv);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ConvTranspose);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, DepthToSpace);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Div);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Dropout);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Elu);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Equal);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Exp);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Flatten);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Floor);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, GRU);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Gather);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Gemm);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, GlobalAveragePool);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, GlobalLpPool);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, GlobalMaxPool);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Greater);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, HardSigmoid);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Hardmax);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Identity);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, If);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, InstanceNormalization);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, LRN);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, LSTM);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, LeakyRelu);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Less);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Log);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, LogSoftmax);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Loop);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, LpNormalization);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, LpPool);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, MatMul);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Max);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, MaxPool);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, MaxRoiPool);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Mean);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Min);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Mul);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Neg);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Not);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Or);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, PRelu);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Pad);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Pow);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, RNN);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, RandomNormal);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, RandomNormalLike);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, RandomUniform);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, RandomUniformLike);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Reciprocal);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ReduceL1);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ReduceL2);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ReduceLogSum);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ReduceLogSumExp);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ReduceMax);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ReduceMean);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ReduceMin);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ReduceProd);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ReduceSum);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ReduceSumSquare);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Relu);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Reshape);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Selu);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Shape);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Sigmoid);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Size);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Slice);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Softmax);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Softplus);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Softsign);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, SpaceToDepth);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Split);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Sqrt);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Squeeze);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Sub);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Sum);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Tanh);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Tile);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, TopK);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Transpose);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Unsqueeze);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Upsample);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Xor);

// Iterate over schema from ai.onnx version 1
class OpSet_Onnx_ver1 {
 public:
  static void ForEachSchema(std::function<void(OpSchema&&)> fn) {
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Abs)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Add)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, And)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ArgMax)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ArgMin)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, AveragePool)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, BatchNormalization)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Cast)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Ceil)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Clip)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Concat)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Constant)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Conv)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ConvTranspose)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, DepthToSpace)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Div)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Dropout)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Elu)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Equal)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Exp)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Flatten)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Floor)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, GRU)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Gather)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Gemm)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, GlobalAveragePool)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, GlobalLpPool)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, GlobalMaxPool)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Greater)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, HardSigmoid)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Hardmax)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Identity)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, If)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, InstanceNormalization)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, LRN)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, LSTM)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, LeakyRelu)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Less)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Log)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, LogSoftmax)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Loop)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, LpNormalization)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, LpPool)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, MatMul)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Max)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, MaxPool)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, MaxRoiPool)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Mean)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Min)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Mul)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Neg)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Not)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Or)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, PRelu)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Pad)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Pow)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, RNN)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, RandomNormal)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, RandomNormalLike)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, RandomUniform)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, RandomUniformLike)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Reciprocal)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ReduceL1)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ReduceL2)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ReduceLogSum)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ReduceLogSumExp)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ReduceMax)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ReduceMean)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ReduceMin)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ReduceProd)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ReduceSum)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ReduceSumSquare)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Relu)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Reshape)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Selu)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Shape)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Sigmoid)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Size)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Slice)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Softmax)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Softplus)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Softsign)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, SpaceToDepth)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Split)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Sqrt)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Squeeze)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Sub)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Sum)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Tanh)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Tile)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, TopK)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Transpose)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Unsqueeze)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Upsample)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Xor)>());
  }
};

// Forward declarations for ai.onnx version 2
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 2, GlobalLpPool);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 2, LpPool);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 2, Pad);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 2, Split);

// Iterate over schema from ai.onnx version 2
class OpSet_Onnx_ver2 {
 public:
  static void ForEachSchema(std::function<void(OpSchema&&)> fn) {
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 2, GlobalLpPool)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 2, LpPool)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 2, Pad)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 2, Split)>());
  }
};

// Forward declarations for ai.onnx version 3
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 3, GRU);

// Iterate over schema from ai.onnx version 3
class OpSet_Onnx_ver3 {
 public:
  static void ForEachSchema(std::function<void(OpSchema&&)> fn) {
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 3, GRU)>());
  }
};

// Forward declarations for ai.onnx version 4
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 4, Concat);

// Iterate over schema from ai.onnx version 4
class OpSet_Onnx_ver4 {
 public:
  static void ForEachSchema(std::function<void(OpSchema&&)> fn) {
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 4, Concat)>());
  }
};

// Forward declarations for ai.onnx version 5
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 5, Reshape);

// Iterate over schema from ai.onnx version 5
class OpSet_Onnx_ver5 {
 public:
  static void ForEachSchema(std::function<void(OpSchema&&)> fn) {
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 5, Reshape)>());
  }
};

// Forward declarations for ai.onnx version 6
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Abs);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Add);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, BatchNormalization);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Cast);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Ceil);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Clip);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Div);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Dropout);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Elu);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Exp);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Floor);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Gemm);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, HardSigmoid);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, InstanceNormalization);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, LeakyRelu);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Log);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Max);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Mean);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Min);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Mul);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Neg);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, PRelu);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Reciprocal);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Relu);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Selu);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Sigmoid);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Sqrt);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Sub);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Sum);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Tanh);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Tile);

// Iterate over schema from ai.onnx version 6
class OpSet_Onnx_ver6 {
 public:
  static void ForEachSchema(std::function<void(OpSchema&&)> fn) {
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Abs)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Add)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, BatchNormalization)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Cast)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Ceil)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Clip)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Div)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Dropout)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Elu)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Exp)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Floor)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Gemm)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, HardSigmoid)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, InstanceNormalization)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, LeakyRelu)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Log)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Max)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Mean)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Min)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Mul)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Neg)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, PRelu)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Reciprocal)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Relu)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Selu)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Sigmoid)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Sqrt)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Sub)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Sum)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Tanh)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Tile)>());
  }
};

// Forward declarations for ai.onnx version 7
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Acos);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Add);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, And);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Asin);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Atan);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, AveragePool);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, BatchNormalization);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Cos);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Div);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Dropout);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Equal);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Gemm);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Greater);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, GRU);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Less);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, LSTM);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Mul);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Or);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Pow);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, RNN);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Sin);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Sub);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Tan);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Upsample);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Multinomial);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Xor);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, PRelu);

// Iterate over schema from ai.onnx version 7
class OpSet_Onnx_ver7 {
 public:
  static void ForEachSchema(std::function<void(OpSchema&&)> fn) {
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Acos)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Add)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, And)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Asin)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Atan)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, AveragePool)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, BatchNormalization)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Cos)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Div)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Dropout)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Equal)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Gemm)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Greater)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, GRU)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Less)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, LSTM)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Mul)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Or)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Pow)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, RNN)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Sin)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Sub)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Tan)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Upsample)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Multinomial)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Xor)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, PRelu)>());
  }
};

// Forward declarations for ai.onnx version 8
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 8, Expand);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 8, Max);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 8, Min);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 8, Sum);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 8, Mean);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 8, MaxPool);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 8, Scan);

// Iterate over schema from ai.onnx version 8
class OpSet_Onnx_ver8 {
 public:
  static void ForEachSchema(std::function<void(OpSchema&&)> fn) {
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 8, Expand)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 8, Min)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 8, Max)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 8, Sum)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 8, Mean)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 8, MaxPool)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 8, Scan)>());
  }
};

// Forward declarations for ai.onnx version 9
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, BatchNormalization);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Compress);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, ConstantOfShape);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, EyeLike);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Greater);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Less);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Upsample);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, MaxUnpool);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Constant);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, MatMul);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, OneHot);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, PRelu);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Gemm);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Flatten);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Sinh);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Cosh);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Asinh);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Acosh);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Atanh);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Shrink);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, IsNaN);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Sign);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Scan);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Erf);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Scatter);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Where);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Cast);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, NonZero);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, TfIdfVectorizer);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, MeanVarianceNormalization);

// Iterate over schema from ai.onnx version 9
class OpSet_Onnx_ver9 {
 public:
  static void ForEachSchema(std::function<void(OpSchema&&)> fn) {
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, BatchNormalization)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Compress)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, ConstantOfShape)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, EyeLike)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Greater)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Less)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Upsample)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, MaxUnpool)>());
    // Add more types' support to Constant/MatMul/PRelu/Gemm/Flatten op.
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Constant)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, MatMul)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, OneHot)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, PRelu)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Gemm)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Flatten)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Scatter)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Sinh)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Cosh)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Asinh)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Acosh)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Atanh)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Shrink)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, IsNaN)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Sign)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Scan)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Erf)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Cast)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Where)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, NonZero)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, TfIdfVectorizer)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, MeanVarianceNormalization)>());
  }
};

// Forward declarations for ai.onnx version 10
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, StringNormalizer);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, Upsample);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, Resize);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, TopK);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, MaxPool);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, Mod);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, AveragePool);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, Slice);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, ThresholdedRelu);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, Dropout);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, MatMulInteger);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, QLinearMatMul);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, ConvInteger);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, QLinearConv);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, QuantizeLinear);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, DequantizeLinear);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, IsInf);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, NonMaxSuppression);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, ReverseSequence);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, RoiAlign);

// Iterate over schema from ai.onnx version 10
class OpSet_Onnx_ver10 {
 public:
  static void ForEachSchema(std::function<void(OpSchema&&)> fn) {
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, Upsample)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, Resize)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, StringNormalizer)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, TopK)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, MaxPool)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, Mod)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, AveragePool)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, Slice)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, ThresholdedRelu)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, Dropout)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, MatMulInteger)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, QLinearMatMul)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, ConvInteger)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, QLinearConv)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, QuantizeLinear)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, DequantizeLinear)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, IsInf)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, NonMaxSuppression)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, ReverseSequence)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, RoiAlign)>());
  }
};

// Forward declarations for ai.onnx version 11
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Loop);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, CumSum);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Round);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, BitShift);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Unique);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, TopK);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, DepthToSpace);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Equal);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Constant);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, DynamicQuantizeLinear);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, GatherElements);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, ScatterElements);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Scatter);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Clip);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Resize);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Range);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Det);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, ScatterND);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, GatherND);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Gather);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, OneHot);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Slice);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Squeeze);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Unsqueeze);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Flatten);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, ArgMax);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, ArgMin);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, ReduceL1);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, ReduceL2);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, ReduceLogSum);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, ReduceLogSumExp);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, ReduceMax);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, ReduceMean);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, ReduceMin);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, ReduceProd);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, ReduceSum);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, ReduceSumSquare);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Compress);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Concat);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Hardmax);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, LogSoftmax);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Softmax);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Scan);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Split);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, AveragePool);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, MaxPool);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, MaxUnpool);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, LpPool);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Conv);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, ConvTranspose);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, SequenceEmpty);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, SequenceConstruct);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, SequenceInsert);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, SequenceAt);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, SequenceErase);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, SequenceLength);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, SplitToSequence);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, ConcatFromSequence);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Pad);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Gemm);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, If);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, NonMaxSuppression);

// Iterate over schema from ai.onnx version 11
class OpSet_Onnx_ver11 {
 public:
  static void ForEachSchema(std::function<void(OpSchema&&)> fn) {
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Loop)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, BitShift)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Unique)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, CumSum)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Round)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, TopK)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, DepthToSpace)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Equal)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Constant)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, DynamicQuantizeLinear)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, GatherElements)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, ScatterElements)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Scatter)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Clip)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Resize)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Range)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Det)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, ScatterND)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, GatherND)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Gather)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, OneHot)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Slice)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Squeeze)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Unsqueeze)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Flatten)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, ArgMin)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, ArgMax)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, ReduceL1)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, ReduceL2)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, ReduceLogSum)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, ReduceLogSumExp)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, ReduceMax)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, ReduceMean)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, ReduceMin)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, ReduceProd)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, ReduceSum)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, ReduceSumSquare)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Compress)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Concat)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Hardmax)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, LogSoftmax)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Softmax)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Scan)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Split)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, AveragePool)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, MaxPool)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, MaxUnpool)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, LpPool)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Conv)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, ConvTranspose)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, SequenceEmpty)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, SequenceConstruct)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, SequenceInsert)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, SequenceAt)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, SequenceErase)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, SequenceLength)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, SplitToSequence)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, ConcatFromSequence)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Pad)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Gemm)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, If)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, NonMaxSuppression)>());
  }
};

// Forward declarations for ai.onnx version 12
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 12, ArgMax);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 12, ArgMin);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 12, Clip);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 12, Einsum);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 12, MaxPool);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 12, ReduceMax);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 12, ReduceMin);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 12, GatherND);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 12, NegativeLogLikelihoodLoss);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 12, Dropout);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 12, Constant);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 12, Celu);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 12, Max);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 12, Min);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 12, LessOrEqual);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 12, GreaterOrEqual);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 12, SoftmaxCrossEntropyLoss);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 12, Pow);

// Iterate over schema from ai.onnx version 12
class OpSet_Onnx_ver12 {
 public:
  static void ForEachSchema(std::function<void(OpSchema&&)> fn) {
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 12, ArgMax)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 12, ArgMin)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 12, Clip)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 12, Einsum)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 12, MaxPool)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 12, ReduceMax)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 12, ReduceMin)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 12, GatherND)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 12, NegativeLogLikelihoodLoss)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 12, Dropout)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 12, Constant)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 12, Celu)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 12, Max)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 12, Min)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 12, LessOrEqual)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 12, GreaterOrEqual)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 12, SoftmaxCrossEntropyLoss)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 12, Pow)>());
  }
};
// Forward declarations for ai.onnx version 13
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Constant);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Greater);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Less);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Equal);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Add);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Sub);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Mul);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Div);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Softmax);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, LogSoftmax);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Hardmax);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Mod);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Neg);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Abs);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Reciprocal);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Floor);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Ceil);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Sqrt);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Relu);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Exp);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Log);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Tanh);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Pow);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Sigmoid);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Max);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Min);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Sum);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Mean);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Clip);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Gemm);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, MatMul);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Expand);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Sign);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Erf);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, SoftmaxCrossEntropyLoss);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, NegativeLogLikelihoodLoss);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Dropout);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Flatten);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, LRN);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, MeanVarianceNormalization);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, ReduceMax);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, ReduceMin);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, ReduceSum);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, ReduceSumSquare);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, ReduceMean);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, ReduceProd);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, ReduceLogSum);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, ReduceLogSumExp);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, ReduceL1);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, ReduceL2);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, ArgMax);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, ArgMin);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Cast);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Reshape);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Shape);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Size);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Concat);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Split);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Slice);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Transpose);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, ScatterND);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, ScatterElements);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Gather);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, GatherElements);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Squeeze);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Unsqueeze);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, SpaceToDepth);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, DepthToSpace);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Tile);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Resize);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Identity);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, IsNaN);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, NonZero);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, GatherND);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Pad);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, QuantizeLinear);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, DequantizeLinear);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Loop);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, If);

// Iterate over schema from ai.onnx version 13
class OpSet_Onnx_ver13 {
 public:
  static void ForEachSchema(std::function<void(OpSchema&&)> fn) {
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Constant)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Greater)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Less)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Equal)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Add)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Sub)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Mul)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Div)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Softmax)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, LogSoftmax)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Hardmax)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Mod)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Neg)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Abs)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Reciprocal)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Floor)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Ceil)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Sqrt)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Relu)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Exp)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Log)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Tanh)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Pow)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Sigmoid)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Max)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Min)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Sum)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Mean)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Clip)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Gemm)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, MatMul)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Expand)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Sign)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Erf)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, SoftmaxCrossEntropyLoss)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, NegativeLogLikelihoodLoss)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Dropout)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Flatten)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, LRN)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, MeanVarianceNormalization)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, ReduceMax)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, ReduceMin)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, ReduceSum)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, ReduceSumSquare)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, ReduceMean)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, ReduceProd)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, ReduceLogSum)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, ReduceLogSumExp)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, ReduceL1)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, ReduceL2)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, ArgMax)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, ArgMin)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Cast)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Reshape)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Shape)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Size)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Concat)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Split)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Slice)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Transpose)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, ScatterND)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, ScatterElements)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Gather)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, GatherElements)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Squeeze)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Unsqueeze)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, SpaceToDepth)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, DepthToSpace)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Tile)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Resize)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Identity)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, IsNaN)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, NonZero)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, GatherND)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Pad)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, QuantizeLinear)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, DequantizeLinear)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Loop)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, If)>());
  }
};

// Forward declarations for ai.onnx version 14
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 14, CumSum);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 14, Relu);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 14, Reshape);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 14, GRU);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 14, LSTM);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 14, RNN);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 14, Trilu);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 14, BatchNormalization);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 14, HardSwish);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 14, Add);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 14, Sub);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 14, Mul);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 14, Div);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 14, Identity);

// Iterate over schema from ai.onnx version 14
class OpSet_Onnx_ver14 {
 public:
  static void ForEachSchema(std::function<void(OpSchema&&)> fn) {
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 14, CumSum)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 14, Relu)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 14, Reshape)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 14, GRU)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 14, LSTM)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 14, RNN)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 14, Trilu)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 14, BatchNormalization)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 14, HardSwish)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 14, Add)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 14, Sub)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 14, Mul)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 14, Div)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 14, Identity)>());
  }
};

// Forward declarations for ai.onnx version 15
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 15, BatchNormalization);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 15, Bernoulli);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 15, Pow);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 15, Optional);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 15, OptionalHasElement);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 15, OptionalGetElement);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 15, CastLike);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 15, Shape);

// Iterate over schema from ai.onnx version 15
class OpSet_Onnx_ver15 {
 public:
  static void ForEachSchema(std::function<void(OpSchema&&)> fn) {
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 15, BatchNormalization)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 15, Bernoulli)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 15, Pow)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 15, Optional)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 15, OptionalHasElement)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 15, OptionalGetElement)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 15, CastLike)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 15, Shape)>());
  }
};

// Forward declarations for ai.onnx version 16
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 16, RoiAlign);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 16, ScatterND);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 16, ScatterElements);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 16, If);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 16, Loop);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 16, Identity);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 16, Where);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 16, GridSample);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 16, Scan);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 16, LessOrEqual);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 16, GreaterOrEqual);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 16, LeakyRelu);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 16, PRelu);

// Iterate over schema from ai.onnx version 16
class OpSet_Onnx_ver16 {
 public:
  static void ForEachSchema(std::function<void(OpSchema&&)> fn) {
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 16, RoiAlign)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 16, ScatterND)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 16, ScatterElements)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 16, If)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 16, Loop)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 16, Identity)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 16, Where)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 16, GridSample)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 16, Scan)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 16, LessOrEqual)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 16, GreaterOrEqual)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 16, LeakyRelu)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 16, PRelu)>());
  }
};

// Forward declarations for ai.onnx version 17
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 17, LayerNormalization);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 17, SequenceMap);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 17, DFT);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 17, HannWindow);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 17, HammingWindow);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 17, BlackmanWindow);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 17, MelWeightMatrix);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 17, STFT);

// Iterate over schema from ai.onnx version 17
class OpSet_Onnx_ver17 {
 public:
  static void ForEachSchema(std::function<void(OpSchema&&)> fn) {
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 17, LayerNormalization)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 17, SequenceMap)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 17, DFT)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 17, HannWindow)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 17, HammingWindow)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 17, BlackmanWindow)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 17, MelWeightMatrix)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 17, STFT)>());
  }
};

// Forward declarations for ai.onnx version 18
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 18, Pad);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 18, CenterCropPad);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 18, Resize);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 18, Mish);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 18, OptionalGetElement);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 18, OptionalHasElement);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 18, Split);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 18, Col2Im);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 18, ScatterND);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 18, ScatterElements);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 18, ReduceSumSquare);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 18, ReduceLogSum);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 18, ReduceLogSumExp);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 18, ReduceL1);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 18, ReduceL2);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 18, ReduceMax);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 18, ReduceMin);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 18, ReduceMean);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 18, ReduceProd);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 18, BitwiseAnd);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 18, BitwiseOr);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 18, BitwiseXor);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 18, BitwiseNot);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 18, GroupNormalization);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 18, LpPool);

// Iterate over schema from ai.onnx version 18
class OpSet_Onnx_ver18 {
 public:
  static void ForEachSchema(std::function<void(OpSchema&&)> fn) {
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 18, Pad)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 18, CenterCropPad)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 18, Resize)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 18, Mish)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 18, OptionalGetElement)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 18, Split)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 18, OptionalHasElement)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 18, Col2Im)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 18, ScatterND)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 18, ScatterElements)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 18, ReduceSumSquare)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 18, ReduceLogSum)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 18, ReduceLogSumExp)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 18, ReduceL1)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 18, ReduceL2)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 18, ReduceMax)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 18, ReduceMin)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 18, ReduceMean)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 18, ReduceProd)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 18, BitwiseAnd)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 18, BitwiseOr)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 18, BitwiseXor)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 18, BitwiseNot)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 18, GroupNormalization)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 18, LpPool)>());
  }
};

// Forward declarations for ai.onnx version 19
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 19, Equal);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 19, AveragePool);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 19, Cast);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 19, CastLike);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 19, Constant);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 19, DeformConv);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 19, DequantizeLinear);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 19, Identity);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 19, If);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 19, Loop);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 19, Pad);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 19, QuantizeLinear);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 19, Resize);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 19, Reshape);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 19, Scan);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 19, Shape);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 19, Size);

// Iterate over schema from ai.onnx version 19
class OpSet_Onnx_ver19 {
 public:
  static void ForEachSchema(std::function<void(OpSchema&&)> fn) {
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 19, Equal)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 19, AveragePool)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 19, Cast)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 19, CastLike)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 19, Constant)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 19, DeformConv)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 19, DequantizeLinear)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 19, Identity)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 19, If)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 19, Loop)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 19, Pad)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 19, QuantizeLinear)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 19, Reshape)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 19, Resize)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 19, Scan)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 19, Shape)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 19, Size)>());
  }
};

inline void RegisterOnnxOperatorSetSchema() {
  RegisterOpSetSchema<OpSet_Onnx_ver1>();
  RegisterOpSetSchema<OpSet_Onnx_ver2>();
  RegisterOpSetSchema<OpSet_Onnx_ver3>();
  RegisterOpSetSchema<OpSet_Onnx_ver4>();
  RegisterOpSetSchema<OpSet_Onnx_ver5>();
  RegisterOpSetSchema<OpSet_Onnx_ver6>();
  RegisterOpSetSchema<OpSet_Onnx_ver7>();
  RegisterOpSetSchema<OpSet_Onnx_ver8>();
  RegisterOpSetSchema<OpSet_Onnx_ver9>();
  RegisterOpSetSchema<OpSet_Onnx_ver10>();
  RegisterOpSetSchema<OpSet_Onnx_ver11>();
  RegisterOpSetSchema<OpSet_Onnx_ver12>();
  RegisterOpSetSchema<OpSet_Onnx_ver13>();
  RegisterOpSetSchema<OpSet_Onnx_ver14>();
  RegisterOpSetSchema<OpSet_Onnx_ver15>();
  RegisterOpSetSchema<OpSet_Onnx_ver16>();
  RegisterOpSetSchema<OpSet_Onnx_ver17>();
  RegisterOpSetSchema<OpSet_Onnx_ver18>();
  RegisterOpSetSchema<OpSet_Onnx_ver19>();
  // 0 means all versions of ONNX schema have been loaded
  OpSchemaRegistry::Instance()->SetLoadedSchemaVersion(0);
}

inline void RegisterOnnxOperatorSetSchema(int target_version) {
  // Update here if opset_version bumps
  // These calls for schema registration here are required to be in descending order for this to work correctly
  RegisterOpSetSchema<OpSet_Onnx_ver19>(target_version);
  RegisterOpSetSchema<OpSet_Onnx_ver18>(target_version);
  RegisterOpSetSchema<OpSet_Onnx_ver17>(target_version);
  RegisterOpSetSchema<OpSet_Onnx_ver16>(target_version);
  RegisterOpSetSchema<OpSet_Onnx_ver15>(target_version);
  RegisterOpSetSchema<OpSet_Onnx_ver14>(target_version);
  RegisterOpSetSchema<OpSet_Onnx_ver13>(target_version);
  RegisterOpSetSchema<OpSet_Onnx_ver12>(target_version);
  RegisterOpSetSchema<OpSet_Onnx_ver11>(target_version);
  RegisterOpSetSchema<OpSet_Onnx_ver10>(target_version);
  RegisterOpSetSchema<OpSet_Onnx_ver9>(target_version);
  RegisterOpSetSchema<OpSet_Onnx_ver8>(target_version);
  RegisterOpSetSchema<OpSet_Onnx_ver7>(target_version);
  RegisterOpSetSchema<OpSet_Onnx_ver6>(target_version);
  RegisterOpSetSchema<OpSet_Onnx_ver5>(target_version);
  RegisterOpSetSchema<OpSet_Onnx_ver4>(target_version);
  RegisterOpSetSchema<OpSet_Onnx_ver3>(target_version);
  RegisterOpSetSchema<OpSet_Onnx_ver2>(target_version);
  RegisterOpSetSchema<OpSet_Onnx_ver1>(target_version);
  // Sets to record the loaded version and prevent the full operator check in Debug mode
  OpSchemaRegistry::Instance()->SetLoadedSchemaVersion(target_version);
}

} // namespace ONNX_NAMESPACE
